//! Quantized scale-header policy for `secant sanitize` (Milestone S-4).
//!
//! A `Q8_0` block is `{f16 d; i8 qs[32]}` and dequantizes to `d * qs[i]`. A
//! `Q4_K` super-block is `{f16 d; f16 dmin; u8 scales[12]; u8 qs[128]}` and
//! dequantizes to `d * sc_j * q - dmin * m_j` with 6-bit `sc_j`, `m_j` and
//! 4-bit `q`. The f16 scale fields multiply every element of their block, so
//! one hostile scale (NaN, Inf, subnormal, or a finite value large enough to
//! push the block past the policy bound) poisons 32 or 256 weights at once and
//! is the value that reaches the `QUANT_SCALE_DIV_OR_OVERFLOW` sink class of
//! `secant scan-sinks` (a reciprocal or division by the scale in a consumer).
//!
//! Finding kinds, per scale field:
//!   * `quant_scale_nan`, `quant_scale_inf`, `quant_scale_subnormal`: the
//!     IEEE class of the f16 bits.
//!   * `quant_scale_extreme`: finite, and `|scale| * M > max_abs`, where `M` is
//!     the largest multiplier the block actually applies to that field
//!     (`Q8_0 d`: `max |qs|`; `Q4_K d`: `max_j sc_j * max q in sub-block j`;
//!     `Q4_K dmin`: `max_j m_j`).
//!
//! and per block:
//!   * `quant_block_extreme`: no field is NaN, Inf or extreme, yet some
//!     dequantized element has `|x| > max_abs`. Only reachable when the two
//!     `Q4_K` terms have opposite signs (ggml's quantizer writes `d, dmin >= 0`).
//!     It has no single-field repair and always remains a finding.
//!
//! Repairs (each opt-in, same flags as the float path): NaN to `+0`
//! (`--clamp-nonfinite`; the block dequantizes to zeros for that term), Inf and
//! extreme finite scales to the signed largest f16 `v` with `v * M <= max_abs`
//! (`--clamp-nonfinite` / `--clamp-extreme`), subnormal to the signed zero
//! (`--flush-subnormal`). Flushing a subnormal SCALE zeroes up to 256 weights,
//! not one: it is a bigger edit than flushing one subnormal float.
//!
//! All arithmetic is exact: f16 values are widened to f64 and multiplied by
//! small integers (at most 63 * 15 = 945), so no product rounds.

use crate::quant::{get_scale_min_k4, QuantFormat};
use crate::security::sanitize::{classify, Kind};

pub const KIND_NAN: &str = "quant_scale_nan";
pub const KIND_INF: &str = "quant_scale_inf";
pub const KIND_SUBNORMAL: &str = "quant_scale_subnormal";
pub const KIND_EXTREME: &str = "quant_scale_extreme";
pub const KIND_BLOCK_EXTREME: &str = "quant_block_extreme";

/// Policy tag appended to the sanitizer's `policy_version` when a file holds
/// quantized tensors.
pub const POLICY_TAG: &str = "quant-scale-headers-v1";

/// One f16 scale field of a block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleField {
    /// `d` or `dmin`.
    pub name: &'static str,
    /// Byte offset of the field inside the block.
    pub offset: usize,
    pub bits: u16,
    pub value: f64,
    pub kind: Kind,
    /// Largest multiplier the block applies to this field (exact integer).
    pub multiplier: f64,
}

impl ScaleField {
    /// Finite and `|value| * multiplier > max_abs`.
    pub fn extreme(&self, max_abs: f64) -> bool {
        self.kind != Kind::Nan && self.kind != Kind::Inf && self.value.abs() * self.multiplier > max_abs
    }
}

/// One finding inside a block.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockFinding {
    pub kind: &'static str,
    /// Field name, or empty for a block-level finding.
    pub field: &'static str,
    /// Byte offset inside the block (0 for a block-level finding).
    pub offset: usize,
    pub bits: u16,
}

fn le_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn field(name: &'static str, blk: &[u8], offset: usize, multiplier: f64) -> ScaleField {
    let bits = le_u16(&blk[offset..]);
    let (value, kind) = classify(bits as u64, "F16");
    ScaleField { name, offset, bits, value, kind, multiplier }
}

/// The scale fields of one block, with their multipliers. `blk` must be
/// exactly one block of `fmt`.
pub fn block_fields(fmt: QuantFormat, blk: &[u8]) -> Vec<ScaleField> {
    assert_eq!(blk.len(), fmt.block_bytes(), "block_fields: one {} block", fmt.name());
    match fmt {
        QuantFormat::Q8_0 => {
            let m = blk[2..].iter().map(|&q| (q as i8 as i32).unsigned_abs()).max().unwrap_or(0);
            vec![field("d", blk, 0, m as f64)]
        }
        QuantFormat::Q4K => {
            let (scales, qs) = (&blk[4..16], &blk[16..]);
            let (mut md, mut mm) = (0u32, 0u32);
            for j in 0..8 {
                let (sc, m) = get_scale_min_k4(j, scales);
                let chunk = &qs[(j / 2) * 32..(j / 2) * 32 + 32];
                let qmax = chunk.iter().map(|&b| if j % 2 == 0 { b & 0xF } else { b >> 4 }).max().unwrap_or(0);
                md = md.max(sc as u32 * qmax as u32);
                mm = mm.max(m as u32);
            }
            vec![field("d", blk, 0, md as f64), field("dmin", blk, 2, mm as f64)]
        }
    }
}

/// Largest `|x|` over the block's dequantized elements, in exact f64. NaN if
/// any element is NaN; infinite if any element is infinite.
pub fn realized_max_abs(fmt: QuantFormat, blk: &[u8]) -> f64 {
    assert_eq!(blk.len(), fmt.block_bytes(), "realized_max_abs: one {} block", fmt.name());
    let mut worst = 0f64;
    let mut see = |x: f64| {
        if x.is_nan() || worst.is_nan() {
            worst = f64::NAN;
        } else if x.abs() > worst {
            worst = x.abs();
        }
    };
    match fmt {
        QuantFormat::Q8_0 => {
            let (d, _) = classify(le_u16(blk) as u64, "F16");
            for &q in &blk[2..] {
                see(d * (q as i8) as f64);
            }
        }
        QuantFormat::Q4K => {
            let (d, _) = classify(le_u16(blk) as u64, "F16");
            let (dmin, _) = classify(le_u16(&blk[2..]) as u64, "F16");
            let (scales, qs) = (&blk[4..16], &blk[16..]);
            for j in 0..8 {
                let (sc, m) = get_scale_min_k4(j, scales);
                for &b in &qs[(j / 2) * 32..(j / 2) * 32 + 32] {
                    let q = if j % 2 == 0 { b & 0xF } else { b >> 4 };
                    see(d * sc as f64 * q as f64 - dmin * m as f64);
                }
            }
        }
    }
    worst
}

/// Largest non-negative finite f16 `v` with `v * multiplier <= max_abs`
/// (binary search over the ordered positive f16 encodings). A zero multiplier
/// leaves the field inert; the largest f16 not above `max_abs` is returned.
pub fn bounded_scale_bits(multiplier: f64, max_abs: f64) -> u16 {
    let (mut lo, mut hi) = (0u16, 0x7bffu16);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let (v, _) = classify(mid as u64, "F16");
        let ok = if multiplier > 0.0 { v * multiplier <= max_abs } else { v <= max_abs };
        if ok {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Every policy finding of one block, in field order, block-level last.
pub fn scan_block(fmt: QuantFormat, blk: &[u8], max_abs: f64) -> Vec<BlockFinding> {
    let mut out = Vec::new();
    let mut attributed = false;
    for f in block_fields(fmt, blk) {
        let class = match f.kind {
            Kind::Nan => Some(KIND_NAN),
            Kind::Inf => Some(KIND_INF),
            Kind::Subnormal => Some(KIND_SUBNORMAL),
            Kind::Normal => None,
        };
        if let Some(kind) = class {
            out.push(BlockFinding { kind, field: f.name, offset: f.offset, bits: f.bits });
        }
        attributed |= matches!(f.kind, Kind::Nan | Kind::Inf);
        if f.extreme(max_abs) {
            out.push(BlockFinding { kind: KIND_EXTREME, field: f.name, offset: f.offset, bits: f.bits });
            attributed = true;
        }
    }
    if !attributed && realized_max_abs(fmt, blk) > max_abs {
        out.push(BlockFinding { kind: KIND_BLOCK_EXTREME, field: "", offset: 0, bits: 0 });
    }
    out
}

/// Which opt-in repairs are enabled.
#[derive(Debug, Clone, Copy, Default)]
pub struct Repairs {
    pub clamp_nonfinite: bool,
    pub clamp_extreme: bool,
    pub flush_subnormal: bool,
}

/// Rewrite the repairable scale fields of one block in place. Returns the
/// finding kinds repaired, one entry per rewritten field. Classification uses
/// the bits as they were on entry (a repair never feeds another).
pub fn repair_block(fmt: QuantFormat, blk: &mut [u8], max_abs: f64, r: Repairs) -> Vec<&'static str> {
    let mut done = Vec::new();
    for f in block_fields(fmt, blk) {
        let sign = f.bits & 0x8000;
        let repl: Option<(u16, &'static str)> = if r.clamp_nonfinite && f.kind == Kind::Nan {
            Some((0, KIND_NAN))
        } else if r.clamp_nonfinite && f.kind == Kind::Inf {
            Some((bounded_scale_bits(f.multiplier, max_abs) | sign, KIND_INF))
        } else if r.flush_subnormal && f.kind == Kind::Subnormal {
            Some((sign, KIND_SUBNORMAL))
        } else if r.clamp_extreme && f.extreme(max_abs) {
            Some((bounded_scale_bits(f.multiplier, max_abs) | sign, KIND_EXTREME))
        } else {
            None
        };
        if let Some((bits, kind)) = repl {
            blk[f.offset..f.offset + 2].copy_from_slice(&bits.to_le_bytes());
            done.push(kind);
        }
    }
    done
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{quantize_row_q4_k, quantize_row_q8_0, Q4_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, QK8_0, QK_K};

    const F16_NAN: u16 = 0x7e00;
    const F16_POS_INF: u16 = 0x7c00;
    const F16_NEG_INF: u16 = 0xfc00;
    const F16_MAX: u16 = 0x7bff; // 65504
    const F16_SUB: u16 = 0x0001; // 2^-24

    fn q8_block(scale: f32) -> Vec<u8> {
        let x: Vec<f32> = (0..QK8_0).map(|i| (i as f32 - 15.5) * scale).collect();
        let mut b = vec![0u8; Q8_0_BLOCK_BYTES];
        quantize_row_q8_0(&x, &mut b).unwrap();
        b
    }

    fn q4k_block() -> Vec<u8> {
        let x: Vec<f32> = (0..QK_K).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let mut b = vec![0u8; Q4_K_BLOCK_BYTES];
        quantize_row_q4_k(&x, &mut b).unwrap();
        b
    }

    fn set(b: &mut [u8], off: usize, bits: u16) {
        b[off..off + 2].copy_from_slice(&bits.to_le_bytes());
    }

    fn kinds(f: &[BlockFinding]) -> Vec<(&str, &str)> {
        f.iter().map(|x| (x.kind, x.field)).collect()
    }

    #[test]
    fn ggml_quantized_blocks_are_clean() {
        // control: genuine ggml-quantized blocks carry no scale finding
        assert!(scan_block(QuantFormat::Q8_0, &q8_block(0.01), 1e4).is_empty());
        assert!(scan_block(QuantFormat::Q4K, &q4k_block(), 1e4).is_empty());
        // and the multipliers match the format's ranges
        let f = block_fields(QuantFormat::Q8_0, &q8_block(0.01));
        assert_eq!(f[0].multiplier, 127.0);
        let f = block_fields(QuantFormat::Q4K, &q4k_block());
        assert!(f[0].multiplier > 0.0 && f[0].multiplier <= 945.0 && f[1].multiplier <= 63.0, "{:?}", f);
    }

    #[test]
    fn nonfinite_and_subnormal_scales_are_detected_per_field() {
        for (bits, kind) in [(F16_NAN, KIND_NAN), (F16_POS_INF, KIND_INF), (F16_NEG_INF, KIND_INF), (F16_SUB, KIND_SUBNORMAL)] {
            let mut b = q8_block(0.01);
            set(&mut b, 0, bits);
            assert_eq!(kinds(&scan_block(QuantFormat::Q8_0, &b, 1e4)), vec![(kind, "d")], "{:#06x}", bits);
            for (off, name) in [(0, "d"), (2, "dmin")] {
                let mut b = q4k_block();
                set(&mut b, off, bits);
                let got = scan_block(QuantFormat::Q4K, &b, 1e4);
                assert_eq!(kinds(&got), vec![(kind, name)], "{:#06x} at {}", bits, name);
                assert_eq!(got[0].offset, off);
                assert_eq!(got[0].bits, bits);
            }
        }
    }

    #[test]
    fn extreme_finite_scale_uses_the_realized_multiplier() {
        // 65504 * 127 > 1e4: extreme
        let mut b = q8_block(0.01);
        set(&mut b, 0, F16_MAX);
        assert_eq!(kinds(&scan_block(QuantFormat::Q8_0, &b, 1e4)), vec![(KIND_EXTREME, "d")]);
        // the same scale over an all-zero block multiplies nothing: clean
        let mut z = vec![0u8; Q8_0_BLOCK_BYTES];
        set(&mut z, 0, F16_MAX);
        assert!(scan_block(QuantFormat::Q8_0, &z, 1e4).is_empty());
        // boundary: |d| * M == max_abs is allowed, one ulp above is not
        let mut one = vec![0u8; Q8_0_BLOCK_BYTES];
        one[2] = 100; // q = 100
        set(&mut one, 0, 0x5640); // f16 100.0
        assert!(scan_block(QuantFormat::Q8_0, &one, 1e4).is_empty());
        set(&mut one, 0, 0x5641);
        assert_eq!(kinds(&scan_block(QuantFormat::Q8_0, &one, 1e4)), vec![(KIND_EXTREME, "d")]);
        // Q4_K dmin extreme
        let mut k = q4k_block();
        set(&mut k, 2, F16_MAX);
        assert_eq!(kinds(&scan_block(QuantFormat::Q4K, &k, 1e4)), vec![(KIND_EXTREME, "dmin")]);
    }

    #[test]
    fn opposite_sign_q4k_terms_are_a_block_level_finding() {
        // d * sc * q and dmin * m each stay under max_abs, but a negative d
        // makes them add: |x| = 2 * 600 > 1000.
        let mut b = vec![0u8; Q4_K_BLOCK_BYTES];
        b[4] = 1; // sc_0 = 1
        b[8] = 1; // m_0 = 1
        b[16] = 0x0F; // q = 15 in sub-block 0
        set(&mut b, 0, 0xd100); // d = -40.0 -> d*sc*q = -600
        set(&mut b, 2, 0x60b0); // dmin = 600.0 -> -dmin*m = -600
        let f = block_fields(QuantFormat::Q4K, &b);
        assert_eq!((f[0].value, f[1].value), (-40.0, 600.0));
        assert!(!f[0].extreme(1000.0) && !f[1].extreme(1000.0));
        assert_eq!(realized_max_abs(QuantFormat::Q4K, &b), 1200.0);
        assert_eq!(kinds(&scan_block(QuantFormat::Q4K, &b, 1000.0)), vec![(KIND_BLOCK_EXTREME, "")]);
        // and no repair addresses it
        let mut c = b.clone();
        let all = Repairs { clamp_nonfinite: true, clamp_extreme: true, flush_subnormal: true };
        assert!(repair_block(QuantFormat::Q4K, &mut c, 1000.0, all).is_empty());
        assert_eq!(c, b);
    }

    #[test]
    fn repairs_leave_a_clean_block_and_keep_the_sign() {
        let all = Repairs { clamp_nonfinite: true, clamp_extreme: true, flush_subnormal: true };
        for fmt in [QuantFormat::Q8_0, QuantFormat::Q4K] {
            let base = if fmt == QuantFormat::Q8_0 { q8_block(0.01) } else { q4k_block() };
            let offsets: &[usize] = if fmt == QuantFormat::Q8_0 { &[0] } else { &[0, 2] };
            for &off in offsets {
                for bits in [F16_NAN, F16_POS_INF, F16_NEG_INF, F16_SUB, 0x8001, F16_MAX] {
                    let mut b = base.clone();
                    set(&mut b, off, bits);
                    let before = scan_block(fmt, &b, 1e4);
                    assert!(!before.is_empty());
                    let done = repair_block(fmt, &mut b, 1e4, all);
                    assert_eq!(done.len(), 1, "{:?} {:#06x}", fmt, bits);
                    let after = scan_block(fmt, &b, 1e4);
                    assert!(after.is_empty(), "{:?} {:#06x}@{}: {:?}", fmt, bits, off, after);
                    let new = le_u16(&b[off..]);
                    if bits != F16_NAN {
                        assert_eq!(new & 0x8000, bits & 0x8000, "sign kept");
                    } else {
                        assert_eq!(new, 0);
                    }
                    assert!(realized_max_abs(fmt, &b) <= 1e4);
                }
            }
        }
    }

    #[test]
    fn repairs_are_opt_in_individually() {
        let mut b = q8_block(0.01);
        set(&mut b, 0, F16_POS_INF);
        let orig = b.clone();
        assert!(repair_block(QuantFormat::Q8_0, &mut b, 1e4, Repairs { clamp_extreme: true, flush_subnormal: true, ..Default::default() })
            .is_empty());
        assert_eq!(b, orig);
        assert_eq!(repair_block(QuantFormat::Q8_0, &mut b, 1e4, Repairs { clamp_nonfinite: true, ..Default::default() }), vec![KIND_INF]);
        // the clamped scale is the LARGEST admissible one: one ulp up is extreme
        let v = le_u16(&b);
        let m = block_fields(QuantFormat::Q8_0, &b)[0].multiplier;
        assert!(classify(v as u64, "F16").0 * m <= 1e4);
        assert!(classify(v as u64 + 1, "F16").0 * m > 1e4);
    }

    #[test]
    fn bounded_scale_bits_edges() {
        assert_eq!(bounded_scale_bits(0.0, 1e9), F16_MAX);
        assert_eq!(bounded_scale_bits(1.0, 1e9), F16_MAX);
        assert_eq!(classify(bounded_scale_bits(127.0, 1e4) as u64, "F16").0, 78.6875); // largest f16 <= 78.74
        assert_eq!(bounded_scale_bits(945.0, 1e-12), 0);
    }
}
