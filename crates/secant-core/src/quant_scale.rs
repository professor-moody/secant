//! Quantized scale-header policy for `secant sanitize` (Milestone S-4).
//!
//! A `Q8_0` block is `{f16 d; i8 qs[32]}` and dequantizes to `d * qs[i]`. A
//! `Q4_K` super-block is `{f16 d; f16 dmin; u8 scales[12]; u8 qs[128]}` and
//! dequantizes to `d * sc_j * q - dmin * m_j` with 6-bit `sc_j`, `m_j` and
//! 4-bit `q`. The f16 scale fields multiply every element of their block, so
//! one non-finite scale poisons 32 or 256 weights at once. Rule B3-2A treats
//! these fields as multipliers: zero and FP16 subnormal values are valid, and
//! the default policy rejects only exponent `0x1f` encodings (NaN or Inf).
//!
//! Finding kinds, per scale field:
//!   * `quant_scale_nan`, `quant_scale_inf`: default Rule B3-2A findings.
//!   * `quant_scale_subnormal`: optional `strict_subnormal` diagnostic.
//!   * `quant_scale_extreme`: optional `check_extreme` diagnostic for a finite
//!     value where `|scale| * M > max_abs`, where `M` is
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
//! Repairs are opt-in. `--clamp-nonfinite` zeroes the entire block when any
//! scale header is NaN or Inf, including `d`, `dmin`, scale metadata and `qs`.
//! The optional diagnostic repairs retain their prior behavior: extreme finite
//! scales are clamped with `--clamp-extreme`, and subnormals are flushed with
//! `--flush-subnormal`.
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

/// GGUF block-quantized formats with IEEE-754 `f16` block scale headers scanned
/// by Rule B3-2A (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`,
/// `Q4_K`, `Q5_K`, `Q6_K`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScaleFormat {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl From<QuantFormat> for QuantScaleFormat {
    fn from(fmt: QuantFormat) -> Self {
        match fmt {
            QuantFormat::Q8_0 => QuantScaleFormat::Q8_0,
            QuantFormat::Q4K => QuantScaleFormat::Q4K,
        }
    }
}

impl QuantScaleFormat {
    pub fn from_dtype(d: &str) -> Option<QuantScaleFormat> {
        match d {
            "Q4_0" => Some(QuantScaleFormat::Q4_0),
            "Q4_1" => Some(QuantScaleFormat::Q4_1),
            "Q5_0" => Some(QuantScaleFormat::Q5_0),
            "Q5_1" => Some(QuantScaleFormat::Q5_1),
            "Q8_0" => Some(QuantScaleFormat::Q8_0),
            "Q2_K" => Some(QuantScaleFormat::Q2K),
            "Q3_K" => Some(QuantScaleFormat::Q3K),
            "Q4_K" => Some(QuantScaleFormat::Q4K),
            "Q5_K" => Some(QuantScaleFormat::Q5K),
            "Q6_K" => Some(QuantScaleFormat::Q6K),
            _ => None,
        }
    }

    pub fn block_elems(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
        }
    }

    pub const ALL: [Self; 10] = [
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
    ];

    pub fn fields(self) -> &'static [(&'static str, usize)] {
        match self {
            Self::Q4_0 | Self::Q5_0 | Self::Q8_0 => &[("d", 0)],
            Self::Q4_1 | Self::Q5_1 => &[("d", 0), ("m", 2)],
            Self::Q2K => &[("d", 80), ("dmin", 82)],
            Self::Q3K => &[("d", 108)],
            Self::Q4K | Self::Q5K => &[("d", 0), ("dmin", 2)],
            Self::Q6K => &[("d", 208)],
        }
    }
}

/// One f16 scale field of a block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleField {
    /// `d`, `dmin`, or `m`.
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

fn visit_q5_nibbles(qh_bytes: &[u8], qs: &[u8], mut f: impl FnMut(u8)) {
    let qh = u32::from_le_bytes([qh_bytes[0], qh_bytes[1], qh_bytes[2], qh_bytes[3]]);
    for (j, &b) in qs.iter().enumerate() {
        let xh0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
        f((b & 0x0F) | xh0);
        f((b >> 4) | xh1);
    }
}

fn visit_q2_k_terms(blk: &[u8], mut f: impl FnMut(u32, u32)) {
    let scales = &blk[0..16];
    let qs = &blk[16..80];
    let mut is = 0usize;
    for n in 0..2 {
        let q = &qs[n * 32..(n + 1) * 32];
        for shift in [0u32, 2, 4, 6] {
            let sc0 = (scales[is] & 0xF) as u32;
            let m0 = (scales[is] >> 4) as u32;
            is += 1;
            for &b in &q[0..16] {
                f(sc0 * ((b >> shift) & 3) as u32, m0);
            }
            let sc1 = (scales[is] & 0xF) as u32;
            let m1 = (scales[is] >> 4) as u32;
            is += 1;
            for &b in &q[16..32] {
                f(sc1 * ((b >> shift) & 3) as u32, m1);
            }
        }
    }
}

fn unpack_q3_k_scales(scales: &[u8]) -> [i8; 16] {
    let kmask1: u32 = 0x0303_0303;
    let kmask2: u32 = 0x0f0f_0f0f;
    let a0 = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
    let a1 = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
    let tmp = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
    let u0 = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
    let u1 = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
    let u2 = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    let u3 = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    let mut out = [0i8; 16];
    for (idx, w) in [u0, u1, u2, u3].into_iter().enumerate() {
        let bytes = w.to_le_bytes();
        for b in 0..4 {
            out[idx * 4 + b] = bytes[b] as i8;
        }
    }
    out
}

fn visit_q3_k_products(blk: &[u8], mut f: impl FnMut(i32)) {
    let hmask = &blk[0..32];
    let qs = &blk[32..96];
    let sc = unpack_q3_k_scales(&blk[96..108]);
    let mut is = 0usize;
    let mut m = 1u8;
    for n in 0..2 {
        let q = &qs[n * 32..(n + 1) * 32];
        for shift in [0u32, 2, 4, 6] {
            let dl0 = (sc[is] as i32) - 32;
            is += 1;
            for l in 0..16 {
                let q3 = ((q[l] >> shift) & 3) as i32 - if (hmask[l] & m) != 0 { 0 } else { 4 };
                f(dl0 * q3);
            }
            let dl1 = (sc[is] as i32) - 32;
            is += 1;
            for l in 16..32 {
                let q3 = ((q[l] >> shift) & 3) as i32 - if (hmask[l] & m) != 0 { 0 } else { 4 };
                f(dl1 * q3);
            }
            m = m.wrapping_shl(1);
        }
    }
}

fn visit_q5_k_terms(blk: &[u8], mut f: impl FnMut(u32, u32)) {
    let scales = &blk[4..16];
    let qh = &blk[16..48];
    let qs = &blk[48..176];
    for j in 0..8 {
        let (sc, m) = get_scale_min_k4(j, scales);
        let chunk = &qs[(j / 2) * 32..(j / 2) * 32 + 32];
        let u = 1u8 << j;
        for l in 0..32 {
            let q4 = if j % 2 == 0 { chunk[l] & 0xF } else { chunk[l] >> 4 };
            let q5 = q4 + if (qh[l] & u) != 0 { 16 } else { 0 };
            f(sc as u32 * q5 as u32, m as u32);
        }
    }
}

fn visit_q6_k_products(blk: &[u8], mut f: impl FnMut(i32)) {
    let ql = &blk[0..128];
    let qh = &blk[128..192];
    let scales = &blk[192..208];
    for n in 0..2 {
        let ql_h = &ql[n * 64..(n + 1) * 64];
        let qh_h = &qh[n * 32..(n + 1) * 32];
        let sc_h = &scales[n * 8..(n + 1) * 8];
        for l in 0..32 {
            let is = l / 16;
            let q1 = (((ql_h[l] & 0xF) | ((qh_h[l] & 3) << 4)) as i32) - 32;
            let q2 = (((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4)) as i32) - 32;
            let q3 = (((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) as i32) - 32;
            let q4 = (((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) as i32) - 32;
            f((sc_h[is] as i8 as i32) * q1);
            f((sc_h[is + 2] as i8 as i32) * q2);
            f((sc_h[is + 4] as i8 as i32) * q3);
            f((sc_h[is + 6] as i8 as i32) * q4);
        }
    }
}

/// The scale fields of one block, with their multipliers. `blk` must be
/// exactly one block of `fmt`.
pub fn block_fields(fmt: impl Into<QuantScaleFormat>, blk: &[u8]) -> Vec<ScaleField> {
    let fmt = fmt.into();
    assert_eq!(blk.len(), fmt.block_bytes(), "block_fields: one {} block", fmt.name());
    match fmt {
        QuantScaleFormat::Q4_0 => {
            let mut m = 0u32;
            for &b in &blk[2..18] {
                m = m.max(((b & 0xF) as i32 - 8).unsigned_abs());
                m = m.max(((b >> 4) as i32 - 8).unsigned_abs());
            }
            vec![field("d", blk, 0, m as f64)]
        }
        QuantScaleFormat::Q4_1 => {
            let mut m = 0u32;
            for &b in &blk[4..20] {
                m = m.max((b & 0xF) as u32).max((b >> 4) as u32);
            }
            vec![field("d", blk, 0, m as f64), field("m", blk, 2, 1.0)]
        }
        QuantScaleFormat::Q5_0 => {
            let mut m = 0u32;
            visit_q5_nibbles(&blk[2..6], &blk[6..22], |q| {
                m = m.max((q as i32 - 16).unsigned_abs());
            });
            vec![field("d", blk, 0, m as f64)]
        }
        QuantScaleFormat::Q5_1 => {
            let mut m = 0u32;
            visit_q5_nibbles(&blk[4..8], &blk[8..24], |q| {
                m = m.max(q as u32);
            });
            vec![field("d", blk, 0, m as f64), field("m", blk, 2, 1.0)]
        }
        QuantScaleFormat::Q8_0 => {
            let m = blk[2..].iter().map(|&q| (q as i8 as i32).unsigned_abs()).max().unwrap_or(0);
            vec![field("d", blk, 0, m as f64)]
        }
        QuantScaleFormat::Q2K => {
            let (mut md, mut mm) = (0u32, 0u32);
            visit_q2_k_terms(blk, |dq, m| {
                md = md.max(dq);
                mm = mm.max(m);
            });
            vec![field("d", blk, 80, md as f64), field("dmin", blk, 82, mm as f64)]
        }
        QuantScaleFormat::Q3K => {
            let mut md = 0u32;
            visit_q3_k_products(blk, |p| {
                md = md.max(p.unsigned_abs());
            });
            vec![field("d", blk, 108, md as f64)]
        }
        QuantScaleFormat::Q4K => {
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
        QuantScaleFormat::Q5K => {
            let (mut md, mut mm) = (0u32, 0u32);
            visit_q5_k_terms(blk, |dq, m| {
                md = md.max(dq);
                mm = mm.max(m);
            });
            vec![field("d", blk, 0, md as f64), field("dmin", blk, 2, mm as f64)]
        }
        QuantScaleFormat::Q6K => {
            let mut md = 0u32;
            visit_q6_k_products(blk, |p| {
                md = md.max(p.unsigned_abs());
            });
            vec![field("d", blk, 208, md as f64)]
        }
    }
}

/// Largest `|x|` over the block's dequantized elements, in exact f64. NaN if
/// any element is NaN; infinite if any element is infinite.
pub fn realized_max_abs(fmt: impl Into<QuantScaleFormat>, blk: &[u8]) -> f64 {
    let fmt = fmt.into();
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
        QuantScaleFormat::Q4_0 => {
            let (d, _) = classify(le_u16(blk) as u64, "F16");
            for &b in &blk[2..18] {
                see(d * ((b & 0xF) as i32 - 8) as f64);
                see(d * ((b >> 4) as i32 - 8) as f64);
            }
        }
        QuantScaleFormat::Q4_1 => {
            let (d, _) = classify(le_u16(blk) as u64, "F16");
            let (m, _) = classify(le_u16(&blk[2..]) as u64, "F16");
            for &b in &blk[4..20] {
                see(d * (b & 0xF) as f64 + m);
                see(d * (b >> 4) as f64 + m);
            }
        }
        QuantScaleFormat::Q5_0 => {
            let (d, _) = classify(le_u16(blk) as u64, "F16");
            visit_q5_nibbles(&blk[2..6], &blk[6..22], |q| {
                see(d * (q as i32 - 16) as f64);
            });
        }
        QuantScaleFormat::Q5_1 => {
            let (d, _) = classify(le_u16(blk) as u64, "F16");
            let (m, _) = classify(le_u16(&blk[2..]) as u64, "F16");
            visit_q5_nibbles(&blk[4..8], &blk[8..24], |q| {
                see(d * q as f64 + m);
            });
        }
        QuantScaleFormat::Q8_0 => {
            let (d, _) = classify(le_u16(blk) as u64, "F16");
            for &q in &blk[2..] {
                see(d * (q as i8) as f64);
            }
        }
        QuantScaleFormat::Q2K => {
            let (d, _) = classify(le_u16(&blk[80..]) as u64, "F16");
            let (dmin, _) = classify(le_u16(&blk[82..]) as u64, "F16");
            visit_q2_k_terms(blk, |dq, m| {
                see(d * dq as f64 - dmin * m as f64);
            });
        }
        QuantScaleFormat::Q3K => {
            let (d, _) = classify(le_u16(&blk[108..]) as u64, "F16");
            visit_q3_k_products(blk, |p| {
                see(d * p as f64);
            });
        }
        QuantScaleFormat::Q4K => {
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
        QuantScaleFormat::Q5K => {
            let (d, _) = classify(le_u16(blk) as u64, "F16");
            let (dmin, _) = classify(le_u16(&blk[2..]) as u64, "F16");
            visit_q5_k_terms(blk, |dq, m| {
                see(d * dq as f64 - dmin * m as f64);
            });
        }
        QuantScaleFormat::Q6K => {
            let (d, _) = classify(le_u16(&blk[208..]) as u64, "F16");
            visit_q6_k_products(blk, |p| {
                see(d * p as f64);
            });
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

/// Optional diagnostics beyond the zero-false-positive B3-2A baseline.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScanPolicy {
    pub strict_subnormal: bool,
    pub check_extreme: bool,
}

/// Every default B3-2A finding of one block, in field order. Zero, FP16
/// subnormals and finite values pass; only exponent `0x1f` encodings fail.
pub fn scan_block(fmt: impl Into<QuantScaleFormat>, blk: &[u8], max_abs: f64) -> Vec<BlockFinding> {
    scan_block_with_policy(fmt.into(), blk, max_abs, ScanPolicy::default())
}

/// B3-2A findings plus explicitly requested diagnostics, in field order and
/// with any block-level extreme finding last.
pub fn scan_block_with_policy(fmt: impl Into<QuantScaleFormat>, blk: &[u8], max_abs: f64, policy: ScanPolicy) -> Vec<BlockFinding> {
    let fmt = fmt.into();
    let mut out = Vec::new();
    let mut attributed = false;
    for f in block_fields(fmt, blk) {
        let class = match f.kind {
            Kind::Nan => Some(KIND_NAN),
            Kind::Inf => Some(KIND_INF),
            Kind::Subnormal if policy.strict_subnormal => Some(KIND_SUBNORMAL),
            Kind::Subnormal => None,
            Kind::Normal => None,
        };
        if let Some(kind) = class {
            out.push(BlockFinding { kind, field: f.name, offset: f.offset, bits: f.bits });
        }
        attributed |= matches!(f.kind, Kind::Nan | Kind::Inf);
        if policy.check_extreme && f.extreme(max_abs) {
            out.push(BlockFinding { kind: KIND_EXTREME, field: f.name, offset: f.offset, bits: f.bits });
            attributed = true;
        }
    }
    if policy.check_extreme && !attributed && realized_max_abs(fmt, blk) > max_abs {
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
pub fn repair_block(fmt: impl Into<QuantScaleFormat>, blk: &mut [u8], max_abs: f64, r: Repairs) -> Vec<&'static str> {
    let fmt = fmt.into();
    let fields = block_fields(fmt, blk);
    if r.clamp_nonfinite && fields.iter().any(|f| matches!(f.kind, Kind::Nan | Kind::Inf)) {
        let done = fields
            .iter()
            .filter_map(|f| match f.kind {
                Kind::Nan => Some(KIND_NAN),
                Kind::Inf => Some(KIND_INF),
                _ => None,
            })
            .collect();
        blk.fill(0);
        return done;
    }
    let mut done = Vec::new();
    for f in fields {
        let sign = f.bits & 0x8000;
        let repl: Option<(u16, &'static str)> = if r.flush_subnormal && f.kind == Kind::Subnormal {
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
    fn b3_2a_default_allows_zero_subnormal_and_extreme_finite_scales() {
        let mut q8 = q8_block(0.01);
        for bits in [0x0000, 0x8000, F16_SUB, 0x8001, 0x02fc, F16_MAX] {
            set(&mut q8, 0, bits);
            assert!(scan_block(QuantFormat::Q8_0, &q8, 1e4).is_empty(), "{:#06x}", bits);
        }
        let mut q4 = q4k_block();
        for off in [0, 2] {
            for bits in [0x0000, F16_SUB, 0x02fc, F16_MAX] {
                set(&mut q4, off, bits);
                assert!(scan_block(QuantFormat::Q4K, &q4, 1e4).is_empty(), "{:#06x} at {}", bits, off);
            }
        }
    }

    #[test]
    fn b3_2a_default_detects_nan_and_inf_per_field() {
        for (bits, kind) in [(F16_NAN, KIND_NAN), (F16_POS_INF, KIND_INF), (F16_NEG_INF, KIND_INF)] {
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
    fn optional_diagnostics_detect_subnormal_and_realized_extreme() {
        let strict = ScanPolicy { strict_subnormal: true, check_extreme: true };
        let mut sub = q8_block(0.01);
        set(&mut sub, 0, F16_SUB);
        assert_eq!(kinds(&scan_block_with_policy(QuantFormat::Q8_0, &sub, 1e4, strict)), vec![(KIND_SUBNORMAL, "d")]);
        // 65504 * 127 > 1e4: extreme
        let mut b = q8_block(0.01);
        set(&mut b, 0, F16_MAX);
        assert_eq!(kinds(&scan_block_with_policy(QuantFormat::Q8_0, &b, 1e4, strict)), vec![(KIND_EXTREME, "d")]);
        // the same scale over an all-zero block multiplies nothing: clean
        let mut z = vec![0u8; Q8_0_BLOCK_BYTES];
        set(&mut z, 0, F16_MAX);
        assert!(scan_block_with_policy(QuantFormat::Q8_0, &z, 1e4, strict).is_empty());
        // boundary: |d| * M == max_abs is allowed, one ulp above is not
        let mut one = vec![0u8; Q8_0_BLOCK_BYTES];
        one[2] = 100; // q = 100
        set(&mut one, 0, 0x5640); // f16 100.0
        assert!(scan_block_with_policy(QuantFormat::Q8_0, &one, 1e4, strict).is_empty());
        set(&mut one, 0, 0x5641);
        assert_eq!(kinds(&scan_block_with_policy(QuantFormat::Q8_0, &one, 1e4, strict)), vec![(KIND_EXTREME, "d")]);
        // Q4_K dmin extreme
        let mut k = q4k_block();
        set(&mut k, 2, F16_MAX);
        assert_eq!(kinds(&scan_block_with_policy(QuantFormat::Q4K, &k, 1e4, strict)), vec![(KIND_EXTREME, "dmin")]);
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
        let strict = ScanPolicy { strict_subnormal: true, check_extreme: true };
        assert_eq!(kinds(&scan_block_with_policy(QuantFormat::Q4K, &b, 1000.0, strict)), vec![(KIND_BLOCK_EXTREME, "")]);
        // and no repair addresses it
        let mut c = b.clone();
        let all = Repairs { clamp_nonfinite: true, clamp_extreme: true, flush_subnormal: true };
        assert!(repair_block(QuantFormat::Q4K, &mut c, 1000.0, all).is_empty());
        assert_eq!(c, b);
    }

    #[test]
    fn repairs_leave_a_clean_block_and_keep_the_sign() {
        let all = Repairs { clamp_nonfinite: true, clamp_extreme: true, flush_subnormal: true };
        let strict = ScanPolicy { strict_subnormal: true, check_extreme: true };
        for fmt in [QuantFormat::Q8_0, QuantFormat::Q4K] {
            let base = if fmt == QuantFormat::Q8_0 { q8_block(0.01) } else { q4k_block() };
            let offsets: &[usize] = if fmt == QuantFormat::Q8_0 { &[0] } else { &[0, 2] };
            for &off in offsets {
                for bits in [F16_NAN, F16_POS_INF, F16_NEG_INF, F16_SUB, 0x8001, F16_MAX] {
                    let mut b = base.clone();
                    set(&mut b, off, bits);
                    let before = scan_block_with_policy(fmt, &b, 1e4, strict);
                    assert!(!before.is_empty());
                    let done = repair_block(fmt, &mut b, 1e4, all);
                    assert_eq!(done.len(), 1, "{:?} {:#06x}", fmt, bits);
                    let after = scan_block_with_policy(fmt, &b, 1e4, strict);
                    assert!(after.is_empty(), "{:?} {:#06x}@{}: {:?}", fmt, bits, off, after);
                    if matches!(bits, F16_NAN | F16_POS_INF | F16_NEG_INF) {
                        assert!(b.iter().all(|&x| x == 0), "non-finite header must zero the block");
                    } else {
                        let new = le_u16(&b[off..]);
                        assert_eq!(new & 0x8000, bits & 0x8000, "sign kept");
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
        assert!(b.iter().all(|&x| x == 0));
    }

    #[test]
    fn bounded_scale_bits_edges() {
        assert_eq!(bounded_scale_bits(0.0, 1e9), F16_MAX);
        assert_eq!(bounded_scale_bits(1.0, 1e9), F16_MAX);
        assert_eq!(classify(bounded_scale_bits(127.0, 1e4) as u64, "F16").0, 78.6875); // largest f16 <= 78.74
        assert_eq!(bounded_scale_bits(945.0, 1e-12), 0);
    }
}
