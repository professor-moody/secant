//! Quantized scale-header policy for `secant sanitize` (Milestone S-4).
//!
//! Covers all 24 reviewed GGUF block-quantized formats:
//!   * Standard: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`
//!   * K-quants: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`
//!   * I-quants: `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ1_S`, `IQ4_NL`, `IQ3_S`,
//!     `IQ2_S`, `IQ4_XS`, `IQ1_M`
//!   * Ternary & microscaling: `TQ1_0`, `TQ2_0`, `MXFP4`
//!
//! Block scale headers (`d`, `dmin`, `m`, `s`, or `MXFP4` E8M0 `e`) multiply
//! every element of their block (32 or 256 weights at once), so one non-finite
//! scale poisons the entire block. Rule B3-2A treats these fields as
//! multipliers: zero and finite subnormal values are valid, and the default
//! policy rejects only non-finite encodings (`FP16`/`FP32` NaN or Inf,
//! `IQ1_M` packed `FP16` NaN or Inf, and `MXFP4` E8M0 `0xFF` NaN).
//!
//! Finding kinds, per scale field:
//!   * `quant_scale_nan`, `quant_scale_inf`: default Rule B3-2A findings.
//!   * `quant_scale_subnormal`: optional `strict_subnormal` diagnostic.
//!   * `quant_scale_extreme`: optional `check_extreme` diagnostic for a finite
//!     value where `|scale| * M > max_abs`, where `M` is the largest multiplier
//!     the block actually applies to that field.
//!
//! and per block:
//!   * `quant_block_extreme`: no field is NaN, Inf or extreme, yet some
//!     dequantized element has `|x| > max_abs`. Only reachable when two terms
//!     combine across opposite signs (for example `Q4_K` with `d < 0, dmin > 0`).
//!     It has no single-field repair and always remains a finding.
//!
//! Repairs are opt-in. `--clamp-nonfinite` zeroes the entire block when any
//! scale header is NaN or Inf, including `d`, `dmin`, `m`, `s`, `e`, scale
//! metadata and `qs`. Optional diagnostic repairs clamp extreme finite scales
//! with `--clamp-extreme` and flush subnormals with `--flush-subnormal`.

use crate::dtype::{f32_from_e8m0_half, BITS_E8M0_NAN};
use crate::quant::{get_scale_min_k4, QuantFormat};
use crate::quant_scale_tables::{
    iq1s_grid_max_abs, IQ2S_GRID_MAX, IQ2XS_GRID_MAX, IQ2XXS_GRID_MAX, IQ3S_GRID_MAX, IQ3XXS_GRID_MAX, KVALUES_IQ4NL, KVALUES_MXFP4,
};
use crate::security::sanitize::{classify, Kind};

pub const KIND_NAN: &str = "quant_scale_nan";
pub const KIND_INF: &str = "quant_scale_inf";
pub const KIND_SUBNORMAL: &str = "quant_scale_subnormal";
pub const KIND_EXTREME: &str = "quant_scale_extreme";
pub const KIND_BLOCK_EXTREME: &str = "quant_block_extreme";

/// Binary encoding of a quantized block scale header field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleEncoding {
    /// Contiguous 2-byte little-endian IEEE-754 binary16 at `offset..offset + 2`.
    F16,
    /// Contiguous 4-byte little-endian IEEE-754 binary32 at `offset..offset + 4` (`Q8_K`).
    F32,
    /// 16-bit IEEE-754 binary16 packed in the top 4 bits of the four little-endian
    /// `u16` words of `IQ1_M` `scales[0..8]` starting at `offset` (48).
    Iq1mPackedF16,
    /// 1-byte OCP MX E8M0 shared exponent at `offset` (`MXFP4`; `0xFF` is NaN).
    E8M0,
}

impl ScaleEncoding {
    /// Read the raw field bit pattern from one block.
    pub fn read_bits(self, blk: &[u8], offset: usize) -> u32 {
        match self {
            Self::F16 => le_u16(&blk[offset..]) as u32,
            Self::F32 => le_u32(&blk[offset..]),
            Self::Iq1mPackedF16 => {
                let sc0 = le_u16(&blk[offset..]);
                let sc1 = le_u16(&blk[offset + 2..]);
                let sc2 = le_u16(&blk[offset + 4..]);
                let sc3 = le_u16(&blk[offset + 6..]);
                ((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000)) as u32
            }
            Self::E8M0 => blk[offset] as u32,
        }
    }

    /// Write a raw field bit pattern into one block, preserving any sibling
    /// packed bits (such as the 12-bit sub-block scales in `IQ1_M`).
    pub fn write_bits(self, blk: &mut [u8], offset: usize, bits: u32) {
        match self {
            Self::F16 => blk[offset..offset + 2].copy_from_slice(&(bits as u16).to_le_bytes()),
            Self::F32 => blk[offset..offset + 4].copy_from_slice(&bits.to_le_bytes()),
            Self::Iq1mPackedF16 => {
                let d = bits as u16;
                let sc0 = (le_u16(&blk[offset..]) & 0x0fff) | ((d & 0x000f) << 12);
                let sc1 = (le_u16(&blk[offset + 2..]) & 0x0fff) | ((d & 0x00f0) << 8);
                let sc2 = (le_u16(&blk[offset + 4..]) & 0x0fff) | ((d & 0x0f00) << 4);
                let sc3 = (le_u16(&blk[offset + 6..]) & 0x0fff) | (d & 0xf000);
                blk[offset..offset + 2].copy_from_slice(&sc0.to_le_bytes());
                blk[offset + 2..offset + 4].copy_from_slice(&sc1.to_le_bytes());
                blk[offset + 4..offset + 6].copy_from_slice(&sc2.to_le_bytes());
                blk[offset + 6..offset + 8].copy_from_slice(&sc3.to_le_bytes());
            }
            Self::E8M0 => blk[offset] = bits as u8,
        }
    }

    /// Decode and classify a raw field bit pattern.
    pub fn classify_bits(self, bits: u32) -> (f64, Kind) {
        match self {
            Self::F16 | Self::Iq1mPackedF16 => classify((bits & 0xffff) as u64, "F16"),
            Self::F32 => classify(bits as u64, "F32"),
            Self::E8M0 => {
                let e = bits as u8;
                if e == BITS_E8M0_NAN {
                    (f64::NAN, Kind::Nan)
                } else {
                    (f32_from_e8m0_half(e) as f64, Kind::Normal)
                }
            }
        }
    }

    /// True if the raw bits encode signed zero (`+-0.0`). Always false for E8M0.
    pub fn is_zero(self, bits: u32) -> bool {
        match self {
            Self::F16 | Self::Iq1mPackedF16 => (bits & 0x7fff) == 0,
            Self::F32 => (bits & 0x7fff_ffff) == 0,
            Self::E8M0 => false,
        }
    }

    /// Format the raw field bits as a fixed-width lowercase hex string.
    pub fn format_bits(self, bits: u32) -> String {
        match self {
            Self::F16 | Self::Iq1mPackedF16 => format!("0x{:04x}", bits as u16),
            Self::F32 => format!("0x{:08x}", bits),
            Self::E8M0 => format!("0x{:02x}", bits as u8),
        }
    }

    /// Canonical finite `1.0` bit pattern for fixtures and tests.
    pub fn unit_bits(self) -> u32 {
        match self {
            Self::F16 | Self::Iq1mPackedF16 => 0x3c00,
            Self::F32 => 0x3f80_0000,
            Self::E8M0 => 0x80,
        }
    }

    /// Canonical quiet NaN bit pattern for fixtures and tests.
    pub fn nan_bits(self) -> u32 {
        match self {
            Self::F16 | Self::Iq1mPackedF16 => 0x7e00,
            Self::F32 => 0x7fc0_0000,
            Self::E8M0 => BITS_E8M0_NAN as u32,
        }
    }

    /// Canonical `+Inf` bit pattern (`None` for E8M0 which has no Inf encoding).
    pub fn pos_inf_bits(self) -> Option<u32> {
        match self {
            Self::F16 | Self::Iq1mPackedF16 => Some(0x7c00),
            Self::F32 => Some(0x7f80_0000),
            Self::E8M0 => None,
        }
    }
}

/// Static descriptor of one scale header field inside a quantized block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldSpec {
    pub name: &'static str,
    pub offset: usize,
    pub encoding: ScaleEncoding,
}

/// GGUF block-quantized formats scanned by Rule B3-2A (all 24 reviewed
/// standard, K-quant, IQ, ternary, and MXFP4 formats).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScaleFormat {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    IQ1M,
    TQ1_0,
    TQ2_0,
    MXFP4,
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
            "Q4_0" => Some(Self::Q4_0),
            "Q4_1" => Some(Self::Q4_1),
            "Q5_0" => Some(Self::Q5_0),
            "Q5_1" => Some(Self::Q5_1),
            "Q8_0" => Some(Self::Q8_0),
            "Q8_1" => Some(Self::Q8_1),
            "Q2_K" => Some(Self::Q2K),
            "Q3_K" => Some(Self::Q3K),
            "Q4_K" => Some(Self::Q4K),
            "Q5_K" => Some(Self::Q5K),
            "Q6_K" => Some(Self::Q6K),
            "Q8_K" => Some(Self::Q8K),
            "IQ2_XXS" => Some(Self::IQ2XXS),
            "IQ2_XS" => Some(Self::IQ2XS),
            "IQ3_XXS" => Some(Self::IQ3XXS),
            "IQ1_S" => Some(Self::IQ1S),
            "IQ4_NL" => Some(Self::IQ4NL),
            "IQ3_S" => Some(Self::IQ3S),
            "IQ2_S" => Some(Self::IQ2S),
            "IQ4_XS" => Some(Self::IQ4XS),
            "IQ1_M" => Some(Self::IQ1M),
            "TQ1_0" => Some(Self::TQ1_0),
            "TQ2_0" => Some(Self::TQ2_0),
            "MXFP4" => Some(Self::MXFP4),
            _ => None,
        }
    }

    pub fn block_elems(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 | Self::IQ4NL | Self::MXFP4 => 32,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ3XXS
            | Self::IQ1S
            | Self::IQ3S
            | Self::IQ2S
            | Self::IQ4XS
            | Self::IQ1M
            | Self::TQ1_0
            | Self::TQ2_0 => 256,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ3XXS => 98,
            Self::IQ1S => 50,
            Self::IQ4NL => 18,
            Self::IQ3S => 110,
            Self::IQ2S => 82,
            Self::IQ4XS => 136,
            Self::IQ1M => 56,
            Self::TQ1_0 => 54,
            Self::TQ2_0 => 66,
            Self::MXFP4 => 17,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ1S => "IQ1_S",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ3S => "IQ3_S",
            Self::IQ2S => "IQ2_S",
            Self::IQ4XS => "IQ4_XS",
            Self::IQ1M => "IQ1_M",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::MXFP4 => "MXFP4",
        }
    }

    pub const ALL: [Self; 24] = [
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Q8K,
        Self::IQ2XXS,
        Self::IQ2XS,
        Self::IQ3XXS,
        Self::IQ1S,
        Self::IQ4NL,
        Self::IQ3S,
        Self::IQ2S,
        Self::IQ4XS,
        Self::IQ1M,
        Self::TQ1_0,
        Self::TQ2_0,
        Self::MXFP4,
    ];

    /// Scale header field names and byte offsets inside one block.
    pub fn fields(self) -> &'static [(&'static str, usize)] {
        match self {
            Self::Q4_0
            | Self::Q5_0
            | Self::Q8_0
            | Self::Q8K
            | Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ3XXS
            | Self::IQ1S
            | Self::IQ4NL
            | Self::IQ3S
            | Self::IQ2S
            | Self::IQ4XS => &[("d", 0)],
            Self::Q4_1 | Self::Q5_1 => &[("d", 0), ("m", 2)],
            Self::Q8_1 => &[("d", 0), ("s", 2)],
            Self::Q2K => &[("d", 80), ("dmin", 82)],
            Self::Q3K => &[("d", 108)],
            Self::Q4K | Self::Q5K => &[("d", 0), ("dmin", 2)],
            Self::Q6K => &[("d", 208)],
            Self::IQ1M => &[("d", 48)],
            Self::TQ1_0 => &[("d", 52)],
            Self::TQ2_0 => &[("d", 64)],
            Self::MXFP4 => &[("e", 0)],
        }
    }

    /// Full scale header field descriptors (name, byte offset, and binary encoding).
    pub fn field_specs(self) -> &'static [FieldSpec] {
        const F16_D0: &[FieldSpec] = &[FieldSpec { name: "d", offset: 0, encoding: ScaleEncoding::F16 }];
        const F16_DM0: &[FieldSpec] = &[
            FieldSpec { name: "d", offset: 0, encoding: ScaleEncoding::F16 },
            FieldSpec { name: "m", offset: 2, encoding: ScaleEncoding::F16 },
        ];
        const F16_DS0: &[FieldSpec] = &[
            FieldSpec { name: "d", offset: 0, encoding: ScaleEncoding::F16 },
            FieldSpec { name: "s", offset: 2, encoding: ScaleEncoding::F16 },
        ];
        const F16_DDMIN0: &[FieldSpec] = &[
            FieldSpec { name: "d", offset: 0, encoding: ScaleEncoding::F16 },
            FieldSpec { name: "dmin", offset: 2, encoding: ScaleEncoding::F16 },
        ];
        const F16_Q2K: &[FieldSpec] = &[
            FieldSpec { name: "d", offset: 80, encoding: ScaleEncoding::F16 },
            FieldSpec { name: "dmin", offset: 82, encoding: ScaleEncoding::F16 },
        ];
        const F16_Q3K: &[FieldSpec] = &[FieldSpec { name: "d", offset: 108, encoding: ScaleEncoding::F16 }];
        const F16_Q6K: &[FieldSpec] = &[FieldSpec { name: "d", offset: 208, encoding: ScaleEncoding::F16 }];
        const F32_Q8K: &[FieldSpec] = &[FieldSpec { name: "d", offset: 0, encoding: ScaleEncoding::F32 }];
        const PACKED_IQ1M: &[FieldSpec] = &[FieldSpec { name: "d", offset: 48, encoding: ScaleEncoding::Iq1mPackedF16 }];
        const F16_TQ1: &[FieldSpec] = &[FieldSpec { name: "d", offset: 52, encoding: ScaleEncoding::F16 }];
        const F16_TQ2: &[FieldSpec] = &[FieldSpec { name: "d", offset: 64, encoding: ScaleEncoding::F16 }];
        const E8M0_MXFP4: &[FieldSpec] = &[FieldSpec { name: "e", offset: 0, encoding: ScaleEncoding::E8M0 }];
        match self {
            Self::Q4_0
            | Self::Q5_0
            | Self::Q8_0
            | Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ3XXS
            | Self::IQ1S
            | Self::IQ4NL
            | Self::IQ3S
            | Self::IQ2S
            | Self::IQ4XS => F16_D0,
            Self::Q4_1 | Self::Q5_1 => F16_DM0,
            Self::Q8_1 => F16_DS0,
            Self::Q2K => F16_Q2K,
            Self::Q3K => F16_Q3K,
            Self::Q4K | Self::Q5K => F16_DDMIN0,
            Self::Q6K => F16_Q6K,
            Self::Q8K => F32_Q8K,
            Self::IQ1M => PACKED_IQ1M,
            Self::TQ1_0 => F16_TQ1,
            Self::TQ2_0 => F16_TQ2,
            Self::MXFP4 => E8M0_MXFP4,
        }
    }
}

/// One scale header field of a block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleField {
    /// `d`, `dmin`, `m`, `s`, or `e`.
    pub name: &'static str,
    /// Byte offset of the field inside the block.
    pub offset: usize,
    pub encoding: ScaleEncoding,
    /// Raw field bit pattern (`u8` for E8M0, `u16` for FP16/IQ1_M, `u32` for FP32).
    pub bits: u32,
    pub value: f64,
    pub kind: Kind,
    /// Largest multiplier the block applies to this field.
    pub multiplier: f64,
}

impl ScaleField {
    /// True if this field encodes signed zero (`+-0.0`).
    pub fn is_zero(&self) -> bool {
        self.encoding.is_zero(self.bits)
    }

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
    pub encoding: ScaleEncoding,
    pub bits: u32,
}

impl BlockFinding {
    /// Render the raw bits for JSON/SARIF receipts (`""` for block-level findings).
    pub fn format_bits(&self) -> String {
        if self.field.is_empty() {
            String::new()
        } else {
            self.encoding.format_bits(self.bits)
        }
    }
}

fn le_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn le_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn make_field(spec: FieldSpec, blk: &[u8], multiplier: f64) -> ScaleField {
    let bits = spec.encoding.read_bits(blk, spec.offset);
    let (value, kind) = spec.encoding.classify_bits(bits);
    ScaleField { name: spec.name, offset: spec.offset, encoding: spec.encoding, bits, value, kind, multiplier }
}

fn visit_q5_nibbles(qh_bytes: &[u8], qs: &[u8], mut f: impl FnMut(u8)) {
    let qh = le_u32(qh_bytes);
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
    let a0 = le_u32(&scales[0..4]);
    let a1 = le_u32(&scales[4..8]);
    let tmp = le_u32(&scales[8..12]);
    let u0 = (a0 & kmask2) | ((tmp & kmask1) << 4);
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

fn iq2_xxs_multiplier(blk: &[u8]) -> f64 {
    let mut m = 0.0f64;
    for ib32 in 0..8 {
        let base = 2 + ib32 * 8;
        let aux0 = le_u32(&blk[base..]);
        let aux1 = le_u32(&blk[base + 4..]);
        let sf = (2 * (aux1 >> 28) + 1) as f64 * 0.125;
        for l in 0..4 {
            let idx = ((aux0 >> (8 * l)) & 0xff) as usize;
            m = m.max(sf * IQ2XXS_GRID_MAX[idx] as f64);
        }
    }
    m
}

fn iq2_xs_multiplier(blk: &[u8]) -> f64 {
    let qs = &blk[2..66];
    let scales = &blk[66..74];
    let mut m = 0.0f64;
    for ib32 in 0..8 {
        let sf0 = (2 * (scales[ib32] & 0xf) as u32 + 1) as f64 * 0.125;
        let sf1 = (2 * (scales[ib32] >> 4) as u32 + 1) as f64 * 0.125;
        for l in 0..4 {
            let q = le_u16(&qs[(4 * ib32 + l) * 2..]);
            let idx = (q & 511) as usize;
            let sf = if l < 2 { sf0 } else { sf1 };
            m = m.max(sf * IQ2XS_GRID_MAX[idx] as f64);
        }
    }
    m
}

fn iq2_s_multiplier(blk: &[u8]) -> f64 {
    let qs = &blk[2..34];
    let qh = &blk[66..74];
    let scales = &blk[74..82];
    let mut m = 0.0f64;
    for ib32 in 0..8 {
        let sf0 = (2 * (scales[ib32] & 0xf) as u32 + 1) as f64 * 0.125;
        let sf1 = (2 * (scales[ib32] >> 4) as u32 + 1) as f64 * 0.125;
        for l in 0..4 {
            let idx = (qs[4 * ib32 + l] as usize) | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
            let sf = if l < 2 { sf0 } else { sf1 };
            m = m.max(sf * IQ2S_GRID_MAX[idx] as f64);
        }
    }
    m
}

fn iq3_xxs_multiplier(blk: &[u8]) -> f64 {
    let qs = &blk[2..66];
    let sc = &blk[66..98];
    let mut m = 0.0f64;
    for ib32 in 0..8 {
        let aux32 = le_u32(&sc[4 * ib32..]);
        let sf = (2 * (aux32 >> 28) + 1) as f64 * 0.25;
        for k in 0..8 {
            let idx = qs[8 * ib32 + k] as usize;
            m = m.max(sf * IQ3XXS_GRID_MAX[idx] as f64);
        }
    }
    m
}

fn iq3_s_multiplier(blk: &[u8]) -> f64 {
    let qs = &blk[2..66];
    let qh = &blk[66..74];
    let scales = &blk[106..110];
    let mut m = 0.0f64;
    for ib64 in 0..4 {
        let sf1 = (1 + 2 * (scales[ib64] & 0xf) as u32) as f64;
        let sf2 = (1 + 2 * (scales[ib64] >> 4) as u32) as f64;
        let qh0 = qh[2 * ib64] as usize;
        let base0 = 16 * ib64;
        for l in 0..4 {
            let idx1 = (qs[base0 + 2 * l] as usize) | ((qh0 << (8 - 2 * l)) & 256);
            let idx2 = (qs[base0 + 2 * l + 1] as usize) | ((qh0 << (7 - 2 * l)) & 256);
            m = m.max(sf1 * IQ3S_GRID_MAX[idx1] as f64).max(sf1 * IQ3S_GRID_MAX[idx2] as f64);
        }
        let qh1 = qh[2 * ib64 + 1] as usize;
        let base1 = base0 + 8;
        for l in 0..4 {
            let idx1 = (qs[base1 + 2 * l] as usize) | ((qh1 << (8 - 2 * l)) & 256);
            let idx2 = (qs[base1 + 2 * l + 1] as usize) | ((qh1 << (7 - 2 * l)) & 256);
            m = m.max(sf2 * IQ3S_GRID_MAX[idx1] as f64).max(sf2 * IQ3S_GRID_MAX[idx2] as f64);
        }
    }
    m
}

fn iq1_s_multiplier(blk: &[u8]) -> f64 {
    let qs = &blk[2..34];
    let qh = &blk[34..50];
    let mut m = 0.0f64;
    for ib in 0..8 {
        let qhi = le_u16(&qh[2 * ib..]);
        let sf = (2 * ((qhi >> 12) & 7) + 1) as f64;
        let neg_delta = (qhi & 0x8000) != 0;
        for l in 0..4 {
            let idx = (qs[4 * ib + l] as usize) | ((((qhi >> (3 * l)) & 7) as usize) << 8);
            m = m.max(sf * iq1s_grid_max_abs(idx, neg_delta));
        }
    }
    m
}

fn iq1_m_multiplier(blk: &[u8]) -> f64 {
    let qs = &blk[0..32];
    let qh = &blk[32..48];
    let sc = [le_u16(&blk[48..]), le_u16(&blk[50..]), le_u16(&blk[52..]), le_u16(&blk[54..])];
    let mut m = 0.0f64;
    for ib in 0..8 {
        let scw = sc[ib / 2];
        let shift = 6 * (ib % 2);
        let sf1 = (2 * ((scw >> shift) & 7) + 1) as f64;
        let sf2 = (2 * ((scw >> (shift + 3)) & 7) + 1) as f64;
        let qsb = &qs[4 * ib..4 * ib + 4];
        let qhb = &qh[2 * ib..2 * ib + 2];
        let idx0 = (qsb[0] as usize) | (((qhb[0] as usize) << 8) & 0x700);
        let idx1 = (qsb[1] as usize) | (((qhb[0] as usize) << 4) & 0x700);
        let idx2 = (qsb[2] as usize) | (((qhb[1] as usize) << 8) & 0x700);
        let idx3 = (qsb[3] as usize) | (((qhb[1] as usize) << 4) & 0x700);
        m = m
            .max(sf1 * iq1s_grid_max_abs(idx0, (qhb[0] & 0x08) != 0))
            .max(sf1 * iq1s_grid_max_abs(idx1, (qhb[0] & 0x80) != 0))
            .max(sf2 * iq1s_grid_max_abs(idx2, (qhb[1] & 0x08) != 0))
            .max(sf2 * iq1s_grid_max_abs(idx3, (qhb[1] & 0x80) != 0));
    }
    m
}

fn iq4_nl_multiplier(blk: &[u8]) -> f64 {
    let mut m = 0u32;
    for &b in &blk[2..18] {
        m = m.max(KVALUES_IQ4NL[(b & 0xf) as usize].unsigned_abs() as u32).max(KVALUES_IQ4NL[(b >> 4) as usize].unsigned_abs() as u32);
    }
    m as f64
}

fn iq4_xs_multiplier(blk: &[u8]) -> f64 {
    let scales_h = le_u16(&blk[2..4]);
    let scales_l = &blk[4..8];
    let qs = &blk[8..136];
    let mut m = 0u32;
    for ib in 0..8 {
        let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32) | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
        let dl_m = (ls - 32).unsigned_abs();
        for &b in &qs[16 * ib..16 * ib + 16] {
            let q0 = KVALUES_IQ4NL[(b & 0xf) as usize].unsigned_abs() as u32;
            let q1 = KVALUES_IQ4NL[(b >> 4) as usize].unsigned_abs() as u32;
            m = m.max(dl_m * q0).max(dl_m * q1);
        }
    }
    m as f64
}

fn tq1_0_multiplier(blk: &[u8]) -> f64 {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let mut m = 0u32;
    for &b in &blk[0..48] {
        for &p in &POW3 {
            let q = b.wrapping_mul(p);
            let xi = (((q as u16) * 3) >> 8) as i32 - 1;
            m = m.max(xi.unsigned_abs());
            if m == 1 {
                return 1.0;
            }
        }
    }
    for &b in &blk[48..52] {
        for &p in &POW3[..4] {
            let q = b.wrapping_mul(p);
            let xi = (((q as u16) * 3) >> 8) as i32 - 1;
            m = m.max(xi.unsigned_abs());
            if m == 1 {
                return 1.0;
            }
        }
    }
    m as f64
}

fn tq2_0_multiplier(blk: &[u8]) -> f64 {
    let mut m = 0u32;
    for &b in &blk[0..64] {
        for l in 0..4 {
            let q = ((b >> (2 * l)) & 3) as i32 - 1;
            m = m.max(q.unsigned_abs());
            if m == 2 {
                return 2.0;
            }
        }
    }
    m as f64
}

fn mxfp4_multiplier(blk: &[u8]) -> f64 {
    let mut m = 0u32;
    for &b in &blk[1..17] {
        m = m.max(KVALUES_MXFP4[(b & 0xf) as usize].unsigned_abs() as u32).max(KVALUES_MXFP4[(b >> 4) as usize].unsigned_abs() as u32);
    }
    m as f64
}

/// Fast header-only scale fields of one block (`multiplier` set to `0.0`).
/// Used on the default B3-2A hot path (`check_extreme == false`) so scanning a
/// multi-gigabyte GGUF inspects only the scale header bytes of every block.
pub fn header_fields(fmt: impl Into<QuantScaleFormat>, blk: &[u8]) -> Vec<ScaleField> {
    let fmt = fmt.into();
    assert_eq!(blk.len(), fmt.block_bytes(), "header_fields: one {} block", fmt.name());
    fmt.field_specs().iter().map(|&spec| make_field(spec, blk, 0.0)).collect()
}

/// The scale fields of one block, with their exact realized multipliers. `blk`
/// must be exactly one block of `fmt`.
pub fn block_fields(fmt: impl Into<QuantScaleFormat>, blk: &[u8]) -> Vec<ScaleField> {
    let fmt = fmt.into();
    assert_eq!(blk.len(), fmt.block_bytes(), "block_fields: one {} block", fmt.name());
    let specs = fmt.field_specs();
    match fmt {
        QuantScaleFormat::Q4_0 => {
            let mut m = 0u32;
            for &b in &blk[2..18] {
                m = m.max(((b & 0xF) as i32 - 8).unsigned_abs());
                m = m.max(((b >> 4) as i32 - 8).unsigned_abs());
            }
            vec![make_field(specs[0], blk, m as f64)]
        }
        QuantScaleFormat::Q4_1 => {
            let mut m = 0u32;
            for &b in &blk[4..20] {
                m = m.max((b & 0xF) as u32).max((b >> 4) as u32);
            }
            vec![make_field(specs[0], blk, m as f64), make_field(specs[1], blk, 1.0)]
        }
        QuantScaleFormat::Q5_0 => {
            let mut m = 0u32;
            visit_q5_nibbles(&blk[2..6], &blk[6..22], |q| {
                m = m.max((q as i32 - 16).unsigned_abs());
            });
            vec![make_field(specs[0], blk, m as f64)]
        }
        QuantScaleFormat::Q5_1 => {
            let mut m = 0u32;
            visit_q5_nibbles(&blk[4..8], &blk[8..24], |q| {
                m = m.max(q as u32);
            });
            vec![make_field(specs[0], blk, m as f64), make_field(specs[1], blk, 1.0)]
        }
        QuantScaleFormat::Q8_0 => {
            let m = blk[2..].iter().map(|&q| (q as i8 as i32).unsigned_abs()).max().unwrap_or(0);
            vec![make_field(specs[0], blk, m as f64)]
        }
        QuantScaleFormat::Q8_1 => {
            let m = blk[4..36].iter().map(|&q| (q as i8 as i32).unsigned_abs()).max().unwrap_or(0);
            vec![make_field(specs[0], blk, m as f64), make_field(specs[1], blk, 1.0)]
        }
        QuantScaleFormat::Q2K => {
            let (mut md, mut mm) = (0u32, 0u32);
            visit_q2_k_terms(blk, |dq, m| {
                md = md.max(dq);
                mm = mm.max(m);
            });
            vec![make_field(specs[0], blk, md as f64), make_field(specs[1], blk, mm as f64)]
        }
        QuantScaleFormat::Q3K => {
            let mut md = 0u32;
            visit_q3_k_products(blk, |p| {
                md = md.max(p.unsigned_abs());
            });
            vec![make_field(specs[0], blk, md as f64)]
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
            vec![make_field(specs[0], blk, md as f64), make_field(specs[1], blk, mm as f64)]
        }
        QuantScaleFormat::Q5K => {
            let (mut md, mut mm) = (0u32, 0u32);
            visit_q5_k_terms(blk, |dq, m| {
                md = md.max(dq);
                mm = mm.max(m);
            });
            vec![make_field(specs[0], blk, md as f64), make_field(specs[1], blk, mm as f64)]
        }
        QuantScaleFormat::Q6K => {
            let mut md = 0u32;
            visit_q6_k_products(blk, |p| {
                md = md.max(p.unsigned_abs());
            });
            vec![make_field(specs[0], blk, md as f64)]
        }
        QuantScaleFormat::Q8K => {
            let m = blk[4..260].iter().map(|&q| (q as i8 as i32).unsigned_abs()).max().unwrap_or(0);
            vec![make_field(specs[0], blk, m as f64)]
        }
        QuantScaleFormat::IQ2XXS => vec![make_field(specs[0], blk, iq2_xxs_multiplier(blk))],
        QuantScaleFormat::IQ2XS => vec![make_field(specs[0], blk, iq2_xs_multiplier(blk))],
        QuantScaleFormat::IQ3XXS => vec![make_field(specs[0], blk, iq3_xxs_multiplier(blk))],
        QuantScaleFormat::IQ1S => vec![make_field(specs[0], blk, iq1_s_multiplier(blk))],
        QuantScaleFormat::IQ4NL => vec![make_field(specs[0], blk, iq4_nl_multiplier(blk))],
        QuantScaleFormat::IQ3S => vec![make_field(specs[0], blk, iq3_s_multiplier(blk))],
        QuantScaleFormat::IQ2S => vec![make_field(specs[0], blk, iq2_s_multiplier(blk))],
        QuantScaleFormat::IQ4XS => vec![make_field(specs[0], blk, iq4_xs_multiplier(blk))],
        QuantScaleFormat::IQ1M => vec![make_field(specs[0], blk, iq1_m_multiplier(blk))],
        QuantScaleFormat::TQ1_0 => vec![make_field(specs[0], blk, tq1_0_multiplier(blk))],
        QuantScaleFormat::TQ2_0 => vec![make_field(specs[0], blk, tq2_0_multiplier(blk))],
        QuantScaleFormat::MXFP4 => vec![make_field(specs[0], blk, mxfp4_multiplier(blk))],
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
        QuantScaleFormat::Q8_1 => {
            let (d, _) = classify(le_u16(blk) as u64, "F16");
            for &q in &blk[4..36] {
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
        QuantScaleFormat::Q8K => {
            let (d, _) = classify(le_u32(blk) as u64, "F32");
            for &q in &blk[4..260] {
                see(d * (q as i8) as f64);
            }
        }
        QuantScaleFormat::IQ2XXS
        | QuantScaleFormat::IQ2XS
        | QuantScaleFormat::IQ3XXS
        | QuantScaleFormat::IQ1S
        | QuantScaleFormat::IQ4NL
        | QuantScaleFormat::IQ3S
        | QuantScaleFormat::IQ2S
        | QuantScaleFormat::IQ4XS
        | QuantScaleFormat::IQ1M
        | QuantScaleFormat::TQ1_0
        | QuantScaleFormat::TQ2_0
        | QuantScaleFormat::MXFP4 => {
            // Single-multiplier formats: every dequantized element is +/-scale * c_i.
            let f = block_fields(fmt, blk)[0];
            if f.value.is_nan() {
                return f64::NAN;
            }
            if f.value.is_infinite() {
                return if f.multiplier == 0.0 { f64::NAN } else { f64::INFINITY };
            }
            return f.value.abs() * f.multiplier;
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

/// Largest non-negative finite f32 `v` with `v * multiplier <= max_abs`.
pub fn bounded_scale_bits_f32(multiplier: f64, max_abs: f64) -> u32 {
    let (mut lo, mut hi) = (0u32, 0x7f7f_ffffu32);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let v = f32::from_bits(mid) as f64;
        let ok = if multiplier > 0.0 { v * multiplier <= max_abs } else { v <= max_abs };
        if ok {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Largest valid E8M0 exponent byte `e in 0x00..=0xFE` with
/// `f32_from_e8m0_half(e) * multiplier <= max_abs`, or `None` if even `e = 0`
/// exceeds `max_abs` (E8M0 has no zero encoding).
pub fn bounded_scale_bits_e8m0(multiplier: f64, max_abs: f64) -> Option<u8> {
    let check = |e: u8| {
        let v = f32_from_e8m0_half(e) as f64;
        if multiplier > 0.0 {
            v * multiplier <= max_abs
        } else {
            v <= max_abs
        }
    };
    if !check(0) {
        return None;
    }
    let (mut lo, mut hi) = (0u8, 0xfeu8);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if check(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo)
}

/// Optional diagnostics beyond the zero-false-positive B3-2A baseline.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScanPolicy {
    pub strict_subnormal: bool,
    pub check_extreme: bool,
}

/// Every default B3-2A finding of one block, in field order. Zero, finite
/// subnormals and finite values pass; only non-finite scale encodings fail.
pub fn scan_block(fmt: impl Into<QuantScaleFormat>, blk: &[u8], max_abs: f64) -> Vec<BlockFinding> {
    scan_block_with_policy(fmt.into(), blk, max_abs, ScanPolicy::default())
}

/// B3-2A findings plus explicitly requested diagnostics, in field order and
/// with any block-level extreme finding last.
pub fn scan_block_with_policy(fmt: impl Into<QuantScaleFormat>, blk: &[u8], max_abs: f64, policy: ScanPolicy) -> Vec<BlockFinding> {
    let fmt = fmt.into();
    let fields = if policy.check_extreme { block_fields(fmt, blk) } else { header_fields(fmt, blk) };
    let mut out = Vec::new();
    let mut attributed = false;
    for f in fields {
        let class = match f.kind {
            Kind::Nan => Some(KIND_NAN),
            Kind::Inf => Some(KIND_INF),
            Kind::Subnormal if policy.strict_subnormal => Some(KIND_SUBNORMAL),
            Kind::Subnormal => None,
            Kind::Normal => None,
        };
        if let Some(kind) = class {
            out.push(BlockFinding { kind, field: f.name, offset: f.offset, encoding: f.encoding, bits: f.bits });
        }
        attributed |= matches!(f.kind, Kind::Nan | Kind::Inf);
        if policy.check_extreme && f.extreme(max_abs) {
            out.push(BlockFinding { kind: KIND_EXTREME, field: f.name, offset: f.offset, encoding: f.encoding, bits: f.bits });
            attributed = true;
        }
    }
    if policy.check_extreme && !attributed && realized_max_abs(fmt, blk) > max_abs {
        out.push(BlockFinding { kind: KIND_BLOCK_EXTREME, field: "", offset: 0, encoding: ScaleEncoding::F16, bits: 0 });
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
    let fields = if r.clamp_extreme { block_fields(fmt, blk) } else { header_fields(fmt, blk) };
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
        let repl: Option<(u32, &'static str)> = match f.encoding {
            ScaleEncoding::F16 | ScaleEncoding::Iq1mPackedF16 => {
                let sign = f.bits & 0x8000;
                if r.flush_subnormal && f.kind == Kind::Subnormal {
                    Some((sign, KIND_SUBNORMAL))
                } else if r.clamp_extreme && f.extreme(max_abs) {
                    Some((bounded_scale_bits(f.multiplier, max_abs) as u32 | sign, KIND_EXTREME))
                } else {
                    None
                }
            }
            ScaleEncoding::F32 => {
                let sign = f.bits & 0x8000_0000;
                if r.flush_subnormal && f.kind == Kind::Subnormal {
                    Some((sign, KIND_SUBNORMAL))
                } else if r.clamp_extreme && f.extreme(max_abs) {
                    Some((bounded_scale_bits_f32(f.multiplier, max_abs) | sign, KIND_EXTREME))
                } else {
                    None
                }
            }
            ScaleEncoding::E8M0 => {
                if r.clamp_extreme && f.extreme(max_abs) {
                    bounded_scale_bits_e8m0(f.multiplier, max_abs).map(|e| (e as u32, KIND_EXTREME))
                } else {
                    None
                }
            }
        };
        if let Some((bits, kind)) = repl {
            f.encoding.write_bits(blk, f.offset, bits);
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

    /// Build one deterministic non-trivial clean block for any of the 24 formats.
    fn clean_block(fmt: QuantScaleFormat) -> Vec<u8> {
        let mut b = vec![0u8; fmt.block_bytes()];
        // Populate payload bytes with deterministic non-zero values so multipliers > 0.
        for (i, byte) in b.iter_mut().enumerate() {
            *byte = ((i * 29 + 7) % 251) as u8;
        }
        // Set all scale header fields to finite 1.0 (preserving sibling bits for IQ1_M).
        for &spec in fmt.field_specs() {
            spec.encoding.write_bits(&mut b, spec.offset, spec.encoding.unit_bits());
        }
        b
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
                assert_eq!(got[0].bits, bits as u32);
            }
        }
    }

    #[test]
    fn all_24_formats_clean_poison_and_repair_matrix() {
        assert_eq!(QuantScaleFormat::ALL.len(), 24);
        let strict = ScanPolicy { strict_subnormal: true, check_extreme: true };
        let all_repairs = Repairs { clamp_nonfinite: true, clamp_extreme: true, flush_subnormal: true };

        for fmt in QuantScaleFormat::ALL {
            assert_eq!(QuantScaleFormat::from_dtype(fmt.name()), Some(fmt));
            assert_eq!(fmt.fields().len(), fmt.field_specs().len());

            // 1. Clean finite block passes under both default and strict policies.
            let base = clean_block(fmt);
            assert!(scan_block(fmt, &base, 1e4).is_empty(), "{} clean default", fmt.name());
            assert!(scan_block_with_policy(fmt, &base, 1e4, strict).is_empty(), "{} clean strict", fmt.name());
            // All-zero block also passes cleanly under default and strict policies.
            let zero_blk = vec![0u8; fmt.block_bytes()];
            assert!(scan_block(fmt, &zero_blk, 1e4).is_empty(), "{} zero default", fmt.name());
            assert!(scan_block_with_policy(fmt, &zero_blk, 1e4, strict).is_empty(), "{} zero strict", fmt.name());

            for &spec in fmt.field_specs() {
                // 2. Planted NaN is caught and zeroed cleanly by --clamp-nonfinite.
                let mut b_nan = base.clone();
                spec.encoding.write_bits(&mut b_nan, spec.offset, spec.encoding.nan_bits());
                let got = scan_block(fmt, &b_nan, 1e4);
                assert_eq!(kinds(&got), vec![(KIND_NAN, spec.name)], "{} NaN", fmt.name());
                assert_eq!(got[0].offset, spec.offset);
                assert_eq!(got[0].bits, spec.encoding.nan_bits());
                assert!(realized_max_abs(fmt, &b_nan).is_nan() || fmt == QuantScaleFormat::Q8_1 && spec.name == "s");
                let done = repair_block(fmt, &mut b_nan, 1e4, Repairs { clamp_nonfinite: true, ..Default::default() });
                assert_eq!(done, vec![KIND_NAN], "{} NaN repair", fmt.name());
                assert!(b_nan.iter().all(|&x| x == 0), "{} NaN repair zeroes block", fmt.name());
                assert!(scan_block_with_policy(fmt, &b_nan, 1e4, strict).is_empty(), "{} post-NaN rescan", fmt.name());

                // 3. Planted +Inf (where representable) is caught and zeroed cleanly.
                if let Some(inf_bits) = spec.encoding.pos_inf_bits() {
                    let mut b_inf = base.clone();
                    spec.encoding.write_bits(&mut b_inf, spec.offset, inf_bits);
                    let got = scan_block(fmt, &b_inf, 1e4);
                    assert_eq!(kinds(&got), vec![(KIND_INF, spec.name)], "{} Inf", fmt.name());
                    assert_eq!(got[0].offset, spec.offset);
                    assert_eq!(got[0].bits, inf_bits);
                    let done = repair_block(fmt, &mut b_inf, 1e4, Repairs { clamp_nonfinite: true, ..Default::default() });
                    assert_eq!(done, vec![KIND_INF], "{} Inf repair", fmt.name());
                    assert!(b_inf.iter().all(|&x| x == 0));
                    assert!(scan_block_with_policy(fmt, &b_inf, 1e4, strict).is_empty());
                }

                // 4. Planted extreme finite scale passes default B3-2A, fails check_extreme, and clamps cleanly.
                let extreme_bits = match spec.encoding {
                    ScaleEncoding::F16 | ScaleEncoding::Iq1mPackedF16 => F16_MAX as u32,
                    ScaleEncoding::F32 => 0x7f7f_ffffu32,
                    ScaleEncoding::E8M0 => 0xfeu32, // 2^126
                };
                let mut b_ext = base.clone();
                spec.encoding.write_bits(&mut b_ext, spec.offset, extreme_bits);
                assert!(scan_block(fmt, &b_ext, 1e4).is_empty(), "{} extreme finite must pass default B3-2A", fmt.name());
                let got_ext = scan_block_with_policy(fmt, &b_ext, 1e4, strict);
                assert_eq!(kinds(&got_ext), vec![(KIND_EXTREME, spec.name)], "{} extreme strict", fmt.name());
                let done_ext = repair_block(fmt, &mut b_ext, 1e4, all_repairs);
                assert_eq!(done_ext, vec![KIND_EXTREME], "{} extreme repair", fmt.name());
                let after_ext = scan_block_with_policy(fmt, &b_ext, 1e4, strict);
                if matches!(fmt, QuantScaleFormat::Q4_1 | QuantScaleFormat::Q5_1) && spec.name == "m" {
                    // Q4_1 / Q5_1 add `d * q + m`: clamping `m` alone to `max_abs`
                    // with positive `d = 1.0` leaves `d * q + m > max_abs`, which
                    // is reported as the unrepairable block-level finding.
                    assert_eq!(kinds(&after_ext), vec![(KIND_BLOCK_EXTREME, "")], "{} additive block extreme", fmt.name());
                } else {
                    assert!(after_ext.is_empty(), "{} post-extreme rescan: {:?}", fmt.name(), after_ext);
                    assert!(realized_max_abs(fmt, &b_ext) <= 1e4, "{} post-extreme realized_max_abs", fmt.name());
                }
            }
        }
    }

    #[test]
    fn iq1_m_packed_scale_preserves_subblock_scales_on_read_and_repair() {
        let mut blk = clean_block(QuantScaleFormat::IQ1M);
        // Save the lower 12 bits of all 4 u16 words in scales[0..8] (bytes 48..56).
        let orig_low12: [u16; 4] =
            [le_u16(&blk[48..]) & 0x0fff, le_u16(&blk[50..]) & 0x0fff, le_u16(&blk[52..]) & 0x0fff, le_u16(&blk[54..]) & 0x0fff];
        // Write negative subnormal 0x8001 into packed d.
        ScaleEncoding::Iq1mPackedF16.write_bits(&mut blk, 48, 0x8001);
        assert_eq!(ScaleEncoding::Iq1mPackedF16.read_bits(&blk, 48), 0x8001);
        // Verify low 12 bits untouched.
        for (i, &want) in orig_low12.iter().enumerate() {
            assert_eq!(le_u16(&blk[48 + 2 * i..]) & 0x0fff, want);
        }
        // Default B3-2A allows subnormal; strict catches it; flush_subnormal flushes to -0.0 (0x8000) preserving low 12 bits.
        assert!(scan_block(QuantScaleFormat::IQ1M, &blk, 1e4).is_empty());
        let strict = ScanPolicy { strict_subnormal: true, check_extreme: false };
        let got = scan_block_with_policy(QuantScaleFormat::IQ1M, &blk, 1e4, strict);
        assert_eq!(kinds(&got), vec![(KIND_SUBNORMAL, "d")]);
        assert_eq!(got[0].format_bits(), "0x8001");
        let done = repair_block(QuantScaleFormat::IQ1M, &mut blk, 1e4, Repairs { flush_subnormal: true, ..Default::default() });
        assert_eq!(done, vec![KIND_SUBNORMAL]);
        assert_eq!(ScaleEncoding::Iq1mPackedF16.read_bits(&blk, 48), 0x8000);
        for (i, &want) in orig_low12.iter().enumerate() {
            assert_eq!(le_u16(&blk[48 + 2 * i..]) & 0x0fff, want);
        }
    }

    #[test]
    fn q8_k_f32_scale_and_mxfp4_e8m0_edge_cases() {
        let strict = ScanPolicy { strict_subnormal: true, check_extreme: true };

        // Q8_K: F32 subnormal 0x8000_0001
        let mut q8k = clean_block(QuantScaleFormat::Q8K);
        ScaleEncoding::F32.write_bits(&mut q8k, 0, 0x8000_0001);
        assert!(scan_block(QuantScaleFormat::Q8K, &q8k, 1e4).is_empty());
        let got = scan_block_with_policy(QuantScaleFormat::Q8K, &q8k, 1e4, strict);
        assert_eq!(kinds(&got), vec![(KIND_SUBNORMAL, "d")]);
        assert_eq!(got[0].format_bits(), "0x80000001");
        repair_block(QuantScaleFormat::Q8K, &mut q8k, 1e4, Repairs { flush_subnormal: true, ..Default::default() });
        assert_eq!(ScaleEncoding::F32.read_bits(&q8k, 0), 0x8000_0000);

        // MXFP4: exhaustive check across all 256 E8M0 bytes under default B3-2A
        let mut mxfp4 = clean_block(QuantScaleFormat::MXFP4);
        for e in 0u8..=255 {
            mxfp4[0] = e;
            let f = scan_block(QuantScaleFormat::MXFP4, &mxfp4, 1e4);
            if e == 0xff {
                assert_eq!(kinds(&f), vec![(KIND_NAN, "e")]);
                assert_eq!(f[0].format_bits(), "0xff");
            } else {
                assert!(f.is_empty(), "MXFP4 e=0x{:02x} should be clean under default B3-2A", e);
            }
        }
        // MXFP4 extreme unrepairable when max_abs < 2^-128 * multiplier (E8M0 has no zero)
        mxfp4[0] = 0x80;
        assert_eq!(bounded_scale_bits_e8m0(12.0, 1e-45), None);
        assert!(repair_block(QuantScaleFormat::MXFP4, &mut mxfp4, 1e-45, Repairs { clamp_extreme: true, ..Default::default() }).is_empty());
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
        // f32 and e8m0 bounds
        let f32_clamped = f32::from_bits(bounded_scale_bits_f32(127.0, 1e4)) as f64;
        assert!(f32_clamped * 127.0 <= 1e4 && (f32::from_bits(bounded_scale_bits_f32(127.0, 1e4) + 1) as f64) * 127.0 > 1e4);
        assert_eq!(bounded_scale_bits_e8m0(12.0, 1e4), Some(137)); // 2^(137-128) = 512; 512 * 12 = 6144 <= 1e4; 1024 * 12 = 12288 > 1e4
    }
}
