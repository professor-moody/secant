//! dtype labels, scalar codecs and the dtype-aware tensor buffer. Mirrors
//! `internal/wsp/{bytesops,lowprecision,semantic}.go` bit for bit.

use crate::error::{Error, Result};

pub const BITS_F32_NAN: u32 = 0x7FC0_0000;
pub const BITS_F32_POS_INF: u32 = 0x7F80_0000;
pub const BITS_F32_NEG_INF: u32 = 0xFF80_0000;
pub const BITS_BF16_NAN: u16 = 0x7FC0;
pub const BITS_BF16_POS_INF: u16 = 0x7F80;
pub const BITS_F16_NAN: u16 = 0x7E00;
pub const BITS_F16_POS_INF: u16 = 0x7C00;

pub fn le_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
pub fn le_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
pub fn le_u64(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    u64::from_le_bytes(a)
}
pub fn le_f32(b: &[u8]) -> f32 {
    f32::from_bits(le_u32(b))
}
pub fn le_f64(b: &[u8]) -> f64 {
    f64::from_bits(le_u64(b))
}
pub fn put_u16(b: &mut [u8], v: u16) {
    b[..2].copy_from_slice(&v.to_le_bytes());
}
pub fn put_u32(b: &mut [u8], v: u32) {
    b[..4].copy_from_slice(&v.to_le_bytes());
}
pub fn put_u64(b: &mut [u8], v: u64) {
    b[..8].copy_from_slice(&v.to_le_bytes());
}
pub fn put_f32(b: &mut [u8], v: f32) {
    put_u32(b, v.to_bits());
}
pub fn put_f64(b: &mut [u8], v: f64) {
    put_u64(b, v.to_bits());
}

/// float32 -> bfloat16 bit pattern, round-to-nearest-even; NaN/Inf keep the
/// top half of the f32 pattern.
pub fn bf16_from_f32(f: f32) -> u16 {
    let b = f.to_bits();
    if f.is_nan() || f.is_infinite() {
        return (b >> 16) as u16;
    }
    (b.wrapping_add(0x7fff).wrapping_add((b >> 16) & 1) >> 16) as u16
}

pub fn f32_from_bf16(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// float32 -> IEEE half. NaN stays NaN, overflow saturates to Inf, ties to
/// even, results below the smallest normal flush to a signed zero (the same
/// behaviour as the oracle, NOT full IEEE subnormal rounding).
pub fn f16_from_f32(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let frac = bits & 0x007F_FFFF;
    if exp == 128 {
        if frac != 0 {
            return (sign << 15) | 0x7E00;
        }
        return (sign << 15) | 0x7C00;
    }
    if exp > 15 {
        return (sign << 15) | 0x7C00;
    }
    if exp < -25 {
        return sign << 15;
    }
    if exp < -14 {
        let shift = (-1 - exp) as u32; // 14..24
        let mant = 0x0080_0000u32 | frac;
        let mut half = (mant >> shift) as u16;
        let half_bit = 1u32 << (shift - 1);
        let sticky_mask = half_bit - 1;
        if (mant & half_bit) != 0 && ((mant & sticky_mask) != 0 || (half & 1) != 0) {
            half = half.wrapping_add(1);
        }
        return (sign << 15) | half;
    }
    let mut half = (sign << 15) | ((((exp + 15) & 0x1F) as u16) << 10) | ((frac >> 13) as u16);
    if frac & 0x1000 != 0 && (frac & 0x0FFF != 0 || half & 1 != 0) {
        half = half.wrapping_add(1);
    }
    half
}

pub fn f32_from_f16(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as i32;
    let mut frac = (h & 0x3FF) as u32;
    match exp {
        0 => {
            if frac == 0 {
                return f32::from_bits(sign << 31);
            }
            // Decode F16 subnormal: value = frac * 2^-24, matching Go's
            // Float32FromF16 in internal/wsp/lowprecision.go.
            let mut e: i32 = 0;
            while frac & 0x400 == 0 {
                frac <<= 1;
                e -= 1;
            }
            frac &= 0x3FF;
            f32::from_bits(sign << 31 | ((127 - 15 + e + 1) as u32) << 23 | frac << 13)
        }
        0x1F => {
            if frac == 0 {
                f32::from_bits(sign << 31 | 0xFF << 23)
            } else {
                f32::from_bits(sign << 31 | 0xFF << 23 | frac << 13)
            }
        }
        _ => f32::from_bits(sign << 31 | ((exp - 15 + 127) as u32) << 23 | frac << 13),
    }
}

/// Byte width of a float/int dtype label; 0 for quantized/unknown labels.
pub fn dtype_size(dtype: &str) -> i64 {
    match dtype {
        "F32" | "I32" => 4,
        "F16" | "BF16" => 2,
        "F64" | "I64" => 8,
        _ => 0,
    }
}

pub fn is_float_dtype(d: &str) -> bool {
    matches!(d, "F32" | "F16" | "BF16" | "F64")
}

pub const SEM_NAN: &str = "nan";
pub const SEM_POS_INF: &str = "+inf";
pub const SEM_NEG_INF: &str = "-inf";
pub const SEM_MIN: &str = "min";
pub const SEM_MAX: &str = "max";
pub const SEM_ZERO: &str = "zero";
pub const SEMANTIC_TOKENS: [&str; 6] = [SEM_NAN, SEM_POS_INF, SEM_NEG_INF, SEM_MIN, SEM_MAX, SEM_ZERO];

/// dtype-exact bit pattern of one semantic token (see the oracle's table).
pub fn encode_semantic(dtype: &str, token: &str) -> Result<u64> {
    let fail = || Err(Error::msg(format!("semantic {:?} is not defined for dtype {}", token, dtype)));
    let v: u64 = match (dtype, token) {
        ("F32", SEM_NAN) => BITS_F32_NAN as u64,
        ("F32", SEM_POS_INF) => BITS_F32_POS_INF as u64,
        ("F32", SEM_NEG_INF) => BITS_F32_NEG_INF as u64,
        ("F32", SEM_MIN) => 0xFF7F_FFFF,
        ("F32", SEM_MAX) => 0x7F7F_FFFF,
        ("BF16", SEM_NAN) => BITS_BF16_NAN as u64,
        ("BF16", SEM_POS_INF) => BITS_BF16_POS_INF as u64,
        ("BF16", SEM_NEG_INF) => 0xFF80,
        ("BF16", SEM_MIN) => 0xFF7F,
        ("BF16", SEM_MAX) => 0x7F7F,
        ("F16", SEM_NAN) => BITS_F16_NAN as u64,
        ("F16", SEM_POS_INF) => BITS_F16_POS_INF as u64,
        ("F16", SEM_NEG_INF) => 0xFC00,
        ("F16", SEM_MIN) => 0xFBFF,
        ("F16", SEM_MAX) => 0x7BFF,
        ("F64", SEM_NAN) => 0x7FF8_0000_0000_0000,
        ("F64", SEM_POS_INF) => 0x7FF0_0000_0000_0000,
        ("F64", SEM_NEG_INF) => 0xFFF0_0000_0000_0000,
        ("F64", SEM_MIN) => 0xFFEF_FFFF_FFFF_FFFF,
        ("F64", SEM_MAX) => 0x7FEF_FFFF_FFFF_FFFF,
        ("I32", SEM_MIN) => 0x8000_0000,
        ("I32", SEM_MAX) => 0x7FFF_FFFF,
        ("I64", SEM_MIN) => 0x8000_0000_0000_0000,
        ("I64", SEM_MAX) => 0x7FFF_FFFF_FFFF_FFFF,
        ("F32" | "BF16" | "F16" | "F64" | "I32" | "I64", SEM_ZERO) => 0,
        ("F32" | "BF16" | "F16" | "F64" | "I32" | "I64", _) => return fail(),
        _ => return Err(Error::msg(format!("semantic encoding: unsupported dtype {:?}", dtype))),
    };
    Ok(v)
}

/// dtype-aware view over one tensor's payload bytes.
pub struct TensorBuffer {
    pub dtype: String,
    pub data: Vec<u8>,
}

impl TensorBuffer {
    pub fn new(dtype: &str, data: Vec<u8>) -> Self {
        TensorBuffer { dtype: dtype.to_string(), data }
    }

    /// Element `i` widened to float64.
    pub fn get(&self, i: i64) -> f64 {
        let i = i as usize;
        match self.dtype.as_str() {
            "F32" => le_f32(&self.data[i * 4..]) as f64,
            "F16" => f32_from_f16(le_u16(&self.data[i * 2..])) as f64,
            "BF16" => f32_from_bf16(le_u16(&self.data[i * 2..])) as f64,
            "F64" => le_f64(&self.data[i * 8..]),
            "I32" => (le_u32(&self.data[i * 4..]) as i32) as f64,
            "I64" => (le_u64(&self.data[i * 8..]) as i64) as f64,
            _ => 0.0,
        }
    }

    /// Store `v` with the tensor's dtype rounding (integer dtypes: no-op).
    pub fn set_float(&mut self, i: i64, v: f64) {
        let i = i as usize;
        match self.dtype.as_str() {
            "F32" => put_f32(&mut self.data[i * 4..], v as f32),
            "F16" => put_u16(&mut self.data[i * 2..], f16_from_f32(v as f32)),
            "BF16" => put_u16(&mut self.data[i * 2..], bf16_from_f32(v as f32)),
            "F64" => put_f64(&mut self.data[i * 8..], v),
            _ => {}
        }
    }

    /// Exact dtype-width bit pattern (low bits for the 16-bit dtypes).
    pub fn set_bits(&mut self, i: i64, bits: u32) {
        let i = i as usize;
        match self.dtype.as_str() {
            "F32" | "I32" => put_u32(&mut self.data[i * 4..], bits),
            "F16" | "BF16" => put_u16(&mut self.data[i * 2..], (bits & 0xFFFF) as u16),
            "F64" => {
                put_u32(&mut self.data[i * 8..], bits);
                put_u32(&mut self.data[i * 8 + 4..], 0);
            }
            _ => {}
        }
    }

    pub fn set_bits64(&mut self, i: i64, bits: u64) {
        let i = i as usize;
        if self.dtype == "F64" || self.dtype == "I64" {
            put_u64(&mut self.data[i * 8..], bits);
        }
    }

    /// Exact two's-complement integer store; fails closed on overflow or a
    /// non-integer dtype.
    pub fn set_int(&mut self, i: i64, v: i64) -> Result<()> {
        let iu = i as usize;
        match self.dtype.as_str() {
            "I32" => {
                if v < i32::MIN as i64 || v > i32::MAX as i64 {
                    return Err(Error::msg(format!("I64 value {} does not fit I32", v)));
                }
                put_u32(&mut self.data[iu * 4..], (v as i32) as u32);
            }
            "I64" => put_u64(&mut self.data[iu * 8..], v as u64),
            d => return Err(Error::msg(format!("SetInt: dtype {} is not an integer type", d))),
        }
        Ok(())
    }

    pub fn write_semantic(&mut self, i: i64, token: &str) -> Result<()> {
        let bits = encode_semantic(&self.dtype, token)?;
        match self.dtype.as_str() {
            "F32" | "I32" | "F16" | "BF16" => self.set_bits(i, bits as u32),
            "F64" | "I64" => self.set_bits64(i, bits),
            d => return Err(Error::msg(format!("semantic encoding: unsupported dtype {:?}", d))),
        }
        Ok(())
    }
}

/// Decode one element of `dtype` at byte offset `off` (oracle `readElem`).
pub fn read_elem(data: &[u8], dtype: &str, off: usize) -> f64 {
    match dtype {
        "F32" => le_f32(&data[off..]) as f64,
        "BF16" => f32_from_bf16(le_u16(&data[off..])) as f64,
        "F16" => f32_from_f16(le_u16(&data[off..])) as f64,
        "F64" => le_f64(&data[off..]),
        "I32" => (le_u32(&data[off..]) as i32) as f64,
        "I64" => (le_u64(&data[off..]) as i64) as f64,
        _ => 0.0,
    }
}

/// Widen `n` consecutive float elements starting at byte offset `off`.
pub fn decode_row(dtype: &str, data: &[u8], off: usize, n: usize, dst: &mut [f64]) {
    match dtype {
        "F32" => {
            for i in 0..n {
                dst[i] = le_f32(&data[off + i * 4..]) as f64;
            }
        }
        "BF16" => {
            for i in 0..n {
                dst[i] = f32_from_bf16(le_u16(&data[off + i * 2..])) as f64;
            }
        }
        "F16" => {
            for i in 0..n {
                dst[i] = f32_from_f16(le_u16(&data[off + i * 2..])) as f64;
            }
        }
        "F64" => {
            for i in 0..n {
                dst[i] = le_f64(&data[off + i * 8..]);
            }
        }
        _ => {}
    }
}
