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
pub const BITS_F8_E4M3_NAN: u8 = 0x7F;
pub const BITS_F8_E5M2_NAN: u8 = 0x7E;
pub const BITS_F8_E5M2_POS_INF: u8 = 0x7C;

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

/// Converts `f32` to an IEEE-754 half-precision (`binary16`) bit pattern:
/// NaN stays NaN (`0x7E00` payload), overflow saturates to signed Inf,
/// subnormals (`-25 <= exp < -14`) round to nearest-even with sticky-bit
/// tracking, and magnitudes below `2^-25` underflow to signed zero.
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

/// Decode one OCP `E4M3FN` byte (SafeTensors `"F8_E4M3"`): 1 sign bit,
/// 4 exponent bits (bias 7), 3 mantissa bits. `0x7F` and `0xFF` are `NaN`;
/// there is no `Inf` encoding (`0x7E` / `0xFE` decode to `±448.0`), and
/// subnormals (`exp == 0`, `mant != 0`) decode to `(-1)^s * 2^-6 * (m / 8)`.
pub fn f32_from_f8_e4m3(b: u8) -> f32 {
    let sign = ((b >> 7) & 1) as u32;
    let exp = ((b >> 3) & 0x0F) as i32;
    let mut frac = (b & 0x07) as u32;
    if exp == 0x0F && frac == 0x07 {
        return f32::from_bits((sign << 31) | BITS_F32_NAN);
    }
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e: i32 = 0;
        while frac & 0x08 == 0 {
            frac <<= 1;
            e -= 1;
        }
        frac &= 0x07;
        return f32::from_bits((sign << 31) | (((127 - 7 + e + 1) as u32) << 23) | (frac << 20));
    }
    f32::from_bits((sign << 31) | (((exp - 7 + 127) as u32) << 23) | (frac << 20))
}

/// Decode one IEEE/OCP `E5M2` byte (SafeTensors `"F8_E5M2"`): 1 sign bit,
/// 5 exponent bits (bias 15), 2 mantissa bits, IEEE `Inf` (`0x7C`/`0xFC`),
/// `NaN` (`exp == 31`, `mant != 0`), and subnormals `(-1)^s * 2^-14 * (m / 4)`.
pub fn f32_from_f8_e5m2(b: u8) -> f32 {
    let sign = ((b >> 7) & 1) as u32;
    let exp = ((b >> 2) & 0x1F) as i32;
    let mut frac = (b & 0x03) as u32;
    match exp {
        0 => {
            if frac == 0 {
                return f32::from_bits(sign << 31);
            }
            let mut e: i32 = 0;
            while frac & 0x04 == 0 {
                frac <<= 1;
                e -= 1;
            }
            frac &= 0x03;
            f32::from_bits((sign << 31) | (((127 - 15 + e + 1) as u32) << 23) | (frac << 21))
        }
        0x1F => {
            if frac == 0 {
                f32::from_bits((sign << 31) | (0xFF << 23))
            } else {
                f32::from_bits((sign << 31) | BITS_F32_NAN | (frac << 20))
            }
        }
        _ => f32::from_bits((sign << 31) | (((exp - 15 + 127) as u32) << 23) | (frac << 21)),
    }
}

/// Byte width of a float/int/bool dtype label; 0 for quantized/unknown labels.
pub fn dtype_size(dtype: &str) -> i64 {
    match dtype {
        "F8_E4M3" | "F8_E5M2" | "U8" | "BOOL" => 1,
        "F32" | "I32" => 4,
        "F16" | "BF16" => 2,
        "F64" | "I64" => 8,
        _ => 0,
    }
}

pub fn is_float_dtype(d: &str) -> bool {
    matches!(d, "F32" | "F16" | "BF16" | "F64")
}

pub fn is_fp8_dtype(d: &str) -> bool {
    matches!(d, "F8_E4M3" | "F8_E5M2")
}

pub fn is_readable_float_dtype(d: &str) -> bool {
    is_float_dtype(d) || is_fp8_dtype(d)
}

pub fn is_integer_or_bool_dtype(d: &str) -> bool {
    matches!(d, "I32" | "I64" | "U8" | "BOOL")
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
            "F8_E4M3" => f32_from_f8_e4m3(self.data[i]) as f64,
            "F8_E5M2" => f32_from_f8_e5m2(self.data[i]) as f64,
            "U8" | "BOOL" => self.data[i] as f64,
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
        "F8_E4M3" => f32_from_f8_e4m3(data[off]) as f64,
        "F8_E5M2" => f32_from_f8_e5m2(data[off]) as f64,
        "U8" | "BOOL" => data[off] as f64,
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
        "F8_E4M3" => {
            for i in 0..n {
                dst[i] = f32_from_f8_e4m3(data[off + i]) as f64;
            }
        }
        "F8_E5M2" => {
            for i in 0..n {
                dst[i] = f32_from_f8_e5m2(data[off + i]) as f64;
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f8_e4m3_exhaustive_256() {
        assert_eq!(dtype_size("F8_E4M3"), 1);
        assert!(is_fp8_dtype("F8_E4M3"));
        assert!(!is_float_dtype("F8_E4M3"));
        assert!(is_readable_float_dtype("F8_E4M3"));

        // Signed zeros
        assert_eq!(f32_from_f8_e4m3(0x00).to_bits(), 0.0f32.to_bits());
        assert_eq!(f32_from_f8_e4m3(0x80).to_bits(), (-0.0f32).to_bits());

        // NaN only at 0x7F / 0xFF; NO Inf anywhere in E4M3FN
        assert!(f32_from_f8_e4m3(0x7F).is_nan());
        assert!(f32_from_f8_e4m3(0xFF).is_nan());
        assert_eq!(f32_from_f8_e4m3(0x7F).to_bits() >> 31, 0);
        assert_eq!(f32_from_f8_e4m3(0xFF).to_bits() >> 31, 1);

        // Anchor normals: 1.0 (0x38), -1.0 (0xB8), 256.0 (0x78), 448.0 (0x7E), -448.0 (0xFE), min normal 2^-6 (0x08)
        assert_eq!(f32_from_f8_e4m3(0x38), 1.0);
        assert_eq!(f32_from_f8_e4m3(0xB8), -1.0);
        assert_eq!(f32_from_f8_e4m3(0x78), 256.0);
        assert_eq!(f32_from_f8_e4m3(0x7E), 448.0);
        assert_eq!(f32_from_f8_e4m3(0xFE), -448.0);
        assert_eq!(f32_from_f8_e4m3(0x08), 2.0f32.powi(-6));

        // Exhaustive check across all 256 byte patterns
        let mut nan_count = 0;
        for b in 0u8..=255 {
            let v = f32_from_f8_e4m3(b);
            assert!(!v.is_infinite(), "E4M3FN byte 0x{:02x} decoded to Inf", b);
            if (b & 0x7F) == 0x7F {
                assert!(v.is_nan(), "byte 0x{:02x} should be NaN", b);
                nan_count += 1;
                continue;
            }
            let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
            let exp = ((b >> 3) & 0x0F) as i32;
            let mant = (b & 0x07) as f32;
            let expected = if exp == 0 {
                if mant == 0.0 {
                    0.0 * sign
                } else {
                    sign * 2.0f32.powi(-6) * (mant / 8.0)
                }
            } else {
                sign * 2.0f32.powi(exp - 7) * (1.0 + mant / 8.0)
            };
            assert_eq!(v.to_bits(), expected.to_bits(), "mismatch on E4M3FN byte 0x{:02x}: got {} vs {}", b, v, expected);
        }
        assert_eq!(nan_count, 2);

        // Monotonicity on positive finite encodings 0x00..=0x7E
        for b in 0u8..0x7E {
            assert!(f32_from_f8_e4m3(b) < f32_from_f8_e4m3(b + 1), "non-monotone at 0x{:02x}", b);
        }

        // TensorBuffer, read_elem, decode_row parity
        let payload: Vec<u8> = (0u8..=255).collect();
        let tb = TensorBuffer::new("F8_E4M3", payload.clone());
        let mut row = vec![0.0f64; 256];
        decode_row("F8_E4M3", &payload, 0, 256, &mut row);
        for i in 0..256 {
            let a = tb.get(i as i64);
            let b = read_elem(&payload, "F8_E4M3", i);
            let c = row[i];
            if a.is_nan() {
                assert!(b.is_nan() && c.is_nan());
            } else {
                assert_eq!(a.to_bits(), b.to_bits());
                assert_eq!(a.to_bits(), c.to_bits());
            }
        }
    }

    #[test]
    fn f8_e5m2_exhaustive_256() {
        assert_eq!(dtype_size("F8_E5M2"), 1);
        assert!(is_fp8_dtype("F8_E5M2"));
        assert!(!is_float_dtype("F8_E5M2"));
        assert!(is_readable_float_dtype("F8_E5M2"));

        // Anchors: ±0, ±1.0 (0x3C/0xBC), max normal ±57344.0 (0x7B/0xFB), ±Inf (0x7C/0xFC), NaN (0x7D..=0x7F, 0xFD..=0xFF)
        assert_eq!(f32_from_f8_e5m2(0x00).to_bits(), 0.0f32.to_bits());
        assert_eq!(f32_from_f8_e5m2(0x80).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f32_from_f8_e5m2(0x3C), 1.0);
        assert_eq!(f32_from_f8_e5m2(0xBC), -1.0);
        assert_eq!(f32_from_f8_e5m2(0x7B), 57344.0);
        assert_eq!(f32_from_f8_e5m2(0xFB), -57344.0);
        assert_eq!(f32_from_f8_e5m2(0x01), 2.0f32.powi(-16));
        assert_eq!(f32_from_f8_e5m2(0x04), 2.0f32.powi(-14));
        assert_eq!(f32_from_f8_e5m2(0x7C), f32::INFINITY);
        assert_eq!(f32_from_f8_e5m2(0xFC), f32::NEG_INFINITY);

        let mut inf_count = 0;
        let mut nan_count = 0;
        for b in 0u8..=255 {
            let v = f32_from_f8_e5m2(b);
            // IEEE E5M2 is exact top-8-bits of IEEE binary16
            let f16_ref = f32_from_f16((b as u16) << 8);
            if f16_ref.is_nan() {
                assert!(v.is_nan(), "E5M2 byte 0x{:02x} expected NaN", b);
                assert_eq!(v.to_bits() >> 31, f16_ref.to_bits() >> 31);
                nan_count += 1;
            } else {
                if v.is_infinite() {
                    inf_count += 1;
                }
                assert_eq!(v.to_bits(), f16_ref.to_bits(), "mismatch on E5M2 byte 0x{:02x}", b);
            }
        }
        assert_eq!(inf_count, 2);
        assert_eq!(nan_count, 6);

        // Positive finite + +Inf monotonicity 0x00..=0x7C
        for b in 0u8..0x7C {
            assert!(f32_from_f8_e5m2(b) < f32_from_f8_e5m2(b + 1), "non-monotone at 0x{:02x}", b);
        }
    }
}
