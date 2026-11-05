//! Exact `Q8_0` (32-element block) and `Q4_K` (256-element superblock)
//! dequantize / quantize kernels (ggml reference semantics), the `Pi_128`
//! commutation witness and block-aligned `Q8_0` in-place surgery.
//!
//! Kernel provenance: `quantize_row_q8_0_ref`, `dequantize_row_q8_0`,
//! `quantize_row_q4_K_ref` (with `make_qkx2_quants`), `dequantize_row_q4_K`
//! and `get_scale_min_k4` from ggml `ggml-quants.c`. All arithmetic is float32
//! exactly as the C reference, written unfused (a C compiler may contract
//! `a*b-c`; this port does not, so Rust results are host-independent).
//!
//! Layout contract: quantization blocks run along the CONTIGUOUS axis of a
//! GGUF tensor, i.e. along the logical input axis `d_in` (ne[0]); a logical
//! row `out` is therefore `d_in / block` consecutive blocks.

use crate::container::*;
use crate::dtype::*;
use crate::error::{Error, Result};
use crate::numeric::fma_add;
use crate::plan::{Edit, Plan};
use memmap2::MmapMut;
use serde::Serialize;
use std::fs::OpenOptions;

pub const QK8_0: usize = 32;
pub const QK_K: usize = 256;
pub const Q8_0_BLOCK_BYTES: usize = 34;
pub const Q4_K_BLOCK_BYTES: usize = 144;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum QuantFormat {
    #[serde(rename = "Q8_0")]
    Q8_0,
    #[serde(rename = "Q4_K")]
    Q4K,
}

impl QuantFormat {
    pub fn parse(s: &str) -> Result<QuantFormat> {
        match s.to_ascii_lowercase().as_str() {
            "q8_0" | "q8-0" => Ok(QuantFormat::Q8_0),
            "q4_k" | "q4-k" | "q4k" => Ok(QuantFormat::Q4K),
            o => Err(Error::msg(format!("unknown quant format {:?} (want q8_0 or q4_k)", o))),
        }
    }
    pub fn block_elems(self) -> usize {
        match self {
            QuantFormat::Q8_0 => QK8_0,
            QuantFormat::Q4K => QK_K,
        }
    }
    pub fn block_bytes(self) -> usize {
        match self {
            QuantFormat::Q8_0 => Q8_0_BLOCK_BYTES,
            QuantFormat::Q4K => Q4_K_BLOCK_BYTES,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            QuantFormat::Q8_0 => "Q8_0",
            QuantFormat::Q4K => "Q4_K",
        }
    }
    pub fn from_dtype(d: &str) -> Option<QuantFormat> {
        match d {
            "Q8_0" => Some(QuantFormat::Q8_0),
            "Q4_K" => Some(QuantFormat::Q4K),
            _ => None,
        }
    }
}

/// float32 -> IEEE-754 binary16, round to nearest even, with proper
/// subnormals (the hardware conversion ggml's `GGML_FP32_TO_FP16` uses).
pub fn f16_from_f32_ieee(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let man = x & 0x007F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x0200 | (man >> 13) as u16 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut half = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half;
    }
    let mut half = sign | ((e as u16) << 10) | (man >> 13) as u16;
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half += 1;
    }
    half
}

#[inline]
fn nearest_int(fval: f32) -> i32 {
    // ggml: add 1.5*2^23 and read the mantissa (round half to even).
    let val = fval + 12582912.0f32;
    ((val.to_bits() as i32) & 0x007f_ffff) - 0x0040_0000
}

fn check_finite(x: &[f32]) -> Result<()> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error::msg("quantize: input holds NaN/Inf; the block scale would be undefined"));
    }
    Ok(())
}

// ------------------------------------------------------------------ Q8_0

/// Quantize `x` (len multiple of 32) into `out` (34 bytes per block).
pub fn quantize_row_q8_0(x: &[f32], out: &mut [u8]) -> Result<()> {
    if x.len() % QK8_0 != 0 {
        return Err(Error::msg(format!("Q8_0 quantize: {} elements is not a multiple of {}", x.len(), QK8_0)));
    }
    let nb = x.len() / QK8_0;
    if out.len() != nb * Q8_0_BLOCK_BYTES {
        return Err(Error::msg(format!("Q8_0 quantize: output is {} bytes, need {}", out.len(), nb * Q8_0_BLOCK_BYTES)));
    }
    check_finite(x)?;
    for i in 0..nb {
        let blk = &x[i * QK8_0..(i + 1) * QK8_0];
        let mut amax = 0f32;
        for v in blk {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0 };
        let o = &mut out[i * Q8_0_BLOCK_BYTES..(i + 1) * Q8_0_BLOCK_BYTES];
        put_u16(&mut o[..2], f16_from_f32_ieee(d));
        for (j, v) in blk.iter().enumerate() {
            o[2 + j] = ((v * id).round() as i8) as u8;
        }
    }
    Ok(())
}

pub fn dequantize_row_q8_0(b: &[u8], out: &mut [f32]) -> Result<()> {
    if b.len() % Q8_0_BLOCK_BYTES != 0 || out.len() != b.len() / Q8_0_BLOCK_BYTES * QK8_0 {
        return Err(Error::msg("Q8_0 dequantize: byte/element count mismatch"));
    }
    for i in 0..b.len() / Q8_0_BLOCK_BYTES {
        let blk = &b[i * Q8_0_BLOCK_BYTES..(i + 1) * Q8_0_BLOCK_BYTES];
        let d = f32_from_f16(le_u16(&blk[..2]));
        for j in 0..QK8_0 {
            out[i * QK8_0 + j] = (blk[2 + j] as i8) as f32 * d;
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ Q4_K

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// `make_qkx2_quants` from ggml-quants.c (n <= 32).
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        for v in l.iter_mut().take(n) {
            *v = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0f32 / iscale;
    let mut best_mad = 0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min));
        l[i] = li.clamp(0, nmax) as u8;
        let mut diff = scale * l[i] as f32 + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_mad += weights[i] * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = li as u8;
            let w = weights[i];
            let lf = li as f32;
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0f32;
            for i in 0..n {
                let mut diff = this_scale * laux[i] as f32 + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                mad += weights[i] * diff;
            }
            if mad < best_mad {
                l[..n].copy_from_slice(&laux[..n]);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// Quantize `x` (len multiple of 256) into `out` (144 bytes per superblock).
pub fn quantize_row_q4_k(x: &[f32], out: &mut [u8]) -> Result<()> {
    if x.len() % QK_K != 0 {
        return Err(Error::msg(format!("Q4_K quantize: {} elements is not a multiple of {}", x.len(), QK_K)));
    }
    let nb = x.len() / QK_K;
    if out.len() != nb * Q4_K_BLOCK_BYTES {
        return Err(Error::msg(format!("Q4_K quantize: output is {} bytes, need {}", out.len(), nb * Q4_K_BLOCK_BYTES)));
    }
    check_finite(x)?;
    for i in 0..nb {
        let xs = &x[i * QK_K..(i + 1) * QK_K];
        let mut l = [0u8; QK_K];
        let mut laux = [0u8; 32];
        let mut weights = [0f32; 32];
        let mut mins = [0f32; QK_K / 32];
        let mut scales = [0f32; QK_K / 32];
        let (mut max_scale, mut max_min) = (0f32, 0f32);
        for j in 0..QK_K / 32 {
            let mut sum_x2 = 0f32;
            for k in 0..32 {
                sum_x2 += xs[32 * j + k] * xs[32 * j + k];
            }
            let av_x = (sum_x2 / 32.0f32).sqrt();
            for k in 0..32 {
                weights[k] = av_x + xs[32 * j + k].abs();
            }
            scales[j] = make_qkx2_quants(
                32,
                15,
                &xs[32 * j..32 * j + 32],
                &weights,
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                &mut laux,
                -1.0,
                0.1,
                20,
                false,
            );
            if scales[j] > max_scale {
                max_scale = scales[j];
            }
            if mins[j] > max_min {
                max_min = mins[j];
            }
        }
        let inv_scale = if max_scale > 0.0 { 63.0f32 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0f32 / max_min } else { 0.0 };
        let o = &mut out[i * Q4_K_BLOCK_BYTES..(i + 1) * Q4_K_BLOCK_BYTES];
        for b in o.iter_mut() {
            *b = 0;
        }
        // layout: d(2) dmin(2) scales(12) qs(128)
        for j in 0..QK_K / 32 {
            let ls = (nearest_int(inv_scale * scales[j]).clamp(0, 255) as u8).min(63);
            let lm = (nearest_int(inv_min * mins[j]).clamp(0, 255) as u8).min(63);
            if j < 4 {
                o[4 + j] = ls;
                o[4 + j + 4] = lm;
            } else {
                o[4 + j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                o[4 + j - 4] |= (ls >> 4) << 6;
                o[4 + j] |= (lm >> 4) << 6;
            }
        }
        put_u16(&mut o[0..2], f16_from_f32_ieee(max_scale / 63.0f32));
        put_u16(&mut o[2..4], f16_from_f32_ieee(max_min / 63.0f32));
        let dh = f32_from_f16(le_u16(&o[0..2]));
        let dmh = f32_from_f16(le_u16(&o[2..4]));
        let sc_bytes: Vec<u8> = o[4..16].to_vec();
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &sc_bytes);
            let d = dh * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = dmh * m as f32;
            for ii in 0..32 {
                let li = nearest_int((xs[32 * j + ii] + dm) / d).clamp(0, 15);
                l[32 * j + ii] = li as u8;
            }
        }
        let mut qo = 16;
        let mut j = 0;
        while j < QK_K {
            for k in 0..32 {
                o[qo + k] = l[j + k] | (l[j + k + 32] << 4);
            }
            qo += 32;
            j += 64;
        }
    }
    Ok(())
}

pub fn dequantize_row_q4_k(b: &[u8], out: &mut [f32]) -> Result<()> {
    if b.len() % Q4_K_BLOCK_BYTES != 0 || out.len() != b.len() / Q4_K_BLOCK_BYTES * QK_K {
        return Err(Error::msg("Q4_K dequantize: byte/element count mismatch"));
    }
    for i in 0..b.len() / Q4_K_BLOCK_BYTES {
        let blk = &b[i * Q4_K_BLOCK_BYTES..(i + 1) * Q4_K_BLOCK_BYTES];
        let d = f32_from_f16(le_u16(&blk[0..2]));
        let min = f32_from_f16(le_u16(&blk[2..4]));
        let scales = &blk[4..16];
        let q = &blk[16..];
        let mut y = i * QK_K;
        let mut is = 0;
        let mut qo = 0;
        let mut j = 0;
        while j < QK_K {
            let (sc, m) = get_scale_min_k4(is, scales);
            let (d1, m1) = (d * sc as f32, min * m as f32);
            let (sc2, m2v) = get_scale_min_k4(is + 1, scales);
            let (d2, m2) = (d * sc2 as f32, min * m2v as f32);
            for l in 0..32 {
                out[y] = d1 * (q[qo + l] & 0xF) as f32 - m1;
                y += 1;
            }
            for l in 0..32 {
                out[y] = d2 * (q[qo + l] >> 4) as f32 - m2;
                y += 1;
            }
            qo += 32;
            is += 2;
            j += 64;
        }
    }
    Ok(())
}

pub fn quantize_row(fmt: QuantFormat, x: &[f32], out: &mut [u8]) -> Result<()> {
    match fmt {
        QuantFormat::Q8_0 => quantize_row_q8_0(x, out),
        QuantFormat::Q4K => quantize_row_q4_k(x, out),
    }
}

pub fn dequantize_row(fmt: QuantFormat, b: &[u8], out: &mut [f32]) -> Result<()> {
    match fmt {
        QuantFormat::Q8_0 => dequantize_row_q8_0(b, out),
        QuantFormat::Q4K => dequantize_row_q4_k(b, out),
    }
}

/// Dequantize logical row `row` (`cols` elements) of a quantized matrix whose
/// rows are contiguous, into float64 (used by the reader).
pub fn dequant_row_f64(dtype: &str, data: &[u8], row: usize, cols: usize, dst: &mut [f64]) -> Result<()> {
    let fmt = QuantFormat::from_dtype(dtype).ok_or_else(|| Error::msg(format!("no dequantizer for dtype {}", dtype)))?;
    let be = fmt.block_elems();
    if cols % be != 0 {
        return Err(Error::msg(format!("{} rows of {} columns are not block-aligned ({} per block)", dtype, cols, be)));
    }
    let row_bytes = cols / be * fmt.block_bytes();
    let mut tmp = vec![0f32; cols];
    dequantize_row(fmt, &data[row * row_bytes..(row + 1) * row_bytes], &mut tmp)?;
    for i in 0..cols {
        dst[i] = tmp[i] as f64;
    }
    Ok(())
}

// ------------------------------------------------------------------ witness

/// `D^T E` statistics: `D` is `k x d_out` (rows are directions), `E` is
/// `d_out x d_in` row-major.
fn dt_e_stats(dirs: &[Vec<f64>], e: &[f64], d_out: usize, d_in: usize) -> (f64, f64) {
    let mut sq = 0f64;
    let mut max_abs = 0f64;
    for d in dirs {
        for c in 0..d_in {
            let mut s = 0f64;
            for o in 0..d_out {
                s = fma_add(s, d[o], e[o * d_in + c]);
            }
            sq = fma_add(sq, s, s);
            if s.abs() > max_abs {
                max_abs = s.abs();
            }
        }
    }
    (sq.sqrt(), max_abs)
}

/// Round trip a row-major `[d_out x d_in]` float32 matrix through `fmt`
/// (blocks along `d_in`).
pub fn roundtrip_matrix(fmt: QuantFormat, w: &[f32], d_out: usize, d_in: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    let be = fmt.block_elems();
    if d_in % be != 0 {
        return Err(Error::msg(format!("{} needs d_in a multiple of {}, got {}", fmt.name(), be, d_in)));
    }
    if w.len() != d_out * d_in {
        return Err(Error::msg(format!("matrix has {} elements, expected {} x {}", w.len(), d_out, d_in)));
    }
    let row_bytes = d_in / be * fmt.block_bytes();
    let mut bytes = vec![0u8; d_out * row_bytes];
    let mut deq = vec![0f32; d_out * d_in];
    for o in 0..d_out {
        quantize_row(fmt, &w[o * d_in..(o + 1) * d_in], &mut bytes[o * row_bytes..(o + 1) * row_bytes])?;
        dequantize_row(fmt, &bytes[o * row_bytes..(o + 1) * row_bytes], &mut deq[o * d_in..(o + 1) * d_in])?;
    }
    Ok((bytes, deq))
}

#[derive(Debug, Clone, Serialize)]
pub struct CommutationWitness {
    pub schema: &'static str,
    pub format: &'static str,
    pub d_out: usize,
    pub d_in: usize,
    pub directions: usize,
    /// `||D^T (Dequant(Quant(W')) - W')||_2` (Frobenius over the k x d_in matrix).
    pub witness_l2: f64,
    pub witness_max_abs: f64,
    /// `||Dequant(Quant(W')) - W'||_F`, for scale.
    pub quant_error_fro: f64,
    pub w_fro: f64,
    /// `||D^T W'||_F` before quantization (0 for a fully projected W').
    pub dt_w_fro_before: f64,
}

pub const WITNESS_SCHEMA: &str = "secant.quant.witness.v1";

/// The `Pi_128` commutation witness: how much of the protected subspace the
/// quantizer re-introduces. `w` is the edited weight `W'` as row-major
/// `[d_out x d_in]` float32, `dirs` the (orthonormal) protected directions.
pub fn commutation_witness(fmt: QuantFormat, dirs: &[Vec<f64>], w: &[f32], d_out: usize, d_in: usize) -> Result<CommutationWitness> {
    for d in dirs {
        if d.len() != d_out {
            return Err(Error::msg(format!("direction has length {}, expected d_out = {}", d.len(), d_out)));
        }
    }
    let (_bytes, deq) = roundtrip_matrix(fmt, w, d_out, d_in)?;
    let e: Vec<f64> = deq.iter().zip(w.iter()).map(|(q, x)| *q as f64 - *x as f64).collect();
    let wf: Vec<f64> = w.iter().map(|x| *x as f64).collect();
    let (l2, max_abs) = dt_e_stats(dirs, &e, d_out, d_in);
    let (dtw, _) = dt_e_stats(dirs, &wf, d_out, d_in);
    let fro = |v: &[f64]| {
        let mut s = 0f64;
        for x in v {
            s = fma_add(s, *x, *x);
        }
        s.sqrt()
    };
    Ok(CommutationWitness {
        schema: WITNESS_SCHEMA,
        format: fmt.name(),
        d_out,
        d_in,
        directions: dirs.len(),
        witness_l2: l2,
        witness_max_abs: max_abs,
        quant_error_fro: fro(&e),
        w_fro: fro(&wf),
        dt_w_fro_before: dtw,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct PanelEquivariance {
    pub format: &'static str,
    /// Every destination block is an exact, aligned source block.
    pub block_aligned: bool,
    /// `Quant(W Pi)` bytes equal the block-permuted `Quant(W)` bytes (only
    /// decidable when `block_aligned`).
    pub bytes_equal: Option<bool>,
    /// `max |Dequant(Quant(W Pi)) - Dequant(Quant(W)) Pi|`.
    pub max_abs_dequant_diff: f64,
    pub equivariant: bool,
}

/// Does quantization commute with an input-axis column permutation `perm`
/// (`perm[dst_col] = src_col`, so `(W Pi)[:, dst] = W[:, src]`)? For `Pi_128`
/// and `Q8_0` (32 divides 128) it must; for `Q4_K` superblocks (256) that
/// straddle 128-column panels it need not.
pub fn panel_equivariance(fmt: QuantFormat, w: &[f32], d_out: usize, d_in: usize, perm: &[usize]) -> Result<PanelEquivariance> {
    if d_out == 0 || d_in == 0 {
        return Err(Error::msg("d_out and d_in must be positive"));
    }
    let expected_len = d_out.checked_mul(d_in).ok_or_else(|| Error::msg("d_out * d_in overflows usize"))?;
    if w.len() != expected_len {
        return Err(Error::msg(format!("weight slice has {} entries, expected d_out * d_in = {}", w.len(), expected_len)));
    }
    if perm.len() != d_in {
        return Err(Error::msg(format!("permutation has {} entries, d_in is {}", perm.len(), d_in)));
    }
    let mut seen = vec![false; d_in];
    for (dst, &src) in perm.iter().enumerate() {
        if src >= d_in {
            return Err(Error::msg(format!("permutation entry perm[{}] = {} is out of bounds for d_in = {}", dst, src, d_in)));
        }
        if seen[src] {
            return Err(Error::msg(format!("permutation is not bijective: duplicate source column {}", src)));
        }
        seen[src] = true;
    }
    let mut wp = vec![0f32; w.len()];
    for o in 0..d_out {
        for (dst, src) in perm.iter().enumerate() {
            wp[o * d_in + dst] = w[o * d_in + src];
        }
    }
    let (qb_p, dq_p) = roundtrip_matrix(fmt, &wp, d_out, d_in)?;
    let (qb, dq) = roundtrip_matrix(fmt, w, d_out, d_in)?;
    let mut max_diff = 0f64;
    for o in 0..d_out {
        for (dst, src) in perm.iter().enumerate() {
            let d = (dq_p[o * d_in + dst] as f64 - dq[o * d_in + src] as f64).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    let be = fmt.block_elems();
    let bb = fmt.block_bytes();
    let mut aligned = true;
    for blk in 0..d_in / be {
        let s0 = perm[blk * be];
        if s0 % be != 0 || (0..be).any(|i| perm[blk * be + i] != s0 + i) {
            aligned = false;
            break;
        }
    }
    let bytes_equal = if aligned {
        let rb = d_in / be * bb;
        let mut eq = true;
        'outer: for o in 0..d_out {
            for blk in 0..d_in / be {
                let src_blk = perm[blk * be] / be;
                let a = &qb_p[o * rb + blk * bb..o * rb + (blk + 1) * bb];
                let b = &qb[o * rb + src_blk * bb..o * rb + (src_blk + 1) * bb];
                if a != b {
                    eq = false;
                    break 'outer;
                }
            }
        }
        Some(eq)
    } else {
        None
    };
    Ok(PanelEquivariance {
        format: fmt.name(),
        block_aligned: aligned,
        bytes_equal,
        max_abs_dequant_diff: max_diff,
        equivariant: max_diff == 0.0,
    })
}

// ------------------------------------------------------------------ Q8_0 surgery

pub const Q8_PATCH_SCHEMA: &str = "secant.quant.patch.v1";

#[derive(Debug, Clone, Serialize, Default)]
pub struct Q8EditReport {
    pub id: String,
    pub tensor: String,
    pub op: String,
    pub blocks_total: usize,
    /// Blocks containing at least one logically edited element.
    pub blocks_touched: usize,
    /// Touched blocks whose 34 stored bytes actually changed.
    pub blocks_rewritten: usize,
    /// max |Dequant(new) - ideal edited value| over touched elements: the
    /// per-block requantization error budget.
    pub max_requant_error: f64,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Q8PatchReport {
    pub schema: String,
    pub base: String,
    pub out: String,
    pub base_sha256: String,
    pub out_sha256: String,
    pub edits: Vec<Q8EditReport>,
    pub verify: Option<VerifyReport>,
}

fn q8_edit(map: &mut MmapMut, layout: &Layout, edit: &Edit) -> Result<Q8EditReport> {
    let name = edit.target_tensor_name(&layout.format).to_string();
    let t = layout.tensor(&name)?.clone();
    if t.dtype != "Q8_0" {
        return Err(Error::msg(format!("tensor {} is {}, block-aligned surgery needs Q8_0", name, t.dtype)));
    }
    t.require_2d(&edit.op)?;
    if t.logical_in % QK8_0 != 0 {
        return Err(Error::msg(format!("tensor {}: d_in {} is not a multiple of {}", name, t.logical_in, QK8_0)));
    }
    let (d_out, d_in) = (t.logical_out, t.logical_in);
    let bpr = d_in / QK8_0;
    let (lo, hi) = (t.offset as usize, (t.offset + t.size) as usize);
    let orig: Vec<u8> = map[lo..hi].to_vec();
    let mut w = vec![0f32; d_out * d_in];
    dequantize_row_q8_0(&orig, &mut w)?;
    let mut ideal: Vec<f64> = w.iter().map(|x| *x as f64).collect();
    let mut touched = vec![false; d_out * bpr];
    let mut rep =
        Q8EditReport { id: edit.id.clone(), tensor: name.clone(), op: edit.op.clone(), blocks_total: d_out * bpr, ..Default::default() };
    if edit.direction.len() != d_out {
        return Err(Error::msg(format!("direction dim {} != tensor logical d_out {}", edit.direction.len(), d_out)));
    }
    match edit.op.as_str() {
        "rank1-columns" => {
            let mut cols: Vec<i64> = edit.cols.clone();
            if edit.layout == "qwen35-ssm-out-128" {
                if !is_qwen35_ssm_out_tensor(&name) || layout.format != "gguf" {
                    return Err(Error::msg(format!("layout qwen35-ssm-out-128 applies to GGUF blk.N.ssm_out.weight, not {}", name)));
                }
                let lay = SsmOutLayout::from_meta(&layout.metadata)?;
                let p = SsmOutPermutation::new(d_in, &lay)?;
                let mut m = Vec::new();
                for c in &cols {
                    if *c < 0 {
                        return Err(Error::msg(format!("negative column {}", c)));
                    }
                    m.push(p.forward_in_col(*c as usize)? as i64);
                }
                cols = m;
                rep.notes.push("cols mapped from HF order through the Qwen3.5 ssm_out interleave".into());
            }
            if cols.len() != edit.coeffs.len() {
                return Err(Error::msg(format!("cols ({}) and coeffs ({}) misaligned", cols.len(), edit.coeffs.len())));
            }
            for (i, c) in cols.iter().enumerate() {
                if *c < 0 || *c as usize >= d_in {
                    return Err(Error::msg(format!("logical input column {} out of [0,{})", c, d_in)));
                }
                let c = *c as usize;
                for o in 0..d_out {
                    ideal[o * d_in + c] = fma_add(ideal[o * d_in + c], edit.coeffs[i], edit.direction[o]);
                    touched[o * bpr + c / QK8_0] = true;
                }
            }
        }
        "projector-subtract" => {
            let mut g = vec![0f64; d_in];
            for (c, gc) in g.iter_mut().enumerate() {
                let mut s = 0f64;
                for o in 0..d_out {
                    s = fma_add(s, edit.direction[o], ideal[o * d_in + c]);
                }
                *gc = s;
            }
            for c in 0..d_in {
                for o in 0..d_out {
                    let coef = -edit.gamma * edit.direction[o];
                    ideal[o * d_in + c] = fma_add(ideal[o * d_in + c], coef, g[c]);
                    touched[o * bpr + c / QK8_0] = true;
                }
            }
        }
        other => return Err(Error::msg(format!("Q8_0 surgery supports rank1-columns and projector-subtract, not {:?}", other))),
    }
    let mut buf = orig.clone();
    for o in 0..d_out {
        for b in 0..bpr {
            if !touched[o * bpr + b] {
                continue;
            }
            rep.blocks_touched += 1;
            let base = o * d_in + b * QK8_0;
            let blk: Vec<f32> = ideal[base..base + QK8_0].iter().map(|x| *x as f32).collect();
            let mut qb = [0u8; Q8_0_BLOCK_BYTES];
            quantize_row_q8_0(&blk, &mut qb)?;
            let mut dq = [0f32; QK8_0];
            dequantize_row_q8_0(&qb, &mut dq)?;
            for j in 0..QK8_0 {
                let err = (dq[j] as f64 - ideal[base + j]).abs();
                if err > rep.max_requant_error {
                    rep.max_requant_error = err;
                }
            }
            let off = (o * bpr + b) * Q8_0_BLOCK_BYTES;
            if buf[off..off + Q8_0_BLOCK_BYTES] != qb {
                rep.blocks_rewritten += 1;
                buf[off..off + Q8_0_BLOCK_BYTES].copy_from_slice(&qb);
            }
        }
    }
    map[lo..hi].copy_from_slice(&buf);
    Ok(rep)
}

/// Apply rank1-columns / projector-subtract edits to Q8_0 tensors at block
/// granularity: only blocks containing an edited element are re-quantized and
/// rewritten; every other block, tensor and the header stay byte-identical
/// (verified before returning).
pub fn q8_0_patch(base_path: &str, out_path: &str, plan: &Plan, mode: PatchMode) -> Result<Q8PatchReport> {
    let layout = load_layout(base_path)?;
    let (base_hash, _) = file_sha256(base_path)?;
    let target = match mode {
        PatchMode::CopyOnWrite => {
            copy_file(base_path, out_path)?;
            out_path
        }
        PatchMode::InPlace => base_path,
    };
    let snapshot = if mode == PatchMode::InPlace { Some(TensorDigests::of(base_path)?) } else { None };
    let file = OpenOptions::new().read(true).write(true).open(target)?;
    // SAFETY: exclusive writer by contract (fresh copy or the named target).
    let mut map = unsafe { MmapMut::map_mut(&file) }.map_err(|e| Error::msg(format!("mmap {}: {}", target, e)))?;
    let mut rep = Q8PatchReport {
        schema: Q8_PATCH_SCHEMA.into(),
        base: base_path.into(),
        out: out_path.into(),
        base_sha256: base_hash,
        ..Default::default()
    };
    for e in &plan.edits {
        let r = q8_edit(&mut map, &layout, e).map_err(|er| Error::msg(format!("edit {}: {}", e.id, er)))?;
        rep.edits.push(r);
    }
    map.flush()?;
    drop(map);
    drop(file);
    rep.out_sha256 = file_sha256(target)?.0;
    let expect: Vec<String> = rep.edits.iter().filter(|e| e.blocks_rewritten > 0).map(|e| e.tensor.clone()).collect();
    let v = match &snapshot {
        None => verify_patch(base_path, out_path, &expect)?,
        Some(s) => s.verify_after(base_path, &expect)?,
    };
    let ok = v.ok;
    let sum = v.failure_summary();
    rep.verify = Some(v);
    if !ok {
        return Err(Error::Gate(format!("post-patch verify failed: {}", sum)));
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((s >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0) as f32
            })
            .collect()
    }

    #[test]
    fn f16_ieee_matches_known() {
        assert_eq!(f16_from_f32_ieee(1.0), 0x3C00);
        assert_eq!(f16_from_f32_ieee(-2.0), 0xC000);
        assert_eq!(f16_from_f32_ieee(65504.0), 0x7BFF);
        assert_eq!(f16_from_f32_ieee(1e-8), 0);
        // smallest subnormal 2^-24
        assert_eq!(f16_from_f32_ieee(5.960_464_5e-8), 1);
        for h in 0u16..=0x7BFF {
            assert_eq!(f16_from_f32_ieee(f32_from_f16(h)), h, "round trip {:04x}", h);
        }
    }

    #[test]
    fn q8_0_roundtrip_error_bounded() {
        let x = lcg(256, 7);
        let mut q = vec![0u8; 8 * Q8_0_BLOCK_BYTES];
        quantize_row_q8_0(&x, &mut q).unwrap();
        let mut y = vec![0f32; 256];
        dequantize_row_q8_0(&q, &mut y).unwrap();
        for b in 0..8 {
            let amax = x[b * 32..(b + 1) * 32].iter().fold(0f32, |a, v| a.max(v.abs()));
            for j in 0..32 {
                assert!((x[b * 32 + j] - y[b * 32 + j]).abs() <= amax * (0.5 / 127.0 + 1.0 / 1024.0) + 1e-7);
            }
        }
    }

    #[test]
    fn q8_0_zero_block_is_zero() {
        let x = vec![0f32; 32];
        let mut q = vec![0u8; 34];
        quantize_row_q8_0(&x, &mut q).unwrap();
        assert!(q.iter().all(|b| *b == 0));
    }

    #[test]
    fn q4_k_roundtrip_error_bounded() {
        let x = lcg(512, 11);
        let mut q = vec![0u8; 2 * Q4_K_BLOCK_BYTES];
        quantize_row_q4_k(&x, &mut q).unwrap();
        let mut y = vec![0f32; 512];
        dequantize_row_q4_k(&q, &mut y).unwrap();
        let rms = (x.iter().zip(&y).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / 512.0).sqrt();
        assert!(rms < 0.08, "rms {}", rms);
    }

    #[test]
    fn rejects_non_finite_and_misaligned() {
        let mut q = vec![0u8; 34];
        let mut x = vec![0f32; 32];
        x[3] = f32::NAN;
        assert!(quantize_row_q8_0(&x, &mut q).is_err());
        assert!(quantize_row_q8_0(&x[..31], &mut q).is_err());
    }

    #[test]
    fn q8_0_commutes_with_128_panels_q4k_need_not() {
        let (d_out, d_in) = (4usize, 512usize);
        let w = lcg(d_out * d_in, 3);
        // swap panels 1 and 2: whole 128-panels move, but across the two 256-superblocks
        let mut perm = vec![0usize; d_in];
        let order = [0usize, 2, 1, 3];
        for p in 0..4 {
            for t in 0..128 {
                perm[p * 128 + t] = order[p] * 128 + t;
            }
        }
        let r8 = panel_equivariance(QuantFormat::Q8_0, &w, d_out, d_in, &perm).unwrap();
        assert!(r8.block_aligned && r8.bytes_equal == Some(true) && r8.equivariant);
        let r4 = panel_equivariance(QuantFormat::Q4K, &w, d_out, d_in, &perm).unwrap();
        assert!(!r4.block_aligned);
        assert!(!r4.equivariant, "Q4_K superblocks straddle the panels, expected a nonzero difference");
    }

    #[test]
    fn panel_equivariance_rejects_malformed_inputs_without_panicking() {
        let (d_out, d_in) = (2usize, 32usize);
        let w = lcg(d_out * d_in, 7);
        let id: Vec<usize> = (0..d_in).collect();
        // Mismatched matrix length
        assert!(panel_equivariance(QuantFormat::Q8_0, &w[..60], d_out, d_in, &id).is_err());
        // Out-of-bounds permutation entry
        let mut oob = id.clone();
        oob[5] = d_in + 10;
        assert!(panel_equivariance(QuantFormat::Q8_0, &w, d_out, d_in, &oob).is_err());
        // Non-bijective permutation (duplicate source column)
        let mut dup = id.clone();
        dup[7] = dup[0];
        assert!(panel_equivariance(QuantFormat::Q8_0, &w, d_out, d_in, &dup).is_err());
        // Zero dimensions
        assert!(panel_equivariance(QuantFormat::Q8_0, &[], 0, d_in, &id).is_err());
    }
}
