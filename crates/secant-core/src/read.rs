//! Static weight-space reader (`secant read`): per-tensor Frobenius norm,
//! leading singular triplets by deterministic float64 power iteration with
//! deflation, highest-L2 logical input columns (the sparse column selector
//! `compose` consumes), structural anomalies, and the projection of the
//! leading left singular vector `u_1` through a vocabulary unembedding matrix.
//!
//! Determinism contract (mirrors `internal/wsp/read.go`): single-threaded,
//! fixed summation order, a fixed splitmix64 start vector, and every
//! multiply-add as two separately rounded float64 operations (Rust never
//! contracts them), so the same inputs give bit-identical reports on arm64 and
//! amd64. The `--go-arm64-fma` switch deliberately does NOT affect this
//! module: the oracle blocks fusion here with explicit conversions.
//!
//! Extension over the oracle: `Q8_0` / `Q4_K` 2-D tensors are dequantized row
//! by row (ggml float32 semantics) and analysed like any other matrix.

use crate::container::*;
use crate::dtype::*;
use crate::error::{Error, Result};
use crate::numeric::go_fmt_g;
use crate::quant;
use crate::subspace::{npy_bytes_f64, top_k_abs_indices};
use serde::Serialize;
use std::os::unix::fs::MetadataExt;

pub const READ_SCHEMA: &str = "secant.wsp.read.v1";

const MAX_TENSOR_BYTES: i64 = 2 << 30;
const MAX_ITERS: usize = 1000;
const CONVERGE_TOL: f64 = 1e-13;
const RESIDUAL_TOL: f64 = 1e-10;
const START_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const RANK_FLOOR: f64 = 1e-12;
const COLLAPSE_RATIO: f64 = 1e-3;
const SPIKE_MADS: f64 = 10.0;
const SPIKE_MIN_RATIO: f64 = 2.0;
const MAX_MAGNITUDE: f64 = 1e150;
const MAX_EXAMPLES: usize = 32;
const VEC_PREVIEW: usize = 16;
const VOCAB_TOP_N: usize = 16;
const MAD_SCALE: f64 = 1.4826;
const POLICY_VERSION: &str = "wsp-read-v1";
const SIGN_CONVENTION: &str = "sign convention: the largest-magnitude component of v_1 is positive; u_1 follows so that W v_1 = sigma_1 u_1. Promoted and suppressed token lists swap under the opposite sign.";

#[derive(Debug, Clone, Default, Serialize)]
pub struct ReadOptions {
    pub model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub tensor: String,
    pub top_cols: usize,
    pub top_k_singular: usize,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub vocab_tensor: String,
    pub detect_anomalies: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub vectors_out: String,
    /// Secant extension (Milestone S-2; omitted when false): report the
    /// Marchenko-Pastur bulk edge and `sigma_1 / sigma_MP` per 2-D tensor.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub marchenko_pastur: bool,
    /// Flag threshold on `sigma_1 / sigma_MP`; 0 takes the default. Serialized
    /// only with `marchenko_pastur`.
    #[serde(skip_serializing_if = "is_zero_f")]
    pub mp_threshold: f64,
    /// Secant extension (Horizon 2, omitted when empty): a trusted base
    /// checkpoint. Every spectral statistic of a selected tensor then
    /// describes `Delta W = W - W_ref` (same-named tensor, same logical
    /// geometry), so a rank-1 edit that sits inside a trained matrix's own
    /// spectrum becomes the dominant singular direction. The vocabulary
    /// tensor, when given, is read from `model`, not from the reference.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub reference: String,
}

/// Default `sigma_1 / sigma_MP` flag threshold.
pub const MP_DEFAULT_THRESHOLD: f64 = 1.5;

fn is_zero_f(v: &f64) -> bool {
    *v == 0.0
}

#[derive(Debug, Clone, Serialize)]
pub struct VecEntry {
    pub index: usize,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColNorm {
    pub col: usize,
    pub l2: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SingularInfo {
    pub sigma: f64,
    pub iterations: usize,
    pub converged: bool,
    pub residual: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColStats {
    pub min: f64,
    pub median: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Anomalies {
    pub zero_rows: usize,
    pub zero_row_examples: Vec<usize>,
    pub collapsed_rows: usize,
    pub collapsed_row_examples: Vec<usize>,
    pub median_row_l2: f64,
    pub spike_cols: usize,
    pub spike_col_examples: Vec<ColNorm>,
    pub rule: String,
    pub flagged: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VocabProjection {
    pub status: String,
    #[serde(skip_serializing_if = "is_zero")]
    pub vocab_size: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub promoted: Vec<VecEntry>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub suppressed: Vec<VecEntry>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub note: String,
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}
fn is_zero_i(v: &i64) -> bool {
    *v == 0
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorSpectrum {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    #[serde(skip_serializing_if = "is_zero")]
    pub logical_out: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub logical_in: usize,
    pub elements: i64,
    pub status: String,
    pub nonfinite_elements: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frobenius: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub singular: Vec<SingularInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub u1_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub v1_norm: Option<f64>,
    #[serde(rename = "u1_top_abs", skip_serializing_if = "Vec::is_empty")]
    pub u1_top: Vec<VecEntry>,
    #[serde(rename = "v1_top_abs", skip_serializing_if = "Vec::is_empty")]
    pub v1_top: Vec<VecEntry>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_cols: Vec<ColNorm>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub col_l2: Option<ColStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anomalies: Option<Anomalies>,
    #[serde(rename = "vocab_projection", skip_serializing_if = "Option::is_none")]
    pub vocab: Option<VocabProjection>,
    /// Secant extension (`--marchenko-pastur`): bulk edge and spike ratio.
    #[serde(rename = "marchenko_pastur", skip_serializing_if = "Option::is_none")]
    pub spectral_bulk: Option<crate::mp::SpectralBulk>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    /// Secant extension (`--reference`): present on every selected tensor of
    /// a reference run; says whether the statistics above are of `Delta W`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_diff: Option<ReferenceDiff>,
}

/// Per-tensor outcome of a `--reference` comparison. `status` is `compared`
/// only when the reference holds the same-named tensor with the same logical
/// geometry; every other value is `skipped: <reason>` and the tensor carries
/// no `Delta W` statistics (never a number computed against a mismatch).
#[derive(Debug, Clone, Serialize)]
pub struct ReferenceDiff {
    pub status: String,
    /// Always `delta_w` when `status == "compared"`: the Frobenius norm,
    /// column statistics, anomalies and spectrum of the tensor are `W - W_ref`.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub basis: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub reference_dtype: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_frobenius: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_frobenius: Option<f64>,
    /// `||Delta W||_F / ||W_ref||_F`, absent when the reference norm is 0 or undefined.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relative_delta: Option<f64>,
    /// `Delta W` is exactly zero everywhere (the two payloads decode equal).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identical: Option<bool>,
    /// `sigma_1^2 / ||Delta W||_F^2`: near 1 means the whole difference is one
    /// rank-1 direction (a targeted edit); small means diffuse change (a
    /// fine-tune). Absent when the delta is zero or has no computed spectrum.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sigma1_energy_fraction: Option<f64>,
}

/// Report-level provenance of the reference checkpoint (bracket-hashed like
/// the model).
#[derive(Debug, Clone, Serialize)]
pub struct ReferenceInfo {
    pub path: String,
    pub format: String,
    pub file_bytes: i64,
    pub pre_sha256: String,
    pub post_sha256: String,
    pub input_unchanged: bool,
    pub tensors_in_file: usize,
    pub tensors_compared: usize,
    pub tensors_skipped: usize,
}

/// One row of the report-level spike ranking (`--marchenko-pastur`).
#[derive(Debug, Clone, Serialize)]
pub struct SpectralOutlier {
    pub tensor: String,
    pub sigma_1: f64,
    pub sigma_mp: f64,
    pub sigma1_over_mp: f64,
    pub outliers: usize,
    /// `sigma1_over_mp > threshold`: outside the i.i.d. bulk. Expected for
    /// trained weights; not a verdict on its own.
    pub flagged: bool,
    /// The ratio against the same tensor in the other layers.
    pub sibling: crate::mp::Sibling,
    /// Top promoted vocabulary tokens of `u_1` (with `--vocab-tensor`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub promoted_tokens: Vec<VecEntry>,
}

/// Promoted tokens copied into the spike ranking.
const OUTLIER_TOKENS: usize = 8;

#[derive(Debug, Clone, Serialize)]
pub struct ReadPolicy {
    pub version: &'static str,
    pub max_iterations: usize,
    #[serde(rename = "convergence_tol_relative_sigma")]
    pub convergence_tol: f64,
    #[serde(rename = "convergence_tol_residual")]
    pub residual_tol: f64,
    pub start_seed: u64,
    #[serde(rename = "rank_floor_relative")]
    pub rank_floor: f64,
    #[serde(rename = "collapse_ratio_of_median_row_l2")]
    pub collapse_ratio: f64,
    pub spike_mads: f64,
    #[serde(rename = "spike_min_ratio_of_median")]
    pub spike_min_ratio: f64,
    pub max_tensor_bytes: i64,
    pub sign_convention: &'static str,
    #[serde(rename = "fma_free_arithmetic")]
    pub fma_free: bool,
    pub precision: &'static str,
    #[serde(rename = "threads_and_order")]
    pub threads_and_order: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReadReport {
    pub schema: &'static str,
    pub model: String,
    pub format: String,
    pub file_bytes: i64,
    pub pre_sha256: String,
    pub post_sha256: String,
    pub input_unchanged: bool,
    pub options: ReadOptions,
    pub policy: ReadPolicy,
    pub tensors_in_file: usize,
    pub tensors_selected: usize,
    pub tensors_analyzed: usize,
    pub tensors: Vec<TensorSpectrum>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub vector_files: Vec<String>,
    /// Secant extension (`--marchenko-pastur`): every tensor with a defined
    /// ratio, by `sigma1_over_mp` descending (ties by name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spectral_outliers: Option<Vec<SpectralOutlier>>,
    /// Secant extension (`--reference`): identity and counts of the base checkpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<ReferenceInfo>,
}

// ------------------------------------------------------------------ arithmetic

fn dot_nf(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0f64;
    for i in 0..a.len() {
        let p = a[i] * b[i]; // separately rounded product, never contracted
        s += p;
    }
    s
}

fn axpy_nf(acc: &mut [f64], s: f64, x: &[f64]) {
    for i in 0..x.len() {
        let p = s * x[i];
        acc[i] += p;
    }
}

fn norm2_nf(a: &[f64]) -> f64 {
    dot_nf(a, a).sqrt()
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Logical `[rows = d_out x cols = d_in]` matrix over a raw payload.
struct Dense<'a> {
    dtype: String,
    data: &'a [u8],
    rows: usize,
    cols: usize,
    /// `--reference`: when set, every row is `W - W_ref` (same logical
    /// geometry), formed as one separately rounded float64 subtraction after
    /// both rows are widened, so the delta is never materialised.
    sub: Option<Box<Dense<'a>>>,
    scratch: std::cell::RefCell<Vec<f64>>,
}

impl<'a> Dense<'a> {
    fn plain(dtype: String, data: &'a [u8], rows: usize, cols: usize) -> Self {
        Dense { dtype, data, rows, cols, sub: None, scratch: std::cell::RefCell::new(Vec::new()) }
    }

    fn minus(mut self, reference: Dense<'a>) -> Self {
        self.scratch = std::cell::RefCell::new(vec![0f64; self.cols]);
        self.sub = Some(Box::new(reference));
        self
    }

    fn row(&self, r: usize, dst: &mut [f64]) -> Result<()> {
        self.row_stored(r, dst)?;
        if let Some(rf) = &self.sub {
            let mut s = self.scratch.borrow_mut();
            rf.row_stored(r, &mut s)?;
            for i in 0..dst.len() {
                dst[i] -= s[i];
            }
        }
        Ok(())
    }

    fn row_stored(&self, r: usize, dst: &mut [f64]) -> Result<()> {
        if is_float_dtype(&self.dtype) {
            decode_row(&self.dtype, self.data, r * self.cols * dtype_size(&self.dtype) as usize, self.cols, dst);
            Ok(())
        } else {
            quant::dequant_row_f64(&self.dtype, self.data, r, self.cols, dst)
        }
    }
}

struct Triplet {
    sigma: f64,
    u: Vec<f64>,
    v: Vec<f64>,
    info: SingularInfo,
}

fn top_singular(m: &Dense, k: usize) -> Result<Vec<Triplet>> {
    let mut out: Vec<Triplet> = Vec::new();
    let mut row_buf = vec![0f64; m.cols];
    let mut state = START_SEED;
    let mut sigma1 = 0f64;
    for t in 0..k {
        let mut v = vec![0f64; m.cols];
        for x in v.iter_mut() {
            *x = ((splitmix64(&mut state) >> 11) as i64) as f64 / ((1u64 << 52) as f64) - 1.0;
        }
        let deflate = |x: &mut [f64], out: &Vec<Triplet>| {
            for _pass in 0..2 {
                for prev in out {
                    let p = dot_nf(x, &prev.v);
                    axpy_nf(x, -p, &prev.v);
                }
            }
        };
        deflate(&mut v, &out);
        let n = norm2_nf(&v);
        if n == 0.0 {
            break;
        }
        for x in v.iter_mut() {
            *x /= n;
        }
        let mut w = vec![0f64; m.rows];
        let mut acc = vec![0f64; m.cols];
        let mut info = SingularInfo::default();
        let mut sigma = 0f64;
        let mut prev_sigma = -1.0f64;
        for it in 1..=MAX_ITERS {
            for a in acc.iter_mut() {
                *a = 0.0;
            }
            for r in 0..m.rows {
                m.row(r, &mut row_buf)?;
                let s = dot_nf(&row_buf, &v);
                w[r] = s;
                axpy_nf(&mut acc, s, &row_buf);
            }
            let lambda = dot_nf(&w, &w);
            sigma = lambda.sqrt();
            deflate(&mut acc, &out);
            let acc_norm = norm2_nf(&acc);
            info.iterations = it;
            if lambda > 0.0 {
                let mut res = 0f64;
                for i in 0..acc.len() {
                    let lv = lambda * v[i];
                    let d = acc[i] - lv;
                    let dd = d * d;
                    res += dd;
                }
                info.residual = res.sqrt() / lambda;
            }
            if acc_norm == 0.0 || lambda == 0.0 {
                info.converged = true;
                break;
            }
            if prev_sigma >= 0.0 && (sigma - prev_sigma).abs() <= CONVERGE_TOL * sigma && info.residual <= RESIDUAL_TOL {
                info.converged = true;
                break;
            }
            prev_sigma = sigma;
            for i in 0..v.len() {
                v[i] = acc[i] / acc_norm;
            }
        }
        if t == 0 {
            sigma1 = sigma;
        }
        if sigma == 0.0 || sigma < RANK_FLOOR * sigma1 {
            break;
        }
        info.sigma = sigma;
        let mut tv = v.clone();
        let mut tu: Vec<f64> = w.iter().map(|x| x / sigma).collect();
        let mut big = 0;
        for i in 0..tv.len() {
            if tv[i].abs() > tv[big].abs() {
                big = i;
            }
        }
        if tv[big] < 0.0 {
            for x in tv.iter_mut() {
                *x = -*x;
            }
            for x in tu.iter_mut() {
                *x = -*x;
            }
        }
        out.push(Triplet { sigma, u: tu, v: tv, info });
    }
    Ok(out)
}

fn preview_top(v: &[f64], n: usize) -> Vec<VecEntry> {
    top_k_abs_indices(v, n).into_iter().map(|j| VecEntry { index: j, value: v[j] }).collect()
}

fn median_of(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut c = xs.to_vec();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if c.len() % 2 == 1 {
        c[c.len() / 2]
    } else {
        (c[c.len() / 2 - 1] + c[c.len() / 2]) / 2.0
    }
}

fn analyze_anomalies(row_l2: &[f64], col_l2: &[f64]) -> Anomalies {
    let mut a = Anomalies {
        zero_rows: 0,
        zero_row_examples: vec![],
        collapsed_rows: 0,
        collapsed_row_examples: vec![],
        median_row_l2: median_of(row_l2),
        spike_cols: 0,
        spike_col_examples: vec![],
        rule: format!(
            "zero row: L2 == 0; collapsed row: 0 < L2 < {} * median row L2; spike column: L2 > median + {} * {} * MAD and > {} * median column L2",
            go_g(COLLAPSE_RATIO),
            go_g(SPIKE_MADS),
            go_g(MAD_SCALE),
            go_g(SPIKE_MIN_RATIO)
        ),
        flagged: false,
    };
    for (r, l) in row_l2.iter().enumerate() {
        if *l == 0.0 {
            a.zero_rows += 1;
            if a.zero_row_examples.len() < MAX_EXAMPLES {
                a.zero_row_examples.push(r);
            }
        } else if a.median_row_l2 > 0.0 && *l < COLLAPSE_RATIO * a.median_row_l2 {
            a.collapsed_rows += 1;
            if a.collapsed_row_examples.len() < MAX_EXAMPLES {
                a.collapsed_row_examples.push(r);
            }
        }
    }
    let med = median_of(col_l2);
    let dev: Vec<f64> = col_l2.iter().map(|l| (l - med).abs()).collect();
    let mad = median_of(&dev);
    let thr = med + SPIKE_MADS * MAD_SCALE * mad;
    for (c, l) in col_l2.iter().enumerate() {
        if *l > thr && *l > SPIKE_MIN_RATIO * med {
            a.spike_cols += 1;
            if a.spike_col_examples.len() < MAX_EXAMPLES {
                a.spike_col_examples.push(ColNorm { col: c, l2: *l });
            }
        }
    }
    a.flagged = a.zero_rows + a.collapsed_rows + a.spike_cols > 0;
    a
}

fn go_g(v: f64) -> String {
    go_fmt_g(v, None)
}

struct Analyzed {
    spec: TensorSpectrum,
    u1: Option<Vec<f64>>,
    v1: Option<Vec<f64>>,
}

/// What `--reference` gives one tensor: nothing asked, no usable counterpart
/// (reason), or the counterpart's location and the reference file's bytes.
enum RefArg<'a> {
    Off,
    Missing(String),
    On(&'a TensorLocation, &'a [u8]),
}

/// Why `rt` cannot be subtracted from `t`, or `None` when it can.
fn reference_incompatibility(t: &TensorLocation, rt: &TensorLocation) -> Option<String> {
    if rt.dtype == "I32" || rt.dtype == "I64" {
        return Some(format!("reference dtype {} is an integer type", rt.dtype));
    }
    if rt.is_quantized() && quant::QuantFormat::from_dtype(&rt.dtype).is_none() {
        return Some(format!("reference dtype {} has no dequantizer in this build", rt.dtype));
    }
    if rt.size > MAX_TENSOR_BYTES {
        return Some(format!("reference payload {} bytes exceeds the {} byte read budget", rt.size, MAX_TENSOR_BYTES));
    }
    if t.is_2d() {
        if !(rt.is_2d() && rt.logical_out == t.logical_out && rt.logical_in == t.logical_in) {
            return Some(format!(
                "reference shape {:?} (logical {}x{}) differs from {:?} (logical {}x{})",
                rt.shape, rt.logical_out, rt.logical_in, t.shape, t.logical_out, t.logical_in
            ));
        }
        if let Some(f) = quant::QuantFormat::from_dtype(&rt.dtype) {
            if rt.logical_in % f.block_elems() != 0 {
                return Some(format!("reference {} rows of {} columns are not block-aligned", rt.dtype, rt.logical_in));
            }
        }
    } else if t.shape.len() <= 1 {
        if !(rt.shape.len() <= 1 && rt.elements() == t.elements()) {
            return Some(format!("reference shape {:?} differs from {:?}", rt.shape, t.shape));
        }
        if rt.is_quantized() {
            return Some("reference is a 1-D quantized tensor".into());
        }
    }
    None
}

/// Frobenius norm of the stored matrix, `None` if any element is non-finite.
fn stored_frobenius(m: &Dense) -> Result<Option<f64>> {
    let mut buf = vec![0f64; m.cols];
    let mut total = 0f64;
    for r in 0..m.rows {
        m.row_stored(r, &mut buf)?;
        let mut rs = 0f64;
        for x in &buf {
            if x.is_nan() || x.is_infinite() {
                return Ok(None);
            }
            let sq = x * x;
            rs += sq;
        }
        total += rs;
    }
    Ok(Some(total.sqrt()))
}

fn analyze_tensor(
    data_all: &[u8],
    t: &TensorLocation,
    top_cols: usize,
    top_k: usize,
    detect: bool,
    mp: Option<f64>,
    reference: &RefArg,
) -> Result<Analyzed> {
    let mut spec = TensorSpectrum {
        name: t.name.clone(),
        dtype: t.dtype.clone(),
        shape: t.shape.clone(),
        logical_out: t.logical_out,
        logical_in: t.logical_in,
        elements: t.elements(),
        status: "ok".into(),
        nonfinite_elements: 0,
        frobenius: None,
        singular: vec![],
        u1_norm: None,
        v1_norm: None,
        u1_top: vec![],
        v1_top: vec![],
        top_cols: vec![],
        col_l2: None,
        anomalies: None,
        vocab: None,
        spectral_bulk: None,
        notes: vec![],
        reference_diff: None,
    };
    let mut res = Analyzed { spec: spec.clone(), u1: None, v1: None };
    let quantized = t.is_quantized();
    if t.dtype == "I32" || t.dtype == "I64" {
        res.spec.status = "skipped: integer tensor (no spectrum defined)".into();
        return Ok(res);
    }
    if quantized && quant::QuantFormat::from_dtype(&t.dtype).is_none() {
        res.spec.status = format!("skipped: dtype {} has no dequantizer in this build", t.dtype);
        return Ok(res);
    }
    if t.size > MAX_TENSOR_BYTES {
        res.spec.status = format!("skipped: payload {} bytes exceeds the {} byte read budget", t.size, MAX_TENSOR_BYTES);
        return Ok(res);
    }
    let mut compared: Option<(&TensorLocation, &[u8])> = None;
    match reference {
        RefArg::Off => {}
        RefArg::Missing(why) => return Ok(skip_for_reference(res, why)),
        RefArg::On(rt, rdata) => match reference_incompatibility(t, rt) {
            Some(why) => return Ok(skip_for_reference(res, &why)),
            None => compared = Some((*rt, *rdata)),
        },
    }
    let data = &data_all[t.offset as usize..(t.offset + t.size) as usize];
    let m = if t.is_2d() {
        Dense::plain(t.dtype.clone(), data, t.logical_out, t.logical_in)
    } else if t.shape.len() <= 1 {
        if quantized {
            res.spec.status = "skipped: 1-D quantized tensor".into();
            return Ok(res);
        }
        spec.notes.push("1-D tensor treated as a single row: Frobenius and magnitude statistics only, no spectrum".into());
        Dense::plain(t.dtype.clone(), data, 1, spec.elements as usize)
    } else {
        res.spec.status = format!("skipped: {}-D tensor (spectrum is defined on exactly 2-D linear weights)", t.shape.len());
        return Ok(res);
    };
    if m.rows == 0 || m.cols == 0 {
        res.spec.status = "skipped: empty tensor".into();
        return Ok(res);
    }
    if quantized && m.cols % quant::QuantFormat::from_dtype(&t.dtype).unwrap().block_elems() != 0 {
        res.spec.status = format!("skipped: {} rows of {} columns are not block-aligned", t.dtype, m.cols);
        return Ok(res);
    }
    let m = match compared {
        Some((rt, rdata)) => {
            let rdense = Dense::plain(rt.dtype.clone(), &rdata[rt.offset as usize..(rt.offset + rt.size) as usize], m.rows, m.cols);
            let ref_fro = stored_frobenius(&rdense)?;
            spec.reference_diff = Some(ReferenceDiff {
                status: "compared".into(),
                basis: "delta_w".into(),
                reference_dtype: rt.dtype.clone(),
                reference_frobenius: ref_fro,
                delta_frobenius: None,
                relative_delta: None,
                identical: None,
                sigma1_energy_fraction: None,
            });
            m.minus(rdense)
        }
        None => m,
    };

    // Pass 0: finiteness, magnitudes, row and column sums of squares.
    let mut row_sq = vec![0f64; m.rows];
    let mut col_sq = vec![0f64; m.cols];
    let mut buf = vec![0f64; m.cols];
    let mut max_abs = 0f64;
    let mut nonfinite: i64 = 0;
    for r in 0..m.rows {
        m.row(r, &mut buf)?;
        let mut rs = 0f64;
        for (c, x) in buf.iter().enumerate() {
            if x.is_nan() || x.is_infinite() {
                nonfinite += 1;
                continue;
            }
            if x.abs() > max_abs {
                max_abs = x.abs();
            }
            let sq = x * x;
            rs += sq;
            col_sq[c] += sq;
        }
        row_sq[r] = rs;
    }
    spec.nonfinite_elements = nonfinite;
    if nonfinite > 0 {
        spec.status = "nonfinite".into();
        spec.notes
            .push("Frobenius norm and spectrum are undefined for a tensor holding NaN/Inf; use `secant sanitize` to locate them".into());
        res.spec = spec;
        return Ok(res);
    }
    if max_abs > MAX_MAGNITUDE {
        spec.status = format!("skipped: magnitude {} exceeds {}; sums of squares may overflow float64", max_abs, MAX_MAGNITUDE);
        res.spec = spec;
        return Ok(res);
    }
    let mut total = 0f64;
    let mut row_l2 = vec![0f64; m.rows];
    for (r, s) in row_sq.iter().enumerate() {
        total += s;
        row_l2[r] = s.sqrt();
    }
    spec.frobenius = Some(total.sqrt());
    if let Some(d) = spec.reference_diff.as_mut() {
        let delta = total.sqrt();
        d.delta_frobenius = Some(delta);
        d.relative_delta = d.reference_frobenius.filter(|r| *r > 0.0).map(|r| delta / r);
        d.identical = Some(delta == 0.0);
    }
    let col_l2: Vec<f64> = col_sq.iter().map(|s| s.sqrt()).collect();
    for i in top_k_abs_indices(&col_l2, top_cols) {
        spec.top_cols.push(ColNorm { col: i, l2: col_l2[i] });
    }
    let (mut mn, mut mx) = (col_l2[0], col_l2[0]);
    for l in &col_l2 {
        mn = mn.min(*l);
        mx = mx.max(*l);
    }
    spec.col_l2 = Some(ColStats { min: mn, median: median_of(&col_l2), max: mx });
    if detect {
        spec.anomalies = Some(analyze_anomalies(&row_l2, &col_l2));
    }
    if !t.is_2d() {
        res.spec = spec;
        return Ok(res);
    }
    let k = top_k.min(m.rows).min(m.cols);
    let trips = top_singular(&m, k)?;
    for tr in &trips {
        spec.singular.push(tr.info.clone());
    }
    if let Some(threshold) = mp {
        let sigmas: Vec<f64> = trips.iter().map(|t| t.info.sigma).collect();
        spec.spectral_bulk = Some(crate::mp::marchenko_pastur(total, m.rows, m.cols, &sigmas, threshold));
    }
    if trips.is_empty() {
        spec.notes.push("all singular values are zero within the rank floor".into());
        res.spec = spec;
        return Ok(res);
    }
    if trips.len() < k {
        spec.notes.push(format!(
            "only {} singular value(s) above {} * sigma_1; later ones omitted, not reported as zero",
            trips.len(),
            go_g(RANK_FLOOR)
        ));
    }
    if let Some(d) = spec.reference_diff.as_mut() {
        d.sigma1_energy_fraction = Some(trips[0].sigma * trips[0].sigma / total.max(f64::MIN_POSITIVE));
    }
    let (u1, v1) = (trips[0].u.clone(), trips[0].v.clone());
    spec.u1_norm = Some(norm2_nf(&u1));
    spec.v1_norm = Some(norm2_nf(&v1));
    spec.u1_top = preview_top(&u1, VEC_PREVIEW);
    spec.v1_top = preview_top(&v1, VEC_PREVIEW);
    for tr in &trips {
        if !tr.info.converged {
            spec.notes.push(format!(
                "sigma {} did not converge in {} iterations (residual {}); treat as an estimate",
                go_fmt_g(tr.info.sigma, Some(6)),
                MAX_ITERS,
                go_fmt_g(tr.info.residual, Some(3))
            ));
        }
        let _ = tr.sigma;
    }
    res.spec = spec;
    res.u1 = Some(u1);
    res.v1 = Some(v1);
    Ok(res)
}

/// A tensor `--reference` could not be applied to: no statistics, never a
/// number computed against a mismatched or absent counterpart.
fn skip_for_reference(mut res: Analyzed, why: &str) -> Analyzed {
    res.spec.status = format!("skipped: {}", why);
    res.spec.reference_diff = Some(ReferenceDiff {
        status: res.spec.status.clone(),
        basis: String::new(),
        reference_dtype: String::new(),
        reference_frobenius: None,
        delta_frobenius: None,
        relative_delta: None,
        identical: None,
        sigma1_energy_fraction: None,
    });
    res
}

fn top_k_signed(v: &[f64], n: usize, largest: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| {
        let o = v[*a].partial_cmp(&v[*b]).unwrap_or(std::cmp::Ordering::Equal);
        if largest {
            o.reverse()
        } else {
            o
        }
    });
    idx.truncate(n.min(v.len()));
    idx
}

fn project_vocab_dense(e: &Dense, u: &[f64]) -> Result<VocabProjection> {
    let mut z = vec![0f64; e.rows];
    let mut buf = vec![0f64; e.cols];
    for r in 0..e.rows {
        e.row(r, &mut buf)?;
        z[r] = dot_nf(&buf, u);
    }
    let n = VOCAB_TOP_N.min(e.rows);
    let mut vp = VocabProjection { status: "ok".into(), vocab_size: e.rows, note: SIGN_CONVENTION.into(), ..Default::default() };
    for i in top_k_signed(&z, n, true) {
        vp.promoted.push(VecEntry { index: i, value: z[i] });
    }
    for i in top_k_signed(&z, n, false) {
        vp.suppressed.push(VecEntry { index: i, value: z[i] });
    }
    Ok(vp)
}

fn dense_bytes(m: &[f64]) -> Vec<u8> {
    m.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

/// Project `u` (length `d_model`) through a vocabulary unembedding matrix
/// `e` (row-major `[vocab x d_model]`): `z = E u`, returning the 16 most
/// promoted and most suppressed token indices (ties by ascending index).
pub fn project_vocab(e: &[f64], vocab: usize, d_model: usize, u: &[f64]) -> Result<VocabProjection> {
    if e.len() != vocab * d_model || u.len() != d_model {
        return Err(Error::msg(format!(
            "project_vocab: E has {} values, u has {}; expected {} x {} and {}",
            e.len(),
            u.len(),
            vocab,
            d_model,
            d_model
        )));
    }
    let bytes = dense_bytes(e);
    project_vocab_dense(&Dense::plain("F64".into(), &bytes, vocab, d_model), u)
}

/// One singular triplet of a dense matrix.
#[derive(Debug, Clone)]
pub struct SingularTriplet {
    pub sigma: f64,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub info: SingularInfo,
}

/// Leading `k` singular triplets of a row-major `[rows x cols]` float64
/// matrix by the reader's deterministic power iteration with deflation
/// (fixed start vector, sign convention: largest |v| component positive).
pub fn top_singular_triplets(m: &[f64], rows: usize, cols: usize, k: usize) -> Result<Vec<SingularTriplet>> {
    if m.len() != rows * cols || rows == 0 || cols == 0 {
        return Err(Error::msg(format!("top_singular_triplets: {} values for a {} x {} matrix", m.len(), rows, cols)));
    }
    let bytes = dense_bytes(m);
    let d = Dense::plain("F64".into(), &bytes, rows, cols);
    Ok(top_singular(&d, k.min(rows).min(cols))?
        .into_iter()
        .map(|t| SingularTriplet { sigma: t.sigma, u: t.u, v: t.v, info: t.info })
        .collect())
}

/// Go `path.Match` for tensor-name globs (`*`, `?`, `[...]`, `\` escapes;
/// `*` and `?` never match `/`).
pub fn path_match(pattern: &str, name: &str) -> Result<bool> {
    fn m(p: &[char], s: &[char]) -> std::result::Result<bool, String> {
        let mut p = p;
        let mut s = s;
        loop {
            if p.is_empty() {
                return Ok(s.is_empty());
            }
            match p[0] {
                '*' => {
                    let mut rest = p;
                    while !rest.is_empty() && rest[0] == '*' {
                        rest = &rest[1..];
                    }
                    if rest.is_empty() {
                        return Ok(!s.contains(&'/'));
                    }
                    for i in 0..=s.len() {
                        if m(rest, &s[i..])? {
                            return Ok(true);
                        }
                        if i < s.len() && s[i] == '/' {
                            break;
                        }
                    }
                    return Ok(false);
                }
                '?' => {
                    if s.is_empty() || s[0] == '/' {
                        return Ok(false);
                    }
                    p = &p[1..];
                    s = &s[1..];
                }
                '[' => {
                    if s.is_empty() {
                        // still validate the pattern
                        let mut j = 1;
                        if j < p.len() && p[j] == '^' {
                            j += 1;
                        }
                        let mut closed = false;
                        while j < p.len() {
                            if p[j] == '\\' {
                                j += 1;
                            } else if p[j] == ']' && j > 1 {
                                closed = true;
                                break;
                            }
                            j += 1;
                        }
                        if !closed {
                            return Err("syntax error in pattern".into());
                        }
                        return Ok(false);
                    }
                    let c = s[0];
                    let mut j = 1;
                    let mut negate = false;
                    if j < p.len() && p[j] == '^' {
                        negate = true;
                        j += 1;
                    }
                    let mut matched = false;
                    let mut nrange = 0;
                    loop {
                        if j >= p.len() {
                            return Err("syntax error in pattern".into());
                        }
                        if p[j] == ']' && nrange > 0 {
                            j += 1;
                            break;
                        }
                        let mut lo = p[j];
                        if lo == '\\' {
                            j += 1;
                            if j >= p.len() {
                                return Err("syntax error in pattern".into());
                            }
                            lo = p[j];
                        }
                        j += 1;
                        let mut hi = lo;
                        if j < p.len() && p[j] == '-' {
                            j += 1;
                            if j >= p.len() {
                                return Err("syntax error in pattern".into());
                            }
                            hi = p[j];
                            if hi == '\\' {
                                j += 1;
                                if j >= p.len() {
                                    return Err("syntax error in pattern".into());
                                }
                                hi = p[j];
                            }
                            j += 1;
                        }
                        if lo <= c && c <= hi {
                            matched = true;
                        }
                        nrange += 1;
                    }
                    if matched == negate {
                        return Ok(false);
                    }
                    p = &p[j..];
                    s = &s[1..];
                }
                '\\' => {
                    if p.len() < 2 {
                        return Err("syntax error in pattern".into());
                    }
                    if s.is_empty() || s[0] != p[1] {
                        return Ok(false);
                    }
                    p = &p[2..];
                    s = &s[1..];
                }
                c => {
                    if s.is_empty() || s[0] != c {
                        return Ok(false);
                    }
                    p = &p[1..];
                    s = &s[1..];
                }
            }
        }
    }
    let pc: Vec<char> = pattern.chars().collect();
    let sc: Vec<char> = name.chars().collect();
    m(&pc, &sc).map_err(|e| Error::msg(format!("tensor glob {:?}: {}", pattern, e)))
}

fn safe_vector_name(s: &str) -> String {
    s.chars().map(|r| if r.is_ascii_alphanumeric() || r == '.' || r == '-' || r == '_' { r } else { '_' }).collect()
}

fn normalize(mut o: ReadOptions) -> Result<ReadOptions> {
    if o.model.is_empty() {
        return Err(Error::msg("model is required"));
    }
    if o.top_cols == 0 {
        o.top_cols = 64;
    }
    if o.top_k_singular == 0 {
        o.top_k_singular = 4;
    }
    if o.top_cols > 1 << 20 || o.top_k_singular > 64 {
        return Err(Error::msg(format!("top-cols must be in [1,{}] and top-k-singular in [1,64]", 1 << 20)));
    }
    if !o.tensor.is_empty() {
        path_match(&o.tensor, "")?;
    }
    if o.marchenko_pastur {
        if o.mp_threshold == 0.0 {
            o.mp_threshold = MP_DEFAULT_THRESHOLD;
        }
        if o.mp_threshold.is_nan() || o.mp_threshold <= 0.0 || o.mp_threshold.is_infinite() {
            return Err(Error::msg("mp-threshold must be a finite value > 0"));
        }
    } else if o.mp_threshold != 0.0 {
        return Err(Error::msg("mp-threshold requires marchenko-pastur"));
    }
    Ok(o)
}

/// Sanitizer-grade extent checks (names unique, payloads inside the file and
/// non-overlapping, SafeTensors densely packed), as the oracle applies before
/// trusting a layout with value reads.
pub fn validate_strict(l: &Layout, size: i64) -> Result<()> {
    if l.data_offset < 8 || l.data_offset > size || l.tensors.is_empty() {
        return Err(Error::msg("empty or invalid tensor layout"));
    }
    let mut names = std::collections::HashSet::new();
    for t in &l.tensors {
        if !names.insert(t.name.clone()) {
            return Err(Error::msg(format!("duplicate tensor name {:?}", t.name)));
        }
        if t.size < 0 || t.offset < l.data_offset || t.offset > size || t.size > size - t.offset {
            return Err(Error::msg(format!("tensor {:?} has invalid extent", t.name)));
        }
    }
    let mut ranges: Vec<&TensorLocation> = l.tensors.iter().collect();
    ranges.sort_by(|a, b| a.offset.cmp(&b.offset).then(a.size.cmp(&b.size)));
    let mut end = l.data_offset;
    for t in ranges {
        if t.offset < end || (l.format == "safetensors" && t.offset != end) {
            return Err(Error::msg(format!("tensor {:?} overlaps or leaves a SafeTensors hole", t.name)));
        }
        end = t.offset + t.size;
    }
    if l.format == "safetensors" && end != size {
        return Err(Error::msg("unindexed SafeTensors payload bytes"));
    }
    Ok(())
}

pub fn read_policy() -> ReadPolicy {
    ReadPolicy {
        version: POLICY_VERSION,
        max_iterations: MAX_ITERS,
        convergence_tol: CONVERGE_TOL,
        residual_tol: RESIDUAL_TOL,
        start_seed: START_SEED,
        rank_floor: RANK_FLOOR,
        collapse_ratio: COLLAPSE_RATIO,
        spike_mads: SPIKE_MADS,
        spike_min_ratio: SPIKE_MIN_RATIO,
        max_tensor_bytes: MAX_TENSOR_BYTES,
        sign_convention: SIGN_CONVENTION,
        fma_free: true,
        precision: "float64 accumulation over dtype-widened payload values",
        threads_and_order: "single thread, rows ascending, fixed splitmix64 start vector",
    }
}

/// Analyze the selected tensors of a model file. The input is mapped
/// read-only, hashed before and after (bracket), and never written.
pub fn read_model(o: ReadOptions) -> Result<ReadReport> {
    let o = normalize(o)?;
    let f = open_regular_file(&o.model)?;
    let md = f.metadata()?;
    let (pre, _) = file_sha256(&o.model)?;
    let model = Model::open(&o.model).map_err(|e| Error::msg(format!("{}: {}", o.model, e)))?;
    validate_strict(&model.layout, md.len() as i64).map_err(|e| Error::msg(format!("{}: {}", o.model, e)))?;
    let layout = &model.layout;
    let mut rep = ReadReport {
        schema: READ_SCHEMA,
        model: o.model.clone(),
        format: layout.format.clone(),
        file_bytes: md.len() as i64,
        pre_sha256: pre,
        post_sha256: String::new(),
        input_unchanged: false,
        options: o.clone(),
        policy: read_policy(),
        tensors_in_file: layout.tensors.len(),
        tensors_selected: 0,
        tensors_analyzed: 0,
        tensors: vec![],
        vector_files: vec![],
        spectral_outliers: None,
        reference: None,
    };

    // `--reference`: map the base checkpoint read-only under the same bracket.
    let ref_file;
    let ref_model;
    let mut ref_md = None;
    if !o.reference.is_empty() {
        ref_file = open_regular_file(&o.reference)?;
        let rmd = ref_file.metadata()?;
        if rmd.dev() == md.dev() && rmd.ino() == md.ino() {
            return Err(Error::msg(format!(
                "reference {} is the model file itself; a checkpoint has no delta against itself",
                o.reference
            )));
        }
        let (rpre, _) = file_sha256(&o.reference)?;
        let rm = Model::open(&o.reference).map_err(|e| Error::msg(format!("{}: {}", o.reference, e)))?;
        validate_strict(&rm.layout, rmd.len() as i64).map_err(|e| Error::msg(format!("{}: {}", o.reference, e)))?;
        rep.reference = Some(ReferenceInfo {
            path: o.reference.clone(),
            format: rm.layout.format.clone(),
            file_bytes: rmd.len() as i64,
            pre_sha256: rpre,
            post_sha256: String::new(),
            input_unchanged: false,
            tensors_in_file: rm.layout.tensors.len(),
            tensors_compared: 0,
            tensors_skipped: 0,
        });
        ref_md = Some(rmd);
        ref_model = Some(rm);
    } else {
        ref_model = None;
    }

    let mut selected: Vec<&TensorLocation> = Vec::new();
    for t in &layout.tensors {
        if !o.tensor.is_empty() && !path_match(&o.tensor, &t.name)? {
            continue;
        }
        selected.push(t);
    }
    if selected.is_empty() {
        return Err(Error::msg(format!("tensor glob {:?} matched nothing in {}", o.tensor, o.model)));
    }
    rep.tensors_selected = selected.len();

    let mut vocab: Option<Dense> = None;
    if !o.vocab_tensor.is_empty() {
        let vt = layout.tensor(&o.vocab_tensor)?;
        if !vt.is_2d() || !(is_float_dtype(&vt.dtype) || quant::QuantFormat::from_dtype(&vt.dtype).is_some()) {
            return Err(Error::msg(format!(
                "vocab tensor {} must be a 2-D float tensor (dtype {}, shape {:?})",
                vt.name, vt.dtype, vt.shape
            )));
        }
        if vt.size > MAX_TENSOR_BYTES {
            return Err(Error::msg(format!(
                "vocab tensor {}: {} bytes exceeds the {} byte read budget",
                vt.name, vt.size, MAX_TENSOR_BYTES
            )));
        }
        let d = Dense::plain(vt.dtype.clone(), model.tensor_bytes(vt), vt.logical_out, vt.logical_in);
        let mut chk = vec![0f64; d.cols];
        for r in 0..d.rows {
            d.row(r, &mut chk)?;
            if chk.iter().any(|x| x.is_nan() || x.is_infinite()) {
                return Err(Error::msg(format!("vocab tensor {} holds non-finite values; projection undefined", vt.name)));
            }
        }
        vocab = Some(d);
    }

    let mut vec_dir: Option<String> = None;
    if !o.vectors_out.is_empty() {
        let p = std::path::Path::new(&o.vectors_out);
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::create_dir(p).map_err(|e| Error::msg(format!("vectors-out must be a new directory: {}", e)))?;
        vec_dir = Some(o.vectors_out.clone());
    }
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut first_skip: Option<String> = None;
    for t in selected {
        let mp = if o.marchenko_pastur { Some(o.mp_threshold) } else { None };
        let rarg = match &ref_model {
            None => RefArg::Off,
            Some(rm) => match rm.layout.tensors.iter().find(|x| x.name == t.name) {
                Some(rt) => RefArg::On(rt, rm.bytes()),
                None => RefArg::Missing(format!("tensor {:?} is absent from the reference checkpoint", t.name)),
            },
        };
        let mut a = analyze_tensor(model.bytes(), t, o.top_cols, o.top_k_singular, o.detect_anomalies, mp, &rarg)?;
        if let Some(ri) = rep.reference.as_mut() {
            if a.spec.reference_diff.is_none() {
                a.spec.reference_diff = Some(ReferenceDiff {
                    status: a.spec.status.clone(),
                    basis: String::new(),
                    reference_dtype: String::new(),
                    reference_frobenius: None,
                    delta_frobenius: None,
                    relative_delta: None,
                    identical: None,
                    sigma1_energy_fraction: None,
                });
            }
            if a.spec.reference_diff.as_ref().is_some_and(|d| d.status == "compared") {
                ri.tensors_compared += 1;
            } else {
                ri.tensors_skipped += 1;
                if first_skip.is_none() {
                    first_skip = Some(a.spec.status.clone());
                }
            }
        }
        if a.spec.status == "ok" {
            rep.tensors_analyzed += 1;
        }
        if let (Some(v), Some(u1)) = (&vocab, &a.u1) {
            if u1.len() != v.cols {
                a.spec.vocab = Some(VocabProjection {
                    status: format!(
                        "skipped: u_1 lives in the {}-dim output space but the vocab tensor's model dim is {}",
                        u1.len(),
                        v.cols
                    ),
                    ..Default::default()
                });
            } else {
                a.spec.vocab = Some(project_vocab_dense(v, u1)?);
            }
        }
        if let (Some(dir), Some(u1), Some(v1)) = (&vec_dir, &a.u1, &a.v1) {
            let mut stem = safe_vector_name(&t.name);
            while used.contains(&stem) {
                stem.push('_');
            }
            used.insert(stem.clone());
            for (prefix, vec) in [("u1_", u1), ("v1_", v1)] {
                let p = format!("{}/{}{}.npy", dir, prefix, stem);
                let bytes = npy_bytes_f64(vec, &[vec.len()])?;
                let mut fh = std::fs::OpenOptions::new().write(true).create_new(true).open(&p)?;
                std::io::Write::write_all(&mut fh, &bytes)?;
                rep.vector_files.push(p);
            }
        }
        rep.tensors.push(a.spec);
    }
    if o.marchenko_pastur {
        rep.spectral_outliers = Some(spectral_ranking(&rep.tensors));
    }
    let (post, _) = file_sha256(&o.model)?;
    rep.post_sha256 = post;
    let cur = std::fs::metadata(&o.model);
    rep.input_unchanged = match cur {
        Ok(c) => c.dev() == md.dev() && c.ino() == md.ino() && rep.pre_sha256 == rep.post_sha256 && c.len() == md.len(),
        Err(_) => false,
    };
    if !rep.input_unchanged {
        return Err(Error::msg("input changed while it was being read"));
    }
    if let (Some(ri), Some(rmd)) = (rep.reference.as_mut(), ref_md) {
        let (rpost, _) = file_sha256(&o.reference)?;
        ri.post_sha256 = rpost;
        ri.input_unchanged = match std::fs::metadata(&o.reference) {
            Ok(c) => c.dev() == rmd.dev() && c.ino() == rmd.ino() && ri.pre_sha256 == ri.post_sha256 && c.len() == rmd.len(),
            Err(_) => false,
        };
        if !ri.input_unchanged {
            return Err(Error::msg("reference changed while it was being read"));
        }
        if ri.tensors_compared == 0 {
            return Err(Error::msg(format!(
                "reference {} shares no comparable tensor with {} ({} selected tensor(s) skipped, first: {}); tensors are matched by identical name and logical geometry",
                o.reference,
                o.model,
                ri.tensors_skipped,
                first_skip.unwrap_or_default()
            )));
        }
    }
    Ok(rep)
}

/// The report-level spike ranking: tensors with a defined ratio, highest
/// first, each compared with its layer family (`mp::family_key`).
fn spectral_ranking(tensors: &[TensorSpectrum]) -> Vec<SpectralOutlier> {
    let mut out: Vec<SpectralOutlier> = tensors
        .iter()
        .filter_map(|t| {
            let b = t.spectral_bulk.as_ref()?;
            let (mp, ratio) = (b.sigma_mp?, b.sigma1_over_mp?);
            let promoted = match &t.vocab {
                Some(v) if v.status == "ok" => v.promoted.iter().take(OUTLIER_TOKENS).cloned().collect(),
                _ => vec![],
            };
            Some(SpectralOutlier {
                tensor: t.name.clone(),
                sigma_1: t.singular.first()?.sigma,
                sigma_mp: mp,
                sigma1_over_mp: ratio,
                outliers: b.outliers,
                flagged: b.flagged,
                sibling: crate::mp::Sibling {
                    family: crate::mp::family_key(&t.name),
                    family_size: 0,
                    family_median: None,
                    robust_z: None,
                    sibling_outlier: None,
                },
                promoted_tokens: promoted,
            })
        })
        .collect();
    let mut families: std::collections::BTreeMap<String, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, o) in out.iter().enumerate() {
        families.entry(o.sibling.family.clone()).or_default().push(i);
    }
    for (fam, idx) in families {
        let vals: Vec<f64> = idx.iter().map(|&i| out[i].sigma1_over_mp).collect();
        for (&i, s) in idx.iter().zip(crate::mp::sibling_stats(&fam, &vals)) {
            out[i].sibling = s;
        }
    }
    out.sort_by(|a, b| b.sigma1_over_mp.partial_cmp(&a.sigma1_over_mp).unwrap_or(std::cmp::Ordering::Equal).then(a.tensor.cmp(&b.tensor)));
    out
}

#[allow(dead_code)]
fn _z(_: &i64) -> bool {
    is_zero_i(&0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_semantics() {
        assert!(path_match("blk.*.ffn_down.weight", "blk.12.ffn_down.weight").unwrap());
        assert!(!path_match("blk.?.ffn_down.weight", "blk.12.ffn_down.weight").unwrap());
        assert!(path_match("blk.[0-9]*.attn", "blk.7x.attn").unwrap());
        assert!(!path_match("a*", "a/b").unwrap());
        assert!(path_match("[", "x").is_err());
    }

    #[test]
    fn power_iteration_recovers_known_spectrum_and_sign_convention() {
        // diag(5, 3, 1) rotated by a fixed orthogonal map is awkward; use an exactly rank-2 matrix
        // u1 s1 v1^T + u2 s2 v2^T with orthonormal pairs.
        let (s1, s2, s3) = (4.0f64, 1.5f64, 0.5f64);
        let u1 = [0.5, 0.5, 0.5, 0.5];
        let u2 = [0.5, -0.5, 0.5, -0.5];
        let u3 = [0.5, 0.5, -0.5, -0.5];
        let v1 = [0.6, 0.8, 0.0];
        let v2 = [-0.8, 0.6, 0.0];
        let v3 = [0.0, 0.0, 1.0];
        let mut m = vec![0f64; 12];
        for r in 0..4 {
            for c in 0..3 {
                m[r * 3 + c] = s1 * u1[r] * v1[c] + s2 * u2[r] * v2[c] + s3 * u3[r] * v3[c];
            }
        }
        let t = top_singular_triplets(&m, 4, 3, 3).unwrap();
        assert!((t[2].sigma - s3).abs() < 1e-9);
        assert_eq!(t.len(), 3);
        assert!((t[0].sigma - s1).abs() < 1e-9 && (t[1].sigma - s2).abs() < 1e-9);
        // sign convention: largest |v| component positive
        let big = t[0].v.iter().cloned().fold(0.0f64, |a, x| if x.abs() > a.abs() { x } else { a });
        assert!(big > 0.0);
        // W v = sigma u
        for r in 0..4 {
            let wv: f64 = (0..3).map(|c| m[r * 3 + c] * t[0].v[c]).sum();
            assert!((wv - t[0].sigma * t[0].u[r]).abs() < 1e-9);
        }
        // determinism: bit-identical on a rerun
        let t2 = top_singular_triplets(&m, 4, 3, 3).unwrap();
        assert_eq!(t[0].sigma.to_bits(), t2[0].sigma.to_bits());
    }

    #[test]
    fn rank_deficient_matrix_reports_an_honest_non_converged_repeat() {
        // Deflation by projection cannot remove the ~1e-11 residue of v1 left by a finite
        // convergence tolerance, and a zero singular value cannot damp it: the third
        // iteration re-finds sigma_1. The reader must flag it (converged = false), never
        // present it as a converged value. (Same behaviour as the Go oracle.)
        let (s1, s2) = (4.0f64, 1.5f64);
        let (u1, u2) = ([0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5]);
        let (v1, v2) = ([0.6, 0.8, 0.0], [-0.8, 0.6, 0.0]);
        let mut m = vec![0f64; 12];
        for r in 0..4 {
            for c in 0..3 {
                m[r * 3 + c] = s1 * u1[r] * v1[c] + s2 * u2[r] * v2[c];
            }
        }
        let t = top_singular_triplets(&m, 4, 3, 3).unwrap();
        assert!(t[0].info.converged && t[1].info.converged);
        if t.len() == 3 {
            assert!(!t[2].info.converged && t[2].info.residual > 1e-3, "spurious repeat must be flagged");
        }
    }

    #[test]
    fn vocab_projection_ranks_tokens() {
        // 4 tokens, d_model 2; u = [1, 0] => z = first column
        let e = [0.1, 9.0, -3.0, 0.0, 2.0, 0.0, -0.5, 7.0];
        let vp = project_vocab(&e, 4, 2, &[1.0, 0.0]).unwrap();
        assert_eq!(vp.promoted[0].index, 2);
        assert_eq!(vp.suppressed[0].index, 1);
    }

    #[test]
    fn splitmix_first_values() {
        let mut s = START_SEED;
        let a = splitmix64(&mut s);
        let mut s2 = START_SEED;
        assert_eq!(a, splitmix64(&mut s2));
    }
}
