//! Native PEFT LoRA adapter scanning for `secant audit-trojan` (Rank-k
//! spectral trojan detection of `adapter_model.safetensors`).
//!
//! A PEFT LoRA adapter stores, per adapted module, `lora_A.weight`
//! (`[r, d_in]`) and `lora_B.weight` (`[d_out, r]`); the weight delta it adds
//! to the frozen base is `Delta W = s * B A` with `s = lora_alpha / r`
//! (`lora_alpha / sqrt(r)` with `use_rslora`), taken from `adapter_config.json`
//! with PEFT's `rank_pattern` / `alpha_pattern` overrides.
//!
//! The spectrum of `B A` is computed without forming the `d_out x d_in`
//! product. With thin QR factors `B = Q_B R_B` and `A^T = Q_A R_A`,
//!
//! ```text
//! B A = Q_B (R_B R_A^T) Q_A^T,   M = R_B R_A^T  (r x r)
//! ```
//!
//! so the singular values of `B A` are those of the small matrix `M`
//! (one-sided Jacobi, accurate to full float64 relative precision), and the
//! left singular vectors are `Q_B U_M`. The energy of the adapter is exact
//! (`||B A||_F = ||M||_F`) and the whole spectrum has at most `r` values, so
//! the cumulative top-k energy needs no iteration and no convergence caveat.
//!
//! Scope: LoRA `Linear` adapters (`lora_A.weight` / `lora_B.weight`, with or
//! without an adapter-name component). Embedding adapters
//! (`lora_embedding_*`), DoRA magnitude vectors and non-LoRA PEFT families
//! (LoHa, LoKr, ...) are counted and listed under `other_tensors`, never
//! guessed at. `--reference` has no meaning for an adapter (it already is a
//! delta); the CLI rejects the combination.
//!
//! Sign convention: the largest-magnitude component of each `u_i` is
//! positive (the base-checkpoint reader fixes `v_1` instead; no `v` is formed
//! for an adapter).

use crate::container::*;
use crate::dtype::*;
use crate::error::{Error, Result};
use crate::rankk::{self, RankK, RankKOutlier};
use crate::read::{dot_nf, load_vocab, norm2_nf, validate_strict};
use regex::Regex;
use serde::Serialize;
use std::collections::BTreeMap;
use std::os::unix::fs::MetadataExt;

pub const LORA_SCHEMA: &str = "secant.wsp.audit-lora.v1";

const MAX_TENSOR_BYTES: i64 = 2 << 30;
/// Ranks above this are skipped (Jacobi is O(r^3) per sweep); PEFT adapters are far below.
const MAX_RANK: usize = 512;
const MAX_SWEEPS: usize = 60;
const JACOBI_EPS: f64 = 1e-15;
/// A column of `B` or `A^T` with residual below this fraction of its own norm
/// is linearly dependent on the earlier ones: it adds nothing to `B A`.
const QR_DEPENDENT: f64 = 1e-12;
const RANK_FLOOR: f64 = 1e-12;
const OTHER_EXAMPLES: usize = 32;
/// PEFT's `LoraConfig.lora_alpha` default, used only when the config omits it (and said so).
const DEFAULT_ALPHA: f64 = 8.0;

#[derive(Debug, Clone, Default, Serialize)]
pub struct LoraAuditOptions {
    pub model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub adapter_config: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub tensor: String,
    pub rank_k: usize,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub vocab_tensor: String,
    /// Checkpoint that holds `vocab_tensor` (an adapter has no unembedding).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub vocab_model: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdapterConfigInfo {
    pub path: String,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peft_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_model_name_or_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_alpha: Option<f64>,
    pub use_rslora: bool,
    /// GPT-2 style `Conv1D` weights are stored `[in, out]`, so PEFT adds `(B A)^T`: the
    /// spectrum and energy are unchanged but the output directions are the RIGHT singular
    /// vectors, which this scan does not form (no vocabulary projection is attempted).
    pub fan_in_fan_out: bool,
    pub rank_pattern_entries: usize,
    pub alpha_pattern_entries: usize,
}

/// Parsed `adapter_config.json`: just what fixes the scaling `s`.
struct AdapterConfig {
    info: AdapterConfigInfo,
    r: Option<i64>,
    alpha: Option<f64>,
    rslora: bool,
    rank_pattern: BTreeMap<String, i64>,
    alpha_pattern: BTreeMap<String, f64>,
    /// `.*\.<key>$` for every pattern key, rank keys first (PEFT's chain order).
    matchers: Vec<(Regex, String)>,
}

impl AdapterConfig {
    fn parse(path: &str, bytes: &[u8]) -> Result<AdapterConfig> {
        let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| Error::msg(format!("{}: not valid JSON: {}", path, e)))?;
        let o = v.as_object().ok_or_else(|| Error::msg(format!("{}: adapter config is not a JSON object", path)))?;
        let num = |k: &str| o.get(k).and_then(|x| x.as_f64());
        let r = o.get("r").and_then(|x| x.as_i64());
        let alpha = num("lora_alpha");
        let rslora = o.get("use_rslora").and_then(|x| x.as_bool()).unwrap_or(false);
        let mut rank_pattern = BTreeMap::new();
        if let Some(m) = o.get("rank_pattern").and_then(|x| x.as_object()) {
            for (k, x) in m {
                let n = x.as_i64().ok_or_else(|| Error::msg(format!("{}: rank_pattern[{:?}] is not an integer", path, k)))?;
                rank_pattern.insert(k.clone(), n);
            }
        }
        let mut alpha_pattern = BTreeMap::new();
        if let Some(m) = o.get("alpha_pattern").and_then(|x| x.as_object()) {
            for (k, x) in m {
                let n = x.as_f64().ok_or_else(|| Error::msg(format!("{}: alpha_pattern[{:?}] is not a number", path, k)))?;
                alpha_pattern.insert(k.clone(), n);
            }
        }
        let mut matchers = Vec::new();
        for k in rank_pattern.keys().chain(alpha_pattern.keys()) {
            let re = Regex::new(&format!(r"^.*\.{}$", k))
                .map_err(|e| Error::msg(format!("{}: pattern key {:?} is not a usable regex: {}", path, k, e)))?;
            matchers.push((re, k.clone()));
        }
        Ok(AdapterConfig {
            info: AdapterConfigInfo {
                path: path.into(),
                sha256: sha256_hex(bytes),
                peft_type: o.get("peft_type").and_then(|x| x.as_str()).map(String::from),
                base_model_name_or_path: o.get("base_model_name_or_path").and_then(|x| x.as_str()).map(String::from),
                r,
                lora_alpha: alpha,
                use_rslora: rslora,
                fan_in_fan_out: o.get("fan_in_fan_out").and_then(|x| x.as_bool()).unwrap_or(false),
                rank_pattern_entries: rank_pattern.len(),
                alpha_pattern_entries: alpha_pattern.len(),
            },
            r,
            alpha,
            rslora,
            rank_pattern,
            alpha_pattern,
            matchers,
        })
    }

    /// `s` for the module `key` whose tensors have rank `tensor_rank`.
    /// `Err(reason)` when the config contradicts the tensors (never a number
    /// computed against a mismatch).
    fn scaling(&self, key: &str, tensor_rank: usize) -> std::result::Result<(f64, String), String> {
        let name = self.matchers.iter().find(|(re, _)| re.is_match(key)).map(|(_, k)| k.as_str()).unwrap_or(key);
        let r_cfg = self.rank_pattern.get(name).copied().or(self.r);
        if let Some(rc) = r_cfg {
            if rc != tensor_rank as i64 {
                return Err(format!("tensor rank {} disagrees with the adapter_config rank {}", tensor_rank, rc));
            }
        }
        let (alpha, src) = match self.alpha_pattern.get(name).copied().or(self.alpha) {
            Some(a) => (a, "adapter_config"),
            None => (DEFAULT_ALPHA, "peft-default-lora_alpha=8"),
        };
        let denom = if self.rslora { (tensor_rank as f64).sqrt() } else { tensor_rank as f64 };
        Ok((
            alpha / denom,
            format!(
                "{}: lora_alpha {} / {}",
                src,
                crate::numeric::go_fmt_g(alpha, None),
                if self.rslora { "sqrt(r) (use_rslora)" } else { "r" }
            ),
        ))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct VocabSource {
    pub path: String,
    pub tensor: String,
    pub pre_sha256: String,
    pub post_sha256: String,
    pub input_unchanged: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoraModule {
    /// Module path without the PEFT `base_model.model.` prefix.
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    /// `ok`, `zero_delta`, `nonfinite`, or `skipped: <reason>`.
    pub status: String,
    pub rank: usize,
    pub d_out: usize,
    pub d_in: usize,
    pub dtype_a: String,
    pub dtype_b: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scaling: Option<f64>,
    pub scaling_source: String,
    /// `||s B A||_F`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_frobenius: Option<f64>,
    /// Leading singular values of `s B A` (at most `rank_k`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub singular: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank_k: Option<RankK>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct OtherTensors {
    pub count: usize,
    pub dora_magnitude_vectors: usize,
    pub embedding_adapters: usize,
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoraAuditReport {
    pub schema: &'static str,
    pub model: String,
    pub format: String,
    pub file_bytes: i64,
    pub pre_sha256: String,
    pub post_sha256: String,
    pub input_unchanged: bool,
    pub options: LoraAuditOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_config: Option<AdapterConfigInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_source: Option<VocabSource>,
    pub tensors_in_file: usize,
    pub modules_found: usize,
    pub modules_selected: usize,
    pub modules_scanned: usize,
    pub modules: Vec<LoraModule>,
    pub rank_k_outliers: Vec<RankKOutlier>,
    pub other_tensors: OtherTensors,
    pub method: &'static str,
    pub sign_convention: &'static str,
    pub rule: String,
}

#[derive(Clone, Copy, PartialEq)]
enum Side {
    A,
    B,
}

enum Parsed {
    Linear { module: String, adapter: Option<String>, side: Side },
    Embedding,
    Dora,
    Other,
}

fn parse_name(name: &str) -> Parsed {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, p) in parts.iter().enumerate() {
        let side = match *p {
            "lora_A" => Some(Side::A),
            "lora_B" => Some(Side::B),
            _ => None,
        };
        if let Some(side) = side {
            let rest = &parts[i + 1..];
            let adapter = match rest {
                ["weight"] => None,
                [a, "weight"] => Some(a.to_string()),
                _ => return Parsed::Other,
            };
            if i == 0 {
                return Parsed::Other;
            }
            let module = parts[..i].join(".");
            let module = module.strip_prefix("base_model.model.").map(String::from).unwrap_or(module);
            return Parsed::Linear { module, adapter, side };
        }
        if p.starts_with("lora_embedding_") {
            return Parsed::Embedding;
        }
        if *p == "lora_magnitude_vector" {
            return Parsed::Dora;
        }
    }
    Parsed::Other
}

/// True when `layout` is a SafeTensors file holding at least one LoRA
/// `lora_A` / `lora_B` linear tensor (the shape of `adapter_model.safetensors`).
pub fn is_peft_adapter(layout: &Layout) -> bool {
    layout.format == "safetensors" && layout.tensors.iter().any(|t| matches!(parse_name(&t.name), Parsed::Linear { .. }))
}

// ------------------------------------------------------------------ numerics

fn decode_matrix(dtype: &str, data: &[u8], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0f64; rows * cols];
    let w = dtype_size(dtype) as usize;
    for r in 0..rows {
        decode_row(dtype, data, r * cols * w, cols, &mut out[r * cols..(r + 1) * cols]);
    }
    out
}

/// Thin QR of the matrix whose COLUMNS are `cols` (each of length `n`) by
/// modified Gram-Schmidt with one re-orthogonalisation. Returns `Q` columns
/// (a dependent column becomes the zero vector) and the upper-triangular `R`
/// (`r[i][j]`, `i <= j`) with `cols[j] = sum_i R[i][j] Q[i]`.
fn qr_columns(cols: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let c = cols.len();
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(c);
    let mut r = vec![vec![0f64; c]; c];
    for j in 0..c {
        let orig = norm2_nf(&cols[j]);
        let mut v = cols[j].clone();
        for _pass in 0..2 {
            for i in 0..j {
                if r[i][i] == 0.0 {
                    continue; // dependent column: its Q is zero
                }
                let p = dot_nf(&q[i], &v);
                r[i][j] += p;
                for t in 0..v.len() {
                    let d = p * q[i][t];
                    v[t] -= d;
                }
            }
        }
        let n = norm2_nf(&v);
        if n == 0.0 || n <= QR_DEPENDENT * orig {
            q.push(vec![0f64; v.len()]);
        } else {
            r[j][j] = n;
            for x in v.iter_mut() {
                *x /= n;
            }
            q.push(v);
        }
    }
    (q, r)
}

/// One-sided Jacobi (Hestenes) SVD of the square matrix given by its columns.
/// Returns `(sigma_j, u_j)` sorted by sigma descending, and whether every
/// rotation sweep converged. A zero column gets sigma 0 and a zero `u`.
fn jacobi_svd(mut a: Vec<Vec<f64>>) -> (Vec<(f64, Vec<f64>)>, bool) {
    let n = a.len();
    let mut converged = n <= 1;
    for _sweep in 0..MAX_SWEEPS {
        let mut rotated = false;
        for i in 0..n {
            for j in i + 1..n {
                let alpha = dot_nf(&a[i], &a[i]);
                let beta = dot_nf(&a[j], &a[j]);
                let gamma = dot_nf(&a[i], &a[j]);
                if alpha == 0.0 || beta == 0.0 || gamma.abs() <= JACOBI_EPS * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for k in 0..a[i].len() {
                    let (x, y) = (a[i][k], a[j][k]);
                    a[i][k] = c * x - s * y;
                    a[j][k] = s * x + c * y;
                }
            }
        }
        if !rotated {
            converged = true;
            break;
        }
    }
    let mut out: Vec<(f64, Vec<f64>)> = a
        .into_iter()
        .map(|col| {
            let s = norm2_nf(&col);
            if s == 0.0 {
                (0.0, col)
            } else {
                (s, col.into_iter().map(|x| x / s).collect())
            }
        })
        .collect();
    out.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
    (out, converged)
}

/// Spectrum of `B A` for `b` (`[d_out x r]`, row-major) and `a` (`[r x d_in]`):
/// `(sigmas, u_i (length d_out), ||B A||_F^2, converged)`, at most `k` triplets
/// above the rank floor.
fn lowrank_spectrum(b: &[f64], d_out: usize, a: &[f64], d_in: usize, r: usize, k: usize) -> (Vec<(f64, Vec<f64>)>, f64, bool) {
    // columns of B (length d_out) and of A^T = rows of A (length d_in)
    let b_cols: Vec<Vec<f64>> = (0..r).map(|j| (0..d_out).map(|i| b[i * r + j]).collect()).collect();
    let a_cols: Vec<Vec<f64>> = (0..r).map(|j| a[j * d_in..(j + 1) * d_in].to_vec()).collect();
    let (qb, rb) = qr_columns(&b_cols);
    let (_qa, ra) = qr_columns(&a_cols);
    // M = R_B R_A^T: M[i][j] = sum_t R_B[i][t] R_A[j][t]; Jacobi wants columns of M.
    let mut m_cols: Vec<Vec<f64>> = vec![vec![0f64; r]; r];
    let mut frob_sq = 0f64;
    for j in 0..r {
        for i in 0..r {
            let mut s = 0f64;
            for t in 0..r {
                let p = rb[i][t] * ra[j][t];
                s += p;
            }
            m_cols[j][i] = s;
            let sq = s * s;
            frob_sq += sq;
        }
    }
    let (svd, converged) = jacobi_svd(m_cols);
    let sigma1 = svd.first().map(|x| x.0).unwrap_or(0.0);
    let mut out = Vec::new();
    for (sigma, um) in svd.into_iter().take(k) {
        if sigma == 0.0 || sigma < RANK_FLOOR * sigma1 {
            break;
        }
        // u = Q_B u_M
        let mut u = vec![0f64; d_out];
        for (c, coef) in um.iter().enumerate() {
            if *coef != 0.0 {
                for t in 0..d_out {
                    let p = coef * qb[c][t];
                    u[t] += p;
                }
            }
        }
        let mut big = 0;
        for t in 0..d_out {
            if u[t].abs() > u[big].abs() {
                big = t;
            }
        }
        if u[big] < 0.0 {
            for x in u.iter_mut() {
                *x = -*x;
            }
        }
        out.push((sigma, u));
    }
    (out, frob_sq, converged)
}

// ------------------------------------------------------------------ scan

fn bracket_unchanged(path: &str, pre: &str, md: &std::fs::Metadata) -> Result<(String, bool)> {
    let (post, _) = file_sha256(path)?;
    let same = match std::fs::metadata(path) {
        Ok(c) => c.dev() == md.dev() && c.ino() == md.ino() && pre == post && c.len() == md.len(),
        Err(_) => false,
    };
    Ok((post, same))
}

#[derive(Default)]
struct Pair {
    a: Option<usize>,
    b: Option<usize>,
}

/// Scan one PEFT LoRA adapter. Read-only; the adapter file (and the vocab
/// checkpoint, when given) is hashed before and after like every reader here.
pub fn scan_adapter(o: LoraAuditOptions) -> Result<LoraAuditReport> {
    if o.model.is_empty() {
        return Err(Error::msg("model is required"));
    }
    if o.rank_k == 0 {
        return Err(Error::msg("rank-k is required for an adapter scan (1, 2, 4 or 8)"));
    }
    rankk::validate_rank_k(o.rank_k)?;
    if !o.tensor.is_empty() {
        crate::read::path_match(&o.tensor, "")?;
    }
    if !o.vocab_tensor.is_empty() && o.vocab_model.is_empty() {
        return Err(Error::msg("an adapter holds no unembedding: --vocab-tensor needs --vocab-model <checkpoint holding it>"));
    }
    if o.vocab_tensor.is_empty() && !o.vocab_model.is_empty() {
        return Err(Error::msg("--vocab-model needs --vocab-tensor"));
    }
    let f = open_regular_file(&o.model)?;
    let md = f.metadata()?;
    let (pre, _) = file_sha256(&o.model)?;
    let model = Model::open(&o.model).map_err(|e| Error::msg(format!("{}: {}", o.model, e)))?;
    validate_strict(&model.layout, md.len() as i64).map_err(|e| Error::msg(format!("{}: {}", o.model, e)))?;
    let layout = &model.layout;
    if !is_peft_adapter(layout) {
        return Err(Error::msg(format!("{}: not a PEFT LoRA adapter (no SafeTensors lora_A / lora_B linear tensors)", o.model)));
    }

    // adapter_config.json: explicit path, else the sibling file when present
    let cfg_path = if !o.adapter_config.is_empty() {
        Some(o.adapter_config.clone())
    } else {
        let dir = std::path::Path::new(&o.model).parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let p = dir.join("adapter_config.json");
        if p.is_file() {
            Some(p.to_string_lossy().into_owned())
        } else {
            None
        }
    };
    let cfg = match &cfg_path {
        Some(p) => {
            let _ = open_regular_file(p)?;
            let bytes = std::fs::read(p).map_err(|e| Error::msg(format!("read {}: {}", p, e)))?;
            Some(AdapterConfig::parse(p, &bytes)?)
        }
        None => None,
    };

    // optional vocabulary checkpoint
    let vocab_model = if o.vocab_model.is_empty() {
        None
    } else {
        if o.vocab_model == o.model {
            return Err(Error::msg("--vocab-model is the adapter itself, which holds no unembedding"));
        }
        let vf = open_regular_file(&o.vocab_model)?;
        let vmd = vf.metadata()?;
        let (vpre, _) = file_sha256(&o.vocab_model)?;
        let vm = Model::open(&o.vocab_model).map_err(|e| Error::msg(format!("{}: {}", o.vocab_model, e)))?;
        validate_strict(&vm.layout, vmd.len() as i64).map_err(|e| Error::msg(format!("{}: {}", o.vocab_model, e)))?;
        Some((vm, vmd, vpre))
    };
    let vocab = match &vocab_model {
        Some((vm, _, _)) => Some(load_vocab(vm, &o.vocab_tensor)?),
        None => None,
    };

    // group tensors into modules
    let mut pairs: BTreeMap<(String, Option<String>), Pair> = BTreeMap::new();
    let mut other = OtherTensors::default();
    for (i, t) in layout.tensors.iter().enumerate() {
        match parse_name(&t.name) {
            Parsed::Linear { module, adapter, side } => {
                let p = pairs.entry((module, adapter)).or_default();
                match side {
                    Side::A => p.a = Some(i),
                    Side::B => p.b = Some(i),
                }
            }
            Parsed::Embedding => {
                other.count += 1;
                other.embedding_adapters += 1;
                if other.examples.len() < OTHER_EXAMPLES {
                    other.examples.push(t.name.clone());
                }
            }
            Parsed::Dora => {
                other.count += 1;
                other.dora_magnitude_vectors += 1;
                if other.examples.len() < OTHER_EXAMPLES {
                    other.examples.push(t.name.clone());
                }
            }
            Parsed::Other => {
                other.count += 1;
                if other.examples.len() < OTHER_EXAMPLES {
                    other.examples.push(t.name.clone());
                }
            }
        }
    }

    let mut rep = LoraAuditReport {
        schema: LORA_SCHEMA,
        model: o.model.clone(),
        format: layout.format.clone(),
        file_bytes: md.len() as i64,
        pre_sha256: pre,
        post_sha256: String::new(),
        input_unchanged: false,
        options: o.clone(),
        adapter_config: cfg.as_ref().map(|c| c.info.clone()),
        vocab_source: None,
        tensors_in_file: layout.tensors.len(),
        modules_found: pairs.len(),
        modules_selected: 0,
        modules_scanned: 0,
        modules: vec![],
        rank_k_outliers: vec![],
        other_tensors: other,
        method: "Delta W = s B A; sigma(B A) = sigma(R_B R_A^T) from thin QR of B and A^T and a one-sided Jacobi SVD of the r x r factor; u_i = Q_B u_M",
        sign_convention: "the largest-magnitude component of each u_i is positive",
        rule: rankk::rule_text(),
    };

    for ((module, adapter), p) in pairs {
        if !o.tensor.is_empty() && !crate::read::path_match(&o.tensor, &module)? {
            continue;
        }
        rep.modules_selected += 1;
        let mut m = LoraModule {
            name: module.clone(),
            adapter: adapter.clone(),
            status: "ok".into(),
            rank: 0,
            d_out: 0,
            d_in: 0,
            dtype_a: String::new(),
            dtype_b: String::new(),
            scaling: None,
            scaling_source: String::new(),
            delta_frobenius: None,
            singular: vec![],
            rank_k: None,
            notes: vec![],
        };
        let (ta, tb) = match (p.a, p.b) {
            (Some(a), Some(b)) => (&layout.tensors[a], &layout.tensors[b]),
            (Some(_), None) => {
                m.status = "skipped: lora_A present without lora_B".into();
                rep.modules.push(m);
                continue;
            }
            (None, Some(_)) => {
                m.status = "skipped: lora_B present without lora_A".into();
                rep.modules.push(m);
                continue;
            }
            (None, None) => unreachable!("a module is created by one of its tensors"),
        };
        m.dtype_a = ta.dtype.clone();
        m.dtype_b = tb.dtype.clone();
        let skip = |m: &mut LoraModule, why: String| {
            m.status = format!("skipped: {}", why);
        };
        if !ta.is_2d() || !tb.is_2d() {
            skip(&mut m, format!("lora_A {:?} / lora_B {:?} are not both 2-D", ta.shape, tb.shape));
            rep.modules.push(m);
            continue;
        }
        let (r, d_in, d_out) = (ta.logical_out, ta.logical_in, tb.logical_out);
        m.rank = r;
        m.d_in = d_in;
        m.d_out = d_out;
        if tb.logical_in != r {
            skip(&mut m, format!("lora_B has {} columns but lora_A has {} rows (rank disagrees)", tb.logical_in, r));
            rep.modules.push(m);
            continue;
        }
        if !is_readable_float_dtype(&ta.dtype) || !is_readable_float_dtype(&tb.dtype) {
            skip(&mut m, format!("dtype {} / {} is not a float dtype", ta.dtype, tb.dtype));
            rep.modules.push(m);
            continue;
        }
        if r == 0 || r > MAX_RANK {
            skip(&mut m, format!("rank {} outside 1..={}", r, MAX_RANK));
            rep.modules.push(m);
            continue;
        }
        if ta.size > MAX_TENSOR_BYTES || tb.size > MAX_TENSOR_BYTES {
            skip(&mut m, format!("payload exceeds the {} byte read budget", MAX_TENSOR_BYTES));
            rep.modules.push(m);
            continue;
        }
        let (s, src) = match &cfg {
            Some(c) => match c.scaling(&module, r) {
                Ok(x) => x,
                Err(why) => {
                    skip(&mut m, why);
                    rep.modules.push(m);
                    continue;
                }
            },
            None => (
                1.0,
                "none: adapter_config.json not found; Delta W taken as B A (scale 1). Energy fractions do not depend on the scale".into(),
            ),
        };
        m.scaling = Some(s);
        m.scaling_source = src;
        let a = decode_matrix(&ta.dtype, model.tensor_bytes(ta), r, d_in);
        let b = decode_matrix(&tb.dtype, model.tensor_bytes(tb), d_out, r);
        let nonfinite = a.iter().chain(b.iter()).filter(|x| !x.is_finite()).count();
        if nonfinite > 0 {
            m.status = "nonfinite".into();
            m.notes.push(format!(
                "{} NaN/Inf element(s) in lora_A / lora_B; the spectrum is undefined (use `secant sanitize` to locate them)",
                nonfinite
            ));
            rep.modules.push(m);
            continue;
        }
        let (trips, frob_sq_m, converged) = lowrank_spectrum(&b, d_out, &a, d_in, r, o.rank_k);
        let s_abs = s.abs();
        let frob_sq = s_abs * s_abs * frob_sq_m;
        let basis = "lora_ba";
        if trips.is_empty() || frob_sq.is_nan() || frob_sq <= 0.0 || !frob_sq.is_finite() {
            m.status = "zero_delta".into();
            m.delta_frobenius = if frob_sq.is_finite() { Some(frob_sq.sqrt()) } else { None };
            m.rank_k =
                Some(rankk::undefined(basis, o.rank_k, "Delta W is zero (B or A is zero, or the scale is 0) or its energy is not finite"));
            m.notes.push("a freshly initialised LoRA has B = 0, so its delta is exactly zero".into());
            rep.modules.push(m);
            continue;
        }
        rep.modules_scanned += 1;
        let sigmas: Vec<f64> = trips.iter().map(|(sg, _)| s_abs * sg).collect();
        m.delta_frobenius = Some(frob_sq.sqrt());
        m.singular = sigmas.clone();
        let mut rk = rankk::energy_report(basis, o.rank_k, &sigmas, converged, frob_sq);
        if !converged {
            m.notes.push(format!("Jacobi did not converge in {} sweeps; treat the spectrum as an estimate", MAX_SWEEPS));
        }
        if let Some(v) = &vocab {
            if cfg.as_ref().is_some_and(|c| c.info.fan_in_fan_out) {
                rk.vocab = Some(rankk::RankKVocab {
                    status: "skipped: fan_in_fan_out adapter (Conv1D weight stored transposed): the output directions are the right singular vectors of B A, which this scan does not form".into(),
                    ..Default::default()
                });
            } else {
                let dirs: Vec<(f64, Vec<f64>)> = trips.iter().zip(sigmas.iter()).map(|((_, u), sg)| (*sg, u.clone())).collect();
                rk.vocab = Some(rankk::vocab_coherence(|row, buf| v.row(row, buf), v.rows, v.cols, &dirs)?);
            }
        }
        m.rank_k = Some(rk);
        rep.modules.push(m);
    }
    if rep.modules_selected == 0 {
        return Err(Error::msg(format!("tensor glob {:?} matched no adapter module in {}", o.tensor, o.model)));
    }
    if !rep.modules.iter().any(|m| m.status == "ok" || m.status == "zero_delta") {
        let first = rep.modules.first().map(|m| format!("{}: {}", m.name, m.status)).unwrap_or_default();
        return Err(Error::msg(format!(
            "no adapter module of {} could be scanned ({} selected, first: {}); an unscannable adapter is undeterminable, not clean",
            o.model, rep.modules_selected, first
        )));
    }
    rep.rank_k_outliers = rankk::ranking(rep.modules.iter().filter_map(|m| Some((m.name.as_str(), m.rank_k.as_ref()?))));

    let (post, same) = bracket_unchanged(&o.model, &rep.pre_sha256, &md)?;
    rep.post_sha256 = post;
    rep.input_unchanged = same;
    if !same {
        return Err(Error::msg("input changed while it was being read"));
    }
    if let Some((_, vmd, vpre)) = &vocab_model {
        let (vpost, vsame) = bracket_unchanged(&o.vocab_model, vpre, vmd)?;
        rep.vocab_source = Some(VocabSource {
            path: o.vocab_model.clone(),
            tensor: o.vocab_tensor.clone(),
            pre_sha256: vpre.clone(),
            post_sha256: vpost,
            input_unchanged: vsame,
        });
        if !vsame {
            return Err(Error::msg("vocab checkpoint changed while it was being read"));
        }
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::top_singular_triplets;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn rand(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n).map(|_| lcg(&mut s)).collect()
    }

    fn matmul(b: &[f64], d_out: usize, a: &[f64], d_in: usize, r: usize) -> Vec<f64> {
        let mut w = vec![0f64; d_out * d_in];
        for i in 0..d_out {
            for t in 0..r {
                for j in 0..d_in {
                    w[i * d_in + j] += b[i * r + t] * a[t * d_in + j];
                }
            }
        }
        w
    }

    #[test]
    fn names_parse_to_modules_and_other() {
        let lin = |n: &str| match parse_name(n) {
            Parsed::Linear { module, adapter, side } => Some((module, adapter, side == Side::A)),
            _ => None,
        };
        assert_eq!(
            lin("base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight"),
            Some(("model.layers.3.self_attn.q_proj".into(), None, true))
        );
        assert_eq!(lin("x.lora_B.weight"), Some(("x".into(), None, false)));
        assert_eq!(lin("x.lora_B.default.weight"), Some(("x".into(), Some("default".into()), false)));
        assert!(matches!(parse_name("x.lora_embedding_A"), Parsed::Embedding));
        assert!(matches!(parse_name("x.lora_magnitude_vector"), Parsed::Dora));
        assert!(matches!(parse_name("model.embed_tokens.weight"), Parsed::Other));
        assert!(matches!(parse_name("lora_A.weight"), Parsed::Other));
        assert!(matches!(parse_name("x.lora_A.default.extra.weight"), Parsed::Other));
    }

    #[test]
    fn low_rank_path_matches_the_dense_spectrum() {
        let (d_out, d_in, r) = (40, 56, 6);
        let b = rand(d_out * r, 11);
        let mut a = rand(r * d_in, 12);
        // make the spectrum well separated: scale the rows of A
        for t in 0..r {
            for j in 0..d_in {
                a[t * d_in + j] *= 4.0 / (1.0 + t as f64 * 1.7);
            }
        }
        let (got, frob_sq, conv) = lowrank_spectrum(&b, d_out, &a, d_in, r, 6);
        assert!(conv);
        let w = matmul(&b, d_out, &a, d_in, r);
        let dense = top_singular_triplets(&w, d_out, d_in, 6).unwrap();
        assert_eq!(got.len(), dense.len());
        let fro: f64 = w.iter().map(|x| x * x).sum();
        assert!((frob_sq - fro).abs() <= 1e-10 * fro, "{} vs {}", frob_sq, fro);
        for (g, d) in got.iter().zip(dense.iter()) {
            assert!((g.0 - d.sigma).abs() <= 1e-9 * d.sigma, "sigma {} vs {}", g.0, d.sigma);
            // same direction up to sign
            let cos = dot_nf(&g.1, &d.u).abs();
            assert!((cos - 1.0).abs() < 1e-8, "u disagrees: |cos| = {}", cos);
        }
        // energy of the adapter is fully inside its r singular values
        let tot: f64 = got.iter().map(|x| x.0 * x.0).sum();
        assert!((tot - fro).abs() <= 1e-9 * fro);
    }

    #[test]
    fn rank_deficient_and_zero_factors_are_handled() {
        let (d_out, d_in, r) = (12, 20, 4);
        // B has a duplicated column: rank 3
        let mut b = rand(d_out * r, 21);
        for i in 0..d_out {
            b[i * r + 3] = b[i * r + 1];
        }
        let a = rand(r * d_in, 22);
        let (got, fro_m, _) = lowrank_spectrum(&b, d_out, &a, d_in, r, 8);
        let w = matmul(&b, d_out, &a, d_in, r);
        let fro: f64 = w.iter().map(|x| x * x).sum();
        assert!((fro_m - fro).abs() <= 1e-9 * fro);
        assert_eq!(got.len(), 3, "rank 3: the dependent column adds no singular value");
        // zero B: no spectrum at all
        let (none, f0, _) = lowrank_spectrum(&vec![0.0; d_out * r], d_out, &a, d_in, r, 8);
        assert!(none.is_empty() && f0 == 0.0);
    }

    #[test]
    fn scaling_follows_peft_including_patterns_and_rslora() {
        let cfg = |j: &str| AdapterConfig::parse("c.json", j.as_bytes()).unwrap();
        let c = cfg(r#"{"r": 8, "lora_alpha": 16}"#);
        assert_eq!(c.scaling("model.layers.0.q_proj", 8).unwrap().0, 2.0);
        assert!(c.scaling("m", 4).unwrap_err().contains("disagrees"));
        let c = cfg(r#"{"r": 4, "lora_alpha": 8, "use_rslora": true}"#);
        assert_eq!(c.scaling("m", 4).unwrap().0, 4.0); // 8 / sqrt(4)
        let c = cfg(r#"{"r": 8, "lora_alpha": 16, "rank_pattern": {"q_proj": 16}, "alpha_pattern": {"q_proj": 64}}"#);
        assert_eq!(c.scaling("model.layers.0.self_attn.q_proj", 16).unwrap().0, 4.0);
        assert_eq!(c.scaling("model.layers.0.self_attn.v_proj", 8).unwrap().0, 2.0);
        // no lora_alpha in the file: PEFT's default of 8, and the source says so
        let c = cfg(r#"{"r": 8}"#);
        let (s, src) = c.scaling("m", 8).unwrap();
        assert_eq!(s, 1.0);
        assert!(src.starts_with("peft-default"), "{}", src);
        assert!(AdapterConfig::parse("c.json", br#"{"rank_pattern": {"(": 4}}"#).is_err());
        assert!(AdapterConfig::parse("c.json", b"[1]").is_err());
    }
}
