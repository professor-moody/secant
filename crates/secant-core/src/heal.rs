//! Project-then-heal (`secant heal`) and its independent verifier
//! (`secant verify-subspace`). Port of `internal/wsp/heal.go`.
//!
//! A LoRA update to a residual-stream writer is `DeltaW = scale * B A` with
//! `B` in `R^{d_out x r}`. Let `D` be an orthonormal basis of the protected
//! output directions. Replacing `B` by `B_locked = (I - D D^T) B` gives
//! `D^T DeltaW = 0` exactly in real arithmetic. This module establishes that in
//! float64, re-measures it on the bytes actually written, and refuses to
//! publish anything whose measured leakage exceeds the stated tolerance.
//!
//! Quantization is not hidden: rounding `W + DeltaW` into BF16/F16/F32
//! payloads adds an error `D^T e` that no projection of the float64 update can
//! remove. The merge gate therefore measures the realized weight difference on
//! disk (the same computation `verify-subspace` performs) and fails closed.
//!
//! Arithmetic policy (mirrors the oracle):
//! * `dotNF` / `axpyNF` / `float64(a*b)` in the oracle block fusion with an
//!   explicit conversion, so they are plain unfused float64 here.
//! * An ordinary `s += a*b` in the oracle (Gram-Schmidt dot, via
//!   `subspace::dot`) goes through `numeric::fma_add`.

use crate::builder::{build_safetensors, BuildTensor};
use crate::container::{
    file_sha256, is_qwen35_ssm_out_tensor, sha256_hex, verify_patch, Layout, Model, SsmOutLayout, SsmOutPermutation, TensorLocation,
};
use crate::dtype::{decode_row, dtype_size, is_float_dtype, TensorBuffer};
use crate::error::{Error, Result};
use crate::plan::GramSchmidtReport;
use crate::subspace::{gram_schmidt, load_numpy_file, load_vectors};
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const HEAL_SCHEMA: &str = "crucible.wsp.heal.v1";
pub const VERIFY_SUBSPACE_SCHEMA: &str = "crucible.wsp.verify-subspace.v1";

/// Bounds `max|D^T B_locked|` in float64 before serialization.
pub const HEAL_F64_TOL: f64 = 1e-12;
/// Bounds the same quantity re-read from the written LoRA file (B is stored as
/// F32, or F64 when the input was F64).
pub const HEAL_SERIALIZED_TOL: f64 = 1e-6;
/// Default realized-weight leakage tolerance.
pub const DEFAULT_SUBSPACE_TOL: f64 = 1e-5;

const MAX_HEADER: i64 = 128 << 20;

// ---------------------------------------------------------------- formatting

/// Go `%.{prec}e`.
pub fn fmt_e(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "+Inf".into() } else { "-Inf".into() };
    }
    let s = format!("{:.*e}", prec, x);
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let e: i32 = exp.parse().unwrap_or(0);
    format!("{}e{}{:02}", mant, if e < 0 { '-' } else { '+' }, e.abs())
}

/// Go `%.{prec}g` (prec >= 1); shared implementation in `numeric`.
pub fn fmt_g(x: f64, prec: usize) -> String {
    crate::numeric::go_fmt_g(x, Some(prec))
}

/// Go `%v` of an `[]int`.
fn fmt_shape(s: &[usize]) -> String {
    let parts: Vec<String> = s.iter().map(|d| d.to_string()).collect();
    format!("[{}]", parts.join(" "))
}

// ---------------------------------------------------------------- arithmetic

/// Oracle `dotNF`: unfused, accumulated in index order over `a`.
fn dot_nf(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0f64;
    for i in 0..a.len() {
        let p: f64 = a[i] * b[i];
        s += p;
    }
    s
}

/// Oracle `axpyNF`: `acc[i] += float64(s * x[i])`, unfused.
fn axpy_nf(acc: &mut [f64], s: f64, x: &[f64]) {
    for i in 0..x.len() {
        let p: f64 = s * x[i];
        acc[i] += p;
    }
}

fn norm2_nf(a: &[f64]) -> f64 {
    dot_nf(a, a).sqrt()
}

/// `max_{k,j} |D_k . x_j|` for the columns of the row-major `[d x r]` matrix.
fn leakage_of_columns(basis: &[Vec<f64>], m: &[f64], d: usize, r: usize) -> f64 {
    let mut worst = 0f64;
    let mut col = vec![0f64; d];
    for j in 0..r {
        for i in 0..d {
            col[i] = m[i * r + j];
        }
        for e in basis {
            let a = dot_nf(e, &col).abs();
            if a > worst {
                worst = a;
            }
        }
    }
    worst
}

/// Replace every column `x` of the row-major `[d x r]` matrix by
/// `(I - D D^T) x`, applied twice so the residual is at rounding level of the
/// remaining vector rather than of the original one.
fn project_columns(basis: &[Vec<f64>], m: &mut [f64], d: usize, r: usize) {
    let mut col = vec![0f64; d];
    for j in 0..r {
        for i in 0..d {
            col[i] = m[i * r + j];
        }
        for _pass in 0..2 {
            for e in basis {
                let p = dot_nf(e, &col);
                axpy_nf(&mut col, -p, e);
            }
        }
        for i in 0..d {
            m[i * r + j] = col[i];
        }
    }
}

// ---------------------------------------------------------------- protected directions

/// The orthonormalized protected output subspace.
#[derive(Debug, Clone, Serialize)]
pub struct ProtectedSet {
    pub source: String,
    pub names: Vec<String>,
    pub file_sha256: BTreeMap<String, String>,
    pub dim: usize,
    pub k: usize,
    pub gram_schmidt: GramSchmidtReport,
    #[serde(skip)]
    pub basis: Arc<Vec<Vec<f64>>>,
}

fn finite_vec(name: &str, v: &[f64]) -> Result<()> {
    for (i, x) in v.iter().enumerate() {
        if !x.is_finite() {
            return Err(Error::msg(format!("protected direction {:?} has a non-finite entry at {}", name, i)));
        }
    }
    Ok(())
}

fn path_ext(p: &str) -> &str {
    let base = p.rsplit('/').next().unwrap_or(p);
    match base.rfind('.') {
        Some(i) => &base[i..],
        None => "",
    }
}

fn path_base(p: &str) -> &str {
    p.trim_end_matches('/').rsplit('/').next().unwrap_or(p)
}

/// Reads a directory of direction files (every `.npy` / `.json` except `h_*`,
/// `hm_*` and `h_m_*` reference activations) or one `.npy` file (1-D = one
/// direction, 2-D `[k, d]` = k rows), and orthonormalizes them with two-pass
/// modified Gram-Schmidt in float64.
pub fn load_protected_directions(spec: &str) -> Result<ProtectedSet> {
    let md = std::fs::metadata(spec).map_err(|e| Error::msg(format!("vectors {}: {}", spec, e)))?;
    let mut named: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut shas: BTreeMap<String, String> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    if md.is_dir() {
        let vs = load_vectors(spec)?;
        for (stem, v) in vs.by_name {
            if stem.starts_with("h_") || stem.starts_with("hm_") || stem.starts_with("h_m_") {
                continue;
            }
            shas.insert(stem.clone(), vs.sha256.get(&stem).cloned().unwrap_or_default());
            named.insert(stem.clone(), v);
            order.push(stem);
        }
        order.sort();
        if order.is_empty() {
            return Err(Error::msg(format!("vectors dir {} holds only h_*/hm_* reference vectors, no protected directions", spec)));
        }
    } else {
        if path_ext(spec).to_lowercase() != ".npy" {
            return Err(Error::msg(format!("vectors {}: want a directory or a .npy file", spec)));
        }
        let arr = load_numpy_file(spec)?;
        let (sum, _) = file_sha256(spec)?;
        let base = path_base(spec);
        let stem = base[..base.len() - path_ext(spec).len()].to_string();
        let vals = arr.to_f64();
        match arr.shape.len() {
            1 => {
                named.insert(stem.clone(), vals);
                shas.insert(stem.clone(), sum);
                order = vec![stem];
            }
            2 => {
                let (k, d) = (arr.shape[0], arr.shape[1]);
                if k == 0 || d == 0 {
                    return Err(Error::msg(format!("vectors {}: empty shape {}", spec, fmt_shape(&arr.shape))));
                }
                for i in 0..k {
                    let name = format!("{}[{}]", stem, i);
                    named.insert(name.clone(), vals[i * d..(i + 1) * d].to_vec());
                    shas.insert(name.clone(), sum.clone());
                    order.push(name);
                }
            }
            _ => {
                return Err(Error::msg(format!(
                    "vectors {}: shape {} is neither a vector nor a [k, d] matrix",
                    spec,
                    fmt_shape(&arr.shape)
                )));
            }
        }
    }
    for name in &order {
        finite_vec(name, &named[name])?;
    }
    let (basis, gs) = gram_schmidt(&named, &order)?;
    let dim = named[&order[0]].len();
    let k = order.len();
    let mut rows = Vec::with_capacity(k);
    for name in &order {
        rows.push(basis[name].clone());
    }
    Ok(ProtectedSet { source: spec.to_string(), names: order, file_sha256: shas, dim, k, gram_schmidt: gs, basis: Arc::new(rows) })
}

/// Pure numeric entry for the C ABI. `directions` is row-major `k x d_out`
/// (not assumed orthonormal: orthonormalized in float64 with the same
/// two-pass modified Gram-Schmidt `heal` uses). `b` is row-major `d_out x r`
/// and is overwritten with `B_locked = P(P B)`, `P = I - D^T D`, computed in
/// float64 and stored back as f32. Returns `max |D^T B_locked|` measured in
/// float64 over the stored f32 values. No tolerance is applied here: the
/// caller owns the gate.
pub fn project_then_heal_f32(directions: &[f32], k: usize, d_out: usize, b: &mut [f32], r: usize) -> Result<f64> {
    if k == 0 {
        return Err(Error::msg("project_then_heal_f32: k = 0 (no protected directions)"));
    }
    if d_out == 0 || r == 0 {
        return Err(Error::msg(format!("project_then_heal_f32: d_out {} and r {} must be positive", d_out, r)));
    }
    let want_dir = k.checked_mul(d_out).ok_or_else(|| Error::msg("project_then_heal_f32: k * d_out overflows"))?;
    let want_b = d_out.checked_mul(r).ok_or_else(|| Error::msg("project_then_heal_f32: d_out * r overflows"))?;
    if directions.len() != want_dir {
        return Err(Error::msg(format!(
            "project_then_heal_f32: directions holds {} values, k {} x d_out {} needs {}",
            directions.len(),
            k,
            d_out,
            want_dir
        )));
    }
    if b.len() != want_b {
        return Err(Error::msg(format!("project_then_heal_f32: b holds {} values, d_out {} x r {} needs {}", b.len(), d_out, r, want_b)));
    }
    let mut named: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::with_capacity(k);
    for i in 0..k {
        let name = format!("d{:08}", i);
        let row: Vec<f64> = directions[i * d_out..(i + 1) * d_out].iter().map(|x| *x as f64).collect();
        finite_vec(&name, &row)?;
        named.insert(name.clone(), row);
        order.push(name);
    }
    if let Some(i) = b.iter().position(|x| !x.is_finite()) {
        return Err(Error::msg(format!("project_then_heal_f32: b holds a non-finite value at element {}", i)));
    }
    let (basis_map, _) = gram_schmidt(&named, &order)?;
    let basis: Vec<Vec<f64>> = order.iter().map(|n| basis_map[n].clone()).collect();
    let mut m: Vec<f64> = b.iter().map(|x| *x as f64).collect();
    project_columns(&basis, &mut m, d_out, r);
    for (dst, src) in b.iter_mut().zip(m.iter()) {
        *dst = *src as f32;
    }
    if let Some(i) = b.iter().position(|x| !x.is_finite()) {
        return Err(Error::msg(format!("project_then_heal_f32: locked b is non-finite at element {}", i)));
    }
    let stored: Vec<f64> = b.iter().map(|x| *x as f64).collect();
    Ok(leakage_of_columns(&basis, &stored, d_out, r))
}

// ---------------------------------------------------------------- strict layout (sanitizeLayout)

/// Reject duplicate JSON object keys and nesting beyond 16, as the oracle's
/// `sanitizeJSONHeader` does (serde_json otherwise keeps the last duplicate).
fn header_json_strict(b: &[u8]) -> Result<()> {
    use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
    struct Seed(usize);
    struct V(usize);
    impl<'de> DeserializeSeed<'de> for Seed {
        type Value = ();
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<(), D::Error> {
            if self.0 > 16 {
                return Err(serde::de::Error::custom("header nesting exceeds limit"));
            }
            d.deserialize_any(V(self.0))
        }
    }
    macro_rules! scalar {
        ($($name:ident: $t:ty),*) => { $(fn $name<E>(self, _v: $t) -> std::result::Result<(), E> { Ok(()) })* };
    }
    impl<'de> Visitor<'de> for V {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("JSON")
        }
        scalar!(visit_bool: bool, visit_i64: i64, visit_u64: u64, visit_f64: f64, visit_str: &str);
        fn visit_unit<E>(self) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_none<E>(self) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> std::result::Result<(), A::Error> {
            while a.next_element_seed(Seed(self.0 + 1))?.is_some() {}
            Ok(())
        }
        fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> std::result::Result<(), A::Error> {
            let mut seen: HashSet<String> = HashSet::new();
            while let Some(k) = a.next_key::<String>()? {
                if !seen.insert(k.clone()) {
                    return Err(serde::de::Error::custom(format!("duplicate or invalid JSON key {:?}", k)));
                }
                a.next_value_seed(Seed(self.0 + 1))?;
            }
            Ok(())
        }
    }
    let mut de = serde_json::Deserializer::from_slice(b);
    Seed(0).deserialize(&mut de).map_err(|e| Error::msg(e.to_string()))?;
    de.end().map_err(|_| Error::msg("trailing header JSON"))
}

/// The oracle's `sanitizeLayout` re-validation, applied on top of the shared
/// parse: GGUF v2/v3 only, header length bounds, no duplicate JSON keys,
/// extents inside the file, no float-unsupported (quantized) dtypes, no
/// overlaps, no SafeTensors holes or unindexed trailing bytes.
fn strict_layout_checks(bytes: &[u8], l: &Layout) -> Result<()> {
    let size = bytes.len() as i64;
    if bytes.len() < 8 {
        return Err(Error::msg("unexpected EOF"));
    }
    if &bytes[..4] == b"GGUF" {
        let v = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if v != 2 && v != 3 {
            return Err(Error::msg("sanitizer supports little-endian GGUF v2/v3 only"));
        }
    } else {
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        if n < 2 || n > (MAX_HEADER - 8) as u64 || size < 8 || n > (size - 8) as u64 {
            return Err(Error::msg("invalid SafeTensors header length"));
        }
        if bytes[8] != b'{' {
            return Err(Error::msg("SafeTensors header must start with an object"));
        }
        header_json_strict(&bytes[8..8 + n as usize])?;
    }
    if l.data_offset < 8 || l.data_offset > size || l.tensors.is_empty() {
        return Err(Error::msg("empty or invalid tensor layout"));
    }
    let mut names: HashSet<&str> = HashSet::new();
    for t in &l.tensors {
        if !names.insert(t.name.as_str()) {
            return Err(Error::msg(format!("duplicate tensor name {:?}", t.name)));
        }
        let mut n: i64 = 1;
        for d in &t.shape {
            let d = *d as i64;
            if d < 0 || (d != 0 && n > i64::MAX / d) {
                return Err(Error::msg(format!("tensor {:?} shape overflow", t.name)));
            }
            n *= d;
        }
        let w = dtype_size(&t.dtype);
        if w == 0
            || n > i64::MAX / w
            || n * w != t.size
            || t.size < 0
            || t.offset < l.data_offset
            || t.offset > size
            || t.size > size - t.offset
        {
            return Err(Error::msg(format!("tensor {:?} has invalid extent or unsupported dtype", t.name)));
        }
    }
    let mut ranges: Vec<&TensorLocation> = l.tensors.iter().collect();
    ranges.sort_by(|a, b| (a.offset, a.size).cmp(&(b.offset, b.size)));
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

/// A model opened the way the oracle's heal/verify paths open it: parsed,
/// then re-validated with the sanitizer's stricter checks.
struct OpenModel {
    model: Model,
    sha256: String,
    id: Option<FileId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileId(u64, u64);

fn file_id(path: &str) -> Option<FileId> {
    let md = std::fs::metadata(path).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some(FileId(md.dev(), md.ino()))
    }
    #[cfg(not(unix))]
    {
        let m = md.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos() as u64;
        Some(FileId(md.len(), m))
    }
}

fn open_strict(path: &str) -> Result<OpenModel> {
    let id = file_id(path);
    let model = Model::open(path)?;
    let sha256 = sha256_hex(model.bytes());
    strict_layout_checks(model.bytes(), &model.layout).map_err(|e| Error::msg(format!("{}: {}", path, e)))?;
    Ok(OpenModel { model, sha256, id })
}

fn matrix_of(bytes: &[u8], t: &TensorLocation) -> Result<Vec<f64>> {
    let w = dtype_size(&t.dtype);
    let n = (t.size / w) as usize;
    let mut out = vec![0f64; n];
    decode_row(&t.dtype, &bytes[t.offset as usize..(t.offset + t.size) as usize], 0, n, &mut out);
    for (i, x) in out.iter().enumerate() {
        if !x.is_finite() {
            return Err(Error::msg(format!("tensor {} holds a non-finite value at element {}", t.name, i)));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- LoRA parsing

#[derive(Debug, Clone)]
struct LoraPair {
    stem: String,
    a_name: String,
    b_name: String,
    a: TensorLocation,
    b: TensorLocation,
}

/// `\.lora_([AB])(\.|$)`: leftmost match, returns (match start, role, role end).
fn lora_role(name: &str) -> Option<(usize, u8, usize)> {
    let b = name.as_bytes();
    let pat = b".lora_";
    let mut i = 0;
    while i + 7 <= b.len() {
        if &b[i..i + 6] == pat && (b[i + 6] == b'A' || b[i + 6] == b'B') && (i + 7 == b.len() || b[i + 7] == b'.') {
            return Some((i, b[i + 6], i + 7));
        }
        i += 1;
    }
    None
}

/// Pairs lora_A / lora_B tensors and refuses anything else: a DoRA magnitude
/// vector or a saved embedding would change the model in ways the subspace
/// lock does not cover, so it is an error, never a pass-through.
fn parse_lora(l: &Layout) -> Result<Vec<LoraPair>> {
    #[derive(Default)]
    struct Half {
        stem: String,
        a: Option<TensorLocation>,
        b: Option<TensorLocation>,
    }
    let mut pairs: BTreeMap<String, Half> = BTreeMap::new();
    let mut stray: Vec<&str> = Vec::new();
    for t in &l.tensors {
        let (m0, role, m3) = match lora_role(&t.name) {
            Some(m) => m,
            None => {
                stray.push(&t.name);
                continue;
            }
        };
        let stem = t.name[..m0].to_string();
        let key = format!("{}.lora_#{}", &t.name[..m0], &t.name[m3..]);
        let h = pairs.entry(key).or_insert_with(|| Half { stem, ..Default::default() });
        if role == b'A' {
            h.a = Some(t.clone());
        } else {
            h.b = Some(t.clone());
        }
    }
    if !stray.is_empty() {
        return Err(Error::msg(format!(
            "LoRA file holds tensors that are not lora_A/lora_B ({}); refusing to claim a subspace lock over a file with unmodeled updates",
            stray.join(", ")
        )));
    }
    if pairs.is_empty() {
        return Err(Error::msg("no lora_A/lora_B tensor pairs found"));
    }
    let mut out: Vec<LoraPair> = Vec::new();
    for (_, h) in pairs {
        let (a, b) = match (h.a, h.b) {
            (Some(a), Some(b)) => (a, b),
            (a, b) => {
                return Err(Error::msg(format!(
                    "unpaired LoRA tensor for module {} (A={:?} B={:?})",
                    h.stem,
                    a.map(|t| t.name).unwrap_or_default(),
                    b.map(|t| t.name).unwrap_or_default()
                )))
            }
        };
        for t in [&a, &b] {
            if !t.is_2d() || !is_float_dtype(&t.dtype) {
                return Err(Error::msg(format!(
                    "LoRA tensor {} must be a 2-D float matrix (dtype {}, shape {})",
                    t.name,
                    t.dtype,
                    fmt_shape(&t.shape)
                )));
            }
        }
        if b.logical_in != a.logical_out {
            return Err(Error::msg(format!(
                "module {}: B is [{} x {}] but A is [{} x {}]; the rank axes disagree",
                h.stem, b.logical_out, b.logical_in, a.logical_out, a.logical_in
            )));
        }
        out.push(LoraPair { stem: h.stem, a_name: a.name.clone(), b_name: b.name.clone(), a, b });
    }
    out.sort_by(|x, y| x.stem.cmp(&y.stem));
    Ok(out)
}

/// `^(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.(\d+)\.(self_attn|mlp|linear_attn)\.([a-z_]+)$`
/// mapped through the HF-module -> GGUF `blk.N.*` table.
fn hf_to_gguf_name(p: &str) -> Option<String> {
    let mut s = p;
    for pre in ["model.", "language_model.", "model."] {
        if let Some(r) = s.strip_prefix(pre) {
            s = r;
        }
    }
    let s = s.strip_prefix("layers.")?;
    let nd = s.bytes().take_while(|c| c.is_ascii_digit()).count();
    if nd == 0 {
        return None;
    }
    let (layer, rest) = s.split_at(nd);
    let rest = rest.strip_prefix('.')?;
    for kind in ["self_attn", "mlp", "linear_attn"] {
        if let Some(r) = rest.strip_prefix(kind) {
            let proj = r.strip_prefix('.')?;
            if proj.is_empty() || !proj.bytes().all(|c| c.is_ascii_lowercase() || c == b'_') {
                return None;
            }
            let g = match format!("{}.{}", kind, proj).as_str() {
                "self_attn.q_proj" => "attn_q",
                "self_attn.k_proj" => "attn_k",
                "self_attn.v_proj" => "attn_v",
                "self_attn.o_proj" => "attn_output",
                "mlp.gate_proj" => "ffn_gate",
                "mlp.up_proj" => "ffn_up",
                "mlp.down_proj" => "ffn_down",
                "linear_attn.out_proj" => "ssm_out",
                _ => return None,
            };
            return Some(format!("blk.{}.{}.weight", layer, g));
        }
    }
    None
}

struct TargetCand {
    name: String,
    hf_mapped: bool,
}

/// The base tensor names a LoRA module may address.
fn lora_target_candidates(stem: &str, format: &str) -> Vec<TargetCand> {
    let p = stem.strip_prefix("base_model.model.").unwrap_or(stem);
    let mut cands = vec![TargetCand { name: format!("{}.weight", p), hf_mapped: false }];
    if format == "gguf" {
        if let Some(g) = hf_to_gguf_name(p) {
            cands.insert(0, TargetCand { name: g, hf_mapped: true });
        }
    }
    cands
}

// ---------------------------------------------------------------- reports

fn is_false(b: &bool) -> bool {
    !*b
}
fn s_empty(s: &String) -> bool {
    s.is_empty()
}

/// Options for `heal`. `scale == 0` means 1, `tol == 0` means
/// [`DEFAULT_SUBSPACE_TOL`] (as the oracle).
#[derive(Debug, Clone, Default, Serialize)]
pub struct HealOptions {
    pub lora: String,
    pub vectors: String,
    pub out: String,
    #[serde(skip_serializing_if = "s_empty")]
    pub base: String,
    #[serde(skip_serializing_if = "s_empty")]
    pub merge_out: String,
    pub scale: f64,
    pub tol: f64,
}

/// Per-module evidence.
#[derive(Debug, Clone, Default, Serialize)]
pub struct HealPair {
    pub module: String,
    pub a_tensor: String,
    pub b_tensor: String,
    pub d_out: usize,
    pub rank: usize,
    pub d_in: usize,
    pub b_fro_before: f64,
    pub b_fro_after: f64,
    #[serde(rename = "leak_before_f64")]
    pub leak_before: f64,
    #[serde(rename = "leak_after_f64")]
    pub leak_after_f64: f64,
    #[serde(rename = "leak_after_serialized")]
    pub leak_after_serialize: f64,
    #[serde(rename = "serialized_b_dtype")]
    pub serialized_dtype: String,
    #[serde(rename = "a_bytes_unchanged")]
    pub a_unchanged: bool,
    #[serde(rename = "merge_target", skip_serializing_if = "s_empty")]
    pub target: String,
    #[serde(rename = "merge_target_dtype", skip_serializing_if = "s_empty")]
    pub target_dtype: String,
    #[serde(rename = "a_columns_permuted_hf_to_gguf", skip_serializing_if = "is_false")]
    pub a_permuted: bool,
}

/// The optional merge into a base copy.
#[derive(Debug, Clone, Default, Serialize)]
pub struct HealMerge {
    pub base: String,
    pub base_pre_sha256: String,
    pub base_post_sha256: String,
    pub out: String,
    pub out_sha256: String,
    pub scale: f64,
    pub targets: Vec<String>,
    pub header_bytes_identical: bool,
    pub changed_tensors: Vec<String>,
    pub gate: Option<VerifySubspaceReport>,
    pub published: bool,
    #[serde(skip_serializing_if = "s_empty")]
    pub quantization_note: String,
}

/// The full heal payload.
#[derive(Debug, Clone, Serialize)]
pub struct HealReport {
    pub schema: String,
    pub status: String,
    #[serde(skip_serializing_if = "s_empty")]
    pub error: String,
    pub options: Option<HealOptions>,
    pub lora_pre_sha256: String,
    pub lora_post_sha256: String,
    pub protected: Option<ProtectedSet>,
    pub f64_tol: f64,
    pub serialized_tol: f64,
    #[serde(rename = "max_leak_after_f64")]
    pub max_leak_f64: f64,
    #[serde(rename = "max_leak_after_serialized")]
    pub max_leak_serial: f64,
    pub pairs: Vec<HealPair>,
    #[serde(skip_serializing_if = "s_empty")]
    pub out_sha256: String,
    pub out_published: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merge: Option<HealMerge>,
    pub healed_b_dtype_note: String,
}

struct LockedPair {
    pair: LoraPair,
    a: Vec<f64>,
    b: Vec<f64>,
}

// ---------------------------------------------------------------- filesystem helpers

fn clean_path(p: &str) -> String {
    let rooted = p.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for c in p.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                if matches!(parts.last(), Some(l) if *l != "..") {
                    parts.pop();
                } else if !rooted {
                    parts.push("..");
                }
            }
            c => parts.push(c),
        }
    }
    let joined = parts.join("/");
    match (rooted, joined.is_empty()) {
        (true, _) => format!("/{}", joined),
        (false, true) => ".".into(),
        (false, false) => joined,
    }
}

fn parent_dir(p: &str) -> PathBuf {
    match Path::new(p).parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

fn path_is_free(p: &str) -> bool {
    matches!(std::fs::symlink_metadata(p), Err(e) if e.kind() == std::io::ErrorKind::NotFound)
}

fn create_temp(dir: &Path, prefix: &str) -> Result<(File, PathBuf)> {
    static CTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for _ in 0..10000 {
        let n = CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let path = dir.join(format!("{}{}-{}-{}", prefix, std::process::id(), t, n));
        let mut oo = OpenOptions::new();
        oo.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            oo.mode(0o600); // Go's os.CreateTemp
        }
        match oo.open(&path) {
            Ok(f) => return Ok((f, path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err(Error::msg("could not create a unique temporary file"))
}

fn stage_bytes(dir: &Path, data: &[u8]) -> Result<PathBuf> {
    let (mut f, path) = create_temp(dir, ".wsp-heal-")?;
    let r = f.write_all(data).and_then(|_| f.sync_all());
    drop(f);
    if let Err(e) = r {
        let _ = std::fs::remove_file(&path);
        return Err(e.into());
    }
    Ok(path)
}

/// Atomically publish `tmp` as `dst` without replacing anything (a hard link
/// refuses an existing target, symlinks included), then drop `tmp`.
fn publish_file_new(tmp: &Path, dst: &str) -> Result<()> {
    std::fs::hard_link(tmp, dst)?;
    std::fs::remove_file(tmp)?;
    Ok(())
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

// ---------------------------------------------------------------- heal

/// Locks a LoRA away from the protected subspace and optionally merges the
/// locked update into a copy of a base model. All requested outputs are
/// staged, verified against re-read bytes, and only then published; on any
/// failure nothing is left at the destinations. As the oracle, a report is
/// returned even when a gate fails (`Error::Gate`).
pub fn heal(o: &HealOptions) -> (Option<HealReport>, Result<()>) {
    let mut opts = o.clone();
    if opts.scale == 0.0 {
        opts.scale = 1.0;
    }
    if opts.tol == 0.0 {
        opts.tol = DEFAULT_SUBSPACE_TOL;
    }
    let mut rep = HealReport {
        schema: HEAL_SCHEMA.into(),
        status: "failed".into(),
        error: String::new(),
        options: Some(opts.clone()),
        lora_pre_sha256: String::new(),
        lora_post_sha256: String::new(),
        protected: None,
        f64_tol: HEAL_F64_TOL,
        serialized_tol: HEAL_SERIALIZED_TOL,
        max_leak_f64: 0.0,
        max_leak_serial: 0.0,
        pairs: Vec::new(),
        out_sha256: String::new(),
        out_published: false,
        merge: None,
        healed_b_dtype_note: "locked B is stored as F32 (F64 if the input B was F64) regardless of the input dtype: BF16/F16 rounding of B_locked would reintroduce ~1e-3 relative leakage".into(),
    };
    let mut cleanup: Vec<PathBuf> = Vec::new();
    let r = heal_inner(&opts, &mut rep, &mut cleanup);
    if let Err(e) = &r {
        rep.status = "failed".into();
        rep.error = e.to_string();
        rep.out_published = false;
        for p in &cleanup {
            let _ = std::fs::remove_file(p);
        }
    }
    (Some(rep), r)
}

fn heal_inner(o: &HealOptions, rep: &mut HealReport, cleanup: &mut Vec<PathBuf>) -> Result<()> {
    if o.lora.is_empty() || o.vectors.is_empty() || o.out.is_empty() {
        return Err(Error::msg("--lora, --vectors and --out are required"));
    }
    if o.base.is_empty() != o.merge_out.is_empty() {
        return Err(Error::msg("--base and --merge-out must be supplied together"));
    }
    if o.scale < 0.0 || !o.scale.is_finite() || o.tol < 0.0 || !o.tol.is_finite() {
        return Err(Error::msg("scale must be a positive finite number and tol non-negative"));
    }
    for dst in [&o.out, &o.merge_out] {
        if dst.is_empty() {
            continue;
        }
        if !path_is_free(dst) {
            return Err(Error::msg(format!("output must be a new path: {}", dst)));
        }
    }
    if !o.merge_out.is_empty() && clean_path(&o.merge_out) == clean_path(&o.out) {
        return Err(Error::msg("--out and --merge-out must differ"));
    }

    let ps = load_protected_directions(&o.vectors)?;
    rep.protected = Some(ps.clone());

    let lora = open_strict_lora(&o.lora, rep)?;
    let ll = &lora.model.layout;
    if ll.format != "safetensors" {
        return Err(Error::msg(format!("{}: the LoRA input must be SafeTensors, got {}", o.lora, ll.format)));
    }
    let pairs = parse_lora(ll)?;
    let lbytes = lora.model.bytes();

    let mut work: Vec<LockedPair> = Vec::new();
    let mut out_tensors: Vec<BuildTensor> = Vec::new();
    for p in &pairs {
        let (d, r, d_in) = (p.b.logical_out, p.b.logical_in, p.a.logical_in);
        if d != ps.dim {
            return Err(Error::msg(format!(
                "module {}: B has d_out={} but the protected directions have dimension {}; only residual-space writers can be locked (refusing rather than leaving this module unlocked)",
                p.stem, d, ps.dim
            )));
        }
        let mut b = matrix_of(lbytes, &p.b)?;
        let a = matrix_of(lbytes, &p.a)?;
        let mut hp = HealPair {
            module: p.stem.clone(),
            a_tensor: p.a_name.clone(),
            b_tensor: p.b_name.clone(),
            d_out: d,
            rank: r,
            d_in,
            b_fro_before: norm2_nf(&b),
            ..Default::default()
        };
        hp.leak_before = leakage_of_columns(&ps.basis, &b, d, r);
        project_columns(&ps.basis, &mut b, d, r);
        hp.b_fro_after = norm2_nf(&b);
        hp.leak_after_f64 = leakage_of_columns(&ps.basis, &b, d, r);
        if hp.leak_after_f64 >= HEAL_F64_TOL {
            return Err(Error::gate(format!(
                "module {}: float64 residual leakage {} is not below {} (column magnitudes too large for an absolute bound)",
                p.stem,
                fmt_e(hp.leak_after_f64, 3),
                fmt_e(HEAL_F64_TOL, 0)
            )));
        }
        if hp.leak_after_f64 > rep.max_leak_f64 {
            rep.max_leak_f64 = hp.leak_after_f64;
        }
        let serial = if p.b.dtype == "F64" { "F64" } else { "F32" };
        hp.serialized_dtype = serial.to_string();
        let bsz = dtype_size(serial);
        let mut b_buf = TensorBuffer::new(serial, vec![0u8; b.len() * bsz as usize]);
        for (i, x) in b.iter().enumerate() {
            b_buf.set_float(i as i64, *x);
        }
        let a_raw = lora.model.tensor_bytes(&p.a).to_vec();
        work.push(LockedPair { pair: p.clone(), a, b });
        rep.pairs.push(hp);
        out_tensors.push(BuildTensor { name: p.a_name.clone(), dtype: p.a.dtype.clone(), shape: p.a.shape.clone(), data: a_raw });
        out_tensors.push(BuildTensor { name: p.b_name.clone(), dtype: serial.to_string(), shape: p.b.shape.clone(), data: b_buf.data });
    }
    out_tensors.sort_by(|x, y| x.name.cmp(&y.name));
    let mut meta: BTreeMap<String, String> = BTreeMap::new();
    meta.insert("format".into(), "pt".into());
    meta.insert("crucible.wsp.heal".into(), "v1".into());
    meta.insert("crucible.wsp.heal.protected_dim".into(), ps.dim.to_string());
    meta.insert("crucible.wsp.heal.protected_k".into(), ps.k.to_string());
    let healed = build_safetensors(&out_tensors, &meta)?;
    let out_dir = parent_dir(&o.out);
    std::fs::create_dir_all(&out_dir)?;
    let tmp_out = stage_bytes(&out_dir, &healed)?;
    cleanup.push(tmp_out.clone());

    // Re-read the staged bytes: the guarantee is claimed on what was written.
    {
        let staged = Model::open(&s(&tmp_out)).map_err(|e| Error::msg(format!("staged LoRA does not re-parse: {}", e)))?;
        let sb = staged.bytes();
        for (i, w) in work.iter().enumerate() {
            let (bt, at) = match (staged.layout.tensor(&w.pair.b_name), staged.layout.tensor(&w.pair.a_name)) {
                (Ok(b), Ok(a)) => (b, a),
                _ => return Err(Error::msg(format!("staged LoRA lost a tensor of module {}", w.pair.stem))),
            };
            let bm = matrix_of(sb, bt)?;
            rep.pairs[i].leak_after_serialize = leakage_of_columns(&ps.basis, &bm, bt.logical_out, bt.logical_in);
            if rep.pairs[i].leak_after_serialize > rep.max_leak_serial {
                rep.max_leak_serial = rep.pairs[i].leak_after_serialize;
            }
            rep.pairs[i].a_unchanged = staged.tensor_bytes(at) == lora.model.tensor_bytes(&w.pair.a);
            if !rep.pairs[i].a_unchanged {
                return Err(Error::msg(format!("module {}: lora_A bytes changed in the staged file", w.pair.stem)));
            }
            if rep.pairs[i].leak_after_serialize >= HEAL_SERIALIZED_TOL {
                return Err(Error::gate(format!(
                    "module {}: leakage after serialization {} is not below {}",
                    w.pair.stem,
                    fmt_e(rep.pairs[i].leak_after_serialize, 3),
                    fmt_e(HEAL_SERIALIZED_TOL, 0)
                )));
            }
        }
    }
    rep.out_sha256 = file_sha256(&s(&tmp_out))?.0;

    let mut tmp_merge: Option<PathBuf> = None;
    if !o.merge_out.is_empty() {
        let mut m = HealMerge {
            base: o.base.clone(),
            out: o.merge_out.clone(),
            scale: o.scale,
            targets: Vec::new(),
            changed_tensors: Vec::new(),
            quantization_note:
                "W + DeltaW is rounded into the base tensor dtype; the gate measures the REALIZED weight difference, not the float64 intent"
                    .into(),
            ..Default::default()
        };
        let r = heal_merge(o, &ps, &work, rep, cleanup, &mut m);
        rep.merge = Some(m);
        tmp_merge = Some(r?);
    }

    // Post-condition: inputs untouched while we worked.
    let post = file_sha256(&o.lora)?.0;
    rep.lora_post_sha256 = post.clone();
    if post != rep.lora_pre_sha256 {
        return Err(Error::msg("LoRA input changed during healing"));
    }
    // Publish: all outputs or none.
    publish_file_new(&tmp_out, &o.out)?;
    cleanup.push(PathBuf::from(&o.out)); // a later failure must also retract this publication
    if let Some(tm) = tmp_merge {
        publish_file_new(&tm, &o.merge_out)?;
        if let Some(m) = rep.merge.as_mut() {
            m.published = true;
        }
    }
    rep.out_published = true;
    rep.status = "healed".into();
    Ok(())
}

fn open_strict_lora(path: &str, rep: &mut HealReport) -> Result<OpenModel> {
    // Mirrors the oracle order: open, hash (recorded even if layout then fails), validate.
    let id = file_id(path);
    let model = Model::open(path)?;
    rep.lora_pre_sha256 = sha256_hex(model.bytes());
    strict_layout_checks(model.bytes(), &model.layout).map_err(|e| Error::msg(format!("{}: {}", path, e)))?;
    let sha256 = rep.lora_pre_sha256.clone();
    Ok(OpenModel { model, sha256, id })
}

// ---------------------------------------------------------------- merge

fn merge_rows(f: &mut File, t: &TensorLocation, b: &[f64], a: &[f64], d: usize, r: usize, d_in: usize, scale: f64) -> Result<()> {
    let w = dtype_size(&t.dtype);
    let row_bytes = d_in as i64 * w;
    let mut row = TensorBuffer::new(&t.dtype, vec![0u8; row_bytes as usize]);
    let mut old = vec![0f64; d_in];
    let mut delta = vec![0f64; d_in];
    for o in 0..d {
        let off = t.offset + o as i64 * row_bytes;
        f.seek(SeekFrom::Start(off as u64))?;
        f.read_exact(&mut row.data).map_err(|e| Error::msg(format!("read {} row {}: {}", t.name, o, e)))?;
        decode_row(&t.dtype, &row.data, 0, d_in, &mut old);
        for x in delta.iter_mut() {
            *x = 0.0;
        }
        for k in 0..r {
            axpy_nf(&mut delta, b[o * r + k], &a[k * d_in..(k + 1) * d_in]);
        }
        for i in 0..d_in {
            if !old[i].is_finite() {
                return Err(Error::msg(format!(
                    "base tensor {} holds a non-finite value at row {} col {}; refusing to merge into it",
                    t.name, o, i
                )));
            }
            let sd: f64 = scale * delta[i];
            row.set_float(i as i64, old[i] + sd);
        }
        f.seek(SeekFrom::Start(off as u64))?;
        f.write_all(&row.data).map_err(|e| Error::msg(format!("write {} row {}: {}", t.name, o, e)))?;
    }
    Ok(())
}

fn heal_merge(
    o: &HealOptions,
    ps: &ProtectedSet,
    work: &[LockedPair],
    rep: &mut HealReport,
    cleanup: &mut Vec<PathBuf>,
    m: &mut HealMerge,
) -> Result<PathBuf> {
    let base = open_strict(&o.base)?;
    m.base_pre_sha256 = base.sha256.clone();
    let bl = &base.model.layout;

    struct Target {
        t: TensorLocation,
        pair_idx: usize,
        a_eff: Vec<f64>,
    }
    let mut targets: Vec<Target> = Vec::new();
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    for (i, w) in work.iter().enumerate() {
        let mut found: Option<(TensorLocation, bool)> = None;
        let mut tried: Vec<String> = Vec::new();
        for c in lora_target_candidates(&w.pair.stem, &bl.format) {
            tried.push(c.name.clone());
            if let Ok(tt) = bl.tensor(&c.name) {
                found = Some((tt.clone(), c.hf_mapped));
                break;
            }
        }
        let (t, hf_mapped) = match found {
            Some(x) => x,
            None => return Err(Error::msg(format!("module {}: no base tensor found (tried {})", w.pair.stem, tried.join(", ")))),
        };
        let p = &w.pair;
        if !t.is_2d() || !is_float_dtype(&t.dtype) {
            return Err(Error::msg(format!(
                "module {} -> {}: target must be a 2-D float tensor (dtype {}, shape {})",
                p.stem,
                t.name,
                t.dtype,
                fmt_shape(&t.shape)
            )));
        }
        if t.logical_out != p.b.logical_out || t.logical_in != p.a.logical_in {
            return Err(Error::msg(format!(
                "module {} -> {}: base logical [d_out x d_in] = [{} x {}] but LoRA implies [{} x {}]",
                p.stem, t.name, t.logical_out, t.logical_in, p.b.logical_out, p.a.logical_in
            )));
        }
        if let Some(prev) = seen.get(&t.name) {
            return Err(Error::msg(format!("modules {} and {} both target {}", prev, p.stem, t.name)));
        }
        seen.insert(t.name.clone(), p.stem.clone());
        let mut a_eff = w.a.clone();
        if hf_mapped && bl.format == "gguf" && is_qwen35_ssm_out_tensor(&t.name) {
            let lay = SsmOutLayout::from_meta(&bl.metadata)
                .map_err(|e| Error::msg(format!("module {}: qwen3.5 ssm_out layout: {}", p.stem, e)))?;
            let perm = SsmOutPermutation::new(t.logical_in, &lay).map_err(|e| Error::msg(format!("module {}: {}", p.stem, e)))?;
            let (r, d_in) = (p.a.logical_out, p.a.logical_in);
            for k in 0..r {
                for g in 0..d_in {
                    a_eff[k * d_in + g] = w.a[k * d_in + perm.perm()[g]]; // GGUF input column g reads HF column perm[g]
                }
            }
            rep.pairs[i].a_permuted = true;
        }
        rep.pairs[i].target = t.name.clone();
        rep.pairs[i].target_dtype = t.dtype.clone();
        m.targets.push(t.name.clone());
        targets.push(Target { t, pair_idx: i, a_eff });
    }

    let merge_dir = parent_dir(&o.merge_out);
    std::fs::create_dir_all(&merge_dir)?;
    let (mut tmp, tmp_path) = create_temp(&merge_dir, ".wsp-merge-")?;
    cleanup.push(tmp_path.clone());
    tmp.write_all(base.model.bytes())?;
    for tg in &targets {
        let p = &work[tg.pair_idx].pair;
        merge_rows(&mut tmp, &tg.t, &work[tg.pair_idx].b, &tg.a_eff, p.b.logical_out, p.b.logical_in, p.a.logical_in, o.scale)?;
    }
    tmp.sync_all()?;
    drop(tmp);
    m.base_post_sha256 = file_sha256(&o.base)?.0;
    if m.base_post_sha256 != m.base_pre_sha256 {
        return Err(Error::msg("base changed while it was being merged"));
    }

    // Identity: the header and every untouched tensor must be byte-identical.
    let tmp_s = s(&tmp_path);
    let v = verify_patch(&o.base, &tmp_s, &[]).map_err(|e| Error::msg(format!("merge identity check: {}", e)))?;
    m.header_bytes_identical = v.header_identical;
    let changed: Vec<String> = v.changed_tensors.clone().unwrap_or_default();
    m.changed_tensors.extend(changed.iter().cloned());
    if !v.header_identical {
        return Err(Error::msg("merged header differs from the base"));
    }
    let is_target: HashSet<&String> = m.targets.iter().collect();
    for c in &changed {
        if !is_target.contains(c) {
            return Err(Error::msg(format!("merge changed non-target tensor {}", c)));
        }
    }

    let mut gate = VerifySubspaceReport::new(&o.base, &tmp_s, o.tol);
    let gate_opts = VerifySubspaceOptions {
        base: o.base.clone(),
        healed: tmp_s.clone(),
        vectors: String::new(),
        protected: Some(ps.clone()),
        tensors: m.targets.clone(),
        tol: o.tol,
    };
    let gr = verify_subspace_inner(&gate_opts, &mut gate);
    let verdict = gate.verdict.clone();
    let (max_leak, tol, reason) = (gate.max_leak, gate.tol, gate.reason.clone());
    m.gate = Some(gate);
    gr?;
    if verdict != "pass" {
        let msg = format!(
            "merge gate {}: max realized leakage {} vs tol {} ({}); no merged model was published",
            verdict,
            fmt_e(max_leak, 3),
            fmt_e(tol, 1),
            reason
        );
        if verdict == "leak" {
            return Err(Error::gate(msg));
        }
        return Err(Error::msg(msg)); // unknown: could not measure, which is exit 2, not a measured failure
    }
    m.out_sha256 = file_sha256(&tmp_s)?.0;
    Ok(tmp_path)
}

// ---------------------------------------------------------------- verify-subspace

/// Options for `verify-subspace`. `protected`, when set, replaces loading
/// `vectors`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct VerifySubspaceOptions {
    pub base: String,
    pub healed: String,
    #[serde(skip_serializing_if = "s_empty")]
    pub vectors: String,
    #[serde(skip)]
    pub protected: Option<ProtectedSet>,
    pub tensors: Vec<String>,
    pub tol: f64,
}

/// Per-tensor leakage measurement.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SubspaceTensor {
    pub name: String,
    pub dtype: String,
    #[serde(skip_serializing_if = "is_zero")]
    pub logical_out: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub logical_in: usize,
    pub verdict: String,
    #[serde(skip_serializing_if = "s_empty")]
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_leak: Option<f64>,
    #[serde(rename = "argmax_direction", skip_serializing_if = "s_empty")]
    pub argmax_direction: String,
    pub argmax_col: usize,
    pub delta_fro: f64,
    pub changed_elements: i64,
    pub elements: i64,
    pub delta_is_zero: bool,
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}

/// The full verify-subspace payload.
#[derive(Debug, Clone, Serialize)]
pub struct VerifySubspaceReport {
    pub schema: String,
    pub base: String,
    pub healed: String,
    pub base_pre_sha256: String,
    pub base_post_sha256: String,
    pub healed_pre_sha256: String,
    pub healed_post_sha256: String,
    pub inputs_unchanged: bool,
    pub protected: Option<ProtectedSet>,
    pub tol: f64,
    pub tensors: Vec<SubspaceTensor>,
    pub max_leak: f64,
    #[serde(rename = "changed_tensors_not_checked")]
    pub unchecked_changed: Vec<String>,
    #[serde(skip_serializing_if = "s_empty")]
    pub identity_note: String,
    pub verdict: String,
    #[serde(skip_serializing_if = "s_empty")]
    pub reason: String,
}

impl VerifySubspaceReport {
    fn new(base: &str, healed: &str, tol: f64) -> Self {
        VerifySubspaceReport {
            schema: VERIFY_SUBSPACE_SCHEMA.into(),
            base: base.into(),
            healed: healed.into(),
            base_pre_sha256: String::new(),
            base_post_sha256: String::new(),
            healed_pre_sha256: String::new(),
            healed_post_sha256: String::new(),
            inputs_unchanged: false,
            protected: None,
            tol,
            tensors: Vec::new(),
            max_leak: 0.0,
            unchecked_changed: Vec::new(),
            identity_note: String::new(),
            verdict: String::new(),
            reason: String::new(),
        }
    }
}

/// Measures `max_{k,c} |D_k^T (W_healed - W_base)[:, c]|` on each listed
/// tensor, directly from the payload bytes of both files.
pub fn verify_subspace(o: &VerifySubspaceOptions) -> Result<VerifySubspaceReport> {
    let mut rep = VerifySubspaceReport::new(&o.base, &o.healed, o.tol);
    verify_subspace_inner(o, &mut rep)?;
    Ok(rep)
}

fn verify_subspace_inner(o: &VerifySubspaceOptions, rep: &mut VerifySubspaceReport) -> Result<()> {
    if rep.tol == 0.0 {
        rep.tol = DEFAULT_SUBSPACE_TOL;
    }
    if rep.tol < 0.0 || !rep.tol.is_finite() {
        return Err(Error::msg("tol must be a non-negative finite number"));
    }
    if o.base.is_empty() || o.healed.is_empty() || o.tensors.is_empty() {
        return Err(Error::msg("--base, --healed and --tensors are required"));
    }
    let ps: ProtectedSet = match &o.protected {
        Some(p) => p.clone(),
        None => {
            if o.vectors.is_empty() {
                return Err(Error::msg("--vectors is required"));
            }
            load_protected_directions(&o.vectors)?
        }
    };
    rep.protected = Some(ps.clone());

    let base = open_strict(&o.base)?;
    let healed = open_strict(&o.healed)?;
    rep.base_pre_sha256 = base.sha256.clone();
    rep.healed_pre_sha256 = healed.sha256.clone();
    let (bl, hl) = (&base.model.layout, &healed.model.layout);
    let (bb_all, hb_all) = (base.model.bytes(), healed.model.bytes());

    let mut any_leak = false;
    let mut any_unknown = false;
    let mut checked: HashSet<&str> = HashSet::new();
    for name in &o.tensors {
        checked.insert(name.as_str());
        let mut st = SubspaceTensor { name: name.clone(), verdict: "unknown".into(), ..Default::default() };
        let mut fail = |mut st: SubspaceTensor, reason: String, rep: &mut VerifySubspaceReport| {
            st.reason = reason;
            any_unknown = true;
            rep.tensors.push(st);
        };
        let bt = bl.tensor(name);
        let ht = hl.tensor(name);
        let (bt, ht) = match (bt, ht) {
            (Err(_), _) => {
                fail(st, "tensor missing from the base model".into(), rep);
                continue;
            }
            (_, Err(_)) => {
                fail(st, "tensor missing from the healed model".into(), rep);
                continue;
            }
            (Ok(b), Ok(h)) => (b, h),
        };
        st.dtype = bt.dtype.clone();
        st.logical_out = bt.logical_out;
        st.logical_in = bt.logical_in;
        if bt.dtype != ht.dtype || bt.logical_out != ht.logical_out || bt.logical_in != ht.logical_in || bt.shape.len() != ht.shape.len() {
            fail(st, "base and healed tensors differ in dtype or shape".into(), rep);
            continue;
        }
        if !bt.is_2d() || !is_float_dtype(&bt.dtype) {
            let reason = format!("tensor must be a 2-D float matrix (dtype {}, shape {})", bt.dtype, fmt_shape(&bt.shape));
            fail(st, reason, rep);
            continue;
        }
        if bt.logical_out != ps.dim {
            let reason = format!("tensor d_out={} but the protected directions have dimension {}", bt.logical_out, ps.dim);
            fail(st, reason, rep);
            continue;
        }
        let (rows, cols) = (bt.logical_out, bt.logical_in);
        let w = dtype_size(&bt.dtype);
        let rb = (cols as i64 * w) as usize;
        let mut b_row = vec![0f64; cols];
        let mut h_row = vec![0f64; cols];
        let mut d_row = vec![0f64; cols];
        let mut acc: Vec<Vec<f64>> = (0..ps.k).map(|_| vec![0f64; cols]).collect();
        let mut sq = 0f64;
        let mut nonfinite = false;
        let mut r = 0;
        while r < rows && !nonfinite {
            let bo = bt.offset as usize + r * rb;
            let ho = ht.offset as usize + r * rb;
            decode_row(&bt.dtype, &bb_all[bo..bo + rb], 0, cols, &mut b_row);
            decode_row(&ht.dtype, &hb_all[ho..ho + rb], 0, cols, &mut h_row);
            for c in 0..cols {
                let d = h_row[c] - b_row[c];
                if !d.is_finite() {
                    nonfinite = true;
                    break;
                }
                d_row[c] = d;
                let dd: f64 = d * d;
                sq += dd;
                if d != 0.0 {
                    st.changed_elements += 1;
                }
            }
            if nonfinite {
                break;
            }
            for k in 0..ps.k {
                axpy_nf(&mut acc[k], ps.basis[k][r], &d_row);
            }
            r += 1;
        }
        st.elements = rows as i64 * cols as i64;
        if nonfinite {
            fail(st, "a non-finite weight in the base or healed tensor makes the difference undefined".into(), rep);
            continue;
        }
        let mut worst = 0f64;
        for (k, row) in acc.iter().enumerate() {
            for (c, x) in row.iter().enumerate() {
                let a = x.abs();
                if a > worst {
                    worst = a;
                    st.argmax_col = c;
                    st.argmax_direction = ps.names[k].clone();
                }
            }
        }
        st.max_leak = Some(worst);
        st.delta_fro = sq.sqrt();
        st.delta_is_zero = st.changed_elements == 0;
        if worst <= rep.tol {
            st.verdict = "pass".into();
            if st.delta_is_zero {
                st.reason = "vacuous: the healed tensor is bit-identical to the base, so nothing could leak".into();
            }
        } else {
            st.verdict = "leak".into();
            st.reason = format!("max |D^T dW| {} exceeds tol {}", fmt_e(worst, 3), fmt_e(rep.tol, 1));
            any_leak = true;
        }
        if worst > rep.max_leak {
            rep.max_leak = worst;
        }
        rep.tensors.push(st);
    }

    // Which other tensors changed? Informational: they are not covered.
    match verify_patch(&o.base, &o.healed, &[]) {
        Ok(v) => {
            for c in v.changed_tensors.clone().unwrap_or_default() {
                if !checked.contains(c.as_str()) {
                    rep.unchecked_changed.push(c);
                }
            }
            if !v.header_identical {
                rep.identity_note = "the two headers are not byte-identical".into();
            }
        }
        Err(e) => {
            rep.identity_note = format!("whole-model change scan unavailable: {}", e);
        }
    }

    rep.base_post_sha256 = file_sha256(&o.base)?.0;
    rep.healed_post_sha256 = file_sha256(&o.healed)?.0;
    let (bcur, hcur) = (file_id(&o.base), file_id(&o.healed));
    rep.inputs_unchanged = bcur.is_some()
        && hcur.is_some()
        && bcur == base.id
        && hcur == healed.id
        && rep.base_pre_sha256 == rep.base_post_sha256
        && rep.healed_pre_sha256 == rep.healed_post_sha256;
    if !rep.inputs_unchanged {
        rep.verdict = "unknown".into();
        rep.reason = "an input changed while it was being read".into();
    } else if any_unknown {
        rep.verdict = "unknown".into();
        rep.reason = "at least one tensor could not be measured".into();
    } else if any_leak {
        rep.verdict = "leak".into();
        rep.reason = "a protected direction leaked into the weight difference".into();
    } else {
        rep.verdict = "pass".into();
    }
    Ok(())
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }

    #[test]
    fn go_style_float_formats() {
        assert_eq!(fmt_e(1.234567e-7, 3), "1.235e-07");
        assert_eq!(fmt_e(1e-12, 0), "1e-12");
        assert_eq!(fmt_e(1e-5, 1), "1.0e-05");
        assert_eq!(fmt_e(0.0, 3), "0.000e+00");
        assert_eq!(fmt_e(12345.678, 3), "1.235e+04");
        assert_eq!(fmt_g(0.0, 4), "0");
        assert_eq!(fmt_g(1.5, 4), "1.5");
        assert_eq!(fmt_g(100.0, 4), "100");
        assert_eq!(fmt_g(12345.0, 4), "1.234e+04");
        assert_eq!(fmt_g(0.00001234, 4), "1.234e-05");
        assert_eq!(fmt_g(0.0001234, 4), "0.0001234");
        assert_eq!(fmt_g(-3.14161, 4), "-3.142");
        assert_eq!(fmt_g(1234.0, 4), "1234");
    }

    #[test]
    fn lora_role_matches_regex_semantics() {
        assert_eq!(lora_role("x.lora_A.weight"), Some((1, b'A', 8)));
        assert_eq!(lora_role("x.lora_B"), Some((1, b'B', 8)));
        assert_eq!(lora_role("x.lora_C.weight"), None);
        assert_eq!(lora_role("x.lora_Ax"), None);
        assert_eq!(lora_role("lora_A.weight"), None);
    }

    #[test]
    fn hf_names_map_to_gguf() {
        assert_eq!(hf_to_gguf_name("model.layers.3.self_attn.o_proj").as_deref(), Some("blk.3.attn_output.weight"));
        assert_eq!(hf_to_gguf_name("model.language_model.layers.12.linear_attn.out_proj").as_deref(), Some("blk.12.ssm_out.weight"));
        assert_eq!(hf_to_gguf_name("layers.0.mlp.down_proj").as_deref(), Some("blk.0.ffn_down.weight"));
        assert_eq!(hf_to_gguf_name("model.layers.0.mlp.bogus_proj"), None);
        assert_eq!(hf_to_gguf_name("model.layers.x.mlp.down_proj"), None);
        assert_eq!(hf_to_gguf_name("model.layers.0.mlp.down_proj.extra"), None);
    }

    #[test]
    fn clean_path_is_lexical() {
        assert_eq!(clean_path("a//b/./c/../d"), "a/b/d");
        assert_eq!(clean_path("/a/../.."), "/");
        assert_eq!(clean_path(""), ".");
        assert_eq!(clean_path("../x"), "../x");
    }

    #[test]
    fn project_then_heal_removes_protected_subspace_without_orthonormal_input() {
        let (k, d, r) = (3usize, 16usize, 4usize);
        let mut seed = 7u64;
        let mut dirs: Vec<f32> = (0..k * d).map(|_| lcg(&mut seed) as f32).collect();
        // make rows deliberately non-orthogonal and unequal in norm
        for i in 0..d {
            dirs[d + i] += 0.8 * dirs[i];
            dirs[2 * d + i] *= 5.0;
        }
        let orig: Vec<f32> = (0..d * r).map(|_| lcg(&mut seed) as f32 * 3.0).collect();
        let mut b = orig.clone();
        let leak = project_then_heal_f32(&dirs, k, d, &mut b, r).unwrap();
        assert!(leak < 1e-6, "leak {}", leak);
        assert_ne!(orig, b);
        // idempotent in the measurable sense
        let mut b2 = b.clone();
        let leak2 = project_then_heal_f32(&dirs, k, d, &mut b2, r).unwrap();
        assert!(leak2 < 1e-6);
        for (x, y) in b.iter().zip(b2.iter()) {
            assert!((x - y).abs() < 1e-5);
        }
        // the component removed is exactly the span of the directions: a B whose
        // columns already lie in the complement is (numerically) unchanged
        let mut b3 = b.clone();
        project_then_heal_f32(&dirs, k, d, &mut b3, r).unwrap();
        for (x, y) in b.iter().zip(b3.iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn project_then_heal_rejects_bad_inputs() {
        let dirs = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut b = vec![1.0f32; 8];
        assert!(project_then_heal_f32(&dirs, 0, 4, &mut b, 2).is_err(), "k=0");
        assert!(project_then_heal_f32(&dirs, 1, 3, &mut b, 2).is_err(), "dimension mismatch");
        assert!(project_then_heal_f32(&dirs, 1, 4, &mut b[..7], 2).is_err(), "b length");
        let mut nan_b = b.clone();
        nan_b[3] = f32::NAN;
        assert!(project_then_heal_f32(&dirs, 1, 4, &mut nan_b, 2).is_err(), "NaN in b");
        let inf_dirs = vec![f32::INFINITY, 0.0, 0.0, 0.0];
        assert!(project_then_heal_f32(&inf_dirs, 1, 4, &mut b, 2).is_err(), "Inf in directions");
        let dep = vec![1.0f32, 2.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0];
        assert!(project_then_heal_f32(&dep, 2, 4, &mut b, 2).is_err(), "linearly dependent directions");
        assert!(project_then_heal_f32(&[0.0; 4], 1, 4, &mut b, 2).is_err(), "zero direction");
    }

    #[test]
    fn leakage_and_projection_agree_on_axis_basis() {
        let basis = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let mut m = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]; // 3 x 2 row-major
        assert_eq!(leakage_of_columns(&basis, &m, 3, 2), 4.0);
        project_columns(&basis, &mut m, 3, 2);
        assert_eq!(m, vec![0.0, 0.0, 0.0, 0.0, 5.0, 9.0]);
        assert_eq!(leakage_of_columns(&basis, &m, 3, 2), 0.0);
    }

    #[test]
    fn strict_header_rejects_duplicate_keys_and_trailing_data() {
        assert!(header_json_strict(br#"{"a":{"b":1},"c":[1,2]}"#).is_ok());
        assert!(header_json_strict(br#"{"a":1,"a":2}"#).is_err());
        assert!(header_json_strict(br#"{"a":1}x"#).is_err());
    }
}
