//! Subspace algebra: NumPy `.npy` v1.0 codec, two-pass float64 modified
//! Gram-Schmidt, and `compose` (rank1-columns top-K sparse column selector,
//! projector-subtract, `--target-shift` / `--gamma` calibration).
//! Mirrors `internal/wsp/{npy,linalg,compose}.go`.

use crate::container::{load_layout, open_regular_file, sha256_hex, Model};
use crate::dtype::*;
use crate::error::{Error, Result};
use crate::numeric::{fma_add, fma_sub};
use crate::plan::*;
use std::collections::BTreeMap;
use std::io::Read;

// ------------------------------------------------------------------ npy

const NPY_MAGIC: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];

/// Decoded `.npy` payload: C-order float32 or float64 with its shape.
#[derive(Debug, Clone)]
pub struct NumpyArray {
    /// `<f4` or `<f8`.
    pub dtype: String,
    pub shape: Vec<usize>,
    pub f32: Option<Vec<f32>>,
    pub f64: Option<Vec<f64>>,
}

impl NumpyArray {
    pub fn len(&self) -> usize {
        match (&self.f64, &self.f32) {
            (Some(v), _) => v.len(),
            (None, Some(v)) => v.len(),
            _ => 0,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn to_f64(&self) -> Vec<f64> {
        match (&self.f64, &self.f32) {
            (Some(v), _) => v.clone(),
            (None, Some(v)) => v.iter().map(|x| *x as f64).collect(),
            _ => Vec::new(),
        }
    }
}

fn find_after<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    // Matches `'?key'?\s*:\s*` and returns the remainder after the colon.
    let mut start = 0;
    while let Some(p) = text[start..].find(key) {
        let at = start + p;
        let mut rest = &text[at + key.len()..];
        rest = rest.strip_prefix('\'').unwrap_or(rest);
        let r2 = rest.trim_start();
        if let Some(r3) = r2.strip_prefix(':') {
            return Some(r3.trim_start());
        }
        start = at + key.len();
    }
    None
}

pub fn load_numpy(bytes: &[u8]) -> Result<NumpyArray> {
    if bytes.len() < 8 {
        return Err(Error::msg("npy: reading magic: unexpected EOF"));
    }
    if bytes[..6] != NPY_MAGIC {
        return Err(Error::msg(format!("npy: bad magic {:?}", String::from_utf8_lossy(&bytes[..6]))));
    }
    let major = bytes[6];
    let (header_len, hstart) = match major {
        1 => {
            if bytes.len() < 10 {
                return Err(Error::msg("npy: reading v1 header length: unexpected EOF"));
            }
            (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize)
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(Error::msg(format!("npy: reading v{} header length: unexpected EOF", major)));
            }
            (le_u32(&bytes[8..12]) as usize, 12usize)
        }
        v => return Err(Error::msg(format!("npy: unsupported major version {}", v))),
    };
    if header_len == 0 || header_len > 1 << 20 {
        return Err(Error::msg(format!("npy: implausible header length {}", header_len)));
    }
    if bytes.len() < hstart + header_len {
        return Err(Error::msg("npy: reading header body: unexpected EOF"));
    }
    let text = String::from_utf8_lossy(&bytes[hstart..hstart + header_len]).into_owned();
    let descr = {
        let r = find_after(&text, "descr").ok_or_else(|| Error::msg(format!("npy: header has no descr: {:?}", text.trim())))?;
        let r = r.strip_prefix('\'').ok_or_else(|| Error::msg(format!("npy: header has no descr: {:?}", text.trim())))?;
        let end = r.find('\'').ok_or_else(|| Error::msg(format!("npy: header has no descr: {:?}", text.trim())))?;
        r[..end].to_string()
    };
    if let Some(r) = find_after(&text, "fortran_order") {
        if r.starts_with("True") {
            return Err(Error::msg("npy: fortran_order=True is not supported; re-save in C order"));
        }
    }
    let shape_txt = {
        let r = find_after(&text, "shape").ok_or_else(|| Error::msg(format!("npy: header has no shape: {:?}", text.trim())))?;
        let r = r.strip_prefix('(').ok_or_else(|| Error::msg(format!("npy: header has no shape: {:?}", text.trim())))?;
        let end = r.find(')').ok_or_else(|| Error::msg(format!("npy: header has no shape: {:?}", text.trim())))?;
        r[..end].to_string()
    };
    let mut shape = Vec::new();
    for part in shape_txt.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let n: i64 = p.parse().map_err(|e| Error::msg(format!("npy: bad shape element {:?}: {}", p, e)))?;
        if n < 0 {
            return Err(Error::msg(format!("npy: negative shape element {}", n)));
        }
        shape.push(n as usize);
    }
    let elements = shape.iter().fold(1usize, |a, d| a.saturating_mul(*d));
    let data = &bytes[hstart + header_len..];
    match descr.as_str() {
        "<f4" | "f4" => {
            let need = 4usize.checked_mul(elements).ok_or_else(|| Error::msg("npy: element count overflow"))?;
            if data.len() < need {
                return Err(Error::msg(format!("npy: reading {} float32 elements: unexpected EOF", elements)));
            }
            let v = (0..elements).map(|i| f32::from_bits(le_u32(&data[4 * i..]))).collect();
            Ok(NumpyArray { dtype: descr, shape, f32: Some(v), f64: None })
        }
        "<f8" | "f8" => {
            let need = 8usize.checked_mul(elements).ok_or_else(|| Error::msg("npy: element count overflow"))?;
            if data.len() < need {
                return Err(Error::msg(format!("npy: reading {} float64 elements: unexpected EOF", elements)));
            }
            let v = (0..elements).map(|i| f64::from_bits(le_u64(&data[8 * i..]))).collect();
            Ok(NumpyArray { dtype: descr, shape, f32: None, f64: Some(v) })
        }
        other => Err(Error::msg(format!("npy: unsupported descr {:?} (want <f4 or <f8 little-endian float)", other))),
    }
}

pub fn load_numpy_file(path: &str) -> Result<NumpyArray> {
    let mut f = open_regular_file(path)?;
    let mut b = Vec::new();
    f.read_to_end(&mut b)?;
    load_numpy(&b)
}

fn npy_header(dtype: &str, shape: &[usize]) -> Result<Vec<u8>> {
    if shape.is_empty() {
        return Err(Error::msg("npy: empty shape"));
    }
    for n in shape {
        if *n == 0 {
            return Err(Error::msg("npy: non-positive shape element 0"));
        }
    }
    let parts: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
    let trailing = if shape.len() == 1 { "," } else { "" };
    let mut dict = format!("{{'descr': '{}', 'fortran_order': False, 'shape': ({}{}), }}", dtype, parts.join(", "), trailing);
    // numpy `_wrap_header`: hlen counts the trailing newline, then the header
    // is space-padded so that magic(6) + version(2) + len(2) + header is a
    // multiple of 64 bytes. (The Go oracle pads one byte too many; see
    // tool-gaps.json TG-SECANT-ORACLE-NPY-PAD.)
    let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
    dict.push_str(&" ".repeat(pad));
    dict.push('\n');
    let mut out = Vec::new();
    out.extend_from_slice(&NPY_MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    Ok(out)
}

/// `.npy` v1.0 bytes for float64 values, byte-identical to `numpy.save`.
pub fn npy_bytes_f64(values: &[f64], shape: &[usize]) -> Result<Vec<u8>> {
    let elements: usize = shape.iter().product();
    if elements != values.len() {
        return Err(Error::msg(format!("npy: shape {:?} holds {} elements but {} values were given", shape, elements, values.len())));
    }
    let mut out = npy_header("<f8", shape)?;
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    Ok(out)
}

pub fn npy_bytes_f32(values: &[f32], shape: &[usize]) -> Result<Vec<u8>> {
    let elements: usize = shape.iter().product();
    if elements != values.len() {
        return Err(Error::msg(format!("npy: shape {:?} holds {} elements but {} values were given", shape, elements, values.len())));
    }
    let mut out = npy_header("<f4", shape)?;
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    Ok(out)
}

// ------------------------------------------------------------------ linalg

pub const ORTHOGONALITY_TOLERANCE: f64 = 1e-12;

/// Inner product (accumulated in index order; fused only in go-arm64-fma mode).
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = 0f64;
    for i in 0..n {
        s = fma_add(s, a[i], b[i]);
    }
    s
}

pub fn l2_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let (na, nb) = (l2_norm(a), l2_norm(b));
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

/// Indices of the k largest-magnitude entries, |v| descending, ties by
/// ascending index (stable).
pub fn top_k_abs_indices(v: &[f64], k: usize) -> Vec<usize> {
    let k = k.min(v.len());
    if k == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| v[*b].abs().partial_cmp(&v[*a].abs()).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

/// Two-pass modified Gram-Schmidt in float64 over `order`, failing closed on a
/// dependent vector or when `max_{i!=j} |cos| >= 1e-12`.
pub fn gram_schmidt(named: &BTreeMap<String, Vec<f64>>, order: &[String]) -> Result<(BTreeMap<String, Vec<f64>>, GramSchmidtReport)> {
    if order.is_empty() {
        return Err(Error::msg("gram-schmidt: no vectors given"));
    }
    let mut dim: Option<usize> = None;
    for name in order {
        let v = named.get(name).ok_or_else(|| Error::msg(format!("gram-schmidt: vector {:?} listed in order but not provided", name)))?;
        match dim {
            None => dim = Some(v.len()),
            Some(d) if v.len() != d => {
                return Err(Error::msg(format!(
                    "gram-schmidt: vector {:?} has dim {}, want {} (all directions must share one subspace dimension)",
                    name,
                    v.len(),
                    d
                )))
            }
            _ => {}
        }
    }
    if dim == Some(0) {
        return Err(Error::msg("gram-schmidt: zero-dimension vectors"));
    }
    let mut rep = GramSchmidtReport { order: order.to_vec(), passes: 2, ..Default::default() };
    for i in 0..order.len() {
        for j in i + 1..order.len() {
            rep.pre_cosine.insert(format!("{}|{}", order[i], order[j]), cosine(&named[&order[i]], &named[&order[j]]));
        }
    }
    let mut basis: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for name in order {
        let mut u = named[name].clone();
        for _pass in 0..2 {
            for prev in order {
                if let Some(e) = basis.get(prev) {
                    let proj = dot(&u, e);
                    for i in 0..u.len() {
                        u[i] = fma_sub(u[i], proj, e[i]);
                    }
                }
            }
        }
        let norm = l2_norm(&u);
        if norm < 1e-9 {
            return Err(Error::msg(format!(
                "gram-schmidt: vector {:?} is linearly dependent on earlier vectors (residual norm {:.3e})",
                name, norm
            )));
        }
        for x in u.iter_mut() {
            *x /= norm;
        }
        basis.insert(name.clone(), u);
    }
    let mut max_abs = 0f64;
    for i in 0..order.len() {
        for j in i + 1..order.len() {
            let c = cosine(&basis[&order[i]], &basis[&order[j]]);
            rep.post_cosine.insert(format!("{}|{}", order[i], order[j]), c);
            if c.abs() > max_abs {
                max_abs = c.abs();
            }
        }
    }
    rep.post_cosine_max_abs = max_abs;
    if max_abs >= ORTHOGONALITY_TOLERANCE {
        return Err(Error::msg(format!(
            "gram-schmidt: post-orthogonalization max |cos| {:.3e} violates the <{:.0e} bound",
            max_abs, ORTHOGONALITY_TOLERANCE
        )));
    }
    Ok((basis, rep))
}

// ------------------------------------------------------------------ vectors

/// Vectors loaded from a directory, keyed by file stem.
#[derive(Debug, Default)]
pub struct VectorSet {
    pub by_name: BTreeMap<String, Vec<f64>>,
    pub sha256: BTreeMap<String, String>,
    pub files: BTreeMap<String, String>,
}

pub fn load_vectors(dir: &str) -> Result<VectorSet> {
    let rd = std::fs::read_dir(dir).map_err(|e| Error::msg(format!("read vectors dir {}: {}", dir, e)))?;
    let mut names: Vec<(String, bool)> = Vec::new();
    for ent in rd {
        let ent = ent?;
        let is_dir = ent.file_type().map(|t| t.is_dir()).unwrap_or(false);
        names.push((ent.file_name().to_string_lossy().into_owned(), is_dir));
    }
    names.sort();
    let mut vs = VectorSet::default();
    for (name, is_dir) in names {
        if is_dir {
            continue;
        }
        let path = std::path::Path::new(dir).join(&name);
        let ext = path.extension().map(|e| format!(".{}", e.to_string_lossy().to_lowercase())).unwrap_or_default();
        let stem = match path.extension() {
            Some(e) => name[..name.len() - e.len() - 1].to_string(),
            None => name.clone(),
        };
        let raw = std::fs::read(&path)?;
        let values: Vec<f64> = match ext.as_str() {
            ".npy" => load_numpy(&raw).map_err(|e| Error::msg(format!("vector {}: {}", name, e)))?.to_f64(),
            ".json" => {
                if let Ok(v) = serde_json::from_slice::<Vec<f64>>(&raw) {
                    v
                } else {
                    #[derive(serde::Deserialize)]
                    struct Obj {
                        values: Option<Vec<f64>>,
                    }
                    match serde_json::from_slice::<Obj>(&raw) {
                        Ok(Obj { values: Some(v) }) => v,
                        _ => return Err(Error::msg(format!("vector {}: JSON must be a number array or {{\"values\": [...]}}", name))),
                    }
                }
            }
            _ => continue,
        };
        if values.is_empty() {
            return Err(Error::msg(format!("vector {}: empty", name)));
        }
        vs.by_name.insert(stem.clone(), values);
        vs.sha256.insert(stem.clone(), sha256_hex(&raw));
        vs.files.insert(stem, name);
    }
    if vs.by_name.is_empty() {
        return Err(Error::msg(format!("vectors dir {} holds no .npy or .json vector files", dir)));
    }
    Ok(vs)
}

fn vector_layers(vs: &VectorSet, prefix: &str) -> Vec<i64> {
    let mut layers: Vec<i64> = Vec::new();
    for stem in vs.by_name.keys() {
        if let Some(rest) = stem.strip_prefix(prefix) {
            if let Some(ls) = rest.strip_prefix("_L") {
                if let Ok(n) = ls.parse::<i64>() {
                    if n >= 0 && !layers.contains(&n) {
                        layers.push(n);
                    }
                }
            }
        }
    }
    layers.sort();
    layers
}

fn h_ref_stems(token: &str, layer: i64) -> [String; 3] {
    [format!("h_{}_L{}", token, layer), format!("hm_{}_L{}", token, layer), format!("h_m_{}_L{}", token, layer)]
}

/// Frobenius norm of one tensor's payload in float64.
pub fn fro_norm_of_tensor(model_path: &str, tensor: &str) -> Result<f64> {
    let m = Model::open(model_path)?;
    let t = m.layout.tensor(tensor)?;
    if !is_float_dtype(&t.dtype) {
        return Err(Error::msg(format!("Frobenius norm of {} dtype {} not supported", tensor, t.dtype)));
    }
    let w = dtype_size(&t.dtype) as usize;
    let data = m.tensor_bytes(t);
    let mut sum_sq = 0f64;
    for i in 0..(t.size as usize / w) {
        let v = match t.dtype.as_str() {
            "F32" => le_f32(&data[i * 4..]) as f64,
            "BF16" => f32_from_bf16(le_u16(&data[i * 2..])) as f64,
            "F16" => f32_from_f16(le_u16(&data[i * 2..])) as f64,
            _ => le_f64(&data[i * 8..]),
        };
        sum_sq = fma_add(sum_sq, v, v);
    }
    Ok(sum_sq.sqrt())
}

// ------------------------------------------------------------------ compose

#[derive(Debug, Clone, Default)]
pub struct ComposeOptions {
    pub profile_name: String,
    pub profile_path: String,
    pub vectors_dir: String,
    pub target_shift: f64,
    pub has_shift: bool,
    pub gamma: f64,
    pub has_gamma: bool,
    pub base_model: String,
}

struct ResolvedSlot<'a> {
    slot: &'a Slot,
    stem: String,
    layer: i64,
    h_stem: String,
}

fn render(pattern: &str, layer: i64) -> String {
    pattern.replace("%d", &layer.to_string())
}

pub fn compose(mut opts: ComposeOptions) -> Result<Plan> {
    let profile: Profile = if !opts.profile_path.is_empty() {
        let data = std::fs::read(&opts.profile_path).map_err(|e| Error::msg(format!("read profile: {}", e)))?;
        let mut p: Profile =
            serde_json::from_slice(&data).map_err(|e| Error::msg(format!("parse profile {}: {}", opts.profile_path, e)))?;
        if !crate::namespace::accepts(PROFILE_SCHEMA, &p.schema) {
            return Err(Error::msg(format!("profile {}: schema {:?}, want {:?}", opts.profile_path, p.schema, PROFILE_SCHEMA)));
        }
        p.schema = PROFILE_SCHEMA.into();
        p.source = opts.profile_path.clone();
        p
    } else {
        builtin_profile(&opts.profile_name)?
    };
    let vs = if !opts.vectors_dir.is_empty() { Some(load_vectors(&opts.vectors_dir)?) } else { None };

    let mode: &str;
    if opts.has_shift && opts.has_gamma {
        return Err(Error::msg("pass either --target-shift or --gamma, not both"));
    } else if opts.has_gamma {
        mode = "gamma";
    } else if opts.has_shift {
        mode = "target-shift";
    } else if profile.default_target_shift != 0.0 {
        mode = "target-shift";
        opts.target_shift = profile.default_target_shift;
        opts.has_shift = true;
    } else if profile.default_gamma != 0.0 {
        mode = "gamma";
        opts.gamma = profile.default_gamma;
        opts.has_gamma = true;
    } else {
        return Err(Error::msg(format!("profile {} defines no default calibration; pass --target-shift or --gamma", profile.name)));
    }

    let mut plan = Plan { schema: PLAN_SCHEMA.into(), profile: profile.name.clone(), ..Default::default() };

    // 1. Bind direction slots to layers and vector files.
    let mut dirs: Vec<ResolvedSlot> = Vec::new();
    let mut cursor: BTreeMap<String, usize> = BTreeMap::new();
    for slot in &profile.slots {
        if !slot.needs_direction {
            continue;
        }
        let vs =
            vs.as_ref().ok_or_else(|| Error::msg(format!("slot {} needs a direction vector but no --vectors dir was given", slot.id)))?;
        let layers = vector_layers(vs, &format!("d_{}", slot.direction_token));
        if layers.is_empty() {
            return Err(Error::msg(format!(
                "no d_{}_L*.(npy|json) vector files in {} (slot {})",
                slot.direction_token, opts.vectors_dir, slot.id
            )));
        }
        let mut layer = slot.layer;
        if layer < 0 {
            let idx = *cursor.get(&slot.direction_token).unwrap_or(&0);
            if idx >= layers.len() {
                return Err(Error::msg(format!(
                    "slot {}: {} d_{}_L* layer(s) available but this slot has none left to bind; set an explicit layer via a custom profile file",
                    slot.id,
                    layers.len(),
                    slot.direction_token
                )));
            }
            layer = layers[idx];
            cursor.insert(slot.direction_token.clone(), idx + 1);
        }
        let stem = format!("d_{}_L{}", slot.direction_token, layer);
        if !vs.by_name.contains_key(&stem) {
            return Err(Error::msg(format!("slot {}: no vector file {}.(npy|json) for explicit layer {}", slot.id, stem, layer)));
        }
        let mut r = ResolvedSlot { slot, stem, layer, h_stem: String::new() };
        for cand in h_ref_stems(&slot.direction_token, layer) {
            if vs.by_name.contains_key(&cand) {
                r.h_stem = cand;
                break;
            }
        }
        dirs.push(r);
    }

    // 2. Gram-Schmidt across subspaces, in slot order.
    let mut basis: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    if !dirs.is_empty() {
        let vs = vs.as_ref().unwrap();
        let mut named = BTreeMap::new();
        let mut order = Vec::new();
        for r in &dirs {
            named.insert(r.stem.clone(), vs.by_name[&r.stem].clone());
            order.push(r.stem.clone());
        }
        let (b, rep) = gram_schmidt(&named, &order)?;
        basis = b;
        plan.gram_schmidt = Some(rep);
    }

    // 3. Materialize edits.
    for r in &dirs {
        let slot = r.slot;
        let vs = vs.as_ref().unwrap();
        let mut edit = Edit {
            id: slot.id.clone(),
            role: slot.role.clone(),
            op: slot.op.clone(),
            tensor: render(&slot.tensor_pattern, r.layer),
            tensor_st: render(&slot.tensor_st_pattern, r.layer),
            layer: r.layer,
            sign: slot.sign,
            top_k: slot.top_k,
            direction: basis[&r.stem].clone(),
            vector_sha256: vs.sha256[&r.stem].clone(),
            ..Default::default()
        };
        match slot.op.as_str() {
            "rank1-columns" => {
                if r.h_stem.is_empty() {
                    return Err(Error::msg(format!(
                        "slot {}: rank1-columns calibration needs the reference activation {}.(npy|json) (h_/hm_ stem) in {}",
                        slot.id,
                        h_ref_stems(&slot.direction_token, r.layer)[0],
                        opts.vectors_dir
                    )));
                }
                let h = &vs.by_name[&r.h_stem];
                let mut top_k = slot.top_k;
                if top_k <= 0 || top_k as usize > h.len() {
                    top_k = h.len() as i64;
                }
                let cols = top_k_abs_indices(h, top_k as usize);
                let mut energy = 0f64;
                for c in &cols {
                    energy = fma_add(energy, h[*c], h[*c]);
                }
                if energy <= 0.0 {
                    return Err(Error::msg(format!("slot {}: reference activation has zero energy on its top-{} columns", slot.id, top_k)));
                }
                let beta;
                if mode == "target-shift" {
                    let mut shift = slot.target_shift;
                    if shift == 0.0 {
                        shift = opts.target_shift;
                    }
                    if shift == 0.0 {
                        shift = profile.default_target_shift;
                    }
                    if shift == 0.0 {
                        return Err(Error::msg(format!(
                            "slot {}: no target shift (flag, slot default, or profile default) to calibrate against",
                            slot.id
                        )));
                    }
                    beta = shift.abs() / energy;
                    edit.achieved_shift = slot.sign * beta * energy;
                } else {
                    let mut gamma = opts.gamma;
                    if gamma == 0.0 {
                        gamma = profile.default_gamma;
                    }
                    let mut fro_w = profile.ref_fro_norm;
                    if !opts.base_model.is_empty() {
                        let mut f = fro_norm_of_tensor(&opts.base_model, &edit.tensor);
                        if let Err(e) = &f {
                            if load_layout(&opts.base_model).is_ok() && e.message().contains("not found") {
                                f = fro_norm_of_tensor(&opts.base_model, &edit.tensor_st);
                            }
                        }
                        fro_w = f.map_err(|e| Error::msg(format!("slot {}: gamma calibration: {}", slot.id, e)))?;
                    }
                    if fro_w == 0.0 {
                        return Err(Error::msg(format!(
                            "slot {}: gamma calibration needs --base <model> (or profile ref_fro_norm) to measure ||W||_F",
                            slot.id
                        )));
                    }
                    beta = gamma * fro_w / energy.sqrt();
                    edit.achieved_shift = slot.sign * beta * energy;
                    edit.gamma = gamma;
                }
                edit.cols = cols.iter().map(|c| *c as i64).collect();
                edit.coeffs = cols.iter().map(|c| slot.sign * beta * h[*c]).collect();
                let mut fro = 0f64;
                for c in &edit.coeffs {
                    fro = fma_add(fro, *c, *c);
                }
                edit.delta_fro_norm = fro.sqrt();
                edit.h_ref = h.clone();
                if slot.requires_h_ref {
                    edit.notes.push(format!("witness: <h_ref, delta_W^T d_hat> = {:.6} at any model scale", edit.achieved_shift));
                }
            }
            "projector-subtract" => {
                let mut gamma = 0.0;
                if opts.has_gamma {
                    gamma = opts.gamma;
                } else if slot.gamma != 0.0 {
                    gamma = slot.gamma;
                } else if profile.default_gamma != 0.0 {
                    gamma = profile.default_gamma;
                }
                if gamma == 0.0 {
                    gamma = 1.0;
                }
                edit.gamma = gamma;
                edit.notes.push(format!(
                    "W' = (I - {:.3}*d*d^T) W applied as left multiplication on the logical output axis; d^T W' = {:.3}*(1-gamma) * d^T W",
                    gamma,
                    1.0 - gamma
                ));
                // The oracle prints `%.3f` of gamma and `1-gamma`; the format
                // strings above reproduce them with the same specifiers.
            }
            other => {
                return Err(Error::msg(format!(
                    "slot {}: direction slots support rank1-columns and projector-subtract, not {:?}",
                    slot.id, other
                )));
            }
        }
        plan.edits.push(edit);
    }

    // 4. Probe / permutation slots need no vectors.
    for slot in &profile.slots {
        if slot.needs_direction {
            continue;
        }
        let mut layer = slot.layer;
        if layer < 0 && format!("{}{}", slot.tensor_pattern, slot.tensor_st_pattern).contains("%d") {
            layer = 0;
        }
        let mut edit = Edit {
            id: slot.id.clone(),
            role: slot.role.clone(),
            op: slot.op.clone(),
            tensor: render(&slot.tensor_pattern, layer),
            tensor_st: render(&slot.tensor_st_pattern, layer),
            layer,
            sign: slot.sign,
            ..Default::default()
        };
        match slot.op.as_str() {
            "set-value" => {
                let p = slot.probe.as_ref().ok_or_else(|| Error::msg(format!("slot {}: set-value needs a probe spec", slot.id)))?;
                edit.semantics = p.semantics.clone();
                edit.set_bits = p.bits.clone();
                edit.set_ints = p.ints.clone();
                edit.set_indices = p.flat_indices.clone();
                edit.token_rows = p.token_rows.clone();
            }
            "permute-columns" => {
                let p = slot
                    .probe
                    .as_ref()
                    .ok_or_else(|| Error::msg(format!("slot {}: permute-columns needs a probe spec carrying permute", slot.id)))?;
                edit.permutation = p.permute.clone();
            }
            other => return Err(Error::msg(format!("slot {}: unsupported non-direction op {:?}", slot.id, other))),
        }
        edit.notes.extend(slot.notes.iter().cloned());
        plan.edits.push(edit);
    }

    plan.calibration = CalibrationSpec { mode: mode.into(), ..Default::default() };
    if mode == "target-shift" {
        plan.calibration.target_shift = opts.target_shift;
        plan.calibration.note =
            "scale-invariant: <h_ref, delta_W^T d_hat> hits the target shift at any model width/depth (fixes the 0.8B->4B gamma port)"
                .into();
    } else {
        plan.calibration.gamma = opts.gamma;
        plan.calibration.base_model = opts.base_model.clone();
        plan.calibration.note = if opts.base_model.is_empty() {
            "magnitudes sized against the profile ref_fro_norm stand-in".into()
        } else {
            "||delta_W||_F = gamma * ||W||_F measured from the base model".into()
        };
    }
    if plan.edits.is_empty() {
        return Err(Error::msg(format!("profile {} produced no edits", profile.name)));
    }
    Ok(plan)
}

/// Load protected output directions and orthonormalize them (two-pass
/// modified Gram-Schmidt, float64). `spec` is a directory of `d_*.(npy|json)`
/// vectors (reference activations `h_*`/`hm_*`/`h_m_*` are skipped, files in
/// name order) or one `.npy` holding a vector or `[k, d]` rows. Used by the
/// `quant` witness; `heal` has its own loader mirroring the oracle exactly.
pub fn load_directions(spec: &str) -> Result<Vec<Vec<f64>>> {
    let md = std::fs::metadata(spec).map_err(|e| Error::msg(format!("directions {}: {}", spec, e)))?;
    let mut rows: Vec<(String, Vec<f64>)> = Vec::new();
    if md.is_dir() {
        let vs = load_vectors(spec)?;
        for (stem, v) in &vs.by_name {
            if stem.starts_with("h_") || stem.starts_with("hm_") || stem.starts_with("h_m_") {
                continue;
            }
            rows.push((stem.clone(), v.clone()));
        }
    } else {
        let arr = load_numpy_file(spec)?;
        let v = arr.to_f64();
        match arr.shape.len() {
            1 => rows.push(("d0".into(), v)),
            2 => {
                let (k, d) = (arr.shape[0], arr.shape[1]);
                for i in 0..k {
                    rows.push((format!("d{}", i), v[i * d..(i + 1) * d].to_vec()));
                }
            }
            n => return Err(Error::msg(format!("directions {}: {}-D array (want a vector or [k, d])", spec, n))),
        }
    }
    if rows.is_empty() {
        return Err(Error::msg(format!("directions {}: no direction vectors found", spec)));
    }
    for (n, v) in &rows {
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Error::msg(format!("direction {}: holds NaN/Inf", n)));
        }
    }
    let order: Vec<String> = rows.iter().map(|(n, _)| n.clone()).collect();
    let named: BTreeMap<String, Vec<f64>> = rows.into_iter().collect();
    let (basis, _) = gram_schmidt(&named, &order)?;
    Ok(order.iter().map(|n| basis[n].clone()).collect())
}
