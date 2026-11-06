//! `.safetensors` and `.gguf` containers: zero-copy memory-mapped parsing,
//! the logical `[d_out, d_in]` accessor (`TensorLocation::weight`), the
//! Qwen3.5 Gated-DeltaNet `ssm_out` input-axis permutation, the payload
//! patcher and the SHA-256 / byte-identity patch verifier.
//!
//! Orientation contract (identical to the Go oracle, `internal/wsp/layout.go`):
//!
//! ```text
//! GGUF 2-D (ne[0] contiguous): logical_out = ne[1], logical_in = ne[0]
//! SafeTensors 2-D (row-major): logical_out = dims[0], logical_in = dims[1]
//! ```
//!
//! Both reduce to the flat payload element index `out * logical_in + in`.
//!
//! One deliberate extension over the oracle: the oracle rejects a whole GGUF
//! file that contains any quantized tensor. Secant lists quantized tensors
//! (`Q8_0`, `Q4_K`, ...) with their exact block-derived byte size so `inspect`,
//! `verify-patch` and the block-aligned `quant` surgery work on them; every
//! float edit refuses a quantized target.

use crate::dtype::*;
use crate::error::{Error, Result};
use crate::numeric::fma_add;
use crate::plan::{Edit, Plan};
use memmap2::{Mmap, MmapMut};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::Path;

pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Serialize)]
pub struct TensorLocation {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub logical_out: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub logical_in: usize,
    #[serde(rename = "file_offset")]
    pub offset: i64,
    #[serde(rename = "byte_size")]
    pub size: i64,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub gguf_column_major: bool,
    /// ggml type id for GGUF tensors (None for SafeTensors).
    #[serde(skip)]
    pub ggml_type: Option<u32>,
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}

impl TensorLocation {
    pub fn is_2d(&self) -> bool {
        self.shape.len() == 2 && self.logical_out > 0 && self.logical_in > 0
    }

    /// Flat payload element index of logical entry `(out_idx, in_idx)`.
    pub fn weight(&self, out_idx: usize, in_idx: usize) -> Result<i64> {
        if !self.is_2d() {
            return Err(Error::msg(format!(
                "tensor {}: logical Weight(out,in) accessor requires an exactly 2-D tensor (storage dims {:?})",
                self.name, self.shape
            )));
        }
        if out_idx >= self.logical_out {
            return Err(Error::msg(format!("tensor {}: logical output index {} out of [0,{})", self.name, out_idx, self.logical_out)));
        }
        if in_idx >= self.logical_in {
            return Err(Error::msg(format!("tensor {}: logical input index {} out of [0,{})", self.name, in_idx, self.logical_in)));
        }
        Ok(out_idx as i64 * self.logical_in as i64 + in_idx as i64)
    }

    /// `weight` for inner loops whose indices were already bounds-checked.
    #[inline]
    pub fn weight_unchecked(&self, out_idx: usize, in_idx: usize) -> i64 {
        out_idx as i64 * self.logical_in as i64 + in_idx as i64
    }

    pub fn require_2d(&self, op: &str) -> Result<()> {
        if self.is_2d() {
            return Ok(());
        }
        Err(Error::msg(format!(
            "op {:?} requires an exactly 2-D tensor: {} has {} storage dim(s) {:?} (expert/batch dimensions are not supported)",
            op,
            self.name,
            self.shape.len(),
            self.shape
        )))
    }

    /// Element count from the declared shape (valid for quantized tensors too).
    pub fn elements(&self) -> i64 {
        self.shape.iter().fold(1i64, |a, d| a.wrapping_mul(*d as i64))
    }

    /// Flat element index of RAW STORAGE axis (row, col).
    pub fn element_index(&self, row: usize, col: usize) -> i64 {
        if self.gguf_column_major {
            col as i64 * self.rows as i64 + row as i64
        } else {
            row as i64 * self.cols as i64 + col as i64
        }
    }

    pub fn is_quantized(&self) -> bool {
        dtype_size(&self.dtype) == 0
    }
}

#[derive(Debug, Clone)]
pub struct Layout {
    pub format: String,
    pub path: String,
    pub file_size: i64,
    pub data_offset: i64,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<TensorLocation>,
}

impl Layout {
    pub fn tensor(&self, name: &str) -> Result<&TensorLocation> {
        self.tensors.iter().find(|t| t.name == name).ok_or_else(|| Error::msg(format!("tensor {:?} not found in {}", name, self.path)))
    }

    pub fn architecture(&self) -> &str {
        self.metadata.get("general.architecture").map(|s| s.as_str()).unwrap_or("")
    }

    /// Integer metadata value, parsed the way `fmt.Sscanf("%d")` parses it
    /// (leading integer prefix).
    pub fn meta_int(&self, key: &str) -> Option<i64> {
        scan_int_prefix(self.metadata.get(key)?)
    }

    pub fn block_count(&self) -> i64 {
        let key = format!("{}.block_count", self.architecture());
        if let Some(n) = self.meta_int(&key) {
            if n > 0 {
                return n;
            }
        }
        let mut max: i64 = -1;
        for t in &self.tensors {
            let parts: Vec<&str> = t.name.splitn(3, '.').collect();
            if parts.len() >= 2 && parts[0] == "blk" {
                if let Ok(n) = parts[1].parse::<i64>() {
                    if n > max {
                        max = n;
                    }
                }
            }
        }
        if max >= 0 {
            max + 1
        } else {
            0
        }
    }
}

/// `fmt.Sscanf(s, "%d", &n)`: skip leading spaces, optional sign, digits.
pub fn scan_int_prefix(s: &str) -> Option<i64> {
    let t = s.trim_start();
    let b = t.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let ds = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == ds {
        return None;
    }
    t[..i].parse::<i64>().ok()
}

// ------------------------------------------------------------ ggml types

struct GgmlInfo {
    name: &'static str,
    block: u64,
    size: u64,
}

fn ggml_info(t: u32) -> Option<GgmlInfo> {
    let i = |name, block, size| Some(GgmlInfo { name, block, size });
    match t {
        0 => i("F32", 1, 4),
        1 => i("F16", 1, 2),
        2 => i("Q4_0", 32, 18),
        3 => i("Q4_1", 32, 20),
        6 => i("Q5_0", 32, 22),
        7 => i("Q5_1", 32, 24),
        8 => i("Q8_0", 32, 34),
        9 => i("Q8_1", 32, 36),
        10 => i("Q2_K", 256, 84),
        11 => i("Q3_K", 256, 110),
        12 => i("Q4_K", 256, 144),
        13 => i("Q5_K", 256, 176),
        14 => i("Q6_K", 256, 210),
        15 => i("Q8_K", 256, 292),
        16 => i("IQ2_XXS", 256, 66),
        17 => i("IQ2_XS", 256, 74),
        18 => i("IQ3_XXS", 256, 98),
        19 => i("IQ1_S", 256, 50),
        20 => i("IQ4_NL", 32, 18),
        21 => i("IQ3_S", 256, 110),
        22 => i("IQ2_S", 256, 82),
        23 => i("IQ4_XS", 256, 136),
        24 => i("I8", 1, 1),
        25 => i("I16", 1, 2),
        26 => i("I32", 1, 4),
        27 => i("I64", 1, 8),
        28 => i("F64", 1, 8),
        29 => i("IQ1_M", 256, 56),
        30 => i("BF16", 1, 2),
        34 => i("TQ1_0", 256, 54),
        35 => i("TQ2_0", 256, 66),
        39 => i("MXFP4", 32, 17),
        40 => i("NVFP4", 64, 36),
        41 => i("Q1_0", 128, 18),
        42 => i("Q2_0", 64, 18),
        _ => None,
    }
}

pub fn ggml_type_id(dtype: &str) -> Option<u32> {
    match dtype {
        "F32" => Some(0),
        "F16" => Some(1),
        "Q8_0" => Some(8),
        "Q4_K" => Some(12),
        "I32" => Some(26),
        "I64" => Some(27),
        "F64" => Some(28),
        "BF16" => Some(30),
        _ => None,
    }
}

fn tensor_data_size(n: u64, t: u32) -> Result<u64> {
    let info = ggml_info(t).ok_or_else(|| Error::msg(format!("unknown ggml type: {}", t)))?;
    if n > u64::MAX - info.block + 1 {
        return Err(Error::msg(format!("element count overflow: {}", n)));
    }
    let blocks = (n + info.block - 1) / info.block;
    if info.size > 0 && blocks > u64::MAX / info.size {
        return Err(Error::msg(format!("tensor data size overflow: {} blocks * {} bytes", blocks, info.size)));
    }
    Ok(blocks * info.size)
}

// ------------------------------------------------------------ format sniff

pub fn detect_format_bytes(b: &[u8], name: &str) -> Result<&'static str> {
    if b.len() >= 4 && &b[..4] == b"GGUF" {
        return Ok("gguf");
    }
    if b.len() >= 9 {
        let hl = le_u64(&b[..8]);
        if hl > 2 && hl <= 1 << 27 && b[8] == b'{' {
            return Ok("safetensors");
        }
    }
    Err(Error::msg(format!("{}: unrecognized model format (want GGUF magic or a SafeTensors JSON header)", name)))
}

// ------------------------------------------------------------ GGUF scan

struct Scan<'a> {
    b: &'a [u8],
    off: usize,
}

impl<'a> Scan<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.b.len() - self.off.min(self.b.len()) < n {
            return Err(Error::msg(format!("gguf header walk at offset {}: unexpected EOF", self.off)));
        }
        let s = &self.b[self.off..self.off + n];
        self.off += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(le_u32(self.take(4)?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(le_u64(self.take(8)?))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > 64 << 20 {
            return Err(Error::msg(format!("gguf header walk: string length {} implausible at offset {}", n, self.off)));
        }
        let raw = self.take(n as usize)?;
        Ok(String::from_utf8_lossy(raw).into_owned())
    }
    /// Returns a rendered scalar for the u32/u64/string kinds the layout
    /// reports; other kinds are skipped by size.
    fn skip_value(&mut self, vt: u32, depth: u32) -> Result<Option<String>> {
        Ok(match vt {
            0 | 1 | 7 => Some(self.take(1)?[0].to_string()),
            2 | 3 => Some(le_u16(self.take(2)?).to_string()),
            4 | 5 => Some(self.u32()?.to_string()),
            10 | 11 => Some(self.u64()?.to_string()),
            6 => {
                self.take(4)?;
                None
            }
            12 => {
                self.take(8)?;
                None
            }
            8 => Some(self.str()?),
            9 => {
                if depth > 8 {
                    return Err(Error::msg("gguf header walk: array nesting too deep"));
                }
                let elem = self.u32()?;
                let count = self.u64()?;
                if count > 1 << 22 {
                    return Err(Error::msg(format!("gguf header walk: array count {} implausible", count)));
                }
                for _ in 0..count {
                    self.skip_value(elem, depth + 1)?;
                }
                None
            }
            other => return Err(Error::msg(format!("gguf header walk: unsupported metadata value type {}", other))),
        })
    }
}

/// Parse only the header region of a GGUF buffer and compute the payload start.
pub fn scan_gguf_layout(b: &[u8]) -> Result<Layout> {
    let mut s = Scan { b, off: 0 };
    let magic = s.take(4)?;
    if magic != b"GGUF" {
        return Err(Error::msg(format!("gguf: bad magic {:?}", String::from_utf8_lossy(magic))));
    }
    let version = s.u32()?;
    if !(1..=3).contains(&version) {
        return Err(Error::msg(format!("gguf: unsupported version {}", version)));
    }
    let tensor_count = s.u64()?;
    let kv_count = s.u64()?;
    if tensor_count > 1 << 16 || kv_count > 1 << 16 {
        return Err(Error::msg(format!("gguf: implausible counts (tensors {}, kv {})", tensor_count, kv_count)));
    }
    let mut layout =
        Layout { format: "gguf".into(), path: String::new(), file_size: 0, data_offset: 0, metadata: BTreeMap::new(), tensors: Vec::new() };
    for _ in 0..kv_count {
        let key = s.str()?;
        let vt = s.u32()?;
        if let Some(val) = s.skip_value(vt, 0)? {
            if !val.is_empty() {
                layout.metadata.insert(key, val);
            }
        }
    }
    let mut alignment = DEFAULT_ALIGNMENT;
    if let Some(v) = layout.metadata.get("general.alignment") {
        if let Some(a) = scan_int_prefix(v) {
            if a > 0 && a <= 1 << 20 {
                alignment = a as u64;
            }
        }
    }
    let mut seen_names: BTreeMap<String, bool> = BTreeMap::new();
    for _ in 0..tensor_count {
        let name = s.str()?;
        if seen_names.insert(name.clone(), true).is_some() {
            return Err(Error::msg(format!("gguf: duplicate tensor name {:?}", name)));
        }
        let ndims = s.u32()?;
        if ndims > 8 {
            return Err(Error::msg(format!("gguf: tensor {:?} ndims {} out of range", name, ndims)));
        }
        let mut dims: Vec<usize> = Vec::with_capacity(ndims as usize);
        let mut elements: u64 = 1;
        for _ in 0..ndims {
            let n = s.u64()?;
            if n > i32::MAX as u64 {
                return Err(Error::msg(format!("gguf: tensor {:?} dim {} too large", name, n)));
            }
            dims.push(n as usize);
            elements = elements.checked_mul(n).ok_or_else(|| Error::msg(format!("gguf: tensor {:?} element count overflows u64", name)))?;
        }
        let typ = s.u32()?;
        let off = s.u64()?;
        let info = ggml_info(typ).ok_or_else(|| Error::msg(format!("gguf: tensor {:?}: unknown ggml type: {}", name, typ)))?;
        let size = tensor_data_size(elements, typ).map_err(|e| Error::msg(format!("gguf: tensor {:?}: {}", name, e)))?;
        let (rows, cols, lo, li) = match dims.len() {
            0 => (1, 1, 0, 0),
            1 => (dims[0], 1, 0, 0),
            2 => (dims[0], dims[1], dims[1], dims[0]),
            _ => (dims[0], dims[1], 0, 0),
        };
        layout.tensors.push(TensorLocation {
            name,
            dtype: info.name.to_string(),
            shape: dims,
            rows,
            cols,
            logical_out: lo,
            logical_in: li,
            offset: i64::try_from(off).map_err(|_| Error::msg(format!("gguf: tensor offset {} exceeds the signed 64-bit range", off)))?,
            size: i64::try_from(size).map_err(|_| Error::msg("gguf: tensor payload size exceeds the signed 64-bit range"))?,
            gguf_column_major: true,
            ggml_type: Some(typ),
        });
    }
    let header_end = s.off as i64;
    layout.data_offset = header_end;
    if !layout.tensors.is_empty() {
        let rem = header_end % alignment as i64;
        if rem != 0 {
            layout.data_offset = header_end + alignment as i64 - rem;
        }
    }
    for t in &mut layout.tensors {
        t.offset = t
            .offset
            .checked_add(layout.data_offset)
            .ok_or_else(|| Error::msg(format!("gguf: tensor {:?} payload offset overflows the signed 64-bit range", t.name)))?;
    }
    Ok(layout)
}

// ------------------------------------------------------------ SafeTensors scan

/// Fail on a repeated key in any object of `body`. serde_json::Map keeps the
/// last duplicate and Go's encoding/json does the same, but a header carrying
/// one tensor name twice is ambiguous: the two tools could bind different
/// entries and every verifier would accept both results (wiring audit H18).
/// The caller has already parsed `body`, which bounds nesting depth.
fn reject_duplicate_json_keys(body: &[u8]) -> std::result::Result<(), String> {
    use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
    struct Walk;
    impl<'de> DeserializeSeed<'de> for Walk {
        type Value = ();
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<(), D::Error> {
            d.deserialize_any(self)
        }
    }
    impl<'de> Visitor<'de> for Walk {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("JSON")
        }
        fn visit_bool<E>(self, _: bool) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_i64<E>(self, _: i64) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_u64<E>(self, _: u64) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_f64<E>(self, _: f64) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_str<E>(self, _: &str) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_unit<E>(self) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> std::result::Result<(), A::Error> {
            while a.next_element_seed(Walk)?.is_some() {}
            Ok(())
        }
        fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> std::result::Result<(), A::Error> {
            let mut seen = std::collections::BTreeSet::new();
            while let Some(k) = a.next_key::<String>()? {
                if !seen.insert(k.clone()) {
                    return Err(serde::de::Error::custom(format!("duplicate JSON key {:?}", k)));
                }
                a.next_value_seed(Walk)?;
            }
            Ok(())
        }
    }
    let mut de = serde_json::Deserializer::from_slice(body);
    Walk.deserialize(&mut de).map_err(|e| e.to_string())
}

pub fn scan_safetensors_layout(b: &[u8]) -> Result<Layout> {
    if b.len() < 8 {
        return Err(Error::msg("safetensors: reading header length: unexpected EOF"));
    }
    let header_len = le_u64(&b[..8]);
    if header_len < 2 || header_len > 1 << 27 {
        return Err(Error::msg(format!("safetensors: implausible header length {}", header_len)));
    }
    if (b.len() as u64) < 8 + header_len {
        return Err(Error::msg("safetensors: reading header body: unexpected EOF"));
    }
    let body = &b[8..8 + header_len as usize];
    let raw: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(body).map_err(|e| Error::msg(format!("safetensors: header is not JSON: {}", e)))?;
    reject_duplicate_json_keys(body).map_err(|e| Error::msg(format!("safetensors: {}", e)))?;
    let data_offset = 8 + header_len as i64;
    let mut layout = Layout {
        format: "safetensors".into(),
        path: String::new(),
        file_size: 0,
        data_offset,
        metadata: BTreeMap::new(),
        tensors: Vec::new(),
    };
    // serde_json::Map is a BTreeMap: iteration is sorted by name bytes, as sort.Strings.
    for (name, v) in &raw {
        if name == "__metadata__" {
            continue;
        }
        let obj = v.as_object().ok_or_else(|| Error::msg(format!("safetensors: tensor {:?} entry: not an object", name)))?;
        let dtype = obj.get("dtype").and_then(|d| d.as_str()).unwrap_or("").to_string();
        let shape: Vec<i64> = match obj.get("shape") {
            Some(serde_json::Value::Array(a)) => {
                let mut out = Vec::new();
                for x in a {
                    out.push(x.as_i64().ok_or_else(|| Error::msg(format!("safetensors: tensor {:?} entry: bad shape element", name)))?);
                }
                out
            }
            Some(serde_json::Value::Null) | None => Vec::new(),
            _ => return Err(Error::msg(format!("safetensors: tensor {:?} entry: shape is not an array", name))),
        };
        let offs: Vec<i64> = match obj.get("data_offsets") {
            Some(serde_json::Value::Array(a)) => {
                let mut out = Vec::new();
                for x in a {
                    out.push(
                        x.as_i64().ok_or_else(|| Error::msg(format!("safetensors: tensor {:?} entry: bad data_offsets element", name)))?,
                    );
                }
                out
            }
            _ => Vec::new(),
        };
        match dtype.as_str() {
            "F32" | "F16" | "BF16" | "F64" | "I32" | "I64" => {}
            _ => return Err(Error::msg(format!("safetensors: tensor {:?} has unsupported dtype {:?}", name, dtype))),
        }
        if offs.len() != 2 {
            return Err(Error::msg(format!("safetensors: tensor {:?} needs data_offsets [begin,end]", name)));
        }
        let (rows, cols, lo, li) = match shape.len() {
            0 => (1i64, 1i64, 0i64, 0i64),
            1 => (shape[0], 1, 0, 0),
            2 => (shape[0], shape[1], shape[0], shape[1]),
            _ => (shape[0], shape[1], 0, 0),
        };
        let (begin, end) = (offs[0], offs[1]);
        if begin < 0 || end < begin {
            return Err(Error::msg(format!("safetensors: tensor {:?} has invalid data_offsets [{},{})", name, begin, end)));
        }
        if shape.iter().any(|d| *d < 0) {
            return Err(Error::msg(format!("safetensors: tensor {:?} has a negative dimension in {:?}", name, shape)));
        }
        let mut elements: i64 = 1;
        for d in &shape {
            elements = elements
                .checked_mul(*d)
                .ok_or_else(|| Error::msg(format!("safetensors: tensor {:?} shape {:?} element count overflows i64", name, shape)))?;
        }
        let want = elements
            .checked_mul(dtype_size(&dtype))
            .ok_or_else(|| Error::msg(format!("safetensors: tensor {:?} byte size overflows i64", name)))?;
        if end - begin != want {
            return Err(Error::msg(format!(
                "safetensors: tensor {:?} spans {} bytes but shape {:?} dtype {} needs {}",
                name,
                end - begin,
                shape,
                dtype,
                want
            )));
        }
        layout.tensors.push(TensorLocation {
            name: name.clone(),
            dtype,
            shape: shape.iter().map(|d| *d as usize).collect(),
            rows: rows as usize,
            cols: cols as usize,
            logical_out: lo as usize,
            logical_in: li as usize,
            offset: data_offset
                .checked_add(begin)
                .ok_or_else(|| Error::msg(format!("safetensors: tensor {:?} data_offsets overflow", name)))?,
            size: end - begin,
            gguf_column_major: false,
            ggml_type: None,
        });
    }
    Ok(layout)
}

/// Parse the layout of an in-memory model image, validating payload extents.
pub fn layout_of_bytes(b: &[u8], path: &str) -> Result<Layout> {
    let format = detect_format_bytes(b, Path::new(path).file_name().and_then(|s| s.to_str()).unwrap_or(path))?;
    let mut layout = match format {
        "gguf" => scan_gguf_layout(b)?,
        _ => scan_safetensors_layout(b)?,
    };
    layout.path = path.to_string();
    layout.file_size = b.len() as i64;
    for t in &layout.tensors {
        let end = t.offset.checked_add(t.size).ok_or_else(|| Error::msg(format!("tensor {:?} payload extent overflows", t.name)))?;
        if t.offset < 0 || end > layout.file_size {
            return Err(Error::msg(format!("tensor {:?} payload [{},{}) exceeds file size {}", t.name, t.offset, end, layout.file_size)));
        }
    }
    Ok(layout)
}

/// A memory-mapped model file (read-only, zero-copy tensor access).
pub struct Model {
    pub layout: Layout,
    map: Mmap,
}

impl Model {
    pub fn open(path: &str) -> Result<Model> {
        let f = open_regular_file(path)?;
        let len = f.metadata()?.len();
        if len == 0 {
            return Err(Error::msg(format!("{}: empty file", path)));
        }
        // SAFETY: the mapping is read-only; callers that need immutability
        // across a long operation bracket it with sha256 before and after.
        let map = unsafe { Mmap::map(&f) }.map_err(|e| Error::msg(format!("mmap {}: {}", path, e)))?;
        let layout = layout_of_bytes(&map, path)?;
        Ok(Model { layout, map })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.map
    }

    pub fn tensor_bytes(&self, t: &TensorLocation) -> &[u8] {
        &self.map[t.offset as usize..(t.offset + t.size) as usize]
    }
}

pub fn open_regular_file(path: &str) -> Result<File> {
    let f = File::open(path).map_err(|e| Error::msg(format!("open {}: {}", path, e)))?;
    let md = f.metadata().map_err(|e| Error::msg(format!("stat {}: {}", path, e)))?;
    if md.is_dir() {
        return Err(Error::msg(format!("{} is a directory", path)));
    }
    Ok(f)
}

pub fn load_layout(path: &str) -> Result<Layout> {
    Ok(Model::open(path)?.layout)
}

pub fn sha256_hex(b: &[u8]) -> String {
    let d = Sha256::digest(b);
    hex(&d)
}

pub fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{:02x}", x));
    }
    s
}

pub fn file_sha256(path: &str) -> Result<(String, i64)> {
    let mut f = open_regular_file(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n: i64 = 0;
    loop {
        let r = f.read(&mut buf).map_err(|e| Error::msg(format!("hash {}: {}", path, e)))?;
        if r == 0 {
            break;
        }
        h.update(&buf[..r]);
        n += r as i64;
    }
    Ok((hex(&h.finalize()), n))
}

// ------------------------------------------------------------ Qwen3.5 ssm_out

pub const QWEN35_DEFAULT_NUM_V_HEADS: usize = 32;
pub const QWEN35_DEFAULT_HEAD_V_DIM: usize = 128;
pub const QWEN35_DEFAULT_NUM_K_HEADS: usize = 16;
pub const QWEN35_DEFAULT_HEAD_K_DIM: usize = 128;

const META_GROUP_COUNT: &str = ".ssm.group_count";
const META_TIME_STEP_RANK: &str = ".ssm.time_step_rank";
const META_INNER_SIZE: &str = ".ssm.inner_size";
const META_STATE_SIZE: &str = ".ssm.state_size";

/// GDN out-projection geometry derived from canonical GGUF SSM metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SsmOutLayout {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_v_dim: usize,
    pub head_k_dim: usize,
    pub metadata_derived: bool,
}

impl SsmOutLayout {
    pub fn in_width(&self) -> usize {
        self.num_v_heads * self.head_v_dim
    }
    /// r = num_v_heads / num_k_heads (value heads per key head). `validate`
    /// must pass first.
    pub fn ratio(&self) -> usize {
        self.num_v_heads / self.num_k_heads
    }
    pub fn tile(&self) -> usize {
        self.ratio() * self.head_v_dim
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_k_heads == 0 {
            return Err(Error::msg(format!("qwen3.5 ssm_out: num_k_heads {} must be positive", self.num_k_heads)));
        }
        if self.num_v_heads == 0 {
            return Err(Error::msg(format!("qwen3.5 ssm_out: num_v_heads {} must be positive", self.num_v_heads)));
        }
        if self.head_v_dim == 0 {
            return Err(Error::msg(format!("qwen3.5 ssm_out: head_v_dim {} must be positive", self.head_v_dim)));
        }
        if self.num_v_heads % self.num_k_heads != 0 {
            return Err(Error::msg(format!(
                "qwen3.5 ssm_out: num_v_heads {} is not a multiple of num_k_heads {}, so the ratio r = num_v_heads/num_k_heads is not an integer",
                self.num_v_heads, self.num_k_heads
            )));
        }
        let r = self.ratio();
        if r < 1 {
            return Err(Error::msg(format!(
                "qwen3.5 ssm_out: ratio r = num_v_heads {} / num_k_heads {} = {} must be >= 1 for the value-head interleave",
                self.num_v_heads, self.num_k_heads, r
            )));
        }
        if self.num_v_heads > (isize::MAX as usize) / self.head_v_dim {
            return Err(Error::msg(format!(
                "qwen3.5 ssm_out: num_v_heads {} * head_v_dim {} overflows int",
                self.num_v_heads, self.head_v_dim
            )));
        }
        Ok(())
    }

    /// Derive from the canonical `{arch}.ssm.{group_count,time_step_rank,
    /// inner_size,state_size}` keys. All-absent gives the documented legacy
    /// geometry with `metadata_derived = false`; a partial or invalid
    /// canonical block fails closed.
    pub fn from_meta(meta: &BTreeMap<String, String>) -> Result<SsmOutLayout> {
        let arch = meta.get("general.architecture").map(|s| s.trim()).unwrap_or("");
        let g = meta.get(&format!("{}{}", arch, META_GROUP_COUNT));
        let t = meta.get(&format!("{}{}", arch, META_TIME_STEP_RANK));
        let i = meta.get(&format!("{}{}", arch, META_INNER_SIZE));
        let s = meta.get(&format!("{}{}", arch, META_STATE_SIZE));
        if g.is_none() && t.is_none() && i.is_none() && s.is_none() {
            return Ok(SsmOutLayout {
                num_k_heads: QWEN35_DEFAULT_NUM_K_HEADS,
                num_v_heads: QWEN35_DEFAULT_NUM_V_HEADS,
                head_v_dim: QWEN35_DEFAULT_HEAD_V_DIM,
                head_k_dim: QWEN35_DEFAULT_HEAD_K_DIM,
                metadata_derived: false,
            });
        }
        let key = |suffix: &str| format!("{}{}", arch, suffix);
        let require = |v: &Option<&String>, suffix: &str| -> Result<String> {
            match v {
                Some(x) => Ok((*x).clone()),
                None => Err(Error::msg(format!(
                    "qwen3.5 ssm_out: canonical SSM metadata block is partial ({} missing); refusing to mix canonical keys with legacy defaults",
                    key(suffix)
                ))),
            }
        };
        let g = require(&g, META_GROUP_COUNT)?;
        let t = require(&t, META_TIME_STEP_RANK)?;
        let i = require(&i, META_INNER_SIZE)?;
        let s = require(&s, META_STATE_SIZE)?;
        let parse = |suffix: &str, raw: &str| -> Result<usize> {
            let n: i64 = raw
                .trim()
                .parse()
                .map_err(|e| Error::msg(format!("qwen3.5 ssm_out: {} = {:?} is not a valid integer: {}", key(suffix), raw, e)))?;
            if n <= 0 {
                return Err(Error::msg(format!("qwen3.5 ssm_out: {} = {} must be positive", key(suffix), n)));
            }
            Ok(n as usize)
        };
        let num_k = parse(META_GROUP_COUNT, &g)?;
        let num_v = parse(META_TIME_STEP_RANK, &t)?;
        let inner = parse(META_INNER_SIZE, &i)?;
        let head_k = parse(META_STATE_SIZE, &s)?;
        if inner % num_v != 0 {
            return Err(Error::msg(format!(
                "qwen3.5 ssm_out: {} = {} is not divisible by {} = {}, so head_v_dim cannot be derived exactly",
                key(META_INNER_SIZE),
                inner,
                key(META_TIME_STEP_RANK),
                num_v
            )));
        }
        let lay =
            SsmOutLayout { num_k_heads: num_k, num_v_heads: num_v, head_v_dim: inner / num_v, head_k_dim: head_k, metadata_derived: true };
        lay.validate()?;
        Ok(lay)
    }

    /// HF value-head block `h = k*r + j` sits at GGUF block `j*num_k + k`.
    pub fn gguf_block_of_hf(&self, hf_block: usize) -> Result<usize> {
        self.validate()?;
        if hf_block >= self.num_v_heads {
            return Err(Error::msg(format!("qwen3.5 ssm_out: HF block index {} out of [0,{})", hf_block, self.num_v_heads)));
        }
        let r = self.ratio();
        Ok((hf_block % r) * self.num_k_heads + hf_block / r)
    }

    pub fn hf_block_of_gguf(&self, gguf_block: usize) -> Result<usize> {
        self.validate()?;
        if gguf_block >= self.num_v_heads {
            return Err(Error::msg(format!("qwen3.5 ssm_out: GGUF block index {} out of [0,{})", gguf_block, self.num_v_heads)));
        }
        Ok((gguf_block % self.num_k_heads) * self.ratio() + gguf_block / self.num_k_heads)
    }

    /// GGUF block order expressed as the HF block indices they hold.
    pub fn interleave_order(&self) -> Result<Vec<usize>> {
        self.validate()?;
        (0..self.num_v_heads).map(|j| self.hf_block_of_gguf(j)).collect()
    }

    pub fn description(&self) -> String {
        match self.interleave_order() {
            Err(e) => format!("qwen3.5 ssm_out: {}", e),
            Ok(order) => {
                let head: Vec<String> = order.iter().take(8).map(|b| b.to_string()).collect();
                format!(
                    "GGUF blk.N.ssm_out.weight stores linear_attn.out_proj.weight values with the LOGICAL INPUT axis (value_dim = GGUF ne[0], {} columns = {} value-head blocks x head_v_dim {}) permuted: with r = {} value heads per key head ({} key heads), GGUF block j holds HF block {} ...; HF block h = k*r + j lands at GGUF block j*num_k_heads + k; edit plans must map input columns accordingly",
                    self.in_width(), self.num_v_heads, self.head_v_dim, self.ratio(), self.num_k_heads, head.join(",")
                )
            }
        }
    }
}

/// The `ssm_out` input-axis column permutation for a tensor with `in_width`
/// logical input columns. `perm[gguf_col] = hf_col`.
///
/// * [`forward_in_col`] maps an HF-order input column to the GGUF column that
///   stores it (what an HF-order plan must apply before touching GGUF bytes).
/// * [`inverse_in_col`] maps a GGUF input column back to its HF column.
#[derive(Debug, Clone)]
pub struct SsmOutPermutation {
    perm: Vec<usize>,
    inv: Vec<usize>,
}

impl SsmOutPermutation {
    pub fn new(in_width: usize, lay: &SsmOutLayout) -> Result<Self> {
        lay.validate()?;
        if in_width == 0 {
            return Err(Error::msg(format!("qwen3.5 ssm_out: input width {} is not positive", in_width)));
        }
        if in_width != lay.in_width() {
            return Err(Error::msg(format!(
                "qwen3.5 ssm_out: logical input width {} != num_v_heads {} * head_v_dim {} = {}",
                in_width,
                lay.num_v_heads,
                lay.head_v_dim,
                lay.in_width()
            )));
        }
        let mut perm = vec![0usize; in_width];
        for j in 0..lay.num_v_heads {
            let hf = lay.hf_block_of_gguf(j)?;
            for w in 0..lay.head_v_dim {
                perm[j * lay.head_v_dim + w] = hf * lay.head_v_dim + w;
            }
        }
        let mut inv = vec![0usize; in_width];
        for (i, p) in perm.iter().enumerate() {
            inv[*p] = i;
        }
        Ok(SsmOutPermutation { perm, inv })
    }

    pub fn len(&self) -> usize {
        self.perm.len()
    }
    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }
    /// `perm[gguf_col] = hf_col`.
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }
    pub fn forward_in_col(&self, hf_col: usize) -> Result<usize> {
        self.inv
            .get(hf_col)
            .copied()
            .ok_or_else(|| Error::msg(format!("qwen3.5 ssm_out: HF input column {} out of [0,{})", hf_col, self.perm.len())))
    }
    pub fn inverse_in_col(&self, gguf_col: usize) -> Result<usize> {
        self.perm
            .get(gguf_col)
            .copied()
            .ok_or_else(|| Error::msg(format!("qwen3.5 ssm_out: GGUF input column {} out of [0,{})", gguf_col, self.perm.len())))
    }
}

pub fn is_qwen35_ssm_out_tensor(name: &str) -> bool {
    let p: Vec<&str> = name.split('.').collect();
    p.len() == 4 && p[0] == "blk" && p[2] == "ssm_out" && p[3] == "weight" && p[1].parse::<i64>().is_ok()
}

// ------------------------------------------------------------ inspect

#[derive(Debug, Clone, Serialize)]
pub struct TensorSummary {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub elements: i64,
    pub byte_size: i64,
    pub file_offset: i64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SsmDetection {
    pub detected: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub pattern: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<i64>,
    #[serde(skip_serializing_if = "is_zero")]
    pub out_width: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub in_width: usize,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub dtype: String,
    #[serde(skip_serializing_if = "is_zero")]
    pub num_k_heads: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub num_v_heads: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub head_v_dim: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub head_k_dim: usize,
    pub metadata_derived: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub interleave_head: Vec<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub gguf_block_of_hf_head: Vec<usize>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectReport {
    pub schema: String,
    pub path: String,
    pub format: String,
    pub file_size: i64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub architecture: String,
    #[serde(skip_serializing_if = "is_zero_i64")]
    pub block_count: i64,
    #[serde(skip_serializing_if = "is_zero_i64")]
    pub hidden_size: i64,
    #[serde(skip_serializing_if = "is_zero_i64")]
    pub context_length: i64,
    pub data_offset: i64,
    pub tensor_count: usize,
    pub dtype_counts: BTreeMap<String, usize>,
    pub tensors: Vec<TensorSummary>,
    pub qwen35_gdn_ssm_out: Option<SsmDetection>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

fn is_zero_i64(v: &i64) -> bool {
    *v == 0
}

pub const INSPECT_SCHEMA: &str = "secant.wsp.inspect.v1";

pub fn inspect_model(path: &str) -> Result<InspectReport> {
    let layout = load_layout(path)?;
    let arch = layout.architecture().to_string();
    let mut rep = InspectReport {
        schema: INSPECT_SCHEMA.into(),
        path: path.into(),
        format: layout.format.clone(),
        file_size: layout.file_size,
        architecture: arch.clone(),
        block_count: layout.block_count(),
        hidden_size: layout.meta_int(&format!("{}.embedding_length", arch)).unwrap_or(0),
        context_length: layout.meta_int(&format!("{}.context_length", arch)).unwrap_or(0),
        data_offset: layout.data_offset,
        tensor_count: layout.tensors.len(),
        dtype_counts: BTreeMap::new(),
        tensors: Vec::new(),
        qwen35_gdn_ssm_out: None,
        notes: Vec::new(),
    };
    let mut ssm = SsmDetection { pattern: "blk.N.ssm_out.weight".into(), ..Default::default() };
    let mut mismatches: Vec<String> = Vec::new();
    for t in &layout.tensors {
        *rep.dtype_counts.entry(t.dtype.clone()).or_insert(0) += 1;
        let shape = if t.shape.is_empty() { vec![t.rows, t.cols] } else { t.shape.clone() };
        rep.tensors.push(TensorSummary {
            name: t.name.clone(),
            dtype: t.dtype.clone(),
            shape,
            elements: t.elements(),
            byte_size: t.size,
            file_offset: t.offset,
        });
        if layout.format == "gguf" && is_qwen35_ssm_out_tensor(&t.name) {
            if let Some(l) = t.name.split('.').nth(1).and_then(|x| x.parse::<i64>().ok()) {
                ssm.layers.push(l);
            }
            if ssm.dtype.is_empty() {
                ssm.out_width = t.logical_out;
                ssm.in_width = t.logical_in;
                ssm.dtype = t.dtype.clone();
            } else if ssm.out_width != t.logical_out || ssm.in_width != t.logical_in || ssm.dtype != t.dtype {
                mismatches.push(format!(
                    "{} is logical [{} x {}] {} but the family reference is [{} x {}] {}",
                    t.name, t.logical_out, t.logical_in, t.dtype, ssm.out_width, ssm.in_width, ssm.dtype
                ));
            }
        }
    }
    ssm.layers.sort();
    let lay = SsmOutLayout::from_meta(&layout.metadata).map_err(|e| Error::msg(format!("qwen3.5 ssm_out metadata: {}", e)))?;
    ssm.num_k_heads = lay.num_k_heads;
    ssm.num_v_heads = lay.num_v_heads;
    ssm.head_v_dim = lay.head_v_dim;
    ssm.head_k_dim = lay.head_k_dim;
    ssm.metadata_derived = lay.metadata_derived;
    let consistent = mismatches.is_empty();
    ssm.detected = !ssm.layers.is_empty() && consistent && ssm.in_width == lay.in_width();
    if ssm.detected {
        if let Ok(order) = lay.interleave_order() {
            ssm.interleave_head = order[..order.len().min(8)].to_vec();
            let mut g = vec![0usize; lay.num_v_heads];
            for (i, slot) in g.iter_mut().enumerate() {
                match lay.gguf_block_of_hf(i) {
                    Ok(x) => *slot = x,
                    Err(_) => break,
                }
            }
            ssm.gguf_block_of_hf_head = g[..g.len().min(8)].to_vec();
        }
        ssm.description = lay.description();
        let origin = if ssm.metadata_derived {
            format!("canonical {}.ssm.group_count/time_step_rank/inner_size/state_size metadata", layout.architecture())
        } else {
            "the documented legacy defaults (no canonical ssm metadata keys in this header)".to_string()
        };
        rep.notes.push(format!(
            "blk.N.ssm_out.weight detected with r = {} ({} value heads / {} key heads), geometry from {}: HF-order edits must map logical input columns through the {}x{} value-head interleave before GGUF surgery (TG-WSP-004)",
            lay.ratio(), lay.num_v_heads, lay.num_k_heads, origin, lay.num_v_heads, lay.head_v_dim
        ));
    } else if !ssm.layers.is_empty() {
        if !consistent {
            rep.notes.push(format!(
                "blk.N.ssm_out.weight family is inconsistent across layers: {}; interleave mapping disabled",
                mismatches.join("; ")
            ));
        }
        if ssm.in_width != lay.in_width() {
            rep.notes.push(format!(
                "blk.N.ssm_out.weight present but logical input width {} != num_v_heads {} * head_v_dim {} = {} (metadata {:?}); interleave mapping disabled",
                ssm.in_width, lay.num_v_heads, lay.head_v_dim, lay.in_width(), lay
            ));
        }
    }
    rep.qwen35_gdn_ssm_out = Some(ssm);
    Ok(rep)
}

// ------------------------------------------------------------ patch

pub const PATCH_SCHEMA: &str = "secant.wsp.patch.v1";
pub const VERIFY_SCHEMA: &str = "secant.wsp.verify.v1";

#[derive(Debug, Clone, Serialize, Default)]
pub struct TensorPatchReport {
    pub id: String,
    pub tensor: String,
    pub op: String,
    pub offset: i64,
    pub entries_total: i64,
    pub entries_changed: i64,
    pub delta_fro_norm: f64,
    #[serde(rename = "new_sha256")]
    pub tensor_sha256_new: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct VerifyReport {
    pub schema: String,
    pub format: String,
    pub base: String,
    pub patched: String,
    #[serde(rename = "header_bytes_identical")]
    pub header_identical: bool,
    pub header_len: i64,
    pub data_offset_base: i64,
    pub data_offset_patched: i64,
    pub n_tensors: usize,
    #[serde(rename = "n_other_tensors")]
    pub n_tensors_other: i64,
    pub other_tensors_byte_identical: bool,
    pub changed_tensors: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unexpected_changed: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub missing_expected: Vec<String>,
    pub ok: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub failures: Vec<String>,
}

impl VerifyReport {
    pub fn failure_summary(&self) -> String {
        self.failures.join("; ")
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct PatchReport {
    pub schema: String,
    pub base: String,
    pub out: String,
    pub format: String,
    pub base_sha256: String,
    pub out_sha256: String,
    pub out_bytes: i64,
    pub edits: Option<Vec<TensorPatchReport>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verify: Option<VerifyReport>,
}

/// Compare `[offset, offset+size)` of two buffers.
fn ranges_identical(a: &[u8], b: &[u8], offset: i64, size: i64) -> Result<bool> {
    if size < 0 {
        return Err(Error::msg(format!("negative size {}", size)));
    }
    let (o, e) = (offset as usize, (offset + size) as usize);
    if e > a.len() || e > b.len() {
        // The oracle's ReadAt tolerates EOF and compares the short reads.
        let (ra, rb) = (&a[o.min(a.len())..e.min(a.len())], &b[o.min(b.len())..e.min(b.len())]);
        return Ok(ra == rb);
    }
    Ok(a[o..e] == b[o..e])
}

/// Byte-level identity check between a base and a patched model. `expect`
/// names the tensors allowed (and required) to differ; the header region
/// `[0, data_offset)` and every other tensor must be 100% byte-identical.
pub fn verify_patch(base_path: &str, patched_path: &str, expect: &[String]) -> Result<VerifyReport> {
    let base = Model::open(base_path).map_err(|e| Error::msg(format!("base: {}", e)))?;
    let patched = Model::open(patched_path).map_err(|e| Error::msg(format!("patched: {}", e)))?;
    let (bl, pl) = (&base.layout, &patched.layout);
    if bl.format != pl.format {
        return Err(Error::msg(format!(
            "cannot compare {} ({}) with {} ({}): different container formats",
            base_path, bl.format, patched_path, pl.format
        )));
    }
    let mut rep = VerifyReport {
        schema: VERIFY_SCHEMA.into(),
        format: bl.format.clone(),
        base: base_path.into(),
        patched: patched_path.into(),
        data_offset_base: bl.data_offset,
        data_offset_patched: pl.data_offset,
        n_tensors: bl.tensors.len(),
        ..Default::default()
    };
    rep.header_len = bl.data_offset;
    rep.header_identical = bl.data_offset == pl.data_offset;
    if rep.header_identical {
        rep.header_identical = ranges_identical(base.bytes(), patched.bytes(), 0, bl.data_offset)?;
    }
    if !rep.header_identical {
        rep.failures.push(format!(
            "header bytes differ (base data_offset={}, patched data_offset={}); the patch must not rewrite the header",
            bl.data_offset, pl.data_offset
        ));
    }
    let expected: std::collections::HashSet<&str> = expect.iter().map(|s| s.as_str()).collect();
    if bl.tensors.len() != pl.tensors.len() {
        rep.failures.push(format!("tensor counts differ: base {}, patched {}", bl.tensors.len(), pl.tensors.len()));
        return Ok(rep);
    }
    let mut other_identical = true;
    let mut changed: Vec<String> = Vec::new();
    for (bt, pt) in bl.tensors.iter().zip(pl.tensors.iter()) {
        if bt.name != pt.name || bt.offset != pt.offset || bt.size != pt.size {
            other_identical = false;
            rep.failures.push(format!(
                "tensor table entry differs: base {}@[{},{}) vs patched {}@[{},{})",
                bt.name,
                bt.offset,
                bt.offset + bt.size,
                pt.name,
                pt.offset,
                pt.offset + pt.size
            ));
            continue;
        }
        if !ranges_identical(base.bytes(), patched.bytes(), bt.offset, bt.size)? {
            changed.push(bt.name.clone());
            if !expected.contains(bt.name.as_str()) {
                rep.unexpected_changed.push(bt.name.clone());
                other_identical = false;
            }
        }
    }
    for want in expect {
        if !changed.iter().any(|c| c == want) {
            rep.missing_expected.push(want.clone());
        }
    }
    rep.n_tensors_other = rep.n_tensors as i64 - changed.len() as i64;
    rep.other_tensors_byte_identical = other_identical;
    if !other_identical {
        rep.failures.push("non-target tensor bytes differ".into());
    }
    if !rep.unexpected_changed.is_empty() {
        rep.failures.push(format!("unexpected changed tensors: {}", crate::numeric::go_fmt_slice(&rep.unexpected_changed)));
    }
    if !rep.missing_expected.is_empty() {
        rep.failures.push(format!("expected-changed tensors that did not change: {}", crate::numeric::go_fmt_slice(&rep.missing_expected)));
    }
    rep.ok = rep.header_identical && other_identical && rep.unexpected_changed.is_empty() && rep.missing_expected.is_empty();
    rep.changed_tensors = if changed.is_empty() { None } else { Some(changed) };
    Ok(rep)
}

/// Per-tensor SHA-256 snapshot of a model, used to verify an IN-PLACE patch
/// where no pristine second file exists to byte-compare against.
#[derive(Debug, Clone)]
pub struct TensorDigests {
    pub header_sha256: String,
    pub tensors: Vec<(String, String)>,
}

impl TensorDigests {
    pub fn of(path: &str) -> Result<TensorDigests> {
        let m = Model::open(path)?;
        Ok(TensorDigests {
            header_sha256: sha256_hex(&m.bytes()[..m.layout.data_offset as usize]),
            tensors: m.layout.tensors.iter().map(|t| (t.name.clone(), sha256_hex(m.tensor_bytes(t)))).collect(),
        })
    }

    /// SHA-256 non-target verifier: header and every tensor not in `expect`
    /// must equal the snapshot; every tensor in `expect` must differ.
    pub fn verify_after(&self, path: &str, expect: &[String]) -> Result<VerifyReport> {
        let now = TensorDigests::of(path)?;
        let mut rep = VerifyReport {
            schema: VERIFY_SCHEMA.into(),
            format: Model::open(path)?.layout.format,
            base: "(in-place snapshot)".into(),
            patched: path.into(),
            n_tensors: self.tensors.len(),
            ..Default::default()
        };
        rep.header_identical = self.header_sha256 == now.header_sha256;
        if !rep.header_identical {
            rep.failures.push("header bytes differ from the pre-patch snapshot".into());
        }
        if self.tensors.len() != now.tensors.len() {
            rep.failures.push(format!("tensor counts differ: snapshot {}, now {}", self.tensors.len(), now.tensors.len()));
            return Ok(rep);
        }
        let expected: std::collections::HashSet<&str> = expect.iter().map(|s| s.as_str()).collect();
        let mut changed = Vec::new();
        let mut other_identical = true;
        for ((n0, h0), (n1, h1)) in self.tensors.iter().zip(now.tensors.iter()) {
            if n0 != n1 {
                other_identical = false;
                rep.failures.push(format!("tensor table entry differs: {} vs {}", n0, n1));
                continue;
            }
            if h0 != h1 {
                changed.push(n0.clone());
                if !expected.contains(n0.as_str()) {
                    rep.unexpected_changed.push(n0.clone());
                    other_identical = false;
                }
            }
        }
        for want in expect {
            if !changed.iter().any(|c| c == want) {
                rep.missing_expected.push(want.clone());
            }
        }
        rep.n_tensors_other = rep.n_tensors as i64 - changed.len() as i64;
        rep.other_tensors_byte_identical = other_identical;
        rep.ok = rep.header_identical && other_identical && rep.unexpected_changed.is_empty() && rep.missing_expected.is_empty();
        if !rep.ok {
            rep.failures.push("in-place verification failed".into());
        }
        rep.changed_tensors = if changed.is_empty() { None } else { Some(changed) };
        Ok(rep)
    }
}

// ---- patch application

fn apply_set_value(buf: &mut TensorBuffer, t: &TensorLocation, edit: &Edit, notes: &mut Vec<String>) -> Result<()> {
    let n_lists = [edit.semantics.len(), edit.set_bits.len(), edit.set_ints.len()].iter().filter(|l| **l > 0).count();
    let set_one = |buf: &mut TensorBuffer, idx: i64, val_idx: usize| -> Result<()> {
        if !edit.semantics.is_empty() {
            return buf.write_semantic(idx, &edit.semantics[val_idx % edit.semantics.len()]);
        }
        if !edit.set_bits.is_empty() {
            buf.set_bits(idx, edit.set_bits[val_idx % edit.set_bits.len()]);
            return Ok(());
        }
        if !edit.set_ints.is_empty() {
            return buf.set_int(idx, edit.set_ints[val_idx % edit.set_ints.len()]);
        }
        Ok(())
    };
    let elems = t.elements();
    if !edit.token_rows.is_empty() {
        if !t.is_2d() {
            return Err(Error::msg(format!(
                "token-row set-value requires an exactly 2-D tensor: {} has {} storage dim(s)",
                t.name,
                t.shape.len()
            )));
        }
        for (vi, tr) in edit.token_rows.iter().enumerate() {
            if *tr < 0 || *tr >= t.logical_out as i64 {
                return Err(Error::msg(format!("token row {} out of [0,{})", tr, t.logical_out)));
            }
            for inn in 0..t.logical_in {
                set_one(buf, t.weight_unchecked(*tr as usize, inn), vi)?;
            }
        }
        notes.push("whole-token rows programmed with dtype-exact values".into());
    } else if !edit.set_indices.is_empty() {
        for (i, idx) in edit.set_indices.iter().enumerate() {
            if *idx < 0 || *idx >= elems {
                return Err(Error::msg(format!("flat index {} out of [0,{})", idx, elems)));
            }
            set_one(buf, *idx, i)?;
        }
    } else {
        for i in 0..elems {
            set_one(buf, i, 0)?;
        }
        notes.push("whole tensor set to a single exact value".into());
    }
    if n_lists == 0 {
        return Err(Error::msg("set-value carries neither semantics nor bits nor ints"));
    }
    if n_lists > 1 {
        return Err(Error::msg("set-value carries more than one of semantics/bits/ints; pick one"));
    }
    if !edit.set_ints.is_empty() && t.dtype != "I32" && t.dtype != "I64" {
        return Err(Error::msg(format!("ints given but tensor dtype is {}", t.dtype)));
    }
    Ok(())
}

fn apply_permute_columns(buf: &mut TensorBuffer, t: &TensorLocation, layout: &Layout, permutation: &str) -> Result<()> {
    if !t.is_2d() {
        return Err(Error::msg(format!("permute-columns needs an exactly 2-D tensor, got storage dims {:?}", t.shape)));
    }
    let lay = SsmOutLayout::from_meta(&layout.metadata).map_err(|e| Error::msg(format!("qwen3.5 ssm_out layout from metadata: {}", e)))?;
    let p = SsmOutPermutation::new(t.logical_in, &lay)?;
    let perm: Vec<usize> = match permutation {
        "qwen35-ssm-out-128-hf-to-gguf" => p.perm().to_vec(),
        "qwen35-ssm-out-128-gguf-to-hf" => (0..p.len()).map(|i| p.forward_in_col(i).unwrap()).collect(),
        other => return Err(Error::msg(format!("unknown permutation {:?}", other))),
    };
    let width = dtype_size(&t.dtype) as usize;
    let mut out = vec![0u8; t.size as usize];
    for (gguf_in, hf_in) in perm.iter().enumerate() {
        for o in 0..t.logical_out {
            let src = t.weight_unchecked(o, *hf_in) as usize * width;
            let dst = t.weight_unchecked(o, gguf_in) as usize * width;
            out[dst..dst + width].copy_from_slice(&buf.data[src..src + width]);
        }
    }
    buf.data.copy_from_slice(&out);
    Ok(())
}

fn apply_edit(map: &mut MmapMut, layout: &Layout, edit: &Edit) -> Result<TensorPatchReport> {
    let name = edit.target_tensor_name(&layout.format).to_string();
    if name.is_empty() {
        return Err(Error::msg(format!("no tensor name for format {}", layout.format)));
    }
    let t = layout.tensor(&name)?.clone();
    if t.is_quantized() {
        return Err(Error::msg(format!(
            "tensor {} is {}: float weight surgery supports F32/F16/BF16/F64/I32/I64 payloads only (use `secant quant` for block-aligned Q8_0 surgery)",
            name, t.dtype
        )));
    }
    let (lo, hi) = (t.offset as usize, (t.offset + t.size) as usize);
    let mut buf = TensorBuffer::new(&t.dtype, map[lo..hi].to_vec());
    let orig = buf.data.clone();
    let mut report =
        TensorPatchReport { id: edit.id.clone(), tensor: name.clone(), op: edit.op.clone(), offset: t.offset, ..Default::default() };
    let elements = t.size / dtype_size(&t.dtype);
    report.entries_total = elements;

    match edit.op.as_str() {
        "rank1-columns" => {
            t.require_2d(&edit.op)?;
            if edit.direction.len() != t.logical_out {
                return Err(Error::msg(format!(
                    "direction dim {} != tensor logical d_out {} (output-subspace edits act on the output axis of the logical [d_out x d_in] matrix)",
                    edit.direction.len(),
                    t.logical_out
                )));
            }
            let mut cols: Vec<i64> = edit.cols.clone();
            if edit.layout == "qwen35-ssm-out-128" {
                if !is_qwen35_ssm_out_tensor(&name) {
                    return Err(Error::msg(format!("layout qwen35-ssm-out-128 applies to blk.N.ssm_out.weight, not {}", name)));
                }
                if layout.format != "gguf" {
                    return Err(Error::msg(format!(
                        "layout qwen35-ssm-out-128 maps HF columns onto the GGUF input axis; {} ({}) is already in HF order",
                        name, layout.format
                    )));
                }
                let lay = SsmOutLayout::from_meta(&layout.metadata)
                    .map_err(|e| Error::msg(format!("qwen3.5 ssm_out layout from metadata: {}", e)))?;
                let p = SsmOutPermutation::new(t.logical_in, &lay)
                    .map_err(|e| Error::msg(format!("qwen35 ssm_out column mapping (d_in={}, {:?}): {}", t.logical_in, lay, e)))?;
                let mut mapped = Vec::with_capacity(cols.len());
                for c in &cols {
                    if *c < 0 || *c as usize >= t.logical_in {
                        return Err(Error::msg(format!(
                            "qwen35 ssm_out column mapping: qwen3.5 ssm_out: HF input column {} out of [0,{})",
                            c, t.logical_in
                        )));
                    }
                    mapped.push(p.forward_in_col(*c as usize)? as i64);
                }
                cols = mapped;
                report.notes.push(format!(
                    "cols given in HF linear_attn.out_proj order; mapped through the Qwen3.5 value-head interleave on the logical input axis ({} blocks x {}, TG-WSP-004)",
                    lay.num_v_heads, lay.head_v_dim
                ));
            }
            if cols.len() != edit.coeffs.len() {
                return Err(Error::msg(format!("cols ({}) and coeffs ({}) misaligned", cols.len(), edit.coeffs.len())));
            }
            for (i, c) in cols.iter().enumerate() {
                if *c < 0 || *c as usize >= t.logical_in {
                    return Err(Error::msg(format!("logical input column {} out of [0,{})", c, t.logical_in)));
                }
                let a = edit.coeffs[i];
                for o in 0..t.logical_out {
                    let idx = t.weight_unchecked(o, *c as usize);
                    let old = buf.get(idx);
                    buf.set_float(idx, fma_add(old, a, edit.direction[o]));
                }
            }
        }
        "projector-subtract" => {
            t.require_2d(&edit.op)?;
            if edit.direction.len() != t.logical_out {
                return Err(Error::msg(format!(
                    "direction dim {} != tensor logical d_out {} (the refusal/ref-space direction lives in the output space)",
                    edit.direction.len(),
                    t.logical_out
                )));
            }
            let mut g = vec![0f64; t.logical_in];
            for (inn, gi) in g.iter_mut().enumerate() {
                let mut s = 0f64;
                for o in 0..t.logical_out {
                    s = fma_add(s, edit.direction[o], buf.get(t.weight_unchecked(o, inn)));
                }
                *gi = s;
            }
            for inn in 0..t.logical_in {
                for o in 0..t.logical_out {
                    let idx = t.weight_unchecked(o, inn);
                    let coef = -edit.gamma * edit.direction[o];
                    buf.set_float(idx, fma_add(buf.get(idx), coef, g[inn]));
                }
            }
        }
        "set-value" => apply_set_value(&mut buf, &t, edit, &mut report.notes)?,
        "permute-columns" => {
            t.require_2d(&edit.op)?;
            apply_permute_columns(&mut buf, &t, layout, &edit.permutation)?;
            report.notes.push(format!("permutation {}", edit.permutation));
        }
        other => return Err(Error::msg(format!("unknown op {:?}", other))),
    }

    let w = dtype_size(&t.dtype) as usize;
    let mut fro_sq = 0f64;
    let mut saw_non_finite = false;
    for i in 0..elements as usize {
        let off = i * w;
        if orig[off..off + w] == buf.data[off..off + w] {
            continue;
        }
        report.entries_changed += 1;
        let old = read_elem(&orig, &t.dtype, off);
        let newv = read_elem(&buf.data, &t.dtype, off);
        if old.is_nan() || newv.is_nan() || old.is_infinite() || newv.is_infinite() {
            saw_non_finite = true;
            continue;
        }
        let d = newv - old;
        fro_sq = fma_add(fro_sq, d, d);
    }
    report.delta_fro_norm = fro_sq.sqrt();
    if saw_non_finite {
        report.notes.push("delta_fro_norm excludes non-finite (NaN/Inf) probe values; entries_changed counts them".into());
    }
    map[lo..hi].copy_from_slice(&buf.data);
    report.tensor_sha256_new = sha256_hex(&buf.data);
    Ok(report)
}

/// How a patch should be written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PatchMode {
    /// Copy `base` to `out` first (copy-on-write; the base is never touched).
    #[default]
    CopyOnWrite,
    /// Edit `base` itself. `out` is ignored; verification uses a per-tensor
    /// SHA-256 snapshot taken before the first write.
    InPlace,
}

/// Apply every plan edit directly to the payload bytes at their absolute file
/// offsets, then verify and fail closed unless verification passes.
pub fn apply_plan(base_path: &str, out_path: &str, plan: &Plan, mode: PatchMode) -> Result<PatchReport> {
    let (rep, err) = apply_plan_report(base_path, out_path, plan, mode);
    match err {
        Some(e) => Err(e),
        None => Ok(rep.expect("report present on success")),
    }
}

/// Like [`apply_plan`] but also returns the (possibly partial) report when the
/// post-patch verification gate fails, so a caller can still emit the evidence.
pub fn apply_plan_report(base_path: &str, out_path: &str, plan: &Plan, mode: PatchMode) -> (Option<PatchReport>, Option<Error>) {
    let mut report: Option<PatchReport> = None;
    let r = apply_plan_inner(base_path, out_path, plan, mode, &mut report);
    (report, r.err())
}

fn apply_plan_inner(base_path: &str, out_path: &str, plan: &Plan, mode: PatchMode, out_report: &mut Option<PatchReport>) -> Result<()> {
    let layout = load_layout(base_path)?;
    let (base_hash, base_bytes) = file_sha256(base_path)?;
    let target = match mode {
        PatchMode::CopyOnWrite => {
            copy_file(base_path, out_path)?;
            out_path
        }
        PatchMode::InPlace => base_path,
    };
    let snapshot = if mode == PatchMode::InPlace { Some(TensorDigests::of(base_path)?) } else { None };
    let file =
        OpenOptions::new().read(true).write(true).open(target).map_err(|e| Error::msg(format!("open {} for editing: {}", target, e)))?;
    // SAFETY: exclusive writer by contract; the file was just created (copy-on-write) or is the operator-named target.
    let mut map = unsafe { MmapMut::map_mut(&file) }.map_err(|e| Error::msg(format!("mmap {} for editing: {}", target, e)))?;
    let mut report = PatchReport {
        schema: PATCH_SCHEMA.into(),
        base: base_path.into(),
        out: out_path.into(),
        format: layout.format.clone(),
        base_sha256: base_hash,
        out_bytes: base_bytes,
        ..Default::default()
    };
    let mut edits = Vec::new();
    for edit in &plan.edits {
        let tpr = apply_edit(&mut map, &layout, edit).map_err(|e| Error::msg(format!("edit {}: {}", edit.id, e)))?;
        edits.push(tpr);
    }
    map.flush().map_err(|e| Error::msg(format!("sync {}: {}", target, e)))?;
    drop(map);
    drop(file);
    let (out_hash, out_bytes) = file_sha256(target)?;
    report.out_sha256 = out_hash;
    report.out_bytes = out_bytes;
    let expect: Vec<String> = edits.iter().map(|e| e.tensor.clone()).collect();
    report.edits = Some(edits);
    let verify = match &snapshot {
        None => verify_patch(base_path, out_path, &expect),
        Some(s) => s.verify_after(base_path, &expect),
    }
    .map_err(|e| Error::msg(format!("post-patch verify: {}", e)))?;
    let ok = verify.ok;
    let summary = verify.failure_summary();
    report.verify = Some(verify);
    *out_report = Some(report);
    if !ok {
        return Err(Error::Gate(format!("post-patch verify failed: {}", summary)));
    }
    Ok(())
}

pub fn copy_file(src: &str, dst: &str) -> Result<()> {
    let mut input = open_regular_file(src)?;
    let perm = input.metadata()?.permissions();
    let tmp = format!("{}.secant-tmp", dst);
    let mut out =
        OpenOptions::new().create(true).truncate(true).write(true).open(&tmp).map_err(|e| Error::msg(format!("create {}: {}", tmp, e)))?;
    if let Err(e) = std::io::copy(&mut input, &mut out) {
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::msg(format!("copy {} -> {}: {}", src, tmp, e)));
    }
    drop(out);
    let _ = std::fs::set_permissions(&tmp, perm);
    std::fs::rename(&tmp, dst).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        Error::msg(format!("install {}: {}", dst, e))
    })
}
