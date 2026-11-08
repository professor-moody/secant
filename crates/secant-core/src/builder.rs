//! Deterministic container writers shared by `heal`, `security::probe` and the
//! test fixtures. Byte-for-byte compatible with the oracle's
//! `BuildSafetensors` / `BuildGGUF` (`internal/wsp/modelbuild.go` over
//! `pkg/gguf/writer.go`): sorted JSON keys, 8-byte-aligned space-padded
//! header, GGUF v3, every tensor payload zero-padded to the alignment.

use crate::container::{ggml_type_id, hex};
use crate::dtype::dtype_size;
use crate::error::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// One tensor to serialize. `shape` is in SafeTensors / logical order
/// (outermost axis first); the GGUF writer reverses it into ne[] order.
#[derive(Debug, Clone)]
pub struct BuildTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl BuildTensor {
    pub fn elements(&self) -> i64 {
        self.shape.iter().fold(1i64, |a, d| a * *d as i64)
    }
}

fn quant_block(dtype: &str) -> Option<(usize, usize)> {
    crate::quant_scale::QuantScaleFormat::from_dtype(dtype).map(|f| (f.block_elems(), f.block_bytes()))
}

fn validate(tensors: &[BuildTensor]) -> Result<()> {
    if tensors.is_empty() {
        return Err(Error::msg("no tensors to serialize"));
    }
    let mut seen = std::collections::HashSet::new();
    for t in tensors {
        if t.name.is_empty() || !seen.insert(t.name.clone()) {
            return Err(Error::msg(format!("tensor name {:?} is empty or duplicated", t.name)));
        }
        if t.shape.iter().any(|d| *d == 0) {
            return Err(Error::msg(format!("tensor {:?}: non-positive dimension in shape {:?}", t.name, t.shape)));
        }
        if let Some((block, bytes)) = quant_block(&t.dtype) {
            // Secant extension (GGUF only): Q8_0 / Q4_K payloads, blocks along the last axis.
            let last = *t.shape.last().unwrap_or(&0);
            if last % block != 0 || t.data.len() != (t.elements() as usize / block) * bytes {
                return Err(Error::msg(format!(
                    "tensor {:?}: {} payload needs the last axis a multiple of {} and {} bytes per block, got shape {:?} and {} bytes",
                    t.name,
                    t.dtype,
                    block,
                    bytes,
                    t.shape,
                    t.data.len()
                )));
            }
            continue;
        }
        let w = dtype_size(&t.dtype);
        if w == 0 {
            return Err(Error::msg(format!("tensor {:?}: unsupported dtype {:?}", t.name, t.dtype)));
        }
        if t.data.len() as i64 != t.elements() * w {
            return Err(Error::msg(format!(
                "tensor {:?}: {} payload bytes, shape {:?} dtype {} needs {}",
                t.name,
                t.data.len(),
                t.shape,
                t.dtype,
                t.elements() * w
            )));
        }
    }
    Ok(())
}

/// Go `encoding/json` string encoding (HTML-escaping on).
pub fn go_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// SafeTensors bytes: tensors in the given order, space-padded JSON header
/// with sorted keys.
pub fn build_safetensors(tensors: &[BuildTensor], metadata: &BTreeMap<String, String>) -> Result<Vec<u8>> {
    validate(tensors)?;
    if let Some(t) = tensors.iter().find(|t| quant_block(&t.dtype).is_some()) {
        return Err(Error::msg(format!("tensor {:?}: SafeTensors cannot carry {}", t.name, t.dtype)));
    }
    // Keys sorted by byte order (as Go's map marshalling).
    let mut entries: BTreeMap<String, String> = BTreeMap::new();
    if !metadata.is_empty() {
        let kv: Vec<String> = metadata.iter().map(|(k, v)| format!("{}:{}", go_json_string(k), go_json_string(v))).collect();
        entries.insert("__metadata__".into(), format!("{{{}}}", kv.join(",")));
    }
    let mut body: Vec<u8> = Vec::new();
    for t in tensors {
        let begin = body.len();
        body.extend_from_slice(&t.data);
        let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        entries.insert(
            t.name.clone(),
            format!(
                "{{\"dtype\":{},\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                go_json_string(&t.dtype),
                shape.join(","),
                begin,
                body.len()
            ),
        );
    }
    let parts: Vec<String> = entries.iter().map(|(k, v)| format!("{}:{}", go_json_string(k), v)).collect();
    let mut hb = format!("{{{}}}", parts.join(",")).into_bytes();
    while hb.len() % 8 != 0 {
        hb.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hb.len() + body.len());
    out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
    out.extend_from_slice(&hb);
    out.extend_from_slice(&body);
    Ok(out)
}

/// A GGUF metadata value (the writer-supported subset of GGUF value types).
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(u32, Vec<MetaValue>),
}

impl MetaValue {
    fn type_id(&self) -> u32 {
        match self {
            MetaValue::U8(_) => 0,
            MetaValue::I8(_) => 1,
            MetaValue::U16(_) => 2,
            MetaValue::I16(_) => 3,
            MetaValue::U32(_) => 4,
            MetaValue::I32(_) => 5,
            MetaValue::F32(_) => 6,
            MetaValue::Bool(_) => 7,
            MetaValue::Str(_) => 8,
            MetaValue::Array(..) => 9,
            MetaValue::U64(_) => 10,
            MetaValue::I64(_) => 11,
            MetaValue::F64(_) => 12,
        }
    }
    fn write(&self, out: &mut Vec<u8>) {
        match self {
            MetaValue::U8(v) => out.push(*v),
            MetaValue::I8(v) => out.push(*v as u8),
            MetaValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetaValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetaValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetaValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetaValue::F32(v) => out.extend_from_slice(&v.to_bits().to_le_bytes()),
            MetaValue::Bool(v) => out.push(*v as u8),
            MetaValue::Str(s) => {
                out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            MetaValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetaValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetaValue::F64(v) => out.extend_from_slice(&v.to_bits().to_le_bytes()),
            MetaValue::Array(elem, data) => {
                out.extend_from_slice(&elem.to_le_bytes());
                out.extend_from_slice(&(data.len() as u64).to_le_bytes());
                for d in data {
                    d.write(out);
                }
            }
        }
    }
}

/// GGUF v3 bytes. `align` must be a power of two; `general.alignment` is
/// appended when `meta` does not carry it.
pub fn build_gguf(meta: &[(String, MetaValue)], tensors: &[BuildTensor], align: u64) -> Result<Vec<u8>> {
    validate(tensors)?;
    if align == 0 || align & (align - 1) != 0 {
        return Err(Error::msg(format!("alignment {} is not a power of two", align)));
    }
    let mut kvs: Vec<(String, MetaValue)> = meta.to_vec();
    if !kvs.iter().any(|(k, _)| k == "general.alignment") {
        kvs.push(("general.alignment".into(), MetaValue::U32(align as u32)));
    }
    // Alignment used for padding: general.alignment when u32/u64, else `align`.
    let mut pad_align = align;
    for (k, v) in &kvs {
        if k == "general.alignment" {
            match v {
                MetaValue::U32(x) => pad_align = *x as u64,
                MetaValue::U64(x) => pad_align = *x,
                _ => {}
            }
        }
    }
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in &kvs {
        out.extend_from_slice(&(k.len() as u64).to_le_bytes());
        out.extend_from_slice(k.as_bytes());
        out.extend_from_slice(&v.type_id().to_le_bytes());
        v.write(&mut out);
    }
    let mut offset: u64 = 0;
    for t in tensors {
        let ty = ggml_type_id(&t.dtype).ok_or_else(|| Error::msg(format!("no ggml type for dtype {:?}", t.dtype)))?;
        out.extend_from_slice(&(t.name.len() as u64).to_le_bytes());
        out.extend_from_slice(t.name.as_bytes());
        out.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
        for d in t.shape.iter().rev() {
            out.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += (t.data.len() as u64 + align - 1) / align * align;
    }
    if pad_align != 0 {
        let rem = out.len() as u64 % pad_align;
        if rem != 0 {
            let pad = (pad_align - rem).min(1 << 20);
            out.extend(std::iter::repeat(0u8).take(pad as usize));
        }
    }
    let base = out.len();
    out.resize(base + offset as usize, 0);
    let mut cur = base;
    for t in tensors {
        out[cur..cur + t.data.len()].copy_from_slice(&t.data);
        cur += (t.data.len() + align as usize - 1) / align as usize * align as usize;
    }
    Ok(out)
}

/// Create `path` exclusively (fails if it exists), write, fsync; remove the
/// partial file on any error.
pub fn write_file_exclusive(path: &str, data: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    let r = f.write_all(data).and_then(|_| f.sync_all());
    drop(f);
    if let Err(e) = r {
        let _ = std::fs::remove_file(path);
        return Err(Error::msg(e.to_string()));
    }
    Ok(())
}

pub fn sha256_bytes_hex(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}
