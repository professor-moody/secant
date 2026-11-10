//! Bounded, revision-pinned SafeTensors range reader. Never falls back to a
//! whole-shard GET when the server ignores Range.
use crate::util::*;
use clap::Args as ClapArgs;
use reqwest::{
    blocking::{Client, Response},
    header, StatusCode, Url,
};
use secant_core::{
    container::sha256_hex,
    dtype::{f32_from_bf16, f32_from_f16, f32_from_f8_e4m3, f32_from_f8_e5m2},
    Error,
};
use serde_json::{json, Value};
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::Path,
    time::Duration,
};

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Fetch one tensor, or a contiguous first-axis slice, from a Hugging Face SafeTensors model. Resolve --revision to a commit, read the shard index and two bounded header ranges, then request only the selected payload bytes. Servers must return exact HTTP 206 Content-Range responses. F16 and BF16 are widened to F32; other supported numeric dtypes retain their representation. Existing output and receipt files are never overwritten. HF_TOKEN is used when present and nonempty.",
    after_help = "Examples:\n  secant fetch-tensor --repo org/model --tensor model.norm.weight --out norm.npy\n  secant fetch-tensor --repo org/model --revision COMMIT --tensor lm_head.weight --row-start 0 --row-count 32 --out rows.npy --receipt rows.receipt.json --json\n\nDefault receipt: receipt.json beside --out. The default tensor limit is 512 MiB."
)]
pub struct Args {
    /// Hugging Face model identifier (owner/name)
    #[arg(long)]
    repo: String,
    /// Exact tensor name in the SafeTensors header
    #[arg(long)]
    tensor: String,
    /// Branch, tag or immutable commit to resolve before reading
    #[arg(long, default_value = "main")]
    revision: String,
    /// New .npy output file
    #[arg(long)]
    out: String,
    /// New receipt path (default: receipt.json beside the .npy)
    #[arg(long)]
    receipt: Option<String>,
    /// First row on axis 0 (requires --row-count)
    #[arg(long, requires = "row_count")]
    row_start: Option<u64>,
    /// Number of contiguous rows to fetch on axis 0
    #[arg(long)]
    row_count: Option<u64>,
    /// Maximum selected raw tensor bytes
    #[arg(long, default_value_t = 536_870_912)]
    max_bytes: u64,
    /// Hub endpoint, also usable with a local fixture server
    #[arg(long, default_value = "https://huggingface.co")]
    endpoint: String,
    /// Optional pipeline identity included in the receipt
    #[arg(long, default_value = "")]
    attempt_id: String,
    /// Emit the receipt as JSON
    #[arg(long)]
    json: bool,
}

const META_LIMIT: u64 = 16 * 1024 * 1024;
fn err(s: impl ToString) -> Error {
    Error::msg(format!("fetch-tensor: {}", s.to_string()))
}
fn safe_path(s: &str) -> bool {
    !s.is_empty() && s.split('/').all(|p| !p.is_empty() && p != "." && p != ".." && !p.contains(['\\', '?', '#']))
}
fn url(base: &str, parts: &[&str]) -> Result<Url, Error> {
    let mut u = Url::parse(base).map_err(err)?;
    if !["https", "http"].contains(&u.scheme())
        || u.query().is_some()
        || u.fragment().is_some()
        || !u.username().is_empty()
        || u.password().is_some()
    {
        return Err(err("endpoint must be an HTTP(S) URL without credentials, query or fragment"));
    }
    {
        let mut p = u.path_segments_mut().map_err(|_| err("endpoint has no path"))?;
        p.pop_if_empty();
        for part in parts {
            p.push(part);
        }
    }
    Ok(u)
}
fn bounded(response: Response, limit: u64) -> Result<Vec<u8>, Error> {
    if response.content_length().is_some_and(|n| n > limit) {
        return Err(err("response exceeds byte limit"));
    }
    let mut bytes = Vec::new();
    response.take(limit.checked_add(1).ok_or_else(|| err("byte limit overflow"))?).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(err("response exceeds byte limit"));
    }
    Ok(bytes)
}
fn request(client: &Client, u: Url, token: &Option<String>) -> reqwest::blocking::RequestBuilder {
    let r = client.get(u).header(header::ACCEPT_ENCODING, "identity");
    if let Some(t) = token {
        r.bearer_auth(t)
    } else {
        r
    }
}
fn metadata(client: &Client, u: Url, token: &Option<String>, optional: bool) -> Result<Option<Vec<u8>>, Error> {
    let r = request(client, u, token).send().map_err(|e| err(e.without_url()))?;
    if optional && r.status() == StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if r.status() != StatusCode::OK {
        return Err(err(format!("metadata HTTP {}; check repo/revision and HF_TOKEN for gated repositories", r.status())));
    }
    bounded(r, META_LIMIT).map(Some)
}
fn range(client: &Client, u: &Url, token: &Option<String>, start: u64, end: u64, total: Option<u64>) -> Result<(Vec<u8>, u64), Error> {
    if end < start {
        return Err(err("empty or inverted byte range"));
    }
    let n = end.checked_sub(start).and_then(|x| x.checked_add(1)).ok_or_else(|| err("range overflow"))?;
    let r = request(client, u.clone(), token)
        .header(header::RANGE, format!("bytes={}-{}", start, end))
        .send()
        .map_err(|e| err(e.without_url()))?;
    if r.status() != StatusCode::PARTIAL_CONTENT {
        return Err(err(format!("range request returned HTTP {}; refusing whole-shard fallback", r.status())));
    }
    if r.headers().get(header::CONTENT_ENCODING).is_some_and(|v| v != "identity") {
        return Err(err("encoded range response is not byte-addressable"));
    }
    let cr = r.headers().get(header::CONTENT_RANGE).and_then(|v| v.to_str().ok()).ok_or_else(|| err("missing Content-Range"))?;
    let prefix = format!("bytes {}-{}/", start, end);
    let size: u64 = cr.strip_prefix(&prefix).ok_or_else(|| err("Content-Range does not match requested offsets"))?.parse().map_err(err)?;
    if size <= end || total.is_some_and(|n| n != size) {
        return Err(err("inconsistent shard size in Content-Range"));
    }
    let raw = bounded(r, n)?;
    if raw.len() as u64 != n {
        return Err(err("truncated range response"));
    }
    Ok((raw, size))
}

fn npy(dtype: &str, shape: &[u64], raw: &[u8]) -> Result<(Vec<u8>, &'static str), Error> {
    let (descr, width) = match dtype {
        "F64" => ("<f8", 8),
        "F32" => ("<f4", 4),
        "F16" | "BF16" => ("<f4", 2),
        "F8_E4M3" | "F8_E5M2" => ("<f4", 1),
        "I64" => ("<i8", 8),
        "I32" => ("<i4", 4),
        "I16" => ("<i2", 2),
        "I8" => ("|i1", 1),
        "U64" => ("<u8", 8),
        "U32" => ("<u4", 4),
        "U16" => ("<u2", 2),
        "U8" => ("|u1", 1),
        "BOOL" => ("|b1", 1),
        _ => return Err(err(format!("unsupported dtype {}", dtype))),
    };
    if raw.len() % width != 0 {
        return Err(err("unaligned tensor payload"));
    }
    let dims = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
    let mut h =
        format!("{{'descr': '{}', 'fortran_order': False, 'shape': ({}{}), }}", descr, dims, if shape.len() == 1 { "," } else { "" });
    h.push_str(&" ".repeat((64 - (10 + h.len() + 1) % 64) % 64));
    h.push('\n');
    let len = u16::try_from(h.len()).map_err(|_| err("NPY header too large"))?;
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    if dtype == "F16" || dtype == "BF16" {
        for b in raw.chunks_exact(2) {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            let v = if dtype == "F16" { f32_from_f16(bits) } else { f32_from_bf16(bits) };
            out.extend_from_slice(&v.to_le_bytes());
        }
    } else if dtype == "F8_E4M3" || dtype == "F8_E5M2" {
        for &b in raw {
            let v = if dtype == "F8_E4M3" { f32_from_f8_e4m3(b) } else { f32_from_f8_e5m2(b) };
            out.extend_from_slice(&v.to_le_bytes());
        }
    } else {
        out.extend_from_slice(raw);
    }
    Ok((out, descr))
}

fn fetch(a: &Args) -> Result<Value, Error> {
    if !safe_path(&a.repo) || !safe_path(&a.revision) {
        return Err(err("repo and revision must be nonempty relative identifiers"));
    }
    if a.max_bytes == 0 || a.max_bytes > (1u64 << 34) {
        return Err(err("--max-bytes must be in 1..17179869184"));
    }
    let receipt = a.receipt.clone().unwrap_or_else(|| Path::new(&a.out).with_file_name("receipt.json").to_string_lossy().into_owned());
    for p in [&a.out, &receipt] {
        if std::fs::symlink_metadata(p).is_ok() {
            return Err(err(format!("output already exists: {}", p)));
        }
    }
    if !a.out.ends_with(".npy") {
        return Err(err("--out must end in .npy"));
    }
    let token = match std::env::var("HF_TOKEN") {
        Ok(t) if t.is_empty() => return Err(err("HF_TOKEN is present but empty; unset it for public repositories")),
        Ok(t) => Some(t),
        Err(std::env::VarError::NotPresent) => None,
        Err(e) => return Err(err(e)),
    };
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .connect_timeout(Duration::from_secs(15))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(err)?;
    let mut api = vec!["api", "models"];
    api.extend(a.repo.split('/'));
    api.extend(["revision", &a.revision]);
    let info: Value = serde_json::from_slice(&metadata(&client, url(&a.endpoint, &api)?, &token, false)?.unwrap()).map_err(err)?;
    let commit = info["sha"]
        .as_str()
        .filter(|s| s.len() == 40 && s.bytes().all(|c| c.is_ascii_hexdigit()))
        .ok_or_else(|| err("Hub did not return a 40-digit commit SHA"))?;
    let resolve = |file: &str| -> Result<Url, Error> {
        if !safe_path(file) {
            return Err(err("index shard path is not a relative path"));
        }
        let mut parts: Vec<&str> = a.repo.split('/').collect();
        parts.extend(["resolve", commit]);
        parts.extend(file.split('/'));
        url(&a.endpoint, &parts)
    };
    let index = metadata(&client, resolve("model.safetensors.index.json")?, &token, true)?;
    let shard = if let Some(ref b) = index {
        let v: Value = serde_json::from_slice(b).map_err(err)?;
        v["weight_map"][&a.tensor].as_str().ok_or_else(|| err(format!("tensor {:?} is absent from shard index", a.tensor)))?.to_string()
    } else {
        "model.safetensors".into()
    };
    let u = resolve(&shard)?;
    let (prefix, total) = range(&client, &u, &token, 0, 7, None)?;
    let hlen = u64::from_le_bytes(prefix.try_into().unwrap());
    if hlen == 0 || hlen > META_LIMIT || hlen.checked_add(8).is_none_or(|n| n > total) {
        return Err(err("invalid or oversized SafeTensors header"));
    }
    let (header_bytes, _) = range(&client, &u, &token, 8, 7 + hlen, Some(total))?;
    let header: Value = serde_json::from_slice(&header_bytes).map_err(err)?;
    let t = header.get(&a.tensor).ok_or_else(|| err(format!("tensor {:?} is absent from shard header", a.tensor)))?;
    let dtype = t["dtype"].as_str().ok_or_else(|| err("missing tensor dtype"))?;
    let width: u64 = match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "BF16" | "F16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => 1,
        _ => return Err(err(format!("unsupported dtype {}", dtype))),
    };
    let shape: Vec<u64> = t["shape"]
        .as_array()
        .ok_or_else(|| err("missing shape"))?
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| err("invalid shape dimension")))
        .collect::<Result<_, _>>()?;
    let count = shape.iter().try_fold(1u64, |n, d| n.checked_mul(*d).ok_or_else(|| err("shape overflow")))?;
    let offsets = t["data_offsets"].as_array().filter(|v| v.len() == 2).ok_or_else(|| err("invalid tensor offsets"))?;
    let begin = offsets[0].as_u64().ok_or_else(|| err("invalid start offset"))?;
    let end = offsets[1].as_u64().ok_or_else(|| err("invalid end offset"))?;
    let expected_bytes = count.checked_mul(width).ok_or_else(|| err("tensor byte count overflow"))?;
    if end.checked_sub(begin) != Some(expected_bytes) || end > total - 8 - hlen {
        return Err(err("tensor offsets do not match shape/dtype or shard size"));
    }
    let mut selected_shape = shape.clone();
    let (relative_start, length) = if let Some(n) = a.row_count {
        let first = a.row_start.unwrap_or(0);
        let rows = *shape.first().ok_or_else(|| err("cannot row-slice a scalar"))?;
        if n == 0 || rows == 0 || first.checked_add(n).is_none_or(|x| x > rows) {
            return Err(err("row slice is outside tensor shape"));
        }
        let stride = (end - begin) / rows;
        selected_shape[0] = n;
        (begin + first * stride, n * stride)
    } else {
        (begin, end - begin)
    };
    if length > a.max_bytes {
        return Err(err(format!("selected payload {} exceeds --max-bytes {}; select fewer rows or raise the limit", length, a.max_bytes)));
    }
    let start = 8 + hlen + relative_start;
    let raw = if length == 0 { Vec::new() } else { range(&client, &u, &token, start, start + length - 1, Some(total))?.0 };
    let (output, npy_dtype) = npy(dtype, &selected_shape, &raw)?;
    let rep = json!({"schema": "secant.fetch-tensor.v1", "attempt_id": a.attempt_id, "repo": a.repo, "revision": a.revision, "commit": commit, "shard": shard, "tensor": a.tensor, "dtype": dtype, "shape": shape, "selected_shape": selected_shape, "npy_dtype": npy_dtype,
        "byte_range": [start, start + length], "byte_range_convention": "half-open", "tensor_bytes": length, "shard_bytes": total, "header_bytes": hlen, "raw_sha256": sha256_hex(&raw), "header_sha256": sha256_hex(&header_bytes), "index_sha256": index.as_ref().map(|b| sha256_hex(b)), "npy_sha256": sha256_hex(&output), "out": a.out, "receipt": receipt, "range_verified": true});
    let mut out = OpenOptions::new().write(true).create_new(true).open(&a.out)?;
    let publish = (|| -> Result<(), Error> {
        out.write_all(&output)?;
        out.sync_all()?;
        let body = json_pretty(&rep)?;
        let mut rec = OpenOptions::new().write(true).create_new(true).open(&receipt)?;
        if let Err(e) = rec.write_all(body.as_bytes()).and_then(|_| rec.sync_all()) {
            let _ = std::fs::remove_file(&receipt);
            return Err(e.into());
        }
        Ok(())
    })();
    if publish.is_err() {
        let _ = std::fs::remove_file(&a.out);
    }
    publish?;
    Ok(rep)
}

pub fn run(a: &Args) -> i32 {
    match fetch(a) {
        Err(e) => fail(&e),
        Ok(v) => {
            if a.json {
                if let Err(e) = emit_json(&v) {
                    return fail(&e);
                }
            } else {
                section("Tensor range receipt");
                details(&v);
            }
            0
        }
    }
}
