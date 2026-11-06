//! Shared test helpers: deterministic fixtures, the Go-oracle runner, a
//! bit-exact JSON comparator and the legacy-namespace normalizer.
//!
//! The oracle is the reference Go oracle (`wsp` subcommands of the oracle
//! binary at `SECANT_ORACLE_BIN`). Its path comes from `SECANT_ORACLE_BIN`, a
//! THREE-state variable: absent => differential
//! tests print `NOT_RUN` and return; present-but-empty => hard error (a
//! forwarded empty value must never silently disable a comparison); present
//! with a value => used.
#![allow(dead_code)]

use secant_core::builder::*;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub fn oracle_bin() -> Option<String> {
    match std::env::var("SECANT_ORACLE_BIN") {
        Err(std::env::VarError::NotPresent) => None,
        Ok(v) if v.is_empty() => panic!("SECANT_ORACLE_BIN is present but empty: refusing to treat it as unset"),
        Ok(v) => Some(v),
        Err(e) => panic!("SECANT_ORACLE_BIN: {}", e),
    }
}

/// Run the oracle's `wsp <args>`; returns None (after printing NOT_RUN) if no oracle.
pub fn run_oracle(args: &[&str]) -> Option<Output> {
    let bin = oracle_bin()?;
    let mut full = vec!["wsp"];
    full.extend_from_slice(args);
    Some(Command::new(bin).args(&full).output().expect("spawn oracle"))
}

pub fn not_run(test: &str) {
    println!("NOT_RUN {}: SECANT_ORACLE_BIN is absent; differential comparison skipped", test);
}

pub struct Scratch(pub PathBuf);
impl Scratch {
    pub fn new(tag: &str) -> Scratch {
        let p = std::env::temp_dir().join(format!("secant-test-{}-{}-{}", tag, std::process::id(), nanos()));
        std::fs::create_dir_all(&p).unwrap();
        Scratch(p)
    }
    pub fn path(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }
    pub fn dir(&self) -> &Path {
        &self.0
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn nanos() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
}

/// splitmix64-driven deterministic generator (no external RNG crate).
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// uniform in (-1, 1)
    pub fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 52) as f64) - 1.0
    }
    pub fn vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.unit()).collect()
    }
}

pub fn f32_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| (*x as f32).to_bits().to_le_bytes()).collect()
}
pub fn f64_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}
pub fn bf16_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| secant_core::dtype::bf16_from_f32(*x as f32).to_le_bytes()).collect()
}
pub fn f16_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| secant_core::dtype::f16_from_f32(*x as f32).to_le_bytes()).collect()
}
pub fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn tensor(name: &str, dtype: &str, shape: &[usize], rng: &mut Rng) -> BuildTensor {
    let n: usize = shape.iter().product();
    let v = rng.vec(n);
    let data = match dtype {
        "F32" => f32_bytes(&v),
        "F64" => f64_bytes(&v),
        "BF16" => bf16_bytes(&v),
        "F16" => f16_bytes(&v),
        d => panic!("fixture dtype {}", d),
    };
    BuildTensor { name: name.into(), dtype: dtype.into(), shape: shape.to_vec(), data }
}

pub fn write(path: &str, data: &[u8]) {
    std::fs::write(path, data).unwrap();
}

// ------------------------------------------------- legacy-namespace normalizer
//
// The reference oracle still writes schema ids, heal adapter keys and probe
// fixture metadata in the legacy namespace; Secant writes the `secant` one.
// Before comparing, the ORACLE side is mapped to the Secant spelling. The
// Secant side is never normalized: a legacy spelling in Secant output is a
// failure, so the mapping cannot hide a regression.

/// (legacy, secant) spellings: schema ids and heal keys (`<ns>.wsp.`), probe
/// metadata keys (`<ns>.probe.`) and the probe architecture (`<ns>-probe`).
pub fn legacy_spellings() -> &'static [(String, String)] {
    use secant_core::namespace::{LEGACY_NAMESPACE as L, NAMESPACE as N};
    static S: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        vec![
            (format!("{}.wsp.", L), format!("{}.wsp.", N)),
            (format!("{}.probe.", L), format!("{}.probe.", N)),
            (format!("{}-probe", L), format!("{}-probe", N)),
        ]
    })
}

pub fn has_legacy(s: &str) -> bool {
    s.contains(secant_core::namespace::LEGACY_NAMESPACE) && legacy_spellings().iter().any(|(l, _)| s.contains(l.as_str()))
}

/// Map every legacy spelling in `s` to the Secant spelling.
pub fn canon_text(s: &str) -> String {
    if !has_legacy(s) {
        return s.to_string();
    }
    let mut out = s.to_string();
    for (l, n) in legacy_spellings() {
        out = out.replace(l.as_str(), n);
    }
    out
}

/// `canon_text` applied to every object key and string value.
pub fn canon_value(v: &Value) -> Value {
    match v {
        Value::String(s) => Value::String(canon_text(s)),
        Value::Array(a) => Value::Array(a.iter().map(canon_value).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, x)| (canon_text(k), canon_value(x))).collect()),
        x => x.clone(),
    }
}

/// Replace string values that EQUAL a key of `map` (e.g. an oracle file's raw
/// SHA-256 mapped to the SHA-256 of its normalized bytes).
pub fn subst_strings(v: &Value, map: &std::collections::BTreeMap<String, String>) -> Value {
    match v {
        Value::String(s) => Value::String(map.get(s).cloned().unwrap_or_else(|| s.clone())),
        Value::Array(a) => Value::Array(a.iter().map(|x| subst_strings(x, map)).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, x)| (k.clone(), subst_strings(x, map))).collect()),
        x => x.clone(),
    }
}

/// First JSON path whose key or string value carries a legacy spelling.
pub fn find_legacy(v: &Value, path: &str) -> Option<String> {
    match v {
        Value::String(s) if has_legacy(s) => Some(format!("{} = {:?}", path, s)),
        Value::Array(a) => a.iter().enumerate().find_map(|(i, x)| find_legacy(x, &format!("{}[{}]", path, i))),
        Value::Object(o) => o.iter().find_map(|(k, x)| {
            if has_legacy(k) {
                Some(format!("{}/{} (key)", path, k))
            } else {
                find_legacy(x, &format!("{}/{}", path, k))
            }
        }),
        _ => None,
    }
}

/// Normalize a SafeTensors or GGUF file written by the oracle: rename legacy
/// metadata keys / string values and re-derive ONLY what the rename moves
/// (the SafeTensors `__metadata__` key order and 8-byte header padding; the
/// GGUF pre-data alignment padding). Tensor descriptors and payload bytes are
/// copied verbatim. `None` when the bytes carry no legacy spelling or are not
/// a container. Panics (never guesses) when the input does not round-trip
/// under the writer rules it assumes.
pub fn canon_container(b: &[u8]) -> Option<Vec<u8>> {
    if b.len() >= 24 && &b[..4] == b"GGUF" {
        return canon_gguf(b);
    }
    if b.len() >= 9 && b[8] == b'{' {
        let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        if n <= b.len() - 8 {
            return canon_safetensors(b, n);
        }
    }
    None
}

fn canon_safetensors(b: &[u8], n: usize) -> Option<Vec<u8>> {
    let header = std::str::from_utf8(&b[8..8 + n]).ok()?;
    if !has_legacy(header) {
        return None;
    }
    let body = header.trim_end_matches(' ');
    let mut repad = body.as_bytes().to_vec();
    while repad.len() % 8 != 0 {
        repad.push(b' ');
    }
    assert_eq!(repad, header.as_bytes(), "safetensors normalizer: header padding is not the 8-byte space rule");
    let v: Value = serde_json::from_str(body).expect("safetensors normalizer: header is not JSON");
    let meta: std::collections::BTreeMap<String, String> =
        serde_json::from_value(v.get("__metadata__").cloned().expect("legacy spelling outside __metadata__"))
            .expect("__metadata__ is not a string map");
    let orig = serde_json::to_string(&meta).unwrap();
    assert_eq!(body.matches(&orig).count(), 1, "safetensors normalizer: __metadata__ is not the sorted compact form");
    let renamed: std::collections::BTreeMap<String, String> = meta.iter().map(|(k, x)| (canon_text(k), canon_text(x))).collect();
    let mut h = body.replacen(&orig, &serde_json::to_string(&renamed).unwrap(), 1).into_bytes();
    assert!(!has_legacy(std::str::from_utf8(&h).unwrap()), "safetensors normalizer: legacy spelling outside __metadata__");
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&b[8 + n..]);
    Some(out)
}

fn canon_gguf(b: &[u8]) -> Option<Vec<u8>> {
    struct Cur<'a> {
        b: &'a [u8],
        p: usize,
    }
    impl<'a> Cur<'a> {
        fn take(&mut self, n: usize) -> &'a [u8] {
            let s = &self.b[self.p..self.p + n];
            self.p += n;
            s
        }
        fn u32(&mut self) -> u32 {
            u32::from_le_bytes(self.take(4).try_into().unwrap())
        }
        fn u64(&mut self) -> u64 {
            u64::from_le_bytes(self.take(8).try_into().unwrap())
        }
        fn string(&mut self) -> String {
            let n = self.u64() as usize;
            String::from_utf8(self.take(n).to_vec()).expect("gguf normalizer: non-UTF-8 string")
        }
    }
    fn put_str(out: &mut Vec<u8>, s: &str, changed: &mut bool) {
        let c = canon_text(s);
        *changed |= c != s;
        out.extend_from_slice(&(c.len() as u64).to_le_bytes());
        out.extend_from_slice(c.as_bytes());
    }
    fn value(r: &mut Cur, ty: u32, out: &mut Vec<u8>, changed: &mut bool) -> Option<u64> {
        let width = match ty {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4 | 5 | 6 => 4,
            10 | 11 | 12 => 8,
            8 => {
                let s = r.string();
                put_str(out, &s, changed);
                return None;
            }
            9 => {
                let (et, n) = (r.u32(), r.u64());
                out.extend_from_slice(&et.to_le_bytes());
                out.extend_from_slice(&n.to_le_bytes());
                for _ in 0..n {
                    value(r, et, out, changed);
                }
                return None;
            }
            t => panic!("gguf normalizer: unknown value type {}", t),
        };
        let raw = r.take(width);
        out.extend_from_slice(raw);
        match ty {
            4 => Some(u32::from_le_bytes(raw.try_into().unwrap()) as u64),
            10 => Some(u64::from_le_bytes(raw.try_into().unwrap())),
            _ => None,
        }
    }
    let mut r = Cur { b, p: 8 };
    let (n_t, n_kv) = (r.u64(), r.u64());
    let mut out = b[..24].to_vec();
    let (mut changed, mut align) = (false, 32u64);
    for _ in 0..n_kv {
        let key = r.string();
        let ty = r.u32();
        put_str(&mut out, &key, &mut changed);
        out.extend_from_slice(&ty.to_le_bytes());
        let num = value(&mut r, ty, &mut out, &mut changed);
        if key == "general.alignment" {
            align = num.expect("general.alignment is not u32/u64");
        }
    }
    let infos = r.p;
    for _ in 0..n_t {
        let name = r.string();
        assert!(!has_legacy(&name), "gguf normalizer: legacy spelling in tensor name {:?}", name);
        let nd = r.u32() as usize;
        r.take(8 * nd + 4 + 8);
    }
    if !changed {
        return None;
    }
    out.extend_from_slice(&b[infos..r.p]);
    let up = |x: usize| (x as u64).div_ceil(align) as usize * align as usize;
    let data_old = up(r.p);
    assert!(data_old <= b.len() && b[r.p..data_old].iter().all(|x| *x == 0), "gguf normalizer: pre-data padding is not zero fill");
    out.resize(up(out.len()), 0);
    out.extend_from_slice(&b[data_old..]);
    Some(out)
}

/// (raw SHA-256, SHA-256 after `canon_container`) of a file.
pub fn canon_sha256_file(path: &str) -> (String, String) {
    let b = std::fs::read(path).unwrap();
    let raw = secant_core::container::sha256_hex(&b);
    let canon = canon_container(&b).map(|c| secant_core::container::sha256_hex(&c)).unwrap_or_else(|| raw.clone());
    (raw, canon)
}

/// Compare two JSON documents; floats must be bit-identical after parsing.
/// `ignore` lists object keys (at any depth) skipped, e.g. absolute paths.
///
/// `a` is the ORACLE side: its legacy spellings are mapped to the Secant ones
/// first. `b` is the SECANT side: any legacy spelling in it is a difference.
pub fn json_diff(a: &Value, b: &Value, ignore: &[&str]) -> Option<String> {
    if let Some(p) = find_legacy(b, "") {
        return Some(format!("Secant output carries a legacy namespace spelling at {}", p));
    }
    let a = &canon_value(a);
    fn go(a: &Value, b: &Value, ignore: &[&str], path: &str) -> Option<String> {
        match (a, b) {
            (Value::Object(x), Value::Object(y)) => {
                let keys: std::collections::BTreeSet<&String> = x.keys().chain(y.keys()).collect();
                for k in keys {
                    if ignore.contains(&k.as_str()) {
                        continue;
                    }
                    let p = format!("{}/{}", path, k);
                    match (x.get(k), y.get(k)) {
                        (Some(l), Some(r)) => {
                            if let Some(d) = go(l, r, ignore, &p) {
                                return Some(d);
                            }
                        }
                        (l, r) => return Some(format!("{}: present on one side only ({:?} vs {:?})", p, l.is_some(), r.is_some())),
                    }
                }
                None
            }
            (Value::Array(x), Value::Array(y)) => {
                if x.len() != y.len() {
                    return Some(format!("{}: array length {} vs {}", path, x.len(), y.len()));
                }
                for (i, (l, r)) in x.iter().zip(y).enumerate() {
                    if let Some(d) = go(l, r, ignore, &format!("{}[{}]", path, i)) {
                        return Some(d);
                    }
                }
                None
            }
            (Value::Number(x), Value::Number(y)) => {
                if let (Some(l), Some(r)) = (x.as_i64(), y.as_i64()) {
                    return if l == r { None } else { Some(format!("{}: {} vs {}", path, l, r)) };
                }
                if let (Some(l), Some(r)) = (x.as_u64(), y.as_u64()) {
                    return if l == r { None } else { Some(format!("{}: {} vs {}", path, l, r)) };
                }
                let (l, r) = (x.as_f64().unwrap(), y.as_f64().unwrap());
                if l.to_bits() == r.to_bits() || (l == 0.0 && r == 0.0) {
                    None
                } else {
                    Some(format!("{}: float bits differ {:e} ({:016x}) vs {:e} ({:016x})", path, l, l.to_bits(), r, r.to_bits()))
                }
            }
            _ => {
                if a == b {
                    None
                } else {
                    Some(format!("{}: {} vs {}", path, a, b))
                }
            }
        }
    }
    go(a, b, ignore, "")
}

pub fn parse_json(out: &[u8]) -> Value {
    serde_json::from_slice(out).unwrap_or_else(|e| panic!("not JSON: {} in {:?}", e, String::from_utf8_lossy(out)))
}

pub fn sha256_file(path: &str) -> String {
    secant_core::container::file_sha256(path).unwrap().0
}

/// Synthetic Qwen3.5-style GGUF: `blk.N.ssm_out.weight` with ne[0]=d_in,
/// ne[1]=d_out (shape passed logical [d_out, d_in]) plus canonical ssm keys.
/// `num_k=2,num_v=4,head_v=4` => d_in 16.
pub fn qwen_meta(num_k: u32, num_v: u32, head_v: u32, head_k: u32) -> Vec<(String, MetaValue)> {
    vec![
        ("general.architecture".into(), MetaValue::Str("qwen35".into())),
        ("qwen35.block_count".into(), MetaValue::U32(2)),
        ("qwen35.embedding_length".into(), MetaValue::U32(8)),
        ("qwen35.ssm.group_count".into(), MetaValue::U32(num_k)),
        ("qwen35.ssm.time_step_rank".into(), MetaValue::U32(num_v)),
        ("qwen35.ssm.inner_size".into(), MetaValue::U32(num_v * head_v)),
        ("qwen35.ssm.state_size".into(), MetaValue::U32(head_k)),
    ]
}
