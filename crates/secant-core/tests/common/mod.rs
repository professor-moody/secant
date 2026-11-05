//! Shared test helpers: deterministic fixtures, the Go-oracle runner and a
//! bit-exact JSON comparator.
//!
//! The oracle binary is `crucible` (it hosts `crucible wsp ...`). Its path comes
//! from `SECANT_ORACLE_BIN`, a THREE-state variable: absent => differential
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

/// Run `crucible wsp <args>`; returns None (after printing NOT_RUN) if no oracle.
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

/// Compare two JSON documents; floats must be bit-identical after parsing.
/// `ignore` lists object keys (at any depth) skipped, e.g. absolute paths.
pub fn json_diff(a: &Value, b: &Value, ignore: &[&str]) -> Option<String> {
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
