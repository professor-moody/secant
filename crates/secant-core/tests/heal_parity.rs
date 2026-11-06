//! Differential parity of `secant_core::heal` against the reference Go oracle
//! (`wsp heal` / `wsp verify-subspace` subcommands of the oracle binary at
//! `SECANT_ORACLE_BIN`).
//!
//! Run: `SECANT_ORACLE_BIN=/path/to/oracle cargo test -p secant-core --test heal_parity -- --nocapture`
//!
//! Every scenario builds identical inputs on disk, runs the oracle CLI and the
//! Rust library on separate output paths, then compares exit class, the full
//! `--json` report (floats bit-exact, path-valued keys ignored) and the SHA-256
//! of every file written. The oracle stamps the healed adapter with legacy-
//! namespace metadata keys: its output file is hashed after mapping them to the
//! `secant.wsp.heal*` keys (`common::canon_container`; tensor bytes verbatim),
//! and the report's `out_sha256` is mapped the same way. The Rust side is first run with the portable unfused
//! policy; if (and only if) that differs, it is re-run with the arm64-FMA
//! policy so a divergence can be attributed.
mod common;

use common::*;
use secant_core::builder::*;
use secant_core::heal::*;
use secant_core::subspace::npy_bytes_f64;
use serde_json::Value;
use std::sync::Mutex;

/// `numeric::set_go_arm64_fma` is process-global: serialize every test.
static LOCK: Mutex<()> = Mutex::new(());

const D: usize = 16;
const RANK: usize = 3;
/// Keys whose values are filesystem paths (differ between the two runs).
const PATH_KEYS: &[&str] = &["lora", "vectors", "out", "base", "merge_out", "source", "healed"];

fn guard() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// ------------------------------------------------------------ fixtures

fn put_npy(path: &str, vals: &[f64], shape: &[usize]) {
    write(path, &npy_bytes_f64(vals, shape).unwrap());
}

/// Deliberately correlated protected directions of dimension `d`.
fn raw_dirs(d: usize, k: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    for i in 0..k {
        let mut v = rng.vec(d);
        if i == 1 {
            for j in 0..d {
                v[j] += 0.8 * out[0][j];
            }
        }
        out.push(v);
    }
    out
}

fn vectors_dir(sc: &Scratch, name: &str, dirs: &[Vec<f64>], rng: &mut Rng) -> String {
    let dir = sc.path(name);
    std::fs::create_dir_all(&dir).unwrap();
    for (i, v) in dirs.iter().enumerate() {
        put_npy(&format!("{}/d_{}.npy", dir, (b'a' + i as u8) as char), v, &[v.len()]);
    }
    // reference activations must be ignored as protected directions
    let h = rng.vec(dirs[0].len());
    put_npy(&format!("{}/h_ref.npy", dir), &h, &[h.len()]);
    dir
}

fn bytes_of(dtype: &str, v: &[f64]) -> Vec<u8> {
    match dtype {
        "F32" => f32_bytes(v),
        "F64" => f64_bytes(v),
        "BF16" => bf16_bytes(v),
        "F16" => f16_bytes(v),
        d => panic!("dtype {}", d),
    }
}

struct Module {
    stem: String,
    a: Vec<f64>, // r x d_in
    b: Vec<f64>, // d x r
    d_in: usize,
}

fn module(stem: &str, d: usize, rank: usize, d_in: usize, rng: &mut Rng) -> Module {
    let a: Vec<f64> = rng.vec(rank * d_in).iter().map(|x| x * 0.3).collect();
    // B carries a large component along the protected directions by construction (random in R^d).
    let b = rng.vec(d * rank);
    Module { stem: stem.into(), a, b, d_in }
}

fn lora_bytes(mods: &[Module], d: usize, rank: usize, dtype: &str) -> Vec<u8> {
    let mut ts = Vec::new();
    for m in mods {
        ts.push(BuildTensor {
            name: format!("{}.lora_A.weight", m.stem),
            dtype: dtype.into(),
            shape: vec![rank, m.d_in],
            data: bytes_of(dtype, &m.a),
        });
        ts.push(BuildTensor {
            name: format!("{}.lora_B.weight", m.stem),
            dtype: dtype.into(),
            shape: vec![d, rank],
            data: bytes_of(dtype, &m.b),
        });
    }
    build_safetensors(&ts, &Default::default()).unwrap()
}

fn base_matrix(rows: usize, cols: usize, amp: f64) -> Vec<f64> {
    (0..rows * cols).map(|i| amp * ((i * 53 + 7) % 2003) as f64 / 1001.0 - amp).collect()
}

fn base_tensors(names: &[(&str, usize)], d: usize, dtype: &str) -> Vec<BuildTensor> {
    let mut ts: Vec<BuildTensor> = names
        .iter()
        .map(|(n, d_in)| BuildTensor {
            name: (*n).into(),
            dtype: dtype.into(),
            shape: vec![d, *d_in],
            data: bytes_of(dtype, &base_matrix(d, *d_in, 0.5)),
        })
        .collect();
    ts.push(BuildTensor {
        name: "output.norm.weight".into(),
        dtype: dtype.into(),
        shape: vec![d],
        data: bytes_of(dtype, &base_matrix(1, d, 1.0)),
    });
    ts
}

fn llama_meta() -> Vec<(String, MetaValue)> {
    vec![("general.architecture".into(), MetaValue::Str("llama".into())), ("llama.block_count".into(), MetaValue::U32(1))]
}

// ------------------------------------------------------------ runners

#[derive(Debug)]
struct Run {
    code: i32,
    report: Value,
    files: Vec<(String, Option<String>)>, // (label, sha256 if the file exists)
}

/// Secant side: raw SHA-256 of an output; a legacy spelling in it is a failure.
fn sha_if_exists(p: &str) -> Option<String> {
    if std::path::Path::new(p).exists() {
        let (raw, canon) = canon_sha256_file(p);
        assert_eq!(raw, canon, "{}: Secant wrote a legacy namespace spelling", p);
        Some(raw)
    } else {
        None
    }
}

/// Oracle side: SHA-256 after legacy-namespace normalization; records
/// raw -> normalized in `map` when they differ.
fn oracle_sha_if_exists(p: &str, map: &mut std::collections::BTreeMap<String, String>) -> Option<String> {
    if !std::path::Path::new(p).exists() {
        return None;
    }
    let (raw, canon) = canon_sha256_file(p);
    if raw != canon {
        map.insert(raw, canon.clone());
    }
    Some(canon)
}

struct HealCase<'a> {
    lora: &'a str,
    vectors: &'a str,
    base: &'a str,
    scale: f64,
    tol: Option<f64>,
}

fn oracle_heal(c: &HealCase, out: &str, merge: &str) -> Option<Run> {
    let mut args: Vec<String> = vec![
        "heal".into(),
        "--lora".into(),
        c.lora.into(),
        "--vectors".into(),
        c.vectors.into(),
        "--out".into(),
        out.into(),
        "--json".into(),
    ];
    if !c.base.is_empty() {
        args.extend(["--base".into(), c.base.into(), "--merge-out".into(), merge.into()]);
    }
    if c.scale != 0.0 {
        args.extend(["--scale".into(), format!("{:?}", c.scale)]);
    }
    if let Some(t) = c.tol {
        args.extend(["--tol".into(), format!("{:?}", t)]);
    }
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let o = run_oracle(&refs)?;
    let report = if o.stdout.is_empty() { Value::Null } else { parse_json(&o.stdout) };
    let mut map = std::collections::BTreeMap::new();
    let files = vec![
        ("out".into(), oracle_sha_if_exists(out, &mut map)),
        ("merge".into(), if merge.is_empty() { None } else { oracle_sha_if_exists(merge, &mut map) }),
    ];
    // A failed gate leaves no adapter on disk, but the report still carries the
    // staged adapter's `out_sha256`. Re-derive those bytes with a heal-only run
    // (the adapter does not depend on --base/--merge-out) and map the hash ONLY
    // if the re-derived raw SHA-256 equals the reported one, i.e. the bytes are
    // proven identical; otherwise nothing is mapped and the comparison fails.
    if let Some(reported) = report["out_sha256"].as_str() {
        if !reported.is_empty() && files[0].1.is_none() {
            let tmp = Scratch::new("heal-rederive");
            let again = tmp.path("adapter.safetensors");
            let mut a: Vec<String> = vec!["heal".into(), "--lora".into(), c.lora.into(), "--vectors".into(), c.vectors.into()];
            a.extend(["--out".into(), again.clone(), "--json".into()]);
            if let Some(t) = c.tol {
                a.extend(["--tol".into(), format!("{:?}", t)]);
            }
            let r: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            run_oracle(&r)?;
            if std::path::Path::new(&again).exists() {
                let (raw, canon) = canon_sha256_file(&again);
                if raw == reported && raw != canon {
                    map.insert(raw, canon);
                }
            }
        }
    }
    Some(Run { code: o.status.code().unwrap_or(-1), report: subst_strings(&report, &map), files })
}

fn rust_heal(c: &HealCase, out: &str, merge: &str) -> Run {
    let o = HealOptions {
        lora: c.lora.into(),
        vectors: c.vectors.into(),
        out: out.into(),
        base: c.base.into(),
        merge_out: if c.base.is_empty() { String::new() } else { merge.into() },
        scale: c.scale,
        tol: c.tol.unwrap_or(DEFAULT_SUBSPACE_TOL),
    };
    let (rep, res) = heal(&o);
    let code = match res {
        Ok(()) => 0,
        Err(e) => e.exit_code(),
    };
    Run {
        code,
        report: serde_json::to_value(rep.expect("report")).unwrap(),
        files: vec![("out".into(), sha_if_exists(out)), ("merge".into(), if merge.is_empty() { None } else { sha_if_exists(merge) })],
    }
}

fn diff_runs(go: &Run, rs: &Run) -> Option<String> {
    if go.code != rs.code {
        return Some(format!("exit code go={} rust={}", go.code, rs.code));
    }
    if go.files != rs.files {
        return Some(format!("output file sha256 go={:?} rust={:?}", go.files, rs.files));
    }
    json_diff(&go.report, &rs.report, PATH_KEYS)
}

/// Compare one heal scenario. Returns the Rust run (portable policy) so the
/// caller can add scenario-specific assertions.
fn check_heal(tag: &str, c: &HealCase, sc: &Scratch) -> Option<Run> {
    let go = oracle_heal(c, &sc.path(&format!("{}-go-out.safetensors", tag)), &sc.path(&format!("{}-go-merged", tag)));
    let go = match go {
        Some(g) => g,
        None => {
            not_run(tag);
            // still run the Rust side so the scenario exercises the library
            let r = rust_heal(c, &sc.path(&format!("{}-rs-out.safetensors", tag)), &sc.path(&format!("{}-rs-merged", tag)));
            return Some(r);
        }
    };
    let rs = rust_heal(c, &sc.path(&format!("{}-rs-out.safetensors", tag)), &sc.path(&format!("{}-rs-merged", tag)));
    let d_unfused = diff_runs(&go, &rs);
    if d_unfused.is_none() {
        println!("PARITY {} policy=unfused MATCH exit={} out={:?}", tag, go.code, go.files);
        return Some(rs);
    }
    secant_core::numeric::set_go_arm64_fma(true);
    let rs_f = rust_heal(c, &sc.path(&format!("{}-rsf-out.safetensors", tag)), &sc.path(&format!("{}-rsf-merged", tag)));
    secant_core::numeric::set_go_arm64_fma(false);
    let d_fused = diff_runs(&go, &rs_f);
    println!(
        "PARITY {} unfused=DIFF({}) fused={}",
        tag,
        d_unfused.as_ref().unwrap(),
        d_fused.as_deref().map(|d| format!("DIFF({})", d)).unwrap_or("MATCH".into())
    );
    // The oracle binary is an arm64 build whose compiler fuses ordinary `s += a*b`: matching under
    // the fused policy is the expected arm64 result; matching under neither is a porting bug.
    assert!(d_fused.is_none(), "{}: neither the unfused nor the fused policy reproduces the oracle", tag);
    Some(rs_f)
}

// ------------------------------------------------------------ heal: LoRA only

fn standard_mods(rng: &mut Rng) -> Vec<Module> {
    vec![
        module("base_model.model.model.layers.0.self_attn.o_proj", D, RANK, 24, rng),
        module("base_model.model.model.layers.0.mlp.down_proj", D, RANK, 40, rng),
    ]
}

#[test]
fn heal_lora_f32_directory_vectors() {
    let _g = guard();
    let sc = Scratch::new("heal-f32");
    let mut rng = Rng(11);
    let mods = standard_mods(&mut rng);
    let lora = sc.path("adapter.safetensors");
    write(&lora, &lora_bytes(&mods, D, RANK, "F32"));
    let dirs = raw_dirs(D, 3, &mut rng);
    let vec_dir = vectors_dir(&sc, "vec", &dirs, &mut rng);
    let c = HealCase { lora: &lora, vectors: &vec_dir, base: "", scale: 0.0, tol: None };
    let r = check_heal("heal_f32", &c, &sc).unwrap();
    assert_eq!(r.code, 0, "{:?}", r.report);
    let rep = &r.report;
    assert_eq!(rep["status"], "healed");
    assert_eq!(rep["schema"], HEAL_SCHEMA);
    assert!(rep["max_leak_after_f64"].as_f64().unwrap() < 1e-12);
    assert!(rep["max_leak_after_serialized"].as_f64().unwrap() < 1e-6);
    for p in rep["pairs"].as_array().unwrap() {
        assert!(p["leak_before_f64"].as_f64().unwrap() > 0.05, "fixture must exercise the projection: {}", p);
        assert_eq!(p["serialized_b_dtype"], "F32");
        assert_eq!(p["a_bytes_unchanged"], true);
    }
    // the healed adapter is stamped with the secant.wsp.heal metadata keys
    let b = std::fs::read(sc.path("heal_f32-rs-out.safetensors")).unwrap();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
    let m = &h["__metadata__"];
    assert_eq!(m[HEAL_META_KEY], "v1", "{}", m);
    assert_eq!(m[HEAL_META_PROTECTED_DIM], D.to_string(), "{}", m);
    assert_eq!(m[HEAL_META_PROTECTED_K], "3", "{}", m);
    assert_eq!(m["format"], "pt");
    assert_eq!(m.as_object().unwrap().len(), 4, "{}", m);
}

#[test]
fn heal_lora_single_npy_matrix_vectors() {
    let _g = guard();
    let sc = Scratch::new("heal-npy2d");
    let mut rng = Rng(23);
    let mods = standard_mods(&mut rng);
    let lora = sc.path("adapter.safetensors");
    write(&lora, &lora_bytes(&mods, D, RANK, "F32"));
    let dirs = raw_dirs(D, 4, &mut rng);
    let flat: Vec<f64> = dirs.iter().flatten().copied().collect();
    let npy = sc.path("protected.npy");
    put_npy(&npy, &flat, &[4, D]);
    let c = HealCase { lora: &lora, vectors: &npy, base: "", scale: 0.0, tol: None };
    let r = check_heal("heal_npy2d", &c, &sc).unwrap();
    assert_eq!(r.code, 0, "{:?}", r.report);
    assert_eq!(r.report["protected"]["k"], 4);
}

#[test]
fn heal_lora_bf16_and_f64_inputs() {
    let _g = guard();
    for (dtype, want_b) in [("BF16", "F32"), ("F16", "F32"), ("F64", "F64")] {
        let sc = Scratch::new("heal-dtype");
        let mut rng = Rng(31);
        let mods = standard_mods(&mut rng);
        let lora = sc.path("adapter.safetensors");
        write(&lora, &lora_bytes(&mods, D, RANK, dtype));
        let dirs = raw_dirs(D, 3, &mut rng);
        let vec_dir = vectors_dir(&sc, "vec", &dirs, &mut rng);
        let c = HealCase { lora: &lora, vectors: &vec_dir, base: "", scale: 0.0, tol: None };
        let r = check_heal(&format!("heal_{}", dtype.to_lowercase()), &c, &sc).unwrap();
        assert_eq!(r.code, 0, "{:?}", r.report);
        for p in r.report["pairs"].as_array().unwrap() {
            assert_eq!(p["serialized_b_dtype"], want_b);
        }
    }
}

#[test]
fn heal_error_paths_match_oracle() {
    let _g = guard();
    let sc = Scratch::new("heal-err");
    let mut rng = Rng(41);
    let mods = standard_mods(&mut rng);
    let lora = sc.path("adapter.safetensors");
    write(&lora, &lora_bytes(&mods, D, RANK, "F32"));
    // protected dimension 12 != d_out 16: refuse rather than leave a module unlocked
    let dirs = raw_dirs(12, 2, &mut rng);
    let vec_dir = vectors_dir(&sc, "vec12", &dirs, &mut rng);
    let c = HealCase { lora: &lora, vectors: &vec_dir, base: "", scale: 0.0, tol: None };
    let r = check_heal("heal_dim_mismatch", &c, &sc).unwrap();
    assert_eq!(r.code, 2);
    assert_eq!(r.report["status"], "failed");
    assert!(!sc.dir().join("heal_dim_mismatch-rs-out.safetensors").exists(), "nothing published on failure");

    // linearly dependent directions fail closed in Gram-Schmidt
    let mut dep = raw_dirs(D, 2, &mut rng);
    dep[1] = dep[0].iter().map(|x| x * 2.0).collect();
    let vdep = vectors_dir(&sc, "vecdep", &dep, &mut rng);
    let c = HealCase { lora: &lora, vectors: &vdep, base: "", scale: 0.0, tol: None };
    let r = check_heal("heal_dependent", &c, &sc).unwrap();
    assert_eq!(r.code, 2);

    // a stray (non lora_A/B) tensor is refused, never passed through
    let mut ts = vec![BuildTensor { name: "embed.weight".into(), dtype: "F32".into(), shape: vec![D], data: f32_bytes(&rng.vec(D)) }];
    ts.push(BuildTensor { name: "m.lora_A.weight".into(), dtype: "F32".into(), shape: vec![RANK, 8], data: f32_bytes(&rng.vec(RANK * 8)) });
    ts.push(BuildTensor { name: "m.lora_B.weight".into(), dtype: "F32".into(), shape: vec![D, RANK], data: f32_bytes(&rng.vec(D * RANK)) });
    let stray = sc.path("stray.safetensors");
    write(&stray, &build_safetensors(&ts, &Default::default()).unwrap());
    let dirs = raw_dirs(D, 3, &mut rng);
    let ok_vec = vectors_dir(&sc, "vecok", &dirs, &mut rng);
    let c = HealCase { lora: &stray, vectors: &ok_vec, base: "", scale: 0.0, tol: None };
    let r = check_heal("heal_stray", &c, &sc).unwrap();
    assert_eq!(r.code, 2);
}

// ------------------------------------------------------------ heal + merge

#[test]
fn heal_merge_into_safetensors_base() {
    let _g = guard();
    let sc = Scratch::new("merge-st");
    let mut rng = Rng(51);
    let mods = standard_mods(&mut rng);
    let lora = sc.path("adapter.safetensors");
    write(&lora, &lora_bytes(&mods, D, RANK, "F32"));
    let base = sc.path("base.safetensors");
    let bt = base_tensors(&[("model.layers.0.self_attn.o_proj.weight", 24), ("model.layers.0.mlp.down_proj.weight", 40)], D, "F32");
    write(&base, &build_safetensors(&bt, &Default::default()).unwrap());
    let dirs = raw_dirs(D, 3, &mut rng);
    let vec_dir = vectors_dir(&sc, "vec", &dirs, &mut rng);
    let c = HealCase { lora: &lora, vectors: &vec_dir, base: &base, scale: 2.0, tol: None };
    let r = check_heal("merge_st", &c, &sc).unwrap();
    assert_eq!(r.code, 0, "{:?}", r.report["merge"]);
    let m = &r.report["merge"];
    assert_eq!(m["published"], true);
    assert_eq!(m["header_bytes_identical"], true);
    assert_eq!(m["gate"]["verdict"], "pass");
    assert_eq!(m["targets"].as_array().unwrap().len(), 2);
    for t in m["gate"]["tensors"].as_array().unwrap() {
        assert_eq!(t["delta_is_zero"], false, "non-vacuous pass expected");
    }
}

#[test]
fn heal_merge_into_gguf_base_with_hf_name_mapping() {
    let _g = guard();
    let sc = Scratch::new("merge-gguf");
    let mut rng = Rng(61);
    let mods = standard_mods(&mut rng);
    let lora = sc.path("adapter.safetensors");
    write(&lora, &lora_bytes(&mods, D, RANK, "F32"));
    let base = sc.path("base.gguf");
    let bt = base_tensors(&[("blk.0.attn_output.weight", 24), ("blk.0.ffn_down.weight", 40)], D, "F32");
    write(&base, &build_gguf(&llama_meta(), &bt, 32).unwrap());
    let dirs = raw_dirs(D, 3, &mut rng);
    let vec_dir = vectors_dir(&sc, "vec", &dirs, &mut rng);
    let c = HealCase { lora: &lora, vectors: &vec_dir, base: &base, scale: 0.5, tol: None };
    let r = check_heal("merge_gguf", &c, &sc).unwrap();
    assert_eq!(r.code, 0, "{:?}", r.report["merge"]);
    let targets: Vec<&str> = r.report["merge"]["targets"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert!(targets.contains(&"blk.0.attn_output.weight") && targets.contains(&"blk.0.ffn_down.weight"), "{:?}", targets);
}

#[test]
fn heal_merge_bf16_base_fails_the_gate_honestly() {
    let _g = guard();
    let sc = Scratch::new("merge-bf16");
    let mut rng = Rng(71);
    let mods = standard_mods(&mut rng);
    let lora = sc.path("adapter.safetensors");
    write(&lora, &lora_bytes(&mods, D, RANK, "F32"));
    let base = sc.path("base.safetensors");
    let bt = base_tensors(&[("model.layers.0.self_attn.o_proj.weight", 24), ("model.layers.0.mlp.down_proj.weight", 40)], D, "BF16");
    write(&base, &build_safetensors(&bt, &Default::default()).unwrap());
    let dirs = raw_dirs(D, 3, &mut rng);
    let vec_dir = vectors_dir(&sc, "vec", &dirs, &mut rng);
    let c = HealCase { lora: &lora, vectors: &vec_dir, base: &base, scale: 1.0, tol: None };
    let r = check_heal("merge_bf16", &c, &sc).unwrap();
    assert_eq!(r.code, 1, "BF16 rounding must fail the realized-leakage gate (exit 1): {:?}", r.report["error"]);
    assert_eq!(r.report["status"], "failed");
    assert_eq!(r.report["out_published"], false);
    assert_eq!(r.report["merge"]["published"], false);
    assert_eq!(r.report["merge"]["gate"]["verdict"], "leak");
    // all-or-nothing: neither the healed LoRA nor the merged model is left behind
    assert_eq!(r.files, vec![("out".to_string(), None), ("merge".to_string(), None)]);
    // ... and no staging temp file survives the failure (oracle and Rust runs share this dir)
    let leftovers: Vec<String> = std::fs::read_dir(sc.dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(".wsp-"))
        .collect();
    assert!(leftovers.is_empty(), "staging files left behind: {:?}", leftovers);
}

#[test]
fn heal_merge_qwen35_ssm_out_permutes_lora_a_columns() {
    let _g = guard();
    let sc = Scratch::new("merge-qwen");
    let mut rng = Rng(81);
    // num_k=2, num_v=4, head_v=4 => d_in = 16; d_out = 8 (= embedding_length)
    let (d_out, d_in, rank) = (8usize, 16usize, 2usize);
    let stem = "base_model.model.model.layers.0.linear_attn.out_proj";
    let mods = vec![module(stem, d_out, rank, d_in, &mut rng)];
    let lora = sc.path("adapter.safetensors");
    write(&lora, &lora_bytes(&mods, d_out, rank, "F32"));
    let base = sc.path("base.gguf");
    let bt = base_tensors(&[("blk.0.ssm_out.weight", d_in)], d_out, "F32");
    write(&base, &build_gguf(&qwen_meta(2, 4, 4, 4), &bt, 32).unwrap());
    let dirs = raw_dirs(d_out, 2, &mut rng);
    let vec_dir = vectors_dir(&sc, "vec", &dirs, &mut rng);
    let c = HealCase { lora: &lora, vectors: &vec_dir, base: &base, scale: 1.5, tol: None };
    let r = check_heal("merge_qwen_ssm_out", &c, &sc).unwrap();
    assert_eq!(r.code, 0, "{:?}", r.report["merge"]);
    assert_eq!(r.report["pairs"][0]["a_columns_permuted_hf_to_gguf"], true);
    assert_eq!(r.report["pairs"][0]["merge_target"], "blk.0.ssm_out.weight");
    // Independent expectation (no Heal code): GGUF block j holds HF block
    // (j % num_k) * r + j / num_k; GGUF input column g reads HF column perm[g].
    let (num_k, num_v, head_v) = (2usize, 4usize, 4usize);
    let ratio = num_v / num_k;
    let perm: Vec<usize> = (0..d_in).map(|g| ((g / head_v) % num_k * ratio + (g / head_v) / num_k) * head_v + g % head_v).collect();
    assert_ne!(perm, (0..d_in).collect::<Vec<_>>(), "the interleave must be a real permutation in this geometry");
    let read = |path: &str, name: &str| -> Vec<f64> {
        let m = secant_core::container::Model::open(path).unwrap();
        let t = m.layout.tensor(name).unwrap().clone();
        let n = (t.size / secant_core::dtype::dtype_size(&t.dtype)) as usize;
        let mut v = vec![0f64; n];
        secant_core::dtype::decode_row(&t.dtype, m.tensor_bytes(&t), 0, n, &mut v);
        v
    };
    let healed_out = sc.path("rs-direct-out.safetensors");
    let merged = sc.path("rs-direct-merged.gguf");
    let (_, e) = heal(&HealOptions {
        lora: lora.clone(),
        vectors: vec_dir.clone(),
        out: healed_out.clone(),
        base: base.clone(),
        merge_out: merged.clone(),
        scale: 1.5,
        tol: DEFAULT_SUBSPACE_TOL,
    });
    e.unwrap();
    let b_locked = read(&healed_out, &format!("{}.lora_B.weight", stem));
    let a = &mods[0].a;
    let old = read(&base, "blk.0.ssm_out.weight");
    let new = read(&merged, "blk.0.ssm_out.weight");
    let (mut worst_perm, mut worst_plain) = (0f64, 0f64);
    for o in 0..d_out {
        for g in 0..d_in {
            let (mut dp, mut dn) = (0f64, 0f64);
            for k in 0..rank {
                dp += b_locked[o * rank + k] * a[k * d_in + perm[g]];
                dn += b_locked[o * rank + k] * a[k * d_in + g];
            }
            worst_perm = worst_perm.max((new[o * d_in + g] - (old[o * d_in + g] + 1.5 * dp)).abs());
            worst_plain = worst_plain.max((new[o * d_in + g] - (old[o * d_in + g] + 1.5 * dn)).abs());
        }
    }
    assert!(worst_perm < 1e-5, "merged ssm_out does not equal W + s*B_locked*A[:, perm]: {:e}", worst_perm);
    assert!(worst_plain > 1e-2, "the unpermuted expectation must NOT match (else the test cannot see the permutation): {:e}", worst_plain);
}

// ------------------------------------------------------------ verify-subspace

fn oracle_verify(base: &str, healed: &str, vectors: &str, tensors: &[&str], tol: Option<f64>) -> Option<(i32, Value)> {
    let joined = tensors.join(",");
    let mut args: Vec<String> = vec![
        "verify-subspace".into(),
        "--base".into(),
        base.into(),
        "--healed".into(),
        healed.into(),
        "--vectors".into(),
        vectors.into(),
        "--tensors".into(),
        joined,
        "--json".into(),
    ];
    if let Some(t) = tol {
        args.extend(["--tol".into(), format!("{:?}", t)]);
    }
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let o = run_oracle(&refs)?;
    let v = if o.stdout.is_empty() { Value::Null } else { parse_json(&o.stdout) };
    Some((o.status.code().unwrap_or(-1), v))
}

fn rust_verify(base: &str, healed: &str, vectors: &str, tensors: &[&str], tol: Option<f64>) -> (i32, Value) {
    let o = VerifySubspaceOptions {
        base: base.into(),
        healed: healed.into(),
        vectors: vectors.into(),
        protected: None,
        tensors: tensors.iter().map(|s| s.to_string()).collect(),
        tol: tol.unwrap_or(DEFAULT_SUBSPACE_TOL),
    };
    match verify_subspace(&o) {
        Err(_) => (2, Value::Null),
        Ok(rep) => {
            let code = match rep.verdict.as_str() {
                "pass" => 0,
                "leak" => 1,
                _ => 2,
            };
            (code, serde_json::to_value(&rep).unwrap())
        }
    }
}

fn check_verify(tag: &str, base: &str, healed: &str, vectors: &str, tensors: &[&str], tol: Option<f64>) -> (i32, Value) {
    let rs = rust_verify(base, healed, vectors, tensors, tol);
    match oracle_verify(base, healed, vectors, tensors, tol) {
        None => {
            not_run(tag);
        }
        Some(go) => {
            let mut d =
                if go.0 != rs.0 { Some(format!("exit code go={} rust={}", go.0, rs.0)) } else { json_diff(&go.1, &rs.1, PATH_KEYS) };
            if d.is_some() {
                let unfused = d.clone().unwrap();
                secant_core::numeric::set_go_arm64_fma(true);
                let rf = rust_verify(base, healed, vectors, tensors, tol);
                secant_core::numeric::set_go_arm64_fma(false);
                let df = if go.0 != rf.0 { Some("exit".to_string()) } else { json_diff(&go.1, &rf.1, PATH_KEYS) };
                println!(
                    "PARITY {} unfused=DIFF({}) fused={}",
                    tag,
                    unfused,
                    df.as_deref().map(|x| format!("DIFF({})", x)).unwrap_or("MATCH".into())
                );
                // accepted only if the fused (arm64 oracle) policy reproduces it
                d = if df.is_none() { None } else { Some(format!("{} / fused: {}", unfused, df.unwrap())) };
            } else {
                println!("PARITY {} policy=unfused MATCH exit={}", tag, go.0);
            }
            assert!(d.is_none(), "{}: {}", tag, d.unwrap());
        }
    }
    rs
}

/// Build the shared verify-subspace fixture: base, a LOCKED merge (via heal),
/// an UNLOCKED merge (naive B A), and the vectors.
struct VerifyFixture {
    base: String,
    locked: String,
    leaky: String,
    vectors: String,
    target: String,
}

fn verify_fixture(sc: &Scratch) -> VerifyFixture {
    let mut rng = Rng(91);
    let mods = vec![module("base_model.model.model.layers.0.self_attn.o_proj", D, RANK, 24, &mut rng)];
    let lora = sc.path("adapter.safetensors");
    write(&lora, &lora_bytes(&mods, D, RANK, "F32"));
    let target = "model.layers.0.self_attn.o_proj.weight".to_string();
    let bt = base_tensors(&[(&target, 24), ("model.layers.0.mlp.down_proj.weight", 40)], D, "F32");
    let base = sc.path("base.safetensors");
    write(&base, &build_safetensors(&bt, &Default::default()).unwrap());
    let dirs = raw_dirs(D, 3, &mut rng);
    let vectors = vectors_dir(sc, "vec", &dirs, &mut rng);
    // locked merge through the Rust heal (identical to the oracle's by the heal scenarios)
    let locked = sc.path("locked.safetensors");
    let (_, r) = heal(&HealOptions {
        lora: lora.clone(),
        vectors: vectors.clone(),
        out: sc.path("healed-lora.safetensors"),
        base: base.clone(),
        merge_out: locked.clone(),
        scale: 1.0,
        tol: DEFAULT_SUBSPACE_TOL,
    });
    r.unwrap();
    // naive merge: W + B A with the ORIGINAL B, F32 payload
    let m = &mods[0];
    let w0: Vec<f64> = base_matrix(D, 24, 0.5).iter().map(|x| (*x as f32) as f64).collect();
    let mut w = w0.clone();
    for o in 0..D {
        for i in 0..24 {
            let mut dlt = 0f64;
            for k in 0..RANK {
                dlt += m.b[o * RANK + k] * m.a[k * 24 + i];
            }
            w[o * 24 + i] = w0[o * 24 + i] + dlt;
        }
    }
    let mut lt = bt.clone();
    lt[0].data = f32_bytes(&w);
    let leaky = sc.path("leaky.safetensors");
    write(&leaky, &build_safetensors(&lt, &Default::default()).unwrap());
    VerifyFixture { base, locked, leaky, vectors, target }
}

#[test]
fn verify_subspace_pass_leak_unknown_verdicts() {
    let _g = guard();
    let sc = Scratch::new("verify");
    let f = verify_fixture(&sc);
    let t = f.target.as_str();

    // pass: the locked merge is orthogonal to the protected subspace, non-vacuously
    let (code, rep) = check_verify("verify_pass", &f.base, &f.locked, &f.vectors, &[t], None);
    assert_eq!(code, 0, "{}", rep);
    assert_eq!(rep["verdict"], "pass");
    assert_eq!(rep["inputs_unchanged"], true);
    assert_eq!(rep["tensors"][0]["delta_is_zero"], false);

    // leak: the naive (unlocked) merge leaks along a protected direction
    let (code, rep) = check_verify("verify_leak", &f.base, &f.leaky, &f.vectors, &[t], None);
    assert_eq!(code, 1);
    assert_eq!(rep["verdict"], "leak");
    assert!(rep["tensors"][0]["max_leak"].as_f64().unwrap() > 0.05);

    // control: identical models pass and are flagged vacuous
    let (code, rep) = check_verify("verify_vacuous", &f.base, &f.base, &f.vectors, &[t], None);
    assert_eq!(code, 0);
    assert!(rep["tensors"][0]["reason"].as_str().unwrap().contains("vacuous"));

    // unknown: tensor missing from the models
    let (code, rep) = check_verify("verify_missing", &f.base, &f.locked, &f.vectors, &["no.such.weight"], None);
    assert_eq!(code, 2);
    assert_eq!(rep["verdict"], "unknown");

    // unknown: a 1-D tensor is not a 2-D float matrix
    let (code, _) = check_verify("verify_not2d", &f.base, &f.locked, &f.vectors, &["output.norm.weight"], None);
    assert_eq!(code, 2);

    // unchecked changed tensors are surfaced: the locked merge changed only `t`
    let (code, rep) = check_verify("verify_unchecked", &f.base, &f.leaky, &f.vectors, &["model.layers.0.mlp.down_proj.weight"], None);
    assert_eq!(code, 0, "{}", rep);
    assert_eq!(rep["changed_tensors_not_checked"][0], t);

    // tolerance: a huge tol turns the leaky merge into a pass
    let (code, _) = check_verify("verify_loose_tol", &f.base, &f.leaky, &f.vectors, &[t], Some(1e3));
    assert_eq!(code, 0);
}

#[test]
fn verify_subspace_dimension_mismatch_is_unknown() {
    let _g = guard();
    let sc = Scratch::new("verify-dim");
    let f = verify_fixture(&sc);
    let mut rng = Rng(5);
    let dirs = raw_dirs(12, 2, &mut rng);
    let v12 = vectors_dir(&sc, "vec12", &dirs, &mut rng);
    let (code, rep) = check_verify("verify_dim", &f.base, &f.locked, &v12, &[f.target.as_str()], None);
    assert_eq!(code, 2);
    assert_eq!(rep["verdict"], "unknown");
}

// ------------------------------------------------------------ determinism

#[test]
fn heal_is_deterministic_and_idempotent() {
    let _g = guard();
    let sc = Scratch::new("determinism");
    let mut rng = Rng(101);
    let mods = standard_mods(&mut rng);
    let lora = sc.path("adapter.safetensors");
    write(&lora, &lora_bytes(&mods, D, RANK, "F32"));
    let dirs = raw_dirs(D, 3, &mut rng);
    let vec_dir = vectors_dir(&sc, "vec", &dirs, &mut rng);
    let mut sums = Vec::new();
    for i in 0..2 {
        let out = sc.path(&format!("o{}.safetensors", i));
        let (rep, r) = heal(&HealOptions { lora: lora.clone(), vectors: vec_dir.clone(), out, ..Default::default() });
        r.unwrap();
        sums.push(rep.unwrap().out_sha256);
    }
    assert_eq!(sums[0], sums[1]);
    // idempotence: re-healing the healed LoRA leaks (before) at rounding level only
    let (rep, r) = heal(&HealOptions {
        lora: sc.path("o0.safetensors"),
        vectors: vec_dir.clone(),
        out: sc.path("again.safetensors"),
        ..Default::default()
    });
    r.unwrap();
    for p in rep.unwrap().pairs {
        assert!(p.leak_before < 1e-6, "{} leaks {:e} on re-heal", p.module, p.leak_before);
    }
    // never overwrites
    let (rep, r) =
        heal(&HealOptions { lora: lora.clone(), vectors: vec_dir.clone(), out: sc.path("o0.safetensors"), ..Default::default() });
    assert!(r.is_err());
    assert_eq!(rep.unwrap().status, "failed");
}

// ------------------------------------------------------------ C-ABI numeric entry

fn read_f32(path: &str, name: &str) -> Vec<f32> {
    let m = secant_core::container::Model::open(path).unwrap();
    let t = m.layout.tensor(name).unwrap().clone();
    assert_eq!(t.dtype, "F32");
    m.tensor_bytes(&t).chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// `project_then_heal_f32` must produce, bit for bit, the B that `heal` writes
/// for an F32 LoRA when the protected directions are f32-representable.
#[test]
fn project_then_heal_f32_equals_heal_output_bytes() {
    let _g = guard();
    let sc = Scratch::new("p2h");
    let mut rng = Rng(111);
    let mods = standard_mods(&mut rng);
    // lora_B values are stored as f32: round the fixture so the f32 view is the source of truth
    let lora = sc.path("adapter.safetensors");
    write(&lora, &lora_bytes(&mods, D, RANK, "F32"));
    let dirs: Vec<Vec<f64>> = raw_dirs(D, 3, &mut rng).iter().map(|v| v.iter().map(|x| (*x as f32) as f64).collect()).collect();
    let vec_dir = vectors_dir(&sc, "vec", &dirs, &mut rng);
    let out = sc.path("healed.safetensors");
    let (_, r) = heal(&HealOptions { lora: lora.clone(), vectors: vec_dir, out: out.clone(), ..Default::default() });
    r.unwrap();
    let flat: Vec<f32> = dirs.iter().flatten().map(|x| *x as f32).collect();
    for m in &mods {
        let name = format!("{}.lora_B.weight", m.stem);
        let mut b = read_f32(&lora, &name);
        let leak = project_then_heal_f32(&flat, 3, D, &mut b, RANK).unwrap();
        assert!(leak < 1e-6, "{} leak {:e}", m.stem, leak);
        let want = read_f32(&out, &name);
        assert_eq!(b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "{}", m.stem);
    }
}

// ------------------------------------------------------------ robustness sweep

/// Parity must not depend on the one geometry the other tests hand-picked:
/// vary d_out, rank, d_in, k, merge scale and the base dtype width.
#[test]
fn heal_merge_geometry_sweep() {
    let _g = guard();
    let cases: &[(usize, usize, usize, usize, f64, &str, u64)] = &[
        (8, 1, 8, 1, 1.0, "F32", 201),
        (32, 4, 48, 4, 0.25, "F32", 202),
        (24, 5, 40, 2, 3.0, "F64", 203),
        (64, 8, 96, 6, 0.125, "F32", 204),
        (12, 2, 20, 3, 1.0, "F16", 205), // F16 base: rounding fails the gate (exit 1) on both sides
    ];
    for (i, (d, rank, d_in, k, scale, dtype, seed)) in cases.iter().enumerate() {
        let sc = Scratch::new("sweep");
        let mut rng = Rng(*seed);
        let mods = vec![module("base_model.model.model.layers.0.self_attn.o_proj", *d, *rank, *d_in, &mut rng)];
        let lora = sc.path("adapter.safetensors");
        write(&lora, &lora_bytes(&mods, *d, *rank, "F32"));
        let base = sc.path("base.safetensors");
        let bt = base_tensors(&[("model.layers.0.self_attn.o_proj.weight", *d_in)], *d, dtype);
        write(&base, &build_safetensors(&bt, &Default::default()).unwrap());
        let dirs = raw_dirs(*d, *k, &mut rng);
        let vec_dir = vectors_dir(&sc, "vec", &dirs, &mut rng);
        let c = HealCase { lora: &lora, vectors: &vec_dir, base: &base, scale: *scale, tol: None };
        let r = check_heal(&format!("sweep{}_{}x{}x{}_k{}_{}", i, d, rank, d_in, k, dtype), &c, &sc).unwrap();
        let want = if *dtype == "F16" { 1 } else { 0 };
        assert_eq!(r.code, want, "case {}: {:?}", i, r.report["error"]);
    }
}
