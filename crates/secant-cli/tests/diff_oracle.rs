//! CLI-level differential conformance: `secant` vs the reference Go oracle
//! (`wsp` subcommands of the oracle binary at `SECANT_ORACLE_BIN`) on synthetic
//! SafeTensors and GGUF (Qwen3.5 `ssm_out`, F32 / BF16 / Q8_0) fixtures.
//!
//! Run: `SECANT_ORACLE_BIN=/path/to/oracle cargo test -p secant-cli --test diff_oracle -- --nocapture`
//! Without SECANT_ORACLE_BIN the test prints NOT_RUN and passes vacuously by
//! design (the oracle may be unavailable); the evidence table is only
//! meaningful when it prints `ORACLE`.
//!
//! Every scenario is executed against both arithmetic policies of the Rust
//! side: `portable` (unfused float64, identical on every host) and
//! `go-arm64-fma` (reproduces the fused multiply-adds the arm64 Go compiler
//! emits). The host-matching policy must be bit-exact; the other policy's
//! result is reported, not asserted.
//!
//! Plans and profiles fed to BOTH tools carry the legacy schema id (the only
//! one the oracle reads; Secant accepts it on input). The oracle's outputs
//! carry legacy schema ids too; `common::json_diff` maps them to the
//! `secant.wsp.*` ids before comparing and fails on any legacy id in Secant's
//! output.

#[path = "../../secant-core/tests/common/mod.rs"]
mod common;

use common::*;
use secant_core::builder::*;
use secant_core::namespace::legacy_alias;
use secant_core::plan::{PLAN_SCHEMA, PROFILE_SCHEMA};
use secant_core::quant::{quantize_row_q8_0, Q8_0_BLOCK_BYTES};
use serde_json::Value;
use std::collections::BTreeMap;
use std::process::{Command, Output};

const D_OUT: usize = 8;
const D_FFN: usize = 32;
const D_IN_SSM: usize = 512; // num_v 4 x head_v 128
const VOCAB: usize = 12;

fn secant(args: &[&str], fma: bool) -> Output {
    let mut a: Vec<&str> = Vec::new();
    if fma {
        a.push("--go-arm64-fma");
    }
    a.extend_from_slice(args);
    Command::new(env!("CARGO_BIN_EXE_secant")).args(&a).output().expect("spawn secant")
}

fn oracle(args: &[&str]) -> Output {
    let mut full = vec!["wsp"];
    full.extend_from_slice(args);
    Command::new(oracle_bin().unwrap()).args(&full).output().expect("spawn oracle")
}

struct Fx {
    dir: Scratch,
    st: String,
    gguf_f32: String,
    gguf_bf16: String,
    gguf_q8: String,
    vectors: String,
    profile: String,
}

fn i32_tensor(name: &str, vals: &[i32]) -> BuildTensor {
    BuildTensor { name: name.into(), dtype: "I32".into(), shape: vec![vals.len()], data: i32_bytes(vals) }
}

fn build_fixtures() -> Fx {
    let dir = Scratch::new("diff");
    let mut rng = Rng(0x5EC4_17);
    // --- SafeTensors (HF naming)
    let st_tensors = vec![
        tensor("lm_head.weight", "F32", &[VOCAB, D_OUT], &mut rng),
        tensor("model.language_model.layers.0.mlp.down_proj.weight", "F32", &[D_OUT, D_FFN], &mut rng),
        tensor("model.language_model.layers.1.mlp.down_proj.weight", "BF16", &[D_OUT, D_FFN], &mut rng),
        i32_tensor("model.layers.0.mlp.gate.tid2eid", &[1, 2, 3, 0, 1, 2, 3, 0]),
        tensor("model.layers.0.mlp.gate.weight", "F32", &[4, D_OUT], &mut rng),
        tensor("model.layers.0.mlp.e_score_correction_bias", "F32", &[4], &mut rng),
        tensor("model.norm.weight", "F32", &[D_OUT], &mut rng),
    ];
    let st = build_safetensors(&st_tensors, &BTreeMap::new()).unwrap();
    let st_path = dir.path("model.safetensors");
    write(&st_path, &st);

    // --- GGUF Qwen3.5-style
    let meta = qwen_meta(2, 4, 128, 128);
    let mk = |dt: &str, rng: &mut Rng| {
        vec![
            tensor("blk.0.ssm_out.weight", dt, &[D_OUT, D_IN_SSM], rng),
            tensor("blk.1.ssm_out.weight", dt, &[D_OUT, D_IN_SSM], rng),
            tensor("blk.0.ffn_down.weight", dt, &[D_OUT, D_FFN], rng),
            tensor("blk.1.ffn_down.weight", dt, &[D_OUT, D_FFN], rng),
            tensor("output.weight", "F32", &[VOCAB, D_OUT], rng),
            tensor("output_norm.weight", "F32", &[D_OUT], rng),
        ]
    };
    let f32_path = dir.path("qwen_f32.gguf");
    write(&f32_path, &build_gguf(&meta, &mk("F32", &mut rng), 32).unwrap());
    let bf16_path = dir.path("qwen_bf16.gguf");
    write(&bf16_path, &build_gguf(&meta, &mk("BF16", &mut rng), 32).unwrap());
    // Q8_0: quantize random float rows
    let q8 = |name: &str, d_in: usize, rng: &mut Rng| {
        let x: Vec<f32> = rng.vec(D_OUT * d_in).iter().map(|v| *v as f32).collect();
        let mut out = vec![0u8; D_OUT * d_in / 32 * Q8_0_BLOCK_BYTES];
        quantize_row_q8_0(&x, &mut out).unwrap();
        BuildTensor { name: name.into(), dtype: "Q8_0".into(), shape: vec![D_OUT, d_in], data: out }
    };
    let q8_tensors = vec![
        q8("blk.0.ssm_out.weight", D_IN_SSM, &mut rng),
        q8("blk.0.ffn_down.weight", D_FFN, &mut rng),
        tensor("output.weight", "F32", &[VOCAB, D_OUT], &mut rng),
    ];
    let q8_path = dir.path("qwen_q8.gguf");
    write(&q8_path, &build_gguf(&meta, &q8_tensors, 32).unwrap());

    // --- vectors
    let vdir = dir.path("vectors");
    std::fs::create_dir_all(&vdir).unwrap();
    for (stem, n) in [
        ("d_loop_L0", D_OUT),
        ("d_loop_L1", D_OUT),
        ("h_loop_L0", D_FFN),
        ("h_loop_L1", D_FFN),
        ("d_ref_L0", D_OUT),
        ("d_root_L0", D_OUT),
        ("h_root_L0", D_FFN),
    ] {
        let v = rng.vec(n);
        let bytes = if stem.starts_with("d_ref") {
            secant_core::subspace::npy_bytes_f32(&v.iter().map(|x| *x as f32).collect::<Vec<_>>(), &[n]).unwrap()
        } else {
            secant_core::subspace::npy_bytes_f64(&v, &[n]).unwrap()
        };
        write(&format!("{}/{}.npy", vdir, stem), &bytes);
    }
    // custom profile: rank1 + projector on the explicit ffn_down tensors, explicit layers
    let profile = dir.path("profile.json");
    let prof = serde_json::json!({
        "schema": legacy_alias(PROFILE_SCHEMA).unwrap(), "name": "diff-profile", "description": "differential test profile",
        "subspaces": [], "default_target_shift": -3.0, "default_gamma": 0.25,
        "layer_selection": {"strategy": "x", "tool": "x", "banked": "x"},
        "slots": [
          {"id":"R1","role":"loop-suppress","direction_token":"loop","op":"rank1-columns","tensor_pattern":"blk.%d.ffn_down.weight",
           "tensor_st_pattern":"model.language_model.layers.%d.mlp.down_proj.weight","layer":0,"sign":-1,"top_k":5,"requires_h_ref":true,"needs_direction":true},
          {"id":"P1","role":"refusal-ablate","direction_token":"ref","op":"projector-subtract","tensor_pattern":"blk.%d.ffn_down.weight",
           "tensor_st_pattern":"model.language_model.layers.%d.mlp.down_proj.weight","layer":0,"sign":-1,"needs_direction":true,"gamma":0.6}
        ]
    });
    write(&profile, serde_json::to_string_pretty(&prof).unwrap().as_bytes());
    Fx { dir, st: st_path, gguf_f32: f32_path, gguf_bf16: bf16_path, gguf_q8: q8_path, vectors: vdir, profile }
}

const IGNORE: &[&str] = &["path", "model", "base", "out", "patched", "plan_path", "base_model", "vector_files", "file"];

#[derive(Default)]
struct Table {
    rows: Vec<(String, &'static str, bool, String)>,
}

impl Table {
    fn add(&mut self, scenario: &str, variant: &'static str, ok: bool, detail: impl Into<String>) {
        self.rows.push((scenario.to_string(), variant, ok, detail.into()));
    }
}

fn variants() -> [(&'static str, bool); 2] {
    [("portable", false), ("go-arm64-fma", true)]
}

/// CPU architecture of the oracle binary, read from its Mach-O / ELF header
/// (not from the host: a Rosetta-run x86_64 oracle on an arm64 host is x86_64).
fn oracle_arch() -> &'static str {
    let b = std::fs::read(oracle_bin().unwrap()).expect("read oracle binary");
    if b.len() > 20 && b[..4] == [0xcf, 0xfa, 0xed, 0xfe] {
        return match u32::from_le_bytes([b[4], b[5], b[6], b[7]]) {
            0x0100_000c => "aarch64",
            0x0100_0007 => "x86_64",
            _ => "unknown",
        };
    }
    if b.len() > 20 && b[..4] == *b"\x7fELF" {
        return match u16::from_le_bytes([b[18], b[19]]) {
            0xB7 => "aarch64",
            0x3E => "x86_64",
            _ => "unknown",
        };
    }
    "unknown"
}

/// The Go compiler fuses `x*y+z` on arm64 but not on amd64 (GOAMD64<v3).
fn host_variant() -> &'static str {
    match oracle_arch() {
        "aarch64" => "go-arm64-fma",
        "x86_64" => "portable",
        other => panic!("cannot determine the oracle architecture ({}); refusing to pick an arithmetic policy", other),
    }
}

fn exit(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stdout_json(o: &Output) -> Result<Value, String> {
    serde_json::from_slice(&o.stdout)
        .map_err(|e| format!("not JSON ({}): {:?}", e, String::from_utf8_lossy(&o.stdout[..o.stdout.len().min(200)])))
}

/// Compare `secant <args> --json` with the oracle's `wsp <args> --json`.
fn cmp_cmd(t: &mut Table, name: &str, args: &[&str], ignore_extra: &[&str]) {
    let go = oracle(args);
    for (vname, fma) in variants() {
        let rs = secant(args, fma);
        let mut ignore: Vec<&str> = IGNORE.to_vec();
        ignore.extend_from_slice(ignore_extra);
        let (ok, detail) = if exit(&go) != exit(&rs) {
            (
                false,
                format!(
                    "exit codes differ: oracle {} secant {} (secant stderr: {})",
                    exit(&go),
                    exit(&rs),
                    String::from_utf8_lossy(&rs.stderr).trim()
                ),
            )
        } else if go.stdout.is_empty() && rs.stdout.is_empty() {
            (true, format!("exit {} (no JSON on either side)", exit(&go)))
        } else {
            match (stdout_json(&go), stdout_json(&rs)) {
                (Ok(a), Ok(b)) => match json_diff(&a, &b, &ignore) {
                    None => (true, format!("exit {} JSON bit-identical", exit(&go))),
                    Some(d) => (false, d),
                },
                (Err(e), _) => (false, format!("oracle: {}", e)),
                (_, Err(e)) => (false, format!("secant: {}", e)),
            }
        };
        t.add(name, vname, ok, detail);
    }
}

fn file_json(path: &str) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn differential_conformance() {
    if oracle_bin().is_none() {
        not_run("differential_conformance");
        return;
    }
    let fx = build_fixtures();
    let mut t = Table::default();
    let mf = |n: &str| fx.dir.path(n);

    // ---- inspect
    for (n, p) in [("st", &fx.st), ("gguf_f32", &fx.gguf_f32), ("gguf_bf16", &fx.gguf_bf16)] {
        cmp_cmd(&mut t, &format!("inspect/{}", n), &["inspect", "--model", p, "--json"], &[]);
    }
    // Q8_0: the oracle rejects the whole file; Secant lists it. Documented divergence.
    {
        let go = oracle(&["inspect", "--model", &fx.gguf_q8, "--json"]);
        let rs = secant(&["inspect", "--model", &fx.gguf_q8, "--json"], false);
        let ok = exit(&go) == 2 && exit(&rs) == 0;
        t.add(
            "inspect/gguf_q8 (divergence: oracle rejects quantized files)",
            "portable",
            ok,
            format!("oracle exit {} secant exit {}", exit(&go), exit(&rs)),
        );
    }

    // ---- read
    for (n, p, extra) in [
        ("st", &fx.st, vec!["--vocab-tensor", "lm_head.weight"]),
        ("gguf_f32", &fx.gguf_f32, vec!["--vocab-tensor", "output.weight"]),
        ("gguf_bf16", &fx.gguf_bf16, vec!["--vocab-tensor", "output.weight"]),
    ] {
        let mut args = vec!["read", "--model", p.as_str(), "--detect-anomalies", "--top-cols", "5", "--top-k-singular", "3", "--json"];
        args.extend(extra.iter().copied());
        cmp_cmd(&mut t, &format!("read/{}", n), &args, &[]);
    }
    // read --vectors-out: identical array values and 64-byte aligned headers (bit-exact).
    {
        let (a, b) = (mf("vo_go"), mf("vo_rs"));
        let go = oracle(&["read", "--model", &fx.gguf_f32, "--tensor", "blk.0.ffn_down.weight", "--vectors-out", &a, "--json"]);
        let rs = secant(&["read", "--model", &fx.gguf_f32, "--tensor", "blk.0.ffn_down.weight", "--vectors-out", &b, "--json"], false);
        let mut same = exit(&go) == 0 && exit(&rs) == 0;
        let mut detail = String::new();
        for f in ["u1_blk.0.ffn_down.weight.npy", "v1_blk.0.ffn_down.weight.npy"] {
            let (x, y) = (std::fs::read(format!("{}/{}", a, f)).unwrap(), std::fs::read(format!("{}/{}", b, f)).unwrap());
            let hl = |v: &Vec<u8>| 10 + u16::from_le_bytes([v[8], v[9]]) as usize;
            let (hx, hy) = (hl(&x), hl(&y));
            let eq = x == y && hx % 64 == 0 && hy % 64 == 0;
            detail.push_str(&format!(
                "{}: byte-identical, oracle header {} (%64={}), secant header {} (%64={}) ok={} ",
                f,
                hx,
                hx % 64,
                hy,
                hy % 64,
                eq
            ));
            same &= eq;
        }
        t.add("read/vectors-out npy (byte-identical header & data)", "portable", same, detail);
    }
    // Q8_0 read: oracle rejects; Secant analyses the dequantized matrix
    {
        let rs = secant(&["read", "--model", &fx.gguf_q8, "--tensor", "blk.0.ffn_down.weight", "--json"], false);
        let v = stdout_json(&rs);
        let ok = exit(&rs) == 0 && v.as_ref().map(|v| v["tensors"][0]["status"] == "ok").unwrap_or(false);
        t.add("read/gguf_q8 (extension: dequantized spectrum)", "portable", ok, format!("exit {}", exit(&rs)));
    }

    // ---- F16 subnormals: both oracle and Secant decode IEEE-correctly (2^-24 ULP).
    {
        let bits: [u16; 4] = [0x0001, 0x0002, 0x0200, 0x03FF];
        let data: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let t16 = BuildTensor { name: "h".into(), dtype: "F16".into(), shape: vec![2, 2], data };
        let p = mf("f16_subnormal.safetensors");
        write(&p, &build_safetensors(&[t16], &BTreeMap::new()).unwrap());
        let go = oracle(&["read", "--model", &p, "--json"]);
        let rs = secant(&["read", "--model", &p, "--json"], false);
        let fro = |o: &Output| stdout_json(o).map(|v| v["tensors"][0]["frobenius"].as_f64().unwrap()).unwrap_or(f64::NAN);
        let (g, r) = (fro(&go), fro(&rs));
        // exact expectation: sqrt(sum of squares of the true half-precision subnormals)
        let truth: f64 = bits.iter().map(|b| (*b as f64) * 2f64.powi(-24)).map(|x| x * x).sum::<f64>().sqrt();
        let ok = exit(&go) == 0 && exit(&rs) == 0 && (r - truth).abs() <= truth * 1e-15 && (g - r).abs() <= r * 1e-15;
        t.add(
            "read/f16 subnormal (IEEE-754 parity)",
            "portable",
            ok,
            format!("oracle fro {:e}, secant fro {:e}, IEEE truth {:e}", g, r, truth),
        );
    }

    // ---- rank-deficient matrix: deflation re-finds sigma_1 and flags it non-converged (oracle behaviour)
    {
        let (u1, u2) = ([0.5f64, 0.5, 0.5, 0.5], [0.5f64, -0.5, 0.5, -0.5]);
        let (v1, v2) = ([0.6f64, 0.8, 0.0], [-0.8f64, 0.6, 0.0]);
        let mut m = vec![0f64; 12];
        for r in 0..4 {
            for c in 0..3 {
                m[r * 3 + c] = 4.0 * u1[r] * v1[c] + 1.5 * u2[r] * v2[c];
            }
        }
        let t64 = BuildTensor { name: "rd".into(), dtype: "F64".into(), shape: vec![4, 3], data: f64_bytes(&m) };
        let p = mf("rank_deficient.safetensors");
        write(&p, &build_safetensors(&[t64], &BTreeMap::new()).unwrap());
        cmp_cmd(
            &mut t,
            "read/rank-deficient F64 (spurious non-converged triplet)",
            &["read", "--model", &p, "--top-k-singular", "3", "--json"],
            &[],
        );
    }

    // ---- compose (explicit profile) and builtin profiles
    struct C<'a> {
        name: &'a str,
        profile: &'a str,
        vectors: Option<&'a str>,
        extra: Vec<&'a str>,
    }
    let cases = vec![
        C { name: "compose/custom target-shift default", profile: &fx.profile, vectors: Some(&fx.vectors), extra: vec![] },
        C {
            name: "compose/custom --target-shift -7",
            profile: &fx.profile,
            vectors: Some(&fx.vectors),
            extra: vec!["--target-shift", "-7"],
        },
        C {
            name: "compose/custom --gamma 0.2 --base gguf_f32",
            profile: &fx.profile,
            vectors: Some(&fx.vectors),
            extra: vec!["--gamma", "0.2", "--base", &fx.gguf_f32],
        },
        C {
            name: "compose/builtin security-auditor-b1b2b3",
            profile: "security-auditor-b1b2b3",
            vectors: Some(&fx.vectors),
            extra: vec![],
        },
        C {
            name: "compose/builtin adversarial-nan-probe (no calibration: identical usage error)",
            profile: "adversarial-nan-probe",
            vectors: None,
            extra: vec![],
        },
        C {
            name: "compose/builtin adversarial-nan-probe --gamma 0.1 (set-value NaN/+Inf rows)",
            profile: "adversarial-nan-probe",
            vectors: None,
            extra: vec!["--gamma", "0.1"],
        },
        C {
            name: "compose/builtin adversarial-moe-router-probe --gamma 0.1 (set-value ints/-Inf/zero)",
            profile: "adversarial-moe-router-probe",
            vectors: None,
            extra: vec!["--gamma", "0.1"],
        },
    ];
    let mut plans: Vec<(String, String, String)> = Vec::new(); // (name, go_plan, rs_plan_portable)
    for (ci, c) in cases.iter().enumerate() {
        let go_plan = mf(&format!("go_plan_{}.json", ci));
        let mut gargs = vec!["compose", "--profile", c.profile, "--out", &go_plan, "--json"];
        if let Some(v) = c.vectors {
            gargs.extend(["--vectors", v]);
        }
        gargs.extend(c.extra.iter().copied());
        let go = oracle(&gargs);
        let mut portable_plan = String::new();
        for (vname, fma) in variants() {
            let rs_plan = mf(&format!("rs_plan_{}_{}.json", ci, vname));
            let mut rargs = vec!["compose", "--profile", c.profile, "--out", &rs_plan, "--json"];
            if let Some(v) = c.vectors {
                rargs.extend(["--vectors", v]);
            }
            rargs.extend(c.extra.iter().copied());
            let rs = secant(&rargs, fma);
            let mut ok = exit(&go) == exit(&rs);
            let mut detail = format!("exit oracle {} secant {}", exit(&go), exit(&rs));
            if ok && exit(&go) == 0 {
                match json_diff(&file_json(&go_plan), &file_json(&rs_plan), &[]) {
                    None => detail.push_str("; plan file bit-identical"),
                    Some(d) => {
                        ok = false;
                        detail = format!("plan differs: {}", d);
                    }
                }
                if ok {
                    if let Some(d) = json_diff(&stdout_json(&go).unwrap(), &stdout_json(&rs).unwrap(), IGNORE) {
                        ok = false;
                        detail = format!("compose --json differs: {}", d);
                    }
                }
            } else if !ok {
                detail.push_str(&format!(" ({})", String::from_utf8_lossy(&rs.stderr).trim()));
            }
            t.add(c.name, vname, ok, detail);
            if vname == "portable" {
                portable_plan = rs_plan.clone();
            }
        }
        if exit(&go) == 0 {
            plans.push((c.name.to_string(), go_plan, portable_plan));
        }
    }

    // ---- patch: same plan through both tools, plus cross-applied plans; compare bytes
    let targets: Vec<(&str, &String)> = vec![("st", &fx.st), ("gguf_f32", &fx.gguf_f32), ("gguf_bf16", &fx.gguf_bf16)];
    for (pname, go_plan, _rs_plan) in &plans {
        for (tname, base) in &targets {
            let go_out = mf(&format!("go_{}_{}.bin", tname, plans.iter().position(|p| &p.0 == pname).unwrap()));
            let _ = std::fs::remove_file(&go_out);
            let go = oracle(&["patch", "--base", base, "--plan", go_plan, "--out", &go_out, "--json"]);
            for (vname, fma) in variants() {
                let rs_out = mf(&format!("rs_{}_{}_{}.bin", tname, vname, plans.iter().position(|p| &p.0 == pname).unwrap()));
                let _ = std::fs::remove_file(&rs_out);
                let rs = secant(&["patch", "--base", base, "--plan", go_plan, "--out", &rs_out, "--json"], fma);
                let mut ok = exit(&go) == exit(&rs);
                let mut detail = format!("exit oracle {} secant {}", exit(&go), exit(&rs));
                if ok && exit(&go) == 0 {
                    let (h1, h2) = (sha256_file(&go_out), sha256_file(&rs_out));
                    ok = h1 == h2;
                    detail.push_str(&format!("; output sha256 {} {}", if ok { "identical" } else { "DIFFER" }, &h1[..12]));
                    if ok {
                        if let Some(d) = json_diff(&stdout_json(&go).unwrap(), &stdout_json(&rs).unwrap(), IGNORE) {
                            ok = false;
                            detail = format!("patch report differs: {}", d);
                        }
                    }
                } else if !ok {
                    detail.push_str(&format!(" ({})", String::from_utf8_lossy(&rs.stderr).trim()));
                }
                t.add(&format!("patch/{} <- {}", tname, pname), vname, ok, detail);
            }
        }
    }

    // ---- hand-written plans: qwen35 ssm_out layout mapping and permute-columns round trip
    {
        let dir8 = vec![0.5, -0.25, 0.125, 1.0, -1.0, 0.75, 0.0, 0.3];
        let plan = serde_json::json!({
        "schema": legacy_alias(PLAN_SCHEMA).unwrap(), "profile": "hand", "calibration": {"mode": "gamma"},
        "edits": [
          {"id":"S1","role":"x","op":"rank1-columns","tensor":"blk.0.ssm_out.weight","layout":"qwen35-ssm-out-128",
           "direction": dir8, "cols":[0,1,130,257,511], "coeffs":[0.5,-0.25,1.5,0.125,2.0]},
          {"id":"S2","role":"x","op":"permute-columns","tensor":"blk.1.ssm_out.weight","permutation":"qwen35-ssm-out-128-hf-to-gguf"}
        ]});
        let pp = mf("hand_plan.json");
        write(&pp, serde_json::to_string_pretty(&plan).unwrap().as_bytes());
        for (tname, base) in [("gguf_f32", &fx.gguf_f32), ("gguf_bf16", &fx.gguf_bf16)] {
            let go_out = mf(&format!("hand_go_{}.bin", tname));
            let go = oracle(&["patch", "--base", base, "--plan", &pp, "--out", &go_out, "--json"]);
            for (vname, fma) in variants() {
                let rs_out = mf(&format!("hand_rs_{}_{}.bin", tname, vname));
                let rs = secant(&["patch", "--base", base, "--plan", &pp, "--out", &rs_out, "--json"], fma);
                let ok = exit(&go) == 0 && exit(&rs) == 0 && sha256_file(&go_out) == sha256_file(&rs_out);
                t.add(
                    &format!("patch/{} ssm_out layout+permute (hand plan)", tname),
                    vname,
                    ok,
                    format!("exit oracle {} secant {}", exit(&go), exit(&rs)),
                );
            }
            // verify-patch parity on the produced file
            cmp_cmd(
                &mut t,
                &format!("verify-patch/{} ok", tname),
                &[
                    "verify-patch",
                    "--base",
                    base,
                    "--patched",
                    &go_out,
                    "--expect-changed",
                    "blk.0.ssm_out.weight,blk.1.ssm_out.weight",
                    "--json",
                ],
                &[],
            );
            cmp_cmd(
                &mut t,
                &format!("verify-patch/{} unexpected-change", tname),
                &["verify-patch", "--base", base, "--patched", &go_out, "--expect-changed", "blk.0.ssm_out.weight", "--json"],
                &[],
            );
        }
    }

    // ---- report
    println!("\nORACLE {} arch={} (policy asserted: {})", oracle_bin().unwrap(), oracle_arch(), host_variant());
    println!("{:<4} {:<14} {:<62} detail", "ok", "policy", "scenario");
    let mut host_fail = Vec::new();
    for (s, v, ok, d) in &t.rows {
        // DIFF = a mismatch under the policy that is NOT asserted for this oracle's architecture
        // (informational: it is the expected last-bit fusion difference, see the file header).
        let tag = if *ok {
            "PASS"
        } else if *v == host_variant() {
            "FAIL"
        } else {
            "DIFF"
        };
        println!("{:<4} {:<14} {:<62} {}", tag, v, s, d);
        if !*ok && *v == host_variant() {
            host_fail.push(format!("{} [{}]: {}", s, v, d));
        }
    }
    let pass_by: BTreeMap<&str, usize> = variants().iter().map(|(v, _)| (*v, t.rows.iter().filter(|r| r.1 == *v && r.2).count())).collect();
    let total_by: BTreeMap<&str, usize> = variants().iter().map(|(v, _)| (*v, t.rows.iter().filter(|r| r.1 == *v).count())).collect();
    println!("SUMMARY pass/total by policy: {:?} / {:?}", pass_by, total_by);
    assert!(host_fail.is_empty(), "host-policy mismatches vs oracle:\n{}", host_fail.join("\n"));
}

/// Optional case count: `SECANT_DIFF_CASES` absent => 40; present-but-empty => error.
fn sweep_cases() -> usize {
    match std::env::var("SECANT_DIFF_CASES") {
        Err(std::env::VarError::NotPresent) => 40,
        Ok(v) if v.is_empty() => panic!("SECANT_DIFF_CASES is present but empty"),
        Ok(v) => v.parse().expect("SECANT_DIFF_CASES must be an integer"),
        Err(e) => panic!("SECANT_DIFF_CASES: {}", e),
    }
}

/// Randomized geometry sweep: shapes, dtypes, directions, columns, gammas and
/// calibration targets we did not hand-pick. Rank-1, projector and compose are
/// compared against the oracle under the oracle-architecture-matching policy;
/// output files must be byte-identical.
#[test]
fn random_geometry_sweep() {
    if oracle_bin().is_none() {
        not_run("random_geometry_sweep");
        return;
    }
    let fma = host_variant() == "go-arm64-fma";
    let s = Scratch::new("sweep");
    let mut rng = Rng(0xA11CE);
    let n = sweep_cases();
    let mut failures = Vec::new();
    let mut by_dtype: BTreeMap<String, usize> = BTreeMap::new();
    for case in 0..n {
        let d_out = 2 + (rng.next_u64() % 23) as usize;
        let d_in = 3 + (rng.next_u64() % 61) as usize;
        let dtype = ["F32", "BF16", "F16", "F64"][(rng.next_u64() % 4) as usize];
        let gguf = rng.next_u64() % 2 == 0;
        *by_dtype.entry(format!("{}/{}", if gguf { "gguf" } else { "st" }, dtype)).or_insert(0) += 1;
        let name = if gguf { "blk.0.ffn_down.weight" } else { "model.language_model.layers.0.mlp.down_proj.weight" };
        let tensors = vec![tensor(name, dtype, &[d_out, d_in], &mut rng), tensor("other.weight", "F32", &[3, 3], &mut rng)];
        let base = s.path(&format!("m{}.{}", case, if gguf { "gguf" } else { "safetensors" }));
        let bytes = if gguf { build_gguf(&[], &tensors, 32).unwrap() } else { build_safetensors(&tensors, &BTreeMap::new()).unwrap() };
        write(&base, &bytes);
        // random plan: one rank-1 edit and (sometimes) a projector on the same tensor
        let norm = |v: Vec<f64>| {
            let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.into_iter().map(|x| x / n).collect::<Vec<f64>>()
        };
        let k = 1 + (rng.next_u64() % d_in as u64) as usize;
        let cols: Vec<u64> = (0..k).map(|_| rng.next_u64() % d_in as u64).collect();
        let mut edits = vec![serde_json::json!({
            "id": "R", "role": "x", "op": "rank1-columns", "tensor": name, "tensor_safetensors": name,
            "direction": norm(rng.vec(d_out)), "cols": cols, "coeffs": rng.vec(k)})];
        if rng.next_u64() % 2 == 0 {
            edits.push(serde_json::json!({
                "id": "P", "role": "x", "op": "projector-subtract", "tensor": name, "tensor_safetensors": name,
                "direction": norm(rng.vec(d_out)), "gamma": 0.1 + 0.9 * (rng.unit().abs())}));
        }
        let plan = serde_json::json!({"schema": legacy_alias(PLAN_SCHEMA).unwrap(), "profile": "sweep", "calibration": {"mode": "gamma"}, "edits": edits});
        let pp = s.path(&format!("plan{}.json", case));
        write(&pp, plan.to_string().as_bytes());
        let (go_out, rs_out) = (s.path(&format!("go{}.bin", case)), s.path(&format!("rs{}.bin", case)));
        let go = oracle(&["patch", "--base", &base, "--plan", &pp, "--out", &go_out, "--json"]);
        let rs = secant(&["patch", "--base", &base, "--plan", &pp, "--out", &rs_out, "--json"], fma);
        let ok = exit(&go) == 0 && exit(&rs) == 0 && sha256_file(&go_out) == sha256_file(&rs_out);
        if !ok {
            failures.push(format!(
                "case {} {}x{} {} gguf={}: oracle exit {} secant exit {} (secant: {})",
                case,
                d_out,
                d_in,
                dtype,
                gguf,
                exit(&go),
                exit(&rs),
                String::from_utf8_lossy(&rs.stderr).trim()
            ));
            continue;
        }
        if let Some(d) = json_diff(&stdout_json(&go).unwrap(), &stdout_json(&rs).unwrap(), IGNORE) {
            failures.push(format!("case {}: patch report differs: {}", case, d));
        }
        // read the patched output with both tools
        let go_r = oracle(&["read", "--model", &go_out, "--tensor", name, "--top-k-singular", "2", "--detect-anomalies", "--json"]);
        let rs_r = secant(&["read", "--model", &rs_out, "--tensor", name, "--top-k-singular", "2", "--detect-anomalies", "--json"], fma);
        match (stdout_json(&go_r), stdout_json(&rs_r)) {
            (Ok(a), Ok(b)) => {
                if let Some(d) = json_diff(&a, &b, IGNORE) {
                    failures.push(format!("case {}: read differs: {}", case, d));
                }
            }
            _ => failures.push(format!("case {}: read produced no JSON (exit {} / {})", case, exit(&go_r), exit(&rs_r))),
        }
    }
    println!("random sweep: {} cases, {} failures; coverage {:?}", n, failures.len(), by_dtype);
    assert!(failures.is_empty(), "sweep mismatches vs oracle ({}):\n{}", host_variant(), failures.join("\n"));
}
