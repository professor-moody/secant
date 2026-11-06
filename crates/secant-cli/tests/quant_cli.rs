//! End-to-end tests of `secant quant` on synthetic Qwen3.5-style GGUF files:
//! block-aligned Q8_0 surgery (only blocks holding an edited element are
//! rewritten), the commutation witness, and the Pi_128 panel-equivariance
//! check for Q8_0 (must commute) and Q4_K (must not).

#[path = "../../secant-core/tests/common/mod.rs"]
mod common;

use common::*;
use secant_core::builder::*;
use secant_core::namespace::legacy_alias;
use secant_core::plan::PLAN_SCHEMA;
use secant_core::quant::*;
use secant_core::subspace::npy_bytes_f64;
use serde_json::Value;
use std::process::Command;

const D_OUT: usize = 8;
const D_IN: usize = 512;

fn secant(args: &[&str]) -> (i32, Value, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_secant")).args(args).output().unwrap();
    let v = serde_json::from_slice(&o.stdout).unwrap_or(Value::Null);
    (o.status.code().unwrap_or(-1), v, String::from_utf8_lossy(&o.stderr).into_owned())
}

fn quantized_tensor(name: &str, fmt: QuantFormat, x: &[f32], d_out: usize, d_in: usize) -> BuildTensor {
    let (bytes, _) = roundtrip_matrix(fmt, x, d_out, d_in).unwrap();
    BuildTensor { name: name.into(), dtype: fmt.name().into(), shape: vec![d_out, d_in], data: bytes }
}

fn fixture(scratch: &Scratch, fmt: QuantFormat, tag: &str) -> (String, Vec<f32>) {
    let mut rng = Rng(77);
    let x: Vec<f32> = rng.vec(D_OUT * D_IN).iter().map(|v| *v as f32).collect();
    let tensors = vec![
        quantized_tensor("blk.0.ssm_out.weight", fmt, &x, D_OUT, D_IN),
        quantized_tensor("blk.1.ssm_out.weight", fmt, &x.iter().map(|v| v * 0.5).collect::<Vec<_>>(), D_OUT, D_IN),
        tensor("output.weight", "F32", &[12, D_OUT], &mut rng),
    ];
    let p = scratch.path(&format!("{}.gguf", tag));
    write(&p, &build_gguf(&qwen_meta(2, 4, 128, 128), &tensors, 32).unwrap());
    (p, x)
}

#[test]
fn q8_0_surgery_rewrites_only_touched_blocks() {
    let s = Scratch::new("q8patch");
    let (base, _) = fixture(&s, QuantFormat::Q8_0, "q8");
    let mut rng = Rng(5);
    let dir: Vec<f64> = rng.vec(D_OUT);
    // rank-1 on two HF columns (mapped through the interleave) of layer 0 only
    let plan = serde_json::json!({
      "schema": PLAN_SCHEMA, "profile": "q8", "calibration": {"mode": "gamma"},
      "edits": [{"id":"E1","role":"x","op":"rank1-columns","tensor":"blk.0.ssm_out.weight","layout":"qwen35-ssm-out-128",
                 "direction": dir, "cols":[3, 300], "coeffs":[0.5, -0.75]}]});
    let pp = s.path("plan.json");
    write(&pp, serde_json::to_string(&plan).unwrap().as_bytes());
    let out = s.path("out.gguf");
    let (code, v, err) = secant(&["quant", "patch", "--base", &base, "--plan", &pp, "--out", &out, "--json"]);
    assert_eq!(code, 0, "{}", err);
    let e = &v["edits"][0];
    // 2 columns hit 2 distinct blocks in each of the 8 rows
    assert_eq!(e["blocks_touched"], 16);
    assert!(e["blocks_rewritten"].as_u64().unwrap() <= 16 && e["blocks_rewritten"].as_u64().unwrap() > 0);
    assert_eq!(e["blocks_total"], D_OUT * D_IN / 32);
    assert_eq!(v["verify"]["ok"], true);
    // header + every other tensor byte-identical, only blk.0.ssm_out changed
    let (c2, vp, err2) =
        secant(&["verify-patch", "--base", &base, "--patched", &out, "--expect-changed", "blk.0.ssm_out.weight", "--json"]);
    assert_eq!(c2, 0, "{}", err2);
    assert_eq!(vp["changed_tensors"], serde_json::json!(["blk.0.ssm_out.weight"]));
    // untouched blocks of the edited tensor are byte-identical
    let (a, b) = (std::fs::read(&base).unwrap(), std::fs::read(&out).unwrap());
    let m = secant_core::container::Model::open(&base).unwrap();
    let t = m.layout.tensor("blk.0.ssm_out.weight").unwrap();
    let (lo, bpr) = (t.offset as usize, D_IN / 32);
    let lay = secant_core::container::SsmOutLayout::from_meta(&m.layout.metadata).unwrap();
    let p = secant_core::container::SsmOutPermutation::new(D_IN, &lay).unwrap();
    let hit: Vec<usize> = [3usize, 300].iter().map(|c| p.forward_in_col(*c).unwrap() / 32).collect();
    let mut changed_blocks = 0;
    for row in 0..D_OUT {
        for blk in 0..bpr {
            let off = lo + (row * bpr + blk) * Q8_0_BLOCK_BYTES;
            let same = a[off..off + Q8_0_BLOCK_BYTES] == b[off..off + Q8_0_BLOCK_BYTES];
            if !hit.contains(&blk) {
                assert!(same, "untouched block row {} blk {} was rewritten", row, blk);
            } else if !same {
                changed_blocks += 1;
            }
        }
    }
    assert_eq!(changed_blocks as u64, e["blocks_rewritten"].as_u64().unwrap());
    // requantization error is bounded by half a step of the touched block scale + fp16 slack
    assert!(e["max_requant_error"].as_f64().unwrap() < 0.05);
    // in-place mode on a copy gives the same bytes
    let ip = s.path("inplace.gguf");
    std::fs::copy(&base, &ip).unwrap();
    let (c3, v3, err3) = secant(&["quant", "patch", "--base", &ip, "--plan", &pp, "--in-place", "--json"]);
    assert_eq!(c3, 0, "{}", err3);
    assert_eq!(v3["verify"]["ok"], true);
    assert_eq!(sha256_file(&ip), sha256_file(&out));
}

#[test]
fn q8_0_projector_leaves_other_tensors_untouched_and_witness_reports_leakage() {
    let s = Scratch::new("q8proj");
    let (base, _) = fixture(&s, QuantFormat::Q8_0, "q8");
    let mut rng = Rng(9);
    let mut d = rng.vec(D_OUT);
    let n = d.iter().map(|x| x * x).sum::<f64>().sqrt();
    d.iter_mut().for_each(|x| *x /= n);
    // the legacy plan id is still accepted on input (files written by older tools)
    let plan = serde_json::json!({
      "schema": legacy_alias(PLAN_SCHEMA).unwrap(), "profile": "q8p", "calibration": {"mode": "gamma"},
      "edits": [{"id":"P1","role":"x","op":"projector-subtract","tensor":"blk.0.ssm_out.weight","direction": d, "gamma": 1.0}]});
    let pp = s.path("plan.json");
    write(&pp, serde_json::to_string(&plan).unwrap().as_bytes());
    let out = s.path("out.gguf");
    let (code, v, err) = secant(&["quant", "patch", "--base", &base, "--plan", &pp, "--out", &out, "--json"]);
    assert_eq!(code, 0, "{}", err);
    assert_eq!(v["edits"][0]["blocks_touched"], D_OUT * D_IN / 32);
    // witness on the patched tensor: D^T W' is ~0 after the float edit but the
    // requantized tensor re-introduces leakage of the order of the quant error.
    let dpath = s.path("d.npy");
    write(&dpath, &npy_bytes_f64(&d, &[D_OUT]).unwrap());
    let (cw, w, errw) = secant(&[
        "quant",
        "witness",
        "--model",
        &out,
        "--tensor",
        "blk.0.ssm_out.weight",
        "--vectors",
        &dpath,
        "--format",
        "q8_0",
        "--json",
    ]);
    assert_eq!(cw, 0, "{}", errw);
    let l2 = w["witness_l2"].as_f64().unwrap();
    let qerr = w["quant_error_fro"].as_f64().unwrap();
    assert!(l2 >= 0.0 && l2 <= qerr + 1e-9, "||D^T E|| <= ||E|| must hold (got {} vs {})", l2, qerr);
    // stored tensor is already a Q8_0 fixed point of dequant: requantizing it changes little
    assert!(qerr < 0.5);
}

#[test]
fn pi128_panel_equivariance_q8_commutes_q4k_does_not() {
    let s = Scratch::new("pi128");
    let mut rng = Rng(21);
    let d = rng.vec(D_OUT);
    let dpath = s.path("d.npy");
    write(&dpath, &npy_bytes_f64(&d, &[D_OUT]).unwrap());
    for (fmt, flag, commutes) in [(QuantFormat::Q8_0, "q8_0", true), (QuantFormat::Q4K, "q4_k", false)] {
        let (p, _) = fixture(&s, fmt, flag);
        let (code, v, err) = secant(&[
            "quant",
            "witness",
            "--model",
            &p,
            "--tensor",
            "blk.0.ssm_out.weight",
            "--vectors",
            &dpath,
            "--format",
            flag,
            "--pi128",
            "--json",
        ]);
        assert_eq!(code, 0, "{}", err);
        let eq = &v["pi128"];
        assert_eq!(eq["equivariant"], commutes, "{} equivariance: {}", flag, eq);
        if commutes {
            assert_eq!(eq["bytes_equal"], true);
            assert_eq!(eq["block_aligned"], true);
        } else {
            assert_eq!(eq["block_aligned"], false);
            assert!(eq["max_abs_dequant_diff"].as_f64().unwrap() > 0.0);
        }
    }
}
