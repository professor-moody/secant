//! `--reference <base>` on `secant read` / `secant audit-trojan` (Horizon 2):
//! statistics, spectrum and the u_1 vocabulary projection are of
//! `Delta W = W - W_ref`. The fixture is the single-file blind spot this closes:
//! the base already carries a strong dominant direction, so a small planted
//! rank-1 edit is NOT the leading singular direction of the edited file and
//! its tokens are invisible to a static read; against the base it is exactly
//! the leading direction and its planted tokens come out first.

#[path = "../../secant-core/tests/common/mod.rs"]
mod common;

use common::*;
use secant_core::builder::*;
use secant_core::quant::{roundtrip_matrix, QuantFormat};
use serde_json::Value;
use std::collections::BTreeMap;
use std::process::Command;

const D_OUT: usize = 64;
const D_IN: usize = 128;
const VOCAB: usize = 50;
const PLANTED: [usize; 2] = [7, 23];
const THETA_BASE: f64 = 3.0;
const THETA_EDIT: f64 = 1.0;

fn secant(args: &[&str]) -> (i32, Value, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_secant")).args(args).output().unwrap();
    let v = serde_json::from_slice(&o.stdout).unwrap_or(Value::Null);
    (o.status.code().unwrap_or(-1), v, String::from_utf8_lossy(&o.stdout).into_owned() + &String::from_utf8_lossy(&o.stderr))
}

fn noise(n: usize, seed: u64, amp: f64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 52) as f64 - 1.0) * amp
        })
        .collect()
}

fn unit_mod(n: usize, residue: usize) -> Vec<f64> {
    let mut u: Vec<f64> = (0..n).map(|i| if i % 4 == residue { 1.0 } else { 0.0 }).collect();
    let k = u.iter().map(|x| x * x).sum::<f64>().sqrt();
    u.iter_mut().for_each(|x| *x /= k);
    u
}

fn add_rank1(w: &mut [f64], theta: f64, u: &[f64], v: &[f64]) {
    for i in 0..D_OUT {
        for j in 0..D_IN {
            w[i * D_IN + j] += theta * u[i] * v[j];
        }
    }
}

/// (base, edited, unembedding). Base direction is orthogonal to the edit's.
fn fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut base = noise(D_OUT * D_IN, 3, 0.002);
    let v_base: Vec<f64> = (0..D_IN).map(|j| if j % 2 == 0 { 1.0 / ((D_IN / 2) as f64).sqrt() } else { 0.0 }).collect();
    add_rank1(&mut base, THETA_BASE, &unit_mod(D_OUT, 3), &v_base);
    let mut edited = base.clone();
    let v_edit: Vec<f64> = (0..D_IN).map(|j| if j % 2 == 1 && j < 32 { 0.25 } else { 0.0 }).collect();
    let u_edit = unit_mod(D_OUT, 1);
    add_rank1(&mut edited, THETA_EDIT, &u_edit, &v_edit);
    let mut e = noise(VOCAB * D_OUT, 9, 0.01);
    for (k, &row) in PLANTED.iter().enumerate() {
        for i in 0..D_OUT {
            e[row * D_OUT + i] = u_edit[i] * (5.0 - k as f64);
        }
    }
    (base, edited, e)
}

fn f32t(name: &str, shape: &[usize], x: &[f64]) -> BuildTensor {
    BuildTensor { name: name.into(), dtype: "F32".into(), shape: shape.to_vec(), data: f32_bytes(x) }
}

fn q8t(name: &str, x: &[f64]) -> BuildTensor {
    let xf: Vec<f32> = x.iter().map(|v| *v as f32).collect();
    let (bytes, _) = roundtrip_matrix(QuantFormat::Q8_0, &xf, D_OUT, D_IN).unwrap();
    BuildTensor { name: name.into(), dtype: "Q8_0".into(), shape: vec![D_OUT, D_IN], data: bytes }
}

fn st_file(dir: &Scratch, name: &str, ts: &[BuildTensor]) -> String {
    let p = dir.path(name);
    write(&p, &build_safetensors(ts, &BTreeMap::new()).unwrap());
    p
}

fn promoted(v: &Value, tensor: &str) -> Vec<u64> {
    let t = v["tensors"].as_array().unwrap().iter().find(|t| t["name"] == tensor).unwrap();
    t["vocab_projection"]["promoted"].as_array().unwrap().iter().map(|e| e["index"].as_u64().unwrap()).collect()
}

fn tensor<'a>(v: &'a Value, name: &str) -> &'a Value {
    v["tensors"].as_array().unwrap().iter().find(|t| t["name"] == name).unwrap_or_else(|| panic!("{} missing in {}", name, v))
}

#[test]
fn reference_recovers_the_edit_a_single_file_read_cannot_see() {
    let (base, edited, e) = fixture();
    let dir = Scratch::new("ref-diff");
    let head = f32t("output.weight", &[VOCAB, D_OUT], &e);
    let base_st = st_file(&dir, "base.safetensors", &[f32t("blk.0.ffn_down.weight", &[D_OUT, D_IN], &base), head.clone()]);
    let edit_st = st_file(&dir, "edited.safetensors", &[f32t("blk.0.ffn_down.weight", &[D_OUT, D_IN], &edited), head.clone()]);
    let edit_gg = dir.path("edited.gguf");
    write(&edit_gg, &build_gguf(&[], &[q8t("blk.0.ffn_down.weight", &edited), head.clone()], 32).unwrap());

    // control: the file alone leads with the base's own direction, not the edit's tokens
    let (code, alone, out) = secant(&["read", "--model", &edit_st, "--tensor", "blk.0.*", "--vocab-tensor", "output.weight", "--json"]);
    assert_eq!(code, 0, "{}", out);
    assert!(alone.get("reference").is_none() && !out.contains("reference"), "default receipt must carry no reference key");
    assert_ne!(&promoted(&alone, "blk.0.ffn_down.weight")[..2], &[PLANTED[0] as u64, PLANTED[1] as u64], "the gap this closes");
    let s1_alone = tensor(&alone, "blk.0.ffn_down.weight")["singular"][0]["sigma"].as_f64().unwrap();
    assert!((s1_alone - THETA_BASE).abs() < 0.2 * THETA_BASE, "single file sees the base direction: {}", s1_alone);

    // against the base: Delta W is the planted edit
    for (model, what) in [(&edit_st, "safetensors F32"), (&edit_gg, "gguf Q8_0 vs safetensors base")] {
        for cmd in ["read", "audit-trojan"] {
            let (code, v, out) = secant(&[
                cmd,
                "--model",
                model,
                "--reference",
                &base_st,
                "--tensor",
                "blk.0.*",
                "--vocab-tensor",
                "output.weight",
                "--json",
            ]);
            assert_eq!(code, 0, "{} {}: {}", cmd, what, out);
            let t = tensor(&v, "blk.0.ffn_down.weight");
            assert_eq!(&promoted(&v, "blk.0.ffn_down.weight")[..2], &[PLANTED[0] as u64, PLANTED[1] as u64], "{} {}", cmd, what);
            let s1 = t["singular"][0]["sigma"].as_f64().unwrap();
            let tol = if what.contains("Q8_0") { 0.02 } else { 1e-5 };
            assert!((s1 - THETA_EDIT).abs() <= tol * THETA_EDIT, "{} {}: sigma_1 {} vs theta {}", cmd, what, s1, THETA_EDIT);
            let rd = &t["reference_diff"];
            assert_eq!(rd["status"], "compared", "{}", rd);
            assert_eq!(rd["basis"], "delta_w");
            assert_eq!(rd["reference_dtype"], "F32");
            assert_eq!(rd["identical"], false);
            assert!(rd["sigma1_energy_fraction"].as_f64().unwrap() > 0.9, "a planted rank-1 edit is one direction: {}", rd);
            let rel = rd["relative_delta"].as_f64().unwrap();
            assert!((rel - THETA_EDIT / THETA_BASE).abs() < 0.1, "{} {}: relative delta {}", cmd, what, rel);
            assert_eq!(t["frobenius"], rd["delta_frobenius"], "frobenius is of Delta W");
            assert_eq!(v["options"]["reference"], base_st.as_str());
            let r = &v["reference"];
            assert_eq!(r["path"], base_st.as_str());
            assert_eq!(r["format"], "safetensors");
            assert_eq!(r["input_unchanged"], true);
            assert_eq!(r["pre_sha256"], r["post_sha256"]);
            assert_eq!(r["pre_sha256"], sha256_bytes_hex(&std::fs::read(&base_st).unwrap()).as_str());
            assert_eq!(r["tensors_compared"], 1, "only the selected tensor is compared; the vocab head is read from --model");
            assert_eq!(r["tensors_skipped"], 0);
        }
    }
}

#[test]
fn delta_is_exact_float64_subtraction_of_the_widened_payloads() {
    // W - W_ref = [[0.5, 0], [0, 3]] exactly in F32: sigma = 3, 0.5; ||.||_F^2 = 9.25.
    let dir = Scratch::new("ref-exact");
    let w = st_file(&dir, "w.safetensors", &[f32t("w", &[2, 2], &[1.0, 2.0, 3.0, 4.0])]);
    let r = st_file(&dir, "r.safetensors", &[f32t("w", &[2, 2], &[0.5, 2.0, 3.0, 1.0])]);
    let (code, v, out) = secant(&["read", "--model", &w, "--reference", &r, "--json"]);
    assert_eq!(code, 0, "{}", out);
    let t = tensor(&v, "w");
    assert_eq!(t["singular"][0]["sigma"].as_f64().unwrap(), 3.0);
    assert_eq!(t["singular"][1]["sigma"].as_f64().unwrap(), 0.5);
    assert_eq!(t["frobenius"].as_f64().unwrap(), 9.25f64.sqrt());
    assert_eq!(t["reference_diff"]["reference_frobenius"].as_f64().unwrap(), (0.25f64 + 4.0 + 9.0 + 1.0).sqrt());
}

#[test]
fn identical_reference_is_reported_identical_not_as_a_spectrum() {
    let (base, _, _) = fixture();
    let dir = Scratch::new("ref-same");
    let a = st_file(&dir, "a.safetensors", &[f32t("w", &[D_OUT, D_IN], &base)]);
    let b = st_file(&dir, "b.safetensors", &[f32t("w", &[D_OUT, D_IN], &base), f32t("extra", &[4], &[1.0, 2.0, 3.0, 4.0])]);
    let (code, v, out) = secant(&["read", "--model", &a, "--reference", &b, "--json"]);
    assert_eq!(code, 0, "{}", out);
    let t = tensor(&v, "w");
    assert_eq!(t["reference_diff"]["identical"], true);
    assert_eq!(t["reference_diff"]["delta_frobenius"].as_f64().unwrap(), 0.0);
    assert_eq!(t["reference_diff"]["relative_delta"].as_f64().unwrap(), 0.0);
    assert!(t.get("singular").is_none(), "a zero delta has no spectrum: {}", t);
    assert!(t["notes"].to_string().contains("all singular values are zero"));
    assert_eq!(v["reference"]["tensors_in_file"], 2, "tensors only the reference has are not reported");
}

#[test]
fn unmatched_tensors_are_skipped_with_a_reason_never_compared() {
    let (base, edited, _) = fixture();
    let dir = Scratch::new("ref-skip");
    let model = st_file(
        &dir,
        "m.safetensors",
        &[
            f32t("keep", &[D_OUT, D_IN], &edited),
            f32t("renamed_in_model", &[D_OUT, D_IN], &edited),
            f32t("reshaped", &[D_IN, D_OUT], &edited),
            f32t("ints", &[2, 2], &[0.0; 4]),
        ],
    );
    let reference = st_file(
        &dir,
        "r.safetensors",
        &[f32t("keep", &[D_OUT, D_IN], &base), f32t("reshaped", &[D_OUT, D_IN], &base), f32t("ints", &[2, 2], &[0.0; 4])],
    );
    let (code, v, out) = secant(&["read", "--model", &model, "--reference", &reference, "--json"]);
    assert_eq!(code, 0, "{}", out);
    assert_eq!(tensor(&v, "keep")["reference_diff"]["status"], "compared");
    let missing = tensor(&v, "renamed_in_model");
    assert!(missing["status"].as_str().unwrap().contains("absent from the reference"), "{}", missing);
    assert!(missing.get("frobenius").is_none() && missing.get("singular").is_none());
    assert!(missing["reference_diff"]["status"].as_str().unwrap().starts_with("skipped:"));
    let reshaped = tensor(&v, "reshaped");
    assert!(reshaped["status"].as_str().unwrap().contains("differs from"), "{}", reshaped);
    assert!(reshaped.get("frobenius").is_none());
    assert_eq!(v["reference"]["tensors_compared"].as_u64().unwrap() + v["reference"]["tensors_skipped"].as_u64().unwrap(), 4);
    assert_eq!(v["reference"]["tensors_skipped"], 2);
}

#[test]
fn unusable_references_are_usage_or_io_errors_exit_2() {
    let (base, edited, _) = fixture();
    let dir = Scratch::new("ref-bad");
    let m = st_file(&dir, "m.safetensors", &[f32t("a", &[D_OUT, D_IN], &edited)]);
    let disjoint = st_file(&dir, "d.safetensors", &[f32t("b", &[D_OUT, D_IN], &base)]);
    let (code, _, out) = secant(&["read", "--model", &m, "--reference", &disjoint, "--json"]);
    assert_eq!(code, 2);
    assert!(out.contains("shares no comparable tensor") && out.contains("absent from the reference"), "{}", out);
    let (code, _, out) = secant(&["audit-trojan", "--model", &m, "--reference", &m]);
    assert_eq!(code, 2);
    assert!(out.contains("is the model file itself"), "{}", out);
    let (code, _, out) = secant(&["read", "--model", &m, "--reference", &dir.path("nope.safetensors")]);
    assert_eq!(code, 2, "{}", out);
    let junk = dir.path("junk.safetensors");
    write(&junk, b"not a checkpoint");
    assert_eq!(secant(&["read", "--model", &m, "--reference", &junk]).0, 2);
}

#[test]
fn audit_trojan_human_report_names_the_reference_and_the_basis() {
    let (base, edited, e) = fixture();
    let dir = Scratch::new("ref-human");
    let head = f32t("output.weight", &[VOCAB, D_OUT], &e);
    let b = st_file(&dir, "b.safetensors", &[f32t("blk.0.ffn_down.weight", &[D_OUT, D_IN], &base), head.clone()]);
    let m = st_file(&dir, "m.safetensors", &[f32t("blk.0.ffn_down.weight", &[D_OUT, D_IN], &edited), head]);
    let (code, _, text) =
        secant(&["audit-trojan", "--model", &m, "--reference", &b, "--tensor", "blk.*", "--vocab-tensor", "output.weight"]);
    assert_eq!(code, 0, "{}", text);
    assert!(text.contains("Reference SHA-256") && text.contains("Delta W = W - W_ref; 1 compared, 0 skipped"), "{}", text);
    assert!(text.contains("7("), "planted token listed beside the ranked Delta W spike: {}", text);
}
