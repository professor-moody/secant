//! `secant audit-trojan --rank-k <K>` end to end: cumulative top-k singular
//! energy and rank-k vocabulary coherence in single-file mode, in `--reference`
//! (Delta W) mode and on a native PEFT LoRA `adapter_model.safetensors`
//! (the spectrum of `s B A`, checked against a dense reference computed in the
//! test). Also the non-interference contract: turning `--rank-k` on adds keys
//! and changes none of the existing ones.

#[path = "../../secant-core/tests/common/mod.rs"]
mod common;

use common::*;
use secant_core::builder::*;
use secant_core::dtype::{bf16_from_f32, f32_from_bf16};
use secant_core::read::top_singular_triplets;
use serde_json::Value;
use std::collections::BTreeMap;
use std::process::Command;

const D_OUT: usize = 64;
const D_IN: usize = 128;
const VOCAB: usize = 50;
const LAYERS: usize = 6;
const TROJAN: usize = 3;
const TOK_U1: usize = 7;
const TOK_U2: usize = 23;
const TOK_MIX: usize = 31;

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

fn unit_block(n: usize, lo: usize, hi: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|j| if (lo..hi).contains(&j) { 1.0 } else { 0.0 }).collect();
    let k = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter_mut().for_each(|x| *x /= k);
    v
}

fn add_rank1(w: &mut [f64], cols: usize, theta: f64, u: &[f64], v: &[f64]) {
    for i in 0..u.len() {
        for j in 0..cols {
            w[i * cols + j] += theta * u[i] * v[j];
        }
    }
}

fn f32t(name: &str, shape: &[usize], x: &[f64]) -> BuildTensor {
    BuildTensor { name: name.into(), dtype: "F32".into(), shape: shape.to_vec(), data: f32_bytes(x) }
}

fn st(dir: &Scratch, name: &str, ts: &[BuildTensor]) -> String {
    let p = dir.path(name);
    write(&p, &build_safetensors(ts, &BTreeMap::new()).unwrap());
    p
}

/// Rank-2 edit directions: orthogonal output (u1, u2) and input (v1, v2) pairs.
fn edit_dirs() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    (unit_mod(D_OUT, 1), unit_mod(D_OUT, 2), unit_block(D_IN, 0, 16), unit_block(D_IN, 16, 32))
}

/// Unembedding with three planted tokens inside span(u1, u2).
fn unembedding() -> Vec<f64> {
    let (u1, u2, _, _) = edit_dirs();
    let mut e = noise(VOCAB * D_OUT, 9, 0.01);
    let s = 0.5f64.sqrt();
    for i in 0..D_OUT {
        e[TOK_U1 * D_OUT + i] = 5.0 * u1[i];
        e[TOK_U2 * D_OUT + i] = -4.0 * u2[i]; // negative sign: coherence must not care
        e[TOK_MIX * D_OUT + i] = 3.0 * s * (u1[i] + u2[i]);
    }
    e
}

/// Six `blk.N.ffn_down.weight` layers with DIFFERENT amplitudes (the energy
/// fractions are scale-free); layer 3 also carries the rank-2 edit.
fn layers(edit: bool) -> Vec<Vec<f64>> {
    let (u1, u2, v1, v2) = edit_dirs();
    (0..LAYERS)
        .map(|l| {
            let mut w = noise(D_OUT * D_IN, 100 + l as u64, 0.002 * (l + 1) as f64);
            if edit && l == TROJAN {
                add_rank1(&mut w, D_IN, 3.0, &u1, &v1);
                add_rank1(&mut w, D_IN, 2.5, &u2, &v2);
            }
            w
        })
        .collect()
}

fn model_file(dir: &Scratch, name: &str, ls: &[Vec<f64>]) -> String {
    let mut ts: Vec<BuildTensor> =
        ls.iter().enumerate().map(|(l, w)| f32t(&format!("blk.{}.ffn_down.weight", l), &[D_OUT, D_IN], w)).collect();
    ts.push(f32t("output.weight", &[VOCAB, D_OUT], &unembedding()));
    st(dir, name, &ts)
}

fn tokens(v: &Value) -> Vec<(u64, f64)> {
    v.as_array().unwrap().iter().map(|e| (e["index"].as_u64().unwrap(), e["value"].as_f64().unwrap())).collect()
}

fn row<'a>(rank: &'a Value, name: &str) -> &'a Value {
    rank.as_array().unwrap().iter().find(|r| r["tensor"] == name).unwrap_or_else(|| panic!("{} missing from {}", name, rank))
}

#[test]
fn single_file_rank2_edit_has_concentrated_energy_and_names_its_tokens() {
    let dir = Scratch::new("rankk-single");
    let m = model_file(&dir, "edited.safetensors", &layers(true));
    let (code, v, out) =
        secant(&["audit-trojan", "--model", &m, "--tensor", "blk.*", "--rank-k", "2", "--vocab-tensor", "output.weight", "--json"]);
    assert_eq!(code, 0, "{}", out);
    assert_eq!(v["options"]["rank_k"], 2);
    let rk = &v["rank_k_outliers"];
    assert_eq!(rk.as_array().unwrap().len(), LAYERS);
    let top = &rk[0];
    assert_eq!(top["tensor"], "blk.3.ffn_down.weight", "{}", top);
    assert_eq!(top["sibling"]["sibling_outlier"], true, "{}", top);
    assert!(top["sibling"]["robust_z"].as_f64().unwrap() > 3.5);
    let (e1, e2) = (top["sigma1_energy"].as_f64().unwrap(), top["cumulative_energy"].as_f64().unwrap());
    assert!(e1 < 0.7 && e2 > 0.9 && e2 <= 1.0 + 1e-9, "E_1 = {}, E_2 = {}: a rank-2 edit is under-stated by rank-1 alone", e1, e2);
    assert!(e2 > e1 + 0.2, "the second direction must add energy: {} vs {}", e1, e2);
    for other in (0..LAYERS).filter(|l| *l != TROJAN) {
        let r = row(rk, &format!("blk.{}.ffn_down.weight", other));
        assert_eq!(r["sibling"]["sibling_outlier"], false, "{}", r);
        assert!(r["cumulative_energy"].as_f64().unwrap() < 0.3, "{}", r);
    }
    // per-tensor block
    let t = v["tensors"].as_array().unwrap().iter().find(|t| t["name"] == "blk.3.ffn_down.weight").unwrap();
    let b = &t["rank_k"];
    assert_eq!(b["status"], "ok");
    assert_eq!(b["basis"], "w");
    assert_eq!(b["requested_k"], 2);
    assert_eq!(b["k_computed"], 2);
    let ks: Vec<u64> = b["prefix_energy"].as_array().unwrap().iter().map(|p| p["k"].as_u64().unwrap()).collect();
    assert_eq!(ks, vec![1, 2]);
    assert_eq!(b["prefix_energy"][1]["energy"], b["cumulative_energy"]);
    // E_k is the sigma^2 share of ||W||_F^2, recomputable from the receipt
    let fro = t["frobenius"].as_f64().unwrap();
    let s: Vec<f64> = t["singular"].as_array().unwrap().iter().map(|x| x["sigma"].as_f64().unwrap()).collect();
    assert!((b["cumulative_energy"].as_f64().unwrap() - (s[0] * s[0] + s[1] * s[1]) / (fro * fro)).abs() < 1e-12);
    assert!((b["frobenius_sq"].as_f64().unwrap() - fro * fro).abs() <= 1e-12 * fro * fro);
    // rank-2 vocabulary coherence: the three planted tokens lead, sign-blind, far above chance
    let vocab = &b["vocab"];
    assert_eq!(vocab["status"], "ok");
    assert_eq!(vocab["basis_rank"], 2);
    let chance = vocab["chance_level"].as_f64().unwrap();
    assert!((chance - 2.0 / D_OUT as f64).abs() < 1e-15);
    let toks = tokens(&vocab["tokens"]);
    let lead: std::collections::BTreeSet<u64> = toks[..3].iter().map(|t| t.0).collect();
    assert_eq!(lead, [TOK_U1 as u64, TOK_U2 as u64, TOK_MIX as u64].into_iter().collect(), "{:?}", toks);
    assert!(toks[..3].iter().all(|t| t.1 > 0.99), "{:?}", toks);
    assert!(toks[3].1 < 0.5, "an unrelated token is not coherent: {:?}", toks[3]);
    assert!(vocab["top_over_chance"].as_f64().unwrap() > 25.0);
    // per-direction signed projections carry the individual directions
    assert_eq!(vocab["per_direction"].as_array().unwrap().len(), 2);
    assert_eq!(vocab["per_direction"][0]["rank"], 1);
    // the ranking row repeats the leading coherent tokens
    assert_eq!(tokens(&top["coherent_tokens"])[0].0, toks[0].0);
    // human report
    let (code, _, text) = secant(&["audit-trojan", "--model", &m, "--tensor", "blk.*", "--rank-k", "2", "--vocab-tensor", "output.weight"]);
    assert_eq!(code, 0);
    assert!(text.contains("Rank-2 spectral energy of W") && text.contains("blk.3.ffn_down.weight") && text.contains("YES"), "{}", text);
}

#[test]
fn rank_k_adds_keys_and_changes_no_existing_one() {
    let dir = Scratch::new("rankk-nointerfere");
    let m = model_file(&dir, "edited.safetensors", &layers(true));
    let base = ["audit-trojan", "--model", m.as_str(), "--tensor", "blk.*", "--vocab-tensor", "output.weight", "--json"];
    let (_, plain, _) = secant(&base);
    // default run carries no rank-k key anywhere
    assert!(!plain.to_string().contains("rank_k"), "default audit-trojan must not carry rank_k");
    // rank_k 8 > the default top-k-singular 4: extra triplets are computed but must not leak
    let mut args = base.to_vec();
    args.extend(["--rank-k", "8"]);
    let (code, with, out) = secant(&args);
    assert_eq!(code, 0, "{}", out);
    let mut stripped = with.clone();
    stripped.as_object_mut().unwrap().remove("rank_k_outliers").unwrap();
    stripped["options"].as_object_mut().unwrap().remove("rank_k").unwrap();
    for t in stripped["tensors"].as_array_mut().unwrap() {
        t.as_object_mut().unwrap().remove("rank_k");
    }
    assert_eq!(stripped, plain, "enabling --rank-k changed an existing field");
    // the energy curve is monotone and never above 1 (Ky Fan lower bound), converged or not
    for t in with["tensors"].as_array().unwrap() {
        let Some(b) = t.get("rank_k") else { continue };
        let pts: Vec<f64> = b["prefix_energy"].as_array().unwrap().iter().map(|p| p["energy"].as_f64().unwrap()).collect();
        assert_eq!(
            b["prefix_energy"].as_array().unwrap().iter().map(|p| p["k"].as_u64().unwrap()).collect::<Vec<_>>(),
            vec![1, 2, 4, 8],
            "{}",
            t["name"]
        );
        assert!(pts.windows(2).all(|w| w[0] <= w[1]), "{}: {:?}", t["name"], pts);
        assert!(*pts.last().unwrap() <= 1.0 + 1e-9, "{}: {:?}", t["name"], pts);
        assert_eq!(b["k_computed"], 8);
    }
}

#[test]
fn reference_mode_measures_the_delta_and_matches_sigma1_fraction_exactly() {
    let dir = Scratch::new("rankk-ref");
    let base_layers: Vec<Vec<f64>> = (0..LAYERS).map(|l| noise(D_OUT * D_IN, 300 + l as u64, 0.01 * (l + 1) as f64)).collect();
    let (u1, u2, v1, v2) = edit_dirs();
    // edited = base + diffuse fine-tune-like noise everywhere + a rank-2 edit in one layer
    let edited: Vec<Vec<f64>> = (0..LAYERS)
        .map(|l| {
            let mut w = base_layers[l].clone();
            for (x, d) in w.iter_mut().zip(noise(D_OUT * D_IN, 400 + l as u64, 0.002 * (l + 1) as f64)) {
                *x += d;
            }
            if l == TROJAN {
                add_rank1(&mut w, D_IN, 3.0, &u1, &v1);
                add_rank1(&mut w, D_IN, 2.5, &u2, &v2);
            }
            w
        })
        .collect();
    let base = model_file(&dir, "base.safetensors", &base_layers);
    let ed = model_file(&dir, "edited.safetensors", &edited);
    let (code, v, out) = secant(&[
        "audit-trojan",
        "--model",
        &ed,
        "--reference",
        &base,
        "--tensor",
        "blk.*",
        "--rank-k",
        "2",
        "--vocab-tensor",
        "output.weight",
        "--json",
    ]);
    assert_eq!(code, 0, "{}", out);
    let rk = &v["rank_k_outliers"];
    assert_eq!(rk[0]["tensor"], "blk.3.ffn_down.weight");
    assert_eq!(rk[0]["sibling"]["sibling_outlier"], true, "{}", rk[0]);
    assert!(rk[0]["cumulative_energy"].as_f64().unwrap() > 0.98, "a rank-2 delta is all in 2 directions: {}", rk[0]);
    assert!(rk[0]["sigma1_energy"].as_f64().unwrap() < 0.7, "{}", rk[0]);
    for t in v["tensors"].as_array().unwrap() {
        let b = &t["rank_k"];
        assert_eq!(b["basis"], "delta_w", "{}", t["name"]);
        // E_1 is the existing sigma1_energy_fraction, bit for bit
        assert_eq!(b["prefix_energy"][0]["energy"], t["reference_diff"]["sigma1_energy_fraction"], "{}", t["name"]);
    }
    let vocab = &v["tensors"].as_array().unwrap().iter().find(|t| t["name"] == "blk.3.ffn_down.weight").unwrap()["rank_k"]["vocab"];
    let lead: std::collections::BTreeSet<u64> = tokens(&vocab["tokens"])[..3].iter().map(|t| t.0).collect();
    assert_eq!(lead, [TOK_U1 as u64, TOK_U2 as u64, TOK_MIX as u64].into_iter().collect());
    // without --reference the same file is read as W, not Delta W
    let (_, single, _) = secant(&["audit-trojan", "--model", &ed, "--tensor", "blk.*", "--rank-k", "2", "--json"]);
    assert_eq!(single["tensors"][0]["rank_k"]["basis"], "w");
    // a tensor identical to its reference has no delta: undefined, never a number
    let twin = model_file(&dir, "twin.safetensors", &base_layers);
    let (code, v, out) = secant(&["audit-trojan", "--model", &twin, "--reference", &base, "--tensor", "blk.*", "--rank-k", "4", "--json"]);
    assert_eq!(code, 0, "{}", out);
    for t in v["tensors"].as_array().unwrap() {
        assert_eq!(t["reference_diff"]["identical"], true, "{}", t["name"]);
        assert!(t["rank_k"]["status"].as_str().unwrap().starts_with("undefined"), "{}", t["rank_k"]);
        assert!(t["rank_k"].get("cumulative_energy").is_none() && t["rank_k"].get("prefix_energy").is_none());
    }
    assert_eq!(v["rank_k_outliers"].as_array().unwrap().len(), 0);
}

#[test]
fn rank_k_values_are_validated() {
    let dir = Scratch::new("rankk-bad");
    let m = model_file(&dir, "m.safetensors", &layers(false));
    for bad in ["0", "3", "5", "16", "-1", "x"] {
        let (code, _, out) = secant(&["audit-trojan", "--model", &m, "--rank-k", bad]);
        assert_eq!(code, 2, "--rank-k {}: {}", bad, out);
    }
    for good in ["1", "2", "4", "8"] {
        let (code, _, out) = secant(&["audit-trojan", "--model", &m, "--tensor", "blk.0.*", "--rank-k", good, "--json"]);
        assert_eq!(code, 0, "--rank-k {}: {}", good, out);
    }
    // adapter-only flags on a base checkpoint are a usage error, not silently ignored
    assert_eq!(secant(&["audit-trojan", "--model", &m, "--vocab-model", &m]).0, 2);
    assert_eq!(secant(&["audit-trojan", "--model", &m, "--adapter-config", &m]).0, 2);
    // `read` does not grow the flag
    assert_eq!(secant(&["read", "--model", &m, "--rank-k", "2"]).0, 2);
}

// ---------------------------------------------------------------- LoRA

const A_OUT: usize = 32;
const A_IN: usize = 48;
const RANK: usize = 4;
const ALPHA: f64 = 16.0;
const A_VOCAB: usize = 40;
const ZERO_LAYER: usize = 5;

fn bf16t(name: &str, shape: &[usize], x: &[f64]) -> BuildTensor {
    BuildTensor { name: name.into(), dtype: "BF16".into(), shape: shape.to_vec(), data: bf16_bytes(x) }
}

fn rb(x: f64) -> f64 {
    f32_from_bf16(bf16_from_f32(x as f32)) as f64
}

/// `(A [r x d_in], B [d_out x r])` of one module, values already rounded to
/// bf16 so the dense reference and the stored payload agree exactly.
fn lora_pair(l: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = noise(RANK * A_IN, 500 + l as u64, 0.1);
    let mut b = noise(A_OUT * RANK, 600 + l as u64, 0.1);
    if l == TROJAN {
        // one dominant rank-1 direction: B[:, 0] = 2 u, A[0, :] = 1 v
        let u = unit_mod(A_OUT, 1);
        let v = unit_block(A_IN, 0, 12);
        for i in 0..A_OUT {
            b[i * RANK] = 2.0 * u[i];
        }
        a[..A_IN].copy_from_slice(&v[..A_IN]);
    }
    if l == ZERO_LAYER {
        b = vec![0.0; A_OUT * RANK]; // PEFT initialises B to zero
    }
    (a.into_iter().map(rb).collect(), b.into_iter().map(rb).collect())
}

fn module(l: usize) -> String {
    format!("model.layers.{}.mlp.down_proj", l)
}

fn adapter_tensors() -> Vec<BuildTensor> {
    let mut ts = Vec::new();
    for l in 0..LAYERS {
        let (a, b) = lora_pair(l);
        ts.push(bf16t(&format!("base_model.model.{}.lora_A.weight", module(l)), &[RANK, A_IN], &a));
        ts.push(bf16t(&format!("base_model.model.{}.lora_B.weight", module(l)), &[A_OUT, RANK], &b));
    }
    ts
}

fn dense_delta(l: usize, scale: f64) -> Vec<f64> {
    let (a, b) = lora_pair(l);
    let mut w = vec![0f64; A_OUT * A_IN];
    for i in 0..A_OUT {
        for t in 0..RANK {
            for j in 0..A_IN {
                w[i * A_IN + j] += scale * b[i * RANK + t] * a[t * A_IN + j];
            }
        }
    }
    w
}

fn lm_head(dir: &Scratch) -> String {
    let u = unit_mod(A_OUT, 1);
    let mut e = noise(A_VOCAB * A_OUT, 77, 0.01);
    for i in 0..A_OUT {
        e[5 * A_OUT + i] = 3.0 * u[i];
        e[11 * A_OUT + i] = -2.0 * u[i];
    }
    st(dir, "base.safetensors", &[f32t("lm_head.weight", &[A_VOCAB, A_OUT], &e)])
}

#[test]
fn lora_adapter_trojan_layer_is_found_from_b_times_a() {
    let dir = Scratch::new("rankk-lora");
    let ad = st(&dir, "adapter_model.safetensors", &adapter_tensors());
    write(
        &dir.path("adapter_config.json"),
        format!("{{\"peft_type\":\"LORA\",\"r\":{},\"lora_alpha\":{},\"base_model_name_or_path\":\"fixture\"}}", RANK, ALPHA).as_bytes(),
    );
    let head = lm_head(&dir);
    let (code, v, out) =
        secant(&["audit-trojan", "--model", &ad, "--rank-k", "4", "--vocab-tensor", "lm_head.weight", "--vocab-model", &head, "--json"]);
    assert_eq!(code, 0, "{}", out);
    assert_eq!(v["schema"], "secant.wsp.audit-lora.v1");
    assert_eq!(v["input_unchanged"], true);
    assert_eq!(v["pre_sha256"], v["post_sha256"]);
    assert_eq!(v["pre_sha256"], secant_core::container::sha256_hex(&std::fs::read(&ad).unwrap()));
    assert_eq!(v["vocab_source"]["input_unchanged"], true);
    assert_eq!(v["adapter_config"]["lora_alpha"], ALPHA);
    assert_eq!(v["modules_found"], LAYERS);
    assert_eq!(v["modules_scanned"], LAYERS - 1, "the freshly initialised (B = 0) module has no delta");
    let zero = v["modules"].as_array().unwrap().iter().find(|m| m["name"] == module(ZERO_LAYER)).unwrap();
    assert_eq!(zero["status"], "zero_delta", "{}", zero);
    assert!(zero["rank_k"]["status"].as_str().unwrap().starts_with("undefined"));
    // the trojan module ranks first and stands out from its siblings
    let rk = &v["rank_k_outliers"];
    assert_eq!(rk.as_array().unwrap().len(), LAYERS - 1);
    // rank-4 adapters at k = 4 have E_4 = 1 for every module: the ranking uses the informative k = 2
    assert!(rk.as_array().unwrap().iter().all(|r| r["ranking_k"] == 2 && r["requested_k"] == 4 && r["saturated"] == false), "{}", rk);
    assert_eq!(rk[0]["tensor"], module(TROJAN), "{}", rk[0]);
    assert_eq!(rk[0]["sibling"]["sibling_outlier"], true, "{}", rk[0]);
    assert!(rk[0]["sigma1_energy"].as_f64().unwrap() > 0.95, "{}", rk[0]);
    // scaling: lora_alpha / r, and the numbers are those of s B A, checked against a dense product
    for l in [TROJAN, 0, 2] {
        let m = v["modules"].as_array().unwrap().iter().find(|m| m["name"] == module(l)).unwrap();
        assert_eq!(m["status"], "ok", "{}", m);
        assert_eq!(m["scaling"], ALPHA / RANK as f64);
        assert_eq!(
            (m["rank"].as_u64().unwrap(), m["d_out"].as_u64().unwrap(), m["d_in"].as_u64().unwrap()),
            (RANK as u64, A_OUT as u64, A_IN as u64)
        );
        let w = dense_delta(l, ALPHA / RANK as f64);
        let fro: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((m["delta_frobenius"].as_f64().unwrap() - fro).abs() <= 1e-9 * fro, "module {}: {} vs {}", l, m["delta_frobenius"], fro);
        let dense = top_singular_triplets(&w, A_OUT, A_IN, RANK).unwrap();
        let got: Vec<f64> = m["singular"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        assert_eq!(got.len(), dense.len(), "module {}", l);
        for (g, d) in got.iter().zip(dense.iter()) {
            assert!((g - d.sigma).abs() <= 1e-8 * d.sigma, "module {}: sigma {} vs dense {}", l, g, d.sigma);
        }
        let e1: f64 = dense[0].sigma * dense[0].sigma / (fro * fro);
        assert!((m["rank_k"]["prefix_energy"][0]["energy"].as_f64().unwrap() - e1).abs() < 1e-8, "module {}", l);
        // the whole spectrum has at most r values: E_4 is 1, converged
        assert!((m["rank_k"]["cumulative_energy"].as_f64().unwrap() - 1.0).abs() < 1e-9, "{}", m["rank_k"]);
        assert_eq!(m["rank_k"]["basis"], "lora_ba");
        assert_eq!(m["rank_k"]["converged"], true);
    }
    // vocabulary coherence of the trojan module (unembedding from the base checkpoint)
    let m = v["modules"].as_array().unwrap().iter().find(|m| m["name"] == module(TROJAN)).unwrap();
    let vocab = &m["rank_k"]["vocab"];
    assert_eq!(vocab["status"], "ok", "{}", vocab);
    let toks = tokens(&vocab["tokens"]);
    let lead: std::collections::BTreeSet<u64> = toks[..2].iter().map(|t| t.0).collect();
    assert_eq!(lead, [5u64, 11u64].into_iter().collect(), "{:?}", toks);
    assert!(toks[0].1 > 0.99 && toks[1].1 > 0.99, "{:?}", toks);
    // human report and the glob selecting one module
    let (code, _, text) = secant(&["audit-trojan", "--model", &ad, "--rank-k", "2"]);
    assert_eq!(code, 0);
    assert!(
        text.contains("LoRA adapter spectral trojan audit")
            && text.contains("Rank-2 spectral energy of Delta W = s B A")
            && text.contains(&module(TROJAN)),
        "{}",
        text
    );
    let (code, one, out) = secant(&["audit-trojan", "--model", &ad, "--tensor", "model.layers.3.*", "--json"]);
    assert_eq!(code, 0, "{}", out);
    assert_eq!(one["modules_selected"], 1);
    assert_eq!(one["options"]["rank_k"], 8, "adapters default to the full rank-8 curve");
}

#[test]
fn lora_adapter_without_config_uses_unit_scale_and_flags_what_it_cannot_scan() {
    let dir = Scratch::new("rankk-lora-noconf");
    let mut ts = adapter_tensors();
    ts.push(bf16t("base_model.model.model.embed_tokens.lora_embedding_A", &[RANK, 10], &noise(RANK * 10, 1, 0.1)));
    ts.push(bf16t("base_model.model.model.embed_tokens.lora_embedding_B", &[6, RANK], &noise(6 * RANK, 2, 0.1)));
    ts.push(bf16t("base_model.model.model.layers.0.mlp.down_proj.lora_magnitude_vector", &[A_OUT], &noise(A_OUT, 3, 0.1)));
    let ad = st(&dir, "adapter_model.safetensors", &ts);
    let (code, v, out) = secant(&["audit-trojan", "--model", &ad, "--rank-k", "2", "--json"]);
    assert_eq!(code, 0, "{}", out);
    assert!(v.get("adapter_config").is_none() || v["adapter_config"].is_null());
    let m = v["modules"].as_array().unwrap().iter().find(|m| m["name"] == module(TROJAN)).unwrap();
    assert_eq!(m["scaling"], 1.0);
    assert!(m["scaling_source"].as_str().unwrap().starts_with("none"), "{}", m);
    let w = dense_delta(TROJAN, 1.0);
    let fro: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((m["delta_frobenius"].as_f64().unwrap() - fro).abs() <= 1e-9 * fro);
    assert_eq!(v["other_tensors"]["embedding_adapters"], 2);
    assert_eq!(v["other_tensors"]["dora_magnitude_vectors"], 1);
    assert_eq!(v["other_tensors"]["count"], 3);
    // energy fractions do not depend on the scale
    let dir2 = Scratch::new("rankk-lora-scale");
    let ad2 = st(&dir2, "adapter_model.safetensors", &adapter_tensors());
    write(&dir2.path("adapter_config.json"), br#"{"r": 4, "lora_alpha": 100}"#);
    let (_, v2, _) = secant(&["audit-trojan", "--model", &ad2, "--rank-k", "2", "--json"]);
    let e = |v: &Value| row(&v["rank_k_outliers"], &module(TROJAN))["sigma1_energy"].as_f64().unwrap();
    assert!((e(&v) - e(&v2)).abs() < 1e-12, "{} vs {}", e(&v), e(&v2));
}

#[test]
fn fan_in_fan_out_adapter_keeps_its_spectrum_but_never_projects_the_wrong_side() {
    // Conv1D adapters store (B A)^T: energy is unchanged, but u_i (rows of B's column space) is not
    // the output direction, so the vocabulary projection is skipped rather than computed on the wrong side.
    let dir = Scratch::new("rankk-lora-fifo");
    let ad = st(&dir, "adapter_model.safetensors", &adapter_tensors());
    let head = lm_head(&dir);
    let args = |extra: &str| {
        write(&dir.path("adapter_config.json"), format!("{{\"r\": 4, \"lora_alpha\": 16{}}}", extra).as_bytes());
        secant(&["audit-trojan", "--model", &ad, "--rank-k", "2", "--vocab-tensor", "lm_head.weight", "--vocab-model", &head, "--json"])
    };
    let (code, plain, out) = args("");
    assert_eq!(code, 0, "{}", out);
    let (code, fifo, out) = args(", \"fan_in_fan_out\": true");
    assert_eq!(code, 0, "{}", out);
    assert_eq!(fifo["adapter_config"]["fan_in_fan_out"], true);
    assert_eq!(plain["adapter_config"]["fan_in_fan_out"], false);
    let find = |v: &Value| v["modules"].as_array().unwrap().iter().find(|m| m["name"] == module(TROJAN)).unwrap().clone();
    assert_eq!(find(&plain)["rank_k"]["vocab"]["status"], "ok");
    let skipped = find(&fifo)["rank_k"]["vocab"]["status"].as_str().unwrap().to_string();
    assert!(skipped.starts_with("skipped: fan_in_fan_out"), "{}", skipped);
    assert!(find(&fifo)["rank_k"]["vocab"].get("tokens").is_none());
    assert_eq!(
        find(&plain)["rank_k"]["prefix_energy"],
        find(&fifo)["rank_k"]["prefix_energy"],
        "the spectrum does not depend on the storage order"
    );
}

#[test]
fn lora_adapter_fails_closed_on_contradictions_and_bad_combinations() {
    let dir = Scratch::new("rankk-lora-bad");
    let ad = st(&dir, "adapter_model.safetensors", &adapter_tensors());
    // config rank contradicts every module: nothing could be scanned, which is exit 2, not a clean report
    write(&dir.path("adapter_config.json"), br#"{"r": 8, "lora_alpha": 16}"#);
    let (code, _, out) = secant(&["audit-trojan", "--model", &ad, "--json"]);
    assert_eq!(code, 2, "{}", out);
    assert!(out.contains("disagrees"), "{}", out);
    // an explicitly named config that does not exist is an error
    let (code, _, out) = secant(&["audit-trojan", "--model", &ad, "--adapter-config", &dir.path("nope.json"), "--json"]);
    assert_eq!(code, 2, "{}", out);
    write(&dir.path("adapter_config.json"), br#"{"r": 4, "lora_alpha": 16}"#);
    let head = lm_head(&dir);
    // reference and MP threshold are undefined for an adapter
    assert_eq!(secant(&["audit-trojan", "--model", &ad, "--reference", &head]).0, 2);
    assert_eq!(secant(&["audit-trojan", "--model", &ad, "--mp-threshold", "2"]).0, 2);
    // an adapter has no unembedding of its own
    assert_eq!(secant(&["audit-trojan", "--model", &ad, "--vocab-tensor", "lm_head.weight"]).0, 2);
    assert_eq!(secant(&["audit-trojan", "--model", &ad, "--vocab-model", &head]).0, 2);
    // a glob that matches no module
    assert_eq!(secant(&["audit-trojan", "--model", &ad, "--tensor", "nothing.*"]).0, 2);
}
