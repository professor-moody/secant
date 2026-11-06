//! `secant read --marchenko-pastur` and `secant audit-trojan` end to end
//! (Milestone S-2): a planted rank-1 edit detaches from the Marchenko-Pastur
//! bulk, is ranked first and flagged, and its promoted vocabulary tokens are
//! the ones the edit was aligned with; a clean tensor of the same shape is not
//! flagged; quantized (Q8_0) weights give the same verdict; and a default
//! `read` receipt carries none of the new keys (oracle layout unchanged).

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

/// Unit output direction of the planted edit.
fn u_dir() -> Vec<f64> {
    let mut u: Vec<f64> = (0..D_OUT).map(|i| if i % 4 == 1 { 1.0 } else { 0.0 }).collect();
    let n = u.iter().map(|x| x * x).sum::<f64>().sqrt();
    u.iter_mut().for_each(|x| *x /= n);
    u
}

fn tensors() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let amp = 0.02;
    let clean = noise(D_OUT * D_IN, 3, amp);
    let mut spiked = noise(D_OUT * D_IN, 5, amp);
    // bulk edge of uniform[-amp,amp] noise: amp/sqrt(3) * (sqrt(d_out) + sqrt(d_in))
    let edge = amp / 3f64.sqrt() * ((D_OUT as f64).sqrt() + (D_IN as f64).sqrt());
    let theta = 5.0 * edge;
    let u = u_dir();
    let v: Vec<f64> = (0..D_IN).map(|j| if j < 16 { 0.25 } else { 0.0 }).collect(); // unit, positive
    for i in 0..D_OUT {
        for j in 0..D_IN {
            spiked[i * D_IN + j] += theta * u[i] * v[j];
        }
    }
    // unembedding [vocab x d_model]: two rows aligned with u, the rest small noise
    let mut e = noise(VOCAB * D_OUT, 9, 0.01);
    for (k, &row) in PLANTED.iter().enumerate() {
        for i in 0..D_OUT {
            e[row * D_OUT + i] = u[i] * (5.0 - k as f64);
        }
    }
    (clean, spiked, e)
}

fn f32t(name: &str, shape: &[usize], x: &[f64]) -> BuildTensor {
    BuildTensor { name: name.into(), dtype: "F32".into(), shape: shape.to_vec(), data: f32_bytes(x) }
}

fn q8t(name: &str, x: &[f64]) -> BuildTensor {
    let xf: Vec<f32> = x.iter().map(|v| *v as f32).collect();
    let (bytes, _) = roundtrip_matrix(QuantFormat::Q8_0, &xf, D_OUT, D_IN).unwrap();
    BuildTensor { name: name.into(), dtype: "Q8_0".into(), shape: vec![D_OUT, D_IN], data: bytes }
}

fn check_ranking(v: &Value, what: &str) {
    let rank = v["spectral_outliers"].as_array().unwrap_or_else(|| panic!("{}: no ranking: {}", what, v));
    assert_eq!(rank.len(), 2, "{}: the two 2-D weights (the vocab head is selected out)", what);
    assert_eq!(rank[0]["tensor"], "blk.0.ffn_down.weight", "{}: spiked tensor ranks first", what);
    assert!(rank[0]["flagged"].as_bool().unwrap(), "{}", what);
    assert!(rank[0]["sigma1_over_mp"].as_f64().unwrap() > 3.0, "{}: {}", what, rank[0]);
    assert_eq!(rank[0]["outliers"], 1, "{}", what);
    let promoted: Vec<u64> = rank[0]["promoted_tokens"].as_array().unwrap().iter().map(|e| e["index"].as_u64().unwrap()).collect();
    assert_eq!(&promoted[..2], &[PLANTED[0] as u64, PLANTED[1] as u64], "{}: planted tokens promoted first", what);
    assert_eq!(rank[1]["tensor"], "blk.0.attn_q.weight");
    assert!(!rank[1]["flagged"].as_bool().unwrap(), "{}: clean tensor not flagged: {}", what, rank[1]);
    let r1 = rank[1]["sigma1_over_mp"].as_f64().unwrap();
    assert!((0.85..1.1).contains(&r1), "{}: iid tensor sits at the bulk edge: {}", what, r1);
    // the per-tensor block carries the formula's terms
    let t = v["tensors"].as_array().unwrap().iter().find(|t| t["name"] == "blk.0.ffn_down.weight").unwrap();
    let b = &t["marchenko_pastur"];
    let want = b["sigma_bulk"].as_f64().unwrap() * (1.0 + (D_IN as f64 / D_OUT as f64).sqrt());
    assert!((b["sigma_mp"].as_f64().unwrap() - want).abs() <= 1e-12 * want, "{}", b);
    assert_eq!(b["d_out"], D_OUT);
    assert_eq!(b["d_in"], D_IN);
}

#[test]
fn planted_rank_one_edit_is_ranked_flagged_and_names_its_tokens() {
    let (clean, spiked, e) = tensors();
    let dir = Scratch::new("audit");
    let st = dir.path("m.safetensors");
    write(
        &st,
        &build_safetensors(
            &[
                f32t("blk.0.attn_q.weight", &[D_OUT, D_IN], &clean),
                f32t("blk.0.ffn_down.weight", &[D_OUT, D_IN], &spiked),
                f32t("output.weight", &[VOCAB, D_OUT], &e),
            ],
            &BTreeMap::new(),
        )
        .unwrap(),
    );
    let gg = dir.path("m.gguf");
    write(
        &gg,
        &build_gguf(
            &[],
            &[q8t("blk.0.attn_q.weight", &clean), q8t("blk.0.ffn_down.weight", &spiked), f32t("output.weight", &[VOCAB, D_OUT], &e)],
            32,
        )
        .unwrap(),
    );
    for (model, what) in [(&st, "safetensors F32"), (&gg, "gguf Q8_0")] {
        let (code, v, out) = secant(&["audit-trojan", "--model", model, "--tensor", "blk.*", "--vocab-tensor", "output.weight", "--json"]);
        assert_eq!(code, 0, "{}: {}", what, out);
        check_ranking(&v, what);
        assert_eq!(v["options"]["detect_anomalies"], true);
        assert_eq!(v["options"]["marchenko_pastur"], true);
        assert_eq!(v["options"]["mp_threshold"], 1.5);
        // read --marchenko-pastur produces the same ranking
        let (code, r, _) =
            secant(&["read", "--model", model, "--tensor", "blk.*", "--vocab-tensor", "output.weight", "--marchenko-pastur", "--json"]);
        assert_eq!(code, 0);
        assert_eq!(r["spectral_outliers"], v["spectral_outliers"], "{}", what);
    }
    // a threshold above the spike unflags it but keeps the ratio
    let (_, v, _) = secant(&["audit-trojan", "--model", &st, "--tensor", "blk.*", "--mp-threshold", "100", "--json"]);
    assert!(!v["spectral_outliers"][0]["flagged"].as_bool().unwrap());
    assert!(v["spectral_outliers"][0]["sigma1_over_mp"].as_f64().unwrap() > 3.0);
    // human report: ranking table, not JSON, informational exit 0
    let (code, _, text) = secant(&["audit-trojan", "--model", &st, "--tensor", "blk.*", "--vocab-tensor", "output.weight"]);
    assert_eq!(code, 0);
    assert!(
        text.contains("Spectral trojan audit")
            && text.contains("blk.0.ffn_down.weight")
            && text.contains("n/a (1 in family)")
            && text.contains("7("),
        "{}",
        text
    );
    let (_, _, text) = secant(&["read", "--model", &st, "--tensor", "blk.0.ffn_down.weight", "--marchenko-pastur"]);
    assert!(text.contains("OUTSIDE-BULK"), "{}", text);
}

#[test]
fn a_planted_layer_stands_out_from_its_siblings() {
    // Six noise layers of one family with DIFFERENT amplitudes (the ratio is
    // scale-free, so all sit near 1) and a rank-1 edit in layer 3. Only layer 3
    // is a sibling outlier; the singleton family is too small to judge.
    let dir = Scratch::new("audit-siblings");
    let (_, spiked, _) = tensors();
    let mut ts = Vec::new();
    for l in 0..6usize {
        let w = if l == 3 { spiked.clone() } else { noise(D_OUT * D_IN, 100 + l as u64, 0.01 * (l + 1) as f64) };
        ts.push(f32t(&format!("blk.{}.ffn_down.weight", l), &[D_OUT, D_IN], &w));
    }
    ts.push(f32t("output.weight", &[VOCAB, D_OUT], &noise(VOCAB * D_OUT, 1, 0.01)));
    let p = dir.path("fam.safetensors");
    write(&p, &build_safetensors(&ts, &BTreeMap::new()).unwrap());
    let (code, v, out) = secant(&["audit-trojan", "--model", &p, "--json"]);
    assert_eq!(code, 0, "{}", out);
    let rank = v["spectral_outliers"].as_array().unwrap();
    let get = |n: &str| rank.iter().find(|r| r["tensor"] == n).unwrap_or_else(|| panic!("{} missing", n)).clone();
    for l in 0..6 {
        let r = get(&format!("blk.{}.ffn_down.weight", l));
        assert_eq!(r["sibling"]["family"], "blk.*.ffn_down.weight");
        assert_eq!(r["sibling"]["family_size"], 6);
        assert_eq!(r["sibling"]["sibling_outlier"], l == 3, "layer {}: {}", l, r);
    }
    assert!(get("blk.3.ffn_down.weight")["sibling"]["robust_z"].as_f64().unwrap() > 3.5);
    let o = get("output.weight");
    assert_eq!(o["sibling"]["family_size"], 1);
    assert!(o["sibling"]["sibling_outlier"].is_null() && o["sibling"].get("robust_z").is_none());
    let (_, _, text) = secant(&["audit-trojan", "--model", &p]);
    assert!(text.contains("Sibling outliers") && text.contains("1 sibling outlier(s)"), "{}", text);
}

#[test]
fn default_read_receipt_is_unchanged_and_bad_flags_are_usage_errors() {
    let (clean, _, _) = tensors();
    let dir = Scratch::new("audit-default");
    let st = dir.path("m.safetensors");
    write(&st, &build_safetensors(&[f32t("w", &[D_OUT, D_IN], &clean)], &BTreeMap::new()).unwrap());
    let (code, _, raw) = secant(&["read", "--model", &st, "--detect-anomalies", "--json"]);
    assert_eq!(code, 0);
    for key in ["marchenko_pastur", "mp_threshold", "spectral_outliers"] {
        assert!(!raw.contains(key), "default read must not carry {}", key);
    }
    for bad in [
        vec!["read", "--model", st.as_str(), "--mp-threshold", "2"],
        vec!["read", "--model", st.as_str(), "--marchenko-pastur", "--mp-threshold", "-1"],
        vec!["audit-trojan", "--model", st.as_str(), "--mp-threshold", "0"],
        vec!["audit-trojan"],
    ] {
        assert_eq!(secant(&bad).0, 2, "{:?}", bad);
    }
    // a rank-deficient tensor fully explained by its computed spectrum is undefined, not flagged
    let r1: Vec<f64> = (0..8 * 32).map(|k| ((k / 32) as f64 + 1.0) * ((k % 32) as f64 - 15.5)).collect();
    let lr = dir.path("lr.safetensors");
    write(&lr, &build_safetensors(&[f32t("w", &[8, 32], &r1)], &BTreeMap::new()).unwrap());
    let (code, v, _) = secant(&["audit-trojan", "--model", &lr, "--json"]);
    assert_eq!(code, 0);
    let b = &v["tensors"][0]["marchenko_pastur"];
    assert!(b["status"].as_str().unwrap().starts_with("undefined"), "{}", b);
    assert_eq!(b["flagged"], false);
    assert!(b.get("sigma1_over_mp").is_none());
    assert_eq!(v["spectral_outliers"].as_array().unwrap().len(), 0, "undefined tensors are not ranked");
}
