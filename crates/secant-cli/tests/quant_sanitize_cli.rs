//! End-to-end tests of `secant sanitize` on quantized GGUF files (Milestone
//! S-4): Q8_0 and Q4_K scale headers (`d`, `dmin`) are scanned for NaN, Inf,
//! subnormal and extreme values instead of the whole file being refused, the
//! opt-in repairs rewrite only the offending f16 fields, and every other
//! quantized type is still refused with exit 2.

#[path = "../../secant-core/tests/common/mod.rs"]
mod common;

use common::*;
use secant_core::builder::*;
use secant_core::container::scan_gguf_layout;
use secant_core::quant::*;
use serde_json::Value;
use std::process::Command;

const D_IN: usize = 512;
const D_OUT: usize = 4;

fn secant(args: &[&str]) -> (i32, Value, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_secant")).args(args).output().unwrap();
    let v = serde_json::from_slice(&o.stdout).unwrap_or(Value::Null);
    (o.status.code().unwrap_or(-1), v, String::from_utf8_lossy(&o.stderr).into_owned())
}

fn quantized(name: &str, fmt: QuantFormat, scale: f32) -> BuildTensor {
    let x: Vec<f32> = (0..D_OUT * D_IN).map(|i| ((i * 37 % 101) as f32 - 50.0) * scale).collect();
    let (bytes, _) = roundtrip_matrix(fmt, &x, D_OUT, D_IN).unwrap();
    BuildTensor { name: name.into(), dtype: fmt.name().into(), shape: vec![D_OUT, D_IN], data: bytes }
}

/// A GGUF holding one Q8_0, one Q4_K and one F32 norm tensor; returns the
/// bytes and the payload offsets of the two quantized tensors.
fn fixture() -> (Vec<u8>, i64, i64) {
    let norm = BuildTensor { name: "blk.0.attn_norm.weight".into(), dtype: "F32".into(), shape: vec![D_IN], data: f32_bytes(&[1.0; D_IN]) };
    let b = build_gguf(
        &[],
        &[quantized("blk.0.ffn_down.weight", QuantFormat::Q8_0, 0.01), quantized("blk.0.ssm_out.weight", QuantFormat::Q4K, 0.02), norm],
        32,
    )
    .unwrap();
    let l = scan_gguf_layout(&b).unwrap();
    (b, l.tensor("blk.0.ffn_down.weight").unwrap().offset, l.tensor("blk.0.ssm_out.weight").unwrap().offset)
}

fn set16(b: &mut [u8], off: i64, bits: u16) {
    b[off as usize..off as usize + 2].copy_from_slice(&bits.to_le_bytes());
}

fn get16(b: &[u8], off: i64) -> u16 {
    u16::from_le_bytes([b[off as usize], b[off as usize + 1]])
}

fn sanitize(dir: &Scratch, model: &str, tag: &str, extra: &[&str]) -> (i32, Value, String) {
    let receipt = dir.path(&format!("{}.receipt.json", tag));
    let mut args = vec!["sanitize", "--model", model, "--receipt", &receipt, "--attempt-id", "att-s4", "--json"];
    args.extend_from_slice(extra);
    let r = secant(&args);
    let on_disk: Value = serde_json::from_slice(&std::fs::read(&receipt).expect("receipt always written")).unwrap();
    assert_eq!(on_disk["status"], r.1["status"], "receipt file and --json agree");
    r
}

#[test]
fn genuine_q8_0_and_q4_k_files_scan_clean_instead_of_being_refused() {
    let dir = Scratch::new("s4-control");
    let (b, _, _) = fixture();
    let p = dir.path("control.gguf");
    write(&p, &b);
    let (code, rep, err) = sanitize(&dir, &p, "control", &[]);
    assert_eq!(code, 0, "{} {}", rep, err);
    assert_eq!(rep["status"], "clean");
    assert_eq!(rep["policy_version"], "unquantized-static-values-v1+quant-scale-headers-v1");
    let before = &rep["before"];
    assert_eq!(before["quant_tensors"], 2);
    assert_eq!(before["quant_blocks"], (D_OUT * D_IN / 32 + D_OUT * D_IN / 256) as i64);
    assert_eq!(before["float_elements"], D_IN as i64, "the F32 norm is still scanned element by element");
    assert_eq!(before["norm_rows"], 1);
    assert_eq!(before["total_findings"], 0);
    assert!(rep["input_unchanged"].as_bool().unwrap());
}

#[test]
fn crafted_scale_headers_are_found_and_repaired_field_by_field() {
    let dir = Scratch::new("s4-crafted");
    let (mut b, q8, q4) = fixture();
    let orig = b.clone();
    // Q8_0 block 3 d = NaN, block 5 d = 65504 (65504 * 127 > 1e4), Q4_K block 1
    // dmin = +Inf, Q4_K block 0 d = negative subnormal.
    let sites = [
        (q8 + 3 * 34, 0x7e00u16, "quant_scale_nan", "d", 3i64, "blk.0.ffn_down.weight"),
        (q8 + 5 * 34, 0x7bff, "quant_scale_extreme", "d", 5, "blk.0.ffn_down.weight"),
        (q4 + 144 + 2, 0x7c00, "quant_scale_inf", "dmin", 1, "blk.0.ssm_out.weight"),
        (q4, 0x8001, "quant_scale_subnormal", "d", 0, "blk.0.ssm_out.weight"),
    ];
    for s in &sites {
        set16(&mut b, s.0, s.1);
    }
    let p = dir.path("crafted.gguf");
    write(&p, &b);

    let (code, rep, _) = sanitize(&dir, &p, "scan", &[]);
    assert_eq!(code, 1, "findings remain: exit 1");
    assert_eq!(rep["status"], "findings");
    let before = &rep["before"];
    assert_eq!(before["total_findings"], 4);
    for s in &sites {
        assert_eq!(before["counts"][s.2], 1, "{}", s.2);
        let ex = before["examples"].as_array().unwrap().iter().find(|e| e["kind"] == s.2).expect(s.2);
        assert_eq!(ex["field"], s.3);
        assert_eq!(ex["index"], s.4, "block index");
        assert_eq!(ex["tensor"], s.5);
        assert_eq!(ex["file_offset"], s.0, "file_offset points at the f16 field");
        assert_eq!(ex["bits"], format!("0x{:04x}", s.1));
    }
    // row = block / blocks-per-row (512 / 32 = 16 Q8_0 blocks per row)
    assert_eq!(before["examples"].as_array().unwrap().iter().find(|e| e["kind"] == "quant_scale_nan").unwrap()["row"], 0);

    // --clamp-nonfinite alone leaves the extreme and subnormal scales: no publication
    let out1 = dir.path("partial.gguf");
    let (code, rep, _) = sanitize(&dir, &p, "partial", &["--clamp-nonfinite", "--out", &out1]);
    assert_eq!(code, 1);
    assert_eq!(rep["status"], "findings");
    assert!(!rep["output_published"].as_bool().unwrap());
    assert!(!std::path::Path::new(&out1).exists());
    assert_eq!(rep["changes"]["quant_scale_nan"], 1);
    assert_eq!(rep["changes"]["quant_scale_inf"], 1);
    assert_eq!(rep["after"]["total_findings"], 2);

    // all three repairs: remediated, published, and only the four fields differ
    let out = dir.path("clean.gguf");
    let (code, rep, err) = sanitize(&dir, &p, "full", &["--clamp-nonfinite", "--clamp-extreme", "--flush-subnormal", "--out", &out]);
    assert_eq!(code, 0, "{} {}", rep, err);
    assert_eq!(rep["status"], "remediated");
    assert!(rep["output_published"].as_bool().unwrap());
    assert_eq!(rep["after"]["total_findings"], 0);
    assert_eq!(std::fs::read(&p).unwrap(), b, "input never modified");
    let c = std::fs::read(&out).unwrap();
    assert_eq!(c.len(), b.len());
    let changed: Vec<usize> = (0..c.len()).filter(|&i| c[i] != b[i]).collect();
    let allowed: Vec<usize> = sites.iter().flat_map(|s| [s.0 as usize, s.0 as usize + 1]).collect();
    assert!(changed.iter().all(|i| allowed.contains(i)), "only scale bytes change: {:?}", changed);
    assert_eq!(get16(&c, sites[0].0), 0x0000, "NaN scale -> +0");
    assert_eq!(get16(&c, sites[3].0), 0x8000, "negative subnormal -> -0");
    for s in [&sites[1], &sites[2]] {
        let v = get16(&c, s.0);
        assert!(v & 0x8000 == 0 && v < 0x7c00 && v != 0, "{:#06x}", v);
    }
    // the clamped scales are bounded replacements, not the genuine values
    assert_ne!(get16(&c, sites[1].0), get16(&orig, sites[1].0));
    // the published copy rescans clean through the CLI
    let (code, rep, _) = sanitize(&dir, &out, "rescan", &[]);
    assert_eq!(code, 0);
    assert_eq!(rep["status"], "clean");
}

#[test]
fn other_quantized_types_are_still_refused_with_exit_2() {
    let dir = Scratch::new("s4-q4_0");
    let t = quantized("w", QuantFormat::Q8_0, 0.01);
    let mut b = build_gguf(&[], &[t], 32).unwrap();
    // retag the Q8_0 tensor (type 8) as Q4_0 (type 2): its 18-byte blocks fit inside the payload
    let pat: Vec<u8> = [1u64.to_le_bytes().to_vec(), vec![b'w'], 2u32.to_le_bytes().to_vec()].concat();
    let at = b.windows(pat.len()).position(|w| w == pat.as_slice()).unwrap() + pat.len() + 16;
    assert_eq!(u32::from_le_bytes(b[at..at + 4].try_into().unwrap()), 8);
    b[at..at + 4].copy_from_slice(&2u32.to_le_bytes());
    let p = dir.path("q4_0.gguf");
    write(&p, &b);
    let (code, rep, _) = sanitize(&dir, &p, "q4_0", &[]);
    assert_eq!(code, 2);
    assert_eq!(rep["status"], "unknown");
    assert!(rep["error"].as_str().unwrap().contains("unsupported ggml type 2 (quantized)"), "{}", rep["error"]);
}

#[test]
fn a_quantized_norm_tensor_is_unknown_not_clean() {
    let dir = Scratch::new("s4-qnorm");
    let b = build_gguf(&[], &[quantized("blk.0.ffn_norm.weight", QuantFormat::Q8_0, 0.01)], 32).unwrap();
    let p = dir.path("qnorm.gguf");
    write(&p, &b);
    let (code, rep, _) = sanitize(&dir, &p, "qnorm", &[]);
    assert_eq!(code, 2);
    assert_eq!(rep["status"], "unknown");
    assert!(rep["error"].as_str().unwrap().contains("norm-row checks are implemented for unquantized floats only"));
}
