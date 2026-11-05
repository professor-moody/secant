//! Q8_0 / Q4_K kernel conformance against the UNMODIFIED ggml reference
//! (`ggml-quants.c` of a llama.cpp checkout), compiled with
//! `-ffp-contract=off` so the C reference is evaluated unfused.
//!
//! `SECANT_GGML_DIR` (three-state): absent => NOT_RUN; present-but-empty =>
//! hard error; value => path of a llama.cpp checkout (its `ggml/` directory).
//! Requires a C compiler named by `CC` (default `cc`).

mod common;
use common::*;
use secant_core::quant::*;
use std::process::Command;

fn ggml_dir() -> Option<String> {
    match std::env::var("SECANT_GGML_DIR") {
        Err(std::env::VarError::NotPresent) => None,
        Ok(v) if v.is_empty() => panic!("SECANT_GGML_DIR is present but empty: refusing to treat it as unset"),
        Ok(v) => Some(v),
        Err(e) => panic!("SECANT_GGML_DIR: {}", e),
    }
}

fn build_ref(scratch: &Scratch, ggml: &str, contract: &str) -> String {
    let here = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/ggml_ref");
    let cc = std::env::var("CC").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "cc".into());
    let (inc1, inc2) = (format!("{}/ggml/include", ggml), format!("{}/ggml/src", ggml));
    let exe = scratch.path(&format!("ggml_ref_{}", contract));
    let objs: Vec<(String, String)> = vec![
        (format!("{}/ggml_ref.c", here), scratch.path(&format!("a_{}.o", contract))),
        (format!("{}/ggml/src/ggml-quants.c", ggml), scratch.path(&format!("b_{}.o", contract))),
        (format!("{}/stubs.c", here), scratch.path(&format!("c_{}.o", contract))),
    ];
    for (src, obj) in &objs {
        let st = Command::new(&cc)
            .args(["-O1", &format!("-ffp-contract={}", contract), "-std=gnu11", "-w", "-I", &inc1, "-I", &inc2, "-c", src, "-o", obj])
            .status()
            .expect("run C compiler");
        assert!(st.success(), "compiling {} failed", src);
    }
    let mut link = Command::new(&cc);
    for (_, o) in &objs {
        link.arg(o);
    }
    let st = link.args(["-lm", "-o", &exe]).status().unwrap();
    assert!(st.success(), "link failed");
    exe
}

fn inputs() -> Vec<(String, Vec<f32>)> {
    let mut rng = Rng(0xC0FFEE);
    let mut v: Vec<(String, Vec<f32>)> = Vec::new();
    for (name, scale) in [("uniform", 1.0), ("tiny", 1e-3), ("huge", 1e3), ("subnormal-scale", 1e-6)] {
        v.push((name.into(), rng.vec(2048).iter().map(|x| (*x * scale) as f32).collect()));
    }
    // heavy tail: cube of uniform
    v.push(("heavy-tail".into(), rng.vec(2048).iter().map(|x| (x * x * x * 40.0) as f32).collect()));
    // one-sided positive and negative blocks (min >= 0 branch)
    v.push(("positive".into(), rng.vec(1024).iter().map(|x| (x.abs() + 0.1) as f32).collect()));
    v.push(("negative".into(), rng.vec(1024).iter().map(|x| -(x.abs() + 0.1) as f32).collect()));
    // zero and constant blocks mixed in
    let mut mixed: Vec<f32> = rng.vec(1024).iter().map(|x| *x as f32).collect();
    for i in 0..256 {
        mixed[i] = 0.0;
    }
    for i in 256..512 {
        mixed[i] = 0.75;
    }
    v.push(("zero-and-constant-blocks".into(), mixed));
    v
}

fn run_quant(exe: &str, scratch: &Scratch, fmt: &str, x: &[f32], tag: &str) -> Vec<u8> {
    let (i, o) = (scratch.path(&format!("in_{}.f32", tag)), scratch.path(&format!("out_{}.bin", tag)));
    write(&i, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let st = Command::new(exe).args(["quant", fmt, &i, &o]).status().unwrap();
    assert!(st.success());
    std::fs::read(&o).unwrap()
}

fn run_dequant(exe: &str, scratch: &Scratch, fmt: &str, bytes: &[u8], n: usize, tag: &str) -> Vec<f32> {
    let (i, o) = (scratch.path(&format!("dq_in_{}.bin", tag)), scratch.path(&format!("dq_out_{}.f32", tag)));
    write(&i, bytes);
    let st = Command::new(exe).args(["dequant", fmt, &i, &o, &n.to_string()]).status().unwrap();
    assert!(st.success());
    std::fs::read(&o).unwrap().chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn kernels_match_ggml_reference() {
    let Some(ggml) = ggml_dir() else {
        println!("NOT_RUN kernels_match_ggml_reference: SECANT_GGML_DIR is absent");
        return;
    };
    let scratch = Scratch::new("ggml");
    let exe = build_ref(&scratch, &ggml, "off");
    let mut checked = 0;
    for (name, x) in inputs() {
        for (fmt, qf) in [("q8_0", QuantFormat::Q8_0), ("q4_k", QuantFormat::Q4K)] {
            let be = qf.block_elems();
            let n = x.len() / be * be;
            let x = &x[..n];
            let tag = format!("{}_{}", name, fmt);
            let reference = run_quant(&exe, &scratch, fmt, x, &tag);
            let mut mine = vec![0u8; n / be * qf.block_bytes()];
            quantize_row(qf, x, &mut mine).unwrap();
            assert_eq!(mine, reference, "{} {}: quantized bytes differ from ggml", name, fmt);
            // dequant parity both ways, bit-exact
            let c_dq = run_dequant(&exe, &scratch, fmt, &reference, n, &tag);
            let mut r_dq = vec![0f32; n];
            dequantize_row(qf, &reference, &mut r_dq).unwrap();
            for i in 0..n {
                assert_eq!(c_dq[i].to_bits(), r_dq[i].to_bits(), "{} {}: dequant differs at {}", name, fmt, i);
            }
            checked += 1;
            println!("PASS {:<28} {:<5} {} blocks: quantized bytes and dequantized floats bit-identical to ggml", name, fmt, n / be);
        }
    }
    println!("ggml_ref: {} input/format pairs bit-identical (ggml dir {})", checked, ggml);
}
