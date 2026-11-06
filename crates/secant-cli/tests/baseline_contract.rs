//! Direct stdout byte comparison against the preserved pre-change Secant build.
#[path = "../../secant-core/tests/common/mod.rs"]
mod common;
use common::*;
use secant_core::{
    builder::{build_safetensors, BuildTensor},
    subspace::npy_bytes_f64,
};
use std::{collections::BTreeMap, process::Command};

#[test]
fn preserved_baseline_json_and_exit_codes() {
    let baseline = match std::env::var("SECANT_BASELINE_BIN") {
        Ok(p) if !p.is_empty() => p,
        Ok(_) => panic!("SECANT_BASELINE_BIN present but empty"),
        Err(_) => {
            eprintln!("NOT_RUN: SECANT_BASELINE_BIN absent");
            return;
        }
    };
    let dir = Scratch::new("baseline");
    let base = dir.path("base.safetensors");
    let changed = dir.path("changed.safetensors");
    let tensor = |v: f64| BuildTensor { name: "w".into(), dtype: "F32".into(), shape: vec![2, 32], data: f32_bytes(&vec![v; 64]) };
    write(&base, &build_safetensors(&[tensor(0.25)], &BTreeMap::new()).unwrap());
    write(&changed, &build_safetensors(&[tensor(0.5)], &BTreeMap::new()).unwrap());
    let npy = dir.path("w.npy");
    write(&npy, &npy_bytes_f64(&vec![0.25; 64], &[2, 32]).unwrap());
    let dirs = dir.path("d.npy");
    write(&dirs, &npy_bytes_f64(&[1., 0.], &[2]).unwrap());
    let mut cases = vec![
        vec!["inspect", "--model", &base, "--json"],
        vec!["read", "--model", &base, "--json"],
        vec!["verify-patch", "--base", &base, "--patched", &base, "--json"],
        vec!["verify-patch", "--base", &base, "--patched", &changed, "--json"],
        vec!["verify-subspace", "--base", &base, "--healed", &changed, "--tensors", "w", "--vectors", &dirs, "--json"],
        vec!["quant", "roundtrip", "--npy", &npy, "--json"],
        vec!["quant", "witness", "--model", &base, "--tensor", "w", "--vectors", &dirs, "--json"],
    ];
    for cmd in ["inspect", "read", "compose", "patch", "verify-patch", "sanitize", "probe", "heal", "verify-subspace"] {
        cases.push(vec![cmd, "--json"]);
    }
    for args in cases {
        let old = Command::new(&baseline).args(&args).output().unwrap();
        let new = Command::new(env!("CARGO_BIN_EXE_secant")).args(&args).output().unwrap();
        assert_eq!(old.status.code(), new.status.code(), "{:?}", args);
        assert_eq!(old.stdout, new.stdout, "{:?}", args);
    }
}

#[test]
fn retain_benign_readiness_fixtures_when_requested() {
    let dir = match std::env::var("SECANT_EVIDENCE_DIR") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => return,
    };
    std::fs::create_dir_all(dir.join("corpus")).unwrap();
    let w: Vec<f64> = (0..48).map(|i| ((i * 17 % 29) as f64 - 14.) / 7.).collect();
    for (name, data) in [
        ("w.npy", npy_bytes_f64(&w, &[6, 8]).unwrap()),
        ("d.npy", npy_bytes_f64(&[1., 2., 3., 0., 0., 0.], &[6]).unwrap()),
        ("corpus/control.bin", vec![129; 8]),
        ("corpus/wide.bin", (0..128).collect()),
        (
            "model.safetensors",
            build_safetensors(
                &[BuildTensor { name: "w".into(), dtype: "F32".into(), shape: vec![6, 8], data: f32_bytes(&w) }],
                &BTreeMap::new(),
            )
            .unwrap(),
        ),
    ] {
        write(dir.join(name).to_str().unwrap(), &data);
    }
}
