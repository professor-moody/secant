//! End-to-end CLI tests for `secant sanitize --sarif <path>` (SARIF 2.1.0).

#[path = "../../secant-core/tests/common/mod.rs"]
mod common;

use common::*;
use secant_core::builder::*;
use secant_core::container::scan_gguf_layout;
use secant_core::quant::*;
use secant_core::sarif::{SARIF_SCHEMA, TOOL_INFO_URI};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::process::Command;

fn secant(args: &[&str]) -> (i32, Value, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_secant")).args(args).output().unwrap();
    let v = serde_json::from_slice(&o.stdout).unwrap_or(Value::Null);
    (o.status.code().unwrap_or(-1), v, String::from_utf8_lossy(&o.stderr).into_owned())
}

fn assert_sarif_envelope(sarif: &Value, expected_attempt: &str) {
    assert_eq!(sarif["$schema"], SARIF_SCHEMA);
    assert_eq!(sarif["version"], "2.1.0");
    let runs = sarif["runs"].as_array().expect("runs array");
    assert_eq!(runs.len(), 1);
    let run = &runs[0];
    let driver = &run["tool"]["driver"];
    assert_eq!(driver["name"], "secant");
    assert_eq!(driver["semanticVersion"], env!("CARGO_PKG_VERSION"));
    assert_eq!(driver["informationUri"], TOOL_INFO_URI);
    assert_eq!(run["automationDetails"]["id"], format!("secant/sanitize/{}", expected_attempt));

    let rules = driver["rules"].as_array().expect("driver.rules array");
    assert!(!rules.is_empty(), "driver.rules must be populated");
    let mut ids = Vec::with_capacity(rules.len());
    for rule in rules {
        let id = rule["id"].as_str().expect("rule.id string");
        assert!(!id.is_empty());
        assert!(!rule["name"].as_str().expect("rule.name").is_empty());
        assert!(!rule["shortDescription"]["text"].as_str().expect("shortDescription.text").is_empty());
        let lvl = rule["defaultConfiguration"]["level"].as_str().expect("defaultConfiguration.level");
        assert!(matches!(lvl, "error" | "warning" | "note"));
        let tags = rule["properties"]["tags"].as_array().expect("rule.properties.tags");
        assert!(!tags.is_empty());
        ids.push(id.to_string());
    }
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ids, sorted, "driver.rules must be deterministically sorted by id");

    let results = run["results"].as_array().expect("results array");
    for res in results {
        let rule_id = res["ruleId"].as_str().expect("result.ruleId");
        let rule_idx = res["ruleIndex"].as_u64().expect("result.ruleIndex") as usize;
        assert_eq!(rules[rule_idx]["id"].as_str(), Some(rule_id), "ruleIndex must match ruleId");
        let level = res["level"].as_str().expect("result.level");
        assert!(matches!(level, "error" | "warning"));
        let msg = res["message"]["text"].as_str().expect("result.message.text");
        assert!(!msg.is_empty());
        assert!(!msg.contains('\u{2014}'), "SARIF messages must not contain em-dashes");

        let locs = res["locations"].as_array().expect("result.locations");
        assert_eq!(locs.len(), 1);
        let phys = &locs[0]["physicalLocation"];
        let uri = phys["artifactLocation"]["uri"].as_str().expect("artifactLocation.uri");
        assert!(!uri.is_empty(), "artifactLocation.uri must not be empty");
        assert!(!uri.starts_with('/'), "artifactLocation.uri must be relative, got {}", uri);
        assert!(!uri.starts_with("./"), "artifactLocation.uri must strip leading ./, got {}", uri);
        assert!(!uri.contains('\\'), "artifactLocation.uri must use forward slashes, got {}", uri);
        assert!(!uri.contains(".."), "artifactLocation.uri must not contain parent traversal, got {}", uri);

        let region = &phys["region"];
        for field in ["startLine", "startColumn", "endLine", "endColumn"] {
            let v = region[field].as_u64().unwrap_or(0);
            assert!(v >= 1, "region.{} must be >= 1, got {}", field, v);
        }

        let fp = res["partialFingerprints"]["primaryLocationLineHash"].as_str().expect("primaryLocationLineHash");
        assert_eq!(fp.len(), 64, "primaryLocationLineHash must be 64-char SHA-256 hex");
        assert!(fp.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')));
    }
}

#[test]
fn clean_safetensors_writes_empty_sarif_and_exits_zero() {
    let dir = Scratch::new("sarif-clean");
    let model = dir.path("clean.safetensors");
    let receipt = dir.path("clean.receipt.json");
    let sarif_path = dir.path("clean.sarif");

    let bytes = build_safetensors(
        &[BuildTensor { name: "model.embed_tokens.weight".into(), dtype: "F32".into(), shape: vec![16, 4], data: f32_bytes(&[0.25; 64]) }],
        &BTreeMap::new(),
    )
    .unwrap();
    write(&model, &bytes);

    let (code, rep, err) =
        secant(&["sanitize", "--model", &model, "--receipt", &receipt, "--attempt-id", "clean-ci", "--sarif", &sarif_path, "--json"]);
    assert_eq!(code, 0, "{} {}", rep, err);
    assert_eq!(rep["status"], "clean");

    let sarif: Value = serde_json::from_slice(&std::fs::read(&sarif_path).expect("sarif written")).unwrap();
    assert_sarif_envelope(&sarif, "clean-ci");
    assert_eq!(sarif["runs"][0]["results"].as_array().unwrap().len(), 0);
}

#[test]
fn violating_safetensors_with_nan_inf_and_s0_companion_writes_sarif_and_exits_one() {
    let dir = Scratch::new("sarif-violating-st");
    let model = dir.path("poisoned.safetensors");
    let cfg = dir.path("config.json");
    let receipt1 = dir.path("scan1.receipt.json");
    let receipt2 = dir.path("scan2.receipt.json");
    let sarif1_path = dir.path("scan1.sarif");
    let sarif2_path = dir.path("scan2.sarif");

    let mut weights = vec![0.5f64; 64];
    weights[2] = f64::NAN;
    weights[5] = f64::NAN;
    weights[9] = f64::INFINITY;
    let bytes = build_safetensors(
        &[BuildTensor { name: "model.embed_tokens.weight".into(), dtype: "F32".into(), shape: vec![16, 4], data: f32_bytes(&weights) }],
        &BTreeMap::new(),
    )
    .unwrap();
    write(&model, &bytes);
    write(&cfg, br#"{"vocab_size": 16, "eos_token_id": 99}"#);

    let (code1, rep1, err1) =
        secant(&["sanitize", "--model", &model, "--receipt", &receipt1, "--attempt-id", "ci-det", "--sarif", &sarif1_path, "--json"]);
    assert_eq!(code1, 1, "findings must exit 1: {} {}", rep1, err1);
    assert_eq!(rep1["status"], "findings");

    let sarif1: Value = serde_json::from_slice(&std::fs::read(&sarif1_path).expect("sarif1 written")).unwrap();
    assert_sarif_envelope(&sarif1, "ci-det");
    let results = sarif1["runs"][0]["results"].as_array().unwrap();
    assert_eq!(results.len(), 3, "expected nan, inf, and s0_companion_token_id_oob results: {}", sarif1);

    let rule_ids: HashSet<&str> = results.iter().map(|r| r["ruleId"].as_str().unwrap()).collect();
    assert!(rule_ids.contains("SECANT-F-NAN"));
    assert!(rule_ids.contains("SECANT-F-INF"));
    assert!(rule_ids.contains("SECANT-S0-COMPANION-TOKEN-ID-OOB"));

    let nan_res = results.iter().find(|r| r["ruleId"] == "SECANT-F-NAN").unwrap();
    assert_eq!(nan_res["locations"][0]["physicalLocation"]["artifactLocation"]["uri"], "poisoned.safetensors");
    assert_eq!(nan_res["properties"]["count"], 2);
    assert!(nan_res["message"]["text"].as_str().unwrap().contains("model.embed_tokens.weight"));
    assert!(nan_res["message"]["text"].as_str().unwrap().contains("byte offset"));

    let s0_res = results.iter().find(|r| r["ruleId"] == "SECANT-S0-COMPANION-TOKEN-ID-OOB").unwrap();
    assert_eq!(s0_res["locations"][0]["physicalLocation"]["artifactLocation"]["uri"], "config.json");
    assert_eq!(s0_res["properties"]["configKey"], "eos_token_id");
    assert!(s0_res["message"]["text"].as_str().unwrap().contains("eos_token_id"));

    // Determinism: second independent run produces identical SARIF & partialFingerprints.
    let (code2, rep2, err2) =
        secant(&["sanitize", "--model", &model, "--receipt", &receipt2, "--attempt-id", "ci-det", "--sarif", &sarif2_path, "--json"]);
    assert_eq!(code2, 1, "{} {}", rep2, err2);
    let sarif2: Value = serde_json::from_slice(&std::fs::read(&sarif2_path).expect("sarif2 written")).unwrap();
    assert_eq!(sarif1, sarif2, "two runs on the same input must emit identical SARIF");
}

#[test]
fn quantized_gguf_findings_and_remediation_update_sarif_cleanly() {
    let dir = Scratch::new("sarif-gguf");
    let d_in = 512usize;
    let d_out = 2usize;
    let x: Vec<f32> = (0..d_out * d_in).map(|i| ((i * 29 % 89) as f32 - 44.0) * 0.01).collect();
    let (q8_bytes, _) = roundtrip_matrix(QuantFormat::Q8_0, &x, d_out, d_in).unwrap();
    let mut gguf = build_gguf(
        &[],
        &[BuildTensor { name: "blk.0.ffn_down.weight".into(), dtype: "Q8_0".into(), shape: vec![d_out, d_in], data: q8_bytes }],
        32,
    )
    .unwrap();
    let layout = scan_gguf_layout(&gguf).unwrap();
    let q8_off = layout.tensor("blk.0.ffn_down.weight").unwrap().offset as usize;
    // Poison block 2 scale header `d` with FP16 NaN (0x7e00).
    let bad_off = q8_off + 2 * 34;
    gguf[bad_off..bad_off + 2].copy_from_slice(&0x7e00u16.to_le_bytes());

    let model = dir.path("bad-q8.gguf");
    write(&model, &gguf);

    let scan_receipt = dir.path("scan.receipt.json");
    let scan_sarif = dir.path("scan.sarif");
    let (code, _, err) =
        secant(&["sanitize", "--model", &model, "--receipt", &scan_receipt, "--attempt-id", "gguf-ci", "--sarif", &scan_sarif, "--json"]);
    assert_eq!(code, 1, "{}", err);
    let sarif: Value = serde_json::from_slice(&std::fs::read(&scan_sarif).unwrap()).unwrap();
    assert_sarif_envelope(&sarif, "gguf-ci");
    let results = sarif["runs"][0]["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["ruleId"], "SECANT-B3-2A-QUANT-SCALE-NAN");
    assert_eq!(results[0]["properties"]["firstIndex"], 2);
    assert_eq!(results[0]["properties"]["firstField"], "d");
    assert_eq!(results[0]["properties"]["firstFileOffset"], bad_off as i64);
    assert!(results[0]["message"]["text"].as_str().unwrap().contains("block index 2"));

    // Remediating clears findings -> exit 0 and empty SARIF results.
    let fixed = dir.path("fixed-q8.gguf");
    let fix_receipt = dir.path("fix.receipt.json");
    let fix_sarif = dir.path("fix.sarif");
    let (code_fix, rep_fix, err_fix) = secant(&[
        "sanitize",
        "--model",
        &model,
        "--clamp-nonfinite",
        "--out",
        &fixed,
        "--receipt",
        &fix_receipt,
        "--attempt-id",
        "gguf-fix",
        "--sarif",
        &fix_sarif,
        "--json",
    ]);
    assert_eq!(code_fix, 0, "{} {}", rep_fix, err_fix);
    assert_eq!(rep_fix["status"], "remediated");
    let sarif_fixed: Value = serde_json::from_slice(&std::fs::read(&fix_sarif).unwrap()).unwrap();
    assert_sarif_envelope(&sarif_fixed, "gguf-fix");
    assert_eq!(sarif_fixed["runs"][0]["results"].as_array().unwrap().len(), 0);
}

#[test]
fn unwritable_or_invalid_sarif_path_fails_closed_with_exit_two() {
    let dir = Scratch::new("sarif-fail-closed");
    let model = dir.path("clean.safetensors");
    let bytes = build_safetensors(
        &[BuildTensor { name: "w".into(), dtype: "F32".into(), shape: vec![4], data: f32_bytes(&[1.0; 4]) }],
        &BTreeMap::new(),
    )
    .unwrap();
    write(&model, &bytes);

    // 1. Missing parent directory -> exit 2.
    let receipt1 = dir.path("r1.json");
    let bad_sarif = dir.path("no-such-dir/out.sarif");
    let (code1, _, err1) =
        secant(&["sanitize", "--model", &model, "--receipt", &receipt1, "--attempt-id", "fail-1", "--sarif", &bad_sarif]);
    assert_eq!(code1, 2, "missing sarif parent must exit 2: {}", err1);
    assert!(err1.contains("sarif publication"), "stderr should explain sarif failure: {}", err1);

    // 2. Existing directory as --sarif target -> exit 2.
    let receipt2 = dir.path("r2.json");
    let (code2, _, err2) =
        secant(&["sanitize", "--model", &model, "--receipt", &receipt2, "--attempt-id", "fail-2", "--sarif", &dir.0.to_string_lossy()]);
    assert_eq!(code2, 2, "directory sarif target must exit 2: {}", err2);

    // 3. Empty --sarif path -> exit 2.
    let receipt3 = dir.path("r3.json");
    let (code3, _, _) = secant(&["sanitize", "--model", &model, "--receipt", &receipt3, "--attempt-id", "fail-3", "--sarif", ""]);
    assert_eq!(code3, 2);

    // 4. --sarif colliding with --receipt -> exit 2.
    let receipt4 = dir.path("r4.json");
    let (code4, _, _) = secant(&["sanitize", "--model", &model, "--receipt", &receipt4, "--attempt-id", "fail-4", "--sarif", &receipt4]);
    assert_eq!(code4, 2);

    // 5. Missing model -> exit 2 and no misleading SARIF file created.
    let receipt5 = dir.path("r5.json");
    let sarif5 = dir.path("should-not-exist.sarif");
    let (code5, _, _) = secant(&[
        "sanitize",
        "--model",
        &dir.path("missing.safetensors"),
        "--receipt",
        &receipt5,
        "--attempt-id",
        "fail-5",
        "--sarif",
        &sarif5,
    ]);
    assert_eq!(code5, 2);
    assert!(!std::path::Path::new(&sarif5).exists(), "errored scan must not publish a SARIF file");

    // 6. Violating model + unwritable --sarif parent -> fail closed with exit 2 (not 1).
    let bad_model = dir.path("nan.safetensors");
    let bad_bytes = build_safetensors(
        &[BuildTensor { name: "w".into(), dtype: "F32".into(), shape: vec![2], data: f32_bytes(&[f64::NAN, 1.0]) }],
        &BTreeMap::new(),
    )
    .unwrap();
    write(&bad_model, &bad_bytes);
    let receipt6 = dir.path("r6.json");
    let (code6, _, err6) =
        secant(&["sanitize", "--model", &bad_model, "--receipt", &receipt6, "--attempt-id", "fail-6", "--sarif", &bad_sarif]);
    assert_eq!(code6, 2, "unwritable sarif on violating model must fail closed with exit 2: {}", err6);
}
