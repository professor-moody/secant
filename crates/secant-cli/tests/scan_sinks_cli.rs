//! `secant scan-sinks` end to end: a fixture tree exercising every sink class,
//! file selection (extensions, hidden and VCS directories, symlinks, test
//! paths), comment and string blanking, CRLF, invalid UTF-8 and HTML escaping,
//! plus the CLI contract (exit codes, `--out` never clobbers, deterministic
//! digest). With `SECANT_ORACLE_BIN` set, the receipt is also compared against
//! the oracle's `wsp scan-sinks` byte for byte on the fixture and on every
//! directory listed in `SECANT_SINK_TREES` (colon-separated; absent => only
//! the fixture). Excluded from the comparison: the run timestamps, `schema`
//! (legacy namespace in the oracle) and the reworded third limitation.

#[path = "../../secant-core/tests/common/mod.rs"]
mod common;

use common::*;
use serde_json::Value;
use std::process::{Command, Output};

fn secant(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_secant")).args(args).output().unwrap()
}

fn put(root: &Scratch, rel: &str, data: &[u8]) {
    let p = root.0.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, data).unwrap();
}

fn fixture() -> Scratch {
    let s = Scratch::new("sinks");
    put(
        &s,
        "src/ggml-quants.c",
        b"static void dequantize_row_q8_0(const block_q8_0 * x, float * y, int k) {\n    const float id = 1.0f / x->d;\n    y[0] = id;\n}\nstatic void guarded(const block_q8_0 * x, float * y) {\n    if (x->d == 0.0f) return;\n    const float id = 1.0f / x->d;\n    y[0] = id;\n}\n",
    );
    put(
        &s,
        "src/moe/router.cpp",
        b"int route(const float * logits, int n) {\r\n    int e = (int) logits[0];\r\n    return e;\r\n}\r\nint safe_route(const float * logits, int n) {\r\n    if (std::isnan(logits[0])) return 0;\r\n    int e = (int) logits[0];\r\n    return e;\r\n}\r\nvoid gather(const int * expert_ids, float * out, const float * w, int n) {\r\n    out[0] = w[expert_ids[0]];\r\n}\r\nvoid gather_checked(const int * expert_ids, float * out, const float * w, int n) {\r\n    if (expert_ids[0] >= n) return;\r\n    out[0] = w[expert_ids[0]];\r\n}\r\n",
    );
    put(&s, "src/model_loader.cc", b"void load(int n_dims) {\n    GGML_ASSERT(n_dims <= 4 && n_dims > 0);\n    DCHECK(n_dims < 8);\n}\n");
    put(
        &s,
        "kernels/scale.cu",
        b"__global__ void k(const half * s, float * o) {\n    o[0] = x / s->scale;\n    o[1] = __half2float(scale[0]);\n}\n",
    );
    put(&s, "src/pick.rs", b"pub fn pick(weights: &[f32]) -> usize {\n    weights[0] as usize\n}\n");
    // comments, strings, invalid UTF-8 and HTML-escaped characters in a snippet
    put(&s, "src/notes.h", b"// int e = (int) logits[0];\n/* (int) logits[0]\n   still a comment */\nconst char * s = \"(int) logits[0]\";\ninline int f(float * probs) { return (int) probs[0] < 1 && 1 > 0 ? 1 : 0; } // \xff\xfe\n");
    put(&s, "tests/test_router.cpp", b"int t(const float * logits) {\n    return (int) logits[0];\n}\n");
    // never scanned: hidden and VCS directories, other extensions, a symlink
    put(&s, ".hidden/a.c", b"int z(float *logits) { return (int) logits[0]; }\n");
    put(&s, "src/.git/b.c", b"int z(float *logits) { return (int) logits[0]; }\n");
    put(&s, "node_modules/c.c", b"int z(float *logits) { return (int) logits[0]; }\n");
    put(&s, "src/readme.md", b"(int) logits[0]\n");
    std::os::unix::fs::symlink(s.0.join("src/ggml-quants.c"), s.0.join("src/link.c")).unwrap();
    s
}

fn json(o: &Output) -> Value {
    serde_json::from_slice(&o.stdout).unwrap_or_else(|e| panic!("{}: {}", e, String::from_utf8_lossy(&o.stdout)))
}

fn sites(v: &Value) -> Vec<(String, i64, String, bool)> {
    v["findings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            (
                f["file"].as_str().unwrap().into(),
                f["line"].as_i64().unwrap(),
                f["sink_class"].as_str().unwrap().into(),
                f["guard_present"].as_bool().unwrap(),
            )
        })
        .collect()
}

#[test]
fn fixture_receipt_selection_classes_guards_and_order() {
    let s = fixture();
    let src = s.0.to_str().unwrap();
    let o = secant(&["scan-sinks", "--source", src, "--json", "--attempt-id", "att-sinks"]);
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    let v = json(&o);
    assert_eq!(v["schema"], "secant.wsp-sink-scan.v1");
    assert_eq!(v["attempt_id"], "att-sinks");
    assert_eq!(v["options"]["top"], 100);
    // 7 scanned files: hidden/.git/node_modules dirs, .md and the symlink are not
    assert_eq!(v["files_scanned"], 7, "{}", v);
    assert_eq!(v["files_skipped"], 0);
    let got = sites(&v);
    let has = |f: &str, l: i64, c: &str, g: bool| got.contains(&(f.into(), l, c.into(), g));
    assert!(has("src/ggml-quants.c", 2, "QUANT_SCALE_DIV_OR_OVERFLOW", false), "{:?}", got);
    assert!(has("src/ggml-quants.c", 7, "QUANT_SCALE_DIV_OR_OVERFLOW", true), "{:?}", got);
    assert!(has("src/moe/router.cpp", 2, "FLOAT_TO_INDEX_CAST", false), "CRLF lines: {:?}", got);
    assert!(has("src/moe/router.cpp", 7, "FLOAT_TO_INDEX_CAST", true), "{:?}", got);
    assert!(has("src/moe/router.cpp", 11, "UNVALIDATED_ROUTING_OR_INDEX_TENSOR", false), "{:?}", got);
    // the bound check must be on an EARLIER line of the function (oracle rule)
    assert!(has("src/moe/router.cpp", 15, "UNVALIDATED_ROUTING_OR_INDEX_TENSOR", true), "{:?}", got);
    assert!(has("src/model_loader.cc", 2, "WEIGHT_VALUE_RELEASE_ASSERT", false), "{:?}", got);
    assert!(!got.iter().any(|x| x.0 == "src/model_loader.cc" && x.1 == 3), "DCHECK compiles out under NDEBUG: {:?}", got);
    assert!(has("kernels/scale.cu", 2, "QUANT_SCALE_DIV_OR_OVERFLOW", false), "division: {:?}", got);
    assert!(has("kernels/scale.cu", 3, "QUANT_SCALE_DIV_OR_OVERFLOW", false), "fp16 conversion: {:?}", got);
    assert!(has("src/pick.rs", 2, "FLOAT_TO_INDEX_CAST", false), "{:?}", got);
    // only line 5 of notes.h is code; its snippet keeps <, > and & and replaces invalid UTF-8
    let notes: Vec<_> = got.iter().filter(|x| x.0 == "src/notes.h").collect();
    assert_eq!(notes.len(), 1, "{:?}", notes);
    assert_eq!(notes[0].1, 5);
    let raw = String::from_utf8(o.stdout.clone()).unwrap();
    assert!(raw.contains("\\u003c 1 \\u0026\\u0026 1 \\u003e 0"), "Go encoding/json HTML escaping");
    assert!(raw.contains("\u{fffd}\u{fffd}"), "each invalid byte becomes U+FFFD");
    // ranking: scores descending, then file, line
    let f = v["findings"].as_array().unwrap();
    for w in f.windows(2) {
        let (a, b) = (w[0]["risk_score"].as_f64().unwrap(), w[1]["risk_score"].as_f64().unwrap());
        assert!(
            a > b
                || (a == b
                    && (w[0]["file"].as_str() < w[1]["file"].as_str()
                        || (w[0]["file"] == w[1]["file"] && w[0]["line"].as_i64() <= w[1]["line"].as_i64())))
        );
    }
    // the test-path hit is scaled by 0.2 and ranks last
    assert_eq!(f.last().unwrap()["file"], "tests/test_router.cpp");
    // counts are exact and agree with the list
    let total: i64 = v["class_counts"].as_object().unwrap().values().map(|x| x.as_i64().unwrap()).sum();
    assert_eq!(total, v["total_findings"].as_i64().unwrap());
    assert_eq!(total as usize, f.len());
}

#[test]
fn cli_contract_top_out_errors_determinism_and_human_report() {
    let s = fixture();
    let src = s.0.to_str().unwrap();
    // --top truncates the list but not the counts
    let v = json(&secant(&["scan-sinks", "--source", src, "--top", "2", "--json"]));
    assert_eq!(v["findings"].as_array().unwrap().len(), 2);
    assert!(v["truncated"].as_bool().unwrap());
    assert!(v["total_findings"].as_i64().unwrap() > 2);
    // <= 0 takes the default
    assert_eq!(json(&secant(&["scan-sinks", "--source", src, "--top", "-5", "--json"]))["top"], 100);
    // same tree, same digest and findings
    let a = json(&secant(&["scan-sinks", "--source", src, "--json"]));
    let b = json(&secant(&["scan-sinks", "--source", src, "--json"]));
    assert_eq!(a["tree_digest_sha256"], b["tree_digest_sha256"]);
    assert_eq!(a["findings"], b["findings"]);
    // a changed file changes the digest
    put(&s, "src/pick.rs", b"pub fn pick(weights: &[f32]) -> usize {\n    weights[1] as usize\n}\n");
    assert_ne!(json(&secant(&["scan-sinks", "--source", src, "--json"]))["tree_digest_sha256"], a["tree_digest_sha256"]);
    // --out writes a NEW receipt (parents created) and never clobbers
    let out = Scratch::new("sinks-out");
    let rec = out.path("nested/dir/sinks.json");
    let o = secant(&["scan-sinks", "--source", src, "--out", &rec]);
    assert_eq!(o.status.code(), Some(0));
    let on_disk: Value = serde_json::from_slice(&std::fs::read(&rec).unwrap()).unwrap();
    assert_eq!(on_disk["schema"], "secant.wsp-sink-scan.v1");
    assert!(std::fs::read(&rec).unwrap().ends_with(b"}\n"));
    let o = secant(&["scan-sinks", "--source", src, "--out", &rec]);
    assert_eq!(o.status.code(), Some(2), "existing receipt is refused");
    assert!(String::from_utf8_lossy(&o.stderr).contains("file exists"), "{}", String::from_utf8_lossy(&o.stderr));
    // human report by default: not JSON, names the classes
    let o = secant(&["scan-sinks", "--source", src]);
    let text = String::from_utf8(o.stdout).unwrap();
    assert!(text.contains("Weight-value sink scan") && text.contains("QUANT_SCALE_DIV_OR_OVERFLOW") && !text.trim_start().starts_with('{'));
    // usage and I/O errors: exit 2
    assert_eq!(secant(&["scan-sinks"]).status.code(), Some(2));
    assert_eq!(secant(&["scan-sinks", "--source", &s.path("missing")]).status.code(), Some(2));
    assert_eq!(secant(&["scan-sinks", "--source", &s.path("src/pick.rs")]).status.code(), Some(2));
}

/// The pretty receipt minus the lines that legitimately differ.
fn comparable(stdout: &[u8]) -> String {
    String::from_utf8_lossy(stdout)
        .lines()
        .filter(|l| {
            !l.contains("\"started_utc\"")
                && !l.contains("\"finished_utc\"")
                && !l.contains("\"schema\"")
                && !l.contains("a hit is a hypothesis")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn incomplete_and_skipped_files_match_the_oracle() {
    let s = fixture();
    // Add a file > 4 MiB and a file with a line > 1 MiB.
    put(&s, "src/oversize.c", &vec![b'a'; (4 << 20) + 16]);
    let mut long_line = b"int f(float *logits) {\n".to_vec();
    long_line.extend(std::iter::repeat_n(b'x', (1 << 20) + 7));
    long_line.extend_from_slice(b"\n  return (int) logits[0];\n}\n");
    put(&s, "src/long_line.c", &long_line);

    let src = s.0.to_str().unwrap();
    let r = secant(&["scan-sinks", "--source", src, "--top", "1000000", "--json"]);
    assert_eq!(r.status.code(), Some(0));
    let vr = json(&r);
    assert_eq!(vr["files_scanned"], 7);
    assert_eq!(vr["files_skipped"], 2);
    assert_eq!(vr["incomplete"], true);
    let skipped = vr["skipped"].as_array().unwrap();
    assert_eq!(skipped.len(), 2);
    assert_eq!(skipped[0]["path"], "src/long_line.c");
    assert_eq!(skipped[0]["reason"], "scan error: bufio.Scanner: token too long");
    assert_eq!(skipped[1]["path"], "src/oversize.c");
    assert_eq!(skipped[1]["reason"], "file exceeds 4194304 bytes");
    assert!(
        vr["limitations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|l| l.as_str() == Some("INCOMPLETE: 2 file(s) were skipped (see skipped); an absence of findings is not a clean result"))
    );

    let Some(go) = oracle_bin() else { return not_run("incomplete_and_skipped_files_match_the_oracle") };
    let g = Command::new(&go).args(["wsp", "scan-sinks", "--source", src, "--top", "1000000", "--json"]).output().unwrap();
    assert_eq!(g.status.code(), r.status.code());
    assert_eq!(comparable(&g.stdout), comparable(&r.stdout));
}

#[test]
fn receipt_is_byte_identical_to_the_oracle() {
    let Some(go) = oracle_bin() else { return not_run("receipt_is_byte_identical_to_the_oracle") };
    let s = fixture();
    let mut trees = vec![s.0.to_string_lossy().into_owned()];
    match std::env::var("SECANT_SINK_TREES") {
        Err(std::env::VarError::NotPresent) => {}
        Ok(v) if v.is_empty() => panic!("SECANT_SINK_TREES is present but empty"),
        Ok(v) => trees.extend(v.split(':').map(String::from)),
        Err(e) => panic!("SECANT_SINK_TREES: {}", e),
    }
    for t in &trees {
        let g = Command::new(&go).args(["wsp", "scan-sinks", "--source", t, "--top", "1000000", "--json"]).output().unwrap();
        for fma in [true, false] {
            let mut args = vec!["scan-sinks", "--source", t, "--top", "1000000", "--json"];
            if fma {
                args.insert(0, "--go-arm64-fma");
            }
            let r = secant(&args);
            assert_eq!(g.status.code(), r.status.code(), "{}", t);
            let (cg, cr) = (comparable(&g.stdout), comparable(&r.stdout));
            if cg != cr {
                let (vg, vr): (Value, Value) = (serde_json::from_slice(&g.stdout).unwrap(), serde_json::from_slice(&r.stdout).unwrap());
                panic!(
                    "[{} fma={}] receipt differs: files {} vs {}, lines {} vs {}, findings {} vs {}",
                    t,
                    fma,
                    vg["files_scanned"],
                    vr["files_scanned"],
                    vg["lines_scanned"],
                    vr["lines_scanned"],
                    vg["total_findings"],
                    vr["total_findings"]
                );
            }
            let v: Value = serde_json::from_slice(&g.stdout).unwrap();
            println!(
                "PARITY {} fma={}: files={} lines={} findings={} digest={}",
                t, fma, v["files_scanned"], v["lines_scanned"], v["total_findings"], v["tree_digest_sha256"]
            );
        }
    }
}
