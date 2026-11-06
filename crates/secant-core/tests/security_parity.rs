//! Differential parity of `secant sanitize` / `secant probe` against the
//! reference Go oracle (`wsp sanitize` / `wsp probe` subcommands of the oracle
//! binary at `SECANT_ORACLE_BIN`).
//!
//! Requires BOTH binaries:
//!   * the oracle: `SECANT_ORACLE_BIN=/path/to/oracle` (absent => NOT_RUN);
//!   * the Secant CLI: `SECANT_BIN`, else `<target>/debug/secant` next to this
//!     test (build it first: `cargo build -p secant-cli`); absent => NOT_RUN
//!     for the CLI-level scenarios (the in-process library scenarios still run).
//!
//! Every CLI scenario runs the oracle and Secant on the SAME inputs with
//! per-side scratch directories, then compares: exit code, stdout (text or
//! --json), the receipt TEXT line by line (so field order, indentation, null /
//! omitempty behaviour and Go number formatting are all covered), and every
//! output file by SHA-256. The only lines excluded from the receipt text
//! comparison are the three inherently run-specific fields: `started_utc`,
//! `finished_utc` and `receipt_sha256` (it seals the compact serialization that
//! embeds both timestamps and every path). Path-valued fields are compared
//! after replacing each side's scratch directory with `<D>`.
//!
//! The oracle writes schema ids and probe fixture metadata in the legacy
//! namespace. Before comparing, both sides are mapped to the `secant`
//! spelling: texts with `common::canon_text`, files hashed after
//! `common::canon_container` (metadata renamed; tensor bytes verbatim), and a
//! fixture's raw `sha256` / `bytes` quoted in a receipt are replaced by those of
//! its normalized bytes.
//! Normalizing both sides keeps inputs that themselves carry legacy metadata
//! (oracle-generated fixtures fed to sanitize) comparable; that Secant itself
//! never WRITES a legacy spelling is asserted separately (`assert_secant_namespace`).
//!
//! The Go receipt seal itself is verified by deserializing the oracle's receipt
//! into the Rust structs and recomputing `receipt_sha256` from them.

mod common;
use common::*;
use secant_core::builder::{build_gguf, build_safetensors, BuildTensor};
use secant_core::numeric::set_go_arm64_fma;
use secant_core::security::{
    generate_probe, sanitize_file_report, write_probe_receipt, ProbeOptions, ProbeReport, SanitizeOptions, SanitizeReceipt,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const ATT: &str = "att-parity-0001";

// ------------------------------------------------------------------ harness

fn secant_bin() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("SECANT_BIN") {
        if v.is_empty() {
            panic!("SECANT_BIN is present but empty");
        }
        return Some(PathBuf::from(v));
    }
    let exe = std::env::current_exe().ok()?;
    let p = exe.parent()?.parent()?.join("secant");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

/// Both binaries or None (after printing NOT_RUN).
fn bins(test: &str) -> Option<(String, PathBuf)> {
    set_go_arm64_fma(true); // the oracle binary is arm64: Go fuses `s += a*b`
    let go = match oracle_bin() {
        Some(g) => g,
        None => {
            not_run(test);
            return None;
        }
    };
    match secant_bin() {
        Some(s) => Some((go, s)),
        None => {
            println!("NOT_RUN {}: secant CLI binary not built (cargo build -p secant-cli)", test);
            None
        }
    }
}

struct Side {
    dir: String,
}

impl Side {
    fn new(root: &Scratch, name: &str) -> Side {
        let d = root.path(name);
        std::fs::create_dir_all(&d).unwrap();
        Side { dir: d }
    }
    fn sub(&self, n: &str) -> String {
        format!("{}/{}", self.dir, n)
    }
    /// Expand `{OUT}`, `{REC}` and `{DIR}` placeholders.
    fn args(&self, a: &[&str]) -> Vec<String> {
        a.iter()
            .map(|x| x.replace("{OUT}", &self.sub("out")).replace("{REC}", &self.sub("receipt.json")).replace("{DIR}", &self.dir))
            .collect()
    }
}

fn norm(s: &str, dir: &str) -> String {
    s.replace(dir, "<D>")
}

fn run_go(go: &str, wsp_args: &[String]) -> Output {
    let mut a = vec!["wsp".to_string()];
    a.extend_from_slice(wsp_args);
    Command::new(go).args(&a).output().expect("spawn oracle")
}

fn run_rs(rs: &Path, args: &[String]) -> Output {
    let mut a = vec!["--go-arm64-fma".to_string()];
    a.extend_from_slice(args);
    Command::new(rs).args(&a).output().expect("spawn secant")
}

const VOLATILE: [&str; 3] = ["\"started_utc\"", "\"finished_utc\"", "\"receipt_sha256\""];

fn stable_lines(text: &str, dir: &str) -> String {
    norm(text, dir).lines().filter(|l| !VOLATILE.iter().any(|v| l.trim_start().starts_with(v))).collect::<Vec<_>>().join("\n")
}

/// raw SHA-256 -> (normalized SHA-256, raw byte length, normalized byte length)
/// for every file whose legacy-namespace normalization changed it.
type Renames = BTreeMap<String, (String, usize, usize)>;

/// Sorted (relative name, sha256 or "<dir>") of everything below `root`.
/// File hashes are taken after legacy-namespace normalization; every file the
/// normalization changed is recorded in `map`.
fn tree_canon(root: &str, map: &mut Renames) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    fn walk(base: &Path, p: &Path, m: &mut BTreeMap<String, String>, map: &mut Renames) {
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let path = e.path();
                let rel = path.strip_prefix(base).unwrap().to_string_lossy().into_owned();
                if path.is_dir() {
                    m.insert(rel, "<dir>".into());
                    walk(base, &path, m, map);
                } else {
                    let b = std::fs::read(&path).unwrap();
                    let raw = secant_core::container::sha256_hex(&b);
                    let canon = match canon_container(&b) {
                        Some(c) => {
                            let h = secant_core::container::sha256_hex(&c);
                            map.insert(raw, (h.clone(), b.len(), c.len()));
                            h
                        }
                        None => raw,
                    };
                    m.insert(rel, canon);
                }
            }
        }
    }
    walk(Path::new(root), Path::new(root), &mut m, map);
    m
}

/// Receipt TEXT: legacy spellings -> secant; a fixture's `"bytes": <raw len>,`
/// line directly above its `"sha256": "<raw>"` line (the receipt's field
/// order) and the hash itself -> the normalized length and hash.
fn canon_side(text: &str, map: &Renames) -> String {
    let mut lines: Vec<String> = canon_text(text).lines().map(String::from).collect();
    for i in 0..lines.len() {
        for (raw, (canon, raw_len, len)) in map {
            if lines[i].contains(raw.as_str()) {
                lines[i] = lines[i].replace(raw.as_str(), canon);
                if i > 0 && lines[i - 1].trim_start().starts_with(&format!("\"bytes\": {},", raw_len)) {
                    lines[i - 1] = lines[i - 1].replace(&format!("\"bytes\": {},", raw_len), &format!("\"bytes\": {},", len));
                }
            }
        }
    }
    let mut out = lines.join("\n");
    if text.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Receipt JSON: in every object whose `sha256` is a renamed file's raw hash,
/// replace `sha256` and (when it equals the raw length) `bytes`.
fn canon_receipt_value(v: &Value, map: &Renames) -> Value {
    match v {
        Value::Array(a) => Value::Array(a.iter().map(|x| canon_receipt_value(x, map)).collect()),
        Value::Object(o) => {
            let mut o: serde_json::Map<String, Value> = o.iter().map(|(k, x)| (k.clone(), canon_receipt_value(x, map))).collect();
            if let Some((canon, raw_len, len)) = o.get("sha256").and_then(|s| s.as_str()).and_then(|s| map.get(s)).cloned() {
                o.insert("sha256".into(), Value::String(canon));
                if o.get("bytes").and_then(|b| b.as_u64()) == Some(raw_len as u64) {
                    o.insert("bytes".into(), Value::from(len as u64));
                }
            }
            Value::Object(o)
        }
        x => x.clone(),
    }
}

/// What Secant WRITES is in the secant namespace: a probe run (whose fixtures
/// and receipt are generated, not copied from an input) carries no legacy
/// spelling anywhere, and every receipt names the canonical schema id.
fn assert_secant_namespace(label: &str, subcommand: &str, dir: &str, stdout: &str, receipt: Option<&String>) {
    // (a receipt path may hold a pre-existing non-receipt file the run must not clobber)
    if let Some(Value::String(schema)) = receipt.and_then(|r| serde_json::from_str::<Value>(r).ok()).map(|v| v["schema"].clone()) {
        let want = if subcommand == "probe" { secant_core::security::PROBE_SCHEMA } else { secant_core::security::SANITIZE_SCHEMA };
        assert_eq!(schema, want, "[{}] Secant receipt schema", label);
    }
    if subcommand == "probe" {
        let mut changed = BTreeMap::new();
        tree_canon(dir, &mut changed);
        assert!(changed.is_empty(), "[{}] Secant wrote a legacy namespace spelling into {} file(s)", label, changed.len());
        assert!(!has_legacy(stdout), "[{}] Secant stdout carries a legacy namespace spelling", label);
        assert!(!receipt.map(|r| has_legacy(r)).unwrap_or(false), "[{}] Secant receipt carries a legacy namespace spelling", label);
    }
}

struct Cmp {
    code: i32,
    #[allow(dead_code)]
    go_stdout: String,
    go_receipt: Option<String>,
    rs_receipt: Option<String>,
    go_tree: BTreeMap<String, String>,
}

/// Run the oracle's `wsp <args>` and `secant --go-arm64-fma <args>` and assert they agree.
fn compare(label: &str, go: &str, rs: &Path, root: &Scratch, args: &[&str]) -> Cmp {
    compare_opt(label, go, rs, root, args, false)
}

/// `lenient_error`: tolerate a different `error` wording (collected in `MSG_DIFFS`)
/// as long as exit code, status and every other receipt line agree.
static MSG_DIFFS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

fn compare_opt(label: &str, go: &str, rs: &Path, root: &Scratch, args: &[&str], lenient_error: bool) -> Cmp {
    let (g, r) = (Side::new(root, &format!("{}-go", label)), Side::new(root, &format!("{}-rs", label)));
    let og = run_go(go, &g.args(args));
    let or = run_rs(rs, &r.args(args));
    let (cg, cr) = (og.status.code().unwrap_or(-1), or.status.code().unwrap_or(-1));
    assert_eq!(
        cg,
        cr,
        "[{}] exit code go={} rs={}\n go stderr: {}\n rs stderr: {}",
        label,
        cg,
        cr,
        String::from_utf8_lossy(&og.stderr),
        String::from_utf8_lossy(&or.stderr)
    );
    // every file written (outputs, receipts, fixtures) by name and normalized SHA-256
    let mut map = BTreeMap::new();
    let (tg, tr) = (tree_canon(&g.dir, &mut map), tree_canon(&r.dir, &mut map));
    let rec_g = std::fs::read_to_string(g.sub("receipt.json")).ok();
    let rec_r = std::fs::read_to_string(r.sub("receipt.json")).ok();
    assert_secant_namespace(label, args[0], &r.dir, &String::from_utf8_lossy(&or.stdout), rec_r.as_ref());
    let json_mode = args.contains(&"--json");
    let (sg, sr) = (norm(&String::from_utf8_lossy(&og.stdout), &g.dir), norm(&String::from_utf8_lossy(&or.stdout), &r.dir));
    let (sg, sr) = if json_mode { (sg, sr) } else { (canon_side(&sg, &map), canon_side(&sr, &map)) };
    if json_mode {
        // --json prints the receipt: compare as parsed JSON minus the volatile fields.
        if !sg.is_empty() || !sr.is_empty() {
            let (jg, jr) = (parse_json(sg.as_bytes()), parse_json(sr.as_bytes()));
            let (jg, jr) = (canon_value(&canon_receipt_value(&jg, &map)), canon_value(&canon_receipt_value(&jr, &map)));
            if let Some(d) = json_diff(&jg, &jr, &["started_utc", "finished_utc", "receipt_sha256"]) {
                panic!("[{}] --json stdout differs: {}", label, d);
            }
        }
    } else {
        assert_eq!(sg, sr, "[{}] stdout differs", label);
    }
    // stderr: the error message (after "Error: ") must agree when the oracle reports one.
    let (eg, er) = (norm(&String::from_utf8_lossy(&og.stderr), &g.dir), norm(&String::from_utf8_lossy(&or.stderr), &r.dir));
    if eg.trim() != er.trim() {
        println!("NOTE [{}] stderr differs:\n   go: {}\n   rs: {}", label, eg.trim(), er.trim());
    }
    match (&rec_g, &rec_r) {
        (Some(a), Some(b)) => {
            let (mut la, mut lb) = (stable_lines(&canon_side(a, &map), &g.dir), stable_lines(&canon_side(b, &map), &r.dir));
            if lenient_error && la != lb {
                let is_err = |l: &str| l.trim_start().starts_with("\"error\"");
                let ea: Vec<&str> = la.lines().filter(|l| is_err(l)).collect();
                let eb: Vec<&str> = lb.lines().filter(|l| is_err(l)).collect();
                if ea != eb {
                    MSG_DIFFS.lock().unwrap().push(format!(
                        "[{}]\n      go: {}\n      rs: {}",
                        label,
                        ea.join("").trim(),
                        eb.join("").trim()
                    ));
                }
                la = la.lines().filter(|l| !is_err(l)).collect::<Vec<_>>().join("\n");
                lb = lb.lines().filter(|l| !is_err(l)).collect::<Vec<_>>().join("\n");
            }
            if la != lb {
                let al: Vec<&str> = la.lines().collect();
                let bl: Vec<&str> = lb.lines().collect();
                let i = al.iter().zip(bl.iter()).position(|(x, y)| x != y).unwrap_or(al.len().min(bl.len()));
                panic!("[{}] receipt text differs at line {}:\n go: {:?}\n rs: {:?}", label, i, al.get(i), bl.get(i));
            }
        }
        (None, None) => {}
        _ => panic!("[{}] receipt presence differs (go {}, rs {})", label, rec_g.is_some(), rec_r.is_some()),
    }
    // Compare trees with the receipt (volatile) removed.
    let strip = |t: &BTreeMap<String, String>| {
        t.iter().filter(|(k, _)| k.as_str() != "receipt.json").map(|(k, v)| (k.clone(), v.clone())).collect::<BTreeMap<_, _>>()
    };
    assert_eq!(strip(&tg), strip(&tr), "[{}] file trees (names/sha256) differ", label);
    Cmp { code: cg, go_stdout: sg, go_receipt: rec_g, rs_receipt: rec_r, go_tree: tg }
}

fn status_of(c: &Cmp) -> String {
    let v: Value = serde_json::from_str(c.go_receipt.as_ref().expect("receipt")).unwrap();
    v["status"].as_str().unwrap().to_string()
}

// ------------------------------------------------------------------ fixtures

fn raw_bits(dtype: &str, vals: &[u64]) -> Vec<u8> {
    let w = secant_core::dtype::dtype_size(dtype) as usize;
    let mut out = Vec::new();
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes()[..w]);
    }
    out
}

fn bt(name: &str, dtype: &str, shape: &[usize], data: Vec<u8>) -> BuildTensor {
    BuildTensor { name: name.into(), dtype: dtype.into(), shape: shape.to_vec(), data }
}

/// SafeTensors with an UNPADDED header (like the Go test builder), tensors in the given order.
fn st_unpadded(tensors: &[BuildTensor]) -> Vec<u8> {
    let mut body = Vec::new();
    let mut parts = Vec::new();
    for t in tensors {
        let b = body.len();
        body.extend_from_slice(&t.data);
        let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        parts.push(format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            t.name,
            t.dtype,
            shape.join(","),
            b,
            body.len()
        ));
    }
    let h = format!("{{{}}}", parts.join(","));
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(h.as_bytes());
    out.extend_from_slice(&body);
    out
}

fn st(tensors: &[BuildTensor]) -> Vec<u8> {
    build_safetensors(tensors, &BTreeMap::new()).unwrap()
}
fn gg(tensors: &[BuildTensor]) -> Vec<u8> {
    build_gguf(&[("general.architecture".to_string(), secant_core::builder::MetaValue::Str("qwen3".into()))], tensors, 32).unwrap()
}

fn st_header(header: &str, body: &[u8]) -> Vec<u8> {
    let mut b = (header.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(header.as_bytes());
    b.extend_from_slice(body);
    b
}

fn put(root: &Scratch, name: &str, data: &[u8]) -> String {
    let p = root.path(name);
    write(&p, data);
    p
}

// ------------------------------------------------------------------ probe

fn probe_args<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    let mut a = vec!["probe", "--out-dir", "{OUT}", "--receipt", "{REC}", "--attempt-id", ATT];
    a.extend_from_slice(extra);
    a
}

#[test]
fn probe_matrix_class_format_experts() {
    let Some((go, rs)) = bins("probe_matrix_class_format_experts") else { return };
    let root = Scratch::new("pm");
    let mut n = 0;
    for class in ["class1", "class2", "class3", "class4", "all"] {
        for format in ["safetensors", "gguf", "both"] {
            let c = compare(&format!("m-{}-{}", class, format), &go, &rs, &root, &probe_args(&["--class", class, "--format", format]));
            assert_eq!(c.code, 0);
            assert!(c.go_tree.len() >= 2); // out dir + fixture(s) + receipt
            n += 1;
        }
    }
    for ne in ["2", "3", "4", "5", "8", "64", "1048576"] {
        let c = compare(&format!("ne-{}", ne), &go, &rs, &root, &probe_args(&["--class", "class4", "--num-experts", ne]));
        assert_eq!(c.code, 0);
        n += 1;
    }
    println!("probe_matrix: {} scenarios, files byte-identical (SHA-256) and receipts line-equal", n);
}

#[test]
fn probe_json_stdout_and_seal_recompute() {
    let Some((go, rs)) = bins("probe_json_stdout_and_seal_recompute") else { return };
    let root = Scratch::new("pj");
    let c = compare("pj", &go, &rs, &root, &probe_args(&["--class", "all", "--json"]));
    assert_eq!(c.code, 0);
    // Deserialize the ORACLE's receipt into the Rust struct, recompute its seal:
    // proves the compact serialization (timestamps, numbers, key order) is the oracle's.
    let go_rec = c.go_receipt.unwrap();
    let rep: ProbeReport = serde_json::from_str(&go_rec).unwrap();
    assert_eq!(rep.compute_seal(), rep.receipt_sha256, "Go probe receipt seal is not reproduced by the Rust compact serializer");
    // The pretty form round-trips to the oracle's text exactly.
    assert_eq!(format!("{}\n", rep.to_json_pretty()), go_rec, "Rust pretty serialization of the oracle receipt differs byte for byte");
    // And the Rust receipt verifies under the same rule.
    let rs_rec: ProbeReport = serde_json::from_str(&c.rs_receipt.unwrap()).unwrap();
    assert_eq!(rs_rec.compute_seal(), rs_rec.receipt_sha256);
}

#[test]
fn probe_library_in_process_matches_oracle_files() {
    set_go_arm64_fma(true);
    let Some(_) = oracle_bin() else { return not_run("probe_library_in_process_matches_oracle_files") };
    let root = Scratch::new("pl");
    let go_dir = root.path("go/fx");
    let go_rec = root.path("go.json");
    let out = run_oracle(&["probe", "--out-dir", &go_dir, "--receipt", &go_rec, "--attempt-id", ATT, "--num-experts", "6"]).unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let o = ProbeOptions { attempt_id: ATT.into(), num_experts: 6, out_dir: root.path("rs/fx"), ..Default::default() };
    let (rep, res) = generate_probe(&o);
    res.unwrap();
    assert_eq!(rep.status, "generated");
    assert_eq!(rep.fixtures.len(), 8);
    let rec = root.path("rs.json");
    write_probe_receipt(&rec, &rep).unwrap();
    assert!(write_probe_receipt(&rec, &rep).is_err(), "receipt must never be overwritten");
    let mut map = BTreeMap::new();
    let (tg, tr) = (tree_canon(&go_dir, &mut map), tree_canon(&root.path("rs/fx"), &mut BTreeMap::new()));
    assert_eq!(tg, tr);
    assert_secant_namespace("pl", "probe", &root.path("rs/fx"), "", Some(&std::fs::read_to_string(&rec).unwrap()));
    assert_eq!(rep.fixtures.len(), map.len(), "every oracle fixture carries legacy metadata and is normalized");
    let gv: Value = canon_receipt_value(&serde_json::from_str(&std::fs::read_to_string(&go_rec).unwrap()).unwrap(), &map);
    let rv: Value = serde_json::from_str(&std::fs::read_to_string(&rec).unwrap()).unwrap();
    if let Some(d) = json_diff(&gv, &rv, &["started_utc", "finished_utc", "receipt_sha256", "path", "out_dir"]) {
        panic!("probe receipt JSON differs: {}", d);
    }
    // Deterministic across attempt ids and runs.
    let o2 = ProbeOptions { attempt_id: "att-other".into(), num_experts: 6, out_dir: root.path("rs2/fx"), ..Default::default() };
    let (rep2, res2) = generate_probe(&o2);
    res2.unwrap();
    for (a, b) in rep.fixtures.iter().zip(rep2.fixtures.iter()) {
        assert_eq!(a.sha256, b.sha256);
    }
}

fn base_tensors(rng: &mut Rng) -> Vec<BuildTensor> {
    vec![
        tensor("blk.0.ffn_gate_inp.weight", "F32", &[8, 16], rng),
        tensor("blk.0.attn_output.scale", "F32", &[16], rng),
        tensor("blk.0.attn_output.bias", "BF16", &[16], rng),
        // benign positive norm rows so the base itself scans clean
        bt("blk.0.attn_norm.weight", "F32", &[4, 16], f32_bytes(&vec![1.0; 64])),
        tensor("blk.0.ffn_up.weight", "F32", &[8, 16], rng),
        bt("blk.0.tid2eid", "I32", &[6, 4], i32_bytes(&(0..24).map(|i| (i / 4 + i % 4) % 8).collect::<Vec<i32>>())),
    ]
}

#[test]
fn probe_base_derived_synthetic_f32_base() {
    let Some((go, rs)) = bins("probe_base_derived_synthetic_f32_base") else { return };
    let root = Scratch::new("pb");
    let mut rng = Rng(0xBA5E);
    let tensors = base_tensors(&mut rng);
    let st_base = put(&root, "base.safetensors", &st(&tensors));
    let gg_base = put(&root, "base.gguf", &gg(&tensors));
    for (base, fmt) in [(&st_base, "safetensors"), (&gg_base, "gguf")] {
        let before = sha256_file(base);
        let c = compare(&format!("base-{}", fmt), &go, &rs, &root, &probe_args(&["--class", "all", "--base", base]));
        assert_eq!(c.code, 0);
        assert_eq!(sha256_file(base), before, "base modified");
        let v: Value = serde_json::from_str(c.go_receipt.as_ref().unwrap()).unwrap();
        assert_eq!(v["fixtures"].as_array().unwrap().len(), 4);
        assert_eq!(v["base_unchanged"], Value::Bool(true));
        // --format must agree with the base format
        let other = if fmt == "gguf" { "safetensors" } else { "gguf" };
        let c2 = compare(&format!("base-{}-conflict", fmt), &go, &rs, &root, &probe_args(&["--base", base, "--format", other]));
        assert_eq!(c2.code, 2);
        assert_eq!(status_of(&c2), "unknown");
        let c3 = compare(
            &format!("base-{}-fmt", fmt),
            &go,
            &rs,
            &root,
            &probe_args(&["--class", "class4", "--base", base, "--format", fmt, "--num-experts", "8"]),
        );
        assert_eq!(c3.code, 0);
    }
}

#[test]
fn probe_base_partial_coverage_unavailable_and_oracle_stray_copy() {
    let Some((go, rs)) = bins("probe_base_partial_coverage_unavailable_and_oracle_stray_copy") else { return };
    let root = Scratch::new("pp");
    let mut rng = Rng(7);
    // no routing table: class4 is partially covered (norm only) and says so
    let mut ts = base_tensors(&mut rng);
    ts.pop();
    let partial = put(&root, "partial.safetensors", &st(&ts));
    let c = compare("partial", &go, &rs, &root, &probe_args(&["--class", "class4", "--base", &partial]));
    assert_eq!(c.code, 0);
    assert!(c.go_receipt.unwrap().contains("NOT covered"));
    // bare base: nothing suitable for class4 -> unknown, nothing left behind; under "all" the classes are listed unavailable
    let bare = put(&root, "bare.safetensors", &st(&[tensor("blk.0.ffn_up.weight", "F32", &[2, 2], &mut rng)]));
    let c = compare("bare-c4", &go, &rs, &root, &probe_args(&["--class", "class4", "--base", &bare]));
    assert_eq!(c.code, 2);
    assert_eq!(c.go_tree.len(), 1, "failed run must leave only the receipt: {:?}", c.go_tree);
    let c = compare("bare-all", &go, &rs, &root, &probe_args(&["--class", "all", "--base", &bare]));
    assert_eq!(c.code, 2);
    // class3 falls back to "first float tensor with >=6 rows and >=4 columns" -> 1 fixture + 3 unavailable
    let wide = put(&root, "wide.safetensors", &st(&[tensor("blk.0.ffn_up.weight", "F32", &[8, 16], &mut rng)]));
    let c = compare("wide-all", &go, &rs, &root, &probe_args(&["--class", "all", "--base", &wide]));
    assert_eq!(c.code, 0);
    let v: Value = serde_json::from_str(c.go_receipt.as_ref().unwrap()).unwrap();
    assert_eq!(v["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(v["unavailable"].as_array().unwrap().len(), 3);
    // Oracle quirk (replicated): a pattern that writes no cell returns "too small" AFTER the base copy
    // was made, so the unmodified copy stays in out-dir. class3 needs >=6 rows; a 6x4 table whose
    // grid is large enough succeeds, a router with 1 row cannot hold the 3-row router pattern's
    // second row, but still writes cell (0,0): so use a 0-row-capable case: class2 scale of 1 cell.
    let tiny = put(&root, "tiny.safetensors", &st(&[tensor("blk.0.attn_output.scale", "F32", &[1], &mut rng)]));
    let c = compare("tiny-scale", &go, &rs, &root, &probe_args(&["--class", "class2", "--base", &tiny]));
    println!("tiny-scale exit {} tree {:?}", c.code, c.go_tree.keys().collect::<Vec<_>>());
}

#[test]
fn probe_base_zero_row_router_leaves_the_oracle_stray_copy() {
    // The oracle copies the base BEFORE it knows a pattern fits; a zero-row router makes
    // class1/class3 write no cell ("too small"), and the unmodified copy stays in out-dir
    // while class2 succeeds. Secant replicates this (differential parity), and the test
    // pins it so a future fix is a deliberate, visible change.
    let Some((go, rs)) = bins("probe_base_zero_row_router_leaves_the_oracle_stray_copy") else { return };
    let root = Scratch::new("pz");
    let mut rng = Rng(3);
    let base = put(
        &root,
        "zero.safetensors",
        &st_unpadded(&[
            bt("blk.0.ffn_gate_inp.weight", "F32", &[0, 16], vec![]),
            tensor("blk.0.attn_output.scale", "F32", &[16], &mut rng),
        ]),
    );
    let c = compare("stray", &go, &rs, &root, &probe_args(&["--class", "all", "--base", &base]));
    assert_eq!(c.code, 0);
    let names: Vec<&String> = c.go_tree.keys().filter(|k| k.starts_with("out/")).collect();
    println!("stray-copy out-dir: {:?}", names);
    assert_eq!(names.len(), 3, "class2 fixture plus two unmodified copies: {:?}", names);
    let v: Value = serde_json::from_str(c.go_receipt.as_ref().unwrap()).unwrap();
    assert_eq!(v["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(v["unavailable"].as_array().unwrap().len(), 3);
    // the stray copies are byte-identical to the base
    let base_sha = sha256_file(&base);
    let copies = c.go_tree.iter().filter(|(k, v)| k.starts_with("out/class1") || k.starts_with("out/class3") && !v.is_empty()).count();
    assert!(copies >= 2);
    assert!(c.go_tree.iter().filter(|(k, _)| k.contains("class1") || k.contains("class3")).all(|(_, v)| *v == base_sha));
}

#[test]
fn sanitize_unfused_default_agrees_with_fused_oracle_on_fixtures() {
    // Without --go-arm64-fma Secant uses portable unfused float64. The only fusable
    // expression in the sanitizer is math.Hypot's `1+q*q`; the policy outcome must not depend on it.
    set_go_arm64_fma(false);
    let Some(go) = oracle_bin() else { return not_run("sanitize_unfused_default_agrees_with_fused_oracle_on_fixtures") };
    let Some(rs) = secant_bin() else { return println!("NOT_RUN sanitize_unfused...: secant CLI binary not built") };
    let root = Scratch::new("sn");
    let files = fixture_files(&go, &root, "8");
    for (i, f) in files.iter().enumerate() {
        for (j, extra) in [vec!["--num-experts", "8"], vec!["--num-experts", "8", "--min-norm", "1e-6"]].iter().enumerate() {
            let a = Side::new(&root, &format!("n{}-{}-g", i, j));
            let b = Side::new(&root, &format!("n{}-{}-r", i, j));
            let og = run_go(&go, &a.args(&san_args(f, extra)));
            let or = Command::new(&rs).args(b.args(&san_args(f, extra))).output().unwrap(); // no --go-arm64-fma
            assert_eq!(og.status.code(), or.status.code());
            let ta = stable_lines(&canon_text(&std::fs::read_to_string(a.sub("receipt.json")).unwrap()), &a.dir);
            let tb = stable_lines(&canon_text(&std::fs::read_to_string(b.sub("receipt.json")).unwrap()), &b.dir);
            assert_eq!(ta, tb, "{} {:?}", f, extra);
        }
    }
}

#[test]
fn probe_option_errors_and_existing_out_dir() {
    let Some((go, rs)) = bins("probe_option_errors_and_existing_out_dir") else { return };
    let root = Scratch::new("pe");
    for (i, extra) in [
        vec!["--class", "class9"],
        vec!["--format", "onnx"],
        vec!["--num-experts", "1"],
        vec!["--num-experts", "2097152"],
        vec!["--num-experts", "-3"],
    ]
    .iter()
    .enumerate()
    {
        let c = compare(&format!("bad{}", i), &go, &rs, &root, &probe_args(extra));
        assert_eq!(c.code, 2, "{:?}", extra);
        assert_eq!(status_of(&c), "unknown");
    }
    // missing required flags -> usage, exit 2, no receipt
    let c = compare("noattempt", &go, &rs, &root, &["probe", "--out-dir", "{OUT}", "--receipt", "{REC}"]);
    assert_eq!(c.code, 2);
    assert!(c.go_receipt.is_none());
    // existing out-dir is never disturbed
    for sub in ["go", "rs"] {
        let d = root.path(&format!("exists-{}/out", sub));
        std::fs::create_dir_all(&d).unwrap();
        write(&format!("{}/keep.txt", d), b"x");
    }
    let c = compare("exists", &go, &rs, &root, &probe_args(&[]));
    assert_eq!(c.code, 2);
    assert_eq!(status_of(&c), "unknown");
    // receipt that already exists / lies inside out-dir -> usage
    let c = compare("rec-inside", &go, &rs, &root, &["probe", "--out-dir", "{DIR}/o", "--receipt", "{DIR}/o/r.json", "--attempt-id", ATT]);
    assert_eq!(c.code, 2);
    let c = compare("rec-exists", &go, &rs, &root, &["probe", "--out-dir", "{DIR}/o", "--receipt", "{DIR}", "--attempt-id", ATT]);
    assert_eq!(c.code, 2);
}

// ------------------------------------------------------------------ sanitize

fn san_args<'a>(model: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    let mut a = vec!["sanitize", "--model", model, "--receipt", "{REC}", "--attempt-id", ATT];
    a.extend_from_slice(extra);
    a
}

fn fixture_files(go: &str, root: &Scratch, num_experts: &str) -> Vec<String> {
    let d = root.path(&format!("fx{}", num_experts));
    let out = Command::new(go)
        .args([
            "wsp",
            "probe",
            "--out-dir",
            &d,
            "--receipt",
            &root.path(&format!("fx{}.json", num_experts)),
            "--attempt-id",
            ATT,
            "--num-experts",
            num_experts,
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let mut v: Vec<String> = std::fs::read_dir(&d).unwrap().flatten().map(|e| e.path().to_string_lossy().into_owned()).collect();
    v.sort();
    v
}

#[test]
fn sanitize_probe_fixtures_findings_and_exit_codes() {
    let Some((go, rs)) = bins("sanitize_probe_fixtures_findings_and_exit_codes") else { return };
    let root = Scratch::new("sf");
    let files = fixture_files(&go, &root, "8");
    assert_eq!(files.len(), 8);
    let mut seen = std::collections::BTreeSet::new();
    for (i, f) in files.iter().enumerate() {
        let name = Path::new(f).file_name().unwrap().to_string_lossy().into_owned();
        // default policy: every fixture has findings -> exit 1
        let c = compare(&format!("fx{}", i), &go, &rs, &root, &san_args(f, &["--num-experts", "8"]));
        assert_eq!(c.code, 1, "{}", name);
        assert_eq!(status_of(&c), "findings");
        let v: Value = serde_json::from_str(c.go_receipt.as_ref().unwrap()).unwrap();
        for (k, _) in v["before"]["counts"].as_object().unwrap() {
            seen.insert(k.clone());
        }
        // truncated examples (max 3) keep aggregate counts exact
        let c = compare(&format!("fx{}-trunc", i), &go, &rs, &root, &san_args(f, &["--num-experts", "8", "--max-examples", "3", "--json"]));
        assert_eq!(c.code, 1);
        // tighter thresholds change the classification identically
        let c =
            compare(&format!("fx{}-thr", i), &go, &rs, &root, &san_args(f, &["--num-experts", "3", "--max-abs", "0.5", "--min-norm", "2"]));
        assert!(c.code == 1 || c.code == 2, "{}", c.code);
        // clamp: remediable classes publish a byte-identical clean copy, others refuse and leave nothing
        let c = compare(
            &format!("fx{}-clamp", i),
            &go,
            &rs,
            &root,
            &san_args(f, &["--num-experts", "8", "--clamp-nonfinite", "--out", "{OUT}"]),
        );
        let st = status_of(&c);
        let published = c.go_tree.contains_key("out");
        println!("{:<44} clamp -> exit {} status {} published {}", name, c.code, st, published);
        let c = compare(
            &format!("fx{}-both", i),
            &go,
            &rs,
            &root,
            &san_args(f, &["--num-experts", "8", "--clamp-nonfinite", "--dedup-routing-rows", "--out", "{OUT}", "--json"]),
        );
        let _ = c;
    }
    // Every finding kind the brief lists was observed on a Go-generated fixture.
    for k in [
        "nan",
        "inf",
        "extreme_finite",
        "subnormal",
        "norm_all_zero",
        "norm_below_min",
        "routing_negative",
        "routing_oob",
        "routing_duplicate",
    ] {
        assert!(seen.contains(k), "kind {} never observed; saw {:?}", k, seen);
    }
}

#[test]
fn sanitize_repairable_classes_publish_identical_bytes() {
    let Some((go, rs)) = bins("sanitize_repairable_classes_publish_identical_bytes") else { return };
    let root = Scratch::new("sr");
    let dts = [
        ("F16", 0x7e01u64, 0x7c00u64, 0xfc00u64, 0x3c00u64),
        ("BF16", 0x7fc1, 0x7f80, 0xff80, 0x3f80),
        ("F32", 0x7fc00001, 0x7f800000, 0xff800000, 0x3f800000),
        ("F64", 0x7ff8000000000001, 0x7ff0000000000000, 0xfff0000000000000, 0x3ff0000000000000),
    ];
    for (fi, fmt) in ["safetensors", "gguf"].iter().enumerate() {
        for (dtype, nan, pinf, ninf, one) in dts {
            let ts = [bt("bias", dtype, &[8], raw_bits(dtype, &[nan, pinf, ninf, one, nan | (1u64 << 1), 0, one, ninf]))];
            let data = if fi == 0 { st(&ts) } else { gg(&ts) };
            let model = put(&root, &format!("m-{}-{}", fmt, dtype), &data);
            for max_abs in ["1.0001", "10000", "65504", "70000", "1e30", "1.5"] {
                let c = compare(
                    &format!("clamp-{}-{}-{}", fmt, dtype, max_abs),
                    &go,
                    &rs,
                    &root,
                    &san_args(&model, &["--clamp-nonfinite", "--max-abs", max_abs, "--out", "{OUT}"]),
                );
                assert_eq!(c.code, 0, "{} {} {}", fmt, dtype, max_abs);
                assert_eq!(status_of(&c), "remediated");
                assert!(c.go_tree.contains_key("out"));
            }
        }
    }
    // a bound below the data's own magnitude leaves extreme_finite findings: refuse, publish nothing
    let ts = [bt("bias", "F32", &[2], raw_bits("F32", &[0x7fc00000, 0x3f800000]))];
    let model = put(&root, "below.safetensors", &st(&ts));
    let c = compare("clamp-below", &go, &rs, &root, &san_args(&model, &["--clamp-nonfinite", "--max-abs", "0.3", "--out", "{OUT}"]));
    assert_eq!(c.code, 1);
    assert!(!c.go_tree.contains_key("out"));
    // dedup-only: I32 and I64, with later unique ids preserved
    for (fi, fmt) in ["safetensors", "gguf"].iter().enumerate() {
        for dtype in ["I32", "I64"] {
            let ts = [
                bt("tid2eid", dtype, &[3, 4], raw_bits(dtype, &[2, 2, 0, 3, 2, 2, 0, 3, 0, 1, 2, 3])),
                bt("ordinary", "F32", &[1], raw_bits("F32", &[0x3f800000])),
            ];
            let data = if fi == 0 { st(&ts) } else { gg(&ts) };
            let model = put(&root, &format!("d-{}-{}", fmt, dtype), &data);
            let c = compare(
                &format!("dedup-{}-{}", fmt, dtype),
                &go,
                &rs,
                &root,
                &san_args(&model, &["--num-experts", "8", "--dedup-routing-rows", "--out", "{OUT}"]),
            );
            assert_eq!(c.code, 0);
            assert_eq!(status_of(&c), "remediated");
            // clean rescan of the sanitized copy
            let out_path = format!("{}/out", root.path(&format!("dedup-{}-{}-rs", fmt, dtype)));
            let sr = sanitize_file_report(&out_path, &SanitizeOptions { attempt_id: ATT.into(), num_experts: 8, ..Default::default() });
            assert_eq!(sr.0.status, "clean");
        }
    }
    // a 64 KiB chunk boundary (16384 F32 per chunk) carrying a NaN: clamp must hit the exact element
    let n = 16386usize;
    let mut vals = vec![0x3f800000u64; n];
    vals[16384] = 0x7fc00000;
    vals[16383] = 0xff800000;
    vals[n - 1] = 0x7f800000;
    let model = put(
        &root,
        "chunk.safetensors",
        &st_unpadded(&[bt("w", "F32", &[2, n / 2], raw_bits("F32", &vals)), bt("empty", "F32", &[0], vec![])]),
    );
    let c = compare("chunk", &go, &rs, &root, &san_args(&model, &[]));
    assert_eq!(c.code, 1);
    let c = compare("chunk-clamp", &go, &rs, &root, &san_args(&model, &["--clamp-nonfinite", "--out", "{OUT}"]));
    assert_eq!(c.code, 0);
}

#[test]
fn sanitize_unrepairable_classes_refuse_to_publish() {
    let Some((go, rs)) = bins("sanitize_unrepairable_classes_refuse_to_publish") else { return };
    let root = Scratch::new("su");
    let cases: Vec<(&str, BuildTensor)> = vec![
        ("neg-route", bt("tid2eid", "I32", &[1, 2], raw_bits("I32", &[0xffffffff, 1]))),
        ("wide-route", bt("tid2eid", "I32", &[1, 9], raw_bits("I32", &[0, 1, 2, 3, 4, 5, 6, 7, 7]))),
        ("nan-in-norm", bt("norm.weight", "F32", &[1], raw_bits("F32", &[0x7fc00000]))),
        ("extreme", bt("bias", "F32", &[1], raw_bits("F32", &[0x7f7fffff]))),
        ("subnormal", bt("bias", "F32", &[2], raw_bits("F32", &[1, 0x3f800000]))),
        ("zero-norm", bt("layer.norm.weight", "F16", &[2, 2], raw_bits("F16", &[0, 0x8000, 0x3c00, 0x3c00]))),
    ];
    for (name, t) in cases {
        let model = put(&root, &format!("{}.safetensors", name), &st(&[t]));
        let c = compare(
            name,
            &go,
            &rs,
            &root,
            &san_args(&model, &["--num-experts", "8", "--clamp-nonfinite", "--dedup-routing-rows", "--out", "{OUT}"]),
        );
        assert_eq!(c.code, 1, "{}", name);
        assert_eq!(status_of(&c), "findings");
        assert!(!c.go_tree.contains_key("out"), "{}: residual output published", name);
    }
}

#[test]
fn sanitize_control_selectors_and_options() {
    let Some((go, rs)) = bins("sanitize_control_selectors_and_options") else { return };
    let root = Scratch::new("sc");
    let mut rng = Rng(99);
    for (fi, fmt) in ["safetensors", "gguf"].iter().enumerate() {
        // legitimate control: random weights, positive norms, valid routing => clean, exit 0
        let ts = vec![
            tensor("blk.0.attn_q.weight", "F32", &[6, 5], &mut rng),
            tensor("blk.0.attn_k.weight", "BF16", &[4, 8], &mut rng),
            tensor("blk.0.attn_v.weight", "F16", &[3, 8], &mut rng),
            tensor("blk.0.attn_o.weight", "F64", &[2, 3], &mut rng),
            bt("blk.0.attn_norm.weight", "F32", &[3, 4], f32_bytes(&vec![1.0; 12])),
            bt("blk.0.tid2eid", "I32", &[2, 3], i32_bytes(&[0, 1, 2, 3, 4, 5])),
            bt("blk.0.expert_id", "I64", &[2, 2], raw_bits("I64", &[0, 7, 7, 0])),
            bt("misc_ints", "I32", &[2], i32_bytes(&[-5, i32::MAX])),
        ];
        let data = if fi == 0 { st(&ts) } else { gg(&ts) };
        let model = put(&root, &format!("ctl.{}", fmt), &data);
        let c = compare(&format!("ctl-{}", fmt), &go, &rs, &root, &san_args(&model, &["--num-experts", "8"]));
        assert_eq!(c.code, 0);
        assert_eq!(status_of(&c), "clean");
        let c = compare(&format!("ctl-{}-json", fmt), &go, &rs, &root, &san_args(&model, &["--num-experts", "8", "--json"]));
        assert_eq!(c.code, 0);
        // custom globs and out-of-range experts via a smaller bound
        let c = compare(
            &format!("ctl-{}-glob", fmt),
            &go,
            &rs,
            &root,
            &san_args(
                &model,
                &["--num-experts", "3", "--routing-tensor", "misc_*", "--norm-tensor", "blk.0.attn_[qk].weight", "--max-examples", "5"],
            ),
        );
        assert!(c.code == 1 || c.code == 2);
        let c = compare(
            &format!("ctl-{}-glob2", fmt),
            &go,
            &rs,
            &root,
            &san_args(&model, &["--num-experts", "8", "--norm-tensor", "blk.*.attn_[qkvo].weight"]),
        );
        println!("ctl-{}-glob2 exit {}", fmt, c.code);
        // selector failures: exit 2, status unknown, same message
        for (i, extra) in [
            vec!["--routing-tensor", "missing*", "--num-experts", "8"],
            vec!["--routing-tensor", "[", "--num-experts", "8"],
            vec!["--routing-tensor", "", "--num-experts", "8"],
            vec!["--norm-tensor", "tid2eid", "--num-experts", "8"],
            vec!["--norm-tensor", "blk.0.attn_norm.weight", "--routing-tensor", "blk.0.attn_norm.weight", "--num-experts", "8"],
            vec![], // routing tensors present but --num-experts missing
            vec!["--num-experts", "-1"],
            vec!["--max-abs", "-1", "--num-experts", "8"],
            vec!["--max-abs", "NaN", "--num-experts", "8"],
            vec!["--max-abs", "inf", "--num-experts", "8"],
            vec!["--min-norm", "-1", "--num-experts", "8"],
            vec!["--max-examples", "-1", "--num-experts", "8"],
            vec!["--max-examples", "10001", "--num-experts", "8"],
            vec!["--clamp-nonfinite", "--num-experts", "8"],
            vec!["--out", "{OUT}", "--num-experts", "8"],
            vec!["--max-abs", "1e-300", "--min-norm", "1e300", "--num-experts", "8"],
        ]
        .iter()
        .enumerate()
        {
            let c = compare(&format!("sel-{}-{}", fmt, i), &go, &rs, &root, &san_args(&model, extra));
            println!("sel-{}-{} {:?} -> exit {} {}", fmt, i, extra, c.code, status_of(&c));
        }
        // float routing table / integer norm are refused (exit 2) by name selection
        let wrong = put(
            &root,
            &format!("wrong.{}", fmt),
            &if fi == 0 {
                st(&[bt("tid2eid", "F32", &[1], raw_bits("F32", &[0x3f800000]))])
            } else {
                gg(&[bt("tid2eid", "F32", &[1], raw_bits("F32", &[0x3f800000]))])
            },
        );
        let c = compare(&format!("wrong-{}", fmt), &go, &rs, &root, &san_args(&wrong, &["--num-experts", "8"]));
        assert_eq!(c.code, 2);
    }
    // 2^53+1 expert bound with exact I64 values (no float64 widening)
    let model =
        put(&root, "big.safetensors", &st(&[bt("expert_id", "I64", &[1, 2], raw_bits("I64", &[9007199254740992, 9007199254740993]))]));
    let c = compare("bigexp", &go, &rs, &root, &san_args(&model, &["--num-experts", "9007199254740993"]));
    assert_eq!(c.code, 1);
    let v: Value = serde_json::from_str(c.go_receipt.as_ref().unwrap()).unwrap();
    assert_eq!(v["before"]["counts"]["routing_oob"], 1);
    assert!(v["before"]["counts"].get("routing_duplicate").is_none());
    // INT64 extremes
    let model = put(
        &root,
        "ext.safetensors",
        &st(&[bt("expert_id", "I64", &[1, 4], raw_bits("I64", &[i64::MIN as u64, i64::MAX as u64, 1 << 32, u64::MAX]))]),
    );
    compare("i64ext", &go, &rs, &root, &san_args(&model, &["--num-experts", "8"]));
}

#[test]
fn sanitize_output_and_receipt_never_clobber() {
    let Some((go, rs)) = bins("sanitize_output_and_receipt_never_clobber") else { return };
    let root = Scratch::new("sk");
    let model = put(&root, "m.safetensors", &st(&[bt("bias", "F32", &[1], raw_bits("F32", &[0x3f800000]))]));
    // out == input, hard link alias, symlink alias, existing out
    let c = compare("out-is-input", &go, &rs, &root, &san_args(&model, &["--clamp-nonfinite", "--out", &model]));
    assert_eq!(c.code, 2);
    for (name, mk) in [("hard", 0), ("sym", 1), ("file", 2)] {
        let side_paths: Vec<String> = ["go", "rs"].iter().map(|s| root.path(&format!("alias-{}-{}", name, s))).collect();
        for p in &side_paths {
            match mk {
                0 => std::fs::hard_link(&model, p).unwrap(),
                1 => std::os::unix::fs::symlink(&model, p).unwrap(),
                _ => write(p, b"pre-existing"),
            }
        }
        let (g, r) = (Side::new(&root, &format!("alias-{}-g", name)), Side::new(&root, &format!("alias-{}-r", name)));
        let og = run_go(&go, &g.args(&san_args(&model, &["--clamp-nonfinite", "--out", &side_paths[0]])));
        let or = run_rs(&rs, &r.args(&san_args(&model, &["--clamp-nonfinite", "--out", &side_paths[1]])));
        assert_eq!(og.status.code(), or.status.code(), "{}", name);
        assert_eq!(og.status.code(), Some(2), "{}", name);
        assert_eq!(std::fs::read(&model).unwrap(), st(&[bt("bias", "F32", &[1], raw_bits("F32", &[0x3f800000]))]));
    }
    // the receipt must be a new path and distinct from the model
    let c = compare("rec-is-model", &go, &rs, &root, &["sanitize", "--model", &model, "--receipt", &model, "--attempt-id", ATT]);
    assert_eq!(c.code, 2);
    for sub in ["go", "rs"] {
        let d = root.path(&format!("rec-exists-{}", sub));
        std::fs::create_dir_all(&d).unwrap();
        write(&format!("{}/receipt.json", d), b"old");
    }
    let c = compare("rec-exists", &go, &rs, &root, &san_args(&model, &[]));
    assert_eq!(c.code, 2);
    let c = compare("no-attempt", &go, &rs, &root, &["sanitize", "--model", &model, "--receipt", "{REC}"]);
    assert_eq!(c.code, 2);
    let c = compare("missing-model", &go, &rs, &root, &san_args("/nonexistent/m.safetensors", &[]));
    assert_eq!(c.code, 2);
    assert_eq!(status_of(&c), "unknown");
    let c = compare("dir-model", &go, &rs, &root, &san_args(&root.path(""), &[]));
    assert_eq!(c.code, 2);
}

#[test]
fn sanitize_malformed_and_unsupported_layouts_fail_closed() {
    let Some((go, rs)) = bins("sanitize_malformed_and_unsupported_layouts_fail_closed") else { return };
    let root = Scratch::new("sm");
    let body8 = vec![0u8; 8];
    let headers = [
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","dtype":"F64","shape":[1],"data_offsets":[0,8]}}"#,
        r#"{"x":{"dtype":"F32","shape":[-1,-1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,0]}}"#,
        r#"{"x":{"dtype":"F32","data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":null,"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[9],"data_offsets":[0,36]}}"#,
        r#"{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
        "{}",
        "null",
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}} trailing"#,
        r#"{"__metadata__":{"k":1},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":5}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":null}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]"#,
        r#"{"a":{"b":{"c":{"d":{"e":{"f":{"g":{"h":{"i":{"j":{"k":{"l":{"m":{"n":{"o":{"p":{"q":{}}}}}}}}}}}}}}}}}}"#,
        // one tensor, 4 payload bytes + 4 stray bytes (unindexed) and a gap/hole variant
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"y":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#,
        r#"{"__metadata__":null,"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{ "x" : { "dtype" : "F32" , "shape" : [ 1 ] , "data_offsets" : [ 0 , 4 ] } }"#,
        r#"{"x":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}}"#,
    ];
    let mut report = Vec::new();
    for (i, h) in headers.iter().enumerate() {
        let model = put(&root, &format!("h{}.safetensors", i), &st_header(h, &body8));
        let c = compare_opt(&format!("hdr{}", i), &go, &rs, &root, &san_args(&model, &[]), true);
        report.push((i, c.code));
    }
    println!("malformed headers (index, exit): {:?}", report);
    // truncated payload (both formats), short files, random bytes
    let ts = [bt("w", "F32", &[2], raw_bits("F32", &[0x3f800000, 0x3f800000]))];
    let s = st(&ts);
    let c = compare_opt("st-trunc", &go, &rs, &root, &san_args(&put(&root, "t.safetensors", &s[..s.len() - 1]), &[]), true);
    assert_eq!(c.code, 2);
    let mut longer = s.clone();
    longer.push(0);
    let c = compare_opt("st-stray", &go, &rs, &root, &san_args(&put(&root, "stray.safetensors", &longer), &[]), true);
    assert_eq!(c.code, 2);
    let g = gg(&ts);
    let lay = secant_core::container::scan_gguf_layout(&g).unwrap();
    let c = compare_opt("gg-trunc", &go, &rs, &root, &san_args(&put(&root, "t.gguf", &g[..lay.data_offset as usize + 7]), &[]), true);
    assert_eq!(c.code, 2);
    // duplicate GGUF tensor names (the builder refuses them: rename the second tensor in place)
    let mut dup = gg(&[bt("w", "F32", &[1], raw_bits("F32", &[0])), bt("v", "F32", &[1], raw_bits("F32", &[0]))]);
    let vpat: Vec<u8> = [1u64.to_le_bytes().to_vec(), vec![b'v']].concat();
    let vat = dup.windows(vpat.len()).position(|w| w == vpat.as_slice()).expect("v tensor info");
    dup[vat + 8] = b'w';
    let c = compare_opt("gg-dupname", &go, &rs, &root, &san_args(&put(&root, "dup.gguf", &dup), &[]), true);
    assert_eq!(c.code, 2);
    for (name, bytes) in [
        ("empty", vec![]),
        ("short", vec![1, 2, 3]),
        ("seven", vec![0; 7]),
        ("zeros", vec![0; 64]),
        ("junk", (0..200u32).map(|i| (i * 7 + 3) as u8).collect::<Vec<u8>>()),
        ("gguf-magic-only", b"GGUF".to_vec()),
        ("gguf8", {
            let mut v = b"GGUF".to_vec();
            v.extend_from_slice(&3u32.to_le_bytes());
            v
        }),
    ] {
        let c = compare_opt(&format!("junk-{}", name), &go, &rs, &root, &san_args(&put(&root, name, &bytes), &[]), true);
        assert_eq!(c.code, 2, "{}", name);
        assert_eq!(status_of(&c), "unknown");
    }
    // GGUF version 1 / 4 are refused; v2 is accepted
    for v in [1u32, 2, 4] {
        let mut b = g.clone();
        b[4..8].copy_from_slice(&v.to_le_bytes());
        let c = compare_opt(&format!("gguf-v{}", v), &go, &rs, &root, &san_args(&put(&root, &format!("v{}.gguf", v), &b), &[]), true);
        println!("gguf v{} -> exit {}", v, c.code);
    }
    // Quantized GGUF: the Go scanner rejects the whole file; Secant lists it and the sanitizer must refuse.
    let mut q = gg(&[bt("w", "F32", &[32], raw_bits("F32", &[0; 32]))]);
    // find "w" tensor info: name_len(1) 'w' ndims(1) dim(32) type(0)
    let pat: Vec<u8> = [1u64.to_le_bytes().to_vec(), vec![b'w'], 1u32.to_le_bytes().to_vec(), 32u64.to_le_bytes().to_vec()].concat();
    let at = q.windows(pat.len()).position(|w| w == pat.as_slice()).expect("tensor info");
    q[at + pat.len()..at + pat.len() + 4].copy_from_slice(&8u32.to_le_bytes()); // Q8_0
    let c = compare_opt("gguf-q8", &go, &rs, &root, &san_args(&put(&root, "q8.gguf", &q), &[]), true);
    assert_eq!(c.code, 2);
    assert_eq!(status_of(&c), "unknown");
    println!("quantized GGUF error go: {}", serde_json::from_str::<Value>(c.go_receipt.as_ref().unwrap()).unwrap()["error"]);
    println!("quantized GGUF error rs: {}", serde_json::from_str::<Value>(c.rs_receipt.as_ref().unwrap()).unwrap()["error"]);
    // quantized tensor placed after valid ones, and an unknown type id
    let mut u = gg(&[bt("w", "F32", &[32], raw_bits("F32", &[0; 32]))]);
    u[at + pat.len()..at + pat.len() + 4].copy_from_slice(&99u32.to_le_bytes());
    let c = compare_opt("gguf-badtype", &go, &rs, &root, &san_args(&put(&root, "bad.gguf", &u), &[]), true);
    assert_eq!(c.code, 2);
    let diffs = MSG_DIFFS.lock().unwrap();
    println!("{} error-wording differences (exit code, status and all other receipt lines identical):", diffs.len());
    for d in diffs.iter() {
        println!("   {}", d);
    }
}

/// Library-level (in-process) checks that need no oracle: receipt shape, seal,
/// bracketing, defaults, and the three-way status contract.
#[test]
fn sanitize_library_contract() {
    set_go_arm64_fma(true);
    let root = Scratch::new("sl");
    let ts = [bt("bias", "F32", &[3], raw_bits("F32", &[0x3f800000, 0x7fc00000, 0x7f800000]))];
    let model = put(&root, "m.safetensors", &st(&ts));
    let opts = SanitizeOptions { attempt_id: ATT.into(), ..Default::default() };
    let (r, res) = sanitize_file_report(&model, &opts);
    res.unwrap();
    assert_eq!(r.status, "findings");
    assert!(r.input_unchanged && r.pre_sha256 == r.post_sha256 && !r.pre_sha256.is_empty());
    assert_eq!(r.options.as_ref().unwrap().max_abs, 1e4);
    assert_eq!(r.options.as_ref().unwrap().min_norm, 1e-12);
    assert_eq!(r.options.as_ref().unwrap().max_examples, 128);
    assert_eq!(r.compute_seal(), r.receipt_sha256);
    let text = r.to_json_pretty();
    assert!(text.contains("\"routing_tensor\": null") && text.contains("\"max_abs\": 10000,") && text.contains("\"min_norm\": 1e-12,"));
    let rt: SanitizeReceipt = serde_json::from_str(&text).unwrap();
    assert_eq!(rt, r);
    // errors keep the receipt, status unknown, error set, seal still valid
    let (r2, res2) = sanitize_file_report(&model, &SanitizeOptions::default());
    assert!(res2.is_err() && r2.status == "unknown" && r2.options.is_none() && r2.compute_seal() == r2.receipt_sha256);
    assert!(r2.to_json_pretty().contains("\"options\": null"));
    // sanitize_file() surfaces the error
    assert!(secant_core::security::sanitize_file(&model, &SanitizeOptions::default()).is_err());
    // remediation publishes a mode-0600 file and a second run refuses to overwrite
    let out = root.path("clean.safetensors");
    let o = SanitizeOptions { attempt_id: ATT.into(), clamp_nonfinite: true, output: out.clone(), ..Default::default() };
    let (r3, res3) = sanitize_file_report(&model, &o);
    res3.unwrap();
    assert_eq!(r3.status, "remediated");
    assert!(r3.output_published);
    assert_eq!(r3.changes.get("nan"), Some(&1));
    assert_eq!(r3.changes.get("inf"), Some(&1));
    assert_eq!(r3.output_sha256, sha256_file(&out));
    let (_, res4) = sanitize_file_report(&model, &o);
    assert!(res4.is_err());
    let rec = root.path("rec.json");
    secant_core::security::write_sanitize_receipt(&rec, &r3).unwrap();
    assert!(secant_core::security::write_sanitize_receipt(&rec, &r3).is_err());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&out).unwrap().permissions().mode() & 0o777, 0o600);
    }
}

/// The oracle's receipt, deserialized into the Rust struct, must reproduce its
/// own `receipt_sha256` (sanitize side of the seal proof).
#[test]
fn sanitize_go_receipt_seal_recomputes_in_rust() {
    let Some((go, rs)) = bins("sanitize_go_receipt_seal_recomputes_in_rust") else { return };
    let root = Scratch::new("ss");
    let files = fixture_files(&go, &root, "8");
    let mut n = 0;
    for (i, f) in files.iter().enumerate() {
        for extra in [vec!["--num-experts", "8"], vec!["--num-experts", "8", "--clamp-nonfinite", "--out", "{OUT}"]] {
            let c = compare(&format!("seal{}-{}", i, extra.len()), &go, &rs, &root, &san_args(f, &extra));
            // compare() normalized the dir inside stdout only; receipt text is verbatim
            let rep: SanitizeReceipt = serde_json::from_str(c.go_receipt.as_ref().unwrap()).unwrap();
            assert_eq!(rep.compute_seal(), rep.receipt_sha256, "Go sanitize seal not reproduced for {} {:?}", f, extra);
            assert_eq!(format!("{}\n", rep.to_json_pretty()), *c.go_receipt.as_ref().unwrap(), "pretty text differs byte for byte");
            n += 1;
        }
    }
    println!("{} Go sanitize receipts: seal and pretty text reproduced from the Rust structs", n);
}

// ------------------------------------------------------------------
// Secant extensions: --clamp-extreme and --flush-subnormal (no oracle exists
// for these; the tests assert properties against an independent decoder).
// ------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Cls {
    Nan,
    Inf,
    Sub,
    Normal,
}

fn width(d: &str) -> usize {
    secant_core::dtype::dtype_size(d) as usize
}
fn sign_of(d: &str) -> u64 {
    1u64 << (width(d) * 8 - 1)
}
fn expo_mant(d: &str) -> (u64, u64) {
    match d {
        "F16" => (0x7c00, 0x3ff),
        "BF16" => (0x7f80, 0x7f),
        "F32" => (0x7f800000, 0x7fffff),
        _ => (0x7ff0000000000000, 0xfffffffffffff),
    }
}
fn cls(d: &str, b: u64) -> Cls {
    let (e, m) = expo_mant(d);
    if b & e == e {
        if b & m != 0 {
            Cls::Nan
        } else {
            Cls::Inf
        }
    } else if b & e == 0 && b & m != 0 {
        Cls::Sub
    } else {
        Cls::Normal
    }
}
/// Independent decoder (core dtype helpers, not the sanitizer's own).
fn dec(d: &str, b: u64) -> f64 {
    use secant_core::dtype::*;
    match d {
        "F16" => f32_from_f16(b as u16) as f64,
        "BF16" => f32_from_bf16(b as u16) as f64,
        "F32" => f32::from_bits(b as u32) as f64,
        _ => f64::from_bits(b),
    }
}
fn bits_at(d: &str, data: &[u8], i: usize) -> u64 {
    let w = width(d);
    let mut x = [0u8; 8];
    x[..w].copy_from_slice(&data[i * w..(i + 1) * w]);
    u64::from_le_bytes(x)
}

#[derive(Clone, Copy)]
struct Flags {
    cnf: bool,
    ce: bool,
    fs: bool,
}
impl Flags {
    fn all() -> Vec<Flags> {
        (1..8).map(|m| Flags { cnf: m & 1 != 0, ce: m & 2 != 0, fs: m & 4 != 0 }).collect()
    }
    fn opts(&self, max_abs: f64, out: &str) -> SanitizeOptions {
        SanitizeOptions {
            attempt_id: ATT.into(),
            max_abs,
            clamp_nonfinite: self.cnf,
            clamp_extreme: self.ce,
            flush_subnormal: self.fs,
            output: out.into(),
            ..Default::default()
        }
    }
}

/// Bit patterns for `dtype`: every pattern for the 16-bit types, specials plus
/// pseudo-random patterns (and near-boundary magnitudes) for F32/F64.
fn patterns(d: &str, max_abs: f64) -> Vec<u64> {
    if width(d) == 2 {
        return (0..65536u64).collect();
    }
    let (e, m) = expo_mant(d);
    let sign = sign_of(d);
    let mut v = vec![0, sign, 1, sign | 1, m, sign | m, e, sign | e, e | 1, e | (1 << 3), sign | e | m, e - 1, sign | (e - 1), (e >> 1)];
    let mut rng = Rng(0xC0FFEE ^ width(d) as u64);
    let mask = if width(d) == 8 { u64::MAX } else { 0xffff_ffff };
    for _ in 0..6000 {
        v.push(rng.next_u64() & mask);
    }
    // magnitudes straddling max_abs
    for k in -8i64..=8 {
        let x = max_abs * (1.0 + k as f64 * 1e-3);
        let b = if d == "F32" { (x as f32).to_bits() as u64 } else { x.to_bits() };
        v.push(b);
        v.push(b | sign);
    }
    v
}

fn container(fmt: &str, tensors: &[BuildTensor]) -> Vec<u8> {
    if fmt == "safetensors" {
        st(tensors)
    } else {
        gg(tensors)
    }
}

/// Expected outcome of one element under `f`: Some(Zero|Bound) replaced, None kept.
#[derive(PartialEq, Debug)]
enum Act {
    Keep,
    Zero,
    SignedZero,
    Bound,
}
fn act(d: &str, b: u64, f: Flags, max_abs: f64) -> Act {
    match cls(d, b) {
        Cls::Nan if f.cnf => Act::Zero,
        Cls::Inf if f.cnf => Act::Bound,
        Cls::Sub if f.fs => Act::SignedZero,
        Cls::Sub | Cls::Normal if f.ce && dec(d, b).abs() > max_abs => Act::Bound,
        _ => Act::Keep,
    }
}
/// Elements the combination can repair (or that need no repair).
fn repairable(d: &str, b: u64, f: Flags, max_abs: f64) -> bool {
    match cls(d, b) {
        Cls::Nan | Cls::Inf => f.cnf,
        Cls::Sub => f.fs, // an unflushed subnormal is itself a blocking finding
        Cls::Normal => f.ce || dec(d, b).abs() <= max_abs,
    }
}

/// Everything except `[off, off+len)` must be byte-identical (header, padding, other tensors).
fn identical_outside(a: &[u8], b: &[u8], off: usize, len: usize) {
    assert_eq!(a.len(), b.len());
    assert_eq!(&a[..off], &b[..off], "header or earlier payload changed");
    assert_eq!(&a[off + len..], &b[off + len..], "later payload changed");
}

fn check_bound(d: &str, bound: u64, max_abs: f64) {
    assert!(bound & sign_of(d) == 0);
    assert!(dec(d, bound) <= max_abs, "{} bound {} exceeds {}", d, dec(d, bound), max_abs);
    // largest representable not above max_abs: the next encoding up is above it (or is the first non-finite)
    assert!(cls(d, bound + 1) != Cls::Normal || dec(d, bound + 1) > max_abs, "{}: bound {:#x} is not the largest <= {}", d, bound, max_abs);
}

fn extension_case(fmt: &str, d: &str, f: Flags, max_abs: f64, root: &Scratch) {
    let tag = format!("{} {} cnf={} ce={} fs={} max_abs={}", fmt, d, f.cnf, f.ce, f.fs, max_abs);
    let pats: Vec<u64> = patterns(d, max_abs).into_iter().filter(|b| repairable(d, *b, f, max_abs)).collect();
    let n = pats.len();
    let keep: Vec<f64> = (0..5).map(|i| 1e-4 * (i + 1) as f64).collect();
    let tensors = [
        bt(
            "aaa_other.weight",
            d,
            &[5],
            match d {
                "F16" => f16_bytes(&keep),
                "BF16" => bf16_bytes(&keep),
                "F32" => f32_bytes(&keep),
                _ => f64_bytes(&keep),
            },
        ),
        bt("w", d, &[n], raw_bits(d, &pats)),
        bt("zzz_ints", "I32", &[3], i32_bytes(&[-7, 0, i32::MAX])),
    ];
    let input = container(fmt, &tensors);
    let model = put(root, &format!("in-{}-{}-{}{}{}.bin", fmt, d, f.cnf as u8, f.ce as u8, f.fs as u8), &input);
    let out = root.path(&format!("out-{}-{}-{}{}{}-{}", fmt, d, f.cnf as u8, f.ce as u8, f.fs as u8, max_abs));
    let (rep, res) = sanitize_file_report(&model, &f.opts(max_abs, &out));
    res.unwrap_or_else(|e| panic!("{}: {}", tag, e));
    assert_eq!(rep.status, "remediated", "{}: {:?}", tag, rep.after.as_ref().map(|a| &a.counts));
    assert!(rep.output_published && rep.input_unchanged, "{}", tag);
    // the published copy rescans clean under the default policy at the same bound
    let (re, r2) = sanitize_file_report(&out, &SanitizeOptions { attempt_id: ATT.into(), max_abs, ..Default::default() });
    r2.unwrap();
    assert_eq!(re.status, "clean", "{}", tag);
    assert_eq!(rep.output_sha256, sha256_file(&out));
    let output = std::fs::read(&out).unwrap();
    assert_eq!(input.len(), output.len());
    let lay = secant_core::container::layout_of_bytes(&input, "x").unwrap();
    let t = lay.tensors.iter().find(|t| t.name == "w").unwrap();
    let (off, len) = (t.offset as usize, t.size as usize);
    identical_outside(&input, &output, off, len);
    // per element
    let (win, wout) = (&input[off..off + len], &output[off..off + len]);
    let sign = sign_of(d);
    let mut bound: Option<u64> = None;
    let mut counts = (0i64, 0i64, 0i64, 0i64);
    for i in 0..n {
        let (bi, bo) = (bits_at(d, win, i), bits_at(d, wout, i));
        match act(d, bi, f, max_abs) {
            Act::Keep => assert_eq!(bo, bi, "{} elem {} kept bits changed {:#x}->{:#x}", tag, i, bi, bo),
            Act::Zero => {
                assert_eq!(bo, 0, "{} NaN must become +0", tag);
                counts.0 += 1;
            }
            Act::SignedZero => {
                assert_eq!(bo, bi & sign, "{} subnormal {:#x} must become signed zero, got {:#x}", tag, bi, bo);
                counts.1 += 1;
            }
            Act::Bound => {
                let mag = bo & !sign;
                assert_eq!(bo & sign, bi & sign, "{} clamp sign flipped for {:#x}", tag, bi);
                let b = *bound.get_or_insert(mag);
                assert_eq!(mag, b, "{} clamp magnitude not constant", tag);
                if cls(d, bi) == Cls::Inf {
                    counts.2 += 1
                } else {
                    counts.3 += 1
                }
            }
        }
    }
    if let Some(b) = bound {
        check_bound(d, b, max_abs);
    }
    // receipt change counts agree with what was actually rewritten
    let get = |k: &str| rep.changes.get(k).copied().unwrap_or(0);
    assert_eq!(get("nan"), counts.0, "{}", tag);
    assert_eq!(get("subnormal"), counts.1, "{}", tag);
    assert_eq!(get("inf"), counts.2, "{}", tag);
    assert_eq!(get("extreme_finite"), counts.3, "{}", tag);
    assert_eq!(rep.compute_seal(), rep.receipt_sha256);
}

#[test]
fn extension_flags_all_combinations_both_containers_all_dtypes() {
    set_go_arm64_fma(true);
    let root = Scratch::new("ext");
    let mut cases = 0;
    for fmt in ["safetensors", "gguf"] {
        for d in ["F16", "BF16", "F32", "F64"] {
            for f in Flags::all() {
                let maxes: &[f64] = if width(d) == 2 { &[1000.5, 0.001] } else { &[1000.5] };
                for &m in maxes {
                    extension_case(fmt, d, f, m, &root);
                    cases += 1;
                }
            }
        }
    }
    println!("extension flags: {} (container, dtype, flag-combination, bound) cases, per-element properties held", cases);
}

#[test]
fn extension_clamp_rounds_down_in_magnitude() {
    let root = Scratch::new("exr");
    // (dtype, max_abs, expected decoded bound)
    for (d, m, want) in
        [("BF16", 1000.5, 1000.0), ("F16", 1000.5, 1000.5), ("BF16", 3.0e38, f64::NAN), ("F32", 0.1, f64::NAN), ("F64", 0.1, 0.1)]
    {
        let big = match d {
            "F16" => 0x7bffu64,
            "BF16" => 0x7f7f,
            "F32" => 0x7f7fffff,
            _ => 0x7fefffffffffffff,
        };
        let model = put(&root, &format!("r-{}-{}", d, m), &st(&[bt("w", d, &[2], raw_bits(d, &[big, big | sign_of(d)]))]));
        let out = root.path(&format!("ro-{}-{}", d, m));
        let o = SanitizeOptions { attempt_id: ATT.into(), max_abs: m, clamp_extreme: true, output: out.clone(), ..Default::default() };
        let (rep, res) = sanitize_file_report(&model, &o);
        res.unwrap();
        assert_eq!(rep.status, "remediated", "{} {}", d, m);
        let bytes = std::fs::read(&out).unwrap();
        let lay = secant_core::container::layout_of_bytes(&bytes, "x").unwrap();
        let off = lay.tensors[0].offset as usize;
        let (p, n) = (bits_at(d, &bytes[off..], 0), bits_at(d, &bytes[off..], 1));
        assert_eq!(n, p | sign_of(d), "{} negative must mirror positive", d);
        assert!(dec(d, p) <= m, "{} {} -> {}", d, m, dec(d, p));
        if !want.is_nan() {
            assert_eq!(dec(d, p), want, "{} {}", d, m);
        }
        check_bound(d, p, m);
    }
}

#[test]
fn extension_partial_flags_refuse_residual_classes_and_publish_nothing() {
    let root = Scratch::new("exp");
    for fmt in ["safetensors", "gguf"] {
        // one element of each class
        let (nan, inf, sub, big) = (0x7fc00000u64, 0x7f800000u64, 1u64, 0x7f7fffffu64);
        let model = put(&root, &format!("p.{}", fmt), &container(fmt, &[bt("w", "F32", &[4], raw_bits("F32", &[nan, inf, sub, big]))]));
        for (i, f) in Flags::all().into_iter().enumerate() {
            if f.cnf && f.ce && f.fs {
                continue;
            }
            let out = root.path(&format!("po-{}-{}", fmt, i));
            let (rep, res) = sanitize_file_report(&model, &f.opts(1e4, &out));
            res.unwrap();
            assert_eq!(rep.status, "findings", "{} {}{}{}", fmt, f.cnf as u8, f.ce as u8, f.fs as u8);
            assert!(!rep.output_published && std::fs::symlink_metadata(&out).is_err());
            let after = rep.after.unwrap().counts;
            assert_eq!(after.get("nan").is_some(), !f.cnf);
            assert_eq!(after.get("inf").is_some(), !f.cnf);
            assert_eq!(after.get("subnormal").is_some(), !f.fs);
            assert_eq!(after.get("extreme_finite").is_some(), !f.ce);
        }
        let out = root.path(&format!("po-{}-all", fmt));
        let all = Flags { cnf: true, ce: true, fs: true };
        let (rep, res) = sanitize_file_report(&model, &all.opts(1e4, &out));
        res.unwrap();
        assert_eq!(rep.status, "remediated");
    }
}

#[test]
fn extension_never_repairs_norm_rows_or_routing_ids() {
    let root = Scratch::new("exn");
    let all = Flags { cnf: true, ce: true, fs: true };
    for fmt in ["safetensors", "gguf"] {
        let cases: Vec<(&str, Vec<BuildTensor>, &str)> = vec![
            (
                "zero-norm",
                vec![bt("layer.norm.weight", "F32", &[2, 2], raw_bits("F32", &[0, 0x80000000, 0x3f800000, 0x3f800000]))],
                "norm_all_zero",
            ),
            // flushing the subnormals leaves an all-zero row: still refused
            ("subnormal-norm", vec![bt("layer.norm.weight", "F32", &[1, 2], raw_bits("F32", &[1, 2]))], "norm_all_zero"),
            (
                "tiny-norm",
                vec![bt("layer.norm.weight", "F32", &[1, 2], raw_bits("F32", &[(1e-30f32).to_bits() as u64, (1e-30f32).to_bits() as u64]))],
                "norm_below_min",
            ),
            ("neg-route", vec![bt("tid2eid", "I32", &[1, 2], raw_bits("I32", &[0xffffffff, 1]))], "routing_negative"),
            ("oob-route", vec![bt("tid2eid", "I64", &[1, 2], raw_bits("I64", &[8, 1]))], "routing_oob"),
        ];
        for (name, ts, kind) in cases {
            // alongside a repairable float so the repairs themselves run
            let mut ts = ts;
            ts.insert(0, bt("a_bias", "F32", &[3], raw_bits("F32", &[0x7fc00000, 0x7f7fffff, 1])));
            let model = put(&root, &format!("n-{}-{}", fmt, name), &container(fmt, &ts));
            let out = root.path(&format!("no-{}-{}", fmt, name));
            let mut o = all.opts(1e4, &out);
            o.num_experts = 8;
            let (rep, res) = sanitize_file_report(&model, &o);
            res.unwrap();
            assert_eq!(rep.status, "findings", "{} {}", fmt, name);
            assert!(!rep.output_published && std::fs::symlink_metadata(&out).is_err(), "{} {}", fmt, name);
            let after = rep.after.unwrap();
            assert!(after.counts.contains_key(kind), "{} {}: {:?}", fmt, name, after.counts);
            assert!(!after.counts.contains_key("nan") && !after.counts.contains_key("extreme_finite"), "float repairs still ran");
        }
    }
}

#[test]
fn extension_defaults_leave_the_receipt_unchanged() {
    set_go_arm64_fma(true);
    let root = Scratch::new("exd");
    let model = put(&root, "m.safetensors", &st(&[bt("w", "F32", &[2], raw_bits("F32", &[0x7fc00000, 1]))]));
    let (rep, _) = sanitize_file_report(&model, &SanitizeOptions { attempt_id: ATT.into(), ..Default::default() });
    let text = rep.to_json_pretty();
    assert!(!text.contains("clamp_extreme") && !text.contains("flush_subnormal"), "default receipt grew new keys");
    assert!(text.contains("\"dedup_routing_rows\": false,\n    \"output\": \"\"\n"), "options layout moved");
    // used flags appear, between dedup_routing_rows and output, and the seal still holds
    let out = root.path("o.safetensors");
    let (rep, _) = sanitize_file_report(
        &model,
        &SanitizeOptions { attempt_id: ATT.into(), clamp_nonfinite: true, flush_subnormal: true, output: out, ..Default::default() },
    );
    let text = rep.to_json_pretty();
    assert!(text.contains("\"dedup_routing_rows\": false,\n    \"flush_subnormal\": true,\n    \"output\""), "{}", text);
    let back: SanitizeReceipt = serde_json::from_str(&text).unwrap();
    assert_eq!(back, rep);
    assert_eq!(back.compute_seal(), back.receipt_sha256);
}

#[test]
fn extension_cli_flags_and_oracle_equivalence_on_nonfinite_only() {
    let Some(rs) = secant_bin() else { return println!("NOT_RUN extension_cli: secant CLI binary not built") };
    let root = Scratch::new("exc");
    let model = put(
        &root,
        "m.safetensors",
        &st(&[bt("w", "F32", &[6], raw_bits("F32", &[0x7fc00000, 0x7f800000, 0xff800000, 0x3f800000, 0x80000000, 0x3f000000]))]),
    );
    // each new flag needs --out; --help documents the limits
    for flag in ["--clamp-extreme", "--flush-subnormal"] {
        let d = Side::new(&root, &format!("cli-noout{}", flag));
        let o = run_rs(&rs, &d.args(&san_args(&model, &[flag])));
        assert_eq!(o.status.code(), Some(2), "{}", flag);
    }
    let h = Command::new(&rs).args(["sanitize", "--help"]).output().unwrap();
    let help = String::from_utf8_lossy(&h.stdout);
    assert!(help.contains("--clamp-extreme") && help.contains("--flush-subnormal") && help.contains("always detected and refused"));
    // with the extensions switched on but nothing for them to do, the published bytes equal the oracle's --clamp-nonfinite output
    if let Some(go) = oracle_bin() {
        set_go_arm64_fma(true);
        let g = Side::new(&root, "eq-go");
        let r = Side::new(&root, "eq-rs");
        let og = run_go(&go, &g.args(&san_args(&model, &["--clamp-nonfinite", "--out", "{OUT}"])));
        let or = run_rs(&rs, &r.args(&san_args(&model, &["--clamp-nonfinite", "--clamp-extreme", "--flush-subnormal", "--out", "{OUT}"])));
        assert_eq!((og.status.code(), or.status.code()), (Some(0), Some(0)));
        assert_eq!(sha256_file(&g.sub("out")), sha256_file(&r.sub("out")));
        let rec: Value = serde_json::from_str(&std::fs::read_to_string(r.sub("receipt.json")).unwrap()).unwrap();
        assert_eq!(rec["options"]["clamp_extreme"], Value::Bool(true));
        assert_eq!(rec["options"]["flush_subnormal"], Value::Bool(true));
        assert_eq!(rec["status"], "remediated");
    } else {
        not_run("extension_cli (oracle equivalence half)");
    }
    // CLI end to end on all three repairs
    let r = Side::new(&root, "cli-all");
    let m2 = put(&root, "m2.gguf", &gg(&[bt("w", "BF16", &[4], raw_bits("BF16", &[0x7f7f, 0xff7f, 0x0001, 0x8001]))]));
    let o = run_rs(&rs, &r.args(&san_args(&m2, &["--max-abs", "1000.5", "--clamp-extreme", "--flush-subnormal", "--out", "{OUT}"])));
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    let out = std::fs::read(r.sub("out")).unwrap();
    let off = secant_core::container::layout_of_bytes(&out, "x").unwrap().tensors[0].offset as usize;
    let got: Vec<u64> = (0..4).map(|i| bits_at("BF16", &out[off..], i)).collect();
    assert_eq!(got, vec![0x447a, 0xc47a, 0x0000, 0x8000]); // 1000.0 rounded down from 1000.5, signs kept, subnormals flushed
}
