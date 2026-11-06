#[path = "../../secant-core/tests/common/mod.rs"]
mod common;
use common::*;
use secant_core::{
    builder::{build_safetensors, BuildTensor},
    subspace::{load_numpy_file, npy_bytes_f64},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::TcpListener,
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_secant")).args(args).env_remove("HF_TOKEN").output().unwrap()
}

#[test]
fn all_help_pages_have_examples_and_human_reports_are_not_json() {
    for command in [
        "inspect",
        "read",
        "compose",
        "patch",
        "verify-patch",
        "sanitize",
        "probe",
        "heal",
        "verify-subspace",
        "quant",
        "rotation-test",
        "fetch-tensor",
    ] {
        let o = run(&[command, "--help"]);
        assert!(o.status.success(), "{}", command);
        let text = String::from_utf8(o.stdout).unwrap();
        assert!(text.contains("Example"), "{}: {}", command, text);
    }
    for sub in ["witness", "patch", "roundtrip"] {
        assert!(run(&["quant", sub, "--help"]).status.success());
    }
    let dir = Scratch::new("human");
    let path = dir.path("w.npy");
    write(&path, &npy_bytes_f64(&vec![0.25; 64], &[2, 32]).unwrap());
    let o = run(&["quant", "roundtrip", "--npy", &path]);
    assert!(o.status.success());
    let text = String::from_utf8(o.stdout).unwrap();
    assert!(text.contains("Quantization round-trip") && text.contains("rmse") && !text.contains("\"rows\""));
    let json = run(&["quant", "roundtrip", "--npy", &path, "--json"]);
    assert_eq!(serde_json::from_slice::<Value>(&json.stdout).unwrap()["rows"], 2);
}

// Minimal in-process HTTP fixture, with bounded lifetime even when a CLI fails
// before issuing every request. No external services or helper scripts.
struct Server {
    endpoint: String,
    done: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<usize>>,
}
impl Server {
    fn new(mode: &'static str, dtype: &str) -> Self {
        let data = if dtype == "BF16" { vec![0x80, 0x3f, 0, 0x40, 0x40, 0x40, 0x80, 0x40] } else { f32_bytes(&[1.0, 2.0, 3.0, 4.0]) };
        let mut shard =
            build_safetensors(&[BuildTensor { name: "w".into(), dtype: dtype.into(), shape: vec![2, 2], data }], &BTreeMap::new()).unwrap();
        if mode == "byte-overflow" || mode == "bad-offsets" {
            let h = serde_json::to_vec(
                &json!({"w": {"dtype":"F64", "shape": if mode == "byte-overflow" {vec![u64::MAX]} else {vec![2]}, "data_offsets": [16,0]}}),
            )
            .unwrap();
            shard = (h.len() as u64).to_le_bytes().to_vec();
            shard.extend(h);
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let done = Arc::new(AtomicBool::new(false));
        let stop = done.clone();
        let join = thread::spawn(move || {
            let mut served = 0;
            while !stop.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(v) => v,
                    Err(_) => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                };
                stream.set_nonblocking(false).unwrap();
                stream.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
                let mut req = Vec::new();
                let mut b = [0; 1];
                while stream.read(&mut b).unwrap_or(0) == 1 {
                    req.push(b[0]);
                    if req.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8(req).unwrap();
                let (status, body, extra) = if request.starts_with("GET /api/models/org/model/revision/main ") {
                    ("200 OK", serde_json::to_vec(&json!({"sha": "a".repeat(40)})).unwrap(), String::new())
                } else {
                    assert!(request.contains(&format!("/org/model/resolve/{}/", "a".repeat(40))), "not pinned: {}", request);
                    if request.contains("model.safetensors.index.json") {
                        if mode == "unsharded" {
                            ("404 Not Found", vec![], String::new())
                        } else {
                            ("200 OK", serde_json::to_vec(&json!({"weight_map": {"w": "model.safetensors"}})).unwrap(), String::new())
                        }
                    } else {
                        let lower = request.to_lowercase();
                        let range = lower.lines().find_map(|l| l.strip_prefix("range: bytes=")).unwrap();
                        let (start, end) = range.split_once('-').unwrap();
                        let (start, end): (usize, usize) = (start.parse().unwrap(), end.trim().parse().unwrap());
                        if mode == "ignore" {
                            ("200 OK", vec![], String::new())
                        } else {
                            let mut body = shard[start..=end].to_vec();
                            served += body.len();
                            if mode == "short" {
                                body.pop();
                            }
                            let reported_start = if mode == "wrong-range" { start + 1 } else { start };
                            ("206 Partial Content", body, format!("Content-Range: bytes {}-{}/{}\r\n", reported_start, end, shard.len()))
                        }
                    }
                };
                let h = format!("HTTP/1.1 {}\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n", status, body.len(), extra);
                let _ = stream.write_all(h.as_bytes());
                let _ = stream.write_all(&body);
            }
            served
        });
        Self { endpoint, done, join: Some(join) }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        self.join.take().unwrap().join().unwrap();
    }
}

#[test]
fn fetch_pins_commit_slices_payload_widens_bf16_and_never_overwrites() {
    for mode in ["normal", "unsharded"] {
        for dtype in ["F32", "BF16"] {
            let server = Server::new(mode, dtype);
            let dir = Scratch::new("fetch");
            let out = dir.path("w.npy");
            let args = [
                "fetch-tensor",
                "--repo",
                "org/model",
                "--tensor",
                "w",
                "--endpoint",
                &server.endpoint,
                "--out",
                &out,
                "--row-start",
                "1",
                "--row-count",
                "1",
                "--attempt-id",
                "test",
                "--json",
            ];
            let o = run(&args);
            assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
            let r: Value = serde_json::from_slice(&o.stdout).unwrap();
            assert_eq!(r["commit"], "a".repeat(40));
            assert_eq!(r["selected_shape"], json!([1, 2]));
            assert_eq!(r["attempt_id"], "test");
            let arr = load_numpy_file(&out).unwrap();
            assert_eq!(arr.shape, vec![1, 2]);
            assert_eq!(arr.to_f64(), vec![3.0, 4.0]);
            let receipt: Value = serde_json::from_slice(&std::fs::read(dir.path("receipt.json")).unwrap()).unwrap();
            assert_eq!(r, receipt);
            assert_eq!(run(&args).status.code(), Some(2));
            assert_eq!(load_numpy_file(&out).unwrap().to_f64(), vec![3.0, 4.0]);
        }
    }
}

#[test]
fn fetch_refuses_ignored_wrong_truncated_or_overbudget_ranges_without_outputs() {
    for mode in ["ignore", "wrong-range", "short", "normal", "byte-overflow", "bad-offsets"] {
        let server = Server::new(mode, "F32");
        let dir = Scratch::new("fetch-fail");
        let out = dir.path("w.npy");
        let o = run(&[
            "fetch-tensor",
            "--repo",
            "org/model",
            "--tensor",
            "w",
            "--endpoint",
            &server.endpoint,
            "--out",
            &out,
            "--max-bytes",
            "4",
        ]);
        assert_eq!(o.status.code(), Some(2), "{}", String::from_utf8_lossy(&o.stderr));
        assert!(!Path::new(&out).exists());
        assert!(!Path::new(&dir.path("receipt.json")).exists());
    }
}
use std::path::Path;
