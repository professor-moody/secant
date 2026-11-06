//! Regression witness for the GGUF header-window growth loop in
//! `security.rs::sanitize_layout` (the `ends_with(": EOF")` predicate).
//!
//! The scanner walks a 1 MiB prefix and, on a short read, grows the window 4x.
//! `go_scan_error` rewrites `...at offset N: unexpected EOF` to `...at offset N: EOF`
//! when N equals the window length, so a header whose metadata ends EXACTLY on a
//! window edge used to be refused with exit 2 although the file is valid.
//!
//! Each case builds a tensor-bearing GGUF v3 (one F32 tensor `w`, 4 elements) whose
//! single string metadata value is sized so that a field ends at a chosen offset.
//! Controls (edge falls mid-field) pass on the old predicate too; the exact-edge
//! cases are red on the old predicate and green on the new one.

mod common;
use common::*;
use secant_core::security::{sanitize_file_report, SanitizeOptions};

const ATT: &str = "att-983da2c58c4878fc";

fn put_str(out: &mut Vec<u8>, s: &[u8]) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s);
}

/// GGUF v3: header(24) + kv("k" -> string of `vlen` bytes) + one F32 tensor "w" [4].
/// Returns the file bytes and the offset at which the metadata value ends.
fn gguf_with_value_len(vlen: usize) -> (Vec<u8>, usize) {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    b.extend_from_slice(&1u64.to_le_bytes()); // kv_count
    put_str(&mut b, b"k");
    b.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
    put_str(&mut b, &vec![b'A'; vlen]);
    let value_end = b.len();
    put_str(&mut b, b"w");
    b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    b.extend_from_slice(&4u64.to_le_bytes()); // dim0
    b.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32
    b.extend_from_slice(&0u64.to_le_bytes()); // data offset within the data section
    while b.len() % 32 != 0 {
        b.push(0);
    }
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    (b, value_end)
}

/// vlen such that the metadata value ends exactly at `edge`.
fn vlen_for_edge(edge: usize) -> usize {
    // 24 header + (8 + 1) key + 4 type + 8 value length prefix
    edge - (24 + 9 + 4 + 8)
}

fn run(tag: &str, vlen: usize, expect_value_end: Option<usize>) -> (String, String) {
    let root = Scratch::new(tag);
    let (bytes, value_end) = gguf_with_value_len(vlen);
    if let Some(e) = expect_value_end {
        assert_eq!(value_end, e, "fixture geometry");
    }
    let p = root.path("m.gguf");
    std::fs::write(&p, &bytes).unwrap();
    let opts = SanitizeOptions { attempt_id: ATT.into(), ..Default::default() };
    let (r, res) = sanitize_file_report(&p, &opts);
    let err = res.err().map(|e| e.message().to_string()).unwrap_or_default();
    (r.status.clone(), err)
}

#[test]
fn control_edge_falls_mid_field_1mib() {
    // value ends 7 bytes after the 1 MiB edge: the failing take() starts before the edge,
    // so the message keeps "unexpected EOF" and the old predicate already retried.
    let (status, err) = run("w1", vlen_for_edge((1 << 20) + 7), None);
    assert_eq!((status.as_str(), err.as_str()), ("clean", ""), "mid-field edge");
}

#[test]
fn exact_edge_1mib_is_not_refused() {
    let (status, err) = run("w2", vlen_for_edge(1 << 20), Some(1 << 20));
    assert_eq!((status.as_str(), err.as_str()), ("clean", ""), "exact 1 MiB edge");
}

#[test]
fn exact_edge_4mib_after_one_regrow_is_not_refused() {
    let (status, err) = run("w3", vlen_for_edge(4 << 20), Some(4 << 20));
    assert_eq!((status.as_str(), err.as_str()), ("clean", ""), "exact 4 MiB edge");
}

#[test]
fn truncated_header_still_fails_closed() {
    // header claims a 3 MiB value but the file stops at 1.5 MiB: genuine truncation must
    // stay an error (status unknown), not loop forever and not report clean.
    let (mut bytes, _) = gguf_with_value_len(3 << 20);
    bytes.truncate((1 << 20) + (1 << 19));
    let root = Scratch::new("w4");
    let p = root.path("m.gguf");
    std::fs::write(&p, &bytes).unwrap();
    let opts = SanitizeOptions { attempt_id: ATT.into(), ..Default::default() };
    let (r, res) = sanitize_file_report(&p, &opts);
    assert!(res.is_err(), "truncated header must be refused");
    assert_eq!(r.status, "unknown");
}
