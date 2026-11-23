#!/usr/bin/env python3
"""GGUF K-quant, IQ, ternary, and MXFP4 value-layer sanitization benchmark.

Builds minimal, loadable 1-layer GGUF v3 checkpoints across all 24 reviewed
GGUF quantized block formats (clean controls + planted NaN/Inf/extreme scale
header cases), runs `secant sanitize --dry-run --json --sarif`, and optionally
runs a side-by-side differential comparison against `llama.cpp --check-tensors`
when `--llama-cli` is supplied.

Only stdlib Python is required on Linux and macOS. Output goes to a new
`--run-dir` directory with a verified `SHA256SUMS` manifest, `summary.json`,
and a Markdown FACTS report (`summary.md`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time


DEFAULT_ATTEMPT_ID = "att-gguf-quant-coverage-v1"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    offset: int
    encoding: str  # "f16" | "f32" | "iq1m_packed_f16" | "e8m0"


@dataclass(frozen=True)
class QuantFormatSpec:
    name: str
    family: str  # "standard" | "k-quant" | "iq" | "ternary" | "mxfp4"
    ggml_type: int
    block_elems: int
    block_bytes: int
    fields: tuple[FieldSpec, ...]


FORMATS: tuple[QuantFormatSpec, ...] = (
    QuantFormatSpec("Q4_0", "standard", 2, 32, 18, (FieldSpec("d", 0, "f16"),)),
    QuantFormatSpec("Q4_1", "standard", 3, 32, 20, (FieldSpec("d", 0, "f16"), FieldSpec("m", 2, "f16"))),
    QuantFormatSpec("Q5_0", "standard", 6, 32, 22, (FieldSpec("d", 0, "f16"),)),
    QuantFormatSpec("Q5_1", "standard", 7, 32, 24, (FieldSpec("d", 0, "f16"), FieldSpec("m", 2, "f16"))),
    QuantFormatSpec("Q8_0", "standard", 8, 32, 34, (FieldSpec("d", 0, "f16"),)),
    QuantFormatSpec("Q8_1", "standard", 9, 32, 36, (FieldSpec("d", 0, "f16"), FieldSpec("s", 2, "f16"))),
    QuantFormatSpec("Q2_K", "k-quant", 10, 256, 84, (FieldSpec("d", 80, "f16"), FieldSpec("dmin", 82, "f16"))),
    QuantFormatSpec("Q3_K", "k-quant", 11, 256, 110, (FieldSpec("d", 108, "f16"),)),
    QuantFormatSpec("Q4_K", "k-quant", 12, 256, 144, (FieldSpec("d", 0, "f16"), FieldSpec("dmin", 2, "f16"))),
    QuantFormatSpec("Q5_K", "k-quant", 13, 256, 176, (FieldSpec("d", 0, "f16"), FieldSpec("dmin", 2, "f16"))),
    QuantFormatSpec("Q6_K", "k-quant", 14, 256, 210, (FieldSpec("d", 208, "f16"),)),
    QuantFormatSpec("Q8_K", "k-quant", 15, 256, 292, (FieldSpec("d", 0, "f32"),)),
    QuantFormatSpec("IQ2_XXS", "iq", 16, 256, 66, (FieldSpec("d", 0, "f16"),)),
    QuantFormatSpec("IQ2_XS", "iq", 17, 256, 74, (FieldSpec("d", 0, "f16"),)),
    QuantFormatSpec("IQ3_XXS", "iq", 18, 256, 98, (FieldSpec("d", 0, "f16"),)),
    QuantFormatSpec("IQ1_S", "iq", 19, 256, 50, (FieldSpec("d", 0, "f16"),)),
    QuantFormatSpec("IQ4_NL", "iq", 20, 32, 18, (FieldSpec("d", 0, "f16"),)),
    QuantFormatSpec("IQ3_S", "iq", 21, 256, 110, (FieldSpec("d", 0, "f16"),)),
    QuantFormatSpec("IQ2_S", "iq", 22, 256, 82, (FieldSpec("d", 0, "f16"),)),
    QuantFormatSpec("IQ4_XS", "iq", 23, 256, 136, (FieldSpec("d", 0, "f16"),)),
    QuantFormatSpec("IQ1_M", "iq", 29, 256, 56, (FieldSpec("d", 48, "iq1m_packed_f16"),)),
    QuantFormatSpec("TQ1_0", "ternary", 34, 256, 54, (FieldSpec("d", 52, "f16"),)),
    QuantFormatSpec("TQ2_0", "ternary", 35, 256, 66, (FieldSpec("d", 64, "f16"),)),
    QuantFormatSpec("MXFP4", "mxfp4", 39, 32, 17, (FieldSpec("e", 0, "e8m0"),)),
)

FORMATS_BY_NAME: dict[str, QuantFormatSpec] = {spec.name: spec for spec in FORMATS}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json_new(path: Path, obj: object) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def encoding_bits(encoding: str, kind: str) -> int | None:
    if encoding in ("f16", "iq1m_packed_f16"):
        if kind == "unit":
            return 0x3C00
        if kind == "nan":
            return 0x7E00
        if kind == "inf":
            return 0x7C00
        if kind == "extreme":
            return 0x7BFF
    elif encoding == "f32":
        if kind == "unit":
            return 0x3F800000
        if kind == "nan":
            return 0x7FC00000
        if kind == "inf":
            return 0x7F800000
        if kind == "extreme":
            return 0x7F7FFFFF
    elif encoding == "e8m0":
        if kind == "unit":
            return 0x80
        if kind == "nan":
            return 0xFF
        if kind == "inf":
            return None
        if kind == "extreme":
            return 0xFE
    raise ValueError(f"unsupported encoding/kind: {encoding}/{kind}")


def write_field_bits(blk: bytearray, field: FieldSpec, bits: int) -> None:
    off = field.offset
    if field.encoding == "f16":
        blk[off : off + 2] = struct.pack("<H", bits & 0xFFFF)
    elif field.encoding == "f32":
        blk[off : off + 4] = struct.pack("<I", bits & 0xFFFFFFFF)
    elif field.encoding == "iq1m_packed_f16":
        d = bits & 0xFFFF
        for i, shift in enumerate((12, 8, 4, 0)):
            nibble_mask = 0x000F << (4 * i)
            word = struct.unpack_from("<H", blk, off + 2 * i)[0]
            word = (word & 0x0FFF) | ((d & nibble_mask) << shift)
            struct.pack_into("<H", blk, off + 2 * i, word)
    elif field.encoding == "e8m0":
        blk[off] = bits & 0xFF
    else:
        raise ValueError(f"unknown encoding: {field.encoding}")


def read_field_bits(blk: bytes | bytearray, field: FieldSpec) -> int:
    off = field.offset
    if field.encoding == "f16":
        return struct.unpack_from("<H", blk, off)[0]
    if field.encoding == "f32":
        return struct.unpack_from("<I", blk, off)[0]
    if field.encoding == "iq1m_packed_f16":
        sc = struct.unpack_from("<4H", blk, off)
        return (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000)
    if field.encoding == "e8m0":
        return blk[off]
    raise ValueError(f"unknown encoding: {field.encoding}")


def build_quant_payload(spec: QuantFormatSpec, n_rows: int, n_cols: int, poison: tuple[FieldSpec, str] | None) -> bytes:
    """Build deterministic quantized tensor bytes for shape [n_cols, n_rows] (GGUF ne0=n_cols)."""
    if n_cols % spec.block_elems != 0:
        raise ValueError(f"{spec.name} requires n_cols divisible by {spec.block_elems}, got {n_cols}")
    n_blocks = n_rows * (n_cols // spec.block_elems)
    buf = bytearray(n_blocks * spec.block_bytes)
    for blk_idx in range(n_blocks):
        blk = bytearray((i * 29 + blk_idx * 17 + spec.ggml_type + 7) % 251 for i in range(spec.block_bytes))
        for field in spec.fields:
            unit = encoding_bits(field.encoding, "unit")
            assert unit is not None
            write_field_bits(blk, field, unit)
        if blk_idx == 0 and poison is not None:
            p_field, p_kind = poison
            p_bits = encoding_bits(p_field.encoding, p_kind)
            if p_bits is None:
                raise ValueError(f"{spec.name}.{p_field.name} has no {p_kind} encoding")
            write_field_bits(blk, p_field, p_bits)
        start = blk_idx * spec.block_bytes
        buf[start : start + spec.block_bytes] = blk
    return bytes(buf)


def _w_str(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _w_kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _w_str(key) + struct.pack("<I", vtype) + payload


def _w_arr(key: str, elem_type: int, items: list[bytes]) -> bytes:
    return _w_str(key) + struct.pack("<IIQ", 9, elem_type, len(items)) + b"".join(items)


def build_loadable_llama_gguf(spec: QuantFormatSpec, poison: tuple[FieldSpec, str] | None) -> bytes:
    """Build a minimal 1-layer llama GGUF v3 with `blk.0.ffn_down.weight` in `spec`."""
    vocab = 32
    n_embd = 256
    n_ff = 256
    n_head = 4
    n_head_kv = 4
    n_ctx = 64

    tok_strings = ["<unk>", "<s>", "</s>", "\u2581", "a", "b"] + [f"t{i}" for i in range(vocab - 6)]
    toks = [_w_str(s) for s in tok_strings]
    scores = [struct.pack("<f", 0.0) for _ in range(vocab)]
    ttypes = [struct.pack("<i", 3 if i < 3 else 1) for i in range(vocab)]

    kvs = [
        _w_kv("general.architecture", 8, _w_str("llama")),
        _w_kv("general.name", 8, _w_str(f"secant-quant-bench-{spec.name.lower()}")),
        _w_kv("general.alignment", 4, struct.pack("<I", 32)),
        _w_kv("llama.context_length", 4, struct.pack("<I", n_ctx)),
        _w_kv("llama.embedding_length", 4, struct.pack("<I", n_embd)),
        _w_kv("llama.block_count", 4, struct.pack("<I", 1)),
        _w_kv("llama.feed_forward_length", 4, struct.pack("<I", n_ff)),
        _w_kv("llama.attention.head_count", 4, struct.pack("<I", n_head)),
        _w_kv("llama.attention.head_count_kv", 4, struct.pack("<I", n_head_kv)),
        _w_kv("llama.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", 1e-5)),
        _w_kv("llama.rope.dimension_count", 4, struct.pack("<I", n_embd // n_head)),
        _w_kv("tokenizer.ggml.model", 8, _w_str("llama")),
        _w_arr("tokenizer.ggml.tokens", 8, toks),
        _w_arr("tokenizer.ggml.scores", 6, scores),
        _w_arr("tokenizer.ggml.token_type", 5, ttypes),
        _w_kv("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 1)),
        _w_kv("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 2)),
        _w_kv("tokenizer.ggml.unknown_token_id", 4, struct.pack("<I", 0)),
    ]

    f32_ones = struct.pack(f"<{n_embd}f", *([1.0] * n_embd))
    f32_embd = struct.pack(f"<{n_embd * vocab}f", *([0.01] * (n_embd * vocab)))
    f32_mat = struct.pack(f"<{n_embd * n_embd}f", *([0.01] * (n_embd * n_embd)))
    target_bytes = build_quant_payload(spec, n_rows=n_embd, n_cols=n_ff, poison=poison)

    tensors = [
        ("token_embd.weight", [n_embd, vocab], 0, f32_embd),
        ("output_norm.weight", [n_embd], 0, f32_ones),
        ("output.weight", [n_embd, vocab], 0, f32_embd),
        ("blk.0.attn_norm.weight", [n_embd], 0, f32_ones),
        ("blk.0.attn_q.weight", [n_embd, n_embd], 0, f32_mat),
        ("blk.0.attn_k.weight", [n_embd, n_embd], 0, f32_mat),
        ("blk.0.attn_v.weight", [n_embd, n_embd], 0, f32_mat),
        ("blk.0.attn_output.weight", [n_embd, n_embd], 0, f32_mat),
        ("blk.0.ffn_norm.weight", [n_embd], 0, f32_ones),
        ("blk.0.ffn_gate.weight", [n_embd, n_ff], 0, f32_mat),
        ("blk.0.ffn_up.weight", [n_embd, n_ff], 0, f32_mat),
        ("blk.0.ffn_down.weight", [n_ff, n_embd], spec.ggml_type, target_bytes),
    ]

    out = bytearray()
    out += b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    for kv in kvs:
        out += kv
    off = 0
    for name, dims, tid, data in tensors:
        out += _w_str(name) + struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<IQ", tid, off)
        off += ((len(data) + 31) // 32) * 32
    out += b"\x00" * ((32 - (len(out) % 32)) % 32)
    for _, _, _, data in tensors:
        out += data
        out += b"\x00" * ((32 - (len(data) % 32)) % 32)
    return bytes(out)


def run_secant(binary: Path, model_path: Path, receipt_path: Path, sarif_path: Path, attempt_id: str) -> dict:
    cmd = [
        str(binary),
        "sanitize",
        "--model",
        str(model_path),
        "--dry-run",
        "--json",
        "--receipt",
        str(receipt_path),
        "--sarif",
        str(sarif_path),
        "--attempt-id",
        attempt_id,
    ]
    t0 = time.perf_counter_ns()
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
    sarif = json.loads(sarif_path.read_text(encoding="utf-8")) if sarif_path.is_file() else {}
    sarif_rules = [r["ruleId"] for r in sarif.get("runs", [{}])[0].get("results", [])] if sarif else []
    before = receipt.get("before") or {}
    return {
        "exit_code": proc.returncode,
        "status": receipt.get("status", "ERROR"),
        "wall_ms": round(elapsed_ms, 3),
        "total_findings": before.get("total_findings", 0),
        "counts": before.get("counts", {}),
        "first_example": (before.get("examples") or [None])[0],
        "sarif_result_count": len(sarif_rules),
        "sarif_rule_ids": sorted(set(sarif_rules)),
    }


def classify_llama_output(returncode: int, stderr: str) -> str:
    low = stderr.lower()
    if returncode == 0:
        return "clean"
    if "ggml_validate_row_data: found nan" in low:
        return "rejected_nan"
    if "ggml_validate_row_data: found inf" in low:
        return "rejected_inf"
    if "ggml_validate_row_data: found invalid e value" in low:
        return "rejected_invalid_e8m0"
    if "ggml_validate_row_data: invalid type" in low:
        return "unsupported_type"
    if "found tensors with invalid data" in low or "has invalid data" in low:
        return "rejected_invalid_data"
    if "cannot be used with" in low or "unsupported" in low or "not supported" in low or "failed to load model" in low:
        return "loader_error"
    return f"other_exit_{returncode}"


def detect_llama_non_interactive_flags(llama_cli: Path) -> list[str]:
    """Probe `llama-cli --help` once to choose modern `--single-turn` or legacy `-no-cnv`."""
    try:
        proc = subprocess.run([str(llama_cli), "--help"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
        help_text = proc.stdout + "\n" + proc.stderr
    except (OSError, subprocess.SubprocessError):
        return []
    if "--single-turn" in help_text:
        flags = ["--single-turn"]
        if "--simple-io" in help_text:
            flags.append("--simple-io")
        return flags
    if "-no-cnv" in help_text:
        return ["-no-cnv"]
    return []


def run_llama_check(llama_cli: Path, model_path: Path, extra_flags: list[str] | None = None) -> dict:
    cmd = [
        str(llama_cli),
        "-m",
        str(model_path),
        "--check-tensors",
        "-p",
        "a",
        "-n",
        "1",
        "--no-warmup",
        "-ngl",
        "0",
    ] + (extra_flags or [])
    t0 = time.perf_counter_ns()
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    verdict = classify_llama_output(proc.returncode, proc.stderr)
    excerpt = [
        line.strip()
        for line in proc.stderr.splitlines()
        if any(tok in line.lower() for tok in ("ggml_validate_row_data", "has invalid data", "error loading model", "invalid type"))
    ][:4]
    return {
        "exit_code": proc.returncode,
        "verdict": verdict,
        "wall_ms": round(elapsed_ms, 3),
        "excerpt": excerpt,
    }


def seal_manifest(run_dir: Path) -> str:
    lines = [f"{sha256_file(p)}  {p.relative_to(run_dir).as_posix()}\n" for p in sorted(run_dir.rglob("*")) if p.is_file()]
    manifest = run_dir / "SHA256SUMS"
    with manifest.open("x", encoding="utf-8") as f:
        f.writelines(lines)
    for line in lines:
        digest, rel = line.rstrip("\n").split("  ", 1)
        if sha256_file(run_dir / rel) != digest:
            raise RuntimeError(f"manifest verification failed for {rel}")
    return sha256_file(manifest)


def render_markdown(summary: dict) -> str:
    s = summary["secant_summary"]
    l = summary["llama_cpp_summary"]
    lines = [
        "# GGUF K-Quant, IQ, Ternary, and MXFP4 Value-Layer Coverage Facts",
        "",
        f"Attempt: `{summary['attempt_id']}`. Platform: `{summary['platform']}`.",
        f"Formats covered: **{summary['formats_tested']}** ({', '.join(summary['format_names'])}).",
        "",
        "## Headline Results",
        "",
        f"- Clean controls: **{s['clean_passed']}/{s['clean_total']}** (`0.0%` synthetic FPR, exit `0`, `0` findings).",
        f"- Planted NaN scale headers: **{s['nan_caught']}/{s['nan_total']}** caught (exit `1`, valid SARIF `SECANT-B3-2A-QUANT-SCALE-NAN`).",
        f"- Planted Inf scale headers: **{s['inf_caught']}/{s['inf_total']}** caught (exit `1`, valid SARIF `SECANT-B3-2A-QUANT-SCALE-INF`).",
        f"- Secant mean scan latency: **{s['mean_wall_ms']:.2f} ms** per 1-layer checkpoint.",
    ]
    if l["status"] == "RUN":
        lines += [
            f"- `llama.cpp --check-tensors` differential status: **RUN** across `{l['cases_run']}` cases "
            f"(mean load+check latency **{l['mean_wall_ms']:.2f} ms**; "
            f"clean accepted: **{l['clean_accepted']}/{l['clean_total']}**, "
            f"planted NaN rejected by validator: **{l['nan_rejected_validator']}/{l['nan_total']}**, "
            f"planted Inf rejected by validator: **{l['inf_rejected_validator']}/{l['inf_total']}**, "
            f"unsupported/loader-only types: **{l['unsupported_or_loader_cases']}**).",
        ]
    else:
        lines.append("- `llama.cpp --check-tensors` differential status: **NOT_RUN** (`--llama-cli` not supplied).")

    lines += [
        "",
        "## Per-Format Differential Matrix",
        "",
        "| Format | Family | ggml ID | Case | Secant Exit | Secant Status | Secant Finding / Bits | Secant ms | llama.cpp Verdict | llama.cpp ms |",
        "|---|---|---:|---|---:|---|---|---:|---|---:|",
    ]
    for row in summary["cases"]:
        sec = row["secant"]
        ex = sec.get("first_example") or {}
        detail = f"{ex.get('kind', '')} {ex.get('field', '')}={ex.get('bits', '')}".strip() if ex else "clean"
        lc = row.get("llama_cpp")
        lc_verdict = lc["verdict"] if lc else "NOT_RUN"
        lc_ms = f"{lc['wall_ms']:.1f}" if lc else "-"
        lines.append(
            f"| `{row['format']}` | {row['family']} | {row['ggml_type']} | `{row['case']}` | "
            f"{sec['exit_code']} | `{sec['status']}` | `{detail}` | {sec['wall_ms']:.1f} | `{lc_verdict}` | {lc_ms} |"
        )
    lines += [
        "",
        f"Gate: `{summary['gate']}`. Evidence manifest SHA-256: `{summary['evidence_manifest_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def run_benchmark(
    run_dir: Path,
    binary: Path,
    llama_cli: Path | None,
    attempt_id: str,
    selected_formats: list[QuantFormatSpec],
) -> tuple[int, dict]:
    run_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("fixtures", "receipts", "sarif"):
        (run_dir / sub).mkdir()

    llama_flags = detect_llama_non_interactive_flags(llama_cli) if llama_cli is not None else []
    cases_out: list[dict] = []
    for spec in selected_formats:
        # 1. Clean control
        variants: list[tuple[str, tuple[FieldSpec, str] | None, int, str | None]] = [("clean", None, 0, None)]
        # 2. Planted NaN per header field
        for field in spec.fields:
            variants.append((f"nan_{field.name}", (field, "nan"), 1, "quant_scale_nan"))
        # 3. Planted +Inf per float header field
        for field in spec.fields:
            if encoding_bits(field.encoding, "inf") is not None:
                variants.append((f"inf_{field.name}", (field, "inf"), 1, "quant_scale_inf"))

        for case_name, poison, expected_exit, expected_kind in variants:
            slug = f"{spec.name.lower()}_{case_name}"
            model_path = run_dir / "fixtures" / f"{slug}.gguf"
            model_bytes = build_loadable_llama_gguf(spec, poison)
            model_path.write_bytes(model_bytes)

            sec = run_secant(
                binary=binary,
                model_path=model_path,
                receipt_path=run_dir / "receipts" / f"{slug}.json",
                sarif_path=run_dir / "sarif" / f"{slug}.sarif",
                attempt_id=attempt_id,
            )
            errors: list[str] = []
            if sec["exit_code"] != expected_exit:
                errors.append(f"expected exit {expected_exit}, got {sec['exit_code']}")
            if expected_kind is None:
                if sec["total_findings"] != 0 or sec["sarif_result_count"] != 0:
                    errors.append("clean control reported findings")
            else:
                if sec["counts"].get(expected_kind, 0) != 1:
                    errors.append(f"expected 1 {expected_kind}, got {sec['counts']}")
                want_rule = "SECANT-B3-2A-QUANT-SCALE-NAN" if expected_kind == "quant_scale_nan" else "SECANT-B3-2A-QUANT-SCALE-INF"
                if sec["sarif_rule_ids"] != [want_rule]:
                    errors.append(f"expected SARIF rule [{want_rule}], got {sec['sarif_rule_ids']}")

            llama_res = run_llama_check(llama_cli, model_path, llama_flags) if llama_cli is not None else None
            cases_out.append(
                {
                    "format": spec.name,
                    "family": spec.family,
                    "ggml_type": spec.ggml_type,
                    "case": case_name,
                    "expected_exit": expected_exit,
                    "expected_kind": expected_kind,
                    "fixture_sha256": sha256_file(model_path),
                    "secant": sec,
                    "llama_cpp": llama_res,
                    "errors": errors,
                }
            )

    clean_rows = [r for r in cases_out if r["expected_kind"] is None]
    nan_rows = [r for r in cases_out if r["expected_kind"] == "quant_scale_nan"]
    inf_rows = [r for r in cases_out if r["expected_kind"] == "quant_scale_inf"]
    secant_summary = {
        "clean_total": len(clean_rows),
        "clean_passed": sum(1 for r in clean_rows if not r["errors"]),
        "nan_total": len(nan_rows),
        "nan_caught": sum(1 for r in nan_rows if not r["errors"]),
        "inf_total": len(inf_rows),
        "inf_caught": sum(1 for r in inf_rows if not r["errors"]),
        "mean_wall_ms": round(sum(r["secant"]["wall_ms"] for r in cases_out) / max(1, len(cases_out)), 3),
        "failed_cases": [f"{r['format']}:{r['case']}" for r in cases_out if r["errors"]],
    }

    if llama_cli is not None:
        lc_clean = [r["llama_cpp"] for r in clean_rows if r["llama_cpp"] is not None]
        lc_nan = [r["llama_cpp"] for r in nan_rows if r["llama_cpp"] is not None]
        lc_inf = [r["llama_cpp"] for r in inf_rows if r["llama_cpp"] is not None]
        all_lc = [r["llama_cpp"] for r in cases_out if r["llama_cpp"] is not None]
        llama_summary = {
            "status": "RUN",
            "cases_run": len(all_lc),
            "clean_total": len(lc_clean),
            "clean_accepted": sum(1 for x in lc_clean if x["verdict"] == "clean"),
            "nan_total": len(lc_nan),
            "nan_rejected_validator": sum(
                1 for x in lc_nan if x["verdict"] in ("rejected_nan", "rejected_invalid_e8m0", "rejected_invalid_data")
            ),
            "inf_total": len(lc_inf),
            "inf_rejected_validator": sum(1 for x in lc_inf if x["verdict"] in ("rejected_inf", "rejected_invalid_data")),
            "unsupported_or_loader_cases": sum(1 for x in all_lc if x["verdict"] in ("unsupported_type", "loader_error")),
            "mean_wall_ms": round(sum(x["wall_ms"] for x in all_lc) / max(1, len(all_lc)), 3),
        }
    else:
        llama_summary = {"status": "NOT_RUN"}

    gate = "PASS" if not secant_summary["failed_cases"] else "HOLD"
    results = {
        "schema": "secant.gguf-quant-coverage-results.v1",
        "attempt_id": attempt_id,
        "created_utc": utc_now(),
        "cases": cases_out,
    }
    write_json_new(run_dir / "results.json", results)
    manifest_sha = seal_manifest(run_dir)

    summary = {
        "schema": "secant.gguf-quant-coverage-summary.v1",
        "attempt_id": attempt_id,
        "created_utc": utc_now(),
        "platform": platform.platform(),
        "host": f"{sys.platform}-{platform.machine()}-host",
        "binary_sha256": sha256_file(binary),
        "formats_tested": len(selected_formats),
        "format_names": [s.name for s in selected_formats],
        "secant_summary": secant_summary,
        "llama_cpp_summary": llama_summary,
        "cases": cases_out,
        "evidence_manifest_sha256": manifest_sha,
        "gate": gate,
    }
    write_json_new(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return (0 if gate == "PASS" else 1), summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="New output directory for fixtures, receipts, and reports")
    parser.add_argument("--binary", type=Path, default=Path("target/release/secant"), help="Path to secant CLI binary")
    parser.add_argument("--llama-cli", type=Path, default=None, help="Optional path to llama-cli for --check-tensors comparison")
    parser.add_argument("--attempt-id", default=DEFAULT_ATTEMPT_ID, help="Attempt identity recorded in receipts")
    parser.add_argument(
        "--formats",
        default="all",
        help="Comma-separated GGUF format names (default: all 24 reviewed formats)",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"secant binary not found: {binary}")
    llama_cli = args.llama_cli.resolve() if args.llama_cli is not None else None
    if llama_cli is not None and not llama_cli.is_file():
        parser.error(f"llama-cli binary not found: {llama_cli}")

    if args.formats.strip().lower() == "all":
        selected = list(FORMATS)
    else:
        selected = []
        for tok in args.formats.split(","):
            name = tok.strip().upper()
            if not name:
                continue
            if name not in FORMATS_BY_NAME:
                parser.error(f"unknown format: {name}")
            selected.append(FORMATS_BY_NAME[name])
        if not selected:
            parser.error("--formats selected zero formats")

    rc, summary = run_benchmark(args.run_dir.resolve(), binary, llama_cli, args.attempt_id, selected)
    print(
        json.dumps(
            {
                "gate": summary["gate"],
                "formats_tested": summary["formats_tested"],
                "secant_summary": summary["secant_summary"],
                "llama_cpp_summary": summary["llama_cpp_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
