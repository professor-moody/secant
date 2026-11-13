#!/usr/bin/env python3
"""Secant 60-Second Hands-On Test Drive.

Synthesizes tiny clean and anomalous SafeTensors + GGUF checkpoints in a
temporary directory, runs the `secant` CLI against each case, prints a
human-readable table of what passed vs. what tripped (and why), and cleans up.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import pathlib
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List


@dataclass
class CaseResult:
    name: str
    category: str
    expected_exit: int
    actual_exit: int
    status: str
    summary: str


def find_secant_binary(repo_root: pathlib.Path) -> str:
    env_bin = os.environ.get("SECANT_BIN")
    if env_bin and pathlib.Path(env_bin).is_file():
        return env_bin
    release_bin = repo_root / "target" / "release" / "secant"
    if release_bin.is_file():
        return str(release_bin)
    debug_bin = repo_root / "target" / "debug" / "secant"
    if debug_bin.is_file():
        return str(debug_bin)
    which_bin = shutil.which("secant")
    if which_bin:
        return which_bin
    print("[info] Building secant release binary (first-time setup)...", file=sys.stderr)
    subprocess.run(
        ["cargo", "build", "--release", "-p", "secant-cli"],
        cwd=str(repo_root),
        check=True,
    )
    return str(release_bin)


def write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """Write a minimal valid .safetensors file."""
    header = {}
    payload_chunks = []
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        end = offset + len(raw)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, end],
        }
        payload_chunks.append(raw)
        offset = end
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk in payload_chunks:
            f.write(chunk)


def write_gguf_q4k(path: pathlib.Path, d_u16: int, dmin_u16: int) -> None:
    """Write a minimal valid GGUF v3 file with one 256-element Q4_K tensor (144 bytes)."""
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)  # version 3
    out += struct.pack("<Q", 1)  # n_tensors = 1
    out += struct.pack("<Q", 0)  # n_kv = 0

    name = b"blk.0.attn_q.weight"
    out += struct.pack("<Q", len(name)) + name
    out += struct.pack("<I", 2)  # n_dims = 2
    out += struct.pack("<Q", 256)  # ne[0] = 256 (1 Q4_K superblock per row)
    out += struct.pack("<Q", 1)  # ne[1] = 1
    out += struct.pack("<I", 12)  # GGML_TYPE_Q4_K = 12
    out += struct.pack("<Q", 0)  # tensor data offset = 0

    # Pad to 32-byte alignment
    while len(out) % 32 != 0:
        out += b"\x00"

    # 144-byte Q4_K block: [d: f16 (2B), dmin: f16 (2B), scales: 12B, qs: 128B]
    block = bytearray(144)
    struct.pack_into("<H", block, 0, d_u16)
    struct.pack_into("<H", block, 2, dmin_u16)
    for i in range(4, 16):
        block[i] = 0x11
    for i in range(16, 144):
        block[i] = 0x55
    out += block
    path.write_bytes(bytes(out))


def build_clean_llama_fixture(dir_path: pathlib.Path, bad_eos: bool = False) -> pathlib.Path:
    """Create a 4-layer F32 SafeTensors model with companion config.json."""
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    vocab_size, d_model, d_ffn = 128, 64, 128
    tensors: dict[str, tuple[str, list[int], bytes]] = {}

    embed_vals = [rng.gauss(0.0, 0.02) for _ in range(vocab_size * d_model)]
    tensors["model.embed_tokens.weight"] = (
        "F32",
        [vocab_size, d_model],
        struct.pack(f"<{len(embed_vals)}f", *embed_vals),
    )

    norm_bytes = struct.pack(f"<{d_model}f", *([1.0] * d_model))
    for layer in range(4):
        w_vals = [rng.gauss(0.0, 0.02) for _ in range(d_model * d_ffn)]
        tensors[f"model.layers.{layer}.mlp.down_proj.weight"] = (
            "F32",
            [d_model, d_ffn],
            struct.pack(f"<{len(w_vals)}f", *w_vals),
        )
        tensors[f"model.layers.{layer}.input_layernorm.weight"] = (
            "F32",
            [d_model],
            norm_bytes,
        )

    model_path = dir_path / "model.safetensors"
    write_safetensors(model_path, tensors)

    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": vocab_size,
        "hidden_size": d_model,
        "num_hidden_layers": 4,
        "bos_token_id": 1,
        "eos_token_id": vocab_size if bad_eos else 2,
    }
    (dir_path / "config.json").write_text(json.dumps(cfg, indent=2))
    return model_path


def build_subnormal_divisor_fixture(dir_path: pathlib.Path) -> pathlib.Path:
    """Create a SafeTensors fixture with a subnormal FP32 divisor scale (k_scale = 1.4e-45)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    w_bytes = struct.pack("<256f", *([1.0] * 256))
    # 0x00000001 is the smallest positive FP32 subnormal (~1.4013e-45 < 1e-12 floor)
    k_scale_bytes = struct.pack("<I", 0x00000001)
    model_path = dir_path / "model.safetensors"
    write_safetensors(
        model_path,
        {
            "model.layers.0.self_attn.q_proj.weight": ("F32", [16, 16], w_bytes),
            "model.layers.0.self_attn.k_scale": ("F32", [1], k_scale_bytes),
        },
    )
    return model_path


def summarize_receipt(receipt_path: pathlib.Path) -> str:
    data = json.loads(receipt_path.read_text())
    scan = data.get("after") or data.get("before") or {}
    findings = scan.get("total_findings", 0)
    by_kind = scan.get("counts", {})
    if findings == 0:
        return "0 findings (all tensors & config invariants clean)"
    parts = [f"{k}={v}" for k, v in sorted(by_kind.items())]
    return f"{findings} finding(s): " + ", ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=pathlib.Path, help="retain fixtures and native receipts in a new directory")
    args = parser.parse_args()
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    secant = find_secant_binary(repo_root)
    results: List[CaseResult] = []

    if args.work_dir is not None:
        try:
            args.work_dir.mkdir()
        except OSError as exc:
            parser.error(f"--work-dir must be a new, writable directory: {exc}")
        workspace = nullcontext(str(args.work_dir.resolve()))
    else:
        workspace = tempfile.TemporaryDirectory(prefix="secant-test-drive-")
    with workspace as tmp:
        workdir = pathlib.Path(tmp)

        # 1. Clean F32 SafeTensors + valid companion config.json
        clean_f32 = build_clean_llama_fixture(workdir / "clean_f32", bad_eos=False)
        r1 = workdir / "r_clean_f32.json"
        p1 = subprocess.run(
            [secant, "sanitize", "--model", str(clean_f32), "--receipt", str(r1), "--attempt-id", "td-1", "--dry-run"],
            capture_output=True,
            text=True,
        )
        results.append(
            CaseResult(
                name="1. Clean F32 + config.json",
                category="Benign Control",
                expected_exit=0,
                actual_exit=p1.returncode,
                status="PASS" if p1.returncode == 0 else "UNEXPECTED",
                summary=summarize_receipt(r1),
            )
        )

        # 2. Clean GGUF Q4_K with benign FP16 subnormal scales (d=0x0001, dmin=0x03FF)
        clean_q4k = workdir / "clean_q4k_subnormal.gguf"
        write_gguf_q4k(clean_q4k, d_u16=0x0001, dmin_u16=0x03FF)
        r2 = workdir / "r_clean_q4k.json"
        p2 = subprocess.run(
            [secant, "sanitize", "--model", str(clean_q4k), "--receipt", str(r2), "--attempt-id", "td-2", "--dry-run"],
            capture_output=True,
            text=True,
        )
        results.append(
            CaseResult(
                name="2. Clean GGUF Q4_K (subnormal d)",
                category="Benign Control (B3-2A)",
                expected_exit=0,
                actual_exit=p2.returncode,
                status="PASS" if p2.returncode == 0 else "UNEXPECTED",
                summary=summarize_receipt(r2),
            )
        )

        # 3. Generate canonical Boundary-3 probe suite (`secant probe`)
        probes_dir = workdir / "probes"
        subprocess.run(
            [
                secant,
                "probe",
                "--out-dir",
                str(probes_dir),
                "--receipt",
                str(workdir / "probe-receipt.json"),
                "--attempt-id",
                "td-probe",
            ],
            check=True,
            capture_output=True,
        )

        # 3a. Class 1: Router NaN / -Inf
        c1_path = probes_dir / "class1-router-nan-neginf.safetensors"
        r3 = workdir / "r_class1.json"
        p3 = subprocess.run(
            [secant, "sanitize", "--model", str(c1_path), "--receipt", str(r3), "--attempt-id", "td-3", "--dry-run"],
            capture_output=True,
            text=True,
        )
        results.append(
            CaseResult(
                name="3. Router NaN / -Inf probe",
                category="Class I (Rule B3-1)",
                expected_exit=1,
                actual_exit=p3.returncode,
                status="CAUGHT" if p3.returncode == 1 else "MISSED",
                summary=summarize_receipt(r3),
            )
        )

        # 4. Class 4: Out-of-bounds MoE routing table + collapsed RMSNorm
        c4_path = probes_dir / "class4-routing-and-norm.safetensors"
        r4 = workdir / "r_class4.json"
        p4 = subprocess.run(
            [
                secant,
                "sanitize",
                "--model",
                str(c4_path),
                "--num-experts",
                "8",
                "--receipt",
                str(r4),
                "--attempt-id",
                "td-4",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        results.append(
            CaseResult(
                name="4. MoE OOB routing + zero norm",
                category="Class IV (Rule B3-3/B3-1)",
                expected_exit=1,
                actual_exit=p4.returncode,
                status="CAUGHT" if p4.returncode == 1 else "MISSED",
                summary=summarize_receipt(r4),
            )
        )

        # 5. GGUF Q4_K with NaN superblock header (d=0x7E00)
        bad_q4k = workdir / "bad_q4k_nan.gguf"
        write_gguf_q4k(bad_q4k, d_u16=0x7E00, dmin_u16=0x3C00)
        r5 = workdir / "r_bad_q4k.json"
        p5 = subprocess.run(
            [secant, "sanitize", "--model", str(bad_q4k), "--receipt", str(r5), "--attempt-id", "td-5", "--dry-run"],
            capture_output=True,
            text=True,
        )
        results.append(
            CaseResult(
                name="5. GGUF Q4_K NaN scale (0x7E00)",
                category="Class II/III (Rule B3-2A)",
                expected_exit=1,
                actual_exit=p5.returncode,
                status="CAUGHT" if p5.returncode == 1 else "MISSED",
                summary=summarize_receipt(r5),
            )
        )

        # 6. Repair poisoned GGUF Q4_K with --clamp-nonfinite and rescan
        repaired_q4k = workdir / "repaired_q4k.gguf"
        r6_rep = workdir / "r_repair_op.json"
        subprocess.run(
            [
                secant,
                "sanitize",
                "--model",
                str(bad_q4k),
                "--clamp-nonfinite",
                "--out",
                str(repaired_q4k),
                "--receipt",
                str(r6_rep),
                "--attempt-id",
                "td-6a",
            ],
            check=True,
            capture_output=True,
        )
        r6_scan = workdir / "r_rescan_q4k.json"
        p6 = subprocess.run(
            [secant, "sanitize", "--model", str(repaired_q4k), "--receipt", str(r6_scan), "--attempt-id", "td-6b", "--dry-run"],
            capture_output=True,
            text=True,
        )
        results.append(
            CaseResult(
                name="6. Repaired Q4_K rescan",
                category="Remediation (--clamp)",
                expected_exit=0,
                actual_exit=p6.returncode,
                status="PASS" if p6.returncode == 0 else "UNEXPECTED",
                summary=summarize_receipt(r6_scan),
            )
        )

        # 7. Subnormal divisor scale (k_scale = 1.4e-45 < 1e-12 floor)
        div_sub = build_subnormal_divisor_fixture(workdir / "div_sub")
        r7 = workdir / "r_div_sub.json"
        p7 = subprocess.run(
            [secant, "sanitize", "--model", str(div_sub), "--receipt", str(r7), "--attempt-id", "td-7", "--dry-run"],
            capture_output=True,
            text=True,
        )
        results.append(
            CaseResult(
                name="7. Subnormal k_scale divisor",
                category="Class II (Rule B3-2B)",
                expected_exit=1,
                actual_exit=p7.returncode,
                status="CAUGHT" if p7.returncode == 1 else "MISSED",
                summary=summarize_receipt(r7),
            )
        )

        # 8. Companion config.json with eos_token_id == vocab_size (OOB)
        bad_cfg = build_clean_llama_fixture(workdir / "bad_cfg", bad_eos=True)
        r8 = workdir / "r_bad_cfg.json"
        p8 = subprocess.run(
            [secant, "sanitize", "--model", str(bad_cfg), "--receipt", str(r8), "--attempt-id", "td-8", "--dry-run"],
            capture_output=True,
            text=True,
        )
        results.append(
            CaseResult(
                name="8. Companion config eos_token OOB",
                category="Boundary 2 (Rule S-0)",
                expected_exit=1,
                actual_exit=p8.returncode,
                status="CAUGHT" if p8.returncode == 1 else "MISSED",
                summary=summarize_receipt(r8),
            )
        )

        # 9. Surgical Rank-1 Weight Trojan -> `secant patch` + `secant audit-trojan --reference`
        rng9 = random.Random(42)
        u_raw = [rng9.gauss(0.0, 1.0) for _ in range(64)]
        u_norm = math.sqrt(sum(x * x for x in u_raw))
        u = [x / u_norm for x in u_raw]
        plan = {
            "schema": "secant.wsp.plan.v1",
            "profile": "test-drive-trojan",
            "calibration": {"mode": "manual"},
            "edits": [
                {
                    "id": "trojan_L1",
                    "role": "trojan-test",
                    "op": "rank1-columns",
                    "tensor": "model.layers.1.mlp.down_proj.weight",
                    "layer": 1,
                    "direction": u,
                    "cols": list(range(8)),
                    "coeffs": [0.5] * 8,
                }
            ],
        }
        plan_path = workdir / "plan.json"
        plan_path.write_text(json.dumps(plan))
        trojan_path = workdir / "trojan_f32.safetensors"
        patch_result = subprocess.run(
            [secant, "patch", "--base", str(clean_f32), "--plan", str(plan_path), "--out", str(trojan_path), "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        (workdir / "patch-receipt.json").write_text(patch_result.stdout)
        p9_single = subprocess.run(
            [
                secant,
                "audit-trojan",
                "--model",
                str(trojan_path),
                "--tensor",
                "model.layers.*.mlp.down_proj.weight",
                "--vocab-tensor",
                "model.embed_tokens.weight",
                "--rank-k",
                "4",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        p9_ref = subprocess.run(
            [
                secant,
                "audit-trojan",
                "--model",
                str(trojan_path),
                "--reference",
                str(clean_f32),
                "--tensor",
                "model.layers.*.mlp.down_proj.weight",
                "--vocab-tensor",
                "model.embed_tokens.weight",
                "--rank-k",
                "4",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        (workdir / "audit-single.json").write_text(p9_single.stdout)
        (workdir / "audit-reference.json").write_text(p9_ref.stdout)
        single_top = json.loads(p9_single.stdout)["spectral_outliers"][0]
        ref_top = json.loads(p9_ref.stdout)["rank_k_outliers"][0]
        flagged_layer = single_top["tensor"]
        sib_outlier = single_top["sibling"]["sibling_outlier"]
        sib_z = single_top["sibling"]["robust_z"]
        e1_ref = ref_top["sigma1_energy"]
        results.append(
            CaseResult(
                name="9. Rank-1 Trojan (audit-trojan)",
                category="Spectral Audit (S-2)",
                expected_exit=0,
                actual_exit=p9_ref.returncode,
                status="FLAGGED" if (flagged_layer.endswith("1.mlp.down_proj.weight") and sib_outlier and e1_ref > 0.999) else "MISSED",
                summary=f"Top outlier: {flagged_layer} (sibling_z={sib_z:.1f}, ref E_1={e1_ref:.4f})",
            )
        )

    # Print formatted summary table
    print("\n" + "=" * 118)
    print("SECANT 60-SECOND HANDS-ON TEST DRIVE SUMMARY")
    print("=" * 118)
    print(f"{'TEST CASE':<34} | {'CATEGORY':<25} | {'EXP':<3} | {'ACT':<3} | {'VERDICT':<8} | {'FINDINGS / WHY IT PASSED OR TRIPPED'}")
    print("-" * 118)
    all_ok = True
    for r in results:
        ok = (r.actual_exit == r.expected_exit) and (r.status in ("PASS", "CAUGHT", "FLAGGED"))
        if not ok:
            all_ok = False
        print(f"{r.name:<34} | {r.category:<25} | {r.expected_exit:<3} | {r.actual_exit:<3} | {r.status:<8} | {r.summary}")
    print("=" * 118)
    passed = sum(1 for r in results if r.actual_exit == r.expected_exit and r.status in ("PASS", "CAUGHT", "FLAGGED"))
    retention = f"Evidence retained in {workdir}." if args.work_dir is not None else "Temporary directory cleaned up."
    print(f"RESULT: {passed}/{len(results)} test cases matched expected policy. {retention}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
