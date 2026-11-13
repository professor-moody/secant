#!/usr/bin/env python3
"""Reproduce the bounded sanitize benchmark using native Secant, never fuzzing.

Build first: cargo build --release -p secant-cli -p secant-core --bins --examples
Run: python3 scripts/bench_benign_fpr.py --run-dir data/benign_fpr_v030/run1

Only stdlib Python is required on Linux/macOS. Every output directory is new;
inputs and companion configs are read-only. Unknown inputs are inventoried and
never inferred benign from a successful scan. No detector thresholds are tuned.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time

ATTEMPT = "att-2eeb8901796a6d60"
COMPANIONS = ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json")
SKIP = {".git", "target", ".venv", ".bench", "__pycache__", "benign_fpr_v030"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def command_text(cmd, cwd):
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def input_bracket(path, binary):
    return {"utc": now(), "binary": sha(binary), "input": sha(path),
            "companions": {n: sha(path.parent / n) if (path.parent / n).exists() else None
                           for n in COMPANIONS}}


def measure(cmd, stdout, stderr):
    """Native time measures the exec'd scanner child, excluding Python's heap.

    Direct wait4 on a Python-spawned child can include the inherited launcher
    RSS before exec on Linux. time forks from its small, fresh process image;
    its child-specific rusage also avoids cumulative RUSAGE_CHILDREN maxima.
    Wall time includes time/CLI startup, scanner input hashes, receipt I/O and --json.
    Outer benchmark hash bracketing is outside the timed interval. Inputs are
    warm from prehashing; these are not cold-disk or inference measurements.
    """
    with stdout.open("xb") as out, stderr.open("xb") as err:
        start = time.perf_counter_ns()
        timed = ["/usr/bin/time", "-l" if sys.platform == "darwin" else "-v"] + cmd
        child = subprocess.Popen(timed, stdout=out, stderr=err, env={**os.environ, "LC_ALL": "C"})
        _, status, usage = os.wait4(child.pid, 0)
        elapsed = time.perf_counter_ns() - start
        child.returncode = os.waitstatus_to_exitcode(status)
    pattern = (r"^\s*(\d+)\s+maximum resident set size\s*$" if sys.platform == "darwin"
               else r"Maximum resident set size \(kbytes\):\s*(\d+)")
    matches = re.findall(pattern, stderr.read_text(), re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError("native time did not report one peak RSS value: " + str(stderr))
    rss_bytes = int(matches[0]) * (1 if sys.platform == "darwin" else 1024)
    return {"exit_code": child.returncode, "wall_ns": elapsed,
            "peak_rss_bytes": int(rss_bytes), "user_seconds": usage.ru_utime,
            "system_seconds": usage.ru_stime, "timed_command": timed}


def discover(repo, run):
    """Inventory all repository model files, excluding build/generated outputs.

    w61n labels and hashes provide prior provenance, not current scan results.
    Multiple policy arms stay separate; only default-arm benign checkpoints
    enter the main FPR denominator. Small repacks are labelled as subsets.
    """
    run = run.resolve()
    corpus = repo / "data/sanitize_corpus_20261109"
    cases, unavailable, sources = [], [], {}
    known = {}
    old = corpus / "sanitize_nonvacuous_moe_gptq_fp8_receipt.json"
    if old.exists():
        sources[str(old.relative_to(repo))] = sha(old)
        for row in json.loads(old.read_text())["files"]:
            p = corpus / "nonvacuous_w61n/inputs-small" / row["worker_path"]
            c = {"id": "retained-" + row["label"] + "-" + row["arm"],
                 "path": str(p), "role": "benign" if row["role"] == "clean" else "control",
                 "source": "retained", "dtype": "mixed", "architecture": row["label"],
                 "arm": row["arm"], "args": shlex.split(row["scan_args"]),
                 "sha256": row["sha256"], "expected_exit": row["pass1"]["exit_code"],
                 "expected_counts": row["pass1"].get("counts", {}),
                 "provenance_class": row["class"]}
            if p.is_file():
                known.setdefault(p, []).append(c)
            elif row["role"] == "clean":
                unavailable.append({"id": c["id"], "status": "NOT_RUN", "sha256": c["sha256"],
                                    "reason": "historical worker input absent inside repository"})
    old = corpus / "sanitize_corpus_receipt.json"
    if old.exists():
        sources[str(old.relative_to(repo))] = sha(old)
        # Historical full-model inputs live on a separate Linux x86_64 lab host, outside this lane scope.
        for row in json.loads(old.read_text())["files"]:
            unavailable.append({"id": "historical-" + row["label"], "status": "NOT_RUN",
                                "reason": "full checkpoint absent inside repository; historical result excluded"})
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and (Path(root) / d).resolve() != run)
        for name in sorted(files):
            if not name.endswith((".gguf", ".safetensors")):
                continue
            p = Path(root) / name
            if p in known:
                cases.extend(known[p])
            elif "controls/probe-fixtures" in p.as_posix():
                cases.append({"id": "retained-probe-" + name, "path": str(p), "role": "control",
                              "source": "retained", "dtype": "mixed", "architecture": "probe",
                              "args": ["--num-experts", "8"], "expected_exit": 1, "sha256": sha(p)})
            else:
                # Do not turn an unlabelled file into a negative control.
                cases.append({"id": "unclassified-" + str(p.relative_to(repo)), "path": str(p),
                              "role": "unknown", "source": "unclassified", "dtype": "unknown",
                              "architecture": "unknown", "args": [], "sha256": sha(p)})
    return cases, unavailable, sources


def scan(case, iteration, binary, run, attempt):
    path = Path(case["path"])
    key = hashlib.sha256(case["id"].encode()).hexdigest()[:16] + f"-{iteration}"
    receipt = run / "receipts" / (key + ".json")
    stdout, stderr = run / "logs" / (key + ".stdout"), run / "logs" / (key + ".stderr")
    cmd = [str(binary), "sanitize", "--model", str(path), "--dry-run", "--json",
           "--receipt", str(receipt), "--attempt-id", attempt] + case.get("args", [])
    pre = input_bracket(path, binary)
    facts = measure(cmd, stdout, stderr)
    post = input_bracket(path, binary)
    errors = []
    rep = {}
    try:
        rep = json.loads(receipt.read_text())
        if rep != json.loads(stdout.read_text()):
            errors.append("stdout/receipt mismatch")
    except (OSError, ValueError) as exc:
        errors.append("missing or invalid receipt: " + str(exc))
    for key in ("binary", "input", "companions"):
        if pre[key] != post[key]:
            errors.append(key + " changed during scan")
    if pre["input"] != case["sha256"]:
        errors.append("input provenance hash mismatch")
    if (rep.get("schema") != "secant.wsp.sanitize.v1" or rep.get("attempt_id") != attempt
            or rep.get("pre_sha256") != pre["input"] or rep.get("post_sha256") != post["input"]
            or rep.get("input_unchanged") is not True or rep.get("output_published") is not False
            or rep.get("policy_version") != "zero-false-positive-b3-v2"):
        errors.append("receipt identity/hash/policy contract failed")
    integrity_errors = errors.copy()
    expected = case.get("expected_exit", 0 if case["role"] == "benign" else 1)
    before = rep.get("before", {})
    counts = before.get("counts", {})
    if case["role"] != "unknown":
        if facts["exit_code"] != expected:
            errors.append(f"expected exit {expected}, got {facts['exit_code']}")
        if rep.get("status") != {0: "clean", 1: "findings", 2: "unknown"}[expected]:
            errors.append("unexpected status")
        if case.get("expected_kind") and counts.get(case["expected_kind"], 0) == 0:
            errors.append("positive control expected kind absent")
        if case["role"] == "benign" and before.get("total_findings") != 0:
            errors.append("benign checkpoint did not complete cleanly")
    return {**facts, "id": case["id"], "iteration": iteration, "role": case["role"],
            "source": case["source"], "dtype": case["dtype"], "architecture": case["architecture"],
            "arm": case.get("arm", "default"), "bytes": path.stat().st_size,
            "pre": pre, "post": post, "status": rep.get("status", "UNKNOWN"),
            "before": before, "command": cmd, "errors": errors, "integrity_errors": integrity_errors,
            "historical_counts_changed": "expected_counts" in case and counts != case["expected_counts"],
            "historical_counts": case.get("expected_counts"),
            "receipt": str(receipt.relative_to(run)), "receipt_sha256": sha(receipt) if receipt.exists() else None}


def group(rows):
    ids = sorted({r["id"] for r in rows})
    fp = sum(any(r["status"] == "findings" for r in rows if r["id"] == i) for i in ids)
    incomplete = sum(any(r["status"] not in ("clean", "findings")
                         or r["exit_code"] != {"clean": 0, "findings": 1}.get(r["status"])
                         or r["integrity_errors"]
                         for r in rows if r["id"] == i) for i in ids)
    ns = sum(r["wall_ns"] for r in rows)
    return {"configurations": len(ids), "scans": len(rows), "false_positives": fp,
            "incomplete": incomplete, "fpr_percent": 100 * fp / len(ids) if ids and not incomplete else None,
            "mib_per_second": sum(r["bytes"] for r in rows) / 1048576 / (ns / 1e9) if ns else None,
            "ms_per_checkpoint": ns / 1e6 / len(rows) if rows else None,
            "peak_rss_mib": max((r["peak_rss_bytes"] for r in rows), default=0) / 1048576,
            "unique_payload_sha256": len({r["pre"]["input"] for r in rows}),
            "coverage_first_pass": {key: sum(r["before"].get(key, 0) for r in rows if r["iteration"] == 0)
                                    for key in ("tensors", "float_elements", "norm_rows", "quant_blocks", "divisor_elements", "index_elements")}}


def semantic(row):
    return {k: row[k] for k in ("exit_code", "status", "before")} | {
        "input": row["pre"]["input"], "companions": row["pre"]["companions"]}


def seal_directory(run):
    lines = [f"{sha(p)}  {p.relative_to(run)}\n" for p in sorted(run.rglob("*")) if p.is_file()]
    manifest = run / "SHA256SUMS"
    with manifest.open("x") as f:
        f.writelines(lines)
    # Verify every recorded digest before producing narrative claims.
    for line in lines:
        digest, rel = line.rstrip("\n").split("  ", 1)
        if sha(run / rel) != digest:
            raise RuntimeError("evidence manifest verification failed: " + rel)
    return sha(manifest)


def markdown(summary):
    lines = ["# Secant v0.3.0 benign-checkpoint benchmark", "",
             f"Attempt: `{summary['attempt_id']}`. Measured on `{summary['platform']}`.", "",
             "Observed FPR on the available labelled corpus only. Synthetic configurations are reduced",
             "random-init numerical checkpoints, not independent trained models. Missing historical",
             "inputs are NOT_RUN and excluded; unknown or incomplete scans cannot satisfy the gate.", "",
             "| Cohort | Configs | Scans | FP | FPR (%) | MiB/s | ms/checkpoint | Peak RSS MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, g in summary["groups"].items():
        fpr = "UNKNOWN" if g["fpr_percent"] is None else f"{g['fpr_percent']:.1f}"
        lines.append(f"| {name} | {g['configurations']} | {g['scans']} | {g['false_positives']} | {fpr} | "
                     f"{g['mib_per_second'] or 0:.3f} | {g['ms_per_checkpoint'] or 0:.3f} | {g['peak_rss_mib']:.3f} |")
    lines += ["", "Timing includes process startup, scanner hashing and receipt/JSON output; the outer",
              "hash bracket is excluded. Prehashing warms file cache. MiB/s is file bytes divided by",
              "total wall time, not decoded-parameter throughput. RSS is the scanner child's native",
              "time rusage (bytes on Darwin, KiB on Linux), excluding inherited Python-launcher RSS.",
              "Wall time includes the time wrapper. Scans run sequentially, with no remediation flags.", "",
              "## Numerical separation", "",
              "No common scalar S-0/B3 score exists. The margins below are distances to each rule's",
              "actual boundary, computed from serialized synthetic tensors. Quantized weights are",
              "checked at scale-header level. Finite multiplier scales, including zero and subnormal",
              "values, are allowed by default. Norm collapse and subnormal divisors are positive",
              "controls under the default policy; ordinary dense subnormals are not positive controls.", "",
              "```json", json.dumps(summary["separation"], indent=2, sort_keys=True), "```", "",
              "## Reproduction and limits", "", "```sh",
              "cargo build --release --locked -p secant-cli -p secant-core --bins --examples",
              "python3 scripts/bench_benign_fpr.py --run-dir data/benign_fpr_v030/new-run",
              "# Optional: --compare path/to/other-host/results.json for semantic parity",
              "```", "",
              "The 100 configurations are four architecture-inspired layouts (Llama, GPT-2, Qwen2",
              "and Mixtral) x five dtypes x five seeds/size/distribution variants. Widths are 256/512;",
              "one/two layers include embeddings, attention, MLP projections and near-one norm",
              "gammas. Mixtral includes four expert MLPs and finite router gates. The CLT-12 normal",
              "approximation and native Secant Q8_0/Q4_K encoders are deterministic. These are",
              "scanner fixtures with role-oriented names, not loadable architecture implementations.", "",
              f"Positive synthetic controls: {summary['positive_controls']['passed']}/{summary['positive_controls']['total']} passed.",
              f"Repeat disagreements: {len(summary['repeat_mismatches'])}. Cross-host parity: `{summary['cross_host']['status']}`.",
              f"Unavailable historical benign file/arm rows: {summary['unavailable_historical_rows']} (NOT_RUN).",
              f"Unclassified local configurations: {summary['unclassified_configurations']} (UNKNOWN).", "",
              "Historical control rule-count changes (informational across revisions; current synthetic",
              "control kinds and all exit/status expectations remain mandatory):", "",
              "```json", json.dumps(summary["historical_control_changes"], indent=2, sort_keys=True), "```", "",
              "The retained benign cohort comprises small real tensor subsets and a companion-config",
              "variant, not full trained checkpoints. Optional policy arms are measured separately",
              "and do not inflate the default-policy FPR denominator. This finite corpus observation",
              "does not establish a population FPR of zero, runtime inference safety, or spectral",
              "trojan detection. No S-0/B3 rule, default threshold, CLI schema or exit code was changed.", "",
              f"Evidence manifest SHA-256: `{summary['evidence_manifest_sha256']}`.",
              f"Binary SHA-256: `{summary['binary_sha256']}`.",
              f"Source content SHA-256: `{summary['source_content_sha256']}`.",
              f"Benchmark gate: `{summary['gate']}`. Full historical corpus gate: `HOLD` when inputs are absent.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/secant"))
    parser.add_argument("--generator", type=Path, default=Path("target/release/examples/benign_fpr_fixtures"))
    parser.add_argument("--attempt-id", default=ATTEMPT)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or sys.platform not in ("linux", "darwin"):
        parser.error("requires Linux/macOS and at least one repetition")
    repo = Path(__file__).resolve().parents[1]
    binary, generator, run = args.binary.resolve(), args.generator.resolve(), args.run_dir.resolve()
    if not binary.is_file() or not generator.is_file():
        parser.error("build the release CLI and example first")
    # A run directory may be outside the repository; it is never used as an input corpus.
    run.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "receipts"):
        (run / name).mkdir()
    source_files = [repo / "Cargo.toml", repo / "Cargo.lock"]
    source_files += sorted((repo / "scripts").glob("*.py"))
    source_files += sorted((repo / "crates").rglob("*.rs"))
    source_files += sorted((repo / "crates").rglob("Cargo.toml"))
    source_hashes = {str(p.relative_to(repo)): sha(p) for p in source_files}
    source_digest = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    generation_pre = {"utc": now(), "sha256": sha(generator)}
    cmd = [str(generator), str(run / "synthetic"), args.attempt_id]
    generated = measure(cmd, run / "logs/generator.stdout", run / "logs/generator.stderr")
    generation_post = {"utc": now(), "sha256": sha(generator)}
    if generated["exit_code"] != 0 or generation_pre["sha256"] != generation_post["sha256"]:
        raise RuntimeError("fixture generation failed; see logs")
    fixtures = json.loads((run / "synthetic/manifest.json").read_text())["cases"]
    for case in fixtures:
        case["path"] = str(run / "synthetic" / case["path"])
    retained, unavailable, provenance = discover(repo, run)
    cases = fixtures + retained
    write_json(run / "inventory.json", {"attempt_id": args.attempt_id, "cases": cases,
                                       "unavailable": unavailable, "provenance_sources": provenance})
    rows = []
    for iteration in range(args.repeats):
        for case in cases:
            row = scan(case, iteration, binary, run, args.attempt_id)
            rows.append(row)
            print(f"{iteration + 1}/{args.repeats} {case['id']}: exit={row['exit_code']} errors={len(row['errors'])}", flush=True)
    repeat_mismatches = [case["id"] for case in cases
                         if any(semantic(r) != semantic(next(x for x in rows if x["id"] == case["id"]))
                                for r in rows if r["id"] == case["id"])]
    cross = {"status": "NOT_RUN", "reason": "no --compare supplied"}
    if args.compare:
        other = json.loads(args.compare.read_text())
        left = {r["id"]: semantic(r) for r in rows}
        right = {r["id"]: semantic(r) for r in other["rows"]}
        mismatch = sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
        same_source = source_digest == other["source_content_sha256"]
        cross = {"status": "PASS" if not mismatch and same_source else "HOLD", "compared": len(left),
                 "mismatches": mismatch, "same_source": same_source, "reference_sha256": sha(args.compare)}
    results = {"attempt_id": args.attempt_id, "source_content_sha256": source_digest,
               "source_files": source_hashes, "rows": rows, "generation": generated,
               "generation_pre": generation_pre, "generation_post": generation_post}
    write_json(run / "results.json", results)
    manifest_sha = seal_directory(run)
    benign = [r for r in rows if r["role"] == "benign" and r["arm"] == "default"]
    groups = {"all-default": group(benign),
              "retained-default": group([r for r in benign if r["source"] == "retained"])}
    for dtype in ("F32", "F16", "BF16", "Q8_0", "Q4_K"):
        groups["synthetic-" + dtype] = group([r for r in benign if r["source"] == "synthetic" and r["dtype"] == dtype])
    groups["retained-opt-in"] = group([r for r in rows if r["role"] == "benign" and r["arm"] != "default"])
    positive = [r for r in rows if r["role"] == "positive"]
    stats = [c["metrics"] for c in fixtures if c["role"] == "benign"]
    controls = [c for c in fixtures if c["role"] == "positive"]
    separation = {"benign_min_norm_l2": min(s["min_norm_l2"] for s in stats), "min_norm_threshold": 1e-12,
                  "benign_min_divisor": min(s["min_divisor"] for s in stats), "divisor_floor_f32": 9.999999960041972e-13,
                  "benign_max_dense_abs": max(s["dense_max_abs"] for s in stats), "max_abs_threshold": 10000,
                  "benign_max_quant_scale_abs": max(s["max_quant_scale_abs"] for s in stats),
                  "s0_safetensors_min_token_headroom": min(c["metrics"]["token_id_headroom"] for c in fixtures
                                                           if c["role"] == "benign" and not c["dtype"].startswith("Q")),
                  "s0_moe_expert_used_headroom": min(c["metrics"]["expert_used_headroom"] for c in fixtures
                                                      if c["role"] == "benign" and c["architecture"] == "mixtral"),
                  "controls": {c["id"]: {"expected_kind": c["expected_kind"], "metrics": c["metrics"]} for c in controls}}
    failed = [r["id"] for r in rows if r["errors"]]
    unclassified = sum(c["role"] == "unknown" for c in cases)
    historical_changes = [{"id": r["id"], "prior_counts": r["historical_counts"], "current_counts": r["before"].get("counts", {})}
                          for r in rows if r["iteration"] == 0 and r["historical_counts_changed"]]
    gate = "PASS" if (not failed and not repeat_mismatches and not unclassified
                      and groups["all-default"]["fpr_percent"] == 0 and cross["status"] != "HOLD") else "HOLD"
    summary = {"schema": "secant.benign-fpr-benchmark.v1", "attempt_id": args.attempt_id, "created_utc": now(),
               "platform": platform.platform(), "host": f"{sys.platform}-{platform.machine()}-host", "groups": groups, "gate": gate,
               "source_commit": command_text(["git", "rev-parse", "HEAD"], repo) if (repo / ".git").exists() else "UNKNOWN",
               "source_content_sha256": source_digest, "binary_sha256": sha(binary), "repeats": args.repeats,
               "evidence_manifest_sha256": manifest_sha, "evidence_directory": str(run),
               "positive_controls": {"total": len(positive), "passed": sum(not r["errors"] for r in positive)},
               "separation": separation, "repeat_mismatches": repeat_mismatches, "cross_host": cross,
               "unavailable_historical_rows": len(unavailable), "unclassified_configurations": unclassified,
               "historical_control_changes": historical_changes,
               "failed_rows": failed, "python": platform.python_version(),
               "cargo": command_text(["cargo", "--version"], repo), "rustc": command_text(["rustc", "--version"], repo)}
    write_json(run / "summary.json", summary)
    with (run / "summary.md").open("x") as f:
        f.write(markdown(summary))
    print(json.dumps({"gate": gate, "groups": groups}, indent=2))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
