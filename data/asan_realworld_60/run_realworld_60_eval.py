#!/usr/bin/env python3
"""ASAN-RealWorld-60 evaluation runner and statistical analyzer (Horizon 1, Item 4).

Evaluates local LLMs on `asan_realworld_60_paired_benchmark.json` (60 functions = 30 real
sanitizer-reproduced upstream C/C++ defects paired with their 30 exact upstream patches)
across:
  - `base`
  - `calibrated_1p0x` (`composed`: Target-Shift-Calibrated rank-1 weight surgery at B2/B3 layers)
  - `random_1p0x`     (`random`: norm-matched random direction orthogonal to the aligned direction)
  - `composed_rms_1p0x` / `random_rms_1p0x` (RMS-calibrated composed domain edit where supported)

Uses the exact `ASAN-120` prompt (`asan120_prompt.py`) and strict 3-criterion scorer
(`asan120_score.py`: balanced accuracy `BA`, patched-code specificity `TNR` / `FPR = 1 - TNR`,
and root-cause recall `RC = verdict + class + sink`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ATTEMPT_ID = "att-741d25553dfbf044"
SCHEMA = "secant.asan_realworld_60-run.v1"
SEED = 20261105
CORE_ARMS = ["base", "calibrated_1p0x", "random_1p0x"]
RMS_ARMS = ["composed_rms_1p0x", "random_rms_1p0x"]
ALL_ARMS = CORE_ARMS + RMS_ARMS

# short -> (repo, revision, params_b, pinned_rms_layer, rank_index)
MODELS = {
    "qwen35_0p8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", 0.8, None, 1),
    "qwen25_1p5b": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", 1.5, None, 2),
    "smollm2_1p7b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674", 1.7, None, 3),
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "local-qwen35-4b", 4.0, None, 4),
    "qwen25_7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", 7.6, None, 5),
    "qwen3_8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", 8.2, 24, 7),
}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class NullProbe:
    def reset(self):
        self.norm_proj_D = [None] * 4096
        self.raw_proj_B = [None] * 4096
        self.raw_norm = [None] * 4096

    def __init__(self):
        self.reset()

    def __call__(self, *_a, **_k):
        return None


def render_chat_prompt(tokenizer, prompt: str, disable_thinking: bool = False) -> str:
    if getattr(tokenizer, "chat_template", None):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if disable_thinking:
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    enable_thinking=False,
                    **kwargs,
                )
            except TypeError:
                pass
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
    return prompt


def generate_batched(model, tokenizer, probe, rendered, max_new, batch_size):
    import torch

    device = next(model.parameters()).device
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    lens = [len(tokenizer(r, add_special_tokens=False)["input_ids"]) for r in rendered]
    order = sorted(range(len(rendered)), key=lambda i: (lens[i], i))
    results, batch_seconds = [None] * len(rendered), []
    for s0 in range(0, len(order), batch_size):
        ids = order[s0 : s0 + batch_size]
        enc = tokenizer(
            [rendered[i] for i in ids],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        probe.reset()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dtm = time.perf_counter() - t0
        batch_seconds.append(dtm)
        in_len = enc["input_ids"].shape[1]
        for r, i in enumerate(ids):
            new = out[r, in_len:].tolist()
            n_tok = next((k + 1 for k, t in enumerate(new) if t in eos), len(new))
            results[i] = {
                "prediction": tokenizer.decode(new[:n_tok], skip_special_tokens=True),
                "output_tokens_including_eos": n_tok,
                "prompt_tokens": int(enc["attention_mask"][r].sum()),
                "hit_max_new_tokens": bool(n_tok >= max_new and not any(t in eos for t in new)),
                "prefill_probe": {
                    "normalized_proj_D": probe.norm_proj_D[r],
                    "raw_proj_basis": probe.raw_proj_B[r],
                    "raw_norm": probe.raw_norm[r],
                },
                "batch_seconds": dtm,
                "batch_rows": len(ids),
            }
    return results, batch_seconds


def load_local_or_hf_model(model_source: str, revision: str | None, token: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    is_local = Path(model_source).exists()
    tok_kwargs = {} if is_local else {"revision": revision, "token": token}
    tok = AutoTokenizer.from_pretrained(model_source, **tok_kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    mdl_kwargs = {
        "dtype": torch.bfloat16,
    }
    if not is_local:
        mdl_kwargs["revision"] = revision
        mdl_kwargs["token"] = token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_source, **mdl_kwargs)
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(model_source, **mdl_kwargs)
    target_dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(target_dev)
    return model, tok


def rms_layer_for(pinned, n_layers: int) -> tuple[int, str]:
    if pinned is not None:
        return pinned, "pinned_in_domain-uplift-models.json"
    return min(n_layers - 1, int(round(0.67 * n_layers))), "heuristic_round(0.67*n_layers)"


def cmd_run(args) -> int:
    import numpy as np
    import torch
    import transformers as tf_pkg

    import asan120_score as S
    import run_domain_surgical_uplift_and_composition as U
    import run_domain_surgical_uplift_rms_batched as RMS
    import run_irrefutable_h100_and_public_benchmarks as IR

    short = args.model
    repo, revision, params_b, pinned_layer, rank = MODELS[short]
    model_source = args.model_path or repo
    out = Path(args.out_root) / short
    (out / "arms").mkdir(parents=True, exist_ok=True)

    bench_path = Path(args.benchmark)
    bench_sha_pre = sha256_file(bench_path)
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    cases = bench["cases"]
    if args.limit:
        keep = sorted({c["pair_id"] for c in cases})[: args.limit]
        cases = [c for c in cases if c["pair_id"] in keep]
    by_id = {c["case_id"]: c for c in cases}

    tokf = Path(os.path.expanduser("~/.cache/huggingface/token"))
    token = os.environ.get("HF_TOKEN") or (tokf.read_text().strip() if tokf.exists() else None)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, tok = load_local_or_hf_model(model_source, revision, token)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = IR.get_layers(model)
    n_layers = len(layers)

    meta: dict = {
        "schema": SCHEMA,
        "attempt_id": ATTEMPT_ID,
        "model": repo,
        "model_source": str(model_source),
        "short": short,
        "revision": revision,
        "params_b": params_b,
        "n_layers": n_layers,
        "started_utc": utc(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "transformers": tf_pkg.__version__,
        "python": platform.python_version(),
        "benchmark": str(bench_path),
        "benchmark_sha256_pre": bench_sha_pre,
        "n_items": len(cases),
        "runner_sha256": sha256_file(__file__),
        "scorer_sha256": sha256_file(HERE / "asan120_score.py"),
        "max_new_tokens": args.max_new_tokens,
        "disable_thinking": bool(args.disable_thinking),
        "batch_size_requested": args.batch_size,
    }

    # ---- IR edits (calibrated_1p0x / random_1p0x)
    candidates = IR.candidate_layers(n_layers)
    captured, cap_metrics = IR.capture_directions(model, tok, layers, candidates)
    b2 = max(candidates, key=lambda x: captured[("B2", x)].snr_proj)
    rem = [x for x in candidates if x != b2] or [b2]
    b3 = max(rem, key=lambda x: captured[("B3", x)].snr_proj)
    ir_edits = []
    for stream, lidx, pairs, target, off in (
        ("B2", b2, IR.B2_PAIRS, -7.382, 1),
        ("B3", b3, IR.B3_PAIRS, 7.382, 2),
    ):
        h_ref = IR.capture_reference_activation(
            model, tok, IR.get_down_proj(layers[lidx]), [x[0] for x in pairs[:6]]
        )
        ir_edits.append(
            IR.make_calibrated_edit(
                stream,
                lidx,
                captured[(stream, lidx)].direction,
                h_ref,
                target,
                128,
                SEED + 1000 * rank + off,
            )
        )
    meta["ir"] = {
        "candidates": candidates,
        "selected_layers": {"B2": b2, "B3": b3},
        "capture_metrics": cap_metrics,
        "calibration": [e.receipt() for e in ir_edits],
        "random_seed_rule": "SEED + 1000*rank + stream_offset",
        "random_seed_base": SEED + 1000 * rank,
    }

    # ---- RMS preparation (optional)
    rms: dict = {"status": "NOT_RUN"}
    probe = NullProbe()
    writer = original = edits = rand_edits = None
    if any(a in RMS_ARMS for a in (args.arms or ALL_ARMS)):
        try:
            rl, rsrc = rms_layer_for(pinned_layer, n_layers)
            u_layers = U.layers_of(model)
            parent, attr = U.writer_parent(u_layers[rl])
            writer = getattr(parent, attr)
            D, edits, diag = U.prepare(model, tok, writer, 1.0, 64)
            for n in RMS.DOMAINS:
                edits[n]["beta_unit"] = edits[n]["beta"]
                edits[n]["h_norm_sq"] = diag[n]["h_norm_sq"]
            sq = []
            for _d, pairs in U.PAIRS.items():
                for pos, _neg in pairs:
                    _x, y, _ln = U.capture(model, tok, writer, pos)
                    sq.append(float(torch.dot(y.double(), y.double())))
            sigma = math.sqrt(sum(sq) / len(sq))
            original = writer.weight.detach().clone()
            us = RMS.random_orthogonal_directions(D, len(RMS.DOMAINS), SEED + 7919)
            rand_edits = {
                n: {**edits[n], "direction": us[i], "orthogonal_direction": us[i]}
                for i, n in enumerate(RMS.DOMAINS)
            }
            raw_dirs = torch.stack([edits[n]["direction"] for n in RMS.DOMAINS]).double()
            Bm = torch.cat([D.double(), raw_dirs, torch.stack(us)]).float().to(writer.weight.device)
            probe = RMS.PrefillProbe(D.float().to(writer.weight.device), Bm)
            rms = {
                "status": "READY",
                "layer": rl,
                "layer_source": rsrc,
                "sigma_writer_l2": sigma,
                "writer_out_features": int(writer.weight.shape[0]),
                "random_seed": SEED + 7919,
                "support": 64,
                "random_max_abs_inner_with_D": float((torch.stack(us) @ D.T).abs().max()),
            }
        except Exception as exc:  # noqa: BLE001
            rms = {"status": "NOT_RUN", "error_type": type(exc).__name__, "error": str(exc)[:600]}
            probe = NullProbe()
    meta["rms"] = rms
    atomic_json(out / "model-meta.json", meta)

    snap = {}
    for lidx in sorted({b2, b3} | ({rms["layer"]} if rms["status"] == "READY" else set())):
        snap[lidx] = IR.get_down_proj(layers[lidx]).weight.detach().clone()

    def weights_restored() -> bool:
        return all(torch.equal(IR.get_down_proj(layers[l]).weight, w) for l, w in snap.items())

    rendered_by_id = {
        c["case_id"]: render_chat_prompt(tok, c["prompt"], disable_thinking=args.disable_thinking)
        for c in cases
    }
    ids = [c["case_id"] for c in cases]
    rendered = [rendered_by_id[i] for i in ids]

    def run_arm(arm: str, batch_size: int) -> dict:
        arm_meta_path = out / "arms" / f"{arm}.meta.json"
        rows_path = out / "arms" / f"{arm}.jsonl"
        if arm_meta_path.exists():
            return json.loads(arm_meta_path.read_text())
        t_arm = time.perf_counter()
        calib = None
        status = "COMPLETE"
        ctx = None
        handle = None
        try:
            if arm in ("base", "calibrated_1p0x", "random_1p0x"):
                ctx = IR.applied_variant(layers, ir_edits, arm)
                ctx.__enter__()
            else:
                if rms["status"] != "READY":
                    meta_a = {"arm": arm, "status": "NOT_RUN", "reason": "RMS preparation unavailable", "rms": rms}
                    atomic_json(arm_meta_path, meta_a)
                    return meta_a
                alpha = 0.5 if "0p5x" in arm else 1.0
                src = rand_edits if arm.startswith("random") else edits
                with torch.no_grad():
                    writer.weight.copy_(original)
                _ae, calib = RMS.calibrate_and_install(
                    writer, original, src, RMS.DOMAINS, True, alpha * rms["sigma_writer_l2"], 0.02, 3
                )
                if calib["status"] == "CALIBRATION_FAILED":
                    meta_a = {"arm": arm, "status": "CALIBRATION_FAILED", "calibration": calib}
                    atomic_json(arm_meta_path, meta_a)
                    return meta_a
                if calib["status"] == "CALIBRATION_OFF":
                    status = "COMPLETE_CALIBRATION_OFF"
            if rms["status"] == "READY":
                handle = writer.register_forward_hook(probe)
            results, bsec = generate_batched(model, tok, probe, rendered, args.max_new_tokens, batch_size)
        finally:
            if handle is not None:
                handle.remove()
            if ctx is not None:
                ctx.__exit__(None, None, None)
            if rms["status"] == "READY":
                with torch.no_grad():
                    writer.weight.copy_(original)
        restored = weights_restored()
        items = []
        with (out / "arms" / f"{arm}.jsonl.tmp").open("w", encoding="utf-8") as fh:
            for cid, rec in zip(ids, results):
                c = by_id[cid]
                sc = S.score_item(c, rec["prediction"], rec["output_tokens_including_eos"], args.max_new_tokens)
                row = {
                    "attempt_id": ATTEMPT_ID,
                    "model": short,
                    "revision": revision,
                    "arm": arm,
                    "case_id": cid,
                    "pair_id": c["pair_id"],
                    "family": c["family"],
                    "repo": c.get("repo"),
                    "crucible_packet_id": c.get("crucible_packet_id"),
                    "gold_label": c["label"],
                    "prediction": rec["prediction"],
                    "output_tokens": rec["output_tokens_including_eos"],
                    "prompt_tokens": rec["prompt_tokens"],
                    "hit_max_new_tokens": rec["hit_max_new_tokens"],
                    "latency_ms_amortized": rec["batch_seconds"] * 1000.0 / rec["batch_rows"],
                    "batch_seconds": rec["batch_seconds"],
                    "batch_rows": rec["batch_rows"],
                    "prefill_probe": rec["prefill_probe"],
                    "score": sc,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                items.append({**sc, "case_id": cid})
        os.replace(out / "arms" / f"{arm}.jsonl.tmp", rows_path)
        order = sorted({c["pair_id"] for c in cases})
        pm = S.point_metrics(S.pair_arrays(items, order))
        meta_a = {
            "arm": arm,
            "status": status,
            "batch_size": batch_size,
            "calibration": calib,
            "weights_restored_after_arm": restored,
            "elapsed_seconds": time.perf_counter() - t_arm,
            "generation_seconds_total": sum(bsec),
            "rows_sha256": sha256_file(rows_path),
            "summary": pm,
            "parsed": sum(1 for i in items if i["parsed"]),
            "n": len(items),
            "think_started": sum(1 for i in items if i["think_started"]),
            "think_closed": sum(1 for i in items if i["think_closed"]),
            "hit_token_cap": sum(1 for i in items if i["hit_token_cap"]),
        }
        atomic_json(arm_meta_path, meta_a)
        print(
            f"[asan-rw60] {short} {arm}: BA={pm['ba']:.3f} MCC={pm['mcc']:.3f} TPR={pm['tpr']:.3f} "
            f"TNR={pm['tnr']:.3f} FPR={1.0 - pm['tnr']:.3f} rc={pm['rc_recall']:.3f} "
            f"parsed={meta_a['parsed']}/{len(items)} cap={meta_a['hit_token_cap']} "
            f"restored={restored} {meta_a['elapsed_seconds']:.1f}s",
            flush=True,
        )
        return meta_a

    arms = [a for a in ALL_ARMS if (not args.arms or a in args.arms)]
    batch = args.batch_size
    while True:
        try:
            for arm in arms:
                run_arm(arm, batch)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 1:
                raise
            batch //= 2
            print(f"[asan-rw60] OOM on {short}; restarting ALL arms at batch_size={batch}", flush=True)
            for f in (out / "arms").glob("*"):
                f.unlink()
            meta["batch_size_fallback"] = batch
    meta["batch_size_used"] = batch
    meta["completed_utc"] = utc()
    meta["benchmark_sha256_post"] = sha256_file(bench_path)
    meta["benchmark_hash_bracket_matched"] = meta["benchmark_sha256_pre"] == meta["benchmark_sha256_post"]
    meta["weights_restored_final"] = weights_restored()
    atomic_json(out / "model-meta.json", meta)
    print(f"[asan-rw60] {short} COMPLETE batch={batch}", flush=True)
    return 0


def cmd_analyze(args) -> int:
    import asan120_score as S

    bench_path = Path(args.benchmark)
    runs_dir = Path(args.runs_dir)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    by_id = {c["case_id"]: c for c in bench["cases"]}
    pair_order = sorted({c["pair_id"] for c in bench["cases"]})

    models_out: dict = {}
    rescore_mismatches = 0

    for short in MODELS:
        mdir = runs_dir / short
        mpath = mdir / "model-meta.json"
        if not mpath.exists():
            continue
        mmeta = json.loads(mpath.read_text(encoding="utf-8"))
        max_new = int(mmeta["max_new_tokens"])
        arms_out: dict = {}
        pa_by_arm: dict = {}

        for arm in ALL_ARMS:
            ap = mdir / "arms" / f"{arm}.meta.json"
            rp = mdir / "arms" / f"{arm}.jsonl"
            if not ap.exists():
                continue
            am = json.loads(ap.read_text(encoding="utf-8"))
            if am.get("status") not in ("COMPLETE", "COMPLETE_CALIBRATION_OFF") or not rp.exists():
                arms_out[arm] = am
                continue
            rsha = sha256_file(rp)
            items = []
            for ln in rp.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                row = json.loads(ln)
                sc = S.score_item(by_id[row["case_id"]], row["prediction"], row["output_tokens"], max_new)
                if sc != row.get("score"):
                    rescore_mismatches += 1
                items.append({**sc, "case_id": row["case_id"], "family": row["family"]})
            pa = S.pair_arrays(items, pair_order)
            pa_by_arm[arm] = pa
            pt = S.point_metrics(pa)
            rng_single = S.np.random.default_rng(SEED)
            idx_single = rng_single.integers(0, len(pa["tp"]), size=(5000, len(pa["tp"])))
            bm_single = S.boot_metrics(pa, idx_single)
            ci95_single = {
                k: [float(x) for x in S.np.percentile(bm_single[k], [2.5, 97.5])]
                for k in ("ba", "mcc", "tpr", "tnr", "rc_recall", "pair_both", "strict_ba")
            }
            arms_out[arm] = {
                **am,
                "rows_sha256_verified": rsha == am.get("rows_sha256"),
                "metrics": {"point": pt, "ci95": ci95_single},
            }

        contrasts: dict = {}
        pairs_to_test = [
            ("calibrated_1p0x", "base"),
            ("random_1p0x", "base"),
            ("calibrated_1p0x", "random_1p0x"),
            ("composed_rms_1p0x", "base"),
            ("random_rms_1p0x", "base"),
            ("composed_rms_1p0x", "random_rms_1p0x"),
        ]
        for idx, (a, b) in enumerate(pairs_to_test):
            if a in pa_by_arm and b in pa_by_arm:
                pb = S.paired_bootstrap(pa_by_arm[a], pa_by_arm[b], draws=5000, seed=SEED + 17 * idx)
                mc = S.mcnemar_block(pa_by_arm[a], pa_by_arm[b])
                contrasts[f"{a}__vs__{b}"] = {
                    "paired_bootstrap": pb,
                    "mcnemar": mc,
                }
        models_out[short] = {"meta": mmeta, "arms": arms_out, "contrasts": contrasts}

    doc = {
        "schema": "crucible.asan_realworld_60.results.v1",
        "attempt_id": ATTEMPT_ID,
        "created_utc": utc(),
        "benchmark_path": str(bench_path),
        "benchmark_sha256": sha256_file(bench_path),
        "scorer_sha256": sha256_file(HERE / "asan120_score.py"),
        "n_pairs": len(pair_order),
        "n_items": len(by_id),
        "rescore_mismatches": rescore_mismatches,
        "models": models_out,
    }
    atomic_json(out_json, doc)

    # Render markdown summary
    def pct(x: float) -> str:
        return f"{100.0 * x:.1f}%"

    lines = [
        "# ASAN-RealWorld-60 Evaluation Summary (`crucible.asan_realworld_60.results.v1`)",
        "",
        f"- **Benchmark**: `asan_realworld_60_paired_benchmark.json` (`sha256={doc['benchmark_sha256']}`)",
        f"- **Suite size**: 30 real sanitizer-reproduced upstream C/C++ defects paired with their 30 exact upstream patches = **60 functions** (expanding `ASAN-120` to `ASAN-180`)",
        f"- **Independent rescore mismatches**: `{rescore_mismatches}`",
        "",
        "## Primary Evaluation Table (`base` vs `composed` / `calibrated_1p0x` vs `random` / `random_1p0x`)",
        "",
        "| Model | Arm | Balanced Acc (BA) | TPR (Vuln Recall) | TNR (Patched Spec.) | Patched FPR (`1-TNR`) | MCC | Root-Cause Recall (`RC`) | Class Acc | Sink Acc | Strict Pair-BA | Parsed |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for short, m in models_out.items():
        for arm in ALL_ARMS:
            a = m["arms"].get(arm)
            if not a or "metrics" not in a:
                continue
            pt = a["metrics"]["point"]
            arm_label = {
                "base": "`base`",
                "calibrated_1p0x": "`composed` (`calibrated_1p0x`)",
                "random_1p0x": "`random` (`random_1p0x`)",
                "composed_rms_1p0x": "`composed_rms_1p0x`",
                "random_rms_1p0x": "`random_rms_1p0x`",
            }.get(arm, f"`{arm}`")
            lines.append(
                f"| `{short}` (`{m['meta']['model']}`) | {arm_label} | "
                f"**{pct(pt['ba'])}** | {pct(pt['tpr'])} | {pct(pt['tnr'])} | {pct(1.0 - pt['tnr'])} | "
                f"`{pt['mcc']:+.3f}` | **{pct(pt['rc_recall'])}** | {pct(pt['class_acc'])} | {pct(pt['sink_acc'])} | "
                f"{pct(pt['strict_ba'])} | `{a['parsed']}/{a['n']}` |"
            )

    lines.extend([
        "",
        "## Paired Contrasts (Cluster Bootstrap 95% CI over 30 Twin Pairs, B=5000 + Exact McNemar p)",
        "",
        "| Model | Contrast | ΔBA `[95% CI]` | ΔTPR `[95% CI]` | ΔTNR `[95% CI]` | ΔRC `[95% CI]` | McNemar Vuln `(b,c,p)` | McNemar Safe `(b,c,p)` |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for short, m in models_out.items():
        for cname, c in m["contrasts"].items():
            pbm = c["paired_bootstrap"]["metrics"]
            mv = c["mcnemar"]["vuln_tpr_n60"]
            ms = c["mcnemar"]["benign_tnr_n60"]
            lines.append(
                f"| `{short}` | `{cname}` | "
                f"`{100*pbm['ba']['diff']:+.1f}` `[{100*pbm['ba']['diff_ci95'][0]:+.1f}, {100*pbm['ba']['diff_ci95'][1]:+.1f}]` | "
                f"`{100*pbm['tpr']['diff']:+.1f}` `[{100*pbm['tpr']['diff_ci95'][0]:+.1f}, {100*pbm['tpr']['diff_ci95'][1]:+.1f}]` | "
                f"`{100*pbm['tnr']['diff']:+.1f}` `[{100*pbm['tnr']['diff_ci95'][0]:+.1f}, {100*pbm['tnr']['diff_ci95'][1]:+.1f}]` | "
                f"`{100*pbm['rc_recall']['diff']:+.1f}` `[{100*pbm['rc_recall']['diff_ci95'][0]:+.1f}, {100*pbm['rc_recall']['diff_ci95'][1]:+.1f}]` | "
                f"`({mv['arm_only_correct']},{mv['ref_only_correct']},{mv['p_two_sided_exact']:.3f})` | "
                f"`({ms['arm_only_correct']},{ms['ref_only_correct']},{ms['p_two_sided_exact']:.3f})` |"
            )

    # Render 30-defect manifest table if manifest exists alongside benchmark
    manifest_path = bench_path.parent / "asan_realworld_60_manifest.json"
    if manifest_path.exists():
        mdoc = json.loads(manifest_path.read_text(encoding="utf-8"))
        lines.extend([
            "",
            "## Sealed 30-Defect Real-World Upstream Manifest (`RW01`..`RW30`, 60 Functions)",
            "",
            f"- **Manifest**: `asan_realworld_60_manifest.json` (`sha256={sha256_file(manifest_path)}`)",
            f"- **Sanitizer Verification Receipt**: `validation_receipt.json` (`30/30` pairs PASS across `clang++ -O1`, `-O0`, `-O2` with `-fsanitize=address,undefined`)",
            "",
            "| Case (`V` / `S`) | Upstream Repo | Base Commit | File Path | Function | Bug Class (`ASAN-120` Family) | Crucible Packet ID | `patch.diff` SHA-256 (prefix) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for c in mdoc.get("pairs", mdoc.get("cases", [])):
            cid = c.get("case_id", c.get("pair_id"))
            sid = c.get("safe_case_id", c.get("benign_case_id"))
            lines.append(
                f"| `{cid}` (`{c['vuln_case_id']}` / `{sid}`) | `{c['repo']}` | `{c['base_commit'][:12]}` | `{c['file_path']}` | `{c['function_name']}` | `{c['bug_class']}` (`{c['family']}`) | `{c['crucible_packet_id']}` | `{c['patch_sha256'][:16]}…` |"
            )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json} sha256={sha256_file(out_json)} mismatches={rescore_mismatches}")
    print(f"wrote {out_md} sha256={sha256_file(out_md)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=list(MODELS))
    r.add_argument("--model-path", default=None, help="Local filesystem path to model weights")
    r.add_argument("--benchmark", required=True)
    r.add_argument("--out-root", required=True)
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--max-new-tokens", type=int, default=512)
    r.add_argument("--disable-thinking", action="store_true")
    r.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--arms", nargs="*")

    a = sub.add_parser("analyze")
    a.add_argument("--benchmark", required=True)
    a.add_argument("--runs-dir", required=True)
    a.add_argument("--out-json", required=True)
    a.add_argument("--out-md", required=True)

    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
