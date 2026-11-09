#!/usr/bin/env python3
"""ASAN-120 scorer and paired statistics (pure python + numpy; no GPU, no network).

Attempt: att-741d25553dfbf044

Scoring is lexical and mechanical, not human-judged. Every branch that cannot decide returns an explicit
UNPARSED / None so that a useless answer looks useless (CLAUDE.md rule 10).

Per-item fields
  verdict          VULNERABLE | SAFE | None (UNPARSED)
  bug_class        one of CLASSES | NONE | None (UNPARSED)
  class_correct    gold class == parsed class (first label only when the field lists several)
  sink_line_correct  first integer in the SINK field is a gold SINK/ROOT line (vulnerable items only)
  sink_symbol_correct a gold sink symbol appears as an identifier in the SINK field
  frame_correct    sink_line_correct or sink_symbol_correct
  root_cause_correct  verdict VULNERABLE and class_correct and frame_correct (mirrors ASAN30)
  terms_hit        a gold term appears as an identifier in the ROOT_CAUSE field (diagnostic only)
  think_started / think_closed / hit_token_cap / refusal
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

CLASSES = (
    ("HEAP_BUFFER_OVERFLOW", "CWE-122"),
    ("STACK_BUFFER_OVERFLOW", "CWE-121"),
    ("USE_AFTER_FREE", "CWE-416"),
    ("INTEGER_OVERFLOW", "CWE-190"),
    ("SIGN_CONVERSION", "CWE-195"),
    ("OFF_BY_ONE", "CWE-193"),
    ("NULL_DEREFERENCE", "CWE-476"),
    ("DOUBLE_FREE", "CWE-415"),
    ("UNINITIALIZED_READ", "CWE-908"),
    ("ASSERT_ONLY_CHECK", "CWE-617"),
    ("TYPE_CONFUSION", "CWE-843"),
    ("UNBOUNDED_RECURSION", "CWE-674"),
)
LABELS = tuple(c for c, _ in CLASSES)
CWE_TO_LABEL = {cwe.split("-")[1]: lab for lab, cwe in CLASSES}
REFUSAL_RE = re.compile(
    r"(i can(?:not|'t| not) (?:assist|help|analyze|provide|triage)|i must decline|"
    r"unable to (?:analyze|assist|help|triage)|against my guidelines)", re.I)
_DECOR = r"[\s\*_`#>\-\[\(]*"
VERDICT_RE = re.compile(r"^" + _DECOR + r"VERDICT" + _DECOR + r"[:=]" + _DECOR + r"(VULNERABLE|SAFE)\b",
                        re.I | re.M)


def _field(answer: str, name: str, start: int) -> str | None:
    pat = re.compile(r"^" + _DECOR + name + _DECOR + r"[:=]" + r"[\s\*_`]*(.*)$", re.I | re.M)
    m = pat.search(answer, start)
    if m is None and start:
        ms = list(pat.finditer(answer))
        m = ms[-1] if ms else None
    return None if m is None else m.group(1).strip().strip("*_` ")


def split_think(text: str) -> tuple[str, bool, bool]:
    """Return (answer_text, think_started, think_closed). Unclosed thinking yields an empty answer."""
    started = "<think>" in text or "</think>" in text
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1], True, True
    if "<think>" in text:
        return "", True, False
    return text, False, False


def normalize_label(field: str | None) -> tuple[str | None, bool]:
    """Map a BUG_CLASS field to (label | 'NONE' | None, multi_label_flag)."""
    if field is None:
        return None, False
    up = re.sub(r"[^A-Z0-9]+", "_", field.upper()).strip("_")
    hits: list[tuple[int, str]] = []
    for lab in LABELS:
        i = up.find(lab)
        if i >= 0:
            hits.append((i, lab))
    for m in re.finditer(r"CWE[-_ ]?(\d{3})", field.upper()):
        lab = CWE_TO_LABEL.get(m.group(1))
        if lab:
            hits.append((m.start(), lab))
    if hits:
        hits.sort()
        distinct = {h[1] for h in hits}
        return hits[0][1], len(distinct) > 1
    if re.fullmatch(r"(NONE|N_A|NA|NO_BUG|NOT_APPLICABLE)", up) or up.startswith("NONE"):
        return "NONE", False
    return None, False


def parse_response(text: str) -> dict[str, Any]:
    answer, started, closed = split_think(text)
    out: dict[str, Any] = {
        "think_started": started, "think_closed": closed, "answer_chars": len(answer.strip()),
        "verdict": None, "bug_class": None, "class_multi": False, "sink_field": None,
        "sink_first_int": None, "root_cause_field": None, "parse_note": "",
    }
    ms = list(VERDICT_RE.finditer(answer))
    if not ms:
        out["parse_note"] = "UNPARSED:no-verdict-line" + ("" if closed or not started else ":think-unclosed")
        return out
    last = ms[-1]
    out["verdict"] = last.group(1).upper()
    pos = last.end()
    out["bug_class"], out["class_multi"] = normalize_label(_field(answer, "BUG_CLASS", pos))
    sink = _field(answer, "SINK", pos)
    out["sink_field"] = sink
    if sink:
        m = re.search(r"\d+", sink)
        out["sink_first_int"] = int(m.group(0)) if m else None
    out["root_cause_field"] = _field(answer, "ROOT_CAUSE", pos)
    return out


def _ident_hit(text: str | None, words: Sequence[str]) -> bool:
    if not text:
        return False
    return any(re.search(r"(?<![A-Za-z0-9_])" + re.escape(w) + r"(?![A-Za-z0-9_])", text) for w in words)


def score_item(case: Mapping[str, Any], text: str, n_tokens: int, max_tokens: int) -> dict[str, Any]:
    p = parse_response(text)
    gold_vuln = case["label"] == "VULNERABLE"
    verdict = p["verdict"]
    s: dict[str, Any] = {
        "gold_label": case["label"], "verdict": verdict, "parsed": verdict is not None,
        "verdict_correct": verdict == case["label"], "bug_class": p["bug_class"],
        "class_multi": p["class_multi"], "think_started": p["think_started"],
        "think_closed": p["think_closed"], "hit_token_cap": bool(n_tokens >= max_tokens),
        "refusal": bool(REFUSAL_RE.search(split_think(text)[0])) or p["answer_chars"] < 20,
        "parse_note": p["parse_note"], "sink_first_int": p["sink_first_int"],
    }
    if gold_vuln:
        gold_lines = set(case["sink_lines"]) | set(case["root_lines"])
        s["class_correct"] = p["bug_class"] == case["gold_class"]
        s["sink_line_correct"] = p["sink_first_int"] in gold_lines if p["sink_first_int"] is not None else False
        s["sink_symbol_correct"] = _ident_hit(p["sink_field"], case["sink_symbols"])
        s["frame_correct"] = s["sink_line_correct"] or s["sink_symbol_correct"]
        s["root_cause_correct"] = bool(verdict == "VULNERABLE" and s["class_correct"] and s["frame_correct"])
        s["terms_hit"] = _ident_hit(p["root_cause_field"], case["terms"])
    else:
        s["hallucinated_bug"] = verdict == "VULNERABLE"
        s["class_none_when_safe"] = p["bug_class"] == "NONE"
        s["terms_hit"] = _ident_hit(p["root_cause_field"], case["terms"])
    return s


# ----------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------
def mcnemar_exact_two_sided(b: int, c: int) -> float:
    """Exact two-sided McNemar on the discordant counts (b: A right/B wrong, c: A wrong/B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def holm(pvals: Sequence[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, pvals[i] * (m - rank)))
        adj[i] = run
    return adj


def confusion_from_pairs(tp: np.ndarray, fn: np.ndarray, tn: np.ndarray, fp: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics from per-pair 0/1 indicators; arrays may be (P,) or (draws, P) and are summed on the last axis.
    Unparsed outputs are counted as errors (FN for a vulnerable item, FP for a safe item)."""
    TP, FN, TN, FP = (a.sum(axis=-1).astype(float) for a in (tp, fn, tn, fp))
    tpr = TP / np.maximum(TP + FN, 1)
    tnr = TN / np.maximum(TN + FP, 1)
    den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = np.where(den > 0, (TP * TN - FP * FN) / np.where(den > 0, den, 1), 0.0)
    return {"tpr": tpr, "tnr": tnr, "ba": (tpr + tnr) / 2, "mcc": mcc}


def pair_arrays(items: Sequence[Mapping[str, Any]], order: Sequence[str]) -> dict[str, np.ndarray]:
    """items: scored records keyed by case_id 'Fxx-Pn-V' / '-S'. Returns per-pair indicator arrays."""
    by = {it["case_id"]: it for it in items}
    tp, fn, tn, fp, rc, cls, snk, both = ([] for _ in range(8))
    for pid in order:
        v, sf = by[pid + "-V"], by[pid + "-S"]
        vv, sv = v["verdict"], sf["verdict"]
        tp.append(vv == "VULNERABLE"); fn.append(vv != "VULNERABLE")
        tn.append(sv == "SAFE"); fp.append(sv != "SAFE")
        rc.append(bool(v["root_cause_correct"])); cls.append(bool(v["class_correct"]))
        snk.append(bool(v["frame_correct"]))
        both.append(vv == "VULNERABLE" and sv == "SAFE")
    a = lambda x: np.array(x, dtype=np.int64)
    return {"tp": a(tp), "fn": a(fn), "tn": a(tn), "fp": a(fp), "rc": a(rc), "cls": a(cls), "snk": a(snk),
            "both": a(both)}


def point_metrics(pa: Mapping[str, np.ndarray]) -> dict[str, float]:
    m = confusion_from_pairs(pa["tp"], pa["fn"], pa["tn"], pa["fp"])
    n = len(pa["tp"])
    out = {k: float(v) for k, v in m.items()}
    out.update({"rc_recall": float(pa["rc"].mean()), "class_acc": float(pa["cls"].mean()),
                "sink_acc": float(pa["snk"].mean()), "pair_both": float(pa["both"].mean()),
                "strict_ba": (float(pa["rc"].mean()) + float(m["tnr"])) / 2, "n_pairs": n})
    return out


def boot_metrics(pa: Mapping[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    g = lambda k: pa[k][idx]
    m = confusion_from_pairs(g("tp"), g("fn"), g("tn"), g("fp"))
    m["rc_recall"] = g("rc").mean(axis=-1)
    m["pair_both"] = g("both").mean(axis=-1)
    m["strict_ba"] = (m["rc_recall"] + m["tnr"]) / 2
    return m


def paired_bootstrap(pa_arm: Mapping[str, np.ndarray], pa_ref: Mapping[str, np.ndarray],
                     draws: int = 10000, seed: int = 20261105) -> dict[str, Any]:
    """Cluster bootstrap over matched pairs: each draw resamples pair indices (same indices for both arms)."""
    n = len(pa_arm["tp"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    a, r = boot_metrics(pa_arm, idx), boot_metrics(pa_ref, idx)
    out: dict[str, Any] = {"draws": draws, "seed": seed, "resample_unit": "matched pair", "metrics": {}}
    obs_a, obs_r = point_metrics(pa_arm), point_metrics(pa_ref)
    for k in ("ba", "mcc", "tpr", "tnr", "rc_recall", "pair_both", "strict_ba"):
        d = a[k] - r[k]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["metrics"][k] = {"arm": obs_a[k], "ref": obs_r[k], "diff": obs_a[k] - obs_r[k],
                             "diff_ci95": [float(lo), float(hi)],
                             "arm_ci95": [float(x) for x in np.percentile(a[k], [2.5, 97.5])]}
    return out


def mcnemar_block(pa_arm: Mapping[str, np.ndarray], pa_ref: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Item-level exact McNemar on three correctness definitions (arm vs ref)."""
    def disc(x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
        return int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())

    res: dict[str, Any] = {}
    # verdict correctness over all 120 items: vulnerable item right = tp, safe item right = tn
    xa = np.concatenate([pa_arm["tp"], pa_arm["tn"]]); xr = np.concatenate([pa_ref["tp"], pa_ref["tn"]])
    for name, (x, y) in {"verdict_correct_n120": (xa, xr), "root_cause_n60": (pa_arm["rc"], pa_ref["rc"]),
                         "benign_tnr_n60": (pa_arm["tn"], pa_ref["tn"]),
                         "vuln_tpr_n60": (pa_arm["tp"], pa_ref["tp"]),
                         "pair_both_n60": (pa_arm["both"], pa_ref["both"])}.items():
        b, c = disc(x, y)
        res[name] = {"arm_only_correct": b, "ref_only_correct": c, "n_items": int(len(x)),
                     "p_two_sided_exact": mcnemar_exact_two_sided(b, c)}
    return res
