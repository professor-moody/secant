"""Check that incomplete scans and changing evidence cannot report zero FPR."""

import copy
import tempfile
import sys
import unittest
from pathlib import Path

from bench_benign_fpr import discover, group, measure


def row(name="a", status="clean", code=0, iteration=0):
    return {"id": name, "status": status, "exit_code": code, "iteration": iteration,
            "integrity_errors": [], "before": {}, "wall_ns": 1000000,
            "bytes": 1048576, "peak_rss_bytes": 1048576, "pre": {"input": name}}


class DenominatorTests(unittest.TestCase):
    def test_rss_is_per_command_not_cumulative_and_units_are_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            big = measure([sys.executable, "-c", "x = bytearray(64 * 1024 * 1024)"], p / "big.out", p / "big.err")
            small = measure([sys.executable, "-c", "pass"], p / "small.out", p / "small.err")
            self.assertEqual((big["exit_code"], small["exit_code"]), (0, 0))
            self.assertGreater(big["peak_rss_bytes"], 64 * 1024 * 1024)
            self.assertGreater(big["peak_rss_bytes"] - small["peak_rss_bytes"], 32 * 1024 * 1024)

    def test_repeats_do_not_inflate_checkpoint_denominator(self):
        r = group([row(), row(iteration=1), row("b", "findings", 1), row("b", iteration=1)])
        self.assertEqual((r["configurations"], r["scans"], r["false_positives"], r["fpr_percent"]), (2, 4, 1, 50.0))

    def test_incomplete_or_contradictory_exit_prevents_zero_fpr(self):
        for status, code in [("unknown", 2), ("clean", -9), ("clean", 1), ("findings", 0)]:
            with self.subTest(status=status, code=code):
                self.assertIsNone(group([row(), row("b", status, code)])["fpr_percent"])

    def test_changed_evidence_prevents_zero_fpr_even_if_scanner_says_clean(self):
        bad = copy.deepcopy(row())
        bad["integrity_errors"] = ["binary changed during scan"]
        self.assertIsNone(group([bad])["fpr_percent"])
        self.assertIsNone(group([])["fpr_percent"])

    def test_unlabelled_model_is_not_assumed_benign_and_run_directory_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "unknown.safetensors").write_bytes(b"unlabelled")
            run = repo / "custom-run-name"
            run.mkdir()
            (run / "generated.gguf").write_bytes(b"generated")
            cases, _, _ = discover(repo, run)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["role"], "unknown")


if __name__ == "__main__":
    unittest.main()
