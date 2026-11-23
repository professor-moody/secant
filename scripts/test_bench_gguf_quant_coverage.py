"""Unit tests for scripts/bench_gguf_quant_coverage.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gguf_quant_coverage as bench  # noqa: E402


REPO = Path(__file__).resolve().parents[1]


def find_secant_binary() -> Path | None:
    env_bin = os.environ.get("SECANT_BIN")
    if env_bin:
        p = Path(env_bin)
        return p if p.is_file() else None
    candidates = [REPO / rel for rel in ("target/release/secant", "target/debug/secant") if (REPO / rel).is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


class TestBenchGgufQuantCoverage(unittest.TestCase):
    def test_all_24_formats_registered_with_unique_ids(self) -> None:
        self.assertEqual(len(bench.FORMATS), 24)
        self.assertEqual(len({s.name for s in bench.FORMATS}), 24)
        self.assertEqual(len({s.ggml_type for s in bench.FORMATS}), 24)
        for spec in bench.FORMATS:
            self.assertIn(spec.block_elems, (32, 256))
            self.assertGreater(spec.block_bytes, 0)
            self.assertGreater(len(spec.fields), 0)

    def test_iq1_m_packed_f16_roundtrip_preserves_subblock_scales(self) -> None:
        spec = bench.FORMATS_BY_NAME["IQ1_M"]
        blk = bytearray(range(spec.block_bytes))
        orig_low12 = [int.from_bytes(blk[48 + 2 * i : 50 + 2 * i], "little") & 0x0FFF for i in range(4)]
        for bits in (0x3C00, 0x7E00, 0x7C00, 0x8001, 0x0000):
            bench.write_field_bits(blk, spec.fields[0], bits)
            self.assertEqual(bench.read_field_bits(blk, spec.fields[0]), bits)
            for i, want in enumerate(orig_low12):
                got = int.from_bytes(blk[48 + 2 * i : 50 + 2 * i], "little") & 0x0FFF
                self.assertEqual(got, want)

    def test_q8_k_f32_and_mxfp4_e8m0_encodings(self) -> None:
        q8k = bench.FORMATS_BY_NAME["Q8_K"]
        blk = bytearray(q8k.block_bytes)
        bench.write_field_bits(blk, q8k.fields[0], 0x7FC00000)
        self.assertEqual(bench.read_field_bits(blk, q8k.fields[0]), 0x7FC00000)

        mxfp4 = bench.FORMATS_BY_NAME["MXFP4"]
        mblk = bytearray(mxfp4.block_bytes)
        bench.write_field_bits(mblk, mxfp4.fields[0], 0xFF)
        self.assertEqual(bench.read_field_bits(mblk, mxfp4.fields[0]), 0xFF)
        self.assertIsNone(bench.encoding_bits("e8m0", "inf"))

    def test_classify_llama_output(self) -> None:
        self.assertEqual(bench.classify_llama_output(0, ""), "clean")
        self.assertEqual(bench.classify_llama_output(1, "ggml_validate_row_data: found nan value at block 0"), "rejected_nan")
        self.assertEqual(bench.classify_llama_output(1, "ggml_validate_row_data: found inf value at block 0"), "rejected_inf")
        self.assertEqual(
            bench.classify_llama_output(1, "ggml_validate_row_data: found invalid e value 255 at block 0"),
            "rejected_invalid_e8m0",
        )
        self.assertEqual(bench.classify_llama_output(1, "ggml_validate_row_data: invalid type 9"), "unsupported_type")

    def test_end_to_end_subset_with_mock_llama_cli(self) -> None:
        binary = find_secant_binary()
        if binary is None:
            self.skipTest("secant binary not built yet")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mock_llama = tmp_path / "mock-llama-cli"
            mock_llama.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "if '--help' in sys.argv:\n"
                "    print('--single-turn --simple-io')\n"
                "    sys.exit(0)\n"
                "m = sys.argv[sys.argv.index('-m') + 1]\n"
                "if 'clean' in m:\n"
                "    sys.exit(0)\n"
                "if 'mxfp4' in m:\n"
                "    sys.stderr.write('ggml_validate_row_data: found invalid e value 255 at block 0\\n')\n"
                "    sys.exit(1)\n"
                "if '_inf_' in m:\n"
                "    sys.stderr.write('ggml_validate_row_data: found inf value at block 0\\n')\n"
                "    sys.exit(1)\n"
                "sys.stderr.write('ggml_validate_row_data: found nan value at block 0\\n')\n"
                "sys.exit(1)\n",
                encoding="utf-8",
            )
            os.chmod(mock_llama, os.stat(mock_llama).st_mode | stat.S_IXUSR)

            run_dir = tmp_path / "run"
            subset = [bench.FORMATS_BY_NAME[n] for n in ("Q4_K", "Q8_K", "IQ1_M", "IQ4_XS", "MXFP4")]
            rc, summary = bench.run_benchmark(run_dir, binary, mock_llama, "att-unit", subset)
            self.assertEqual(rc, 0, json.dumps(summary["secant_summary"], indent=2))
            self.assertEqual(summary["gate"], "PASS")
            self.assertEqual(summary["secant_summary"]["clean_passed"], 5)
            self.assertEqual(summary["secant_summary"]["nan_caught"], 6)
            self.assertEqual(summary["secant_summary"]["inf_caught"], 5)
            self.assertEqual(summary["llama_cpp_summary"]["status"], "RUN")
            self.assertEqual(summary["llama_cpp_summary"]["clean_accepted"], 5)
            self.assertEqual(summary["llama_cpp_summary"]["nan_rejected_validator"], 6)
            self.assertEqual(summary["llama_cpp_summary"]["inf_rejected_validator"], 5)
            self.assertTrue((run_dir / "SHA256SUMS").is_file())
            self.assertTrue((run_dir / "summary.md").is_file())
            self.assertTrue((run_dir / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
