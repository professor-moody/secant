# Secant 60-Second Hands-On Test Drive

This directory contains a self-contained test drive that builds and runs `secant` through 9 checks on synthetic checkpoints in a temporary directory, prints a human-readable summary table of what passed vs. what tripped (and why), and removes all temporary files automatically.

## Requirements

- Rust toolchain (`cargo`, only needed if `./target/release/secant` is not yet built)
- Python 3.9+ (uses only the Python standard library; no `pip` or virtualenv required)

## Run the Test Drive

From the repository root:

```bash
python3 examples/test-drive/test_drive.py
```

To retain the generated files and native JSON receipts for review, choose a new directory:

```sh
python3 examples/test-drive/test_drive.py --work-dir /tmp/secant-test-drive-evidence
```

An existing directory is refused. Without `--work-dir`, temporary files are removed automatically. Set `SECANT_BIN` to select a specific executable; otherwise the script prefers the repository's release binary, then debug, then the binary on `PATH`.

## What It Tests

| # | Test Case | Format | Rule / Command | Expected Result |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Clean 4-layer `F32` model + valid `config.json` | `.safetensors` | `secant sanitize` | **PASS** (`exit 0`, 0 findings) |
| **2** | Clean `Q4_K` model with benign FP16 subnormal scales (`d=0x0001`, `dmin=0x03FF`) | `.gguf` | `secant sanitize` (`Rule B3-2A`) | **PASS** (`exit 0`, 0 false positives on subnormal multiplier scales) |
| **3** | MoE router `NaN` / `-Inf` sentinel exhaustion probe (`class1`) | `.safetensors` | `secant sanitize` (`Rule B3-1`) | **CAUGHT** (`exit 1`, `nan=51`, `inf=51`) |
| **4** | Out-of-bounds MoE routing IDs + all-zero RMSNorm (`class4`) | `.safetensors` | `secant sanitize --num-experts 8` (`Rule B3-3` & `B3-1`) | **CAUGHT** (`exit 1`, `routing_negative`, `routing_oob`, `routing_duplicate`, `norm_all_zero`, `norm_below_min`) |
| **5** | Corrupted `Q4_K` superblock header (`d=0x7E00` FP16 `NaN`) | `.gguf` | `secant sanitize` (`Rule B3-2A`) | **CAUGHT** (`exit 1`, `quant_scale_nan=1`) |
| **6** | Remediated `Q4_K` checkpoint (`--clamp-nonfinite`) rescan | `.gguf` | `secant sanitize --clamp-nonfinite --out ...` | **PASS** (`exit 0`, poisoned block zeroed and verified clean on rescan) |
| **7** | Subnormal FP32 divisor scale (`k_scale = 1.4e-45 < 1e-12`) | `.safetensors` | `secant sanitize` (`Rule B3-2B`) | **CAUGHT** (`exit 1`, `divisor_scale_below_floor=1`) |
| **8** | Companion `config.json` with `eos_token_id == vocab_size` | `.safetensors` + `config.json` | `secant sanitize` (`Rule S-0`) | **CAUGHT** (`exit 1`, `s0_companion_token_id_oob=1`) |
| **9** | Synthetic rank-one matrix change in layer 1 MLP | `.safetensors` | `secant patch` + `secant audit-trojan` | **FLAGGED** (`exit 0`, `model.layers.1.mlp.down_proj.weight` ranked `#1` with `sibling_z > 150` and reference `E_1 = 1.0000`) |

## Sample Output

```text
======================================================================================================================
SECANT 60-SECOND HANDS-ON TEST DRIVE SUMMARY
======================================================================================================================
TEST CASE                          | CATEGORY                  | EXP | ACT | VERDICT  | FINDINGS / WHY IT PASSED OR TRIPPED
----------------------------------------------------------------------------------------------------------------------
1. Clean F32 + config.json         | Benign Control            | 0   | 0   | PASS     | 0 findings (all tensors & config invariants clean)
2. Clean GGUF Q4_K (subnormal d)   | Benign Control (B3-2A)    | 0   | 0   | PASS     | 0 findings (all tensors & config invariants clean)
3. Router NaN / -Inf probe         | Class I (Rule B3-1)       | 1   | 1   | CAUGHT   | 102 finding(s): inf=51, nan=51
4. MoE OOB routing + zero norm     | Class IV (Rule B3-3/B3-1) | 1   | 1   | CAUGHT   | 20 finding(s): norm_all_zero=3, norm_below_min=2, routing_duplicate=4, routing_negative=5, routing_oob=6
5. GGUF Q4_K NaN scale (0x7E00)    | Class II/III (Rule B3-2A) | 1   | 1   | CAUGHT   | 1 finding(s): quant_scale_nan=1
6. Repaired Q4_K rescan            | Remediation (--clamp)     | 0   | 0   | PASS     | 0 findings (all tensors & config invariants clean)
7. Subnormal k_scale divisor       | Class II (Rule B3-2B)     | 1   | 1   | CAUGHT   | 1 finding(s): divisor_scale_below_floor=1
8. Companion config eos_token OOB  | Boundary 2 (Rule S-0)     | 1   | 1   | CAUGHT   | 1 finding(s): s0_companion_token_id_oob=1
9. Rank-1 Trojan (audit-trojan)    | Spectral Audit (S-2)      | 0   | 0   | FLAGGED  | Top outlier: model.layers.1.mlp.down_proj.weight (sibling_z=156.1, ref E_1=1.0000)
======================================================================================================================
RESULT: 9/9 test cases matched expected policy. Temporary directory cleaned up.
```

The spectral check is a numerical demonstration, not a behavioral backdoor test. A completed audit exits `0` even when its report contains outliers. First-time compilation is outside the short test-drive runtime.

## Go Further: All 24 GGUF Formats (`class5` Probe + `llama.cpp` Differential Benchmark)

The 9-case test drive above covers `Q4_K` as its representative GGUF block format. To exercise all **24 reviewed GGUF formats** (`Q4_0`..`Q8_1`, `Q2_K`..`Q8_K`, `IQ1_S`..`IQ4_XS`, `IQ1_M`, `TQ1_0`, `TQ2_0`, `MXFP4`), run the opt-in `class5` probe or the 83-case differential benchmark from the repository root:

```sh
# 1. Generate one deterministic 8,128-byte GGUF fixture spanning all 24 formats
c5_dir=$(mktemp -d)
./target/release/secant probe --class class5 --format gguf \
  --out-dir "$c5_dir/c5" --receipt "$c5_dir/c5-probe.json" --attempt-id demo-c5
# -> class5-quant-scales.gguf (24 tensors, 71 blocks, 59 findings, sha256 = 7af1ef9437b10194f299293e5528fd05ec62ce203d06553082851df7775c6e82)

# 2. Scan with SARIF 2.1.0 output (expect exit 1: 30 quant_scale_nan + 29 quant_scale_inf)
./target/release/secant sanitize --model "$c5_dir/c5/class5-quant-scales.gguf" --dry-run \
  --receipt "$c5_dir/c5-scan.json" --sarif "$c5_dir/c5-scan.sarif" --attempt-id demo-c5

# 3. Repair into a verified copy (expect exit 0, status="remediated", 0 findings remaining)
./target/release/secant sanitize --model "$c5_dir/c5/class5-quant-scales.gguf" --clamp-nonfinite \
  --out "$c5_dir/c5-fixed.gguf" --receipt "$c5_dir/c5-fix.json" --sarif "$c5_dir/c5-fix.sarif" --attempt-id demo-c5-fix
# -> output_sha256 = 9712c61401dd60afa3ea2dc52c9d7d89d55bac249200f73e082d86aa8f4bb8ed
```

To benchmark all 24 formats individually (24 clean controls, 30 planted NaN scale headers, and 29 planted Inf scale headers = 83 loadable 1-layer GGUF v3 checkpoints) and optionally compare side by side against `llama.cpp --check-tensors`:

```sh
python3 scripts/bench_gguf_quant_coverage.py \
  --run-dir "$c5_dir/bench-24fmt" \
  --binary ./target/release/secant \
  --llama-cli /path/to/llama-cli   # omit --llama-cli to run Secant-only
```

The script emits `summary.md`, `summary.json`, per-case receipts and SARIF 2.1.0 logs, and a verified `SHA256SUMS` manifest.

## Next Steps

- [Practical recipes](../../docs/RECIPES.md) cover scanning, repair, fine-tuning, and quantized-export verification.

- **[Offensive Researcher's Guide](../../docs/OFFENSIVE-RESEARCHERS-GUIDE.md)**: Step-by-step CLI recipes for crafting synthetic weight-space and quantization-skew test cases.
- **[PyTorch 10-Line Healing Guide](../../docs/PYTORCH-10-LINE-HEALING-GUIDE.md)**: Constrain supplied output directions during PyTorch LoRA fine-tuning with `ProjectThenHealLinear`.
