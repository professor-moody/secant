# Qwen3.5 Multi-Seed Random Controls & Fine Interior Dose-Response Grid (2026-11-09)

This directory contains the execution receipts for Horizon 1, Item 2:

1. **`qwen35_4b_dose_and_rand3_receipt.json`** (`sha256: eae628d0611e382b5dd21f0719cf28deddc53dda6dbff92564aa9e2e88eaa5ca`):
   - **Target**: `Qwen3.5-4B` (`Q8_0` GGUF, `1,536`-token decoding budget, `30`-item `ASAN-30` evaluation suite).
   - **3-Seed Norm-Matched Random Control Ensemble (`1.00x`)**:
     - `cal738rand1` (seed `20261105`, GGUF `sha256: 6226fb5bc5554f4488f50a69d5139d0d650576bf36514b06441566fc4ed965d4`): `0 / 30` (`0.0%`) `</think>` closed, `0 / 30` strict bug class, `0 / 30` culprit frame, `0 / 30` strict root cause, `1,536.0` mean tokens.
     - `cal738rand2` (seed `20261106`, GGUF `sha256: 8b98945e534bc8d078671688894e6ac244f660a70f61f62155bcfa9d52eb5996`): `0 / 30` (`0.0%`) `</think>` closed, `0 / 30` strict bug class, `0 / 30` culprit frame, `0 / 30` strict root cause, `1,536.0` mean tokens.
     - `cal738rand3` (seed `20261107`, GGUF `sha256: 706b8cc899951d4690f26c347500e2f0415e537c69102d0e363179fa42e76cf2`): `0 / 30` (`0.0%`) `</think>` closed, `0 / 30` strict bug class, `0 / 30` culprit frame, `0 / 30` strict root cause, `1,536.0` mean tokens.
     - **3-Seed Random Ensemble (`rand1..rand3`)**: `0.0 +- 0.0 / 30` (`0.0%`) across all metrics.
   - **7-Point Dose-Response Ramp (`0.50x .. 1.50x`)**:
     - `a05` (`0.50x`): `0 / 30` (`0.0%`) closed, `0 / 30` class, `0 / 30` frame, `0 / 30` strict root cause, `1,536.0` mean tokens.
     - `a06` (`0.60x`): `2 / 30` (`6.7%`) closed, `2 / 30` class, `2 / 30` frame, `2 / 30` strict root cause, `1,500.7` mean tokens.
     - `a07` (`0.70x`): `5 / 30` (`16.7%`) closed, `5 / 30` class, `4 / 30` frame, `4 / 30` strict root cause, `1,449.9` mean tokens.
     - `a08` (`0.80x`): `12 / 30` (`40.0%`) closed, `8 / 30` class, `8 / 30` frame, `7 / 30` strict root cause, `1,303.7` mean tokens.
     - `a09` (`0.90x`): `14 / 30` (`46.7%`) closed, `10 / 30` class, `10 / 30` frame, `8 / 30` strict root cause, `1,209.6` mean tokens.
     - `cal738` (`1.00x`): `21 / 30` (`70.0%`) closed, `18 / 30` class, `17 / 30` frame, `15 / 30` strict root cause, `1,004.7` mean tokens.
     - `a15` (`1.50x`): `30 / 30` (`100.0%`) closed, `22 / 30` class, `19 / 30` frame, `16 / 30` strict root cause, `467.8` mean tokens.

2. **`qwen35_08b_multiseed_receipt.json`** (`sha256: f7da83186596c764f4711e4d2293772751dbcdaf75eac754faa45d90713e6dc2`):
   - **Target**: `Qwen3.5-0.8B` (`BF16`, `MATH-500` `N = 500`, `max_gen_toks = 2,048`).
   - **6-Arm Comparison**:
     - `base_thinking_on`: `164 / 500` (`32.80%`), `29.2%` closed, `1,659.7` mean tokens.
     - `base_thinking_off`: `254 / 500` (`50.80%`), `100.0%` closed, `585.1` mean tokens.
     - `secant_b1_edit`: `274 / 500` (`54.80%`), `70.6%` closed, `1,226.4` mean tokens.
     - `random_norm_matched` (seed 1, `42..44`): `163 / 500` (`32.60%`), `30.0%` closed, `1,648.6` mean tokens.
     - `random_norm_matched_seed2` (seed 2, `142..144`): `165 / 500` (`33.00%`), `30.8%` closed, `1,636.6` mean tokens.
     - `random_norm_matched_seed3` (seed 3, `242..244`): `154 / 500` (`30.80%`), `29.6%` closed, `1,659.4` mean tokens.
     - **3-Seed Random Ensemble (`seed1..seed3`)**: `160.67 +- 5.86 / 500` (`32.13% +- 1.17%`), `30.13% +- 0.61%` closed (`p = 3.24e-38` vs. `secant_b1_edit`).
