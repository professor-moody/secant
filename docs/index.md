# Secant Scan, Diff, and Patch Model Files on Disk

Secant is a single Rust CLI to scan, diff against a base model, and patch `.safetensors` and `.gguf` files directly on disk, with no Python and no GPU required. It reads the file format itself, so no model is loaded into PyTorch, llama.cpp, or vLLM. An optional Python package adds a LoRA layer that keeps chosen directions fixed during fine-tuning.

## Four Things You Can Do

**Diff a downloaded checkpoint against its base model.** Compare a community fine-tune, merge, or quant with the official base and list every tensor that changed, including embeddings, routers, and the output head. `audit-trojan --reference` ranks the changes and shows which vocabulary tokens the largest change promotes.

```sh
secant read --model finetune.safetensors --reference base.safetensors --top-k-singular 1 --top-cols 1
secant audit-trojan --model finetune.safetensors --reference base.safetensors --vocab-tensor lm_head.weight
```

**Scan for broken or poisoned values before loading.** `sanitize` checks for `NaN` and `Inf` weights, bad GGUF block scales, zero divisor scales, out-of-range MoE expert indices, and token IDs in `config.json` that point past the vocabulary. Add `--clamp-nonfinite --out` to write a repaired copy that is published only after a clean rescan.

```sh
secant sanitize --model model.gguf --receipt scan.json --attempt-id pre-load
```

**Patch weights in place on disk.** Describe the edit in a JSON plan; `patch` writes only those tensors, checks that every other byte matches the original, and records SHA-256 before and after. `quant patch` does the same for `Q8_0` blocks.

```sh
secant patch --base model.safetensors --plan plan.json --in-place --json > patch-receipt.json
```

**Keep chosen directions fixed during LoRA fine-tuning.** `secant.nn.ProjectThenHealLinear` wraps an `nn.Linear` so the adapter cannot write into directions you choose, and merges back into a plain layer for inference.

## Start Here

- **[Quickstart](quickstart.md#try-the-four-jobs-on-demo-files)**: run all four jobs on tiny generated files in a few minutes, with no model download.
- **[Practical recipes](RECIPES.md)**: copy-paste commands for diffing, scanning, repairing, patching, and verifying real checkpoints.
- **[PyTorch guide](PYTORCH-10-LINE-HEALING-GUIDE.md)**: a runnable `ProjectThenHealLinear` example with leakage checks in float32 and float64.
- **[Offensive researcher's guide](OFFENSIVE-RESEARCHERS-GUIDE.md)**: the scanner's rules, benign controls, and how to evaluate a model-file scanner.

## Scope

A clean scan means the listed checks passed; it cannot see a behavioral backdoor in normal-looking weights. Diffing needs a trusted base with the same tensor names. Patching supports float tensors and `Q8_0` blocks; `Q4_K` can be read, scanned, and verified but not patched, and the patcher holds the file in memory. The LoRA lock covers the wrapped layers, so measure leakage again after merging or quantizing.

## See It Run

This is a real terminal recording. `secant patch` edits two tensors in a copy of the 3.4 GB `SmolLM2-1.7B-Instruct` checkpoint, and the output hash matches the one recorded when the original experiment ran. The roughly 30 s of disk I/O spent copying and hashing is cut from the recording so there is no blank pause. The [quickstart](quickstart.md) shows all six steps, from `inspect` to `verify-patch`, with their tapes and text transcripts.

```{figure} assets/secant-patch.gif
:name: fig-index-vhs-patch
:alt: Terminal recording of secant patch editing two tensors of SmolLM2-1.7B-Instruct and printing the output SHA-256
:width: 100%

`secant patch` on a real checkpoint: 104,755 of 16,777,216 entries change in the `L15` MLP writer, and every tensor outside the plan stays byte-identical.
```

## How the Editing Pipeline Works

The research side of Secant builds calibrated edits from captured activations and then protects them during later training. Three measured properties carry it:

- **Edits touch few bytes.** A calibrated rank-one edit changes only the input columns where the reference activation is largest. On `Q8_0`, only the 34-byte blocks those columns intersect are re-encoded; in the `Qwen3.5-4B` test that was `0.20%` of the file's bytes, with every other block bit-for-bit identical.
- **Later fine-tuning cannot undo them.** Masking the gradient is not enough under AdamW, whose per-coordinate scaling rotates updates back into the edited directions. `ProjectThenHealLinear` projects the adapter in `float64` on every forward pass, holding leakage at `5e-18` to `1e-17` in `float64` and about `1e-10` in `float32`.
- **Value checks run before native kernels.** Container validators check offsets and lengths; `sanitize` checks the values themselves, separating multiplicative block scales (rule `B3-2A`, which allows legitimate `Q8_0`/`Q4_K` subnormals), divisor scales (`B3-2B`), and integer routing tables (`B3-3`).

```{figure} assets/secant-pipeline.svg
:name: fig-secant-pipeline
:alt: Four-stage Secant diagram. Stage 1 shows control and target activation distributions and the per-layer SNR curve of SmolLM2-1.7B with its Borda winner L12. Stage 2 shows two-pass Gram-Schmidt orthogonalization and the sparse top-64 input-column support. Stage 3 shows Q8_0 32-weight blocks fitting 128-column panels exactly while a 256-weight Q4_K superblock is split. Stage 4 shows the ProjectThenHealLinear update staying in the plane orthogonal to D while an AdamW step drifts out of it.
:width: 100%

The four stages and the geometry behind each one. **1. Capture and rank:** the target prompts' projections separate from the control prompts' by `m / σ = 40.48` at `L12`, and `secant analyze` picks that layer by Borda rank over four activation streams. **2. Orthogonalize and size:** two `float64` Gram-Schmidt passes take the two directions from cosine `0.1236` to `1.12e-17`, and the `B3` edit touches only the 64 input columns where the reference activation is largest. **3. Patch stored blocks:** a 128-column panel holds exactly four `Q8_0` blocks, but cuts a `Q4_K` superblock in half. **4. Lock later training:** `ProjectThenHealLinear` keeps the adapter update in the plane perpendicular to `D`. AdamW's per-coordinate scaling tilts a gradient-masked update back out of that plane. Panels marked DATA are drawn from real files; panels marked SCHEMATIC are sketches whose labels quote measured values.
```

## What the Reported Experiments Show

Different experiments answer different questions. The paper reports results across checkpoints from `0.8B` to `72.7B` parameters, but the two largest stress tests used a single checkpoint shard. No single experiment establishes every capability across that range.

| Experiment | Result | What it establishes |
|---|---|---|
| `Qwen3.5-0.8B` `MATH-500` 4-arm control (`n = 100`, `4,096` tokens) | `base_thinking_on` `1.0%` (`1%` closed); `base_thinking_off` `48.0%`; `secant_b1_edit` `24.0%` (`64%` closed, `31.0%` whole-response); `random_norm_matched` `0.0%` (`2%` closed, `p = 1.19e-7` vs. `secant_b1_edit`) | `enable_thinking=False` reaches `48.0%`, close to the `48.1%` Takehi et al. reported for the weight edit under a different protocol (`32,768` tokens, stochastic decoding), while with `<think>` enabled the targeted `B1` edit raises `</think>` completion `64x` and beats a norm-matched random edit (`24.0%` vs. `0.0%`). |
| Reference-activation calibration, 15 models (`0.8B` to `32.8B`) | Maximum `float64` shift error `<= 7.99e-15` | The solved step size realizes the exact target shift at the reference activation across model widths. |
| Public benchmarks after the edit, 14 models (`0.8B` to `32.8B`) | Calibrated `1.0x` edits stay within `-2.0` to `+2.1` points of base across `GSM8K`, `MATH-500`, `MMLU-Pro`, `HumanEval`, and `ASan-30` | Pooled, unhealed surgical edits leave general benchmarks within about 2 points of base, with one significant pooled cost (`MMLU-Pro` `-2.0`); single 100-item cells range from `-12` to `+19`. |
| Reasoning-loop test, `Qwen3.5-4B`, 30 crash traces | Completion within the token limit rose from `0/30` (base and copied `0.8B` ratio) to `21/30` (`1.0x`) and `30/30` (`1.5x`, `19/30` strict root cause) | On this one model (`n = 30`, one run per arm), reference-activation sizing closed `21/30` at `1.0x` and `30/30` once the loop edit was raised to `1.5x`, whereas copying a fixed norm ratio closed `0/30`. |
| LoRA comparison, 15 models (`0.8B` to `32.8B`) | Gradient-projected AdamW leaked `0.33%` to `1.67%`; `ProjectThenHealLinear` stayed at `1.78e-10` to `8.67e-10` (`F32`) | Forward-pass double projection locks LoRA updates out of the removed subspace under AdamW across all 15 tested architectures. |
| Live causal-LM healing & RMS-relative dosing (`8` models, `0.8B` to `14B`) | `float64` leakage `5.21e-18..9.87e-18` (max-absolute) and `3.18e-17..5.07e-17` (relative); `ProjectThenHealLinear` lifted `Qwen2.5-Math-7B` `GSM8K` `87% -> 93%` and `Qwen2.5-Coder-7B` `HumanEval` `36% -> 42%` | On looping models (`Qwen3-8B`), both targeted and random edits lift `GSM8K` `18% -> 30%` by shortening `<think>` timeouts, showing why random controls are mandatory; direct `heal-the-base` and `healed-random` controls show the `87% -> 93%` `Qwen2.5-Math-7B` `GSM8K` gain comes from the 60-step LoRA pass itself (`93%` on both controls while `ProjectThenHealLinear` cuts domain leakage from `4.75%` to `5.08e-17`), while on `Qwen2.5-Coder-7B` `HumanEval` `heal-the-base` (`37%`) and `healed-random` (`35%`) do not reproduce `42%` (`p = 0.332` and `0.189`, neither surviving Holm correction). |
| Paired `ASAN-120` vulnerability & benign twins (`9` models x `7` arms, `7,560` rows) | `60` vulnerable C/C++ functions paired with `60` patched benign twins. Non-reasoning models (`0.8B` to `8B`) collapse to one-sided guessing (`BA = 0.475..0.533`). On `<think>` reasoning models, the `composed_rms_1p0x` edit beats its random control `4/4` on balanced accuracy (sign test `p = 0.125` across 9 models) and lifts strict root-cause recall from `45%` to `55%` on `Qwen3-14B` (95% CI `0` to `+21.7`; no Holm survivor) | Patched benign twins expose models that label almost every snippet `VULNERABLE` (`93%` to `100%` false-positive rate on `Llama-3.1-8B` and `SmolLM2-1.7B`). |
| Two-stage spectral Trojan detector (`secant audit-trojan`, 24 audits) | Whole-matrix spectral metrics trigger near $g^* = \sigma_1(W)/\|W\|_F$ (tested to `16%` dose; no signal at the pre-specified `0.10%` to `4.00%` doses; trigger token in the top 2 of the vocabulary projection at `8%`, rank `#1` at `10%`, sibling flag at `12%`) | Stage 1 peels outlier singular values against the Marchenko-Pastur bulk and checks row/column spikes; Stage 2 adds the cross-layer sibling Z-score and `--vocab-tensor` projection. Together they flag a planted rank-one edit once it approaches the layer's top singular value; below that, single-file inspection shows no signal. |
| Role-aware Boundary-3 sanitization (`secant sanitize`, 10 GGUF formats & SafeTensors) | Scans `186,870,528` quantized blocks (`10 / 10` checkpoints, six GGUF and four SafeTensors including small test models, with `8.99M` benign subnormal `Q4_K` scales) with `0` observed false positives (`B3-1` and `B3-2A` on that corpus; `B3-2B` and `B3-3` validated at rule level on synthetic GPTQ/AWQ/FP8/routing fixtures) | Separating multiplicative block scales (`B3-2A`) from divisor scales (`B3-2B`) and routing indices (`B3-3`) eliminates subnormal false alarms on production GGUF models. |
| Static spectral atlas (`46`-file inventory, `2,619` weight matrices) | `2,619` analyzed 2D tensors across `14` full-model reads and `16` scoped variants; SafeTensors and GGUF copies agree bit-for-bit | Baseline spectral distribution of unedited open-weight models across `.safetensors` and `.gguf`. |

```{figure} assets/benchmark-retention-14models.svg
:name: fig-index-benchmark-retention
:alt: Small multiples of paired change from base for 14 models on GSM8K, MATH-500, MMLU-Pro, HumanEval and ASan-30, comparing calibrated edits with norm-matched random edits
:width: 100%

What a calibrated `B2 + B3` edit does to public benchmarks on 14 models. Each dot is a paired change from the same model's unedited score on the same items. Blue is the calibrated edit; orange is a random direction with the same columns and the same Frobenius norm. Almost every dot sits near zero, and blue and orange overlap. Pooled over all 14 models, each benchmark moves by about 2 points or less. The full method and every number are in the [paper, Section 4.2](#sec-paper-public-benchmarks).
```

```{figure} assets/h100-15model-leakage-log10.svg
:name: fig-index-leakage-log10
:alt: Log-scale relative leakage at step 60 for unconstrained LoRA, gradient-projected AdamW and ProjectThenHealLinear on 15 models, plus trajectories over 60 steps
:width: 100%

Leakage into the protected direction after 60 AdamW steps, on a log scale. Masking the gradient (yellow) still leaks `0.33%` to `1.67%`. `ProjectThenHealLinear` (green) stays between `1.78e-10` and `8.67e-10` on all 15 models, 6.5 million to 38 million times lower than the gradient-masked arm on the same model.
```

```{figure} assets/qwen4b-phase-transition.svg
:name: fig-qwen4b-phase-transition
:alt: Completion and attribution scores at different loop-edit strengths on 30 Qwen3.5-4B crash traces
:width: 100%

On this 30-prompt test, the calibrated edit completed 21 answers at the standard dose. Increasing only the loop-edit component (`B1`) to 1.5 times that dose completed all 30, with 23 correct culprit frames and 19 correct root causes. The other two edit components stayed at their standard dose.
```

```{figure} assets/asan120-paired-audit-uplift.svg
:name: fig-index-asan120
:alt: ASAN-120 paired vulnerability and benign control benchmark across 9 models and 7 steering arms
:width: 100%

Paired `ASAN-120` evaluation across 9 models (`60` vulnerable C/C++ snippets paired with `60` patched benign twins, `7,560` completions). Non-reasoning models (`0.8B` to `8B`) collapse to trivial one-sided predictors (`BA ≈ 0.50`, `RC <= 0.067`), whereas on the 4 `<think>` reasoning models `composed_rms_1p0x` beats `random_rms_1p0x` `4/4` on balanced accuracy (`6W / 1L / 2T` overall) and lifts strict root-cause recall (`RC`) from `0.183` to `0.250` on `R1-Llama-8B` and from `0.450` to `0.550` on `Qwen3-14B`. Method and tables: [paper, Section 4.5](#sec-paper-asan120).
```

```{figure} assets/trojan-detection-phase-transition.svg
:name: fig-index-trojan-phase
:alt: Planted rank-1 trojan detection against relative Frobenius dose on SmolLM2-1.7B and Qwen3.5-4B
:width: 100%

The audit side of the same tools. A rank-1 edit planted in one layer showed no signal at the pre-specified `0.10%` to `4.00%` doses. The tested post-hoc doses bracket the nominal crossover $g^* = \sigma_1(W)/\|W\|_F$ (`6.57%` Qwen, `8.65%` SmolLM2): $\sigma_2$ moved at `6%`, the trigger token reached rank 1 or 2 in the vocabulary projection at `8%`, and the automatic sibling test flagged the layer at `12%`. Method and tables: [payload checks, Section 5.1](#sec-trojan-dose-curve).
```

## Go Deeper

- [Theory and calibration](theory-and-calibration.md): understand direction selection, non-overlapping edits, and reference-based edit sizing before the equations.
- [Editing quantized weights](quantized-surgery.md): understand which compressed blocks can change and why `Q4_K` patching is refused.
- [Project-Then-Heal](project-then-heal.md): understand the training constraint, its numerical limits, and the measured results.
- [CLI, Python, and C reference](cli-and-python-reference.md): find command templates, API signatures, exit codes, and receipt fields.
- [Payload checks](payload-security.md): learn what the static scanner checks and what each result means.
- [Scientific paper](PAPER.md): read the derivations, experimental tables, related work, and limitations together.

Secant builds on weight-space programming, representation engineering, and constrained model editing [@rikiyat2026wsp; @zou2023representation; @zou2024circuitbreakers; @arditi2024refusal]. The paper separates prior work from the methods evaluated here.
