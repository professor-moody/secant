# Secant: Inspect and Edit Model Weights

**Secant** reads, edits, and checks the numerical weights stored in language-model checkpoints. Its Rust command-line tool works with `.safetensors` and `.gguf` files. Its Python package supports **LoRA**, a fine-tuning method that learns a small adapter while keeping the original weights fixed.

The workflow has three parts: choose a direction in a layer's activations, make a measured weight edit along that direction, and check the result. A direction is a vector that captures a difference between two sets of prompts. It is a useful experimental signal, not a complete description of a behavior.

Secant also offers **Project-Then-Heal**: a LoRA layer that keeps adapter updates perpendicular to specified directions. This constraint applies at the selected layer's output. It does not guarantee that the whole model will preserve a behavior after fine-tuning.

```{figure} assets/secant-pipeline.svg
:name: fig-secant-pipeline
:alt: Four-stage Secant diagram. Stage 1 shows control and target activation distributions and the per-layer SNR curve of SmolLM2-1.7B with its Borda winner L12. Stage 2 shows two-pass Gram-Schmidt orthogonalization and the sparse top-64 input-column support. Stage 3 shows Q8_0 32-weight blocks fitting 128-column panels exactly while a 256-weight Q4_K superblock is split. Stage 4 shows the ProjectThenHealLinear update staying in the plane orthogonal to D while an AdamW step drifts out of it.
:width: 100%

The four stages and the geometry behind each one. **1. Capture and rank:** the target prompts' projections separate from the control prompts' by `m / σ = 40.48` at `L12`, and `secant analyze` picks that layer by Borda rank over four activation streams. **2. Orthogonalize and size:** two `float64` Gram-Schmidt passes take the two directions from cosine `0.1236` to `1.12e-17`, and the `B3` edit touches only the 64 input columns where the reference activation is largest. **3. Patch stored blocks:** a 128-column panel holds exactly four `Q8_0` blocks, but cuts a `Q4_K` superblock in half. **4. Lock later training:** `ProjectThenHealLinear` keeps the adapter update in the plane perpendicular to `D`. AdamW's per-coordinate scaling tilts a gradient-masked update back out of that plane. Panels marked DATA are drawn from real files; panels marked SCHEMATIC are sketches whose labels quote measured values.
```

## What You Can Do

- **Inspect weights without running the model.** `secant inspect` lists tensors and their storage layout. `secant read` measures matrix structure and can map leading directions to vocabulary scores. These scores describe static weights, not generated answers.
- **Choose and size an edit.** Capture differences between prompt groups, rank candidate layers with `secant analyze`, and create a plan with `secant compose`. Target-Shift Calibration sizes an edit for one reference activation; held-out prompts test whether it changes behavior as intended.
- **Edit a checkpoint and check its bytes.** `secant patch` handles supported floating-point weights. `secant quant patch` handles `Q8_0`, a compressed format that stores 32 weights per block. Both write a copy by default. `secant verify-patch` checks that the header and tensors outside your change list stayed identical.
- **Constrain later fine-tuning.** `secant.nn.ProjectThenHealLinear` removes the protected components from the adapter on every forward pass. `secant heal` applies that projection once to an existing adapter. Verify again after merging or changing storage precision.
- **Check unusual weight values.** `secant sanitize` scans supported payloads for non-finite values and other configured findings. A clean scan is limited to those checks; it is not a runtime safety certificate.

## See It Run

This is a real terminal recording, not a drawing: `secant patch` edits two tensors in a copy of the 3.4 GB `SmolLM2-1.7B-Instruct` checkpoint, and the output hash matches the one recorded when the original experiment ran. The roughly 30 s spent copying and hashing is cut from the recording, and bash `time` prints the real elapsed time on screen. The [quickstart](quickstart.md) shows all six steps, from `inspect` to `verify-patch`, with their tapes and text transcripts.

```{figure} assets/secant-patch.gif
:name: fig-index-vhs-patch
:alt: Terminal recording of secant patch editing two tensors of SmolLM2-1.7B-Instruct and printing the output SHA-256
:width: 100%

`secant patch` on a real checkpoint: 104,755 of 16,777,216 entries change in the `L15` MLP writer, and every tensor outside the plan stays byte-identical.
```

## What the Reported Experiments Show

Different experiments answer different questions. The paper reports results across checkpoints from `0.8B` to `72.7B` parameters, but the two largest stress tests used a single checkpoint shard. No single experiment establishes every capability across that range.

| Experiment | Result | What it establishes |
|---|---|---|
| Reference-activation calibration, 15 models (`0.8B` to `32.8B`) | Maximum reported `float64` shift error about `7.99e-15` | The edit size matches the requested shift at the reference input. Behavioral benchmarks were `NOT_RUN` in this sweep. |
| Public benchmarks after the edit, 14 models (`0.8B` to `32.8B`) | Calibrated `1.0x` edits: 0 of 70 model-by-benchmark cells change significantly (exact McNemar, Holm). Pooled shifts are `-2.0` to `+2.1` points | Retention on first-N items of GSM8K, MATH-500, MMLU-Pro, HumanEval, and a 30-trace crash-triage set. Equal-size random edits behave the same way, so this shows the edit does little harm. It does not show a benefit. |
| Reasoning-loop test, `Qwen3.5-4B`, 30 crash traces | Completion within the token limit rose from `0/30` to `21/30`, then `30/30` at a higher loop-edit dose | A measured improvement on this prompt set. Completing an answer is separate from identifying the correct root cause (`19/30` at the higher dose). |
| LoRA comparison, 15 models | Gradient-projected AdamW retained `0.33%` to `1.67%` relative leakage; Project-Then-Heal measured `1.78e-10` to `8.67e-10` | A local constraint on the adapter's weight update, measured with the same metric in all three arms. |
| C/C++ root-cause benchmark, `Qwen2.5-Coder-32B-Instruct` | Correct cases rose from `2/30` to `12/30` (reported paired `p = 0.00195`) | Performance on the reported suite, not a count of new vulnerabilities discovered. |
| Live fine-tuning, `Qwen3.5-0.8B` and `SmolLM2-1.7B` | Similar held-out perplexity to unconstrained LoRA, with lower local leakage | Evidence from two short runs. Both Qwen arms overfit the small training batch by step 60. |
| RMS-relative dosing, `Qwen2.5-Math-7B-Instruct`, 14 arms | The earlier unit shift was `1.34%` of the writer's output norm. At the full norm, composed edits matched a norm-matched random control at every dose. The Project-Then-Heal arm moved `GSM8K` from `87` to `93` of 100 (unadjusted `p = 0.031`, Holm `1.0`) | How to size an edit so it is large enough to matter. The `GSM8K` gain does not survive correction, and no control separates the composition from the healing step. |
| Planted rank-1 trojan, `SmolLM2-1.7B` and `Qwen3.5-4B`, 24 static audits | Nothing at doses `0.10%` to `4%`. The trigger token is rank 1 or 2 in the vocabulary projection at `8%`, and the automatic sibling flag fires at `12%`. 0 false positives on the unedited bases | When `secant audit-trojan` sees one rank-1 edit in one layer: once its singular value reaches the trained $\sigma_1$. The doses above `4%` were added post-hoc. |
| Quantized scale sanitize, `Q8_0` and `Q4_K` | The three planted hostile scales are each found exactly once and repaired. Clean quantizer output also fails the default gate, on `0.20%` (`Q8_0`) to `0.53` per super-block (`Q4_K`) subnormal scales | Read the per-kind counts, not the pass or fail status. |
| Static atlas, 46 local checkpoints | 429 `read` receipts, `2,619` tensors, `0` non-finite; SafeTensors and GGUF copies agree bit-for-bit | What unedited weights look like to these readers. |

```{figure} assets/benchmark-retention-14models.svg
:name: fig-index-benchmark-retention
:alt: Small multiples of paired change from base for 14 models on GSM8K, MATH-500, MMLU-Pro, HumanEval and ASan-30, comparing calibrated edits with norm-matched random edits
:width: 100%

What a calibrated `B2 + B3` edit does to public benchmarks on 14 models. Each dot is a paired change from the same model's unedited score on the same items. Blue is the calibrated edit; orange is a random direction with the same columns and the same Frobenius norm. Almost every dot sits near zero, and blue and orange overlap. Pooled over all 14 models, each benchmark moves by about 2 points or less. The full method and every number are in the [paper, Section 4.4](#sec-paper-public-benchmarks).
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

```{figure} assets/trojan-detection-phase-transition.svg
:name: fig-index-trojan-phase
:alt: Planted rank-1 trojan detection against relative Frobenius dose on SmolLM2-1.7B and Qwen3.5-4B
:width: 100%

The audit side of the same tools. A rank-1 edit planted in one layer stays invisible until its singular value reaches that layer's trained $\sigma_1$, at `6.57%` (Qwen) and `8.65%` (SmolLM2) of the weight norm. Past that point the trigger token appears in the vocabulary projection, and at `12%` the automatic sibling test flags the layer. Method and tables: [payload checks, Section 5.1](#sec-trojan-dose-curve).
```

## Where to Start

- [Quickstart](quickstart.md): build the tools, try a small local example, and watch the retained terminal recordings.
- [Theory and calibration](theory-and-calibration.md): understand direction selection, non-overlapping edits, and reference-based edit sizing before the equations.
- [Editing quantized weights](quantized-surgery.md): understand which compressed blocks can change and why `Q4_K` patching is refused.
- [Project-Then-Heal](project-then-heal.md): understand the training constraint, its numerical limits, and the measured results.
- [CLI, Python, and C reference](cli-and-python-reference.md): find command templates, API signatures, exit codes, and receipt fields.
- [Payload checks](payload-security.md): learn what the static scanner checks and what each result means.
- [Scientific paper](PAPER.md): read the derivations, experimental tables, related work, and limitations together.

Secant builds on weight-space programming, representation engineering, and constrained model editing [@rikiyat2026wsp; @zou2023representation; @zou2024circuitbreakers; @arditi2024refusal]. The paper separates prior work from the methods evaluated here.
