# Secant: Direct Weight-Space Surgery, Payload Auditing, and Subspace-Locked LoRA

**Secant** is a Rust and PyTorch toolkit for inspecting, editing, and auditing language-model weights directly on disk. Its Rust CLI and C-ABI library operate on `.safetensors` and `.gguf` checkpoints via memory-mapped I/O without loading full models into GPU RAM, while its Python package (`secant.nn`, `secant.capture`) captures contrastive activation subspaces and constrains PyTorch LoRA fine-tuning.

Standard model adaptation and safety pipelines leave three practical gaps:
1. **Editing weights without re-quantizing whole checkpoints:** Modifying a single behavior in a multi-gigabyte checkpoint typically requires loading full `BF16` weights into memory, applying an update, and re-quantizing the entire file. Secant calibrates sparse rank-1 updates against a layer's reference activation and patches only the intersecting 34-byte `Q8_0` blocks or floating-point columns in place, leaving over `98%` of the file byte-for-bit untouched and verified by pre/post SHA-256 receipts.
2. **Preventing post-edit LoRA fine-tuning from erasing surgical edits:** When a patched model is fine-tuned with LoRA under AdamW, AdamW's coordinate-wise second-moment scaling (`sqrt(v_hat_t) + eps`) rotates even orthogonalized gradients back into the edited subspace. **`ProjectThenHealLinear`** enforces a twice-applied `float64` orthogonal projection (`B_locked = P_perp(P_perp(B_tilde))`) inside the forward pass, keeping adapter updates in the left nullspace of the protected directions down to the working-dtype precision floor (`~1e-18` in `float64`, `~1e-10` in `float32`).
3. **Auditing numerical payload invariants ("Boundary 3") before native kernels execute:** Structural container validators verify header offsets and byte lengths without inspecting payload values. `secant sanitize`, `secant probe`, and `secant audit-trojan` audit weight matrices, quantized block scales (`Rule B3-2A` permitting legitimate `Q8_0`/`Q4_K` FP16 subnormals while catching `NaN`/`Inf` headers), divisor scales (`Rule B3-2B`), integer routing tables (`Rule B3-3`), and spectral rank-1 anomalies without a GPU.

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
| Reference-activation calibration, 15 models (0.8B to 32.8B) | Maximum reported float64 shift error about 7.99e-15 | The edit size matches the requested shift at the reference input. Behavioral benchmarks were NOT\_RUN in this sweep. |
| Public benchmarks after the edit, 14 models (0.8B to 32.8B) | Calibrated 1.0x edits: 0 of 70 model-by-benchmark cells change significantly (exact McNemar, Holm). Pooled shifts are -2.0 to +2.1 points | Retention on first-N items of GSM8K, MATH-500, MMLU-Pro, HumanEval, and a 30-trace crash-triage set. Equal-size random edits behave the same way, so this shows the edit does little harm. It does not show a benefit. |
| Reasoning-loop test, Qwen3.5-4B, 30 crash traces | Completion within the token limit rose from 0/30 to 21/30, then 30/30 at a higher loop-edit dose | A measured improvement on this prompt set. Completing an answer is separate from identifying the correct root cause (19/30 at the higher dose). |
| LoRA comparison, 15 models | Gradient-projected AdamW retained 0.33% to 1.67% relative leakage; Project-Then-Heal measured 1.78e-10 to 8.67e-10 | A local constraint on the adapter's weight update, measured with the same metric in all three arms. |
| C/C++ root-cause benchmark, Qwen2.5-Coder-32B-Instruct | Correct cases rose from 2/30 to 12/30 (reported paired p = 0.00195) | Performance on the reported suite, not a count of new vulnerabilities discovered. |
| Live causal-LM healing, 8 models (0.8B to 14B) | Two early runs (Qwen3.5-0.8B, SmolLM2-1.7B) plus six H100 models (Qwen3-4B, Qwen2.5-Math-7B, Qwen2.5-Coder-7B, R1-Llama-8B, Qwen3-8B, Qwen3-14B). H100 relative leakage: 3.18e-17 to 5.07e-17 | Real-text healing with distinct metrics: early runs report maximum absolute leakage; H100 reports relative leakage. The 24,000 benchmark rows belong to the six-model RMS study. See [live-healing coverage](#sec-paper-live-healing). |
| RMS-relative dosing, 6 models (4B to 14B), 24,000 rows | σwriter spans 5.85 to 78.15 (13.4x), so a fixed unit shift ranges from 1.28% to 17.10% of the writer's output norm. At α = 1.0, unhealed composed edits matched norm-matched random controls across all 6 models. The ProjectThenHealLinear arm (3.18e-17 to 5.07e-17 leakage) moved GSM8K 87 -&gt; 93 on Qwen2.5-Math-7B-Instruct (p = 0.031) and HumanEval 36 -&gt; 42 on Qwen2.5-Coder-7B-Instruct (p = 0.238, Holm 1.0) | How to size an edit against σwriter so doses are comparable across layers and models. The healed gains do not survive multiple-comparison correction, and no control separates the composition from the healing step. |
| Paired ASAN-120 vulnerability & benign twins, 9 models (0.8B to 32.8B) × 7 arms (7,560 rows) | 60 vulnerable C/C++ snippets paired with 60 patched benign twins (60/60 compiler/sanitizer matrix PASS). Non-reasoning models (0.8B-8B) collapse to trivial one-sided predictors (BA = 0.475..0.533, RC &lt;= 0.067). On the 4 &lt;think&gt; reasoning models (8B-32.8B), composed\_rms\_1p0x beats random\_rms\_1p0x 4/4 on balanced accuracy (6W / 1L / 2T overall, sign-test p = 0.125, mean dBA = +0.017) and lifts strict root-cause recall RC from 0.183 to 0.250 on R1-Llama-8B and from 0.450 to 0.550 on Qwen3-14B | Why vulnerability benchmarks need patched benign twins (TNR / pair) alongside strict root-cause (RC = verdict + class + sink) scoring: benign controls expose 93-98% VULNERABLE rubber-stamping on SmolLM2-1.7B and Llama-3.1-8B (TNR &lt;= 0.067). No single-model McNemar test survives Holm correction across the 117-test family (min p\_Holm = 0.2197). |
| Planted rank-1 trojan, SmolLM2-1.7B and Qwen3.5-4B, 24 static audits | Nothing at doses 0.10% to 4%. The trigger token is rank 1 or 2 in the vocabulary projection at 8%, and the automatic sibling flag fires at 12%. 0 false positives on the unedited bases | When secant audit-trojan sees one rank-1 edit in one layer: once its singular value reaches the trained σ₁. The doses above 4% were added post-hoc. |
| Quantized scale & Boundary-3 sanitize, Q8\_0..Q6\_K & .safetensors | Rule B3-2A passes 10 / 10 production checkpoints (2,194 total tensors, 912 quantized tensors, 186,870,528 blocks, 272,730 subnormal Q8\_0 scales, and 8,992,791 subnormal Q4\_K header scales; receipt `399143323a...`, plus an earlier unreceipted 2,723-tensor / 16,139-subnormal dogfood run down to 4.55e-05) with 0 false positives, while catching 100% of planted NaN/Inf block headers ((bits &amp; 0x7C00) == 0x7C00), near-zero divisor scales (Rule B3-2B, &lt; 1e-12f), and OOB routing indices (Rule B3-3) | Separating multiplicative block scales (B3-2A) from divisor scales (B3-2B) eliminates false positives on real quantized checkpoints while blocking non-finite and out-of-range kernel inputs. |
| Static atlas, 46-file local inventory (2,619 analyzed weight matrices: 2,574 across 14 full reads and 45 across 16 scoped variants) | 429 read receipts, 2,619 tensors, 0 non-finite; SafeTensors and GGUF copies agree bit-for-bit | What unedited weights look like to these readers. |

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
