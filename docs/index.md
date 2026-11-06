# Secant: Inspect and Edit Model Weights

**Secant** reads, edits, and checks the numerical weights stored in language-model checkpoints. Its Rust command-line tool works with `.safetensors` and `.gguf` files. Its Python package supports **LoRA**, a fine-tuning method that learns a small adapter while keeping the original weights fixed.

The workflow has three parts: choose a direction in a layer's activations, make a measured weight edit along that direction, and check the result. A direction is a vector that captures a difference between two sets of prompts. It is a useful experimental signal, not a complete description of a behavior.

Secant also offers **Project-Then-Heal**: a LoRA layer that keeps adapter updates perpendicular to specified directions. This constraint applies at the selected layer's output. It does not guarantee that the whole model will preserve a behavior after fine-tuning.

```{figure} assets/secant-pipeline.svg
:name: fig-secant-pipeline
:alt: Secant workflow from inspecting weights through composing edits, patching files, and constraining LoRA updates
:width: 100%

Inspect the checkpoint, prepare an edit, write and verify the changed weights, then constrain any later LoRA training.
```

## What You Can Do

- **Inspect weights without running the model.** `secant inspect` lists tensors and their storage layout. `secant read` measures matrix structure and can map leading directions to vocabulary scores. These scores describe static weights, not generated answers.
- **Choose and size an edit.** Capture differences between prompt groups, rank candidate layers with `secant analyze`, and create a plan with `secant compose`. Target-Shift Calibration sizes an edit for one reference activation; held-out prompts test whether it changes behavior as intended.
- **Edit a checkpoint and check its bytes.** `secant patch` handles supported floating-point weights. `secant quant patch` handles `Q8_0`, a compressed format that stores 32 weights per block. Both write a copy by default. `secant verify-patch` checks that the header and tensors outside your change list stayed identical.
- **Constrain later fine-tuning.** `secant.nn.ProjectThenHealLinear` removes the protected components from the adapter on every forward pass. `secant heal` applies that projection once to an existing adapter. Verify again after merging or changing storage precision.
- **Check unusual weight values.** `secant sanitize` scans supported payloads for non-finite values and other configured findings. A clean scan is limited to those checks; it is not a runtime safety certificate.

## See It Run

This is a real terminal recording, not a drawing: `secant patch` edits two tensors in a copy of the 3.4 GB `SmolLM2-1.7B-Instruct` checkpoint, and the output hash matches the one recorded when the original experiment ran. The [quickstart](quickstart.md) shows all six steps, from `inspect` to `verify-patch`, with their tapes and text transcripts.

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
| Reasoning-loop test, `Qwen3.5-4B`, 30 crash traces | Completion within the token limit rose from `0/30` to `21/30`, then `30/30` at a higher loop-edit dose | A measured improvement on this prompt set. Completing an answer is separate from identifying the correct root cause (`19/30` at the higher dose). |
| LoRA comparison, 15 models | Gradient-projected AdamW retained `0.33%` to `1.67%` relative leakage; Project-Then-Heal measured `1.78e-10` to `8.67e-10` | A local constraint on the adapter's weight update, measured with the same metric in all three arms. |
| C/C++ root-cause benchmark, `Qwen2.5-Coder-32B-Instruct` | Correct cases rose from `2/30` to `12/30` (reported paired `p = 0.00195`) | Performance on the reported suite, not a count of new vulnerabilities discovered. |
| Live fine-tuning, `Qwen3.5-0.8B` and `SmolLM2-1.7B` | Similar held-out perplexity to unconstrained LoRA, with lower local leakage | Evidence from two short runs. Both Qwen arms overfit the small training batch by step 60. |

```{figure} assets/qwen4b-phase-transition.svg
:name: fig-qwen4b-phase-transition
:alt: Completion and attribution scores at different loop-edit strengths on 30 Qwen3.5-4B crash traces
:width: 100%

On this 30-prompt test, the calibrated edit completed 21 answers at the standard dose. Increasing only the loop-edit component (`B1`) to 1.5 times that dose completed all 30, with 23 correct culprit frames and 19 correct root causes. The other two edit components stayed at their standard dose.
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
