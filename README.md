<div align="center">

# Secant

**Weight-Space Programming, Direct Quantized Surgery, and Subspace-Locked LoRA (`Project-Then-Heal`)**

[![Documentation](https://img.shields.io/badge/Docs-professor--moody.github.io%2Fsecant-0A66C2?style=flat-square)](https://professor-moody.github.io/secant/)
[![Technical Paper](https://img.shields.io/badge/Paper-MyST_%2F_Quarto-8A2BE2?style=flat-square)](https://professor-moody.github.io/secant/PAPER)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange?style=flat-square)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)](https://www.python.org/)

**[Documentation](https://professor-moody.github.io/secant/)** · **[Quickstart](https://professor-moody.github.io/secant/quickstart)** · **[Theory & Calibration](https://professor-moody.github.io/secant/theory-and-calibration)** · **[Quantized Surgery](https://professor-moody.github.io/secant/quantized-surgery)** · **[Project-Then-Heal](https://professor-moody.github.io/secant/project-then-heal)** · **[Technical Paper](https://professor-moody.github.io/secant/PAPER)**

<br/>

<img src="docs/assets/secant-pipeline.svg" alt="Secant End-to-End Weight-Space Engine" width="100%" />

</div>

---

## Why Secant?

In [*"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, Nov 2026), Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab demonstrated that frontier models can inspect raw transformer weights and edit behavioral circuits directly, such as suppressing `<think>` reasoning loops in `Qwen3.5-0.8B` by modifying `0.016%` of its parameters. In their **"What Comes Next?"** section, they called for:

> *"Turning what currently seems like black magic into a rigorous, reproducible engineering discipline"* and *"auditing without evaluation (e.g., backdoors directly from weights)."*

**We took this on and decided to see what we could do.**

**Secant** is a standalone Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python library (`secant.nn.ProjectThenHealLinear`) for deterministic weight-space reading, scale-calibrated multi-subspace surgery (exercised on 17 checkpoints from `0.8B` to `72.7B` in three test regimes, 9 model families), direct in-place `Q8_0` / `Q4_K` GGUF block editing, static tensor payload security auditing, and subspace-locked LoRA fine-tuning.

---

## Our 7 Empirical Findings in Plain Words

1. **"Safe" model files can carry values that crash or corrupt an unchecked C++ kernel (`secant sanitize`, `secant probe`):** Switching from Python `pickle` files to `.safetensors` stopped runnable Python code in model files, but C++ and GPU math kernels still blindly trust the numbers stored inside the weights to pick Mixture-of-Experts (MoE) routes, divide quantization scales, sort words, and index into memory. A `.safetensors` file with `NaN` in a router bias or a negative integer in a routing table passes every header check, and a kernel that does not check that invariant can then crash or corrupt C++ memory (shown with four synthetic kernels, not a named runtime; because `x86-64` and Apple Silicon `ARM64` convert `NaN` to integers differently in hardware, the same synthetic kernel and input ran quietly on `ARM64` and crashed on `x86-64`).
2. **Small reasoning models often fail on three stacked habits at once (`secant.capture`, `secant compose`):** On C++ crash triage, `Qwen3.5-0.8B` and `4B` get stuck in `<think>` loops (`B1`), falsely refuse benign crash dumps once the loop is opened (`B2`), and blame the crash line (`memcpy`) instead of the caller function that skipped the bounds check (`B3`). Fixing only one habit fails because the next habit in line blocks the answer; `secant compose` makes all three directions perpendicular (orthogonal) and edits them together.
3. **Why searching inside `<think>` for keywords is a mirage (The `93.3%` Copy-the-Input Control):** We did not lose a finding here; we caught a benchmarking trap *before* publishing that proved our real `0%` $\to$ `100%` result is genuine. A naive keyword search on `Qwen3.5-4B`'s unfinished `<think>` rambling scored `86.7%` (`26/30`), but feeding the **raw user prompt itself** (with **zero AI model**) into that same keyword scorer got **`93.3%` (`28/30`)** because every function in the call stack is already listed in the prompt! Under a **strict post-`</think>` scorer** that requires closing `</think>` and naming one culprit function over all the decoys, copying the prompt scores **`0/30` (`0%`)**, the unedited `4B` model scores **`0/30` (`0%`)**, and `Secant`'s calibrated edit (`a15`: `B1` at `1.5x`, `B2` and `B3` at `1.0x`) reaches **`30/30` (`100%`)** closed and **`23/30` (`76.7%`)** exact culprit function.
4. **Why copying an edit ratio across model sizes fails (`0/30`), and how Target-Shift Calibration reached `30/30` on `Qwen3.5-4B`:** A fixed Frobenius-ratio edit ($\gamma = 0.30$) carried over from `0.8B` closed **`0/30`** `<think>` loops on `4B`, because wider layers have different activation scales. **Target-Shift Calibration** solves for the weight step that produces a chosen activation shift at the layer's measured scale: on `Qwen3.5-4B` it closed `0/30`, `21/30` and `30/30` with the `B1` loop edit at $0.5\times$, $1.0\times$ and $1.5\times$ (`B2` and `B3` held at $1.0\times$). It was run end to end on `Qwen3.5-4B` and for the `B3` edit on `SmolLM2-1.7B`; the 14 large-model H100 runs used a fixed-ratio edit instead.
5. **You can audit vocabulary concepts and edit `.safetensors` or `.gguf` files directly on disk without PyTorch (`secant inspect`, `secant read`, `secant patch`, `secant verify-patch`):** `secant read` memory-maps tensors directly from disk, extracts their strongest directions in `float64`, and projects them onto the vocabulary table so you can see the exact words a layer promotes or suppresses without a GPU. `secant patch` edits only the target matrices in place and proves via SHA-256 that the other `98%+` of the file is bit-identical.
6. **8-bit (`Q8_0`) GGUF models can be edited directly on disk, while 4-bit (`Q4_K`) superblocks straddle head boundaries (`secant quant patch`, `secant quant witness`):** Each `Qwen3.5` `ssm_out` value head spans `128` columns, which divides evenly into four 32-weight `Q8_0` blocks ($128 = 4 \times 32$), enabling direct 34-byte block surgery (`98.5%` refusal removal, `10/10` eval agreement with edit-then-quantize). A 256-weight `Q4_K` superblock spans two neighboring 128-column heads with a shared scale, so `secant` blocks unsafe in-place `Q4_K` straddling.
7. **An unconstrained LoRA adapter writes back into edited directions; `ProjectThenHealLinear` keeps it out to rounding level (`secant heal`, `secant verify-subspace`):** `ProjectThenHealLinear` projects the LoRA output factor out of the protected directions on every forward pass ($B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$). In live fine-tuning after real surgery (`Qwen3.5-0.8B`, 60 steps; `SmolLM2-1.7B`, 20 steps) both arms reached the same training loss, and the largest write into the protected directions was $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$ (`float64`) for `ProjectThenHealLinear` against $4.64 \times 10^{-3}$ to $1.05 \times 10^{-2}$ unconstrained. In a separate H100 stress test on 14 checkpoints (`7.2B` to `72.7B`) whose training target deliberately points into the protected directions, locked leakage stayed at $4.82 \times 10^{-10}$ to $1.08 \times 10^{-9}$ (`float64`) while the unconstrained adapter put `8.42%` to `52.55%` of its norm there (a property of that stress task, not a measured erasure in real training).

---

## What Secant Does

| Capability | CLI / Python | What It Solves |
|---|---|---|
| **Weight-Space Reading** | `secant inspect`<br>`secant read` | Deterministic non-FMA `float64` SVD over `.safetensors` and `.gguf` tensors with direct vocabulary unembedding projection (`--vocab-tensor`). |
| **Calibrated Multi-Subspace Surgery** | `secant compose`<br>`secant patch`<br>`secant verify-patch` | Two-pass `float64` Gram-Schmidt orthogonalization + **Target-Shift Calibration** (`beta = \|Delta_target\| / \|\|h_S\|\|_2^2`), with zero-copy `mmap` patching and bit-exact verification of all untouched bytes. |
| **Direct Quantized Surgery** | `secant quant patch`<br>`secant quant witness` | Edits 34-byte `Q8_0` blocks directly inside compressed `.gguf` files (`untouched_blocks_bit_identical: true`, `0.18%` to `0.92%` per-tensor Frobenius residual vs. `BF16` edit-then-quantize) and proves $\Pi_{128}$ block-grid equivariance. |
| **Subspace-Locked LoRA (`Project-Then-Heal`)** | `secant.nn.ProjectThenHealLinear`<br>`secant heal`<br>`secant verify-subspace` | Prevents post-surgical LoRA + AdamW fine-tuning from intruding into excised weight subspaces, with measured writer leakage about $10^{-17}$ (`float64`) in live fine-tuning and below $1.1 \times 10^{-9}$ (`float64`) at step 60 in the H100 stress test. |
| **Payload Security Auditing** | `secant sanitize`<br>`secant probe` | Audits and sanitizes structurally valid `.safetensors` and `.gguf` checkpoints for `NaN`/`Inf`/subnormal floats, out-of-bounds MoE router IDs, and alignment-gap data smuggling without running a forward pass. |

---

## Quickstart

```sh
# 1. Build the Rust CLI & C-ABI static/shared libraries
cargo build --release --workspace

# 2. Install the Python package (PyTorch ProjectThenHealLinear + contrastive capture)
pip install -e python/
```

### Read, Patch, and Verify a Checkpoint in 3 Commands

```sh
# Read top singular directions and project into vocabulary tokens without a GPU
secant read --model model.gguf --tensor "blk.15.ffn_down.weight" --vocab-tensor "token_embd.weight"

# Compose an orthogonal multi-subspace plan and patch a copy (secant patch: BF16/F16/F32; secant quant patch: Q8_0 only)
secant compose --profile profile.json --vectors vec/ --target-shift -4.0 --out plan.json --json
secant quant patch --base model-q8_0.gguf --plan plan.json --out model-patched-q8_0.gguf --json

# Lock subsequent LoRA fine-tuning away from the edited subspace
secant heal --lora lora_adapter.safetensors --vectors vec/ --out lora_locked.safetensors --json
```

---

## Headline Empirical Results (17 Checkpoints, `0.8B` to `72.7B`, Three Regimes)

Full derivations and the per-regime caveats are in the **[Documentation](https://professor-moody.github.io/secant/)** and **[Technical Paper](https://professor-moody.github.io/secant/PAPER)**. Every number below is copied from a banked receipt.

### The three test regimes

Every result on this site comes from one of three regimes. They measure different things and use different metrics, so we never mix them in one column.

| Regime | Checkpoints | What was edited | What "healing" means | Leakage metric |
|---|---|---|---|---|
| **A. Live LM fine-tuning after real surgery** | `Qwen3.5-0.8B` (60 AdamW steps, CPU), `SmolLM2-1.7B-Instruct` (20 AdamW steps, ROCm GPU) | Real composed edits, verified on disk with `verify-patch` | LoRA trained on real causal-LM completions on the edited model | $\max \lvert D \Delta W\rvert $ per module, the same absolute metric for both arms |
| **B. Generative `<think>` benchmark** | `Qwen3.5-4B` | Calibrated `B1+B2+B3` composition with the `B1` edit at $0.5\times$, $1.0\times$, $1.5\times$ (`B2`, `B3` at $1.0\times$), plus unedited, single-family and random-direction arms | No healing run (`NOT_RUN`) | n/a |
| **C. H100 stress test** | 12 full models (`7.2B` to `32.8B`) plus 2 single-shard runs (`70.6B`, `72.7B`) | A **fixed-ratio** rank-1 edit ($\gamma = 0.045\,\Vert W\Vert _{\mathrm{F}}$), compared with an equal-norm random direction | 60 AdamW steps toward a **synthetic adversarial target** $Y_{\text{target}} = Y + 1.5\,v_{B2}$ that lies inside the protected subspace, on the **unedited** `down_proj` | Unconstrained: $\Vert DB\Vert _{\mathrm{F}} / \Vert B\Vert _{\mathrm{F}}$ (relative). `ProjectThenHealLinear`: $\max \lvert D B_{\text{locked}}\rvert $ (absolute). The two are **not** comparable as a ratio |

What regime C does and does not show:
- It shows the lock holds under a gradient that points straight into $\operatorname{span}(D)$: `ProjectThenHealLinear` leakage stayed at rounding level on all 14 checkpoints.
- Because the target lies inside $\operatorname{span}(D)$, the locked arm **cannot** fit it. Its step-60 loss sits at the irreducible floor $2.25 / d_{\text{model}}$ (between `0.007%` and `1.3%` above it on every checkpoint), while the unconstrained arm fits it. So regime C says nothing about learning speed or quality, and it is not evidence of "zero learning penalty".
- The unconstrained percentage is the fraction of the adapter's `B` norm inside $\operatorname{span}(D)$. The task requires that, so it is not a measured "erasure of the surgery" in real training.
- The edit-versus-random ratio compares a direction chosen from activations against a random unit direction. A random direction's projection is expected to be about $1/\sqrt{d_{\text{model}}}$ of an aligned one, so ratios of $10^2$ to $10^3$ are a geometric sanity check, not evidence of behavioural change.
- The regime does **not** test the AdamW gradient-projection argument in [Project-Then-Heal](docs/project-then-heal.md): the unconstrained arm has no projection at all, so its leakage comes from the task gradient.


**Regime B, `Qwen3.5-4B` dose-response (`n = 30`, one greedy run per arm):**

| Arm (`Qwen3.5-4B`, `n = 30`, `1,536` token cap) | Summed `B1` target shift | `</think>` Closed | Strict Bug Class | Strict Culprit Frame | Strict Root Cause | Mean Tokens |
|---|---:|---:|---:|---:|---:|---:|
| **`base`** / **`rand1`** / **`rand2`** (equal-norm random) | n/a | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`g030`** (uncalibrated $\gamma = 0.30$) | n/a | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`a05`** (`B1` at $0.5\times$; `B2` and `B3` held at $1.0\times$) | `-3.691` | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`b1only`** ($\alpha = 1.0\times$, loop edit only) | `-7.382` | `17/30` (`56.7%`) | `15/30` (`50.0%`) | `13/30` (`43.3%`) | `11/30` (`36.7%`) | `1,044` |
| **`cal738`** ($\alpha = 1.0\times$, composed `B1+B2+B3`) | `-7.382` | **`21/30` (`70.0%`)** | **`18/30` (`60.0%`)** | **`15/30` (`50.0%`)** | **`14/30` (`46.7%`)** | `954` |
| **`a15`** (`B1` at $1.5\times$; `B2` and `B3` held at $1.0\times$) | **`-11.073`** | **`30/30` (`100.0%`)** | **`23/30` (`76.7%`)** | **`23/30` (`76.7%`)** | **`19/30` (`63.3%`)** | `789` |

Exact two-sided McNemar tests against `base`: `cal738` closure $p = 9.54 \times 10^{-7}$ (21 vs. 0 discordant), `a15` closure $p = 1.86 \times 10^{-9}$ (30 vs. 0). `a15` vs. `cal738`: closure $p = 0.0039$, frame $p = 0.0078$, bug class $p = 0.125$ (not significant). In both the `a05` and `a15` arms only the `B1` edits (`B1a`, `B1b`) were scaled; `B2` and `B3` were held at their `cal738` ($1.0\times$) values in every arm (`a05`, `a10` and `a15` plans checked edit by edit).

**Regime A, live LM fine-tuning after real surgery.** Both arms train on real completions after a verified edit. Same metric for both arms.

| Model | Steps | Edit evidence | `ProjectThenHealLinear` $\max\lvert D \Delta W\rvert$, `float64` / merged `F32` | Unconstrained $\max\lvert D \Delta W\rvert$ | Held-out perplexity, PTH vs. unconstrained |
|---|---:|---|---:|---:|---|
| `Qwen/Qwen3.5-0.8B` | 60 (CPU) | Composed `B1+B2+B3` GGUF, `verify-patch` exit 0 | `9.87e-18` / `2.49e-10` | `1.05e-02` | step 20: `5.435` vs. `5.423`; step 60: `12.106` vs. `12.357` (both overfit after step 20) |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 20 (ROCm GPU) | `B2` projector at L14 `o_proj`: held-out shift `-2.436` vs. random `-0.0359` (`67.9x`). `B3` calibrated at L15 `down_proj`: `+3.952` vs. `+0.179` (`22.1x`) | `5.21e-18` / `1.46e-10` | `4.64e-03` | step 20: `21.260` vs. `21.255` |

After rounding merged weights to `BF16`, both arms show leakage of order $10^{-4}$ to $10^{-3}$ (`SmolLM2` PTH `6.8e-4` to `9.5e-4`), so the guarantee does not survive a `BF16` merge; `secant heal` stores the adapter in `F32`/`F64` for that reason.

**Regime C, H100 stress test (fixed-ratio edit, adversarial heal target).** Read the regime notes above before comparing columns. Edit layers are the layers of peak $\mathrm{SNR}_{\text{proj}}$ for each family. Shift values are at dose $1.0\times$.

| Model | Params | Layers ($d_{\text{model}}$) | `B2` edit: shift vs. random (ratio, layer) | `B3` edit: shift vs. random (ratio, layer) | Unconstrained $\Vert DB\Vert _{\mathrm{F}}/\Vert B\Vert _{\mathrm{F}}$ | PTH $\max\lvert D B_{\text{locked}}\rvert$, `float64` / merged `F32` |
|---|---:|---:|---:|---:|---:|---:|
| `mistralai/Mistral-7B-Instruct-v0.3` | `7.2B` | `32` (`4096`) | `+0.2567` vs. `-0.000387` (`663.9x`, L12) | `+3.212` vs. `-0.0325` (`98.9x`, L16) | `46.79%` | `7.73e-10` / `3.87e-10` |
| `Qwen/Qwen2.5-7B-Instruct` | `7.6B` | `28` (`3584`) | `+4.654` vs. `+0.0300` (`155.1x`, L10) | `+20.82` vs. `-0.157` (`132.7x`, L18) | `31.34%` | `7.55e-10` / `6.88e-09` |
| `meta-llama/Llama-3.1-8B-Instruct` | `8.0B` | `32` (`4096`) | `+3.539` vs. `-0.0200` (`176.8x`, L24) | `+8.888` vs. `-0.252` (`35.3x`, L16) | `48.03%` | `7.62e-10` / `2.72e-09` |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `8.0B` | `32` (`4096`) | `+2.609` vs. `+0.00205` (`1,269.9x`, L20) | `+8.644` vs. `-0.0603` (`143.3x`, L16) | `52.55%` | `6.39e-10` / `2.13e-09` |
| `Qwen/Qwen3-8B` | `8.2B` | `36` (`4096`) | `+2.187` vs. `-0.00996` (`219.7x`, L12) | `+120.8` vs. `+4.15` (`29.1x`, L24) | `39.53%` | `7.14e-10` / `5.93e-09` |
| `google/gemma-2-9b-it` | `9.2B` | `42` (`3584`) | `+19.08` vs. `+0.149` (`127.9x`, L14) | `+130.0` vs. `+1.49` (`87.2x`, L26) [d] | `41.05%` | `6.53e-10` / `8.50e-10` |
| `allenai/OLMo-2-1124-13B-Instruct` | `13.7B` | `40` (`5120`) | `+1.250` vs. `+0.0121` (`103.3x`, L16) | `+7.341` vs. `-0.0279` (`262.9x`, L20) [d] | `30.73%` | `1.08e-09` / `3.33e-09` |
| `microsoft/phi-4` [p] | `14.7B` | `40` (`5120`) | `-3.399` vs. `+0.0104` (`326.9x` by magnitude, L32) [s] [d] | `+30.76` vs. `+0.632` (`48.7x`, L24) | `8.42%` | `9.51e-10` / `1.70e-08` |
| `Qwen/Qwen3-14B` | `14.8B` | `40` (`5120`) | `+4.548` vs. `+0.0180` (`252.5x`, L16) | `+63.11` vs. `-0.128` (`492.8x`, L24) | `41.33%` | `5.77e-10` / `4.68e-09` |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `14.8B` | `48` (`5120`) | `+3.762` vs. `+0.0564` (`66.7x`, L16) | `+51.94` vs. `+0.801` (`64.9x`, L32) | `32.57%` | `6.11e-10` / `2.78e-09` |
| `google/gemma-2-27b-it` | `27.2B` | `46` (`4608`) | `+542.8` vs. `+3.90` (`139.0x`, L16) | `+5,152` vs. `-26.3` (`196.0x`, L28) [d] | `11.54%` | `7.16e-10` / `7.54e-10` |
| `Qwen/Qwen3-32B` | `32.8B` | `64` (`5120`) | `+18.89` vs. `+0.119` (`158.6x`, L48) | `+41.52` vs. `-0.0813` (`510.5x`, L40) | `30.75%` | `4.82e-10` / `8.29e-09` |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` [x] | `70.6B` | `80` (`8192`) | `+8.306` vs. `-0.0155` (`537.1x`, one L32 `down_proj`) | `NOT_RUN` | `20.24%` | `5.66e-10` / `3.98e-08` |
| `Qwen/Qwen2.5-72B-Instruct` [x] | `72.7B` | `80` (`8192`) | `+9.061` vs. `-0.0675` (`134.2x`, one L40 `down_proj`) | `NOT_RUN` | `36.94%` | `9.85e-10` / `1.58e-08` |

Notes:
- **[d] Dose response is not linear for every edit.** For 20 of the 24 full-model edits, raising the dose from $1.0\times$ to $1.5\times$ multiplied the shift by `1.49` to `1.50`. The exceptions are `phi-4` `B2` (`1.04`), `OLMo-2-13B` `B3` (`1.18`), `gemma-2-9b` `B3` (`1.04`), and `gemma-2-27b` `B3` (`1.07`).
- **[s] `phi-4` `B2` moved the wrong way.** The intended shift is positive; the measured shift is `-3.399` at $1.0\times$ and `-3.519` at $1.5\times$. The `326.9x` ratio is by magnitude only.
- **[p] The `phi-4` capture looks degenerate.** Its peak `B2` $\mathrm{SNR}_{\text{proj}}$ is `27,362.3` with $\mathrm{SNR}_{\text{trace}}$ `60.4` at L32 (and up to `909.6` at L16), against `83.0` to `199.4` and about `1` to `2` on the other full models. This pattern suggests an outlier-dimension artefact rather than a sharper behavioural direction. `phi-4` was run twice; the table uses run 1 (the rerun gives `8.70%` and `9.03e-10` / `1.74e-08`, so the LoRA runs are not bit-reproducible).
- **[x] Single-shard rows.** Each used one `.safetensors` shard and one `down_proj` tensor: no activation capture and no `B3` edit. The receipt does not record how the edit direction was chosen, and the producing script is not banked. The banked run log shows the first `72.7B` attempt failing and contains no `70.6B` run, so these two rows rest on the receipt alone.
- Parameter counts are from the model cards, not from the receipts.
- `ProjectThenHealLinear` leakage over all 60 steps (not only step 60) stayed within $4.82 \times 10^{-10}$ to $1.60 \times 10^{-9}$ (`float64`) and up to $6.98 \times 10^{-8}$ (merged `F32`, `phi-4` step 0).


---

## Documentation

The full scientific documentation is hosted at **[https://professor-moody.github.io/secant/](https://professor-moody.github.io/secant/)**:

1. **[Overview & Empirical Summary](https://professor-moody.github.io/secant/)**: Capabilities, $\alpha$-dose-response phase transition (`0% -> 70% -> 100%`), and cross-family table.
2. **[Quickstart & Live CLI Receipts](https://professor-moody.github.io/secant/quickstart)**: End-to-end CLI and Python examples with SHA-256 bracketed JSON receipts.
3. **[Mathematical Theory & Scale Calibration](https://professor-moody.github.io/secant/theory-and-calibration)**: Contrastive capture, `SNR_proj`, Gram-Schmidt orthogonalization, and Target-Shift Calibration.
4. **[Quantized Block Surgery (`Q8_0` & `Q4_K`)](https://professor-moody.github.io/secant/quantized-surgery)**: The $\Pi_{128}$ Quantization Commutation Theorem and direct 34-byte `Q8_0` block editing.
5. **[Project-Then-Heal Subspace-Locked LoRA](https://professor-moody.github.io/secant/project-then-heal)**: Why AdamW rotates gradients back into excised subspaces and how $B = P_\perp \tilde{B}$ locks writer leakage.
6. **[CLI, Python & C-ABI Reference](https://professor-moody.github.io/secant/cli-and-python-reference)**: Flags, exit codes, JSON schemas, `secant.nn`, and `secant.h` FFI symbols.
7. **[Container Geometry & Payload Security](https://professor-moody.github.io/secant/payload-security)**: Boundary-3 static auditing (`sanitize`, `probe`) for `.safetensors` and `.gguf` files.
8. **[Technical Paper](https://professor-moody.github.io/secant/PAPER)**: Complete self-contained paper with formal propositions and proofs.

---

## Citation & References

If you build on Secant, please cite both the foundational MIT CSAIL Weight-Space Programming work and Secant:

```bibtex
@misc{takehi2026weightspace,
  author       = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  title        = {Sparks of Weight-Space Programming: {LLMs} can natively read \& write neural network weights},
  year         = {2026},
  month        = nov,
  institution  = {MIT CSAIL},
  howpublished = {\url{https://rikiyat.github.io/blog/weight-space-programming/}}
}

@software{moody2026secant,
  author       = {Moody, Professor},
  title        = {Secant: Direct Multi-Subspace Weight Surgery, Quantized Block-Grid Equivariance, and Subspace-Locked LoRA},
  year         = {2026},
  url          = {https://github.com/professor-moody/secant}
}
```
