# Secant: Weight-Space Programming & Subspace-Locked LoRA Healing

**Secant** is an open-source Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python library (`secant`, `secant.nn.ProjectThenHealLinear`) for **deterministic Weight-Space Programming (WSP)**, **in-place quantized block surgery (`Q8_0`)**, and **Subspace-Locked LoRA Healing (`Project-Then-Heal`)** on `.safetensors` and `.gguf` (v2/v3) checkpoints.

Motivated by MIT CSAIL's release by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (*"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*, Nov 2, 2026) [@rikiyat2026wsp] alongside representation-engineering and null-space editing research [@zou2023representation; @zou2024circuitbreakers; @arditi2024refusal], Secant aims to turn direct weight surgery into a **reproducible, hash-bracketed engineering discipline**. It has been exercised on **17 model checkpoints (`0.8B` to `72.7B`)** from **9 model families** (`Qwen3.5`, `Qwen3`, `Qwen2.5`, `SmolLM2`, `Llama-3.1`, `Gemma-2`, `Mistral`, `OLMo-2`, `Phi-4`; the `DeepSeek-R1-Distill` checkpoints are Llama- and Qwen2.5-architecture distillations). Those 17 checkpoints were tested in **three different regimes**, and the numbers from one regime do not transfer to another; see [the regime summary](#regimes) below.

> *"Turning what currently seems like black magic into a rigorous, reproducible engineering discipline... and auditing without evaluation (e.g., backdoors directly from weights)."*
>
> Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, **Sparks of Weight-Space Programming ("What Comes Next?")**, MIT CSAIL (Nov 2, 2026). **We took this on and decided to see what we could do.**

---

## Our 7 Empirical Findings in Plain Words (and How `secant` Identifies or Fixes Each One)

Before diving into equations or CLI flags, here are our seven core findings in plain English, along with the `secant` command that identifies or fixes each issue:

1. **"Safe" model files can still crash C++ memory on the very first run:**
   - **In plain words:** Switching from Python `pickle` files to `.safetensors` stopped attackers from hiding runnable Python code inside a model file. However, once a model loads, C++ and GPU math kernels trust the numbers stored inside the weights to pick Mixture-of-Experts (MoE) routes, divide by quantization scales, sort tokens, and index into memory. A `NaN` ("Not a Number") or a negative number in a routing table passes every format-level safety check and can then crash or corrupt C++ memory on the first forward pass. Because Intel/AMD (`x86-64`) and Apple Silicon (`ARM64`) convert `NaN` to integers differently in hardware, the same file can run quietly on one machine and crash on the other.
   - **How `secant` identifies and fixes it:** Run `secant sanitize --model model.safetensors --receipt receipt.json --attempt-id audit-1 --out clean.safetensors` to scan every tensor on disk for `NaN`/`Inf` values, zero quantization scales, and out-of-range router indices and write a clean copy before any kernel loads it. Use `secant probe` to generate test files that check whether your inference engine survives all four failure classes.

2. **Small reasoning models often fail on three stacked reflexes at once (not just one):**
   - **In plain words:** When you ask a small reasoning model (`Qwen3.5-0.8B` or `4B`) to analyze a C++ crash report, three separate bad habits stack on top of each other: **(1)** it gets stuck in an endless `"Wait, let me double-check..."` loop inside `<think>` until it runs out of tokens (`B1`), **(2)** if you fix the loop so it finally finishes `<think>`, it sometimes refuses to answer because words like `"buffer overflow"` trigger a false-positive safety reflex (`B2`), and **(3)** when it does answer, it blames the line where the program crashed (`memcpy`) instead of the caller that forgot to check the buffer size (`B3`). Fixing only one habit fails because the next habit in line blocks the answer.
   - **How `secant` identifies and fixes it:** `secant.capture` measures the activation direction of each habit and scores every layer (`SNR_proj`) so you target the layer where the habit is sharpest. Then `secant compose` makes the directions mutually orthogonal so the edits do not interfere, and `secant patch` applies them in a single pass on disk.

3. **Why searching inside `<think>` for keywords is a mirage (the `93.3%` copy-the-input control), and how strict post-`</think>` scoring shows the `0%` $\to$ `100%` closure result is real:**
   - **In plain words:** When the unedited `Qwen3.5-4B` model got stuck inside `<think>` (`0/30` finished answers), a keyword search for the right function name anywhere in its output said `"Yes, 86.7% (26/30)"`. As a sanity check we fed the **user's input prompt itself** (no model output at all) into the same keyword scorer. The raw prompt scored **`93.3%` (`28/30`)**, higher than the model, because the crash report already lists every function in the call stack.
   - **What this proved:** You cannot grade a reasoning model by checking whether a keyword from the prompt appears in its output. So the benchmark uses a **strict post-`</think>` scorer** that only gives credit when the model **closes `</think>`** and commits to **one culprit function over all the decoys in the stack trace**. On that strict test: copying the input scores **`0/30`**, the unedited `Qwen3.5-4B` scores **`0/30`**, and the `Secant`-edited `Qwen3.5-4B` (`1.5x` dose) closes `<think>` on **`30/30`** cases, names the correct culprit frame on **`23/30` (`76.7%`)**, and gets the full strict root-cause verdict on **`19/30` (`63.3%`)**. Closing `</think>` is termination, not new knowledge: under the old whole-text scorer the same arm scores `22/30` against the unedited model's `26/30`.
   - **How `secant` identifies and fixes it:** Evaluate on held-out prompts and score strictly on post-`</think>` discriminative verdicts, never on raw substring overlap with the prompt.

4. **Why copying an edit ratio from a small model to a larger one fails (`0/30`), and how Target-Shift Calibration reached `30/30` on `Qwen3.5-4B`:**
   - **In plain words:** Many weight-editing scripts scale an edit as a fixed fraction of a weight matrix's norm. A fixed ratio ($\gamma = 0.30$) carried over from `Qwen3.5-0.8B` closed **`0 out of 30`** loops on `Qwen3.5-4B`, because a wider model has different activation scales. **Target-Shift Calibration** instead solves for the weight step that produces a chosen activation shift at the layer's measured activation scale. On `Qwen3.5-4B` the calibrated loop edit (summed target shift `-7.382`) closed `21/30` at dose $1.0\times$ and `30/30` at $1.5\times$; half the dose ($0.5\times$, `-3.691`) closed `0/30`.
   - **Scope:** Target-Shift Calibration was run end to end on `Qwen3.5-4B` (and for the `B3` edit on `SmolLM2-1.7B`). The 14 large-model H100 rows below did **not** use it: they used a fixed-ratio edit ($\gamma = 0.045\,\|W\|_{\mathrm{F}}$) as a geometric check.
   - **How `secant` identifies and fixes it:** In your `secant compose` profile, set `"calibration": {"mode": "target-shift", "target_shift": -7.3822}` (or scale with $\alpha = 1.5\times$ to `-11.073`).

5. **You can audit a model's vocabulary concepts and edit `.safetensors` or `.gguf` files directly on disk without loading PyTorch:**
   - **In plain words:** Normally, inspecting or editing a model requires loading gigabytes of weights into PyTorch, editing them, and re-converting the whole checkpoint. Instead, `secant` memory-maps the file, reads the strongest singular directions of any layer, and projects them onto the model's vocabulary so you can see which tokens a layer promotes or suppresses without running a prompt. When patching, it rewrites only the target tensors in place and verifies via SHA-256 and byte comparison that the header and every other tensor are unchanged.
   - **How `secant` identifies and fixes it:** Run `secant inspect --model <path>`, `secant read --model <path> --tensor <name> --vocab-tensor <embd> --detect-anomalies`, `secant patch`, and `secant verify-patch`.

6. **8-bit (`Q8_0`) GGUF models can be edited directly on disk, while 4-bit (`Q4_K`) models should be edited before quantization:**
   - **In plain words:** Quantized GGUF models compress weights into small blocks that share a scale. In `Qwen3.5`, each `ssm_out` value head spans `128` columns. Because `Q8_0` groups weights into blocks of `32`, exactly four `Q8_0` blocks fit inside one head (`4 x 32 = 128`). Direct `Q8_0` block surgery removed `98.5%` of the targeted refusal component (`0.5172` $\to$ `0.0079`). By contrast, a `Q4_K` superblock of `256` weights spans **two neighboring 128-column heads** with a shared scale, and re-encoding it in place left `67%` of the component (`0.346`).
   - **How `secant` identifies and fixes it:** Run `secant quant witness` to check block alignment and the quantization noise floor, and run `secant quant patch` on `Q8_0` GGUF files (`secant` refuses in-place `Q4_K` superblock straddling with `SuperblockStraddleError`).

7. **During LoRA fine-tuning after an edit, an unconstrained adapter writes into the edited directions; `ProjectThenHealLinear` keeps it out to rounding level:**
   - **In plain words:** Suppose you remove a habit from a model's weights and then fine-tune with standard LoRA. Nothing stops the adapter from writing back into the directions you edited. `ProjectThenHealLinear` is a drop-in LoRA layer that projects the adapter's output factor out of those directions on every forward pass ($B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$).
   - **What we measured (two regimes, not one):** In **live language-model fine-tuning** after real surgery (`Qwen3.5-0.8B`, 60 steps; `SmolLM2-1.7B`, 20 steps) both arms reached the same training loss (`0.3120` vs. `0.3096` on `0.8B`), and the largest write into the edited directions, $\max |D \Delta W|$, was $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$ (`float64`) for `ProjectThenHealLinear` against $4.64 \times 10^{-3}$ to $1.05 \times 10^{-2}$ for unconstrained LoRA. In a separate **H100 adversarial stress test** on 14 checkpoints (`7.2B` to `72.7B`), where the training target deliberately points into the protected directions, unconstrained LoRA put `8.42%` to `52.55%` of its adapter norm inside them; that number measures the stress test's pull, not erasure in real training.
   - **How `secant` identifies and fixes it:** In Python training, wrap linear layers with `secant.nn.ProjectThenHealLinear.from_linear(layer, rank=16, directions=D)`. For existing LoRA `.safetensors` adapters on disk, run `secant heal --lora adapter.safetensors --vectors D.npy --out healed_lora.safetensors` and verify with `secant verify-subspace`.

```{figure} assets/secant-pipeline.svg
:name: fig-secant-pipeline
:alt: Secant four-stage weight-space programming pipeline
:width: 100%

The four stages of Secant: (1) zero-eval weight reading and Boundary-3 payload sanitization, (2) Gram-Schmidt multi-subspace composition with Target-Shift Calibration, (3) zero-copy `mmap` and 34-byte `Q8_0` quantized block surgery, and (4) `ProjectThenHealLinear` subspace-locked LoRA training.
```

---

(regimes)=
## The Three Test Regimes

Every result on this site comes from one of three regimes. They measure different things and use different metrics, so we never mix them in one column.

| Regime | Checkpoints | What was edited | What "healing" means | Leakage metric |
|---|---|---|---|---|
| **A. Live LM fine-tuning after real surgery** | `Qwen3.5-0.8B` (60 AdamW steps, CPU), `SmolLM2-1.7B-Instruct` (20 AdamW steps, ROCm GPU) | Real composed edits, verified on disk with `verify-patch` | LoRA trained on real causal-LM completions on the edited model | $\max \lvert D \Delta W\rvert $ per module, the same absolute metric for both arms |
| **B. Generative `<think>` benchmark** | `Qwen3.5-4B` | Calibrated `B1+B2+B3` edits at doses $0$, $0.5\times$, $1.0\times$, $1.5\times$, plus single-family and random-direction arms | No healing run (`NOT_RUN`) | n/a |
| **C. H100 stress test** | 12 full models (`7.2B` to `32.8B`) plus 2 single-shard runs (`70.6B`, `72.7B`) | A **fixed-ratio** rank-1 edit ($\gamma = 0.045\,\Vert W\Vert _{\mathrm{F}}$), compared with an equal-norm random direction | 60 AdamW steps toward a **synthetic adversarial target** $Y_{\text{target}} = Y + 1.5\,v_{B2}$ that lies inside the protected subspace, on the **unedited** `down_proj` | Unconstrained: $\Vert DB\Vert _{\mathrm{F}} / \Vert B\Vert _{\mathrm{F}}$ (relative). `ProjectThenHealLinear`: $\max \lvert D B_{\text{locked}}\rvert $ (absolute). The two are **not** comparable as a ratio |

What regime C does and does not show:
- It shows the lock holds under a gradient that points straight into $\operatorname{span}(D)$: `ProjectThenHealLinear` leakage stayed at rounding level on all 14 checkpoints.
- Because the target lies inside $\operatorname{span}(D)$, the locked arm **cannot** fit it. Its step-60 loss sits at the irreducible floor $2.25 / d_{\text{model}}$ (between `0.007%` and `1.3%` above it on every checkpoint), while the unconstrained arm fits it. So regime C says nothing about learning speed or quality, and it is not evidence of "zero learning penalty".
- The unconstrained percentage is the fraction of the adapter's `B` norm inside $\operatorname{span}(D)$. The task requires that, so it is not a measured "erasure of the surgery" in real training.
- The edit-versus-random ratio compares a direction chosen from activations against a random unit direction. A random direction's projection is expected to be about $1/\sqrt{d_{\text{model}}}$ of an aligned one, so ratios of $10^2$ to $10^3$ are a geometric sanity check, not evidence of behavioural change.
- The regime does **not** test the AdamW gradient-projection argument in [Project-Then-Heal](project-then-heal.md): the unconstrained arm has no projection at all, so its leakage comes from the task gradient.

---

## Core Capabilities & Verification Coverage

| Capability / Subcommand | What It Does | Where it has been exercised | Banked evidence |
|---|---|---|---|
| **`secant inspect` & `secant read --detect-anomalies`** | `float64` singular spectrum, vocabulary token projection, and column-norm / spectral anomaly detection directly on disk. | `Qwen3.5-4B` (`.safetensors` and `.gguf`) and one 7B-class `.safetensors` shard via the Rust CLI; top-3 singular values of one `down_proj` tensor on the `70.6B` and `72.7B` shards. | `read` receipts record `input_unchanged: true` with pre/post SHA-256. |
| **`secant sanitize` & `secant probe`** | Scans `.safetensors` and `.gguf` payloads for `NaN`/`Inf` routers, zero scales, and OOB indices; generates Boundary-3 test fixtures. | `Qwen3.5-4B` and one 7B-class `.safetensors` shard; fixtures for the four kernel classes in the [Part I study](payload-security.md). | `sanitize` receipts record `input_unchanged: true` with pre/post SHA-256. |
| **`secant.capture` (`SNR_proj` & `SNR_trace`)** | Contrastive activation capture with held-out signal-to-noise and split-half cosine per layer. | 12 full H100 models (`7.2B` to `32.8B`), `SmolLM2-1.7B`, `Qwen3.5-0.8B`, `Qwen3.5-4B`. Not run on the two single-shard checkpoints. | Peak `B2` $\mathrm{SNR}_{\text{proj}}$ `83.0` to `199.4` on 11 of 12 H100 models; `microsoft/phi-4` is an outlier (see below). |
| **`secant compose` (Target-Shift Calibration)** | Two-pass Gram-Schmidt orthogonalization + analytical step size $\beta = \lvert\Delta_{\text{target}}\rvert / \Vert h_S\Vert _2^2$. | Calibrated end to end on `Qwen3.5-4B` (all `B1+B2+B3` edits) and `SmolLM2-1.7B` (`B3`). Regime C used the fixed-ratio mode instead. | `4B` dose-response table below; `SmolLM2` `B3` design shift `3.8936` vs. target `3.8956`. |
| **`secant patch` & `secant verify-patch`** | Zero-copy `mmap` patching on `.safetensors` and `.gguf` (`F32`/`BF16`/`F16`) with byte-level verification of untouched tensors. | `Qwen3.5-0.8B` and `Qwen3.5-4B` GGUF, `SmolLM2-1.7B` safetensors, one 7B-class safetensors shard [r]. Regime C edits were applied in RAM, not with `secant patch`. | `header_bytes_identical: true`, `other_tensors_byte_identical: true`. |
| **`secant quant witness` & `secant quant patch`** | In-place 34-byte `Q8_0` block surgery and $\Pi_{128}$ alignment / commutation witness vs. `Q4_K`. | `Qwen3.5-0.8B` (`Q8_0`, `Q4_K_S`) and `Qwen3.5-4B` (`Q8_0`, `Q4_K`, 16 witness runs). | `0` differing bytes outside touched `Q8_0` blocks (`240,640` untouched blocks bit-identical on `0.8B`). |
| **Generative `<think>` benchmark (`n = 30`, `1,536` tok)** | Multi-arm ablation plus $\alpha \in \{0, 0.5, 1.0, 1.5\}$ dose-response. | `Qwen3.5-4B` (`BF16` GGUF); 10-case ablation on `Qwen3.5-0.8B`. | `0/30` $\to$ `30/30` `<think>` closure (McNemar $p = 1.86 \times 10^{-9}$); `23/30` correct frame and `19/30` strict root cause at $1.5\times$. |
| **`ProjectThenHealLinear`, `secant heal` & `verify-subspace`** | Double orthogonal projection $P_\perp(P_\perp(\tilde{B}))$ that keeps a LoRA adapter out of the protected directions during training or on disk. | Regime A (`0.8B`, `1.7B`) and regime C (14 checkpoints). Offline `heal` on a `0.8B` adapter and on one 7B-class adapter [r]. | Regime A $\max \lvert D \Delta W\rvert $: $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$ (`float64`), $1.46 \times 10^{-10}$ to $2.49 \times 10^{-10}$ (merged `F32`). Regime C at step 60: $4.82 \times 10^{-10}$ to $1.08 \times 10^{-9}$ (`float64`), $3.87 \times 10^{-10}$ to $3.98 \times 10^{-8}$ (merged `F32`). |


[r] On the `0.8B`, `4B` and `SmolLM2` checkpoints, `patch`, `verify-patch` and `heal` were run with the reference Go implementation of the same subcommands, against which the Rust CLI is parity-tested. The 7B-class `read`, `sanitize`, `patch`, `verify-patch`, `heal` and `verify-subspace` receipts are Rust CLI receipts.
---

## Visual Highlights of Our Empirical Findings

### 1. The $\alpha$ Dose-Response on `Qwen3.5-4B` (Regime B)

Small reasoning models (`Qwen3.5-0.8B`, `Qwen3.5-4B`) frequently get trapped inside an unfinished `<think>...</think>` block when analyzing C++ AddressSanitizer crash dumps (`0/30` strict completion on unedited `Qwen3.5-4B` at a 1,536-token budget).

A fixed Frobenius norm ratio ($\gamma = 0.30$) carried over from `0.8B` closed `0/30` on `4B`. Calibrated edits produced a threshold between $\alpha = 0.5\times$ and $1.0\times$ ([](#fig-qwen4b-phase-transition)). Doses between those two points were not sampled, so the exact threshold location is unknown, and each arm is a single greedy run.

```{figure} assets/qwen4b-phase-transition.svg
:name: fig-qwen4b-phase-transition
:alt: Qwen3.5-4B alpha dose-response chart
:width: 100%

Dose-response on `Qwen3.5-4B` (`n = 30` C++ sanitizer crash-triage benchmark, `1,536`-token cap, greedy, one run per arm). Equal-norm random controls and the uncalibrated ratio stay at `0/30`; calibrated edits close `21/30` at $\alpha = 1.0\times$ and `30/30` at $\alpha = 1.5\times$.
```

| Arm (`Qwen3.5-4B`, `n = 30`, `1,536` token cap) | Summed `B1` target shift | `</think>` Closed | Strict Bug Class | Strict Culprit Frame | Strict Root Cause | Mean Tokens |
|---|---:|---:|---:|---:|---:|---:|
| **`base`** / **`rand1`** / **`rand2`** (equal-norm random) | n/a | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`g030`** (uncalibrated $\gamma = 0.30$) | n/a | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`a05`** ($\alpha = 0.5\times$) | `-3.691` | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`b1only`** ($\alpha = 1.0\times$, loop edit only) | `-7.382` | `17/30` (`56.7%`) | `15/30` (`50.0%`) | `13/30` (`43.3%`) | `11/30` (`36.7%`) | `1,044` |
| **`cal738`** ($\alpha = 1.0\times$, composed `B1+B2+B3`) | `-7.382` | **`21/30` (`70.0%`)** | **`18/30` (`60.0%`)** | **`15/30` (`50.0%`)** | **`14/30` (`46.7%`)** | `954` |
| **`a15`** ($\alpha = 1.5\times$, composed `B1+B2+B3`) | **`-11.073`** | **`30/30` (`100.0%`)** | **`23/30` (`76.7%`)** | **`23/30` (`76.7%`)** | **`19/30` (`63.3%`)** | `789` |

Exact two-sided McNemar tests against `base`: `cal738` closure $p = 9.54 \times 10^{-7}$ (21 vs. 0 discordant), `a15` closure $p = 1.86 \times 10^{-9}$ (30 vs. 0). `a15` vs. `cal738`: closure $p = 0.0039$, frame $p = 0.0078$, bug class $p = 0.125$ (not significant). In the `a05` arm only the `B1` edit was scaled; `B2` and `B3` were held at their `cal738` values.

---

### 2. Cross-Family Scale-Up & `ProjectThenHealLinear` (Regimes A and C)

**Regime A, live LM fine-tuning after real surgery.** Both arms train on real completions after a verified edit. Same metric for both arms.

| Model | Steps | Edit evidence | `ProjectThenHealLinear` $\max\lvert D \Delta W\rvert$, `float64` / merged `F32` | Unconstrained $\max\lvert D \Delta W\rvert$ | Held-out perplexity, PTH vs. unconstrained |
|---|---:|---|---:|---:|---|
| `Qwen/Qwen3.5-0.8B` | 60 (CPU) | Composed `B1+B2+B3` GGUF, `verify-patch` exit 0 | `9.87e-18` / `2.49e-10` | `1.05e-02` | step 20: `5.435` vs. `5.423`; step 60: `12.106` vs. `12.357` (both overfit after step 20) |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 20 (ROCm GPU) | `B2` projector at L14 `o_proj`: held-out shift `-2.436` vs. random `-0.0359` (`67.9x`). `B3` calibrated at L15 `down_proj`: `+3.952` vs. `+0.179` (`22.1x`) | `5.21e-18` / `1.46e-10` | `4.64e-03` | step 20: `21.260` vs. `21.255` |

After rounding merged weights to `BF16`, both arms show leakage of order $10^{-4}$ to $10^{-3}$ (`SmolLM2` PTH `6.8e-4` to `9.5e-4`), so the guarantee does not survive a `BF16` merge; `secant heal` stores the adapter in `F32`/`F64` for that reason.

**Regime C, H100 stress test (fixed-ratio edit, adversarial heal target).** Read the [regime notes](#regimes) before comparing columns. Edit layers are the layers of peak $\mathrm{SNR}_{\text{proj}}$ for each family. Shift values are at dose $1.0\times$.

```{figure} assets/project-then-heal-chart.svg
:name: fig-project-then-heal-chart
:alt: Regime C stress test, unconstrained adapter fraction in span(D) and ProjectThenHealLinear absolute leakage, 14 checkpoints
:width: 100%

Regime C at step 60 on 14 checkpoints. Left bars: fraction of the unconstrained adapter's `B` norm inside $\operatorname{span}(D)$. Right column: `ProjectThenHealLinear` absolute leakage $\max |D B_{\text{locked}}|$. The two quantities are different metrics and are shown side by side, not as a ratio.
```

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

## Documentation Guide

1. **[Quickstart & Live CLI Receipts](quickstart.md)**: Build `secant`, inspect a model on disk, extract directions, compose a calibrated plan, patch `.safetensors` or `.gguf` files, and verify bit-level invariants.
2. **[Mathematical Theory & Scale Calibration](theory-and-calibration.md)**: Derivations of contrastive capture, `SNR_proj`, Gram-Schmidt orthogonalization, and Target-Shift Calibration.
3. **[Quantized Block Surgery (`Q8_0` & `Q4_K`)](quantized-surgery.md)**: `Q8_0` 32-element block alignment vs. `Q4_K` 256-element superblock straddling, plus Path 1 vs. Path 2 measurements.
4. **[Project-Then-Heal Subspace-Locked LoRA](project-then-heal.md)**: How double orthogonal projection $P_\perp(P_\perp(\tilde{B}))$ keeps a LoRA adapter out of the protected directions.
5. **[CLI, Python & C-ABI Reference](cli-and-python-reference.md)**: Command-line flags, exit codes, JSON schemas, `secant.nn` / `secant.capture` APIs, and `secant.h` FFI symbols.
6. **[Container Geometry & Payload Security](payload-security.md)**: checked container extents, the `sanitize` static value scan, `probe` boundary-value fixtures, and `read --detect-anomalies` norm rules.
7. **[Technical Paper](PAPER.md)**: Self-contained write-up with propositions, proofs, and the full tables.
