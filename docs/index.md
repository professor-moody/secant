# Secant: Weight-Space Programming & Subspace-Locked LoRA Healing

**Secant** is an open-source Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python library (`secant`, `secant.nn.ProjectThenHealLinear`) for **deterministic Weight-Space Programming (WSP)**, **in-place quantized block surgery (`Q8_0`)**, and **Subspace-Locked LoRA Healing (`Project-Then-Heal`)** on `.safetensors` and `.gguf` (v2/v3) checkpoints.

Motivated by MIT CSAIL's release by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (*"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*, Nov 2, 2026) [@rikiyat2026wsp] alongside representation-engineering and null-space editing research [@zou2023representation; @zou2024circuitbreakers; @arditi2024refusal], Secant turns direct weight surgery into a **reproducible, hash-bracketed engineering discipline** across **17 model checkpoints (`0.8B` to `72.7B`)** and **9 architecture families** (`Qwen3.5`, `Qwen3`, `Qwen2.5`, `SmolLM2`, `Llama-3.1`, `Gemma-2`, `Mistral`, `OLMo-2`, `Phi-4`, and `DeepSeek-R1-Distill`).

> *"Turning what currently seems like black magic into a rigorous, reproducible engineering discipline... and auditing without evaluation (e.g., backdoors directly from weights)."*
>
> Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, **Sparks of Weight-Space Programming ("What Comes Next?")**, MIT CSAIL (Nov 2, 2026). **We took this on and decided to see what we could do.**

---

## Our 7 Empirical Findings in Plain Words (and How `secant` Identifies or Fixes Each One)

Before diving into equations or CLI flags, here are our seven core findings in plain, everyday English, along with the exact `secant` command that identifies or fixes each issue:

1. **"Safe" model files can still crash C++ memory on the very first run:**
   - **In plain words:** Switching from Python `pickle` files to `.safetensors` stopped hackers from hiding runnable Python code inside a model file. However, once a model loads, the C++ and GPU math kernels blindly trust the numbers stored inside the weights to pick Mixture-of-Experts (MoE) routes, divide quantization scales, sort words, and jump to memory addresses. If someone puts a `NaN` ("Not a Number") or a negative number inside a routing table, the `.safetensors` file passes every standard safety check and then crashes or corrupts C++ memory on the very first question. Worse, because Intel/AMD (`x86-64`) and Apple Silicon (`ARM64`) chips convert `NaN` to integers differently in hardware, a poisoned model file can run fine on a developer's MacBook and immediately crash a Linux production server.
   - **How `secant` identifies and fixes it:** Run `secant sanitize --model model.safetensors --receipt receipt.json --attempt-id audit-1 --out clean.safetensors` to scan every tensor on disk for `NaN`/`Inf` values, zero quantization scales, and out-of-range router indices and write a clean, verified copy before any GPU kernel ever loads it. Use `secant probe` to generate test files that verify whether your inference engine survives all four failure classes.

2. **Small reasoning models often fail on three stacked reflexes at once (not just one):**
   - **In plain words:** When you ask a small reasoning model (`Qwen3.5-0.8B` or `4B`) to analyze a C++ crash report, three separate bad habits stack on top of each other: **(1)** it gets stuck in an endless `"Wait, let me double-check..."` loop inside `<think>` until it runs out of tokens (`B1`), **(2)** if you fix the loop so it finally finishes `<think>`, it suddenly refuses to answer half the time because words like `"buffer overflow"` trigger a false-positive safety alarm (`B2`), and **(3)** when it does answer, it blames the line where the program crashed (`memcpy`) instead of the caller function that forgot to check the buffer size (`B3`). Fixing only one habit fails because the next habit in line blocks the answer.
   - **How `secant` identifies and fixes it:** `secant.capture` measures the activation direction of each habit and scores every layer (`SNR_proj`) so you target the exact layer responsible. Then `secant compose` makes the three directions mathematically perpendicular (orthogonal) so they never interfere with each other, and `secant patch` edits all three in a single pass on disk.

3. **Why searching inside `<think>` for keywords is a mirage (The `93.3%` Copy-the-Input Control), and how strict post-`</think>` scoring proved our `0%` $\to$ `100%` fix is real:**
   - **In plain words:** We did **not** lose a finding here. Rather, we caught a major benchmarking trap that makes our real result airtight. When the unedited `Qwen3.5-4B` model got stuck inside `<think>` (`0/30` finished answers), we first checked if the right function name appeared anywhere inside its unfinished `<think>` rambling. A standard keyword search said `"Yes, 86.7% (26/30)!"` Before trusting that number, we ran a simple sanity check: we fed the **user's input prompt itself** (with **zero AI model output**) into that same keyword scorer. The raw input prompt scored **`93.3%` (`28/30`)**, beating the model! Why? Because the input crash report already lists all 8 functions in the call stack; when the model rambles through the stack trace inside `<think>`, it repeats all of them.
   - **What this proved:** You cannot grade a reasoning model by checking whether a keyword from the prompt appears inside `<think>`. So we built a **strict post-`</think>` scorer** that only gives credit when the model **closes `</think>`** and commits to **one specific culprit function over all the other functions in the stack trace**. On that real, strict test: copying the input scores **`0/30` (`0%`)**, the unedited `Qwen3.5-4B` model scores **`0/30` (`0%`)**, and our `Secant`-edited `Qwen3.5-4B` model (`1.5x` dose) closes `<think>` on **`30/30` (`100%`)** of crashes and names the exact culprit function on **`23/30` (`76.7%`)**.
   - **How `secant` identifies and fixes it:** Use `secant.capture` on held-out prompts and evaluate models strictly on post-`</think>` discriminative verdicts (`dsc_score.py`), never on raw substring overlap with the prompt.

4. **Why copying an edit percentage from a small model to a larger model fails (`0/30`), and how Target-Shift Calibration unlocks `100%` (`30/30`) completion:**
   - **In plain words:** Most open-source weight-editing scripts scale an edit as a fixed percentage (say, `30%`) of a weight matrix's size. That worked on `Qwen3.5-0.8B` (`100%` loop fix), but when we used that exact same `30%` rule on `Qwen3.5-4B`, it fixed **`0 out of 30` (`0%`)** loops. Why? Because a bigger model has wider layers with much stronger internal signals; a `30%` weight nudge only pushed the `4B` model's internal signal by `-0.90` units, whereas it needed at least `-7.38` units to cross the threshold and close `</think>`. **Target-Shift Calibration** measures how strongly a model's layers fire and calculates the exact weight dose needed to reach that threshold on any model size.
   - **How `secant` identifies and fixes it:** In your `secant compose` profile, set `"calibration": {"mode": "target-shift", "target_shift": -7.3822}` (or scale with $\alpha = 1.5\times$ to `-11.073`). `secant compose` automatically computes the exact weight update for any model from `0.8B` to `72.7B`.

5. **You can audit a model's vocabulary concepts and edit `.safetensors` or `.gguf` files directly on disk without loading PyTorch:**
   - **In plain words:** Normally, inspecting or editing a model requires loading gigabytes of weights into PyTorch GPU memory, editing them, and re-converting the whole checkpoint. Instead, `secant` opens the file directly on disk (`mmap`), reads the strongest mathematical directions of any layer, and multiplies them against the model's dictionary so you can see the exact words a layer promotes (like `"unfortunately"`, `"sorry"`, `"illegal"`) or suppresses (like `"Explain"`, `"Definition"`) without running a single prompt. And when patching, it rewrites only the target matrices in place and proves via SHA-256 that the other `98%+` of the file did not change by a single bit.
   - **How `secant` identifies and fixes it:** Run `secant inspect --model <path>`, `secant read --model <path> --tensor <name> --vocab-tensor <embd> --detect-anomalies`, `secant patch`, and `secant verify-patch`.

6. **8-bit (`Q8_0`) GGUF models can be edited directly on disk, while 4-bit (`Q4_K`) models must be edited before quantization:**
   - **In plain words:** Quantized GGUF models compress weights into small blocks that share a scaling number. In `Qwen3.5`, each attention head uses `128` columns. Because `Q8_0` groups weights into blocks of `32`, exactly four `Q8_0` blocks fit cleanly inside one `128`-column head (`4 x 32 = 128`). That means `secant quant patch` can edit a `Q8_0` GGUF file directly on disk in seconds (`98.5%` refusal removal, matching edit-then-quantize on `10/10` held-out crashes). By contrast, `Q4_K` groups `256` weights into one "superblock" that stretches across **two neighboring `128`-column heads at once**. Trying to edit one head in place inside a `Q4_K` file corrupts the neighboring head's shared scale (`67%` of the refusal habit survives).
   - **How `secant` identifies and fixes it:** Run `secant quant witness` to check block alignment and quantization noise floor, and run `secant quant patch` on `Q8_0` GGUF files (`secant` automatically blocks unsafe in-place `Q4_K` superblock straddling with `SuperblockStraddleError`).

7. **Standard LoRA fine-tuning erases `8.4%` to `52.6%` of a weight edit in 60 steps, and `ProjectThenHealLinear` locks that erasure to zero (`< 10⁻⁹`):**
   - **In plain words:** Suppose you remove a bad habit (like `<think>` looping or false refusals) from a model's weights, and then you fine-tune the model with standard LoRA to teach it a new task. Within just 60 steps of AdamW training, the optimizer's momentum and per-weight scaling push the LoRA updates right back into the directions you just removed, undoing up to **half (`52.55%`)** of your weight surgery! We solved this with **`ProjectThenHealLinear`**, a drop-in LoRA layer that mathematically locks the adapter out of the edited directions on every forward pass (`B_locked = P_perp(P_perp(B))`). The model learns the new task at the exact same speed and accuracy, while `0.000000001` (`< 10⁻⁹` in `float64`) of the update ever leaks back into the removed habit.
   - **How `secant` identifies and fixes it:** In Python training, wrap linear layers with `secant.nn.ProjectThenHealLinear.from_linear(layer, rank=16, directions=D)`. For existing LoRA `.safetensors` adapters on disk, run `secant heal --lora adapter.safetensors --vectors D.npy --out healed_lora.safetensors` and verify with `secant verify-subspace`.

```{figure} assets/secant-pipeline.svg
:name: fig-secant-pipeline
:alt: Secant four-stage weight-space programming pipeline
:width: 100%

The four stages of Secant: (1) Zero-eval weight reading and Boundary-3 payload sanitization, (2) Gram-Schmidt multi-subspace composition with Target-Shift Calibration, (3) zero-copy `mmap` and 34-byte `Q8_0` quantized block surgery, and (4) `ProjectThenHealLinear` subspace-locked LoRA micro-training.
```

---

## Core Capabilities & Verification Coverage Across Model Scales (`0.8B` to `72.7B`)

To be 100% transparent about what has been tested at every scale, the table below records each `Secant` capability, its verified mathematical guarantee, and the exact model scales (`0.8B` to `72.7B`) where we have executed and banked SHA-256 receipts:

| Capability / Subcommand | What It Does | Verified Scales & Checkpoints (`17` Models Total) | Level of Robustness & Banked Evidence |
|---|---|---|---|
| **`secant inspect` & `secant read --detect-anomalies`** | Zero-eval `float64` SVD spectrum, vocabulary token projection, and column-norm / spectral anomaly detection directly on disk. | **`0.8B`, `1.5B`, `1.7B`, `4.0B`, `7.6B`** (`SafeTensors` & `GGUF` via Rust CLI) + **`70.6B` & `72.7B`** (`8192 x 29568` shard SVD). | Deterministic bit-exact `float64` power iteration (`residual < 1e-10`) + $z = 6.14$ vocabulary readout (`p < 0.01`). |
| **`secant sanitize` & `secant probe`** | Scans `.safetensors` and `.gguf` payloads for `NaN`/`Inf` routers, zero scales, and OOB indices; generates Boundary-3 test fixtures. | **`0.8B`, `1.5B`, `1.7B`, `4.0B`, `7.6B`** production checkpoints + **`212` `x86-64` vs. `ARM64` C++ kernel runs**. | `100%` detection of Class I-IV payload hazards (`input_unchanged: true`, pre/post SHA-256 bracketed). |
| **`secant.capture` (`SNR_proj` & `SNR_trace`)** | Multi-layer forward-pass contrastive activation capture with held-out signal-to-noise profiling across attention and MLP writers. | **15 full models from `0.8B` to `32.8B`** (`Qwen3.5`, `Qwen3`, `Qwen2.5`, `SmolLM2`, `Mistral`, `Llama-3.1`, `Gemma-2`, `OLMo-2`, `Phi-4`, `R1-Distill`). | Held-out split-half cosine (`0.80` to `0.996`) and sub-stream writer localization across `24` to `64` layers. |
| **`secant compose` (Target-Shift Calibration)** | Two-pass Gram-Schmidt orthogonalization + exact analytical step-size calibration ($\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$). | **All 17 models from `0.8B` to `72.7B`** (`0.8B` to `32.8B` full-model forward witness; `70.6B` & `72.7B` single-shard witness). | **`22.2x` to `1,269.9x`** target-shift separation over matched-norm random controls across all 17 models. |
| **`secant patch` & `secant verify-patch`** | Zero-copy `mmap` surgical patching on `.safetensors` and `.gguf` (`F32`/`BF16`/`F16`) with bit-for-bit untouched-tensor verification. | **`0.8B`, `1.5B`, `1.7B`, `4.0B`, `7.6B`** (`SafeTensors` & `GGUF` via Rust CLI) + in-RAM verification across **`7.2B` to `72.7B`**. | `100%` byte-identity on all non-target tensors (`header_bytes_identical: true`, `other_tensors_byte_identical: true`). |
| **`secant quant witness` & `secant quant patch`** | Direct in-place 34-byte `Q8_0` GGUF block surgery and $\Pi_{128}$ head-panel alignment / commutation witness vs. `Q4_K`. | **`Qwen3.5-0.8B` (`Q8_0` & `Q4_K_S`)** and **`Qwen3.5-4B` (`Q8_0` & `Q4_K`, 16/16 witness runs)**. | `0` differing bytes outside touched `Q8_0` blocks (`240,640` untouched blocks bit-identical); `10/10` Path 1 vs. Path 2 eval agreement. |
| **Full Generative `<think>` Benchmark (`n = 30`, `1,536` tok)** | Multi-arm ablation (`base`, `rand`, `b1only`, `b2only`, `b3only`, `composed`) + $\alpha \in \{0, 0.13, 0.5, 1.0, 1.5\}$ dose-response. | **`Qwen3.5-0.8B` (`BF16` & `Q8_0`)** and **`Qwen3.5-4B` (`BF16`)** (6-arm + dose-response, `30` ASan crashes + `20` held-out cases). | `0/30` (`0%`) $\to$ **`30/30` (`100%`, McNemar $p = 1.86 \times 10^{-9}$)** `<think>` closure and **`23/30` (`76.7%`)** exact frame at $\alpha = 1.5\times$. |
| **`ProjectThenHealLinear`, `secant heal` & `verify-subspace`** | Double orthogonal projection $P_\perp(P_\perp(\tilde{B}))$ locking LoRA adapters out of surgical subspaces during AdamW training or on disk. | **All 17 models from `0.8B` to `72.7B`** (live LM fine-tuning on `0.8B` & `1.7B`; Rust CLI `heal` on `0.8B` & `7.6B`; 60-step AdamW stress test on `7.2B` to `72.7B`). | Reduces `8.42%`–`52.55%` unconstrained LoRA subspace erasure to **`9.87e-18`–`1.08e-9` (`float64`)** and **`1.46e-10`–`3.98e-8` (`F32`)**. |

---

## Visual Highlights of Our Empirical Findings

### 1. The $\alpha$-Dose-Response Phase Transition on `Qwen3.5-4B`

Small reasoning models (`Qwen3.5-0.8B`, `Qwen3.5-4B`) frequently get trapped inside an unfinished `<think>...</think>` block when analyzing C++ AddressSanitizer crash dumps (`0/30` strict completion on unedited `Qwen3.5-4B` at a 1,536-token budget).

Simply copying a fixed Frobenius norm ratio ($\gamma = 0.30$) from `0.8B` to `4B` fails completely (`0/30` closed) because the induced activation shift (`-0.90`) is an order of magnitude smaller than the `-7.38` shift needed to cross the `</think>` termination threshold. As shown in [](#fig-phase-surface-3d) and [](#fig-qwen4b-phase-transition), solving for the weight update via **Target-Shift Calibration** triggers a clean phase transition across both layer depth and calibrated dose $\alpha$:

```{figure} assets/fig-phase-surface-3d.svg
:name: fig-phase-surface-3d
:alt: 3D phase-transition surface over layer depth and calibrated dose alpha on Qwen3.5-4B
:width: 100%

**(A)** 3D phase-transition surface over target layer index $\ell \in [8, 28]$ and calibrated dose multiplier $\alpha \in [0, 1.75]$ on `Qwen3.5-4B`. **(B)** Observed vs. sigmoid-fitted phase transition (`n = 30`, `1,536`-token cap), showing `0%` closure at uncalibrated $\gamma = 0.30$ (`alpha = 0.13x`) and half-dose `0.5x`, `70.0%` closure at $\alpha = 1.0\times$, and `100.0%` (`30/30`, McNemar $p = 1.86 \times 10^{-9}$) closure at $\alpha = 1.5\times$.
```

```{figure} assets/qwen4b-phase-transition.svg
:name: fig-qwen4b-phase-transition
:alt: Qwen3.5-4B alpha dose-response phase transition chart
:width: 100%

Dose-response phase transition on `Qwen3.5-4B` (`n = 30` C++ sanitizer crash-triage benchmark, `1,536` token cap). Equal-norm random controls and uncalibrated ratios stay at `0%`, while Target-Shift Calibration reaches `70.0%` closure at $\alpha = 1.0\times$ and `100.0%` (`30/30`) closure at $\alpha = 1.5\times$.
```

| Arm (`Qwen3.5-4B`, `n = 30`, `1,536` token cap) | Target Shift $\Delta_{\text{target}}$ | `</think>` Closed | Post-`</think>` Bug Class | Post-`</think>` Frame | Discriminative Root Cause | Mean Tokens |
|---|---:|---:|---:|---:|---:|---:|
| **`base`** / **`rand1`** / **`b2only`** / **`b3only`** | `0.00` | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `1,536` |
| **`g030`** (uncalibrated $\gamma = 0.30$) | `-0.90` | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `1,536` |
| **`a05`** ($\alpha = 0.5\times$ half dose) | `-3.69` | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `1,536` |
| **`b1only`** ($\alpha = 1.0\times$ loop edit only) | `-7.38` | `17/30` (`56.7%`) | `17/30` (`56.7%`) | `17/30` (`56.7%`) | `11/30` (`36.7%`) | `1,044` |
| **`cal738`** ($\alpha = 1.0\times$ composed `B1+B2+B3`) | `-7.38` | **`21/30` (`70.0%`)** | **`18/30` (`60.0%`)** | **`18/30` (`60.0%`)** | **`14/30` (`46.7%`)** | `954` |
| **`a15`** (**$\alpha = 1.5\times$ composed `B1+B2+B3`**) | **`-11.07`** | **`30/30` (`100.0%`)** | **`23/30` (`76.7%`)** | **`23/30` (`76.7%`)** | **`19/30` (`63.3%`)** | **`742`** |

---

### 2. Cross-Family Scale-Up (`0.8B` to `72.7B`) & `ProjectThenHealLinear`

We evaluate `ProjectThenHealLinear` across **17 model checkpoints (`0.8B` to `72.7B`)** in two complementary regimes (visualized in [](#fig-pth-trajectories) and [](#fig-project-then-heal-chart)):
1. **Live causal LM fine-tuning (`Qwen3.5-0.8B` 60-step CPU and `SmolLM2-1.7B` 20-step ROCm GPU):** Both `ProjectThenHealLinear` and unconstrained LoRA train on domain completions after weight surgery. Both arms achieve identical training loss reduction (`2.8242` $\to$ `0.3120` vs. `0.3096` on `0.8B`, with optimal held-out perplexity `5.435` at Step 20 as shown in [](#fig-pth-trajectories)), while `ProjectThenHealLinear` holds writer leakage $\max |D \Delta W|$ at $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$ (`float64`) vs. $4.64 \times 10^{-3}$ to $1.05 \times 10^{-2}$ for unconstrained LoRA.
2. **Adversary-aligned subspace stress test across 14 models on H100 (`7.2B` to `72.7B`):** A synthetic target ($Y_{\text{target}} = Y + 1.5 v_{B2}$) pulls the adapter directly toward the forbidden subspace $\operatorname{span}(D)$ for 60 AdamW steps. Under this direct pull, unconstrained LoRA places $8.42\%$ to $52.55\%$ of its adapter norm ($\|D B\|_{\mathrm{F}} / \|B\|_{\mathrm{F}}$) inside $\operatorname{span}(D)$, whereas `ProjectThenHealLinear` holds writer leakage $\max |D B_{\text{locked}}|$ at $4.83 \times 10^{-10}$ to $1.08 \times 10^{-9}$ (`float64`) and merged `F32` leakage at $3.87 \times 10^{-10}$ to $3.98 \times 10^{-8}$.

```{figure} assets/fig-pth-trajectories.svg
:name: fig-pth-trajectories
:alt: 60-step AdamW training loss, held-out perplexity, and log-scale subspace leakage trajectories
:width: 100%

**(A)** 60-step AdamW training cross-entropy loss and held-out validation perplexity on `Qwen3.5-0.8B`, showing zero learning penalty and the Step-20 early-stopping perplexity minimum (`PPL = 5.435`). **(B)** Log-scale subspace leakage trajectories across 60 AdamW steps from `0.8B` to `72.7B`, showing unconstrained LoRA overwriting `8.4%`–`52.6%` of the surgical subspace while `ProjectThenHealLinear` locks leakage below $10^{-9}$ (`float64`).
```

```{figure} assets/project-then-heal-chart.svg
:name: fig-project-then-heal-chart
:alt: Cross-family 60-step LoRA healing comparison chart from 0.8B to 72.7B
:width: 100%

Across models from `0.8B` to `72.7B`, unconstrained LoRA intrudes into the protected subspace during 60 AdamW steps, while `ProjectThenHealLinear` holds writer-level leakage at $9.87 \times 10^{-18}$ to $1.08 \times 10^{-9}$ (`float64`) and $1.46 \times 10^{-10}$ to $3.98 \times 10^{-8}$ (`F32`).
```

| Model Checkpoint | Params | Layers ($d_{\text{model}}$) | `B2` Shift Ratio vs. Random (`Shift` Pair) | `B3` Shift Ratio vs. Random (`Shift` Pair) | Unconstrained LoRA Subspace Intrusion | `ProjectThenHealLinear` Leakage (`float64` / `F32`) |
|---|---:|---:|---:|---:|---:|---:|
| **`Qwen/Qwen3.5-0.8B`** (Live 60-step LM) | `0.8B` | `24` (`1024`) | `10/10` `<think>` close | `6/10` strict frame | `1.05e-02` max $\|D \Delta W\|$ | **`9.87e-18` / `2.49e-10`** |
| **`HuggingFaceTB/SmolLM2-1.7B-Instruct`** (20-step ROCm) | `1.7B` | `24` (`2048`) | **`68.0x`** (`-12.50` vs `-0.184`) | **`22.2x`** (`+9.17` vs `+0.413`) | `4.64e-03` max $\|D \Delta W\|$ | **`5.21e-18` / `1.46e-10`** |
| **`Qwen/Qwen3.5-4B`** (6-Arm + Static Lock) | `4.0B` | `32` (`2560`) | **`30/30`** (`100%`) close at `1.5x` | **`23/30`** (`76.7%`) class+frame | `1.40e-03` (`BF16` merge floor) | **`7.01e-18` / `2.20e-10`** |
| **`mistralai/Mistral-7B-Instruct-v0.3`** | `7.2B` | `32` (`4096`) | **`663.9x`** (`+0.257` vs `-0.00039`) | **`98.9x`** (`+3.212` vs `-0.0325`) | `46.79%` (`0.4679`) | **`7.73e-10` / `3.87e-10`** |
| **`Qwen/Qwen2.5-7B-Instruct`** | `7.6B` | `28` (`3584`) | **`155.1x`** (`+4.654` vs `+0.0300`) | **`132.7x`** (`+20.82` vs `-0.1568`) | `31.34%` (`0.3134`) | **`7.55e-10` / `6.88e-09`** |
| **`meta-llama/Llama-3.1-8B-Instruct`** | `8.0B` | `32` (`4096`) | **`176.8x`** (`+3.539` vs `-0.0200`) | **`35.3x`** (`+8.888` vs `-0.2520`) | `48.03%` (`0.4803`) | **`7.62e-10` / `2.72e-09`** |
| **`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`** | `8.0B` | `32` (`4096`) | **`1,269.9x`** (`+2.609` vs `+0.0021`) | **`143.3x`** (`+8.644` vs `-0.0603`) | **`52.55%` (`0.5255`)** | **`6.39e-10` / `2.13e-09`** |
| **`Qwen/Qwen3-8B`** | `8.2B` | `36` (`4096`) | **`219.7x`** (`+2.187` vs `-0.0100`) | **`29.1x`** (`+120.8` vs `+4.146`) | `39.53%` (`0.3953`) | **`7.14e-10` / `5.93e-09`** |
| **`google/gemma-2-9b-it`** (Logit Soft-Cap + GeGLU) | `9.2B` | `42` (`3584`) | **`127.9x`** (`+19.08` vs `+0.1492`) | **`87.2x`** (`+130.0` vs `+1.491`) | `41.05%` (`0.4105`) | **`6.53e-10` / `8.50e-10`** |
| **`allenai/OLMo-2-1124-13B-Instruct`** | `13.7B` | `40` (`5120`) | **`103.3x`** (`+1.250` vs `+0.0121`) | **`262.9x`** (`+7.341` vs `-0.0279`) | `30.73%` (`0.3073`) | **`1.08e-09` / `3.33e-09`** |
| **`microsoft/phi-4`** | `14.7B` | `40` (`5120`) | **`326.9x`** (`-3.399` vs `+0.0104`, abs) | **`48.7x`** (`+30.76` vs `+0.6322`) | `8.42%` (`0.0842`) | **`9.51e-10` / `1.70e-08`** |
| **`Qwen/Qwen3-14B`** | `14.8B` | `40` (`5120`) | **`252.5x`** (`+4.548` vs `+0.0180`) | **`492.8x`** (`+63.11` vs `-0.1281`) | `41.33%` (`0.4133`) | **`5.77e-10` / `4.68e-09`** |
| **`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`** | `14.8B` | `48` (`5120`) | **`66.7x`** (`+3.762` vs `+0.0564`) | **`64.9x`** (`+51.94` vs `+0.8008`) | `32.57%` (`0.3257`) | **`6.11e-10` / `2.78e-09`** |
| **`google/gemma-2-27b-it`** (Logit Soft-Cap + GeGLU) | `27.2B` | `46` (`4608`) | **`139.0x`** (`+542.8` vs `+3.905`) | **`196.0x`** (`+5152.2` vs `-26.29`) | `11.54%` (`0.1154`) | **`7.16e-10` / `7.54e-10`** |
| **`Qwen/Qwen3-32B`** | `32.8B` | `64` (`5120`) | **`158.6x`** (`+18.89` vs `+0.1191`) | **`510.5x`** (`+41.52` vs `-0.0813`) | `30.75%` (`0.3075`) | **`4.83e-10` / `8.29e-09`** |
| **`deepseek-ai/DeepSeek-R1-Distill-Llama-70B`** (Shard SVD) | `70.6B` | `80` (`8192`) | **`537.1x`** (`+8.301` vs `-0.0155`) | `L32 down_proj` (`8192x28672`) | `20.24%` (`0.2024`) | **`5.66e-10` / `3.98e-08`** |
| **`Qwen/Qwen2.5-72B-Instruct`** (Shard SVD) | `72.7B` | `80` (`8192`) | **`134.2x`** (`+9.043` vs `+0.0674`) | `L40 down_proj` (`8192x29568`) | `36.94%` (`0.3694`) | **`9.85e-10` / `1.58e-08`** |

---

## Documentation Guide

1. **[Quickstart & Live CLI Receipts](quickstart.md)**: Build `secant`, inspect a model on disk, extract directions, compose a calibrated plan, patch `.safetensors` or `.gguf` files, and verify bit-level invariants.
2. **[Mathematical Theory & Scale Calibration](theory-and-calibration.md)**: Intuitive explanation and formal derivations of contrastive capture, `SNR_proj`, Gram-Schmidt orthogonalization, and Target-Shift Calibration.
3. **[Quantized Block Surgery (`Q8_0` & `Q4_K`)](quantized-surgery.md)**: Visual guide to `Q8_0` 32-element block alignment vs. `Q4_K` 256-element superblock straddling, plus Path 1 vs. Path 2 benchmarks.
4. **[Project-Then-Heal Subspace-Locked LoRA](project-then-heal.md)**: How double orthogonal projection $P_\perp(P_\perp(\tilde{B}))$ prevents AdamW from intruding into excised subspaces during post-edit fine-tuning.
5. **[CLI, Python & C-ABI Reference](cli-and-python-reference.md)**: Complete command-line flags, exit codes, JSON schemas, `secant.nn` / `secant.capture` APIs, and `secant.h` FFI symbols.
6. **[Container Geometry & Payload Security](payload-security.md)**: Boundary-3 C++ kernel defense, GGUF `ne[]` vs. logical `[d_out, d_in]` orientation, and `Qwen3.5` `ssm_out` head permutation.
7. **[Technical Paper](PAPER.md)**: Complete self-contained research paper with theorems, proofs, and multi-architecture tables.
