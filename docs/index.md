# Secant: Weight-Space Programming & Subspace-Locked LoRA Healing

**Secant** is an open-source Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python library (`secant`, `secant.nn.ProjectThenHealLinear`) for **deterministic Weight-Space Programming (WSP)**, **in-place quantized block surgery (`Q8_0`)**, and **Subspace-Locked LoRA Healing (`Project-Then-Heal`)** on `.safetensors` and `.gguf` (v2/v3) checkpoints.

Motivated by MIT CSAIL's release by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (*"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*, Nov 2, 2026) [@rikiyat2026wsp] alongside representation-engineering and null-space editing research [@zou2023representation; @zou2024circuitbreakers; @arditi2024refusal], Secant turns direct weight surgery into a **reproducible, hash-bracketed engineering discipline** across **17 model checkpoints (`0.8B` to `72.7B`)** and **9 architecture families** (`Qwen3.5`, `Qwen3`, `Qwen2.5`, `SmolLM2`, `Llama-3.1`, `Gemma-2`, `Mistral`, `OLMo-2`, `Phi-4`, and `DeepSeek-R1-Distill`).

> *"Turning what currently seems like black magic into a rigorous, reproducible engineering discipline... and auditing without evaluation (e.g., backdoors directly from weights)."*
>
> Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, **Sparks of Weight-Space Programming ("What Comes Next?")**, MIT CSAIL (Nov 2, 2026). **We took this on and decided to see what we could do.**

---

## Key Takeaways in Plain Words

Before diving into equations or CLI flags, here are the four core ideas behind Secant:

1. **Read and audit weights directly from disk without running the model:** `secant read` computes the top singular directions of any weight matrix in `float64` and projects them onto the model's token embeddings so you can see which words a layer promotes or suppresses. `secant sanitize` checks `.safetensors` and `.gguf` files for `NaN`/`Inf` values, zero quantization scales, and out-of-range router indices that crash C++ inference kernels after container validation passes.
2. **Calibrate edits by activation shift, not by fixed norm percentages:** An edit that changes `2.7%` of a matrix's Frobenius norm works on `Qwen3.5-0.8B` (`100%` `<think>` loop closure), but that same percentage fails completely (`0/30` closed) on `Qwen3.5-4B` because wider layers have larger internal activations. **Target-Shift Calibration** solves for the exact weight update needed to shift the layer's output by a chosen inner product ($\Delta_{\text{target}}$), restoring `70.0%` closure at $1.0\times$ dose and `100.0%` (`30/30`) closure at $1.5\times$ dose.
3. **`Q8_0` quantized blocks align with 128-column head panels, while `Q4_K` superblocks straddle them:** In `Qwen3.5` GGUF files, `llama.cpp` reorders `ssm_out` columns in 128-column value-head panels ($\Pi_{128}$). Because `Q8_0` quantizes 32 weights per 34-byte block, four `Q8_0` blocks fit inside each 128-column panel ($128 = 4 \times 32$) with zero boundary crossings, enabling in-place quantized block surgery (`98.5%` refusal removal). By contrast, a 256-weight `Q4_K` superblock spans two adjacent 128-column panels with a shared `FP16` scale, so editing one head in place corrupts its neighbor (`67%` of the refusal direction survives).
4. **Lock LoRA inside the forward pass so AdamW cannot undo weight surgery:** If you fine-tune an edited model with standard LoRA, AdamW's per-coordinate variance scaling rotates updates back into the directions you just removed. `ProjectThenHealLinear` parameterizes the LoRA output matrix inside the orthogonal complement $P_\perp = I - D^\top D$ via double projection $B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$, holding writer-level subspace leakage at $9.87 \times 10^{-18}$ (`float64` on `0.8B` live LM training) and $4.83 \times 10^{-10}$ to $1.08 \times 10^{-9}$ (`float64` across `7B` to `32.8B` H100 stress tests).

```{figure} assets/secant-pipeline.svg
:name: fig-secant-pipeline
:alt: Secant four-stage weight-space programming pipeline
:width: 100%

The four stages of Secant: (1) Zero-eval weight reading and Boundary-3 payload sanitization, (2) Gram-Schmidt multi-subspace composition with Target-Shift Calibration, (3) zero-copy `mmap` and 34-byte `Q8_0` quantized block surgery, and (4) `ProjectThenHealLinear` subspace-locked LoRA micro-training.
```

---

## Core Capabilities at a Glance

| Stage | CLI / Python Entrypoint | What It Does & Verified Guarantee |
|---|---|---|
| **1. Read & Sanitize** | `secant inspect`, `secant read`, `secant sanitize`, `secant probe` | Extracts top singular triplets $(\sigma_i, u_i, v_i)$ and vocabulary projections directly from disk via deterministic `float64` power iteration, and scans/remediates Boundary-3 C++ kernel hazards (`NaN`/`Inf` routers, zero scales, out-of-range `I32` index tables). |
| **2. Capture & Compose** | `secant.capture`, `secant compose` | Profiles held-out sub-stream signal-to-noise ratio (`SNR_proj`), orthogonalizes multiple directions via two-pass modified Gram-Schmidt ($\|D D^\top - I_k\|_{\mathrm{F}} < 10^{-14}$), and sizes updates via **Target-Shift Calibration**. |
| **3. Zero-Copy Patch** | `secant patch`, `secant verify-patch`, `secant quant patch` | Applies surgical plans directly to `.safetensors` and `.gguf` (`F32`/`BF16`/`F16`/`Q8_0`) via `memmap2`, proving by SHA-256 that 100% of untouched tensors and `Q8_0` blocks remain bit-for-bit identical. |
| **4. Project-Then-Heal** | `secant.nn.ProjectThenHealLinear`, `secant heal`, `secant verify-subspace` | Parameterizes LoRA inside the orthogonal complement $P_\perp = I - D^\top D$ via double projection, holding writer-level subspace leakage at $9.87 \times 10^{-18}$ to $1.08 \times 10^{-9}$ (`float64`) and $1.46 \times 10^{-10}$ to $1.70 \times 10^{-8}$ (`float32`). |

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
