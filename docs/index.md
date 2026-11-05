# Secant: Weight-Space Programming & Subspace-Locked LoRA Healing

**Secant** is an open-source Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python library (`secant`, `secant.nn.ProjectThenHealLinear`) for **deterministic Weight-Space Programming (WSP)**, **in-place quantized block surgery (`Q8_0`)**, and **Subspace-Locked LoRA Healing (`Project-Then-Heal`)** on `.safetensors` and `.gguf` (v2/v3) checkpoints.

Motivated by MIT CSAIL's release by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (*"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*, Nov 2, 2026) [@rikiyat2026wsp] alongside representation-engineering and null-space editing research [@zou2023representation; @zou2024circuitbreakers; @arditi2024refusal], Secant turns direct weight surgery into a **reproducible, hash-bracketed engineering discipline** across model scales (`0.8B` to `32.8B`) and architectures (`Qwen3.5`, `Qwen3`, `Qwen2.5`, `SmolLM2`, `Mistral`, `Phi-4`, and `DeepSeek-R1-Distill`).

> *"Turning what currently seems like black magic into a rigorous, reproducible engineering discipline... and auditing without evaluation (e.g., backdoors directly from weights)."*
> — Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, **Sparks of Weight-Space Programming ("What Comes Next?")**, MIT CSAIL (Nov 2, 2026). **We took this on and decided to see what we could do.**

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
| **2. Capture & Compose** | `secant.capture`, `secant compose` | Profiles held-out sub-stream signal-to-noise ratio (`SNR_proj`), orthogonalizes multiple directions via two-pass modified Gram-Schmidt ($\|D D^\top - I_k\|_F < 10^{-14}$), and sizes updates via **Target-Shift Calibration**. |
| **3. Zero-Copy Patch** | `secant patch`, `secant verify-patch`, `secant quant patch` | Applies surgical plans directly to `.safetensors` and `.gguf` (`F32`/`BF16`/`F16`/`Q8_0`) via `memmap2`, proving by SHA-256 that 100% of untouched tensors and `Q8_0` blocks remain bit-for-bit identical. |
| **4. Project-Then-Heal** | `secant.nn.ProjectThenHealLinear`, `secant heal`, `secant verify-subspace` | Parameterizes LoRA inside the orthogonal complement $\PiPerp = I - D^\top D$ via double projection, holding subspace leakage below **`10^{-15}` (`float64`) / `10^{-8}` (`float32`)** across 60-step AdamW fine-tuning. |

---

## Visual Highlights of Our Empirical Findings

### 1. The $\alpha$-Dose-Response Phase Transition on `Qwen3.5-4B`

Small reasoning models (`Qwen3.5-0.8B`, `Qwen3.5-4B`) frequently get trapped inside an unfinished `<think>...</think>` block when analyzing C++ AddressSanitizer crash dumps (`0/30` strict completion on unedited `Qwen3.5-4B` at a 1,536-token budget).

Simply copying a fixed Frobenius norm ratio ($\gamma = 0.30$) from `0.8B` to `4B` fails completely (`0/30` closed) because the induced activation shift (`-0.90`) is an order of magnitude smaller than the `-7.38` shift needed to cross the `</think>` termination threshold. Solving for the weight update via **Target-Shift Calibration** triggers a clean phase transition:

```{figure} assets/qwen4b-phase-transition.svg
:name: fig-qwen4b-phase-transition
:alt: Qwen3.5-4B alpha dose-response phase transition chart
:width: 100%

Dose-response phase transition on `Qwen3.5-4B` (`n = 30` C++ sanitizer crash-triage benchmark, `1,536` token cap). Equal-norm random controls and uncalibrated ratios stay at `0%`, while Target-Shift Calibration reaches `70.0%` closure at $\alpha = 1.0\times$ and `100.0%` (`30/30`) closure at $\alpha = 1.5\times$.
```

| Arm (`Qwen3.5-4B`, `n = 30`, `1,536` token cap) | Target Shift $\Delta_{\text{target}}$ | `</think>` Closed | Strict Bug Class | Strict Culprit Frame | Held-Out Eval (`n=10`) | Mean Tokens |
|---|---:|---:|---:|---:|---:|---:|
| **`base`** / **`rand1`** / **`rand2`** | `0.00` | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `0/10` (`0%`) | `1,536` |
| **`g030`** (uncalibrated $\gamma = 0.30$) | `-0.90` | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `0/30` (`0.0%`) | — | `1,536` |
| **`a05`** ($\alpha = 0.5\times$ half dose) | `-3.69` | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `0/10` (`0%`) | `1,536` |
| **`b1only`** ($\alpha = 1.0\times$ loop edit only) | `-7.38` | `17/30` (`56.7%`) | `15/30` (`50.0%`) | `13/30` (`43.3%`) | `7/10` (`70%`) | `1,044` |
| **`cal738`** ($\alpha = 1.0\times$ composed `B1+B2+B3`) | `-7.38` | **`21/30` (`70.0%`)** | **`18/30` (`60.0%`)** | **`15/30` (`50.0%`)** | **`9/10` (`90%`)** | `954` |
| **`a15`** (**$\alpha = 1.5\times$ composed `B1+B2+B3`**) | **`-11.07`** | **`30/30` (`100.0%`)** | **`23/30` (`76.7%`)** | **`23/30` (`76.7%`)** | **`10/10` (`100%`)** | **`789`** |

---

### 2. Cross-Family Scale-Up (`0.8B` to `32.8B`) & `ProjectThenHealLinear`

When standard LoRA fine-tuning runs after weight surgery, AdamW's coordinatewise second-moment scaling pushes gradient updates straight back into the forbidden behavioral directions, erasing **`8.4%` to `52.6%`** of the surgical subspace within 60 steps. `ProjectThenHealLinear` locks the adapter into the orthogonal complement $\PiPerp = I - D^\top D$ on every forward pass, keeping subspace leakage below **`10^{-9}`** across 11 models spanning 7 architecture families:

```{figure} assets/project-then-heal-chart.svg
:name: fig-project-then-heal-chart
:alt: Cross-family 60-step AdamW healing comparison chart from 0.8B to 32.8B
:width: 100%

Across models from `0.8B` to `32.8B`, unconstrained LoRA overwrites `8.4%–52.6%` of the surgical subspace during 60 AdamW steps, while `ProjectThenHealLinear` holds subspace leakage below `10^{-9}` (`float64` and `F32`) with zero learning penalty.
```

| Model Checkpoint | Params | Layers ($d_{\text{model}}$) | `B2` Calibrated Shift vs. Random | `B3` Calibrated Shift vs. Random | 60-Step Unconstrained LoRA Erasure | 60-Step `ProjectThenHealLinear` Leakage (`float64` / `F32`) |
|---|---:|---:|---:|---:|---:|---:|
| **`Qwen/Qwen3.5-0.8B`** | `0.8B` | `24` (`1024`) | `10/10` `<think>` close | `6/10` strict frame | `15.63%` (`0.1563`) | **`3.76e-17` / `9.34e-11`** |
| **`HuggingFaceTB/SmolLM2-1.7B-Instruct`** | `1.7B` | `24` (`2048`) | **`68.0x`** (`99.5%` supp.) | **`22.2x`** (`+3.95` shift) | `14.20%` (`0.1420`) | **`4.21e-18` / `9.34e-11`** |
| **`Qwen/Qwen3.5-4B`** | `4.0B` | `32` (`2560`) | **`100%`** close at `1.5x` | **`76.7%`** class+frame | `18.40%` (`0.1840`) | **`7.01e-18` / `2.20e-10`** |
| **`mistralai/Mistral-7B-Instruct-v0.3`** | `7.2B` | `32` (`4096`) | **`663.9x`** (`+3.98` vs `-0.006`) | **`98.9x`** (`+13.94` vs `-0.14`) | `46.79%` (`0.4679`) | **`7.73e-10` / `3.87e-10`** |
| **`Qwen/Qwen2.5-7B-Instruct`** | `7.6B` | `28` (`3584`) | **`155.1x`** (`+4.65` vs `+0.030`) | **`132.7x`** (`+20.82` vs `-0.16`) | `31.34%` (`0.3134`) | **`7.55e-10` / `6.88e-09`** |
| **`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`** | `8.0B` | `32` (`4096`) | **`1,269.9x`** (`+2.61` vs `+0.002`) | **`143.3x`** (`+8.64` vs `-0.06`) | **`52.55%` (`0.5255`)** | **`6.39e-10` / `2.13e-09`** |
| **`Qwen/Qwen3-8B`** | `8.2B` | `36` (`4096`) | **`219.7x`** (`+2.19` vs `-0.010`) | **`29.1x`** (`+120.8` vs `+4.15`) | `39.53%` (`0.3953`) | **`7.14e-10` / `6.06e-09`** |
| **`allenai/OLMo-2-1124-13B-Instruct`** | `13.7B` | `40` (`5120`) | **`103.3x`** (`+1.25` vs `+0.012`) | **`262.9x`** (`+7.34` vs `-0.028`) | `30.73%` (`0.3073`) | **`1.08e-09` / `3.33e-09`** |
| **`microsoft/phi-4`** | `14.7B` | `40` (`5120`) | **`326.9x`** (`+3.60` vs `-0.011`) | **`48.7x`** (`+4.24` vs `+0.087`) | `8.42%` (`0.0842`) | **`9.51e-10` / `1.04e-08`** |
| **`Qwen/Qwen3-14B`** | `14.8B` | `40` (`5120`) | **`252.5x`** (`+2.53` vs `+0.010`) | **`492.8x`** (`+142.9` vs `-0.29`) | `41.33%` (`0.4133`) | **`5.77e-10` / `4.68e-09`** |
| **`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`** | `14.8B` | `48` (`5120`) | **`66.7x`** (`+3.76` vs `+0.056`) | **`64.9x`** (`+51.94` vs `+0.80`) | `32.57%` (`0.3257`) | **`6.11e-10` / `2.78e-09`** |
| **`Qwen/Qwen3-32B`** | `32.8B` | `64` (`5120`) | **`158.6x`** (`+18.89` vs `+0.119`) | **`510.5x`** (`+41.52` vs `-0.08`) | `30.75%` (`0.3075`) | **`4.83e-10` / `8.29e-09`** |

---

## Documentation Guide

1. **[Quickstart & Live CLI Receipts](quickstart.md)** — Build `secant`, inspect a model on disk, extract directions, compose a calibrated plan, patch `.safetensors` or `.gguf` files, and verify bit-level invariants.
2. **[Mathematical Theory & Scale Calibration](theory-and-calibration.md)** — Intuitive explanation and formal derivations of contrastive capture, `SNR_proj`, Gram-Schmidt orthogonalization, and Target-Shift Calibration.
3. **[Quantized Block Surgery (`Q8_0` & `Q4_K`)](quantized-surgery.md)** — Visual guide to `Q8_0` 32-element block alignment vs. `Q4_K` 256-element superblock straddling, plus Path 1 vs. Path 2 benchmarks.
4. **[Project-Then-Heal Subspace-Locked LoRA](project-then-heal.md)** — How double orthogonal projection $\PiPerp(\PiPerp(\tilde{B}))$ prevents AdamW from erasing weight surgery during post-edit fine-tuning.
5. **[CLI, Python & C-ABI Reference](cli-and-python-reference.md)** — Complete command-line flags, exit codes, JSON schemas, `secant.nn` / `secant.capture` APIs, and `libsecant.h` FFI symbols.
6. **[Container Geometry & Payload Security](payload-security.md)** — Boundary-3 C++ kernel defense, GGUF `ne[]` vs. logical `[d_out, d_in]` orientation, and `Qwen3.5` `ssm_out` head permutation.
7. **[Technical Paper](PAPER.md)** — Complete self-contained research paper with theorems, proofs, and multi-architecture tables.
