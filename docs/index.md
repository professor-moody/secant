# Secant: Weight-Space Programming & Subspace-Locked LoRA Healing

**Secant** is a standalone Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python library (`secant`, `secant.nn.ProjectThenHealLinear`) for **deterministic Weight-Space Programming (WSP)**, **direct quantized block surgery (`Q8_0` / `Q4_K`)**, and **Subspace-Locked LoRA Healing (`Project-Then-Heal`)** on SafeTensors and GGUF (v2/v3) model artifacts.

Motivated by the landmark MIT CSAIL release by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (*"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*, Nov 2, 2026) [@rikiyat2026wsp] alongside representation-engineering and circuit-breaker research [@zou2023representation; @zou2024circuitbreakers; @arditi2024refusal], Secant turns behavioral direction surgery into a **reproducible, hash-bracketed systems discipline** across scales (`0.8B` to `14.8B+`) and model families (`Qwen3.5`, `Qwen3`, `Qwen2.5`, `SmolLM2`, `Mistral`, and `Microsoft Phi-4`).

---

## Core Capabilities at a Glance

| Capability | CLI / Python Entrypoint | Mathematical Guarantee & Empirical Bound |
|---|---|---|
| **Deterministic Weight-Space Reader** | `secant read --model <path> --tensor <name>` | FMA-free `float64` power iteration ($\varepsilon_\sigma = 10^{-13}$), splitmix64 seed `0x9e3779b97f4a7c15`, row-norm MAD anomaly & rank-collapse detection. |
| **Scale-Invariant Multi-Subspace Composition** | `secant compose --calibration target-shift` | Gram-Schmidt column orthogonalization ($V_\perp = V(V^\top V)^{-1/2}$) + Frobenius target-shift scaling ($\gamma_b = \alpha \frac{\tau_b \|W_\ell\|_F}{\|v_b\|_2 \|h_{\text{ref},\ell}\|_2}$). |
| **Zero-Copy SafeTensors & GGUF Surgery** | `secant patch` & `secant verify-patch` | `mmap`-backed sparse column updates; proves 100% of untouched bytes and headers outside target column spans are bit-identical (`bytes_differing_outside_edited_tensors = 0`). |
| **Direct Quantized Block Surgery (`Q8_0` / `Q4_K`)** | `secant quant patch` & `secant quant verify` | Dequantizes only touched 32-element `Q8_0` (34 B) or 256-element `Q4_K` (144 B) super-blocks, applies $\Delta W$, and re-encodes in place (`untouched_blocks_bit_identical: true`). |
| **Subspace-Locked LoRA (`Project-Then-Heal`)** | `secant.nn.ProjectThenHealLinear` & `secant heal` | Double orthogonal projection $B_{\text{locked}} = \PiPerp(\PiPerp(\tilde{B}))$ where $\PiPerp = I - D^\top D$; keeps protected-subspace leakage at **`3.76e-17` (`float64`) / `9.34e-11` (`float32`)** across AdamW fine-tuning. |
| **Payload Security & Sink Auditing** | `secant probe`, `secant sanitize`, `secant scan-sinks` | Detects and strips non-zero GGUF alignment-padding smuggling, NaN/Inf payload poisoning, collapsed rows, and MoE router sink bias. |

---

## Key Empirical Findings

### 1. The $\alpha$-Dose-Response Phase Transition on `Qwen3.5-4B`

In reasoning models (`Qwen3.5-0.8B`, `Qwen3.5-2B`, `Qwen3.5-4B`), analytical crash-triage prompts frequently trap the model inside an unclosed `<think>...</think>` block until the context budget (`1,536` tokens) is exhausted (`86.7%` unclosed rate on `Qwen3.5-4B` baseline).

Applying a multi-subspace `B1+B2+B3` edit plan (`B1`: `<think>` loop termination at layers 15 & 19; `B2`: analytical engagement at layer 12; `B3`: C/C++ root-cause hypothesis priming at layer 13) with **Target-Shift Calibration** reveals a sharp **phase transition** as the global dose multiplier $\alpha$ increases:

| Arm (`Qwen3.5-4B`, `n = 30`, `1,536` max tokens) | Calibration Dose $\alpha$ | `</think>` Closed Rate | `stop = eos` (Clean Stop) | Strict Root-Cause Accuracy | Mean Tokens |
|---|---:|---:|---:|---:|---:|
| **`base`** (unedited `Qwen3.5-4B`) | `0.0x` | `4/30` (`13.3%`) | `4/30` (`13.3%`) | `2/30` (`6.7%`) | `1,429.2` |
| **`fix_g`** (uncalibrated `0.8B` $\gamma$) | uncalibrated | `5/30` (`16.7%`) | `5/30` (`16.7%`) | `3/30` (`10.0%`) | `1,401.5` |
| **`a05`** (half target-shift dose) | `0.5x` | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `0/30` (`0.0%`) | `1,536.0` |
| **`cal738`** (unit target-shift dose) | `1.0x` | `21/30` (`70.0%`) | `21/30` (`70.0%`) | `18/30` (`60.0%`) | `1,093.0` |
| **`a15`** (**1.5x target-shift dose**) | **`1.5x`** | **`30/30` (`100.0%`)** | **`30/30` (`100.0%`)** | **`19/30` (`63.3%`, `73.3%` post-`</think>`)** | **`789.0`** |

### 2. Cross-Family & Large-Model Scale-Up (`0.8B` $\to$ `14.8B` Across 6 Architectures)

Extending Takehi et al.'s `Qwen3.5-0.8B` result [@rikiyat2026wsp] across **six distinct model families and four parameter scales** (`0.8B`, `1.7B–4B`, `7B–8B`, and `14.7B–14.8B`) on AMD ROCm (`gfx1151`) and NVIDIA H100 80GB GPUs demonstrates that **Target-Shift Calibration** and **`Project-Then-Heal` (`ProjectThenHealLinear`)** are architecture-invariant:

| Model Family & Checkpoint | Params | Layers ($d_{\text{model}}$) | `B2` Calibrated Shift vs. Random | `B3` Calibrated Shift vs. Random | 60-Step Unconstrained LoRA Leakage | 60-Step `ProjectThenHealLinear` Leakage (`float64`) |
|---|---:|---:|---:|---:|---:|---:|
| **`Qwen/Qwen3.5-0.8B`** | `0.8B` | `24` (`1024`) | **`35.4x`** (`100%` `</think>` close) | **`28.1x`** (`60.0%` strict RC) | `15.61%` (`0.1561`) | **`3.76e-17`** (`9.34e-11` `F32`) |
| **`HuggingFaceTB/SmolLM2-1.7B-Instruct`** | `1.7B` | `24` (`2048`) | **`68.0x`** (`-12.50` vs `-0.18`) | **`22.2x`** (`+9.17` vs `+0.41`) | `14.20%` (`0.1420`) | **`4.21e-18`** (`9.34e-11` `F32`) |
| **`Qwen/Qwen3.5-4B`** | `4.0B` | `36` (`2560`) | **`100.0%`** `</think>` close at `1.5x` | **`73.3%`** post-`</think>` RC | `18.40%` (`0.1840`) | **`7.01e-18`** (`2.20e-10` `F32`) |
| **`mistralai/Mistral-7B-Instruct-v0.3`** | `7.2B` | `32` (`4096`) | **`663.9x`** (`+3.98` vs `-0.006`) | **`98.9x`** (`+13.94` vs `-0.14`) | `46.79%` (`0.4679`) | **`7.73e-10`** (`3.87e-10` `F32`) |
| **`Qwen/Qwen2.5-7B-Instruct`** | `7.6B` | `28` (`3584`) | **`155.1x`** (`+4.65` vs `+0.030`) | **`132.7x`** (`+20.82` vs `-0.16`) | `31.34%` (`0.3134`) | **`7.55e-10`** (`6.88e-09` `F32`) |
| **`Qwen/Qwen3-8B`** | `8.2B` | `36` (`4096`) | **`219.7x`** (`+2.19` vs `-0.010`) | **`29.1x`** (`+120.8` vs `+4.15`) | `39.53%` (`0.3953`) | **`7.14e-10`** (`6.06e-09` `F32`) |
| **`microsoft/phi-4`** | `14.7B` | `40` (`5120`) | **`326.9x`** (`+3.60` vs `-0.011`) | **`48.7x`** (`+4.24` vs `+0.087`) | `8.42%` (`0.0842`) | **`9.51e-10`** (`1.04e-08` `F32`) |
| **`Qwen/Qwen3-14B`** | `14.8B` | `40` (`5120`) | **`252.5x`** (`+2.53` vs `+0.010`) | **`492.8x`** (`+142.9` vs `-0.29`) | `41.33%` (`0.4133`) | **`5.77e-10`** (`4.68e-09` `F32`) |

### 3. Direct `Q8_0` Block Surgery Matches `BF16` Edit-Then-Quantize (`3.65e-3` Relative Frobenius Error)

Across the 4 edited `ffn_down` tensors of `Qwen3.5-0.8B` and `Qwen3.5-4B`, direct `Q8_0` block surgery (`secant quant patch`, **Path 2**) leaves **100% of untouched 34-byte `Q8_0` blocks bit-identical** to the base `Q8_0` GGUF, matches `BF16` edit-then-quantize (**Path 1**) within **`3.65e-3` (`0.365%`) relative Frobenius error** (well below the `0.02` theoretical bound), and suppresses `96.7%` of `<think>` reasoning loops at $\alpha=1.0\times$ (`29/30` closed) and **`100.0%` at $\alpha=1.5\times$ (`30/30` closed)** under live Vulkan GPU inference (`89–103 tok/s`).

---

## Documentation Structure

1. **[Quickstart & Live CLI Receipts](quickstart.md)** — Build `secant`, inspect a model, extract contrastive directions, compose a calibrated plan, patch `BF16` or `Q8_0` GGUF files, and verify bit-level invariants.
2. **[Mathematical Theory & Scale Calibration](theory-and-calibration.md)** — Formal derivations of contrastive capture, `SNR_proj` vs. `SNR_trace`, Gram-Schmidt multi-subspace orthogonalization, and Target-Shift Calibration.
3. **[Quantized Block Surgery (`Q8_0` & `Q4_K`)](quantized-surgery.md)** — The Quantization Commutation Theorem, `Q8_0` vs. `Q4_K` truncation bounds, and Path 1 vs. Path 2 direct block surgery.
4. **[Project-Then-Heal Subspace-Locked LoRA](project-then-heal.md)** — Why standard LoRA erases weight-space edits (`15.6%` leakage) and how double orthogonal projection $\PiPerp(\PiPerp(\tilde{B}))$ locks leakage to $< 10^{-16}$.
5. **[CLI, Python & C-ABI Reference](cli-and-python-reference.md)** — Complete command-line flags, exit codes, JSON schemas, `secant.nn` / `secant.capture` APIs, and `libsecant.h` FFI symbols.
6. **[Container Geometry & Payload Security](payload-security.md)** — SafeTensors and GGUF v2/v3 structural verification, alignment-gap sanitization, and MoE router sink scanning.
7. **[Technical Paper](PAPER.md)** — Complete self-contained research paper with theorems, proofs, and multi-architecture tables.
