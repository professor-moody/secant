# Secant: Weight-Space Programming & Subspace-Locked LoRA Healing

**Secant** is a standalone Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python library (`secant`, `secant.nn.ProjectThenHealLinear`) for **deterministic Weight-Space Programming (WSP)**, **direct quantized block surgery (`Q8_0` / `Q4_K`)**, and **Subspace-Locked LoRA Healing (`Project-Then-Heal`)** on SafeTensors and GGUF (v2/v3) model artifacts.

Motivated by recent open-weight representation-engineering and circuit-breaker research from MIT and collaborators [@zou2023representation; @zou2024circuitbreakers; @arditi2024refusal] alongside Rikiya T.'s practical formulation of direct weight-space programming [@rikiyat2026wsp], Secant turns behavioral direction surgery into a **reproducible, hash-bracketed systems discipline**.

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

### 2. Cross-Family Generalization (`SmolLM2-1.7B-Instruct`)

On a dense Llama-style transformer from a completely independent training lineage (`HuggingFaceTB/SmolLM2-1.7B-Instruct`, 24 layers, $d_{\text{model}} = 2048$, $d_{\text{ffn}} = 8192$), contrastive direction capture and target-shift composition yield:
- **`68.0x` higher held-out activation shift** on `B2` (`-12.502` vs `-0.184` for an equal-Frobenius-norm random control).
- **`22.2x` higher held-out activation shift** on `B3` (`+9.167` vs `+0.413` for an equal-Frobenius-norm random control).
- **`4.21e-18` (`float64`) / `9.34e-11` (`float32`)** subspace leakage during `ProjectThenHealLinear` GPU healing.

### 3. Direct `Q8_0` Block Surgery Matches `BF16` Edit-Then-Quantize

Across the 4 edited `ffn_down` tensors of `Qwen3.5-0.8B`, direct `Q8_0` block surgery (`secant quant patch`, **Path 2**) leaves all **`240,640` untouched 34-byte blocks 100% bit-identical** to the base `Q8_0` GGUF and matches the dequantized weights of editing in `BF16` first and running `llama-quantize Q8_0` (**Path 1**) to within **`0.18% – 0.25%` relative Frobenius norm** (`mean_abs_diff` $= 1.53 \times 10^{-6}$ to $2.85 \times 10^{-6}$ on layers 13, 15, and 19).

---

## Documentation Structure

1. **[Quickstart & Live CLI Receipts](quickstart.md)** — Build `secant`, inspect a model, extract contrastive directions, compose a calibrated plan, patch `BF16` or `Q8_0` GGUF files, and verify bit-level invariants.
2. **[Mathematical Theory & Scale Calibration](theory-and-calibration.md)** — Formal derivations of contrastive capture, `SNR_proj` vs. `SNR_trace`, Gram-Schmidt multi-subspace orthogonalization, and Target-Shift Calibration.
3. **[Quantized Block Surgery (`Q8_0` & `Q4_K`)](quantized-surgery.md)** — The Quantization Commutation Theorem, `Q8_0` vs. `Q4_K` truncation bounds, and Path 1 vs. Path 2 direct block surgery.
4. **[Project-Then-Heal Subspace-Locked LoRA](project-then-heal.md)** — Why standard LoRA erases weight-space edits (`15.6%` leakage) and how double orthogonal projection $\PiPerp(\PiPerp(\tilde{B}))$ locks leakage to $< 10^{-16}$.
5. **[CLI, Python & C-ABI Reference](cli-and-python-reference.md)** — Complete command-line flags, exit codes, JSON schemas, `secant.nn` / `secant.capture` APIs, and `libsecant.h` FFI symbols.
6. **[Container Geometry & Payload Security](payload-security.md)** — SafeTensors and GGUF v2/v3 structural verification, alignment-gap sanitization, and MoE router sink scanning.
7. **[Technical Paper](PAPER.md)** — Complete self-contained research paper with theorems, proofs, and multi-architecture tables.
