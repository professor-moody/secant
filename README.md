<div align="center">

# Secant

**Deterministic Weight-Space Programming, In-Place Quantized Block Surgery (`Q8_0`), and Subspace-Locked LoRA (`Project-Then-Heal`)**

[![Documentation](https://img.shields.io/badge/Docs-professor--moody.github.io%2Fsecant-0A66C2?style=flat-square)](https://professor-moody.github.io/secant/)
[![Technical Paper](https://img.shields.io/badge/Paper-MyST_%2F_Quarto-8A2BE2?style=flat-square)](https://professor-moody.github.io/secant/paper)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange?style=flat-square)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)](https://www.python.org/)

**[Documentation](https://professor-moody.github.io/secant/)** · **[Quickstart](https://professor-moody.github.io/secant/quickstart)** · **[Theory & Calibration](https://professor-moody.github.io/secant/theory-and-calibration)** · **[Quantized Surgery](https://professor-moody.github.io/secant/quantized-surgery)** · **[Project-Then-Heal](https://professor-moody.github.io/secant/project-then-heal)** · **[Payload Security](https://professor-moody.github.io/secant/payload-security)** · **[Technical Paper](https://professor-moody.github.io/secant/paper)**

<br/>

<img src="docs/assets/secant-pipeline.svg" alt="Secant Four-Stage Architecture" width="100%" />

</div>

---

## Overview

In [*"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, Nov 2026), Takehi, Teoh, Zhang, and Khattab showed that frontier models can inspect transformer weights and edit behavioral circuits directly, calling for *"turning what currently seems like black magic into a rigorous, reproducible engineering discipline"* and *"auditing without evaluation."*

**Secant** is a standalone Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python library (`secant.nn.ProjectThenHealLinear`, `secant.capture`) for verifiable weight-space surgery and subspace-locked post-edit adaptation. Every operation is bracketed by pre/post SHA-256 receipts and explicit floating-point invariants.

---

## Core Scientific & Engineering Findings

Full mathematical derivations, regime boundaries, and SHA-256 receipts are published in the **[Documentation](https://professor-moody.github.io/secant/)** and **[Technical Paper](https://professor-moody.github.io/secant/paper)**.

### 1. Sparse Rank-1 Weight Surgery & Multi-Subspace Composition (`secant.capture`, `secant compose`)
- **Top-K Column-Sparse Rank-1 Surgery:** Instead of modifying every column of an MLP writer matrix `W`, Secant selects the top-`K` (`K = 64` or `128`) columns `S` ordered by reference activation magnitude `|h_ref|` and solves analytically for the step size `beta = shift / ||h_S||_2^2`. Across 15 model architectures (`0.6B` to `32.8B`), sparse updates touch `< 2%` of layer weights while realizing the exact target shift to within `7.99e-15` in `float64`.
- **Two-Pass Float64 Gram-Schmidt Composition:** Sequential rank-1 edits along correlated directions (`security`, `math`, `science`, `code`) overwrite one another (`0.08` to `0.31` shift drift). `secant compose` orthonormalizes the row basis `D` via two-pass modified Gram-Schmidt in `float64` (`max |D D^T - I_k| < 1e-12`, measured `< 1e-15`), eliminating cross-direction interference.
- **Scale-Calibrated Dose Response (`Qwen3.5-4B`):** Copying a fixed Frobenius norm ratio from `0.8B` to `4B` closes `0/30` `<think>` loops because wider layers have different activation scales. Target-Shift Calibration produces a clean transition on `Qwen3.5-4B` (`n = 30`, strict post-`</think>` scoring): `0/30` (`0%`) at baseline and `0.5x` dose, `21/30` (`70%`) at `1.0x`, and `30/30` (`100%`) closed with `23/30` (`76.7%`) exact culprit frame at `1.5x` (paired McNemar `p = 1.86e-9`).

### 2. Direct In-Place `mmap` & 34-Byte `Q8_0` Quantized Block Surgery (`secant patch`, `secant quant patch`)
- **Sub-Tensor Quantized Block Editing on Disk:** Rather than loading full `BF16` models into GPU RAM, editing them, and re-quantizing multi-gigabyte checkpoints from scratch, `secant quant patch` memory-maps `.gguf` files directly on disk, decodes only the 34-byte `Q8_0` blocks (2-byte `f16` scale `d_b` + 32 `int8` quants) that intersect the sparse column support `S`, applies the rank-1 update, and re-packs `d_b = max_i |v_i| / 127` in place. Over `98%` of the file remains bit-for-bit untouched (`0.18%` to `0.92%` Frobenius residual vs. full `BF16` edit-then-quantize).
- **The `Pi_128` Quantization Commutation Boundary (`Q8_0` vs. `Q4_K`):** Because each 128-column attention/SSM head divides into exactly four 32-weight `Q8_0` blocks (`128 = 4 x 32`), `Q8_0` block surgery is head-local. By contrast, a 256-weight `Q4_K` superblock spans two neighboring 128-column heads with a shared superblock scale; `secant quant witness` measures this cross-head leakage and `secant quant patch` refuses unsafe `Q4_K` straddling.

### 3. `Project-Then-Heal`: Subspace-Locked LoRA Under AdamW (`secant.nn.ProjectThenHealLinear`, `secant heal`)
- **Why Gradient Projection Alone Leaks Under AdamW (`108/108` Runs):** When fine-tuning an edited model with standard LoRA (`dW = (alpha / r) B A`) to recover fluency, unconstrained updates write back into `span(D)` (`~1e-2` leakage), erasing the surgical edit. Even when the raw gradient is projected into `ker(D)` (`g_t <- P_perp g_t`) before `AdamW.step()`, AdamW divides each coordinate by its running second-moment estimate `sqrt(v_hat_t) + eps` (a non-isotropic diagonal preconditioner `S_t^{-1}`). Because `D S_t^{-1} P_perp != 0` for non-axis-aligned directions `D`, AdamW rotates orthogonal gradients back into `span(D)` on `100%` of steps (`108/108` runs in our 6-arm, 648-run optimizer ablation).
- **Forward-Graph Double Projection (`P_perp(P_perp B_tilde)`):** Building on subspace-constrained LoRA methods (Safe LoRA, Hsu et al. 2024; SaLoRA, Li et al. ICLR 2025; OPLoRA, Xiong & Xie AAAI 2026; OGPSA, Sun et al. 2026), `ProjectThenHealLinear` locks the LoRA output factor inside the forward pass via a twice-applied `float64` projection `B_locked = P_perp(P_perp(B_tilde))` where `P_perp = I - D^T D` is evaluated as `B - D^T(D B)` in `O(k * d_out * r)` time without ever allocating a `d_out x d_out` matrix.
- **Working-Dtype Precision Floor:** For a true `float64` module (Regime A live causal-LM fine-tuning on `Qwen3.5-0.8B` and `SmolLM2-1.7B`), realized writer leakage `max |D dW|` stays at **`5.21e-18` to `9.87e-18`**. For a `float32` module whose projection is computed in `float64` and rounded back to `float32` (15-model H100 Stage-1 sweep, `0.6B` to `32.8B`), leakage stays at the `float32` rounding floor (**`1.78e-10` to `8.67e-10`**).

### 4. Paired 30-Case C/C++ Sanitizer Audit Uplift & Zero-Eval Payload Auditing
- **Downstream Vulnerability Discovery Uplift:** On our 30-case C/C++ sanitizer vulnerability suite (6 CWE-aligned bug classes), rank-1 security-invariant surgery lifts `Qwen2.5-Coder-32B-Instruct` from `6.7%` (`2/30`) to **`40.0%` (`12/30`, `+10` findings, paired exact two-sided McNemar `p = 0.00195`)** and `Gemma-2-9B-IT` from `13.3%` (`4/30`) to **`23.3%` (`7/30`, `+3` findings)** without increasing reasoning token length (`125.9 -> 122.4` mean tokens).
- **Static Weight & Container Auditing (`secant read`, `secant sanitize`, `secant probe`):** `secant read` computes deterministic non-FMA `float64` singular vectors via `mmap` and projects them onto the vocabulary unembedding matrix (`--vocab-tensor`) without a GPU, while `secant sanitize` scans `.safetensors` and `.gguf` files for `NaN`/`Inf`/subnormal floats, zero-norm rows, and out-of-bounds MoE router IDs.

---

## Summary of Key Empirical Regimes

| Regime / Experiment | Models & Scale | Primary Metric | Unconstrained / Baseline | Secant (`Project-Then-Heal` / Calibrated Surgery) |
| :--- | :--- | :--- | :--- | :--- |
| **6-Arm Optimizer Ablation** (`648` runs) | Synthetic + Layer-16 `down_proj` (`0.8B`, `4B`, `7B`) | Relative subspace leakage `||D dW||_F / ||dW||_F` | Std LoRA: `1.5e-1`<br>Grad-Proj AdamW: `1.56% - 2.98%` | **`5.21e-18 - 9.87e-18`** (`float64` module) |
| **Regime A: Live LM Healing** | `Qwen3.5-0.8B`, `SmolLM2-1.7B` | Post-surgery causal-LM healing leakage `max \|D dW\|` | `4.64e-3 - 1.05e-2` | **`5.21e-18 - 9.87e-18`** (`float64`) / **`<= 1.70e-8`** (`F32`) |
| **15-Model H100 Stage-1 Sweep** | `15` models (`0.6B` to `32.8B`, 7 families) | Exact shift error & `float32` locked LoRA leakage | Grad-Proj AdamW: `0.33% - 1.67%` | Shift error **`<= 7.99e-15`**; leakage **`1.78e-10 - 8.67e-10`** |
| **Regime B: `<think>` Calibration** | `Qwen3.5-4B` (`n = 30` crash traces) | Loop closure & exact culprit attribution | `0/30` (`0%`) at baseline & fixed-ratio | **`30/30` (`100%`)** closed; **`23/30` (`76.7%`)** culprit (`p = 1.86e-9`) |
| **30-Case C/C++ Sanitizer Suite** | `Qwen2.5-Coder-32B-Instruct` (`n = 30`) | Paired sanitizer vulnerability detection | `2/30` (`6.7%`) | **`12/30` (`40.0%`, `+10` findings, `p = 0.00195`)** |

---

## Command & Library Surface

| Stage | CLI / Python Surface | Purpose |
| :--- | :--- | :--- |
| **01 · Audit & Read** | `secant inspect` · `secant read`<br>`secant sanitize` · `secant probe` | Read-only `mmap` layout inspection, deterministic `float64` SVD with vocabulary token projection (`--vocab-tensor`), and static tensor payload sanitization. |
| **02 · Capture & Compose** | `secant.capture`<br>`secant compose` | Contrastive activation capture with held-out `SNR_proj` layer ranking, two-pass `float64` Gram-Schmidt orthogonalization, and Target-Shift Calibration (`beta = shift / ||h_S||_2^2`). |
| **03 · In-Place Patch** | `secant patch` · `secant verify-patch`<br>`secant quant patch` · `secant quant witness` | Zero-copy `mmap` surgery on `BF16`/`F16`/`F32` `.safetensors`/`.gguf` and direct 34-byte `Q8_0` GGUF block editing with SHA-256 byte-range verification. |
| **04 · Subspace-Locked LoRA** | `secant.nn.ProjectThenHealLinear`<br>`secant heal` · `secant verify-subspace` | Forward-pass left-nullspace LoRA locking (`B_locked = P_perp(P_perp(B_tilde))`) during PyTorch training and offline `.safetensors` LoRA adapter healing. |

---

## Quickstart

```sh
# 1. Build the Rust CLI and C-ABI static/shared libraries
cargo build --release --workspace

# 2. Install the Python package (PyTorch ProjectThenHealLinear + contrastive capture)
pip install -e python/
```

### Read, Patch, and Lock a Checkpoint in 3 Commands

```sh
# 1. Read top singular directions on disk and project into vocabulary tokens without a GPU
secant read --model model.gguf --tensor "blk.15.ffn_down.weight" --vocab-tensor "token_embd.weight"

# 2. Compose an orthogonal multi-subspace plan and patch a Q8_0 GGUF checkpoint in place
secant compose --profile profile.json --vectors vec/ --target-shift -4.0 --out plan.json --json
secant quant patch --base model-q8_0.gguf --plan plan.json --out model-patched-q8_0.gguf --json

# 3. Lock a LoRA adapter away from the excised subspace and verify orthogonality
secant heal --lora lora_adapter.safetensors --vectors vec/ --out lora_locked.safetensors --json
secant verify-subspace --lora lora_locked.safetensors --vectors vec/ --json
```

---

## Documentation & Technical Paper

- **[Overview & Empirical Summary](https://professor-moody.github.io/secant/)**: Three-regime taxonomy, methodology controls, and cross-family results.
- **[Quickstart & Live CLI Receipts](https://professor-moody.github.io/secant/quickstart)**: End-to-end CLI and Python workflows with SHA-256 bracketed JSON receipts.
- **[Mathematical Theory & Scale Calibration](https://professor-moody.github.io/secant/theory-and-calibration)**: Contrastive capture, `SNR_proj`, Gram-Schmidt cross-projection compensation, and Target-Shift Calibration.
- **[Quantized Block Surgery (`Q8_0` & `Q4_K`)](https://professor-moody.github.io/secant/quantized-surgery)**: The `Pi_128` Quantization Commutation Theorem and 34-byte `Q8_0` block editing.
- **[Project-Then-Heal Subspace-Locked LoRA](https://professor-moody.github.io/secant/project-then-heal)**: Formal proof of AdamW diagonal preconditioner rotation and double-idempotent forward-pass projection.
- **[Container Geometry & Payload Security](https://professor-moody.github.io/secant/payload-security)**: Boundary-3 static auditing (`sanitize`, `probe`) for `.safetensors` and `.gguf` files.
- **[CLI, Python & C-ABI Reference](https://professor-moody.github.io/secant/cli-and-python-reference)**: Flags, exit codes, JSON schemas, `secant.nn`, and `secant.h` FFI symbols.
- **[Full Technical Paper](https://professor-moody.github.io/secant/paper)**: Self-contained paper with formal propositions, proofs, and limitations.

---

## Citation

```bibtex
@article{takehi2026sparks,
  title   = {Sparks of Weight-Space Programming: LLMs can natively read \& write neural network weights},
  author  = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  year    = {2026},
  month   = nov,
  url     = {https://rikiyat.github.io/blog/weight-space-programming/}
}

@software{moody2026secant,
  title   = {Secant: Deterministic Weight-Space Programming, In-Place Quantized Block Surgery, and Subspace-Locked LoRA},
  author  = {Moody, Nathan},
  year    = {2026},
  url     = {https://github.com/professor-moody/secant}
}
```
