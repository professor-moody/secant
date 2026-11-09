<div align="center">

<img src="docs/assets/secant-logo.svg" alt="Secant Logo" width="84" height="84" />

# Secant

**Deterministic Weight-Space Programming, In-Place Quantized Block Surgery (`Q8_0`), Static Checkpoint Auditing, and Subspace-Locked LoRA (`Project-Then-Heal`)**

[![Documentation](https://img.shields.io/badge/Docs-professor--moody.github.io%2Fsecant-0A66C2?style=flat-square)](https://professor-moody.github.io/secant/)
[![Technical Paper](https://img.shields.io/badge/Paper-MyST_%2F_Quarto-8A2BE2?style=flat-square)](https://professor-moody.github.io/secant/paper)
[![Blog Series](https://img.shields.io/badge/Blog_Series-m00dy.sh-12704f?style=flat-square)](https://m00dy.sh/notes/sparks-to-secant-weight-space-programming-series)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange?style=flat-square)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)](https://www.python.org/)

**[Documentation](https://professor-moody.github.io/secant/)** · **[4-Part Blog Series](https://m00dy.sh/notes/sparks-to-secant-weight-space-programming-series)** · **[Quickstart](https://professor-moody.github.io/secant/quickstart)** · **[Theory & Calibration](https://professor-moody.github.io/secant/theory-and-calibration)** · **[Quantized Surgery](https://professor-moody.github.io/secant/quantized-surgery)** · **[Project-Then-Heal](https://professor-moody.github.io/secant/project-then-heal)** · **[Payload Security](https://professor-moody.github.io/secant/payload-security)** · **[Technical Paper](https://professor-moody.github.io/secant/paper)**

<br/>

<img src="docs/assets/secant-pipeline.svg" alt="Secant Four-Stage Architecture" width="100%" />

</div>

---

## Overview

In [*"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, Nov 2026), Takehi, Teoh, Zhang, and Khattab showed that an LLM agent (`gpt-6-astra`) could inspect activations and edit `0.016%` of `Qwen3.5-0.8B`'s weights to lift its `MATH-500` score from `7.9%` to `48.1%`, while calling for *"turning what currently seems like black magic into a rigorous, reproducible engineering discipline"* and *"auditing without evaluation."* Follow-up reproductions in the community quickly noted that disabling `<think>` on unedited `Qwen3.5-0.8B` achieved the same score (`51.5%` vs. `51.2%`) and that uncalibrated trial-and-error edits were brittle across seeds (`12.9%`).

**Secant** (`v0.2.0`) is a standalone Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and PyTorch library (`secant.nn.ProjectThenHealLinear`, `secant.capture`) built to test and productionize weight-space programming across **`17` models (`0.8B` to `32.8B`)** and **`59,440` H100 benchmark evaluations**, with a separate `17`-checkpoint Project-Then-Heal study reaching `72.7B` on single shards. Every operation is deterministic and bracketed by pre/post SHA-256 receipts.

---

## What We Found Across the Benchmarks

| Benchmark / Test | What We Tested | Result |
| :--- | :--- | :--- |
| **`Qwen3.5-0.8B` `MATH-500` 4-Arm Control** (`n = 100`, `4,096` tokens) | `enable_thinking=True` vs. `enable_thinking=False` vs. `0.016%` `B1` weight edit vs. norm-matched random edit (`qwen35_08b_thinking_off_vs_edit_receipt.json`, `sha256 bc3c982e...c13b`) | `base_thinking_on` timed out on `99%` (`1.0%` final / `12.0%` raw). `base_thinking_off` jumped to **`48.0%`** (matching Takehi et al.'s `48.1%`). With `<think>` kept on, `secant_b1_edit` closed **`64%`** of `<think>` traces (**`24.0%`** final / **`31.0%`** raw, solving `5` items `thinking_off` missed), while `random_norm_matched` closed **`2%`** (**`0.0%`** final, exact McNemar $p = 1.19 \times 10^{-7}$). |
| **Why Mild `<think>` Edits Tie Random Noise (`GSM8K` Control)** | Targeted weight edit vs. norm-matched random weight edit on `Qwen3-8B` (where 98% of baseline runs time out inside `<think>`) | On shallow `GSM8K` loops, both the targeted edit (**`18% -> 30%`**) and a **random weight edit of the same norm (`18% -> 30%`)** gave the exact same boost simply by shortening `<think>` chains. Always run a random control. |
| **`<think>` Loop Closure Across Scale** (`Qwen3.5-4B`, `n = 30`) | Copying our `0.8B` Frobenius-norm cap (`30%`) to `4B` vs. calibrating step size from reference activations (`secant analyze` + `secant compose`) | Baseline and copied `0.8B` ratio closed **`0 / 30`** loops. Activation-calibrated surgery closed **`30 / 30`** loops (`19 / 30` strict root cause, `p = 1.86e-9`) while two random controls closed **`0 / 30`**. |
| **General Benchmark Retention** (`14` models, `0.8B` to `32.8B`) | Does weight surgery alone boost math/code on non-looping models (`GSM8K`, `MATH-500`, `MMLU-Pro`, `HumanEval`)? | **No.** Without LoRA fine-tuning, calibrated edits preserve baseline capability within **`0.5` to `2.0` points** across `14` models. |
| **Subspace-Locked LoRA (`ProjectThenHealLinear`)** | Fine-tuning with LoRA while locking the adapter out of edited directions (`15` models on H100 + `24,000`-eval RMS sweep) | Standard gradient projection leaks **`0.33%` to `1.67%`** within 60 AdamW steps. `ProjectThenHealLinear` holds leakage to **`10^-10` (`F32`) / `10^-17` (`float64`)** while lifting `Qwen2.5-Math-7B` `GSM8K` **`87% -> 93%`** and `Qwen2.5-Coder-7B` `HumanEval` **`36% -> 42%`**. |
| **Paired C/C++ Vulnerability Auditing (`ASAN-120`)** | `60` vulnerable C/C++ functions paired with `60` patched benign twins across `9` models x `7` arms (`7,560` evaluations) | Small non-reasoning models (`0.8B` to `8B`) collapse to one-sided guessing (`BA = 0.475..0.533`). On reasoning models, calibrated surgery beats random controls on **`4 / 4` models**, lifting `Qwen3-14B` strict root-cause recall from **`45%` to `55%`**. |
| **Quantized GGUF Block Surgery** (`Q8_0` vs. `Q4_K`) | In-place `mmap` editing of 34-byte `Q8_0` blocks (`secant patch` / `secant quant patch`) vs. 256-weight `Q4_K` superblocks | Four `32`-weight `Q8_0` blocks fit one `128`-column head panel exactly (`128 = 4 x 32`). Direct block surgery removed **`98.5%`** of the targeted weight component (`0.5172 -> 0.0079`) and left all `240,640` untouched blocks bit-identical, matching edit-then-quantize within `0.179%` to `0.923%` per tensor. A `256`-weight `Q4_K` superblock straddles two heads and retained **`67%`** of the component (`0.346`), so `secant quant patch` refuses it. |
| **Pre-Load Kernel & Trojan Auditing** (`186.8M` GGUF blocks + `7`-point dose sweep) | Role-aware Boundary-3 sanitization (`secant sanitize`) + two-stage SVD Trojan detection (`secant audit-trojan`) | Achieves **`0` false positives across `186.8M` blocks** in 10 tested production GGUF/SafeTensors checkpoints (`B3-1`, `B3-2A`, and `B3-2B` on the production corpus, plus `B3-3` validated at rule level on synthetic GPTQ/AWQ/FP8 fixtures), detects a planted whole-matrix rank-one implant once its dose passes `g* = sigma_1(W) / \|\|W\|\|_F` (trigger token in the top-2 vocabulary projection at `8%`, Rank **`#1`** at `10%`, automatic sibling flag at `12%`, on `SmolLM2-1.7B` and `Qwen3.5-4B`), and adds a Stage-2 column-localized excess check for sparse `0.50%..4%` implants. |

---

## Core Engineering Capabilities

Full mathematical derivations and reproducibility receipts are published in the **[Documentation](https://professor-moody.github.io/secant/)**, **[Technical Paper](https://professor-moody.github.io/secant/paper)**, and **[4-Part Blog Series](https://m00dy.sh/notes/sparks-to-secant-weight-space-programming-series)**.

### 1. Activation-Calibrated Weight Surgery & Multi-Subspace Composition (`secant analyze`, `secant compose`)
- **Top-K Column-Sparse Rank-1 Surgery:** Instead of modifying every column of an MLP writer matrix `W`, Secant selects the top-`K` (`K = 64` or `128`) columns `S` ordered by reference activation magnitude `|h_ref|` and solves analytically for the step size `beta = shift / ||h_S||_2^2`. Across 15 model architectures (`0.8B` to `32.8B`), sparse updates touch `< 2%` of layer weights while realizing the exact target shift at `h_ref` to within `7.99e-15` in `float64`.
- **Two-Pass Float64 Gram-Schmidt Composition:** Sequential rank-1 edits along correlated directions (`B1` loop suppression, `B2` false-refusal removal, `B3` root-cause focus) overwrite one another if applied naively. `secant compose` orthonormalizes the row basis `D` via two-pass modified Gram-Schmidt in `float64` (`max |D D^T - I_k| < 1e-15`) with upper-triangular `R^-1` cross-projection compensation so simultaneous edits do not interfere.

### 2. Direct In-Place `mmap` & 34-Byte `Q8_0` Quantized Block Surgery (`secant patch`, `secant quant patch`)
- **Sub-Tensor Quantized Block Editing on Disk:** Rather than loading full `BF16` models into GPU RAM and re-quantizing multi-gigabyte checkpoints from scratch, `secant quant patch` memory-maps `.gguf` files directly on disk, decodes only the 34-byte `Q8_0` blocks (2-byte `f16` scale `d_b` + 32 `int8` quants) that intersect the sparse column support `S`, applies the rank-1 update, and re-packs `d_b = max_i |v_i| / 127` in place. Over `98%` of the file remains bit-for-bit untouched, verified by `secant verify-patch`.
- **The `Pi_128` Head-Alignment Boundary (`Q8_0` vs. `Q4_K`):** Because each 128-column attention head divides into four 32-weight `Q8_0` blocks (`128 = 4 x 32`), `Q8_0` block surgery stays local to the edited head. By contrast, a 256-weight `Q4_K` superblock spans two neighboring 128-column heads with a shared scale; `secant quant witness` measures this cross-head coupling and `secant quant patch` refuses non-`Q8_0` tensors.

### 3. `Project-Then-Heal`: Subspace-Locked LoRA Under AdamW (`secant.nn.ProjectThenHealLinear`, `secant heal`)
- **Why Gradient Projection Alone Leaks Under AdamW:** When fine-tuning an edited model with standard LoRA (`dW = (alpha / r) B A`), unconstrained updates write back into `span(D)` (`0.82% - 8.51%` relative leakage), erasing the surgical edit. Even if you project the gradient before `AdamW.step()`, AdamW divides each coordinate by its running second-moment estimate `sqrt(v_hat_t) + eps`. That coordinate-wise scaling rotates the update back into the forbidden subspace within 60 steps (`0.33% - 1.67%` leakage across 15 models).
- **Forward-Pass Double Projection (`P_perp(P_perp B_tilde)`):** `ProjectThenHealLinear` locks the LoRA output factor inside the forward pass via a twice-applied `float64` projection `B_locked = P_perp(P_perp(B_tilde))` evaluated in `O(k * d_out * r)` time. Leakage stays at the working-dtype rounding floor: **`5.21e-18` to `9.87e-18`** max-absolute (`3.18e-17` to `5.07e-17` relative) in `float64` and **`1.78e-10` to `8.67e-10`** in `float32`.

### 4. Two-Stage Static Spectral Trojan Detection (`secant audit-trojan` & `secant analyze --detect-trojan`)
- **Stage 1 (Whole-Matrix Spectral Energy & Marchenko-Pastur Bulk Peeling Above `g* = sigma_1(W) / ||W||_F`):** Measures top singular energy fraction (`f_1 = sigma_1^2 / ||W||_F^2`), Marchenko-Pastur spike ratio (`sigma_1 / sigma_MP`), and architectural sibling modified Z-score (`blk.*.ffn_down.weight` / `model.layers.*.mlp.down_proj.weight`). Whole-matrix spectral metrics fire once a rank-one perturbation exceeds the nominal singular-value equality reference `g* = sigma_1(W) / ||W||_F` (`8.65%` on `SmolLM2-1.7B` layer 12 and `6.57%` on `Qwen3.5-4B` layer 12; the planted edit stays submerged at `0.10%..4%` dose, the trigger token reaches Rank `#1` by `10%`, and the sibling test fires at `12%`).
- **Stage 2 (Column-Localized Excess Ratio & Vocabulary Trigger Readout):** Because a rank-one implant restricted to `K = 64` columns concentrates all `||\Delta W||_F^2` into `64 / 3584` (`1.79%`) of the columns, Stage 2 compares the top-64 column norm slice against the median column norm (`col_excess_ratio`) to catch column-sparse implants down to `0.50%` dose, and projects `u_1` onto `--vocab-tensor` to list the promoted trigger tokens.
- **Synthetic Witness Generation (`secant probe`):** Generates deterministic `.safetensors` and `.gguf` test fixtures across all four Boundary-3 failure classes so users can test `secant sanitize` and `secant audit-trojan` out of the box.

### 5. Three Trust Boundaries & Role-Aware Pre-Load Sanitization (`secant sanitize`)

Every model loader crosses three distinct boundaries:
1. **Boundary 1 (Container & Framing):** Header sizes, JSON/KV metadata syntax, and file byte spans.
2. **Boundary 2 (Tensor Topology & Layout):** Dimension counts, shape products, and architecture alignment.
3. **Boundary 3 (Stored Values & Kernel Invariants):** The floating-point values, quantization scales, and integer routing indices consumed directly by C++ and CUDA kernels after `mmap`.

`secant sanitize` enforces five role-aware pre-load rules across `.safetensors` and all 10 GGUF quantization formats:

| Rule | Scope | Pre-load rule |
| :--- | :--- | :--- |
| **S-0** | Container headers and `config.json` | Checked 64-bit spans, matching tensor shapes, scalar `numel == 1`, and `1 <= expert_used_count <= num_experts`. |
| **B3-1** | Unquantized float weights, norms, and router biases | Reject NaN and infinity with dtype-specific bitwise exponent checks. |
| **B3-2A** | Multiplicative GGUF block scales (`Q4_0..Q6_K` `d`, `dmin`, `m`) | Permit `0.0` and FP16 subnormals (`8.99M` occur naturally in clean `Q4_K` superblocks); flag only non-finite exponents `(bits & 0x7C00) == 0x7C00`. |
| **B3-2B** | Divisor activation and KV scales (`input_scale`, `k_scale`, `v_scale`, `*scale_inv`, `rope_freqs`) | Require a finite value at least `--divisor-floor` (default `1e-12f`) and non-zero `rope_freqs` to prevent divide-by-zero and float-to-int cast overflow (`INT32_MIN` on `x86-64` vs. `0`/`INT32_MAX` on `ARM64`). |
| **B3-3** | Routing tensors (`tid2eid`, `expert_id*`) and integer index tensors (`g_idx`, `expert_map`, `full_row_indices`, `hash_indices_table`) | Enforce two-sided bounds (`[0, bound)`, or `[-1, bound)` for `expert_map`) before array indexing, and fail closed (`exit 2`) when an upper bound is unknown. |

<p align="center">
  <img src="docs/assets/asan120-paired-audit-uplift.svg" alt="ASAN-120 paired vulnerability and benign control benchmark across 9 models and 7 steering arms" width="100%" />
</p>

<p align="center">
  <img src="docs/assets/trojan-detection-phase-transition.svg" alt="Static spectral Trojan detection phase transition across relative Frobenius doses" width="100%" />
</p>

---

## Live CLI Demos (`inspect`, `analyze`, `compose`, `patch`, `verify-patch`, `probe`)

<p align="center">
  <img src="docs/assets/secant-patch.gif" alt="secant patch and verify-patch live terminal recording" width="49%" />
  <img src="docs/assets/secant-probe.gif" alt="secant probe live terminal recording" width="49%" />
</p>

---

## Command & Library Surface

| Stage | CLI / Python Surface | Purpose |
| :--- | :--- | :--- |
| **01 · Audit & Read** | `secant inspect` · `secant read`<br>`secant sanitize` · `secant audit-trojan` · `secant probe` | Read-only `mmap` layout inspection, two-stage spectral Trojan detection with vocabulary token projection (`--vocab-tensor`), role-aware 10-format GGUF/SafeTensors sanitization, and deterministic Class I-IV witness generation. |
| **02 · Capture & Compose** | `secant.capture` · `secant analyze`<br>`secant compose` | Contrastive activation capture with held-out `SNR_proj` layer ranking, two-pass `float64` Gram-Schmidt orthogonalization, and Target-Shift / RMS-Relative calibration. |
| **03 · In-Place Patch** | `secant patch` · `secant verify-patch`<br>`secant quant patch` · `secant quant witness` | Zero-copy `mmap` surgery on `BF16`/`F16`/`F32` `.safetensors`/`.gguf` and direct 34-byte `Q8_0` GGUF block editing with SHA-256 byte-range verification. |
| **04 · Subspace-Locked LoRA** | `secant.nn.ProjectThenHealLinear`<br>`secant heal` · `secant verify-subspace` | Forward-pass left-nullspace LoRA locking (`B_locked = P_perp(P_perp(B_tilde))`) during PyTorch training and offline `.safetensors` LoRA adapter healing. |

---

## Quickstart

```sh
# 1. Build the Rust CLI and C-ABI static/shared libraries, and install secant into ~/.cargo/bin
cargo build --release --workspace
cargo install --path crates/secant-cli

# 2. Install the Python package (PyTorch ProjectThenHealLinear + contrastive capture)
pip install -e python/
```

### Audit, Patch, and Lock a Checkpoint in 3 Commands

```sh
# 1. Audit a checkpoint on disk for Boundary-3 kernel hazards and spectral Trojans without a GPU
secant sanitize --model model-Q4_K_M.gguf --receipt /tmp/sanitize.json --attempt-id quickstart --num-experts 8 --dry-run --json
secant audit-trojan --model model.safetensors --vocab-tensor model.embed_tokens.weight --top 16 --json

# 2. Compose an activation-calibrated multi-subspace plan and patch a Q8_0 GGUF checkpoint in place
secant compose --profile profile.json --vectors vec/ --target-shift -4.0 --out plan.json --json
secant quant patch --base model-q8_0.gguf --plan plan.json --out model-patched-q8_0.gguf --json

# 3. Lock a LoRA adapter away from the excised subspace, merge into the base copy, and verify orthogonality
secant heal --lora lora_adapter.safetensors --vectors vec/ --out lora_locked.safetensors --base model.safetensors --merge-out model_healed.safetensors --json
secant verify-subspace --base model.safetensors --healed model_healed.safetensors --vectors vec/ --tensors "model.layers.15.mlp.down_proj.weight" --json
```

### Conformance & Differential Oracle Testing

`cargo test --workspace` runs the standalone unit, property, and CLI integration suite (`152` tests) out of the box. Four optional environment variables enable differential parity tests against external reference binaries and source trees:

- `SECANT_ORACLE_BIN`: Path to the reference oracle binary for byte-exact receipt and exit-code parity across `inspect`, `read`, `analyze`, `compose`, `patch`, `verify-patch`, `heal`, `verify-subspace`, `probe`, `sanitize`, and `scan-sinks`.
- `SECANT_GGML_DIR`: Path to a local `llama.cpp` checkout (containing `ggml/`) to compile and link `ggml`'s C reference `dequantize_row_q8_0` / `dequantize_row_q4_K` against `secant-core`.
- `SECANT_BASELINE_BIN`: Path to a prior release `secant` binary for backward-compatibility contract verification.
- `SECANT_SINK_TREES`: Colon-separated list of C/C++/CUDA/Rust source directories compared byte-for-byte in `scan_sinks_cli`.

---

## Documentation, Blog Series & Technical Paper

- **[4-Part Blog Series ("From Sparks to Secant")](https://m00dy.sh/notes/sparks-to-secant-weight-space-programming-series)**: Plain-English walkthrough and interactive visualizers for Parts I to IV.
- **[Overview & Empirical Summary](https://professor-moody.github.io/secant/)**: Three-regime taxonomy, methodology controls, and cross-family results.
- **[Quickstart & Live CLI Receipts](https://professor-moody.github.io/secant/quickstart)**: End-to-end CLI and Python workflows with SHA-256 bracketed JSON receipts.
- **[Mathematical Theory & Scale Calibration](https://professor-moody.github.io/secant/theory-and-calibration)**: Contrastive capture, `SNR_proj`, Gram-Schmidt cross-projection compensation, and Target-Shift Calibration.
- **[Quantized Block Surgery (`Q8_0` & `Q4_K`)](https://professor-moody.github.io/secant/quantized-surgery)**: The `Pi_128` Quantization Commutation Theorem and 34-byte `Q8_0` block editing.
- **[Project-Then-Heal Subspace-Locked LoRA](https://professor-moody.github.io/secant/project-then-heal)**: Formal proof of AdamW diagonal preconditioner rotation and double-idempotent forward-pass projection.
- **[Container Geometry & Payload Security](https://professor-moody.github.io/secant/payload-security)**: Boundary-3 static auditing (`sanitize`, `audit-trojan`, `probe`) for `.safetensors` and `.gguf` files.
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

@software{keys2026secant,
  title   = {Secant: Deterministic Weight-Space Programming, In-Place Quantized Block Surgery, and Subspace-Locked LoRA},
  author  = {Keys, Nathan},
  year    = {2026},
  url     = {https://github.com/professor-moody/secant}
}
```
