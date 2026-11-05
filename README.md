<div align="center">

# Secant

**Weight-Space Programming, Direct Quantized Surgery, and Subspace-Locked LoRA (`Project-Then-Heal`)**

[![Documentation](https://img.shields.io/badge/Docs-professor--moody.github.io%2Fsecant-0A66C2?style=flat-square)](https://professor-moody.github.io/secant/)
[![Technical Paper](https://img.shields.io/badge/Paper-MyST_%2F_Quarto-8A2BE2?style=flat-square)](https://professor-moody.github.io/secant/PAPER)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange?style=flat-square)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)](https://www.python.org/)

**[Documentation](https://professor-moody.github.io/secant/)** · **[Quickstart](https://professor-moody.github.io/secant/quickstart)** · **[Theory & Calibration](https://professor-moody.github.io/secant/theory-and-calibration)** · **[Quantized Surgery](https://professor-moody.github.io/secant/quantized-surgery)** · **[Project-Then-Heal](https://professor-moody.github.io/secant/project-then-heal)** · **[Technical Paper](https://professor-moody.github.io/secant/PAPER)**

</div>

---

## Why Secant?

In [*"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, Nov 2026), Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab demonstrated that frontier models can inspect raw transformer weights and edit behavioral circuits directly—such as suppressing `<think>` reasoning loops in `Qwen3.5-0.8B` by modifying `0.016%` of its parameters. In their **"What Comes Next?"** section, they called for:

> *"Turning what currently seems like black magic into a rigorous, reproducible engineering discipline"* and *"auditing without evaluation (e.g., backdoors directly from weights)."*

**We took this on and decided to see what we could do.**

**Secant** is a standalone Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python library (`secant.nn.ProjectThenHealLinear`) for deterministic weight-space reading, scale-calibrated multi-subspace surgery (`0.8B` to `14.8B+`), direct in-place `Q8_0` / `Q4_K` GGUF block editing, static tensor payload security auditing, and subspace-locked LoRA fine-tuning.

---

## What Secant Does

| Capability | CLI / Python | What It Solves |
|---|---|---|
| **Weight-Space Reading** | `secant inspect`<br>`secant read` | Deterministic non-FMA `float64` SVD over `.safetensors` and `.gguf` tensors with direct vocabulary unembedding projection (`--vocab-tensor`). |
| **Calibrated Multi-Subspace Surgery** | `secant compose`<br>`secant patch`<br>`secant verify-patch` | Two-pass `float64` Gram-Schmidt orthogonalization + **Target-Shift Calibration** ($\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$) across model widths, with zero-copy `mmap` patching and bit-exact verification of all untouched bytes. |
| **Direct Quantized Surgery** | `secant quant patch`<br>`secant quant witness` | Edits 34-byte `Q8_0` blocks directly inside compressed `.gguf` files (`untouched_blocks_bit_identical: true`, `0.365%` Frobenius residual vs. `BF16` edit-then-quantize) and proves $\Pi_{128}$ block-grid equivariance. |
| **Subspace-Locked LoRA (`Project-Then-Heal`)** | `secant.nn.ProjectThenHealLinear`<br>`secant heal`<br>`secant verify-subspace` | Prevents post-surgical LoRA + AdamW fine-tuning from erasing weight-space edits (`8.4%–46.8%` unconstrained leakage $\to$ **`< 10^{-9}`** in `float64` / **`< 10^{-8}`** in `F32`). |
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

# Compose an orthogonal multi-subspace plan and patch in-place (BF16, F32, or Q8_0 GGUF)
secant compose --profile profile.json --vectors vec/ --out plan.json --json
secant quant patch --base model-q8_0.gguf --plan plan.json --out model-patched-q8_0.gguf --json

# Lock subsequent LoRA fine-tuning away from the edited subspace
secant heal --adapter lora_adapter.safetensors --basis basis.json --out lora_locked.safetensors --json
```

---

## Headline Empirical Results (`0.8B` to `14.8B` Across 6 Model Families)

Full derivations, SHA-256 artifact manifests, and reproduction scripts are in the **[Documentation](https://professor-moody.github.io/secant/)** and **[Technical Paper](https://professor-moody.github.io/secant/PAPER)**:

| Model Family & Checkpoint | Params | Calibrated `B2` Shift vs. Random | Calibrated `B3` Shift vs. Random | Standard LoRA Erasure (60 steps) | `ProjectThenHealLinear` Leakage (`float64`) |
|---|---:|---:|---:|---:|---:|
| **`Qwen/Qwen3.5-0.8B`** | `0.8B` | **`35.4x`** (`100%` `</think>` close) | **`28.1x`** (`60.0%` strict RC) | `15.61%` | **`3.76e-17`** |
| **`HuggingFaceTB/SmolLM2-1.7B-Instruct`** | `1.7B` | **`68.0x`** | **`22.2x`** | `14.20%` | **`4.21e-18`** |
| **`Qwen/Qwen3.5-4B`** | `4.0B` | **`100.0%`** `</think>` close (`1.5x` dose) | **`73.3%`** post-`</think>` RC | `18.40%` | **`7.01e-18`** |
| **`mistralai/Mistral-7B-Instruct-v0.3`** | `7.2B` | **`663.9x`** | **`98.9x`** | `46.79%` | **`7.73e-10`** |
| **`Qwen/Qwen2.5-7B-Instruct`** | `7.6B` | **`155.1x`** | **`132.7x`** | `31.34%` | **`7.55e-10`** |
| **`Qwen/Qwen3-8B`** | `8.2B` | **`219.7x`** | **`29.1x`** | `39.53%` | **`7.14e-10`** |
| **`microsoft/phi-4`** | `14.7B` | **`326.9x`** (`cos = 0.9971`) | **`48.7x`** | `8.42%` | **`9.51e-10`** |
| **`Qwen/Qwen3-14B`** | `14.8B` | **`252.5x`** | **`492.8x`** | `41.33%` | **`5.77e-10`** |

---

## Documentation

The full scientific documentation is hosted at **[https://professor-moody.github.io/secant/](https://professor-moody.github.io/secant/)**:

1. **[Overview & Empirical Summary](https://professor-moody.github.io/secant/)** — Capabilities, $\alpha$-dose-response phase transition (`0% -> 70% -> 100%`), and cross-family table.
2. **[Quickstart & Live CLI Receipts](https://professor-moody.github.io/secant/quickstart)** — End-to-end CLI and Python examples with SHA-256 bracketed JSON receipts.
3. **[Mathematical Theory & Scale Calibration](https://professor-moody.github.io/secant/theory-and-calibration)** — Contrastive capture, `SNR_proj`, Gram-Schmidt orthogonalization, and Target-Shift Calibration.
4. **[Quantized Block Surgery (`Q8_0` & `Q4_K`)](https://professor-moody.github.io/secant/quantized-surgery)** — The $\Pi_{128}$ Quantization Commutation Theorem and direct 34-byte `Q8_0` block editing.
5. **[Project-Then-Heal Subspace-Locked LoRA](https://professor-moody.github.io/secant/project-then-heal)** — Why AdamW rotates gradients back into excised subspaces and how $B = P_\perp \tilde{B}$ locks leakage to $< 10^{-9}$.
6. **[CLI, Python & C-ABI Reference](https://professor-moody.github.io/secant/cli-and-python-reference)** — Flags, exit codes, JSON schemas, `secant.nn`, and `secant.h` FFI symbols.
7. **[Container Geometry & Payload Security](https://professor-moody.github.io/secant/payload-security)** — Boundary-3 static auditing (`sanitize`, `probe`) for `.safetensors` and `.gguf` files.
8. **[Technical Paper](https://professor-moody.github.io/secant/PAPER)** — Complete self-contained paper with formal propositions and proofs.

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
