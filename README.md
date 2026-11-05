# Secant

**Weight-Space Programming, Tensor-Payload Security Auditing, and `Project-Then-Heal` Subspace-Locked Training for `.safetensors` and `.gguf` Checkpoints.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)

**Secant** (`https://github.com/professor-moody/secant`) is a self-contained Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python package (`secant`) for inspecting, auditing, surgically editing, and fine-tuning large language model weights directly inside `.safetensors` and `.gguf` container files—without re-exporting checkpoints through HuggingFace or `convert_hf_to_gguf.py`.

It implements three tightly coupled capabilities:
1. **Boundary-3 Tensor Payload Security (`sanitize`, `probe`)**: Static auditing, sanitization, and adversarial boundary-value synthesis for numerical and index-domain hazards inside structurally valid `.safetensors` and `.gguf` payloads (`NaN`/`Inf`/subnormal floats, out-of-bounds or duplicate MoE/SSM index tables, and zero-norm divisor vectors).
2. **Multi-Subspace Weight-Space Programming (`inspect`, `read`, `compose`, `patch`, `verify-patch`, `quant`)**: Deterministic non-FMA `float64` singular-value spectral reading (with direct vocabulary unembedding projection), two-pass modified Gram-Schmidt orthogonalization, inner-product target-shift calibration across model scales, zero-copy `memmap2` surgical patching (`F32`, `BF16`, `F16`, and block-aligned `Q8_0`), and byte-identity verification of every untouched header and tensor.
3. **`Project-Then-Heal` Subspace-Locked Micro-Training (`heal`, `verify-subspace`, `secant.nn.ProjectThenHealLinear`)**: Forward-pass structural LoRA parameterization ($B = P_\perp \tilde{B}, P_\perp = I_{d_{\text{out}}} - D D^\top$) that holds $D^\top \Delta W = 0$ to machine precision (`< 1e-17` in `float64`, `< 1e-9` in `F32`) across multi-step AdamW backpropagation while preserving full orthogonal learning capacity.

---

## Architecture & Workspace Layout

| Component | Path | Description |
| --- | --- | --- |
| **`secant-core`** | `crates/secant-core` | Pure-Rust core library: zero-copy `memmap2` `.safetensors` and `.gguf` parser, logical `[d_out, d_in]` orientation contract, `Qwen3.5` Gated DeltaNet `ssm_out` input-axis bijection, two-pass `float64` modified Gram-Schmidt (`compose`), deterministic non-FMA `float64` power-iteration SVD (`read`), `Project-Then-Heal` double projection & container merge (`heal`, `verify-subspace`), Class I–IV payload security scanner/synthesizer (`sanitize`, `probe`), and `ggml`-compatible `Q8_0` / `Q4_K` block codecs & surgery (`quant`). |
| **`secant-cli`** | `crates/secant-cli` | Standalone `secant` CLI binary providing 10 subcommands (`inspect`, `read`, `compose`, `patch`, `verify-patch`, `sanitize`, `probe`, `heal`, `verify-subspace`, `quant`) with deterministic `--json` receipts and POSIX exit codes (`0` = pass, `1` = policy/verification gate failed, `2` = usage/I/O error). |
| **`secant-ffi`** | `crates/secant-ffi` | C ABI (`libsecant.a` / `libsecant.so` / `libsecant.dylib`) and header (`crates/secant-ffi/include/secant.h`) for embedding Secant inside C, C++, Go, or Python applications. |
| **`secant` (Python)** | `python/secant` | PyTorch and NumPy integration: `secant.nn.ProjectThenHealLinear` (`nn.Module` drop-in replacement for `nn.Linear` with structural $P_\perp$ forward projection), `secant.capture.capture_contrastive_directions` (contrastive activation extractor & per-stream SNR scorer), and `ctypes` / `subprocess` bindings. |
| **Mathematical Paper & Docs** | `docs/` | Full mathematical derivations, container layout proofs, and empirical benchmarks (`docs/PAPER.md`). |

---

## Quick Start

### 1. Build the Rust CLI & C ABI Library

```sh
cargo build --release --workspace
cargo test --workspace
```

The compiled binary is placed at `target/release/secant` (and C ABI libraries at `target/release/libsecant.{a,so,dylib}`).

### 2. Install the Python Package (`secant`)

```sh
pip install -e python/
pytest python/tests
```

---

## Core CLI Workflows

### A. Static Weight-Space Reading & Vocabulary Projection (`inspect`, `read`)

Inspect container metadata (including automatic detection of `Qwen3.5`'s `32 x 128` Gated DeltaNet `blk.N.ssm_out.weight` value-head permutation) and project the dominant singular vectors of any writer matrix through the unembedding table without loading the model onto a GPU:

```console
$ secant inspect --model qwen35-4b-base-bf16.gguf
qwen35-4b-base-bf16.gguf: gguf, 441 tensors, 8665620384 bytes, payload at +10968992
architecture qwen35, 33 blocks, hidden 2560
  BF16 257 tensor(s)
  F32  184 tensor(s)
qwen3.5 GDN ssm_out: 24 layer(s), logical [2560 x 4096] BF16
  GGUF blk.N.ssm_out.weight stores linear_attn.out_proj.weight values with the LOGICAL INPUT axis
  (value_dim = GGUF ne[0], 4096 columns = 32 value-head blocks x head_v_dim 128) permuted:
  with r = 2 value heads per key head (16 key heads), GGUF block j holds HF block 0,2,4,6,8,10,12,14 ...;
  HF block h = k*r + j lands at GGUF block j*num_k_heads + k; edit plans must map input columns accordingly

$ secant read \
    --model qwen35-4b-base-bf16.gguf \
    --tensor "blk.13.ffn_down.weight" \
    --vocab-tensor "token_embd.weight" \
    --top-k-singular 3 --top-cols 8 --detect-anomalies
read qwen35-4b-base-bf16.gguf (gguf): 1/441 tensors selected, 1 analyzed;
  sha256 6bc66ab9ac4c1d90922c59b42f372d59403fa32da89d1def6e3953299721b95c unchanged=true
  blk.13.ffn_down.weight BF16 [9216, 2560]: ok fro=4.112923e1 s1=2.650751e0 s2=2.338223e0 s3=2.018516e0
      promoted tokens:   151353(1.393e-1) 211741(1.374e-1) 180201(1.318e-1) 200228(1.276e-1) ...
      suppressed tokens: 6843(-4.152e-2) 3648(-4.042e-2) 220(-3.783e-2) 64(-3.775e-2) ...
```

### B. Multi-Subspace Composition & Offset-Addressed Binary Surgery (`compose`, `patch`, `verify-patch`)

Orthogonalize behavioral directions in `float64` (two-pass modified Gram-Schmidt, fail-closed if $\max_{i \neq j} |\langle \hat{d}_i, \hat{d}_j \rangle| \ge 10^{-12}$), calibrate edits via scale-invariant inner-product shift ($\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$), patch only the target byte ranges via `memmap2`, and verify that the header and all other tensors are bit-for-bit identical:

```sh
secant compose \
    --profile evidence/profile_cal738_ssmout.json \
    --vectors vec/real \
    --out plans/cal738.plan.json \
    --json > evidence/compose_cal738.json

secant patch \
    --base gguf/qwen35-4b-base-bf16.gguf \
    --plan plans/cal738.plan.json \
    --out gguf/qwen35-4b-cal738-bf16.gguf \
    --json > evidence/patch_cal738.json

secant verify-patch \
    --base gguf/qwen35-4b-base-bf16.gguf \
    --patched gguf/qwen35-4b-cal738-bf16.gguf \
    --plan plans/cal738.plan.json \
    --json
```

### C. Direct `Q8_0` Quantized Block Surgery & $\Pi_{128}$ Commutation Witness (`quant`)

Inspect block-grid equivariance under the 128-column `ssm_out` permutation $\Pi_{128}$ (`Q8_0` 32-element blocks divide 128 evenly; `Q4_K` 256-element superblocks straddle 128-element panels) and perform direct in-place block surgery on `Q8_0` GGUF tensors while keeping all untouched 34-byte blocks bit-identical:

```sh
# Measure Pi_128 permutation equivariance and quantization error on a tensor
secant quant witness \
    --model gguf/qwen35-4b-base-bf16.gguf \
    --tensor "blk.16.ssm_out.weight" \
    --codec q8_0 \
    --num-k-heads 16 --num-v-heads 32 --head-v-dim 128 \
    --json

# Apply an edit plan directly to a Q8_0 GGUF checkpoint
secant quant patch \
    --base gguf/qwen35-08b-q8_0.gguf \
    --plan plans/b2_q8.plan.json \
    --out gguf/qwen35-08b-q8_0-patched.gguf \
    --json
```

### D. `Project-Then-Heal` Subspace-Locked LoRA (`secant.nn.ProjectThenHealLinear`, `heal`, `verify-subspace`)

Train a LoRA adapter strictly inside the orthogonal complement $P_\perp = I_{d_{\text{out}}} - D D^\top$ of removed surgical directions so AdamW's coordinatewise second-moment preconditioning cannot write back into $\operatorname{span}(D)$:

```python
import torch
from secant.nn import ProjectThenHealLinear

# Wrap any nn.Linear writer layer with an orthonormal protected basis D [d_out, k]
layer = ProjectThenHealLinear.from_linear(
    base_linear,
    rank=8,
    alpha=16.0,
    directions=[d_loop_L15, d_loop_L19, d_ref_L12, d_root_L13],
)

# Forward pass computes y = W_0 x + (alpha / r) * (P_perp @ B_tilde) @ (A @ x)
# Guarantees D^T dW = 0 to machine precision (~1e-18 in float64, ~1e-10 in float32)
out = layer(x)
```

Audit and merge serialized LoRA `.safetensors` adapters on disk:

```sh
secant heal \
    --lora adapter_raw.safetensors \
    --vectors vec/real \
    --out-lora adapter_locked.safetensors \
    --base model.safetensors \
    --out-merged model_healed_f32.safetensors \
    --scale 2.0 --tol 1e-5 --json

secant verify-subspace \
    --base model.safetensors \
    --healed model_healed_f32.safetensors \
    --vectors vec/real \
    --tensors "model.layers.15.mlp.down_proj.weight,model.layers.16.linear_attn.out_proj.weight" \
    --tol 1e-5 --json
```

### E. Boundary-3 Tensor Payload Security (`sanitize`, `probe`)

Audit untrusted `.safetensors` and `.gguf` checkpoints for Class I–IV payload hazards before loading them into C++/CUDA/ROCm runtimes, or synthesize minimal header-valid reproducer fixtures for parser/kernel fuzzing:

```sh
# Audit and remediate a checkpoint (fail-closed exit code 1 if hazards are found)
secant sanitize \
    --model untrusted_model.gguf \
    --out clean_model.gguf \
    --policy clamp \
    --json

# Synthesize a header-valid Class II index-domain OOB probe fixture
secant probe \
    --class index-oob \
    --format safetensors \
    --out fixtures/probe_index_oob.safetensors \
    --json
```

---

## Core Mathematical & Engineering Invariants

Every invariant in Secant is enforced at runtime and covered by unit and cross-ISA regression tests:

1. **Logical `[d_out, d_in]` Orientation Contract**:
   - SafeTensors stores 2-D linear weights as `shape = [d_out, d_in]` (`shape[1]` contiguous).
   - GGUF stores 2-D linear weights as `ne = [d_in, d_out]` (`ne[0]` contiguous).
   - Both reduce to the identical flat row-major element offset `out_idx * d_in + in_idx` via `TensorLocation::weight(out_idx, in_idx)`.
2. **`Qwen3.5` Gated DeltaNet `blk.N.ssm_out.weight` Column Bijection**:
   - Derived from canonical GGUF metadata keys `{arch}.ssm.{group_count, time_step_rank, inner_size, state_size}` with $r = n_v / n_k$.
   - HuggingFace value-head 128-column block $h = k \cdot r + j$ ($k \in \{0,\dots,n_k-1\}, j \in \{0,\dots,r-1\}$) maps bijectively to GGUF block $j \cdot n_k + k$ (`24/24` layers bit-exact on `Qwen3.5-0.8B` and `Qwen3.5-4B`).
3. **Cross-ISA Bit-Identical IEEE-754 `float64` Arithmetic**:
   - All reductions in `read`, `compose`, and `heal` use separately rounded IEEE-754 `float64` operations without hardware fused multiply-add (FMA), producing identical JSON digests across `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`.
4. **Two-Pass Modified Gram-Schmidt Gate**:
   - Orthogonalization fails closed unless $\max_{i \neq j} |\langle \hat{d}_i, \hat{d}_j \rangle| < 10^{-12}$ and $\max_i |\|\hat{d}_i\|_2 - 1| < 10^{-12}$.
5. **`Project-Then-Heal` Double Projection & `BF16` Merge Gate**:
   - Double projection $B_{\text{locked}} = P_\perp (P_\perp \tilde{B})$ holds $\|D^\top B_{\text{locked}}\|_\infty \le 10^{-16}$ in `float64`.
   - `verify-subspace` measures leakage on the **realized on-disk tensor bytes** and fails closed when a `BF16` base merge introduces 7-bit mantissa rounding noise ($\sim 1.4 \times 10^{-3} > 10^{-5}$) along $D$.

---

## Citation

If you use **Secant** in your research or engineering work, please cite:

```bibtex
@software{secant2026,
  author = {Moody, Professor},
  title = {Secant: Weight-Space Programming, Tensor-Payload Security, and Subspace-Locked LoRA for SafeTensors and GGUF},
  year = {2026},
  url = {https://github.com/professor-moody/secant}
}
```

## License

Licensed under the Apache License, Version 2.0.
