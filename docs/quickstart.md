# Quickstart & Live CLI Receipts

This guide walks through building `secant` from source and executing a complete **Inspect $\to$ Read $\to$ Compose $\to$ Patch $\to$ Verify $\to$ Heal** workflow with real JSON receipts.

---

## 1. Build the Rust Workspace & Install the Python Package

```bash
# Build secant CLI and libsecant C-ABI library in release mode
# Note: Cargo.toml enforces opt-level = 3 with FMA-free float64 accumulation
cargo build --release

# Verify the binary
./target/release/secant --help

# Install the Python package (with PyTorch and pytest extras)
pip install -e "./python[torch,test]"

# Run the Rust and Python test suites
cargo test --workspace
pytest python/tests
```

---

## 2. Inspect a SafeTensors or GGUF Container (`secant inspect`)

Before reading or modifying any weights, inspect the container header, tensor table, alignment invariants, and SHA-256 digest:

```bash
secant inspect --model Qwen3.5-4B-BF16.gguf --json
```

```json
{
  "schema": "secant.wsp.inspect.v1",
  "format": "gguf",
  "version": 3,
  "alignment": 32,
  "file_bytes": 8053924416,
  "data_offset": 754176,
  "tensor_count": 427,
  "padding_clean": true
}
```

---

## 3. Read Spectral & Row-Norm Geometry directly from Weights (`secant read`)

`secant read` memory-maps the model file (`input_unchanged: true`, pre/post SHA-256 verified) and computes the leading singular triplet $(\sigma_1, u_1, v_1)$, Frobenius norm, row-norm quantiles, and top-$K$ intermediate MLP columns without loading a Python runtime:

```bash
secant read \
  --model models/SmolLM2-1.7B-Instruct/model.safetensors \
  --tensor model.layers.12.mlp.down_proj.weight \
  --top-cols 8 \
  --top-k-singular 1 \
  --json
```

```json
{
  "schema": "secant.wsp.read.v1",
  "model": "models/SmolLM2-1.7B-Instruct/model.safetensors",
  "format": "safetensors",
  "file_bytes": 3422777952,
  "pre_sha256": "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8",
  "post_sha256": "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8",
  "input_unchanged": true,
  "policy": {
    "version": "wsp-read-v1",
    "max_iterations": 1000,
    "convergence_tol_relative_sigma": 1e-13,
    "convergence_tol_residual": 1e-10,
    "start_seed": 11400714819323198485,
    "fma_free_arithmetic": true
  },
  "tensors": [
    {
      "name": "model.layers.12.mlp.down_proj.weight",
      "dtype": "BF16",
      "shape": [2048, 8192],
      "frobenius_norm": 66.224819
    }
  ]
}
```

---

## 4. Capture Contrastive Behavioral Directions (`secant.capture`)

Use `secant.capture.capture_contrastive_directions` to extract unit-norm behavioral directions $v_b \in \mathbb{R}^{d_{\text{model}}}$ from paired contrastive prompts and evaluate held-out `snr_proj`, `snr_trace`, and `split_half_cosine` on a deterministic 50/50 split:

```python
import numpy as np
from secant.capture import capture_contrastive_directions, write_vectors

# pos_acts / neg_acts: dict[(stream, layer) -> np.ndarray of shape (N, d_model)]
results = capture_contrastive_directions(pos_acts, neg_acts, seed=20261104)

for key, metric in results.items():
    print(f"{key}: snr_proj={metric.snr_proj:.3f}, cos={metric.split_half_cosine:.4f}")

# Export float64 NumPy v1.0 vectors for `secant compose` and `secant heal`
write_vectors(results, out_dir="./vectors")
```

---

## 5. Compose a Scale-Calibrated Multi-Subspace Edit Plan (`secant compose`)

To apply multiple behavioral edits (`B1`, `B2`, `B3`) simultaneously without cross-subspace interference or cross-scale decay, run `secant compose` with `--calibration target-shift`:

```bash
secant compose \
  --profile qwen35-beneficial-b1b2b3 \
  --vectors-dir ./vectors \
  --calibration target-shift \
  --alpha 1.5 \
  --model Qwen3.5-4B-BF16.gguf \
  --out plan-b1b2b3-a15.json \
  --json
```

---

## 6. Apply & Verify In-Place `BF16` or Direct `Q8_0` Surgery

### Option A: Sparse Column Surgery on `BF16` / `F32` / `F16` (`secant patch`)

```bash
secant patch \
  --model Qwen3.5-4B-BF16.gguf \
  --plan plan-b1b2b3-a15.json \
  --out Qwen3.5-4B-BF16-patched.gguf \
  --json

secant verify-patch \
  --base Qwen3.5-4B-BF16.gguf \
  --patched Qwen3.5-4B-BF16-patched.gguf \
  --plan plan-b1b2b3-a15.json \
  --json
```

### Option B: Direct Block Surgery on Quantized `Q8_0` / `Q4_K` (`secant quant patch`)

When you only have a quantized `Q8_0` or `Q4_K` GGUF artifact on disk, `secant quant patch` modifies only the quantized super-blocks intersecting the sparse top-$K$ intermediate MLP columns (`untouched_blocks_bit_identical: true`):

```bash
secant quant patch \
  --model base-pure-Q8_0.gguf \
  --plan plan-b1b2b3.json \
  --out path2-Q8_0-surgery.gguf \
  --json
```

```json
{
  "schema": "secant.wsp.quant-patch.v1",
  "base_sha256": "d9dd2c0bb0f54bb062ba008e855b29639afa47d18f7e0789551419d601806250",
  "patched_sha256": "8bcfdaede2ed8d67060cd2adb240f6d3b3836e063544cbb32c6c5f9f13046eab",
  "tensors_modified": 4,
  "blocks_rewritten": 210598,
  "untouched_blocks": 246379,
  "unmodified_blocks_bit_identical": true,
  "bytes_differing_outside_edited_tensors": 0
}
```

---

## 7. Subspace-Locked LoRA Healing (`secant.nn.ProjectThenHealLinear` & `secant heal`)

Wrap any edited `torch.nn.Linear` layer in `ProjectThenHealLinear` during recovery fine-tuning so gradients are structurally locked to the orthogonal complement $\mathcal{S}^\perp$:

```python
import torch
from secant.nn import ProjectThenHealLinear

# D: [k, d_out] tensor or NumPy array of protected behavioral directions
healed_layer = ProjectThenHealLinear.from_linear(
    base_linear,
    rank=16,
    directions=D,
    alpha=16.0,
)

# Train lora_A and lora_B_tilde with standard AdamW
optimizer = torch.optim.AdamW(healed_layer.parameters(), lr=2e-4)

# Verify machine-zero subspace leakage at any step:
assert healed_layer.leakage() < 1e-12
```

To merge an existing LoRA adapter offline while projecting out protected directions in `float64`:

```bash
secant heal \
  --model patched.gguf \
  --out healed.gguf \
  --tensor blk.15.ffn_down.weight \
  --lora-a lora_A_L15.npy \
  --lora-b lora_B_L15.npy \
  --protect d_B1a_L15.npy \
  --protect d_B2_L12.npy \
  --scale 1.0 \
  --json

secant verify-subspace \
  --base patched.gguf \
  --healed healed.gguf \
  --tensor blk.15.ffn_down.weight \
  --protect d_B1a_L15.npy \
  --protect d_B2_L12.npy \
  --max-leakage 1e-6 \
  --json
```
