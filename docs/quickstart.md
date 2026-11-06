# Quickstart & Live CLI Receipts

This guide walks through building `secant` from source and executing **Probe $\to$ Inspect $\to$ Read $\to$ Sanitize $\to$ Verify** on a deterministic fixture with real JSON receipts, followed by the exact command shapes for **Compose $\to$ Patch $\to$ Heal $\to$ Verify** on your own model.

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

## 2. Generate a Deterministic Fixture and Inspect It (`secant probe`, `secant inspect`)

Every output on this page through Section 3 comes from a real run of the built binary against a fixture that `secant probe` generates byte-deterministically, so you can reproduce each value on your own machine in seconds. `probe` requires `--out-dir`, `--receipt` and `--attempt-id`:

```bash
secant probe --class class3 --format safetensors \
  --out-dir fixtures --receipt probe-receipt.json --attempt-id quickstart-1
# probe: generated; 1 fixture(s) in fixtures; receipt probe-receipt.json
#   class3-nan-sort-arrays.safetensors  findings findings=90 sha256=22ffbb02b3e4d9c5560967bc7908f6991629fae9edf5d23c2b4419731555d503

secant inspect --model fixtures/class3-nan-sort-arrays.safetensors --json
```

```json
{
  "schema": "secant.wsp.inspect.v1",
  "path": "fixtures/class3-nan-sort-arrays.safetensors",
  "format": "safetensors",
  "file_size": 736,
  "block_count": 3,
  "data_offset": 352,
  "tensor_count": 3,
  "dtype_counts": { "BF16": 1, "F16": 1, "F32": 1 },
  "tensors": [
    { "name": "blk.0.router_scores.weight", "dtype": "F32", "shape": [6, 8],
      "elements": 48, "byte_size": 192, "file_offset": 352 }
  ]
}
```

(The `tensors` array is truncated to its first entry here; `read` output below is abridged the same way.)

---

## 3. Read, Sanitize, Re-Read and Verify (`secant read`, `secant sanitize`, `secant verify-patch`)

`secant read` memory-maps the file, hashes it before and after (`input_unchanged`), and refuses to compute a spectrum for a tensor holding NaN or Inf:

```bash
secant read --model fixtures/class3-nan-sort-arrays.safetensors \
  --top-k-singular 1 --top-cols 4 --detect-anomalies --json
```

```json
{
  "schema": "secant.wsp.read.v1",
  "format": "safetensors",
  "file_bytes": 736,
  "pre_sha256": "22ffbb02b3e4d9c5560967bc7908f6991629fae9edf5d23c2b4419731555d503",
  "post_sha256": "22ffbb02b3e4d9c5560967bc7908f6991629fae9edf5d23c2b4419731555d503",
  "input_unchanged": true,
  "tensors_in_file": 3,
  "tensors_selected": 3,
  "tensors_analyzed": 0,
  "tensors": [
    { "name": "blk.0.router_scores.weight", "dtype": "F32", "shape": [6, 8],
      "status": "nonfinite", "nonfinite_elements": 30,
      "notes": ["Frobenius norm and spectrum are undefined for a tensor holding NaN/Inf; use `secant sanitize` to locate them"] }
  ]
}
```

`secant sanitize` scans the values. Without a remediation flag it only reports (exit `1`: findings remain). `--out` without a remediation flag is refused (exit `2`: `remediation flags and a separate --out must be supplied together`). With `--clamp-nonfinite --out` it publishes a clean copy (exit `0`):

```bash
secant sanitize --model fixtures/class3-nan-sort-arrays.safetensors \
  --receipt sanitize-receipt.json --attempt-id quickstart-1
# exit 1; receipt: "status": "findings", "counts": {"inf": 18, "nan": 72}

secant sanitize --model fixtures/class3-nan-sort-arrays.safetensors \
  --receipt clean-receipt.json --attempt-id quickstart-1 \
  --clamp-nonfinite --out clean.safetensors
# exit 0; receipt: "status": "remediated", "output_published": true
# clean.safetensors sha256 1e0b8f013a3f2e78e291fb1976f3b24372b3492716a88dbe64a7137df4f2a403

secant read --model clean.safetensors --tensor 'blk.0.*' --top-k-singular 1 --top-cols 2 --json
# "status": "ok", "frobenius": 24494.897662574545,
# "singular": [{"sigma": 24494.89743144648, "iterations": 3, "converged": true, "residual": 2.715678131663147e-16}]

secant verify-patch --base fixtures/class3-nan-sort-arrays.safetensors --patched clean.safetensors \
  --expect-changed blk.0.router_scores.weight,blk.1.router_scores.weight,blk.2.router_scores.weight --json
# "header_bytes_identical": true, "other_tensors_byte_identical": true, "ok": true (exit 0)
```

Naming only `blk.0.router_scores.weight` in `--expect-changed` makes the same `verify-patch` fail with `unexpected changed tensors: [blk.1.router_scores.weight blk.2.router_scores.weight]` (exit `1`), because the sanitizer also repaired the other two tensors.

These runs were captured with the `secant` binary built from this repository on 2026-11-05; the probe fixture and the sanitized copy are byte-deterministic, so the digests above are the values to expect.

---

## 4. Capture Contrastive Behavioral Directions (`secant.capture`)

Use `secant.capture.capture_contrastive_directions` to extract unit-norm behavioral directions $v_b \in \mathbb{R}^{d_{\text{model}}}$ from paired contrastive prompts and evaluate held-out `snr_proj`, `snr_trace`, and `split_half_cosine` on a deterministic 50/50 split:

```python
from secant.capture import capture_contrastive_directions, write_vectors

# activations: {(stream, layer): (pos [n, d_model], neg [n, d_model])}
results = capture_contrastive_directions(activations=activations, seed=20261104)

for (stream, layer), r in results.items():
    print(f"{stream} L{layer}: {r.status} snr_proj={r.snr_proj:.3f} snr_trace={r.snr_trace:.3f}")

# Writes d_<stream>_L<layer>.npy (NumPy v1.0, float64) for `secant compose --vectors` and `secant heal --vectors`
write_vectors("./vectors", results)
```

---

## 5. Compose a Scale-Calibrated Multi-Subspace Edit Plan (`secant compose`)

To apply multiple behavioral edits (`B1`, `B2`, `B3`) together, `secant compose` orthogonalizes the directions in `float64` and calibrates each step. `--target-shift` selects Target-Shift Calibration ($\beta = \lvert\Delta_{\text{target}}\rvert / \lVert h_S\rVert_2^2$, which needs the `h_*` reference activations in the vectors directory); `--gamma` selects relative Frobenius calibration and measures $\lVert W\rVert_F$ from `--base`. With neither flag, the profile's default applies.

```bash
secant compose \
  --profile security-auditor-b1b2b3 \
  --vectors ./vectors \
  --target-shift -4.0 \
  --out plan-b1b2b3.json \
  --json
```

Built-in profiles: `adversarial-moe-router-probe`, `adversarial-nan-probe`, `refusal-ablate`, `security-auditor-b1b2b3`, `think-loop-fix`; any other value is read as a path to a `secant.wsp.profile.v1` JSON file. There is no `secant profiles` or `secant analyze` subcommand.

---

## 6. Apply & Verify `BF16` or Direct `Q8_0` Surgery

### Option A: Sparse Column Surgery on `BF16` / `F32` / `F16` (`secant patch`)

```bash
secant patch \
  --base Qwen3.5-4B-BF16.gguf \
  --plan plan-b1b2b3.json \
  --out Qwen3.5-4B-BF16-patched.gguf \
  --json

secant verify-patch \
  --base Qwen3.5-4B-BF16.gguf \
  --patched Qwen3.5-4B-BF16-patched.gguf \
  --expect-changed <comma-separated tensors the plan edits> \
  --json
```

`--expect-changed` lists the tensors the plan edits (as named inside the model); every other tensor must be byte-identical or `verify-patch` exits `1`.

### Option B: Direct Block Surgery on Quantized `Q8_0` (`secant quant patch`)

When you only have a `Q8_0` GGUF on disk, `secant quant patch` re-quantizes only the 34-byte `Q8_0` blocks that hold an edited element. It is **`Q8_0` only**: a tensor of any other dtype, including `Q4_K`, is refused with `tensor <name> is <dtype>, block-aligned surgery needs Q8_0`.

```bash
secant quant patch \
  --base base-pure-Q8_0.gguf \
  --plan plan-b1b2b3.json \
  --out path2-Q8_0-surgery.gguf \
  --json

secant verify-patch \
  --base base-pure-Q8_0.gguf \
  --patched path2-Q8_0-surgery.gguf \
  --expect-changed <tensors named in the plan> \
  --json
```

The `--json` receipt has schema `secant.quant.patch.v1` with fields `base`, `out`, `base_sha256`, `out_sha256`, `verify`, and one `edits[]` entry per edit carrying `id`, `tensor`, `op`, `blocks_total`, `blocks_touched`, `blocks_rewritten`, `max_requant_error` and `notes`. There is no `secant quant verify`; `verify-patch` is the byte-difference check.

---

## 7. Subspace-Locked LoRA Healing (`secant.nn.ProjectThenHealLinear` & `secant heal`)

Wrap an edited `torch.nn.Linear` layer in `ProjectThenHealLinear` during recovery fine-tuning so the LoRA update is parameterized inside the orthogonal complement $\mathcal{S}^\perp$:

```python
import torch
from secant.nn import ProjectThenHealLinear

# D: [k, d_out] tensor, NumPy array, or path to a .npy of protected directions
healed_layer = ProjectThenHealLinear.from_linear(
    base_linear,
    rank=16,
    directions=D,
    alpha=16.0,
    module_name="model.layers.15.mlp.down_proj",
)

# Train lora_A and lora_B_tilde with standard AdamW
optimizer = torch.optim.AdamW(
    [healed_layer.lora_A, healed_layer.lora_B_tilde], lr=2e-4
)

# max |D B_locked| in float64; stays at the rounding level of the working dtype
print(healed_layer.leakage())

# Write lora_A / locked lora_B as one .safetensors for `secant heal --lora`
info = healed_layer.export_lora("lora_locked.safetensors")
print(info["scale"])  # pass to `secant heal --scale`
```

To lock an existing LoRA adapter offline (`lora_A`/`lora_B` pairs in one `.safetensors`), optionally merge it into a copy of the base, and verify the merged weights:

```bash
secant heal \
  --lora lora_adapter.safetensors \
  --vectors ./vectors \
  --out lora_locked.safetensors \
  --base patched.safetensors \
  --merge-out healed.safetensors \
  --scale 1.0 \
  --json

secant verify-subspace \
  --base patched.safetensors \
  --healed healed.safetensors \
  --vectors ./vectors \
  --tensors model.layers.15.mlp.down_proj.weight \
  --tol 1e-5 \
  --json
```

`heal` exits `1` and publishes nothing if the float64 residual is not below `1e-12`, the residual re-read from the written file is not below `1e-6`, or the merged weights exceed `--tol`. `verify-subspace` exits `0` (pass), `1` (leak above `--tol`), or `2` (unknown: missing, mis-shaped or non-finite tensor; never pass).
