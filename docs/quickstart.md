# Quickstart & Live CLI Walkthrough

This guide walks through building `secant` and running its **5 core workflows** (covering all 14 CLI subcommands and the Python `secant.nn` / `secant.capture` APIs), from reading weight concepts directly into vocabulary tokens without a GPU to in-place 34-byte `Q8_0` GGUF block surgery and `ProjectThenHealLinear` subspace-locked LoRA training.

---

## 1. Build & Install in 60 Seconds

```bash
# 1. Build the Rust CLI (secant) and C-ABI library (libsecant) in release mode
cargo build --release --workspace

# 2. Verify all 14 subcommands
./target/release/secant --help

# 3. Install the Python package (PyTorch ProjectThenHealLinear + contrastive capture)
pip install -e "./python[torch,test]"

# 4. Optional: run the Rust and Python test suites
cargo test --workspace
pytest python/tests
```

### All 14 CLI Subcommands at a Glance

| Stage | Subcommand | What It Does |
| :--- | :--- | :--- |
| **01 · Audit & Read** | `secant profiles` | Lists built-in architecture profiles (`qwen35-9b`, `llama3-8b`, `security-auditor-b1b2b3`) and tensor templates. |
| | `secant inspect` | Zero-copy `mmap` header and tensor offset/shape inventory (`.safetensors` & `.gguf`). |
| | `secant read` | Deterministic `float64` SVD spectrum, top active columns, and `--vocab-tensor` token projection on disk. |
| | `secant sanitize` | Static Boundary-3 tensor payload scanner and `--clamp-nonfinite` remediator (`NaN`/`Inf`/subnormals/router IDs). |
| | `secant probe` | Deterministic generator for boundary-value test fixtures (`class1`..`class4`). |
| **02 · Analyze & Compose** | `secant analyze` | Ranks layers via Borda aggregation over empirical `SNR_proj` (`resid`/`attn`/`mlp`/`h`) and emits `L_loop*`/`L_ref*`/`L_root*`. |
| | `secant compose` | Two-pass `float64` Gram-Schmidt orthogonalization ($D D^\top = I_k$), `--analysis` slot binding, and Target-Shift Calibration ($\beta = \text{shift} / \|h_S\|_2^2$). |
| **03 · Patch** | `secant patch` | Zero-copy `mmap` sparse rank-1 or projector surgery on `BF16` / `F16` / `F32` `.safetensors` and `.gguf` files. |
| | `secant verify-patch` | Cryptographic pre/post SHA-256 and byte-range verifier proving all unedited tensors are bit-identical. |
| | `secant quant patch` | Direct in-place surgery on 34-byte `Q8_0` `.gguf` blocks without full-model dequantization or re-export. |
| | `secant quant witness` (`verify`) | Measures $\Pi_{128}$ head-grid alignment and quantization commutation error (`Q8_0` vs. `Q4_K`). |
| | `secant quant roundtrip` | Block-level encode/decode error microbenchmark for `Q8_0` and `Q4_K` codecs. |
| **04 · Lock & Heal** | `secant heal` | Projects an existing `.safetensors` LoRA adapter into $\ker(D)$ ($B_{\text{locked}} = P_\perp(P_\perp(B))$) and optionally merges it. |
| | `secant verify-subspace` | Audits any `.safetensors` adapter or merged checkpoint for subspace leakage against `--tol`. |

---

## 2. Workflow 1: Inspect a Model & Read Weight Concepts into Vocabulary Words Without a GPU (`secant inspect`, `secant read`)

Normally, seeing what a neural network layer has learned requires loading the model onto a GPU and running prompts through it. `secant inspect` and `secant read` memory-map `.safetensors` or `.gguf` files directly on disk, compute the top singular directions of any weight matrix in deterministic `float64`, and project those directions straight onto the model's vocabulary unembedding matrix (`--vocab-tensor`) so you can see which tokens a layer promotes or suppresses:

```bash
# 1. Inspect tensor names, dtypes, shapes, and exact byte offsets on disk
secant inspect --model Qwen3.5-4B.safetensors --json

# 2. Read the top singular directions of layer 15's MLP writer and project onto the vocabulary
secant read \
  --model Qwen3.5-4B.safetensors \
  --tensor "model.layers.15.mlp.down_proj.weight" \
  --vocab-tensor "model.embed_tokens.weight" \
  --top-k-singular 3 \
  --top-cols 8 \
  --detect-anomalies \
  --json
```

**What to look for in the JSON receipt (`secant.wsp.read.v1`):**
- `input_unchanged: true` (`pre_sha256 == post_sha256` proves the read was 100% non-destructive).
- `singular[]`: the top singular values $\sigma_1 \ge \sigma_2 \ge \dots$, convergence residuals (`~1e-16`), and the highest-energy input column indices (`top_columns`) that feed that singular direction.
- `vocab_projection`: when `--vocab-tensor` is supplied, each singular vector $u_i \in \mathbb{R}^{d_{\text{model}}}$ is dotted against the vocabulary matrix to surface the highest-affinity and most-suppressed token IDs directly from static weights.

---

## 3. Workflow 2: Capture Behavioral Directions & Compose a Calibrated Surgery Plan (`secant.capture`, `secant compose`)

To steer or suppress multiple behaviors at once (for example: closing endless `<think>` loops `B1`, stopping false-positive security keyword refusals `B2`, and focusing crash triage on the caller frame `B3`), extract the contrastive directions in Python and orthogonalize + calibrate them with `secant compose`:

```python
from secant.capture import capture_contrastive_directions, write_vectors

# activations: dict mapping (stream_name, layer_idx) -> (pos_acts [n_pos, d], neg_acts [n_neg, d])
results = capture_contrastive_directions(activations=activations, seed=20261104)

# Inspect held-out signal-to-noise (snr_proj) and split-half cosine to pick the sharpest layer
for (stream, layer), r in sorted(results.items(), key=lambda kv: -kv[1].snr_proj)[:5]:
    print(f"{stream} L{layer}: status={r.status} snr_proj={r.snr_proj:.1f} cos={r.split_half_cosine:.3f}")

# Save float64 direction vectors (d_*.npy) and reference activations (h_*.npy)
write_vectors("./vectors", results)
```

Now run `secant analyze` to rank empirical layer `SNR_proj` via Borda aggregation (`L_loop*`, `L_ref*`, `L_root*`), and `secant compose` to orthogonalize the directions in `float64` (`max |D D^T - I_k| < 1e-12`, measured `< 1e-15`) and solve analytically for the scale-calibrated step size $\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$ over the top-$K$ sparse columns:

```bash
# 1. Rank layers by empirical SNR_proj across resid / attn / mlp / h streams
secant analyze \
  --in layer_snr.json \
  --profile security-auditor-b1b2b3 \
  --out analysis.json \
  --json

# 2. Orthogonalize directions in float64, bind auto slots from analysis.json, and solve target-shift step
secant compose \
  --profile security-auditor-b1b2b3 \
  --vectors ./vectors \
  --analysis analysis.json \
  --target-shift -7.382 \
  --out plan-b1b2b3.json \
  --json
```

> **Why `--target-shift` matters:** Copying a fixed Frobenius norm ratio ($\gamma = 0.30$) from a `0.8B` model to a `4B` model closes **`0/30`** `<think>` loops because wider layers have different activation scales. Target-Shift Calibration solves for the exact step needed at the target layer's measured activation scale, taking `Qwen3.5-4B` from `0/30` (`0%`) to **`21/30` (`70%`)** at $1.0\times$ and **`30/30` (`100%`)** loop closure (`23/30` exact culprit frame) at $1.5\times$.

---

## 4. Workflow 3: In-Place `mmap` Surgery (`BF16`/`F32`) & Direct 34-Byte `Q8_0` GGUF Block Surgery (`secant patch`, `secant quant patch`, `secant verify-patch`)

### Option A: Zero-Copy `mmap` Surgery on Unquantized `.safetensors` / `.gguf` (`secant patch`)

```bash
secant patch \
  --base Qwen3.5-4B-BF16.gguf \
  --plan plan-b1b2b3.json \
  --out Qwen3.5-4B-BF16-patched.gguf \
  --json

secant verify-patch \
  --base Qwen3.5-4B-BF16.gguf \
  --patched Qwen3.5-4B-BF16-patched.gguf \
  --expect-changed blk.14.attn_output.weight,blk.15.ffn_down.weight,blk.16.ffn_down.weight,blk.17.ffn_down.weight \
  --json
```

### Option B: Direct 34-Byte `Q8_0` Block Surgery on Quantized `.gguf` (`secant quant patch`)

If you only have a quantized `Q8_0` `.gguf` file on disk, you do **not** need PyTorch or the original `BF16` checkpoint. `secant quant patch` opens the `.gguf` file via `mmap`, decodes **only the 34-byte `Q8_0` blocks** (2-byte `f16` scale $d_b$ + 32 `int8` values) that intersect the sparse column support $S$, applies the rank-1 update, and re-packs those blocks in place:

```bash
# 1. Optional: verify Q8_0 vs Q4_K head-grid alignment and quantization noise floor
secant quant witness --plan plan-b1b2b3.json --base Qwen3.5-0.8B-Q8_0.gguf --json

# 2. Patch only the targeted 34-byte Q8_0 blocks in place on disk
secant quant patch \
  --base Qwen3.5-0.8B-Q8_0.gguf \
  --plan plan-b1b2b3.json \
  --out Qwen3.5-0.8B-Q8_0-patched.gguf \
  --json

# 3. Verify that all 331 untouched tensors (and 240,640 untouched Q8_0 blocks) are bit-for-bit identical
secant verify-patch \
  --base Qwen3.5-0.8B-Q8_0.gguf \
  --patched Qwen3.5-0.8B-Q8_0-patched.gguf \
  --expect-changed blk.14.attn_output.weight,blk.15.ffn_down.weight,blk.16.ffn_down.weight,blk.17.ffn_down.weight \
  --json
```

> **Safety Gate (`Q8_0` vs. `Q4_K`):** Each 128-column `ssm_out` head divides evenly into four 32-weight `Q8_0` blocks ($128 = 4 \times 32$), achieving `98.5%` directional suppression with `0.18%–0.92%` Frobenius residual. By contrast, a 256-weight `Q4_K` superblock straddles two neighboring 128-column heads with a shared scale, so `secant quant patch` refuses `Q4_K` tensors (`block-aligned surgery needs Q8_0`).

---

## 5. Workflow 4: Subspace-Locked LoRA Healing (`secant.nn.ProjectThenHealLinear`, `secant heal`, `secant verify-subspace`)

If you fine-tune an edited model with standard LoRA (`dW = (alpha/r) B A`) under `AdamW`, unconstrained updates write straight back into $\mathrm{span}(D)$ (`~10^-2` leakage). Even projecting the gradient before `AdamW.step()` leaks on **`100%` of runs (`108/108`)** because `AdamW`'s coordinate-wise variance division $\sqrt{\hat{v}_t} + \epsilon$ rotates orthogonal gradients back into $\mathrm{span}(D)$.

`ProjectThenHealLinear` locks the LoRA output factor inside the forward pass via double orthogonal projection $B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$ in `float64`:

```python
import torch
from secant.nn import ProjectThenHealLinear

# Wrap any edited nn.Linear layer; D is [k, d_out] (tensor, numpy array, or .npy path)
locked_layer = ProjectThenHealLinear.from_linear(
    edited_linear,
    rank=16,
    directions="vectors/directions.npy",
    alpha=16.0,
    module_name="model.layers.15.mlp.down_proj",
)

# Train lora_A and lora_B_tilde normally with AdamW
optimizer = torch.optim.AdamW([locked_layer.lora_A, locked_layer.lora_B_tilde], lr=2e-4)

# Verify subspace leakage max |D B_locked| (~1e-18 in float64, ~1e-10 in float32)
print("max |D B_locked|:", locked_layer.leakage())

# Export locked LoRA adapter to .safetensors
locked_layer.export_lora("lora_locked.safetensors")
```

To lock or audit an existing `.safetensors` LoRA adapter offline from the CLI:

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

---

## 6. Workflow 5: 5-Second Reproducible Self-Test (`secant probe` $\to$ `inspect` $\to$ `read` $\to$ `sanitize` $\to$ `verify-patch`)

Want to run a byte-deterministic test right now without downloading a multi-gigabyte model? `secant probe` generates a tiny `736`-byte `.safetensors` fixture with planted `NaN`/`Inf` payload faults so you can test `inspect`, `read`, `sanitize`, and `verify-patch` in 5 seconds and compare against the exact SHA-256 digests below:

```bash
# 1. Generate a deterministic 736-byte fixture with 72 NaNs and 18 Infs
secant probe --class class3 --format safetensors \
  --out-dir fixtures --receipt probe-receipt.json --attempt-id quickstart-1
# -> sha256 = 22ffbb02b3e4d9c5560967bc7908f6991629fae9edf5d23c2b4419731555d503

# 2. Inspect its 3 tensors on disk
secant inspect --model fixtures/class3-nan-sort-arrays.safetensors --json

# 3. Scan for non-finite values and remediate into a clean copy
secant sanitize --model fixtures/class3-nan-sort-arrays.safetensors \
  --receipt clean-receipt.json --attempt-id quickstart-1 \
  --clamp-nonfinite --out clean.safetensors
# -> exit 0, status="remediated", clean.safetensors sha256 = 1e0b8f013a3f2e78e291fb1976f3b24372b3492716a88dbe64a7137df4f2a403

# 4. Compute the exact float64 singular value on the repaired tensor
secant read --model clean.safetensors --tensor 'blk.0.*' --top-k-singular 1 --top-cols 2 --json
# -> status="ok", sigma = 24494.89743144648, residual = 2.715678131663147e-16

# 5. Verify header parity and changed-tensor boundaries
secant verify-patch --base fixtures/class3-nan-sort-arrays.safetensors --patched clean.safetensors \
  --expect-changed blk.0.router_scores.weight,blk.1.router_scores.weight,blk.2.router_scores.weight --json
# -> header_bytes_identical=true, other_tensors_byte_identical=true, ok=true (exit 0)
```
