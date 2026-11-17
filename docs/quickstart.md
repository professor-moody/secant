# Quickstart Inspect, Edit, and Verify Weights

Secant is a single Rust CLI to scan, diff against a base model, and patch `.safetensors` and `.gguf` files directly on disk, with no Python and no GPU required. An optional Python layer keeps chosen directions fixed during LoRA fine-tuning.

Start with [the four jobs on demo files](#try-the-four-jobs-on-demo-files): it takes a few minutes and needs no model download. The numbered workflows after it are the full research path: capture directions, compose a calibrated edit, patch a real checkpoint, and lock later training. They are templates for your own checkpoint, activations, and tensor names. You can also [watch the terminal recordings](#watch-it-run) of a full edit on a 1.7B checkpoint first.

---

(try-the-four-jobs-on-demo-files)=
## Try the Four Jobs on Demo Files

From the repository root, build the CLI and generate a folder of tiny SafeTensors and GGUF files. The generator is a standard-library Python script; Secant itself does not use Python. It writes a clean model, a copy with a planted rank-one edit, and files with broken values.

```bash
cargo build --release -p secant-cli
export PATH="$PWD/target/release:$PATH"
python3 examples/test-drive/test_drive.py --work-dir secant-demo
cd secant-demo
```

**1. Diff against the base.** Which tensors differ from the clean model, and by how much?

```bash
secant read --model trojan_f32.safetensors --reference clean_f32/model.safetensors \
  --tensor 'model.layers.*.mlp.down_proj.weight' --top-k-singular 1 --top-cols 1

secant audit-trojan --model trojan_f32.safetensors --reference clean_f32/model.safetensors \
  --tensor 'model.layers.*.mlp.down_proj.weight' --vocab-tensor model.embed_tokens.weight
```

`read` lists a difference of `1.414214e0` for `model.layers.1.mlp.down_proj.weight` and `0.000000e0` for layers 0, 2, and 3. `audit-trojan` marks layers 0, 2, and 3 `identical` and shows layer 1 with all of its change in one direction (`1.0000`), plus the vocabulary tokens that direction promotes. Both exit `0`.

**2. Scan, then repair a copy.**

```bash
secant sanitize --model bad_q4k_nan.gguf --receipt scan.json --attempt-id demo
secant sanitize --model bad_q4k_nan.gguf --clamp-nonfinite --out fixed.gguf \
  --receipt fix.json --attempt-id demo-fix
```

The scan exits `1` with one `quant_scale_nan` finding: a `Q4_K` block whose scale header is `NaN`. The repair zeroes that block, rescans the result, publishes `fixed.gguf`, and exits `0` with status `remediated`.

**3. Patch in place.** Zero the embedding row for token 7 in a copy of the clean model, then check that nothing else moved:

```bash
cp clean_f32/model.safetensors edit-me.safetensors
cat > zero-row.json <<'EOF'
{
  "schema": "secant.wsp.plan.v1",
  "profile": "manual",
  "calibration": {"mode": "manual"},
  "edits": [
    {"id": "zero_token_7", "op": "set-value",
     "tensor": "model.embed_tokens.weight",
     "token_rows": [7], "semantics": ["zero"]}
  ]
}
EOF
secant patch --base edit-me.safetensors --plan zero-row.json --in-place
secant verify-patch --base clean_f32/model.safetensors --patched edit-me.safetensors \
  --expect-changed model.embed_tokens.weight
```

`patch` reports `64 / 8192` entries changed, a passing verification gate, and the new SHA-256. `verify-patch` reports an identical header, 8 of 9 tensors byte-identical, and exits `0`. Run `secant read --model edit-me.safetensors --reference clean_f32/model.safetensors --top-k-singular 1 --top-cols 1` and only `model.embed_tokens.weight` shows a nonzero difference.

**4. Lock directions during fine-tuning.** This one needs PyTorch: `pip install -e "./python[torch]"`, then run the example in the [PyTorch guide](PYTORCH-10-LINE-HEALING-GUIDE.md).

Reference hashes from our runs (macOS arm64, Secant `2e582e8`): `clean_f32/model.safetensors` `64e30a91f547a7ac28e7e84852e1e30e8675294d172a6144d2de812d3b6b8567`, `trojan_f32.safetensors` `fe07e95eba8c2f0d8b52082df919a92e8eb67b3456658e0d2bf3469977542ee7`, patched `edit-me.safetensors` `f84e2a02c52324d0b2d935545efcf150ea9f166a91ceb5d2391eb3fbab9fd108`.

---

## 1. Build and Install

Run from the repository root with Rust/Cargo and Python installed. Installation time depends on your machine and cached dependencies.

```bash
# 1. Build the Rust CLI (secant) and C-ABI library (libsecant) in release mode
cargo build --release --workspace

# 2. Show the available commands
./target/release/secant --help
export PATH="$PWD/target/release:$PATH"

# 3. Install the Python package (PyTorch ProjectThenHealLinear + contrastive capture)
pip install -e "./python[torch,test]"

# 4. Optional: run the Rust and Python test suites
cargo test --workspace
pytest python/tests
```

### Commands Used in This Guide

| Stage | Subcommand | What It Does |
| :--- | :--- | :--- |
| **01 · Audit & Read** | secant profiles | Lists research profiles, such as think-loop-fix and security-auditor-b1b2b3, and their edit slots. |
| | secant inspect | Lists tensor names, shapes, storage types, and byte offsets. |
| | secant read | Measures matrix size, leading directions, and optional vocabulary scores. With --reference, reports how each tensor differs from a base checkpoint. |
| | secant audit-trojan | Ranks matrices by unusual spectral structure; with --reference, ranks the changes against a base checkpoint and lists the tokens they promote. |
| | secant sanitize | Static weight-value scanner and optional repair for NaN, Inf, divisor scales, and routing IDs. Q8\_0 and Q4\_K zero or FP16 subnormal block scales pass by default; --flush-subnormal is opt-in. |
| | secant probe | Deterministic generator for boundary-value test fixtures (class1..class4). |
| **02 · Analyze & Compose** | secant analyze | Combines layer rankings from captured signal-to-noise measurements. |
| | secant compose | Makes supplied directions perpendicular, selects layers from --analysis, and sizes edits from reference activations. |
| **03 · Patch** | secant patch | Edits F32, F16, BF16, F64, I32, and I64 tensors, writing a copy by default or the same file with --in-place. |
| | secant verify-patch | Checks header identity and byte identity outside the supplied tensor allowlist. |
| | secant quant patch | Edits selected Q8\_0 blocks, writing a new checkpoint by default or the same file with --in-place. |
| | secant quant witness (verify) | Measures quantization error and optional reordering of 128-column panels. |
| | secant quant roundtrip | Encodes and decodes a matrix to measure compression error. |
| **04 · Lock & Heal** | secant heal | Removes protected output components from an existing adapter and optionally merges it. |
| | secant verify-subspace | Compares a base and healed checkpoint at selected tensors, checking leakage against --tol. |

---

## 2. Workflow 1 Inspect a Checkpoint and Read Its Weights

`inspect` lists the tensors in a checkpoint. `read` measures matrix structure and can score vocabulary tokens against its leading directions. Neither command runs inference. Treat the vocabulary scores as clues about static weights, not predictions of what the model will say. A memory map lets the program access file bytes without first loading the whole checkpoint into a framework.

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
- `input_unchanged: true`: the model's SHA-256 matched before and after the read.
- `singular[]`: the top singular values $\sigma_1 \ge \sigma_2 \ge \dots$, convergence residuals, and the highest-energy input column indices (`top_columns`) that feed that singular direction.
- `vocab_projection`: when `--vocab-tensor` is supplied, each singular vector $u_i \in \mathbb{R}^{d_{\text{model}}}$ is dotted against the vocabulary matrix to surface the highest-affinity and most-suppressed token IDs directly from static weights.

> **Takeaway:** You can see a checkpoint's layout and the structure of any matrix without loading the model, and the `read` receipt confirms the file was not modified.

---

## 3. Workflow 2 Capture Directions and Prepare an Edit

A contrastive direction is the difference between the average activations of two prompt groups. The snippet below assumes you have already collected those activations at the final prompt token. Check that a direction separates held-out examples before using it in an edit:

```python
from secant.capture import capture_contrastive_directions, write_vectors

# activations: dict mapping (stream_name, layer_idx) -> (pos_acts [n_pos, d], neg_acts [n_neg, d])
results = capture_contrastive_directions(activations=activations, seed=20261104)

# Inspect held-out signal-to-noise (snr_proj) and split-half cosine to pick the sharpest layer
valid = {key: r for key, r in results.items() if r.status == "ok"}
for (stream, layer), r in sorted(valid.items(), key=lambda kv: -kv[1].snr_proj)[:5]:
    print(f"{stream} L{layer}: status={r.status} snr_proj={r.snr_proj:.1f} cos={r.split_half_cosine:.3f}")

# Save float64 direction vectors only (d_*.npy); reference inputs are separate
write_vectors("./vectors", valid)
```

`write_vectors` saves direction files only. It does not save reference activations or the `layer_snr.json` input expected by `analyze`. Before composing, prepare those inputs and a profile that matches your model:

- The capture stream names become `d_<stream>_L<layer>.npy`; they must match the profile's direction tokens.
- A `rank1-columns` slot also needs the corresponding writer-input reference activation, such as `h_<token>_L<layer>.npy`. This has the writer's input width, which may differ from the output-direction width.
- Store metrics outside the vectors directory. The composer interprets vector-directory JSON files as vectors.
- `layer_snr.json` must follow an input layout supported by `secant analyze`; it is not generated by the snippet above. See [Native WSP Extensions](native-wsp-extensions.md).

With those inputs ready, `analyze` combines per-stream layer rankings using Borda voting: each stream contributes points based on rank. `compose` removes overlap between directions and calculates an edit size from the reference activation. Inspect the resulting plan's layers, tensor names, signs, and sizes before patching.

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

> **What calibration guarantees:** The requested shift is calculated for the supplied reference activation. Different inputs and lower-precision storage can change the realized shift. The profile supplies each slot's sign; the flag is not a universal “make every edit negative” switch. The reported `Qwen3.5-4B` completion results also depend on the specific directions, prompts, and doses described in the paper.

> **Takeaway:** `analyze` picks layers, `compose` turns directions into a reviewable plan. Nothing touches a checkpoint until you run `patch`.

---

## 4. Workflow 3 Write and Verify a Patched Copy

Use the model revision and tensor layout for which you prepared the plan. Set the shell variable `CHANGED_TENSORS` to the exact comma-separated checkpoint tensor names allowed to change in that plan. The verifier does not infer this list, and allowing a change does not prove that the intended numerical edit happened.

### Option A Floating-Point Weights

Use `secant patch` for supported floating-point tensors. Keep the base file for verification.

```bash
secant patch \
  --base Qwen3.5-4B-BF16.gguf \
  --plan plan-b1b2b3.json \
  --out Qwen3.5-4B-BF16-patched.gguf \
  --json

secant verify-patch \
  --base Qwen3.5-4B-BF16.gguf \
  --patched Qwen3.5-4B-BF16-patched.gguf \
  --expect-changed "$CHANGED_TENSORS" \
  --json
```

### Option B Compressed `Q8_0` Weights

A `Q8_0` block stores 32 weights in 34 bytes. The patcher decodes and re-encodes blocks that intersect the edit, while retaining untouched block bytes. You need a plan and vectors prepared for this same model and layout. The example writes a new file; `--in-place` is a separate choice that mutates the input.

```bash
# 1. Optional: measure quantization error along compatible output directions
secant quant witness --model Qwen3.5-4B-Q8_0.gguf \
  --tensor blk.15.ffn_down.weight --vectors directions.npy --format q8_0 --json

# 2. Patch selected Q8_0 blocks in a new copy
secant quant patch \
  --base Qwen3.5-4B-Q8_0.gguf \
  --plan plan-b1b2b3.json \
  --out Qwen3.5-4B-Q8_0-patched.gguf \
  --json

# 3. Verify the header and tensors outside the allowed change list
secant verify-patch \
  --base Qwen3.5-4B-Q8_0.gguf \
  --patched Qwen3.5-4B-Q8_0-patched.gguf \
  --expect-changed "$CHANGED_TENSORS" \
  --json
```

> **Format limit:** Four `Q8_0` blocks fit in a 128-column panel. A `Q4_K` superblock spans two such panels and shares encoding parameters across them. `secant quant patch` therefore refuses `Q4_K`. Block alignment does not guarantee that a behavioral edit will succeed; the guide's numerical results are measurements from specific experiments.

> **Takeaway:** Patch a copy, then let `verify-patch` prove that only the tensors you named changed. Steps 4 and 5 of the [recordings](#watch-it-run) show this on a real checkpoint.

---

## 5. Workflow 4 Constrain a LoRA Adapter

LoRA learns an adapter while the original weights stay fixed. Ordinary LoRA can add a component along a direction you removed earlier. Projecting only the gradient may not prevent this because AdamW rescales the gradient's coordinates.

`ProjectThenHealLinear` removes those components whenever the adapter runs. It calculates the projection in `float64` and casts the result back to the working dtype. The constraint is local to the wrapped layer; measure leakage again after a merge or precision change. In this snippet, `edited_linear` is an existing edited `torch.nn.Linear`, and `directions.npy` contains compatible output directions. Replace the original layer in your model with `locked_layer` before training.

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

# Measure the maximum protected component in the working-dtype adapter
print("max |D B_locked|:", locked_layer.leakage())

# Export locked LoRA adapter to .safetensors
locked_layer.export_lora("trained-locked.safetensors")
```

The CLI can project an existing adapter once and optionally merge it. Further ordinary LoRA training can undo that one-time constraint. The following uses separate `.safetensors` inputs, not the GGUF outputs above. Keep only compatible protected directions in `protected-vectors/`; use the adapter's actual `alpha/r` scale (shown here as `1.0`). Choose output names that do not already exist:

```bash
secant heal \
  --lora lora_adapter.safetensors \
  --vectors ./protected-vectors \
  --out lora_locked.safetensors \
  --base patched.safetensors \
  --merge-out healed.safetensors \
  --scale 1.0 \
  --json

secant verify-subspace \
  --base patched.safetensors \
  --healed healed.safetensors \
  --vectors ./protected-vectors \
  --tensors model.layers.15.mlp.down_proj.weight \
  --tol 1e-5 \
  --json
```

> **Takeaway:** Use `ProjectThenHealLinear` while training and `secant heal` for an adapter that already exists, then measure leakage again after any merge or precision change.

---

(local-self-test)=
## 6. Workflow 5 Try a Small Local Self-Test

This small example needs no model download or GPU. It creates a test file with non-finite values, writes a repaired copy, and checks the changed tensor boundaries. Run it in a fresh scratch directory because these commands refuse existing outputs. The hashes below are retained expected values, not timing guarantees. This checks the file tools, not inference-runtime safety.

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

# 4. Compute the leading singular value on the repaired tensor
secant read --model clean.safetensors --tensor 'blk.0.*' --top-k-singular 1 --top-cols 2 --json
# -> status="ok", sigma = 24494.89743144648, residual = 2.715678131663147e-16

# 5. Verify header parity and changed-tensor boundaries
secant verify-patch --base fixtures/class3-nan-sort-arrays.safetensors --patched clean.safetensors \
  --expect-changed blk.0.router_scores.weight,blk.1.router_scores.weight,blk.2.router_scores.weight --json
# -> header_bytes_identical=true, other_tensors_byte_identical=true, ok=true (exit 0)
```

> **Takeaway:** If your hashes match the ones above, your build reproduces the reference outputs byte for byte.

(watch-it-run)=
## 7. Watch It Run Real Terminal Recordings

The six terminal captures below show the compiled `secant` release binary running each step of an edit on [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) (`3.4 GB` `BF16` `.safetensors`, SHA-256 `f55217be...a0e8`), producing the patched file (`4d036572...4f0b`) from the paper's live-healing experiment (Regime A).

**Step 1: `secant inspect` reads the file layout without loading the model.** It reports 218 `BF16` tensors behind a 25,184-byte header.

```{figure} assets/secant-inspect.gif
:name: fig-vhs-inspect
:alt: Terminal recording of secant inspect on SmolLM2-1.7B-Instruct, reporting 218 BF16 tensors and a 25,184-byte payload offset
:width: 100%

`secant inspect --model smollm2-1.7b.safetensors`
```

**Step 2: `secant analyze` ranks layers from the captured signal-to-noise ratios.** For each behavior it combines the rankings from four activation streams (Borda count). The `b2` family peaks at `L11` and `b3` at `L12`. The tool also reports where the hard-coded mid-layer default (`L15`/`L19`) ranks instead.

```{figure} assets/secant-analyze.gif
:name: fig-vhs-analyze
:alt: Terminal recording of secant analyze ranking 24 layers across resid, attn, mlp and h subspaces
:width: 100%

`secant analyze --in layer_snr.json --out analysis.json`
```

This profile pins its layers explicitly (`B2` at `L14`, `B3` at `L15`), so the next step does not read `analysis.json`. To bind layers from the ranking instead, use a profile whose slots set `layer: -1` and pass `--analysis analysis.json` to `compose`.

**Step 3: `secant compose` turns directions into an edit plan.** Gram-Schmidt leaves a largest cosine of `1.125e-17` between the two directions, far below the `1e-12` gate. `B3` edits 64 columns, sized so the shift at its reference activation is `+3.895616`.

```{figure} assets/secant-compose.gif
:name: fig-vhs-compose
:alt: Terminal recording of secant compose producing a two-edit plan with a passing orthogonality gate
:width: 100%

`secant compose --profile profile-smollm2-b2b3.json --vectors vectors --out plan.json`
```

**Step 4: `secant patch` writes the edits into a copy of the file.** `B2` changes 3,425,789 of 4,194,304 entries in `model.layers.14.self_attn.o_proj.weight`. `B3` changes only 104,755 of 16,777,216 entries (0.6%) in `model.layers.15.mlp.down_proj.weight`. `shasum` then confirms the output hash.

```{figure} assets/secant-patch.gif
:name: fig-vhs-patch
:alt: Terminal recording of secant patch editing two tensors and printing the output SHA-256 4d036572
:width: 100%

`secant patch --base smollm2-1.7b.safetensors --plan plan.json --out patched.safetensors`
```

**Step 5: `secant verify-patch` proves nothing else moved.** The header is byte-identical, only the two named tensors differ, and the other 216 tensors are byte-identical to the base.

```{figure} assets/secant-verify-patch.gif
:name: fig-vhs-verify-patch
:alt: Terminal recording of secant verify-patch reporting an identical header and 216 byte-identical tensors
:width: 100%

`secant verify-patch --base smollm2-1.7b.safetensors --patched patched.safetensors --expect-changed ...`
```

**Step 6: `secant probe` generates the self-test fixtures.** It writes the same NaN/Inf fixture as both `.safetensors` and `.gguf`, then inspects the `.gguf` copy.

```{figure} assets/secant-probe.gif
:name: fig-vhs-probe
:alt: Terminal recording of secant probe writing class3 safetensors and GGUF fixtures with 90 findings each, then inspecting the GGUF
:width: 100%

`secant probe --class class3 --out-dir fixtures --receipt probe-receipt.json --attempt-id vhs-demo`
```

The [CLI, Python & C-ABI Reference](cli-and-python-reference.md) documents all sixteen root commands, including `fetch-tensor`, `rotation-test`, `audit-trojan` and `scan-sinks`.
