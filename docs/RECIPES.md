# Practical Secant Recipes

Secant is a single Rust CLI to scan, diff against a base model, and patch `.safetensors` and `.gguf` files directly on disk, with no Python and no GPU. Each recipe below is one job, with commands you can copy and a note on what the result tells you. Commands run from the repository root unless stated otherwise.

## Before You Start

Install the CLI once:

```sh
cargo install --path crates/secant-cli
```

To try everything on tiny generated files first, run `python3 examples/test-drive/test_drive.py --work-dir demo` and follow the [quickstart](quickstart.md#try-the-four-jobs-on-demo-files). It needs only Python's standard library, plus Rust if the CLI is not built yet.

Keep an untouched copy of every original. Receipt and output paths must be new files; Secant refuses to overwrite them.

## Recipe 1 Diff Against a Base Model

Use this when you download a community fine-tune, merge, or re-quantized file and want to know what it changed relative to the official base.

```sh
model=finetune.safetensors
base=base.safetensors
secant read --model "$model" --reference "$base" \
  --top-k-singular 1 --top-cols 1 --json > diff.json
jq -r '.tensors[] | select(.status != "ok" or .frobenius != 0) | [.name, .status, .frobenius] | @tsv' diff.json
```

`read --reference` subtracts each base tensor from the same-named tensor in `--model` and reports the size of the difference. The `jq` line prints only tensors that changed (a nonzero size) or could not be compared (status `skipped: ...`). An empty list means every compared tensor matches the base value for value.

Use `--tensor` to narrow the comparison with a glob, such as `--tensor 'blk.*.ffn_*'`. Embeddings, output heads, norms, and MoE router tensors are compared like any other tensor, so a changed vocabulary row or router shows up here too.

To rank the changed matrices and see which tokens the largest change pushes toward:

```sh
secant audit-trojan --model "$model" --reference "$base" \
  --vocab-tensor lm_head.weight --rank-k 4
```

For each 2-D tensor this reports how big the change is relative to the base weights (`||DW||/||W_REF||`), how much of it lies in a single direction, and the vocabulary tokens that direction promotes. A planted rank-one edit shows a fraction near `1.0000`. Unchanged tensors print `identical`. Use your model's output head name; GGUF files usually call it `output.weight`.

What to know:

- Tensors are matched by name and shape. Hugging Face SafeTensors and GGUF name tensors differently, so compare like with like: a fine-tune's SafeTensors against the base SafeTensors, or a community GGUF against an official GGUF of the base. Float, `Q8_0`, and `Q4_K` tensors can be compared with each other. If no tensor matches, the command exits `2`.
- A quantized file compared with a full-precision base differs everywhere by rounding. Look for tensors whose change is far larger than their neighbors'.
- Both commands are reports. They exit `0` when they finish, whatever they find.
- With no reference at all, `audit-trojan --model "$model"` compares each layer with its siblings. That only flags edits large enough to stand out from the layer's own structure; the [payload checks](payload-security.md) page measures where that threshold sits.

## Recipe 2 Scan a Downloaded Model

Use this after downloading a pinned Hugging Face revision and before vLLM or llama.cpp loads it. Keep the matching `config.json`, `generation_config.json`, `tokenizer_config.json`, and `tokenizer.json` beside the weights so their token IDs get checked too. Scan every shard; one clean shard says nothing about the others.

```sh
model=/path/to/model.safetensors  # a .gguf works too
scan_dir=$(mktemp -d)
secant inspect --model "$model" --json > "$scan_dir/layout.json"
secant sanitize --model "$model" --dry-run \
  --receipt "$scan_dir/scan.json" --attempt-id downloaded-model --json
scan_exit=$?
printf 'sanitize exit = %s; receipt directory = %s\n' "$scan_exit" "$scan_dir"
```

Exit `0` means the checks passed, `1` means findings, `2` means unsupported or invalid input. In the receipt, read `status`, `before.counts`, and `before.examples`.

To upload findings to GitHub Code Scanning, pass `--sarif "$scan_dir/scan.sarif"` (written atomically before exiting `0` or `1`, or exiting `2` if unwritable). In GitHub Actions, set the `sarif-out` input on this repository's action and pass the file to `github/codeql-action/upload-sarif@v3` with `if: always()`.

For MoE routing tables, pass the real expert count with `--num-experts N`. If an index bound cannot be inferred from sibling tensors, pass a verified `--index-bound N`. Use the real values; a guessed bound can hide a finding. Common divisor-scale names (`input_scale`, `k_scale`, `v_scale`, `q_scale`, `y_scale`, `prob_scale`, `*scale_inv`) are checked by default. Add others with `--divisor-scale-tensor GLOB` only when the runtime divides by them.

`inspect` lists layout and types without checking values. `sanitize` checks values against fixed rules, which cannot see a behavioral backdoor in normal-looking weights. If you have the base model, also run [Recipe 1](#recipe-1-diff-against-a-base-model).

## Recipe 3 Repair a Copy and Verify Its Hashes

Use this for a non-finite value or scale after reviewing its scan. Keep the output beside the original so both see the same companion files.

```sh
model=/path/to/model.gguf
repaired=/path/to/model-sanitized.gguf
repair_dir=$(mktemp -d)
secant_bin=$(command -v secant)
shasum -a 256 "$model" "$secant_bin" > "$repair_dir/input-and-tool.sha256"
secant sanitize --model "$model" --clamp-nonfinite --out "$repaired" \
  --receipt "$repair_dir/repair.json" --attempt-id repair-copy --json
repair_exit=$?
shasum -a 256 -c "$repair_dir/input-and-tool.sha256"
printf 'repair exit = %s\n' "$repair_exit"
```

Continue only if repair exited `0` and the hash check passed. The repaired copy is written only after it passes a rescan.

What `--clamp-nonfinite` does: `NaN` becomes `0`, `Inf` becomes `--max-abs` (default `1e4`) with the same sign, a non-finite divisor scale becomes `--divisor-floor`, and a quantized GGUF block with a non-finite scale header is zeroed. Zeroing a block removes its 32 or 256 weights; it does not restore them. Finite zero or negative divisor scales, out-of-range routing IDs, and collapsed norm vectors have no automatic repair. If any remain, nothing is published.

Rescan the output on its own:

```sh
shasum -a 256 "$repaired" "$secant_bin" > "$repair_dir/output-and-tool.sha256"
secant sanitize --model "$repaired" --dry-run \
  --receipt "$repair_dir/rescan.json" --attempt-id repair-rescan --json
rescan_exit=$?
shasum -a 256 -c "$repair_dir/output-and-tool.sha256"
printf 'rescan exit = %s; evidence = %s\n' "$rescan_exit" "$repair_dir"
```

Keep both receipts and both hash files. The repair receipt records the original hash, the output hash, and the findings before and after. A clean file can still be a worse model, so evaluate the repaired copy against the original before deploying it.

## Recipe 4 Patch Weights on Disk

Use this to change specific weights without loading the model into a framework, re-exporting, or re-quantizing the whole file. You describe the edit in a JSON plan and Secant writes only the bytes of the tensors it names.

### Write a plan by hand

A plan is a list of edits. Two ops are practical to write by hand.

`set-value` writes exact values. This plan zeroes embedding rows 7 and 8, and sets two single entries of a norm vector to `0`:

```json
{
  "schema": "secant.wsp.plan.v1",
  "profile": "manual",
  "calibration": {"mode": "manual"},
  "edits": [
    {"id": "zero_tokens", "op": "set-value",
     "tensor": "model.embed_tokens.weight",
     "token_rows": [7, 8], "semantics": ["zero"]},
    {"id": "norm_entries", "op": "set-value",
     "tensor": "model.norm.weight",
     "set_indices": [0, 5], "semantics": ["zero"]}
  ]
}
```

- `token_rows` selects whole rows of a 2-D tensor. `set_indices` selects flat element positions. With neither, the whole tensor gets one value.
- The value comes from exactly one of `semantics` (`zero`, `nan`, `+inf`, `-inf`, `min`, `max`), `set_bits` (raw bit patterns for the tensor's dtype), or `set_ints` (for `I32`/`I64` tensors). A list shorter than the selection repeats.

`rank1-columns` adds a scaled direction to chosen input columns: for each `cols[i]`, column `cols[i]` of the weight matrix gets `coeffs[i] * direction` added to it. `direction` must have the tensor's output width; `secant read --json` reports it as `logical_out` for each tensor.

```json
{"id": "nudge", "op": "rank1-columns",
 "tensor": "model.layers.15.mlp.down_proj.weight",
 "direction": [0.01, -0.02, "... one value per output row ..."],
 "cols": [0, 1, 2, 3], "coeffs": [0.5, 0.5, 0.5, 0.5]}
```

Plans written by `secant compose` use the same format, with directions and sizes computed from captured activations. The [quickstart](quickstart.md) walks through that path.

### Apply and check it

```sh
secant patch --base model.safetensors --plan plan.json --out model-edited.safetensors --json > patch.json
secant verify-patch --base model.safetensors --patched model-edited.safetensors \
  --expect-changed model.embed_tokens.weight,model.norm.weight
```

`patch` applies the edits, checks that the header and every tensor outside the plan are byte-identical to the base, and only then publishes the result. The receipt holds `base_sha256`, `out_sha256`, and the number of entries each edit changed. `verify-patch` repeats the identity check on its own; it exits `1` if any tensor outside `--expect-changed` differs.

To edit the file at its own path, use `--in-place` instead of `--out`. Secant writes the verified result to a temporary file in the same folder and renames it over the original, and refuses to publish if the original changed on disk during the run.

### Q8_0 GGUF files

```sh
secant quant patch --base model-Q8_0.gguf --plan plan.json --out model-Q8_0-edited.gguf --json
```

`quant patch` applies `rank1-columns` edits to `Q8_0` tensors. It decodes only the 34-byte blocks the edit touches, applies the change, and re-encodes them; every other block stays byte-identical. The receipt reports blocks touched and the largest re-quantization error. `--in-place` works the same way as above.

What to know:

- `secant patch` handles `F32`, `F16`, `BF16`, `F64`, `I32`, and `I64` tensors in SafeTensors and GGUF. `secant quant patch` handles `Q8_0`. `Q4_K` and other K-quants are refused, because a K-quant superblock shares scales across 256 weights; see [quantized surgery](quantized-surgery.md).
- The patcher holds the file in memory, about twice its size. For sharded checkpoints, patch the shard that holds the tensor.
- A passing `verify-patch` proves which bytes changed. Whether the model still behaves well needs an evaluation.

## Recipe 5 Fine-Tune and Export in Ten Lines

Install `python3 -m pip install -e './python[torch]'`. Supply an existing `torch.nn.Linear` as `linear_layer`, protected output directions as `protected_dirs`, and training tensors as `inputs` and `targets`. Directions have shape `[k, out_features]`. Choosing good directions takes model-specific evidence; random directions are useful only as a numerical smoke test.

With `torch` imported, these ten lines train one step and export a standard layer:

```python
from secant.nn import ProjectThenHealLinear
locked = ProjectThenHealLinear.from_linear(linear_layer, rank=4, directions=protected_dirs)
optimizer = torch.optim.AdamW([p for p in locked.parameters() if p.requires_grad], lr=1e-3)
optimizer.zero_grad()
loss = (locked(inputs) - targets).square().mean()
loss.backward()
optimizer.step()
assert max(locked.leakage(), locked.merged_leakage()) < 1e-6
exported = torch.nn.Linear(locked.in_features, locked.out_features, bias=locked.bias is not None).to(locked.weight)
exported.load_state_dict({"weight": locked.merge_into(), **({"bias": locked.bias.detach()} if locked.bias is not None else {})})
```

In a real training loop, swap the selected layer for `locked` before training and for `exported` afterwards. Freeze other layers if they also need to stay fixed. The exported layer is an ordinary `nn.Linear` with no adapter, so inference costs the same as before.

`leakage()` measures the adapter; `merged_leakage()` measures the merged weight difference. The `1e-6` threshold is a float32 tolerance chosen for this example. Run in float32 or float64 outside autocast when you rely on it, because lower precision weakens the lock. The [full guide](PYTORCH-10-LINE-HEALING-GUIDE.md) has a runnable example and explains how to choose directions.

## Recipe 6 Verify a Quantized Export

After exporting a base and a healed checkpoint to the same layout and quantization format, check that the stored difference stays out of the protected directions:

```sh
verify_dir=$(mktemp -d)
base=base-q8_0.gguf
healed=healed-q8_0.gguf
vectors=protected.npy
secant_bin=$(command -v secant)
shasum -a 256 "$base" "$healed" "$vectors" "$secant_bin" > "$verify_dir/inputs.sha256"
secant verify-subspace --base "$base" --healed "$healed" \
  --vectors "$vectors" --tensors blk.0.ffn_down.weight \
  --tol 1e-5 --json > "$verify_dir/verification.json"
verify_exit=$?
shasum -a 256 -c "$verify_dir/inputs.sha256"
printf 'verification exit = %s; evidence = %s\n' "$verify_exit" "$verify_dir"
```

The same command accepts `Q4_K` files. Use the tensor names `inspect` reports; `--tensors` takes a comma-separated list, not a glob. Directions must match each tensor's output width, and `--vectors` takes one `.npy` file or a directory.

The check computes the largest protected-direction component of `W_healed - W_base` across the selected tensors. The default `--tol` is `1e-5`. Exit `0` means every selected tensor passed, `1` means leakage exceeded the tolerance, and `2` means the result could not be determined. Also read `unchecked_changed`: it lists changed tensors you did not select.

Quantization can add leakage back even when the float adapter was locked. Pick the tolerance before looking at results, keep a base-versus-identical-copy control, and compare with the float export to separate training drift from rounding.

## Recipe 7 Test a Scanner with Synthetic Fixtures

Use known-bad files to check that a scanner, Secant or your own, catches what it claims to. These are small numerical fixtures. They are scanner inputs, and they do not exercise any runtime.

```sh
# Nine checks, including clean controls, repair, and a planted rank-one edit
python3 examples/test-drive/test_drive.py

# A larger generated corpus
cargo build --release -p secant-core --example benign_fpr_fixtures
fixture_dir=$(mktemp -d)
target/release/examples/benign_fpr_fixtures "$fixture_dir/suite" scanner-check
```

Add `--work-dir` with a new directory to keep the test-drive files and receipts. The generated `suite/manifest.json` lists each file, its SHA-256, its role, and the expected finding. Keep clean and altered files apart when counting false positives and detections, and count each file once.

| Condition | Where it is covered | What a scanner should report |
| --- | --- | --- |
| NaN or Inf weights | Test-drive router case | A non-finite finding and exit `1` |
| Zero or below-floor divisor scale | `quant_sanitize_cli` Rust tests and generated `positive-divisor_subnormal` | A divisor-scale finding; `--clamp-nonfinite` does not repair a finite zero |
| Q8_0 or Q4_K NaN/Inf scale header | Generated `positive-Q8_0-quant_nan`, `positive-Q8_0-quant_inf`, and the Q4_K equivalents | `quant_scale_nan` or `quant_scale_inf`; zero or subnormal multiplier scales are legitimate |
| Out-of-range MoE expert IDs | Test-drive routing case with `--num-experts 8` | Routing findings with the expected bounds |
| Rank-one edit | Test-drive case 9 | The edited layer ranks first, and the `--reference` difference sits in one direction |

For a repeatable benchmark with retained receipts:

```sh
cargo build --release --locked -p secant-cli -p secant-core --bins --examples
python3 scripts/bench_benign_fpr.py --run-dir data/benign_fpr_v030/my-scanner-check
```

Use a new run directory. The runner hashes inputs and binaries before and after each scan and records controls, incomplete checks, and repeats. The [retained benchmark](benchmarks/benign-fpr-v0.3.0.md) explains the denominator and the cross-platform comparison; its JSON SHA-256 is `63ac1125e6a1f87f65acd5c3163e7fdf3ad2c479a1dee8968c8dd0bf80a64fac`.

When testing your own scanner, keep its version and raw output, and report unsupported cases separately from passes.
