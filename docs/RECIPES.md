# Practical Secant Recipes

Start with the job you need to do. Each recipe explains what a passing result means and what still needs checking. All commands run from the repository root unless stated otherwise.

## Before You Start

Install the CLI once with `cargo install --path crates/secant-cli`. For the smallest local demonstration, run `python3 examples/test-drive/test_drive.py`; it needs only Python's standard library and a Rust toolchain if the CLI has not been built. No model download or GPU is required.

Keep a trusted original, use new receipt paths, and retain JSON results alongside input and tool hashes. A static scan checks numerical and structural rules. Behavioral evaluations answer whether a model still performs its intended task.

## Recipe 1 Scan a Downloaded Model

Use this after downloading a pinned Hugging Face revision and before handing its files to vLLM or llama.cpp. Keep the checkpoint's matching `config.json`, `generation_config.json`, `tokenizer_config.json`, and `tokenizer.json` beside it when available. Scan every shard separately; one passing shard does not cover the others.

Set `model` to your local file, then run:

```sh
model=/path/to/model.safetensors  # A supported .gguf also works
scan_dir=$(mktemp -d)
secant inspect --model "$model" --json > "$scan_dir/layout.json"
secant sanitize --model "$model" --dry-run \
  --receipt "$scan_dir/scan.json" --attempt-id downloaded-model --json
scan_exit=$?
printf 'sanitize exit = %s; receipt directory = %s\n' "$scan_exit" "$scan_dir"
```

Read `status`, `before.counts`, and `before.examples` in the receipt. Exit `0` means the requested policy checks completed cleanly; `1` means findings remain; `2` means unknown, unsupported, or invalid input. Only continue an automated loading pipeline after a completed clean result and your other validation gates.

For routing tables, supply the actual expert bound using `--num-experts N`. If index bounds cannot be inferred from sibling tensors, supply a verified `--index-bound N`. Do not guess these values to get a passing scan. Defaults select known divisor-scale names; other names need `--divisor-scale-tensor GLOB` only when their consumer actually divides by those scales.

`inspect` lists layout and types; it does not check all values. `sanitize` does not certify absence of a finite-weight backdoor. When you have a trusted same-layout reference, use `secant read --reference` or `audit-trojan --reference` to examine changes and evaluate the model on held-out behavior tests.

## Recipe 2 Test a Scanner with Synthetic Fixtures

Use numerical fixtures to check whether a scanner sees the conditions it claims to detect. These small files are scanner inputs, not executable model graphs or demonstrations of a runtime vulnerability.

```sh
# Nine checks, including clean controls, repair, and a rank-one matrix change
python3 examples/test-drive/test_drive.py

# A retained native generator supplies a larger numerical corpus
cargo build --release -p secant-core --example benign_fpr_fixtures
fixture_dir=$(mktemp -d)
target/release/examples/benign_fpr_fixtures "$fixture_dir/suite" scanner-check
```

For the nine-case demo, add `--work-dir /tmp/secant-demo-evidence` with a new directory to retain its files and native receipts.

The generated `suite/manifest.json` lists each file, its SHA-256, role, and expected finding kind. Keep clean and altered files separate when calculating false positives and detection counts. Do not count repeated scans as additional models.

| Condition | Existing coverage | What to check |
| --- | --- | --- |
| NaN or Inf weights | Test-drive router case | A non-finite-value finding and exit `1` |
| Zero or below-floor divisor scale | `quant_sanitize_cli` Rust integration tests and generated `positive-divisor_subnormal` | A divisor-scale finding; finite zero is detected but not repaired by `--clamp-nonfinite` |
| Q8_0 or Q4_K NaN/Inf scale header | Generated `positive-Q8_0-quant_nan`, `positive-Q8_0-quant_inf`, and corresponding Q4_K cases | `quant_scale_nan` or `quant_scale_inf`; clean multiplier scales can be zero or subnormal |
| Out-of-range MoE expert IDs | Test-drive routing case with `--num-experts 8` | Routing findings with the expected bounds |
| Rank-one spectral change | Test-drive numerical matrix case | The altered layer is ranked as expected and the reference difference concentrates in one singular direction |

For a repeatable scanner benchmark with retained receipts, use the existing runner:

```sh
cargo build --release --locked -p secant-cli -p secant-core --bins --examples
python3 scripts/bench_benign_fpr.py --run-dir data/benign_fpr_v030/my-scanner-check
```

Choose a new run directory. The runner records input and binary hashes before and after scans, controls, incomplete checks, and repeated measurements. The [retained benchmark](benchmarks/benign-fpr-v0.3.0.md) explains the denominator and cross-platform comparison. Its JSON SHA-256 is `63ac1125e6a1f87f65acd5c3163e7fdf3ad2c479a1dee8968c8dd0bf80a64fac`.

For your own scanner, use these same numerical inputs and benign controls, retain its version and raw results, and report unsupported cases separately from passes. A completed `audit-trojan` command can exit `0` while reporting outliers; inspect the JSON instead of treating that exit as a clean verdict. A rank-one difference identifies structure, not malicious intent.

## Recipe 3 Repair a Copy and Verify Its Hashes

Use this for a non-finite scale or value after reviewing its scan. The example below keeps the output beside the original so both see the same companion files. The output filename and receipt paths must be new.

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

Continue only if repair exited `0` and the original/tool hash check passed. For quantized GGUF, `--clamp-nonfinite` zeroes the entire block containing a non-finite scale. It does not reconstruct the intended weights. Finite zero or negative divisor scales, out-of-range routing IDs, and collapsed norm vectors have no automatic repair under this flag; remaining findings prevent publication.

Verify the output independently:

```sh
shasum -a 256 "$repaired" "$secant_bin" > "$repair_dir/output-and-tool.sha256"
secant sanitize --model "$repaired" --dry-run \
  --receipt "$repair_dir/rescan.json" --attempt-id repair-rescan --json
rescan_exit=$?
shasum -a 256 -c "$repair_dir/output-and-tool.sha256"
printf 'rescan exit = %s; evidence = %s\n' "$rescan_exit" "$repair_dir"
```

Retain both receipts and both hash files. The repair receipt records the original hash, output hash, input stability, and before/after findings. Evaluate output quality against the untouched original and a trusted clean reference before deployment; numerical cleanliness does not restore model quality by itself.

## Recipe 4 Fine-Tune and Export in Ten Lines

Install `python3 -m pip install -e './python[torch]'`. Supply an existing `torch.nn.Linear` as `linear_layer`, protected output directions as `protected_dirs`, and training tensors as `inputs` and `targets`. Directions have shape `[k, out_features]`; choosing them requires model-specific evidence. Random directions are useful only for a numerical smoke test.

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

In a model training loop, replace the selected layer with `locked` before training, then with `exported` for inference. Freeze other layers if the intended constraint requires them to stay fixed. The exported layer has ordinary weights and no adapter operations, so it adds zero inference overhead relative to that standard layer.

`leakage()` measures the projected adapter factor; `merged_leakage()` measures the stored merged-weight difference. The `1e-6` check is a chosen float32 tolerance, not a universal bound for arbitrary weight magnitudes. Use float32 or float64 activations outside autocast when relying on this measurement; lower-precision computation or storage can weaken the lock. The [full guide](PYTORCH-10-LINE-HEALING-GUIDE.md) gives a runnable example and explains how to choose directions.

## Recipe 5 Verify a Quantized Export

After exporting a base and healed checkpoint to the same layout and quantization format, check the stored difference:

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

The same command accepts Q4_K files. Use the actual tensor names reported by `inspect`; `--tensors` accepts a comma-separated list, not a wildcard. Directions must match each selected tensor's output width. `--vectors` accepts one `.npy` file or a directory.

The CLI computes the maximum absolute protected-direction component of `W_healed - W_base`, over all selected input columns. The default `--tol` is `1e-5`. Exit `0` means every selected tensor passed, `1` means leakage exceeded tolerance, and `2` means the result could not be determined. Review `unchecked_changed` too: a passing selected-tensor check does not cover other changed tensors.

Quantization can reintroduce leakage even when the float adapter was locked. Set an acceptance tolerance before examining results, retain a base-versus-identical-copy control, and compare with the float export to distinguish training drift from quantization effects. Secant verifies Q4_K exports but its Rust patch command does not perform Q4_K weight surgery.
