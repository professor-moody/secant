<div align="center">

<img src="docs/assets/secant-logo.svg" alt="Secant Logo" width="84" height="84" />

# Secant

**Scan, diff, and patch `.safetensors` and `.gguf` model files on disk.**

[![Documentation](https://img.shields.io/badge/Docs-professor--moody.github.io%2Fsecant-0A66C2?style=flat-square)](https://professor-moody.github.io/secant/)
[![Technical Paper](https://img.shields.io/badge/Paper-MyST_%2F_Quarto-8A2BE2?style=flat-square)](https://professor-moody.github.io/secant/paper)
[![Blog Series](https://img.shields.io/badge/Blog_Series-m00dy.sh-12704f?style=flat-square)](https://m00dy.sh/notes/sparks-to-secant-weight-space-programming-series)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange?style=flat-square)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)](https://www.python.org/)

[Recipes](docs/RECIPES.md) · [Quickstart](docs/quickstart.md) · [Test drive](examples/test-drive/README.md) · [PyTorch guide](docs/PYTORCH-10-LINE-HEALING-GUIDE.md) · [Technical paper](docs/PAPER.md)

</div>

Secant is a single Rust CLI to scan, diff against a base model, and patch `.safetensors` and `.gguf` files directly on disk. It needs no Python and no GPU. It reads the file format itself, so nothing gets loaded into PyTorch, llama.cpp, or vLLM.

```sh
cargo install --path crates/secant-cli
```

## Four things you can do with it

### 1. Diff a downloaded checkpoint against its base model

Got a community fine-tune, merge, or re-quantized file? Compare it with the official base and see exactly which tensors changed.

```sh
secant read --model finetune.safetensors --reference base.safetensors \
  --top-k-singular 1 --top-cols 1 --json \
  | jq -r '.tensors[] | select(.status != "ok" or .frobenius != 0) | [.name, .status, .frobenius] | @tsv'
```

Each tensor is compared with the same-named tensor in the base. The number printed is the size of the change, and tensors that match the base are filtered out. Tensors the base does not have are listed as skipped.

To rank the changes and see which vocabulary tokens the biggest change pushes toward:

```sh
secant audit-trojan --model finetune.safetensors --reference base.safetensors \
  --vocab-tensor lm_head.weight
```

```text
  TENSOR (DELTA W)                     ||DW||/||W_REF||  SIGMA_1     SIGMA_1^2/||DW||^2  PROMOTED TOKENS OF DW u_1
  model.layers.1.mlp.down_proj.weight  7.8112e-1         1.414214e0  1.0000              22(5.373e-2) 45(5.217e-2) ...
  model.layers.0.mlp.down_proj.weight  0.0000e0          identical   n/a
  model.layers.2.mlp.down_proj.weight  0.0000e0          identical   n/a
```

That output is from the [test drive](examples/test-drive/README.md): a rank-one edit planted in layer 1 shows up as the only changed layer, and all of its change sits in one direction (`1.0000`). Use the tensor names your file actually has; `secant inspect` lists them.

### 2. Scan for broken or poisoned values before loading

```sh
secant sanitize --model model.gguf --receipt scan.json --attempt-id pre-load
```

On CPU, with no inference, `sanitize` checks for:

- `NaN` and `Inf` weights
- bad scale headers in ten GGUF block formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, and Q2_K to Q6_K)
- zero or tiny divisor scales such as `k_scale` and `*scale_inv`
- MoE expert indices outside the expert count (pass `--num-experts N`)
- token IDs in `config.json`, `generation_config.json`, and `tokenizer_config.json` that point past the vocabulary

Exit `0` is clean, `1` means findings, `2` means the scan could not finish. The JSON receipt lists every finding with counts and examples.

To write a repaired copy, add a repair flag and an output path. The copy is published only if it passes a rescan:

```sh
secant sanitize --model model.gguf --clamp-nonfinite --out model-fixed.gguf \
  --receipt fix.json --attempt-id repair
```

### 3. Patch weights in place on disk

Describe the edit in a small JSON plan, then apply it. This one zeroes the embedding row for token 7:

```json
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
```

```sh
secant patch --base model.safetensors --plan zero-row.json --in-place --json > patch-receipt.json
```

Only the bytes of the named tensors change. Before the file is replaced, Secant checks that the header and every other tensor are byte-identical to the original. The receipt records the SHA-256 before (`base_sha256`) and after (`out_sha256`). Drop `--in-place` and pass `--out new.safetensors` to write a copy instead.

`secant patch` handles F32, F16, BF16, and F64 weights (and I32/I64 tables) in both formats. Plans can also add a rank-one update to chosen columns (`rank1-columns`). For `Q8_0` GGUF files, `secant quant patch` applies `rank1-columns` edits and re-encodes only the 34-byte blocks the edit touches. [Recipe 4](docs/RECIPES.md#recipe-4-patch-weights-on-disk) covers both.

### 4. Keep chosen directions fixed during LoRA fine-tuning

This part is an optional Python layer. `ProjectThenHealLinear` wraps an `nn.Linear` so a LoRA adapter cannot write into output directions you choose, even under AdamW. After training it merges into a plain `nn.Linear` with no extra cost at inference.

```python
from secant.nn import ProjectThenHealLinear
locked = ProjectThenHealLinear.from_linear(linear_layer, rank=4, directions=protected_dirs)
optimizer = torch.optim.AdamW([p for p in locked.parameters() if p.requires_grad], lr=1e-3)
# ... train as usual ...
assert max(locked.leakage(), locked.merged_leakage()) < 1e-6
```

Install with `python3 -m pip install -e './python[torch]'`. The [PyTorch guide](docs/PYTORCH-10-LINE-HEALING-GUIDE.md) has a complete runnable example.

## Try it in 30 seconds

```sh
python3 examples/test-drive/test_drive.py
```

This builds tiny SafeTensors and GGUF files, runs nine checks (clean files, broken values, a repair, and a planted rank-one edit found with `--reference`), and prints `RESULT: 9/9`. No model download, GPU, or Python packages are needed. The first run builds the CLI, which needs Rust and can take a few minutes. Add `--work-dir demo` to keep the files, then follow the [quickstart](docs/quickstart.md#try-the-four-jobs-on-demo-files) to run each job on them.

## Recipes

| You want to | Recipe |
| --- | --- |
| See which tensors a fine-tune, merge, or quant changed | [1. Diff against a base model](docs/RECIPES.md#recipe-1-diff-against-a-base-model) |
| Scan a Hugging Face download before vLLM or llama.cpp loads it | [2. Scan a downloaded model](docs/RECIPES.md#recipe-2-scan-a-downloaded-model) |
| Repair broken scales or values into a hash-checked copy | [3. Repair a copy](docs/RECIPES.md#recipe-3-repair-a-copy-and-verify-its-hashes) |
| Edit a few weights without re-exporting the model | [4. Patch weights on disk](docs/RECIPES.md#recipe-4-patch-weights-on-disk) |
| Train a LoRA that cannot touch protected directions | [5. Fine-tune and export in ten lines](docs/RECIPES.md#recipe-5-fine-tune-and-export-in-ten-lines) |
| Check a Q8_0 or Q4_K export kept those directions | [6. Verify a quantized export](docs/RECIPES.md#recipe-6-verify-a-quantized-export) |
| Test a model-file scanner with known-bad fixtures | [7. Test a scanner](docs/RECIPES.md#recipe-7-test-a-scanner-with-synthetic-fixtures) |

## Exit codes

| Exit | `sanitize`, `verify-patch`, `verify-subspace`, `patch` | Next step |
| --- | --- | --- |
| `0` | Passed, or a repaired copy passed its rescan | Keep the receipt |
| `1` | Findings remain, or a verification failed | Read the findings before loading the file |
| `2` | Bad usage, I/O error, unsupported input, or an incomplete check | Treat the result as unknown and rerun |

`read` and `audit-trojan` are reports. They exit `0` whenever they finish, so read their output. Receipt and output paths must be new files.

## Scope

- **A clean scan covers the listed checks.** It will not find a behavioral backdoor hidden in normal-looking weights. On 105 benign test configurations (100 synthetic, 5 retained) the scanner raised no false positives on either test host ([benchmark](docs/benchmarks/benign-fpr-v0.3.0.md)).
- **Diffing needs a trusted base with the same tensor names and shapes.** Without one, `audit-trojan` can only flag edits large enough to stand out from the layer's own structure; the [paper](docs/PAPER.md) measures that threshold. A quantized file compared with a full-precision base shows small rounding changes everywhere, so look for tensors that changed much more than their neighbors.
- **Patching supports float tensors and `Q8_0` blocks.** `Q4_K` files can be read, diffed, scanned, and verified, but not patched. The other GGUF block formats (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q2_K`, `Q3_K`, `Q5_K`, `Q6_K`) are checked by `sanitize` only. The patcher holds the file in memory (about twice its size), so patch sharded checkpoints one shard at a time. `--in-place` writes a verified temporary file in the same folder and renames it over the original, so it needs free disk space equal to the file.
- **The LoRA lock covers the wrapped layers and the directions you pass in.** Measure leakage again after merging or quantizing.

## More

- [Quickstart](docs/quickstart.md): install, the four jobs on demo files, and the full edit workflow on a 1.7B checkpoint with terminal recordings.
- [CLI, Python, and C reference](docs/cli-and-python-reference.md): every command, flag, exit code, and receipt field.
- [Payload checks](docs/payload-security.md): the exact rules `sanitize` applies.
- [Quantized surgery](docs/quantized-surgery.md): how Q8_0 block edits work and why Q4_K patching is refused.
- [Paper](docs/PAPER.md): derivations, experiments on checkpoints from 0.8B to 72.7B parameters, and their limits.

<p align="center">
  <img src="docs/assets/secant-patch.gif" alt="secant patch terminal recording" width="49%" />
  <img src="docs/assets/secant-probe.gif" alt="secant probe terminal recording" width="49%" />
</p>

## Citation

```bibtex
@software{keys2026secant,
  title   = {Secant: Deterministic Weight-Space Programming, In-Place Quantized Block Surgery, and Subspace-Locked LoRA},
  author  = {Keys, Nathan},
  year    = {2026},
  url     = {https://github.com/professor-moody/secant}
}
```
