# CLI, Python, and C Reference

Choose the interface that matches your workflow: shell commands, Python training code, or a C-compatible library call. The Rust CLI and C library use `secant-core`; the Python package also implements capture and the trainable layer.

1. **`secant` CLI** (`crates/secant-cli`): Deterministic command-line tool with structured `--json` receipts.
2. **`secant` Python Package** (`python/secant`): PyTorch `ProjectThenHealLinear`, contrastive capture, and CLI/FFI bridges.
3. **`libsecant` C ABI** (`crates/secant-ffi/include/secant.h`): Stable `extern "C"` shared/static library (`*_v1` symbols).

A **receipt** is a structured record of a command's inputs, measurements, and checks. A successful exit applies to the command's stated scope; it does not certify the model's behavior.

Command shapes were checked against the installed binary's `--help` and Rust argument definitions on 2026-11-06. Python signatures come from `python/secant/nn.py`. Pin the binary and source revision together: installed help describes that executable, not necessarily an unbuilt checkout. Examples with angle-bracket placeholders are templates, not shell-ready commands.

---

## 1. Exit Code Contract

Every `secant` CLI command obeys a three-state exit code contract:

| Exit Code | Meaning | Example |
|---|---|---|
| **0** | **PASS**: Operation or verification gate succeeded. | verify-patch confirmed every non-target tensor is byte-identical; verify-subspace confirmed max abs(D ΔW) ≤ --tol for row basis D with k rows and dout columns. |
| **1** | **GATE FAILED**: A deterministic check ran to completion and found a violation. | verify-subspace measured leakage above --tol; sanitize finished its scan and findings remain. |
| **2** | **USAGE / IO / UNDETERMINABLE**: Invalid arguments, missing file, malformed container, or a verdict that cannot be computed. | Missing --model, an unknown subcommand, a truncated GGUF header, or a verify-subspace tensor holding NaN/Inf (verdict unknown, never pass). |

The C ABI uses a separate five-value return code (Section 4).

---

## 2. `secant` CLI Subcommands

The binary has sixteen functional root subcommands: `profiles`, `inspect`, `read`, `audit-trojan`, `analyze`, `compose`, `patch`, `verify-patch`, `sanitize`, `scan-sinks`, `probe`, `heal`, `verify-subspace`, `quant`, `rotation-test` and `fetch-tensor`, plus Clap's `help`. `quant` contains `witness` (alias `verify`), `patch` and `roundtrip`. `rotation-test` and `fetch-tensor` are described in [Native WSP Extensions](native-wsp-extensions.md). The global `--go-arm64-fma` flag selects the arm64 Go oracle's fused multiply-adds; the default is unfused `float64`.

### `secant profiles`
Lists the five built-in Weight-Space Programming research profiles (`think-loop-fix`, `security-auditor-b1b2b3`, `refusal-ablate`, `adversarial-nan-probe`, `adversarial-moe-router-probe`), their subspace definitions, default target activation shifts, and layer-selection rules.
```bash
secant profiles [--json]
```

### `secant inspect`
Parses SafeTensors or GGUF headers, tensor descriptors, payload offsets, dtype layout, and the Qwen3.5 `ssm_out` interleave without mutating the file.
```bash
secant inspect --model <path.gguf|path.safetensors> [--json]
```

### `secant read`
Reads matrix structure without running the model. A singular triplet describes a leading input direction, output direction, and their scale; the Frobenius norm measures overall matrix size.

Computes deterministic FMA-free `float64` leading singular triplets, Frobenius norms, top input columns, optional anomalies, and an optional vocabulary projection. Read-only.
```bash
secant read \
  --model <path> \
  [--tensor <glob>] \
  [--top-cols <0..1048576, default 64; 0 selects default>] \
  [--top-k-singular <0..64, default 4; 0 selects default>] \
  [--vocab-tensor <unembedding tensor>] \
  [--detect-anomalies] \
  [--marchenko-pastur [--mp-threshold <ratio, default 1.5>]] \
  [--vectors-out <new_dir>] \
  [--json]
```

`--marchenko-pastur` adds a per-tensor `marchenko_pastur` block and a report-level `spectral_outliers` ranking (see [Payload Security](payload-security.md), section 5). `secant audit-trojan --model <path> [--tensor <glob>] [--vocab-tensor <name>] [--top-k-singular <k>] [--mp-threshold <r>] [--top <rows>] [--json]` runs the same reader with `--detect-anomalies --marchenko-pastur` and prints the ranking. Both exit `0` whether or not anything is flagged.

`--reference <base>` (on `read` and `audit-trojan`) replaces `W` by `Delta W = W - W_ref` for every selected tensor, matching tensors by name and logical shape; `--vocab-tensor` is still read from `--model`. Tensors the reference lacks, or holds with another shape, are reported `skipped` with the reason. Each tensor gains `reference_diff` (`status`, `basis` = `delta_w`, `reference_dtype`, `reference_frobenius`, `delta_frobenius`, `relative_delta`, `identical`, `sigma1_energy_fraction`) and the report gains a `reference` block (`path`, `format`, `pre_sha256`, `post_sha256`, `input_unchanged`, `tensors_compared`, `tensors_skipped`). A reference equal to the model file, or with no comparable tensor, exits `2`. Without the flag the receipt has none of these keys.

### `secant analyze`
Combines layer rankings from measurements you already captured. Borda voting assigns points by rank and combines those points across streams; it does not run the model to gather new measurements.

Ranks layers per subspace (`resid`, `attn`, `mlp`, `h`) by precomputed or externally captured per-layer SNR cells (`layer_snr.json` or `sweep_summary.json`), aggregates with Borda rank voting across subspaces to select $L_{\text{loop}}^*$, $L_{\text{ref}}^*$, and $L_{\text{root}}^*$, and diagnoses why hardcoded mid-layer defaults (`L15`/`L19`) miss the bottleneck at other model scales.
```bash
secant analyze \
  --in <layer_snr.json|sweep_summary.json> \
  [--profile <name>] \
  [--out <analysis.json>] \
  [--json]
```

### `secant compose`
Turns directions, reference activations, and an edit profile into a plan. Orthogonalizing removes overlap between directions. Calibration chooses the edit size for the reference input, not for every possible prompt.

Orthogonalizes behavioral directions via two-pass modified Gram-Schmidt in `float64` and calibrates step sizes. `--profile` takes a built-in name (`adversarial-moe-router-probe`, `adversarial-nan-probe`, `refusal-ablate`, `security-auditor-b1b2b3`, `think-loop-fix`) or a `.json` path to a `secant.wsp.profile.v1` file. If neither `--target-shift` nor `--gamma` is given, the profile's `default_target_shift` (or else `default_gamma`) applies. When `--analysis <analysis.json>` (`secant.wsp.analyze.v1`) is provided, automatic layer slots (`layer: -1`) bind in the analysis family's Borda order ($L_{\text{loop}}^*$, $L_{\text{ref}}^*$, $L_{\text{root}}^*$) instead of sorted vector filenames.
```bash
secant compose \
  --profile <name-or-json> \
  --vectors <dir> \
  [--analysis <analysis.json>] \
  [--target-shift <value> | --gamma <value> [--base <model>]] \
  --out <plan.json> \
  [--json]
```

### `secant patch` & `secant verify-patch`
`patch` applies a plan's weight edits to an owned snapshot of a SafeTensors or GGUF file. It validates all edits before writing an exclusively created temporary file, verifies the result, and atomically publishes the output. Copy mode rejects a base/output alias, including hard links and symlinks. `--in-place` atomically replaces the base path; other hard links retain the original inode. Failed edits and publication failures preserve existing base/output bytes. No-op plans are valid.

`verify-patch` checks total file length, header identity, every inter-tensor gap and trailing byte, and 100% byte-identity of every tensor not named in `--expect-changed`. Expected tensors must actually differ. All shared readers/writers reject overlapping nonempty tensor extents, invalid GGUF alignment/offsets, and partial quantized rows; empty SafeTensors extents remain valid.

Callers must serialize external writers to base/output paths through publication. Patching uses private memory and rechecks the pinned input's identity and SHA-256 before publication. Hash bracketing detects changes; it is not synchronization and cannot exclude a writer racing the final rename. Read-only `Model` mappings also require callers to exclude writes and truncation for their lifetime. Patching temporarily holds two full model images in memory.
```bash
secant patch --base <base> --plan <plan.json> --out <patched> [--in-place] [--json]
secant verify-patch --base <base> --patched <patched> --expect-changed <tensor1,tensor2,...> [--json]
```

### `secant quant witness` (`verify`), `secant quant patch` & `secant quant roundtrip`
Measure compression error with `witness` or `roundtrip`; use `patch` to change checkpoint weights. A quantized block groups values that share encoding parameters. A witness is a numerical diagnostic, not a behavioral test.

`quant patch` performs block-aligned surgery on **`Q8_0` tensors only**: only the 34-byte blocks holding an edited element are re-quantized. A tensor of any other dtype (including `Q4_K`) is refused with the error `tensor <name> is <dtype>, block-aligned surgery needs Q8_0` (`crates/secant-core/src/quant.rs`). `--out` is required unless `--in-place` is given. Use `secant verify-patch` for whole-file byte-difference verification, and `secant quant witness` (or `secant quant verify`) to measure $\lVert D(\mathrm{Dequant}(\mathrm{Quant}(W')) - W')\rVert_F$ for `q8_0` or `q4_k`, optionally under the $\Pi_{128}$ panel permutation.
```bash
secant quant patch --base <base-q8_0.gguf> --plan <plan.json> --out <patched-q8_0.gguf> [--in-place] [--json]
secant quant witness --model <model> --tensor <name> --vectors <dir-or-npy> [--format q8_0|q4_k] [--pi128] [--json]
secant quant roundtrip --npy <matrix.npy> [--format q8_0|q4_k] [--json]
```

### `secant heal` & `secant verify-subspace`
`heal` removes protected components from an existing adapter once. `verify-subspace` compares stored base and healed weights at selected tensors. Neither command keeps an exported adapter constrained during later training; use the Python layer for that.

`heal` orthonormalizes the protected directions $D$ (direction `.npy`/`.json` files in a directory, or one `.npy`), replaces every LoRA $B$ in a `lora_A`/`lora_B` `.safetensors` adapter by $B_{\text{locked}} = P_\perp(P_\perp B)$, where $P_\perp = I - D^\top D$, and writes a new healed adapter. With `--base` and `--merge-out` it also adds $\Delta W = \text{scale}\cdot B_{\text{locked}} A$ into a copy of the base and publishes it only if the realized $\max\lvert D \Delta W\rvert \le$ `--tol`. `verify-subspace` recomputes $\Delta W = W_{\text{healed}} - W_{\text{base}}$ from stored bytes for each named tensor.
```bash
secant heal \
  --lora <adapter.safetensors> \
  --vectors <dir-or-npy> \
  --out <healed-adapter.safetensors> \
  [--base <model> --merge-out <merged-model>] \
  [--scale <lora_alpha/r, default 1>] \
  [--tol <default 1e-5>] \
  [--json]

secant verify-subspace \
  --base <pre-heal-model> \
  --healed <healed-model> \
  --vectors <dir-or-npy> \
  --tensors <tensor1,tensor2,...> \
  [--tol <default 1e-5>] \
  [--json]
```

### `secant sanitize`, `secant probe` & `secant read --detect-anomalies`
`sanitize` scans static weight values and can publish a repaired copy. Its checks cover non-finite and extreme values, tiny-norm vectors, divisor scales, and invalid or duplicate routing ids. For `Q8_0` and `Q4_K`, Rule B3-2A permits zero and FP16 subnormal multiplicative block scales by default; `--flush-subnormal` opts into stricter subnormal diagnostics and repair. `--receipt` and `--attempt-id` are required. `--out` must be supplied together with at least one remediation flag (`--clamp-nonfinite`, `--dedup-routing-rows`, `--clamp-extreme`, `--flush-subnormal`); otherwise it fails with `remediation flags and a separate --out must be supplied together`.

`probe` generates header-valid boundary-value fixtures and requires `--out-dir`, `--receipt` and `--attempt-id`; it does not scan a model.

`sanitize` also scans all 10 standard GGUF quantized scale-header formats (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` -> `quant_scale_*` findings); other quantized types exit `2`. `secant probe --class all --format both --out-dir <new_dir> --receipt <new.json> --attempt-id <id>` deterministically generates self-verified synthetic witness bundles across all four Boundary-3 failure classes (`class1` router/gate NaN/-Inf, `class2` scale/bias NaN/+-1e38, `class3` NaN/sign/payload/tie arrays, `class4` I32/I64 routing tables and zero/subnormal norm vectors) so users can test `secant sanitize` out of the box. `read --detect-anomalies` reports `zero_rows`, `collapsed_rows` and `spike_cols`. `audit-trojan` runs a two-stage spectral detector: Stage 1 iteratively peels outlier singular values ($\sigma_i > \sigma_{MP}$) from the Frobenius energy ($\|W\|_F^2 - \sum_{i=1}^k \sigma_i^2$) until the residual Marchenko-Pastur bulk edge $\sigma_{MP}$ converges to a fixed point, alongside row/column norm spike detection; Stage 2 ranks each tensor's peeled $\sigma_1/\sigma_{MP}$ ratio against its architectural sibling layer family via the Iglewicz-Hoaglin modified $z$-score ($z > 3.5$) and projects $u_1$ onto `--vocab-tensor` ($E \cdot u_1$). `scan-sinks` ranks C/C++/CUDA/Rust source sites for the weight-value sink classes. Rules and exit codes: [Payload Security](payload-security.md).
```bash
secant sanitize --model <in_path> --receipt <new.json> --attempt-id <id> \
  [--num-experts <N>] [--routing-tensor <glob>]... [--norm-tensor <glob>]... \
  [--divisor-scale-tensor <glob>]... [--no-default-divisor-scales] \
  [--index-tensor <glob>]... [--index-bound <N>] \
  [--max-abs <default 10000>] [--min-norm <default 1e-12>] [--divisor-floor <default 1e-12>] \
  [--max-examples <1..10000>] [--strict-subnormal] [--dry-run] \
  [--clamp-nonfinite] [--dedup-routing-rows] [--clamp-extreme] [--flush-subnormal] \
  [--out <new_clean_path>] [--json]
secant probe --class <class1|class2|class3|class4|all> --format <safetensors|gguf|both> \
  --out-dir <new_dir> --receipt <new.json> --attempt-id <id> \
  [--base <model>] [--num-experts <2..1048576, default 8>] [--json]
secant read --model <path> [--tensor <glob>] --detect-anomalies [--json]
secant scan-sinks --source <dir> [--top <N, default 100>] [--attempt-id <id>] \
  [--out <new.json>] [--json] [--show <rows, default 25>]
```

---

### Receipt Scope and Field Meanings

Check the schema, producer, and scope before comparing results. Similar field names can measure different objects: the adapter factor, the full weight update, or the error after storage rounding.

Rust emits `secant.wsp.profiles.v1`, `secant.wsp.inspect.v1`, `secant.wsp.read.v1`, `secant.wsp.analyze.v1`, `secant.wsp.compose.v1`, `secant.wsp.patch.v1`, `secant.wsp.verify.v1`, `secant.wsp.heal.v1`, `secant.wsp.verify-subspace.v1`, `secant.wsp.sanitize.v1`, `secant.wsp.probe.v1` and `secant.wsp-sink-scan.v1` (`scan-sinks`) for the corresponding commands; `audit-trojan` writes `secant.wsp.read.v1`. `quant witness` (`verify`) uses `secant.quant.witness.v1`, and `quant patch` uses `secant.quant.patch.v1`. `quant roundtrip` currently emits `format`, `rows`, `cols`, `bytes`, `max_abs_error`, `rmse` and `sha256` without a `schema` field. Retain the producer identity on all archived receipts.

- `read`: `pre_sha256`, `post_sha256` and `input_unchanged` bracket the model read; a successful command can still contain `status: "nonfinite"` tensors.
- `verify-patch`: `header_bytes_identical`, `other_tensors_byte_identical` and `ok` test byte invariants. The allowlist permits changes; it does not establish that the intended numerical edit occurred.
- `heal`: `pairs[].leak_after_f64` and `pairs[].leak_after_serialized` measure $\max|DB|$, while the optional `merge.gate` checks $\max|D\Delta W|$. The latter also depends on $A$, scale and storage rounding. `out_published` and `merge.published` must be checked separately.
- `verify-subspace`: `verdict` and `tensors[].verdict` can be `unknown`. `changed_tensors_not_checked` lists changed tensors outside the requested scope; a scoped pass is not a whole-model invariance claim.
- `quant patch`: `blocks_touched` counts logical support; `blocks_rewritten` counts touched blocks whose bytes changed. Neither is a count of changed scalar weights.
- `quant witness`: `witness_l2` is the Frobenius norm of the `k x d_in` projected error matrix. `dt_w_fro_before` is a legacy field name; with this page's row basis its value is $\|DW\|_F$.

## 3. Python API Reference (`python/secant`)

Use the trainable layer to enforce the constraint during fine-tuning. Use the bridge functions to call the CLI from Python and inspect both its exit code and structured output.

### `secant.nn.ProjectThenHealLinear`

This wraps a linear layer with a trainable LoRA adapter. The base weight and bias are copied and frozen. The protected directions are rows of `D`, with one column per output feature. “Leakage” below means a remaining component along those directions.

Signatures copied from `python/secant/nn.py`:

```python
class ProjectThenHealLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        directions: Any,
        alpha: Optional[float] = None,
        *,
        bias: bool = False,
        lora_bias: bool = False,
        weight: Optional[torch.Tensor] = None,
        base_bias: Optional[torch.Tensor] = None,
        b_init_std: float = 0.0,
        module_name: str = "module",
        dtype: Optional[torch.dtype] = None,
        device: Optional[Any] = None,
    ) -> None: ...

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        directions: Any,
        alpha: Optional[float] = None,
        **kwargs: Any,
    ) -> "ProjectThenHealLinear":
        """Wrap an existing `nn.Linear`: weight/bias are COPIED and frozen."""

    def locked_B(self) -> torch.Tensor:
        """B_locked = P (P B_tilde) [d_out, r], float64 accumulation, cast to the working dtype."""

    def leakage(self) -> float:
        """max |D B_locked| in float64, measured on the stored-dtype B_locked (`lora_B_tilde.dtype`). Note: if `forward(x)` runs with a lower-precision `x.dtype` (e.g. `bfloat16`) or inside `torch.autocast`, `F.linear` casts `B_locked` to that lower dtype (`~1e-3` BF16 floor); wrap `forward()` in `torch.amp.autocast(device_type, enabled=False)` with `float32`/`float64` activations if runtime matmul precision must match `leakage()`."""

    def orthonormality_error(self) -> float:
        """max |D D^T - I| (float64)."""

    def merged_delta(self) -> torch.Tensor:
        """scale * B_locked @ A, [d_out, d_in], float64 accumulation, cast to the working dtype."""

    def merge_into(self, base_weight: Optional[torch.Tensor] = None, *, inplace: bool = False) -> torch.Tensor:
        """Return base_weight + merged_delta() (float64 sum, cast to base_weight's dtype)."""

    def merged_leakage(self, base_weight: Optional[torch.Tensor] = None) -> float:
        """Realized leakage of the merge: max |D (W_merged - W)| over the stored dtype."""

    def lora_tensors(self, module_name: Optional[str] = None, *, locked: bool = True, dtype: Optional[str] = None) -> Dict[str, np.ndarray]:
        """The `<module>.lora_A.weight` / `<module>.lora_B.weight` arrays `export_lora` writes."""

    def export_lora(
        self,
        path: str,
        module_name: Optional[str] = None,
        *,
        locked: bool = True,
        dtype: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Write a LoRA .safetensors for `secant heal --lora`."""


def export_lora_adapter(
    path: str,
    modules: Mapping[str, ProjectThenHealLinear],
    *,
    locked: bool = True,
    dtype: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Write several modules (name -> module) into one LoRA .safetensors."""
```

Notes that follow from the source:

- `leakage()` returns the **maximum absolute entry** of $D B_{\text{locked}}$, not a ratio.
- `merge_into()` returns a **tensor** (the merged weight), not an `nn.Linear`. With `inplace=True` it writes into `base_weight` and returns it.
- `export_lora()` writes one `.safetensors` file holding `<module>.lora_A.weight` $[r, d_{\text{in}}]$ and `<module>.lora_B.weight` $[d_{\text{out}}, r]$ as `F32` (`F64` for `float64` modules). `alpha` is not stored in the tensor file; pass the returned `scale` to `secant heal --scale`.
- `directions` with dependent, zero, NaN/Inf rows, or $k > d_{\text{out}}$ raises `DirectionError` (fail closed). `lora_bias=True` raises `ValueError`.

### `secant` CLI Bridge Functions (`python/secant/_bridge.py`)

Every CLI subcommand is also callable from Python, returning a `CliResult` (`exit_code`, `stdout`, `stderr`, `data` parsed from `--json`):

```python
import secant

profs = secant.profiles()
info  = secant.inspect("model.safetensors")
spec  = secant.read("model.safetensors", tensor="model.layers.12.mlp.down_proj.weight", vocab_tensor="lm_head.weight")
rank  = secant.analyze("layer_snr.json", profile="security-auditor-b1b2b3", out="analysis.json")
plan  = secant.compose("security-auditor-b1b2b3", "plan.json", vectors="directions/", analysis="analysis.json", target_shift=-4.0)
patch = secant.patch("model.safetensors", "plan.json", out="edited.safetensors")
ver   = secant.verify_patch("model.safetensors", "edited.safetensors", expect_changed="model.language_model.layers.12.mlp.down_proj.weight")
heal  = secant.heal("adapter.safetensors", "directions/", "healed_adapter.safetensors")
vsub  = secant.verify_subspace("model.safetensors", "merged.safetensors", "directions/", "model.layers.12.mlp.down_proj.weight")
```

---

## 4. C FFI Reference (`crates/secant-ffi/include/secant.h`)

The foreign-function interface (FFI) lets C-compatible callers project an adapter or measure quantization error. Arrays must have the dimensions and memory layout stated in the header. These functions use their own return codes, not the CLI's exit-code contract.

C declarations from the frozen header; the two leakage comments below use the row-basis convention of this page. The header's original $D^\top$ comments have a transpose error for its declared `k x d_out` storage.

```c
/* Secant C ABI (libsecant.a / libsecant.so / libsecant.dylib). Frozen _v1 signatures.
 * Return codes: 0 ok; 1 null pointer / zero extent; 2 size overflow;
 * 3 refused (NaN/Inf, dependent directions, misaligned shape); 4 caught panic. */

/* Static NUL-terminated version string; never free. */
const char *secant_version_v1(void);

/* Message for the last non-zero return on this thread. */
const char *secant_last_error_v1(void);

/* b <- P(P b), P = I - D^T D. directions: row-major k x d_out (orthonormalized
 * internally in float64); b: row-major d_out x r, modified in place.
 * *max_leak_out (nullable) <- max |D b_locked| over the stored floats. */
int32_t secant_project_then_heal_f32_v1(const float *directions, size_t k, size_t d_out,
                                        float *b, size_t r, double *max_leak_out);

/* ||D (Dequant_Q8_0(Quant_Q8_0(w)) - w)||_F and its max-abs entry.
 * w: row-major d_out x d_in (d_in % 32 == 0); directions: row-major k x d_out. */
int32_t secant_q8_0_commutation_witness_v1(const float *directions, size_t k, size_t d_out,
                                           const float *w, size_t d_in,
                                           double *out_l2, double *out_max_abs);
```

The C ABI locks a LoRA $B$ matrix in place; it does not merge into a weight matrix. Merging is done by `secant heal --base --merge-out` or `ProjectThenHealLinear.merge_into`.
