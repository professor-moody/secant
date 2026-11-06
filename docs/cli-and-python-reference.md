# CLI, Python & C-ABI Reference

Secant exposes three interfaces backed by `secant-core`:
1. **`secant` CLI** (`crates/secant-cli`): Deterministic command-line tool with structured `--json` receipts.
2. **`secant` Python Package** (`python/secant`): PyTorch `ProjectThenHealLinear`, contrastive capture, and CLI/FFI bridges.
3. **`libsecant` C ABI** (`crates/secant-ffi/include/secant.h`): Stable `extern "C"` shared/static library (`*_v1` symbols).

Command shapes were checked against the installed binary's `--help` and Rust argument definitions on 2026-11-06. Python signatures come from `python/secant/nn.py`. Pin the binary and source revision together: installed help describes that executable, not necessarily an unbuilt checkout. Examples with angle-bracket placeholders are templates, not shell-ready commands.

---

## 1. Exit Code Contract

Every `secant` CLI command obeys a three-state exit code contract:

| Exit Code | Meaning | Example |
|---:|---|---|
| **`0`** | **PASS**: Operation or verification gate succeeded. | `verify-patch` confirmed every non-target tensor is byte-identical; `verify-subspace` confirmed $\max \lvert D \Delta W\rvert \le$ `--tol` for row basis $D \in \mathbb{R}^{k \times d_{\text{out}}}$. |
| **`1`** | **GATE FAILED**: A deterministic check ran to completion and found a violation. | `verify-subspace` measured leakage above `--tol`; `sanitize` finished its scan and findings remain. |
| **`2`** | **USAGE / IO / UNDETERMINABLE**: Invalid arguments, missing file, malformed container, or a verdict that cannot be computed. | Missing `--model`, an unknown subcommand, a truncated GGUF header, or a `verify-subspace` tensor holding NaN/Inf (verdict `unknown`, never pass). |

The C ABI uses a separate five-value return code (Section 4).

---

## 2. `secant` CLI Subcommands

The binary has twelve functional root subcommands: `inspect`, `read`, `compose`, `patch`, `verify-patch`, `sanitize`, `probe`, `heal`, `verify-subspace`, `quant`, `rotation-test` and `fetch-tensor`, plus Clap's `help`. `quant` contains `witness`, `patch` and `roundtrip`. The last two root commands are described in [Native WSP Extensions](native-wsp-extensions.md). There is no `profiles`, `analyze`, or `quant verify` subcommand. The global `--go-arm64-fma` flag selects the arm64 Go oracle's fused multiply-adds; the default is unfused `float64`.

### `secant inspect`
Parses SafeTensors or GGUF headers, tensor descriptors, payload offsets, dtype layout, and the Qwen3.5 `ssm_out` interleave without mutating the file.
```bash
secant inspect --model <path.gguf|path.safetensors> [--json]
```

### `secant read`
Computes deterministic FMA-free `float64` leading singular triplets, Frobenius norms, top input columns, optional anomalies, and an optional vocabulary projection. Read-only.
```bash
secant read \
  --model <path> \
  [--tensor <glob>] \
  [--top-cols <0..1048576, default 64; 0 selects default>] \
  [--top-k-singular <0..64, default 4; 0 selects default>] \
  [--vocab-tensor <unembedding tensor>] \
  [--detect-anomalies] \
  [--vectors-out <new_dir>] \
  [--json]
```

### `secant compose`
Orthogonalizes behavioral directions via two-pass modified Gram-Schmidt in `float64` and calibrates step sizes. `--profile` takes a built-in name (`adversarial-moe-router-probe`, `adversarial-nan-probe`, `refusal-ablate`, `security-auditor-b1b2b3`, `think-loop-fix`) or a `.json` path to a `secant.wsp.profile.v1` file. If neither `--target-shift` nor `--gamma` is given, the profile's `default_target_shift` (or else `default_gamma`) applies. `--analysis` is accepted by the parser but always exits `2`: `wsp analyze` is not ported, and layers bind from the sorted vector filenames.
```bash
secant compose \
  --profile <name-or-json> \
  --vectors <dir> \
  [--target-shift <value> | --gamma <value> [--base <model>]] \
  --out <plan.json> \
  [--json]
```

### `secant patch` & `secant verify-patch`
`patch` applies a plan's weight edits directly to payload bytes (`F32`, `F16`, `BF16`) of a SafeTensors or GGUF file, writing a patched copy unless `--in-place` is given. `verify-patch` checks header identity and 100% byte-identity of every tensor not named in `--expect-changed`.
```bash
secant patch --base <base> --plan <plan.json> --out <patched> [--in-place] [--json]
secant verify-patch --base <base> --patched <patched> --expect-changed <tensor1,tensor2,...> [--json]
```

### `secant quant witness`, `secant quant patch` & `secant quant roundtrip`
`quant patch` performs block-aligned surgery on **`Q8_0` tensors only**: only the 34-byte blocks holding an edited element are re-quantized. A tensor of any other dtype (including `Q4_K`) is refused with the error `tensor <name> is <dtype>, block-aligned surgery needs Q8_0` (`crates/secant-core/src/quant.rs`). `--out` is required unless `--in-place` is given. There is no `quant verify`; use `secant verify-patch` for the byte-difference check. `quant witness` measures $\lVert D(\mathrm{Dequant}(\mathrm{Quant}(W')) - W')\rVert_F$ for `q8_0` or `q4_k`, optionally under the $\Pi_{128}$ panel permutation.
```bash
secant quant patch --base <base-q8_0.gguf> --plan <plan.json> --out <patched-q8_0.gguf> [--in-place] [--json]
secant quant witness --model <model> --tensor <name> --vectors <dir-or-npy> [--format q8_0|q4_k] [--pi128] [--json]
secant quant roundtrip --npy <matrix.npy> [--format q8_0|q4_k] [--json]
```

### `secant heal` & `secant verify-subspace`
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
`sanitize` scans static weight values (NaN, Inf, extreme finite, subnormal, zero or tiny-norm vectors, out-of-range and duplicate routing ids) and can publish a repaired copy. `--receipt` and `--attempt-id` are required. `--out` must be supplied together with at least one remediation flag (`--clamp-nonfinite`, `--dedup-routing-rows`, `--clamp-extreme`, `--flush-subnormal`); otherwise it fails with `remediation flags and a separate --out must be supplied together`. `probe` generates header-valid boundary-value fixtures and requires `--out-dir`, `--receipt` and `--attempt-id`; it does not scan a model. `read --detect-anomalies` reports `zero_rows`, `collapsed_rows` and `spike_cols`. Rules and exit codes: [Payload Security](payload-security.md). The standalone `secant` CLI has no `scan-sinks` subcommand.
```bash
secant sanitize --model <in_path> --receipt <new.json> --attempt-id <id> \
  [--num-experts <N>] [--routing-tensor <glob>]... [--norm-tensor <glob>]... \
  [--max-abs <default 10000>] [--min-norm <default 1e-12>] [--max-examples <1..10000>] \
  [--clamp-nonfinite] [--dedup-routing-rows] [--clamp-extreme] [--flush-subnormal] \
  [--out <new_clean_path>] [--json]
secant probe --class <class1|class2|class3|class4|all> --format <safetensors|gguf|both> \
  --out-dir <new_dir> --receipt <new.json> --attempt-id <id> \
  [--base <model>] [--num-experts <2..1048576, default 8>] [--json]
secant read --model <path> [--tensor <glob>] --detect-anomalies [--json]
```

---

### Receipt scope and field meanings

Rust emits `secant.wsp.inspect.v1`, `secant.wsp.read.v1`, `secant.wsp.compose.v1`, `secant.wsp.patch.v1`, `secant.wsp.verify.v1`, `secant.wsp.heal.v1`, `secant.wsp.verify-subspace.v1`, `secant.wsp.sanitize.v1` and `secant.wsp.probe.v1` for the corresponding commands. `quant witness` uses `secant.quant.witness.v1`, and `quant patch` uses `secant.quant.patch.v1`. `quant roundtrip` currently emits `format`, `rows`, `cols`, `bytes`, `max_abs_error`, `rmse` and `sha256` without a `schema` field. Older banked receipts and the Go interface may use a `crucible` prefix; retain the producer identity instead of relabeling those receipts.

- `read`: `pre_sha256`, `post_sha256` and `input_unchanged` bracket the model read; a successful command can still contain `status: "nonfinite"` tensors.
- `verify-patch`: `header_bytes_identical`, `other_tensors_byte_identical` and `ok` test byte invariants. The allowlist permits changes; it does not establish that the intended numerical edit occurred.
- `heal`: `pairs[].leak_after_f64` and `pairs[].leak_after_serialized` measure $\max|DB|$, while the optional `merge.gate` checks $\max|D\Delta W|$. The latter also depends on $A$, scale and storage rounding. `out_published` and `merge.published` must be checked separately.
- `verify-subspace`: `verdict` and `tensors[].verdict` can be `unknown`. `changed_tensors_not_checked` lists changed tensors outside the requested scope; a scoped pass is not a whole-model invariance claim.
- `quant patch`: `blocks_touched` counts logical support; `blocks_rewritten` counts touched blocks whose bytes changed. Neither is a count of changed scalar weights.
- `quant witness`: `witness_l2` is the Frobenius norm of the `k x d_in` projected error matrix. `dt_w_fro_before` is a legacy field name; with this page's row basis its value is $\|DW\|_F$.

## 3. Python API Reference (`python/secant`)

### `secant.nn.ProjectThenHealLinear`

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
        """max |D B_locked| in float64, measured on the working-dtype B_locked that forward() uses."""

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

---

## 4. C FFI Reference (`crates/secant-ffi/include/secant.h`)

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
