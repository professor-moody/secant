# CLI, Python & C-ABI Reference

Secant exposes three synchronized interfaces backed by `secant-core`:
1. **`secant` CLI** (`crates/secant-cli`) — Deterministic command-line tool with structured `--json` receipts.
2. **`secant` Python Package** (`python/secant`) — PyTorch `ProjectThenHealLinear`, contrastive capture, and CLI/FFI bridges.
3. **`libsecant` C ABI** (`crates/secant-ffi/include/secant.h`) — Stable `extern "C"` shared/static library (`*_v1` symbols).

---

## 1. Exit Code Contract

Every `secant` CLI command and C-ABI entry point obeys a strict three-state exit code contract:

| Exit Code | Meaning | Example |
|---:|---|---|
| **`0`** | **PASS** — Operation or verification gate succeeded. | `verify-patch` confirmed zero out-of-plan bytes changed; `verify-subspace` confirmed leakage $\le \tau$. |
| **`1`** | **GATE FAILED** — Deterministic check ran to completion and found a violation. | `verify-subspace` measured leakage $0.156 > 10^{-6}$; `probe` detected non-zero GGUF alignment padding. |
| **`2`** | **USAGE / IO ERROR** — Invalid arguments, missing file, or malformed container. | Missing `--model` flag or truncated GGUF header. |

---

## 2. `secant` CLI Subcommands

### `secant profiles`
Lists built-in weight-space edit profiles (`qwen35-beneficial-b1b2b3`, `qwen35-think-loop-b1`, `smollm2-beneficial-b2b3`, etc.).
```bash
secant profiles --json
```

### `secant inspect`
Parses SafeTensors or GGUF v2/v3 headers, tensor descriptors, offsets, and alignment padding without mutating the file.
```bash
secant inspect --model <path> [--json]
```

### `secant read`
Computes deterministic FMA-free `float64` leading singular triplets $(\sigma_1, u_1, v_1)$, Frobenius norms, row-norm quantiles, and top-$K$ intermediate column indices.
```bash
secant read \
  --model <path> \
  [--tensor <name>] \
  [--top-cols <K>] \
  [--top-k-singular <N>] \
  [--detect-anomalies] \
  [--json]
```

### `secant analyze`
Ranks candidate layers from captured contrastive metrics (`snr_proj`, `snr_trace`, `split_half_cosine`).
```bash
secant analyze --in <capture_metrics.json> --out <analysis.json> [--json]
```

### `secant compose`
Orthogonalizes behavioral directions via two-pass modified Gram-Schmidt in `float64` and calibrates step sizes (`gamma` or `target-shift`).
```bash
secant compose \
  --profile <name-or-json> \
  --vectors-dir <dir> \
  [--calibration gamma|target-shift] \
  [--alpha <float>] \
  [--model <base_model_for_target_shift>] \
  --out <plan.json> \
  [--json]
```

### `secant patch` & `secant verify-patch`
Applies and verifies sparse floating-point (`F32`, `F16`, `BF16`) weight updates on SafeTensors or GGUF containers.
```bash
secant patch --model <base> --plan <plan.json> --out <patched> [--json]
secant verify-patch --base <base> --patched <patched> --plan <plan.json> [--json]
```

### `secant quant patch` & `secant quant verify`
Applies direct in-place block surgery on quantized `Q8_0` and `Q4_K` GGUF tensors (`untouched_blocks_bit_identical: true`) and computes quantization commutation witnesses.
```bash
secant quant patch --model <base-q8.gguf> --plan <plan.json> --out <patched-q8.gguf> [--json]
secant quant verify --base <base.gguf> --patched <patched.gguf> --plan <plan.json> [--json]
```

### `secant heal` & `secant verify-subspace`
Merges a LoRA adapter $(A, \tilde{B})$ into a model tensor after double orthogonal projection $\PiPerp(\PiPerp(\tilde{B}))$ in `float64`, and verifies that protected-subspace leakage is bounded by `--max-leakage`.
```bash
secant heal \
  --model <base> \
  --out <healed> \
  --tensor <tensor_name> \
  --lora-a <A.npy> \
  --lora-b <B.npy> \
  --protect <d1.npy> [--protect <d2.npy> ...] \
  [--scale <alpha_over_r>] \
  [--json]

secant verify-subspace \
  --base <base> \
  --healed <healed> \
  --tensor <tensor_name> \
  --protect <d1.npy> [--protect <d2.npy> ...] \
  [--max-leakage 1e-6] \
  [--json]
```

### `secant probe`, `secant sanitize` & `secant scan-sinks`
Audits and sanitizes container structural anomalies, non-zero GGUF alignment gaps, NaN/Inf weights, and MoE router sink bias.
```bash
secant probe --model <path> [--out <receipt.json>] [--json]
secant sanitize --model <in_path> --out <clean_path> [--receipt <receipt.json>] [--json]
secant scan-sinks --model <path> [--json]
```

---

## 3. Python API Reference (`python/secant`)

### `secant.nn.ProjectThenHealLinear`

```python
class ProjectThenHealLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        directions: Union[np.ndarray, torch.Tensor],
        alpha: Optional[float] = None,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None: ...

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        rank: int,
        directions: Union[np.ndarray, torch.Tensor],
        alpha: Optional[float] = None,
    ) -> "ProjectThenHealLinear": ...

    def locked_B(self) -> torch.Tensor:
        """Returns B_locked = P(P(lora_B_tilde)) with P = I - D^T D."""

    def leakage(self) -> float:
        """Returns ||D @ B_locked||_F / ||B_locked||_F in float64."""

    def merged_delta(self) -> torch.Tensor:
        """Returns (alpha / rank) * (B_locked @ lora_A)."""

    def merge_into(self, target: Optional[torch.nn.Linear] = None) -> torch.nn.Linear:
        """Merges the subspace-locked LoRA delta into the base Linear weight."""

    def export_lora(self, dir_path: Union[str, Path], prefix: str = "") -> dict:
        """Writes lora_A.npy and locked lora_B.npy in NumPy v1.0 format for `secant heal`."""
```

---

## 4. C FFI Reference (`crates/secant-ffi/include/secant.h`)

```c
// Returns the semantic version string of libsecant ("0.1.0").
const char* secant_version_v1(void);

// Applies double orthogonal projection B_locked = P(P(B_tilde)) in float64
// and merges scale * (B_locked @ A) into W_inout (d_out x d_in, row-major f32).
int32_t secant_project_then_heal_f32_v1(
    float* w_inout,
    size_t d_out,
    size_t d_in,
    const float* lora_a,
    const float* lora_b_tilde,
    size_t rank,
    float scale,
    const double* protected_dirs,
    size_t num_dirs,
    double* out_leakage
);

// Computes the Q8_0 quantization commutation witness on a float32 weight buffer.
int32_t secant_q8_0_commutation_witness_v1(
    const float* w_base,
    const float* delta_w,
    size_t n_elements,
    double* out_rel_fro_error
);
```
