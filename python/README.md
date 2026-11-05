# secant (Python)

Python side of the Secant workspace: Project-Then-Heal LoRA for torch,
contrastive direction capture with held-out metrics, and bridges to the
`secant` CLI and the `libsecant` C ABI. Requires Python 3.9+ and numpy;
`secant.nn` additionally needs torch.

    pip install -e '.[torch,test]'     # from this directory

## What is in it

- `secant.nn.ProjectThenHealLinear`: a frozen linear layer plus a LoRA update
  `y = base(x) + scale * (x A^T) B_locked^T` with `B_locked = P (P B_tilde)`,
  `P = I - D^T D`. `D` (k x d_out) holds orthonormal protected output
  directions (two-pass modified Gram-Schmidt in float64 at construction;
  dependent, zero or NaN input raises). Trainable: `lora_A`, `lora_B_tilde`.
  There is no LoRA bias. `P` is applied as `B - D^T (D B)`; no
  `d_out x d_out` matrix is ever built (the test records every op output
  during forward and backward). Methods: `locked_B()`, `leakage()`,
  `merged_delta()`, `merge_into()`, `merged_leakage()`, `export_lora()`,
  `ProjectThenHealLinear.from_linear()`.
- `secant.capture.capture_contrastive_directions`: difference-of-means
  directions per (stream, layer), from activation arrays or from a model with
  forward hooks, with held-out `snr_proj`, `snr_trace` and `split_half_cosine`
  on a deterministic 50/50 split. `write_vectors` writes `d_<stream>_L<N>.npy`
  (NumPy v1.0, float64) for `secant compose` / `secant heal`.
- `secant.run` / `secant.cli` and the helpers `inspect`, `read`, `compose`,
  `heal`, `verify_subspace`, `verify_patch`, `patch`: subprocess wrappers that
  mirror the `secant` CLI flag set. Exit 0 passed, 1 a gate failed (returned),
  2 undeterminable (helpers raise).
- `secant.load_library`, `project_then_heal_f32`, `q8_0_commutation_witness`,
  `library_version`: ctypes calls into the fixed C ABI (`*_v1`).

Binary / library discovery: explicit argument, then `SECANT_BIN` / `SECANT_LIB`
(absent: search; set but empty: error; set: use exactly that), then PATH and
`./target/{release,debug}`.

## Limits worth knowing

- Leakage guarantees are numerical. A float32 module reaches about 1e-8 times
  the entry size of `B_locked`, not exactly zero; bfloat16/float16 modules and
  merges into bf16/f16 weights leak far more and `leakage()` /
  `merged_leakage()` report it.
- `snr_trace` is a plug-in estimate and is biased upward on small samples.
- Tests that need the real binary or library skip with an explicit
  `SKIP-*` reason when it is missing; `tests/test_ffi_shim.py` checks only the
  ctypes plumbing against a small C reference, not libsecant.
- `export_lora` does not store `alpha`; pass the returned `scale` to
  `secant heal --scale` when merging.

## Tests

    python3 -m pytest tests -ra
