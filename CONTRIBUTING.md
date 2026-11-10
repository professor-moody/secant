# Contributing to Secant

Thank you for contributing to Secant (`secant-core`, `secant-cli`, `secant-ffi`, and `python/secant`). Secant is built around deterministic arithmetic, fail-closed binary verification, and reproducible receipts. Please review the invariants below before opening an issue or pull request.

## Workspace Layout

- `crates/secant-core`: Container parsers (`.safetensors`, `.gguf`), two-pass modified Gram-Schmidt subspace orthogonalization, `Q8_0`/`Q4_K` block codecs, `Project-Then-Heal` LoRA locking, `Rule S-0` / `Boundary-3` payload sanitization, and Marchenko-Pastur spectral auditing.
- `crates/secant-cli`: Command-line interface (`secant`) enforcing the `0` / `1` / `2` exit-code contract and JSON receipt schemas (`secant.wsp.*`).
- `crates/secant-ffi`: Frozen `_v1` C ABI (`libsecant`) with checked extent multiplication and panic containment at the FFI boundary.
- `python/`: PyTorch `ProjectThenHealLinear` module and contrastive activation capture utilities.

## Engineering Invariants

1. **No `unsafe` Rust outside `secant-ffi` (and the `memmap2` file-mapping boundary):**
   - New modules in `secant-cli` and `secant-core` must uphold `#![forbid(unsafe_code)]` semantics—do not introduce `unsafe` blocks, raw pointer casts, or unchecked slice indexing for parsing or numerical kernels.
   - All C ABI `unsafe` code is isolated inside `crates/secant-ffi` (`*_v1` functions), where every pointer check, `usize::checked_mul` extent validation, and `catch_unwind` guard must be documented with an explicit `// SAFETY:` comment.
   - Container parsing must use checked 64-bit integer arithmetic (`checked_add`, `checked_mul`) on every header offset and tensor shape.

2. **Deterministic, bit-exact floating-point fencing:**
   - Default `secant` execution must produce **bit-identical** `float64`/`float32` results across `x86_64` and `aarch64` without hardware FMA contraction drift.
   - Route multiply-add/sub inner loops through `secant_core::numeric::{fma_add, fma_sub}` (which evaluate unfused `(a * b) + c` in portable mode) and keep fixed index-order summation—never parallelize reductions in ways that reorder floating-point additions.
   - Preserve IEEE-754 round-to-nearest-even (`RNE`) `F16` and `BF16` conversions (including `F16` subnormals `0x0001..0x03FF`).

3. **Fail-closed I/O, hash bracketing, and exit codes (`0` / `1` / `2`):**
   - Never silently overwrite existing output or receipt files (`create_new` / exclusive file creation unless `--in-place` is explicitly supported and requested).
   - Bracket read/audit operations with `pre_sha256` and `post_sha256` digests (`input_unchanged == true`).
   - Unreadable or malformed sibling companion configs (`config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`), unsupported dtypes (e.g., `F8_E4M3`), or unbounded index tensors must **fail closed** (`status: "unknown"`, exit code `2`), never silently pass as clean.

## Required Verification

Run the full Rust workspace test suite (and Python tests when touching `python/`) before submitting a change:

```sh
cargo test --workspace
cargo clippy --workspace --all-targets
```

Every bug fix, scanner rule, or numerical change must include deterministic unit or integration tests covering both clean inputs and adversarial/edge-case controls.

## AI-Assistance Disclosure & Human Review Policy

AI coding assistants may be used when drafting code, tests, or documentation, subject to two strict requirements:

1. **Mandatory human review and verification:** Every line of code, mathematical invariant, security check, and test assertion must be personally reviewed, understood, and executed locally by the human author prior to submission. Unverified or speculative output is not accepted, and contributors must not post unreviewed AI-generated replies in code review threads.
2. **Concise disclosure:** Pull requests drafted or validated with AI tooling must include a concise 1-sentence disclosure in the PR description (for example: *"AI assistance was used in drafting and validating this change; the patch and tests were reviewed and verified by the author."*).

## Licensing & Conduct

Contributions are licensed under the [Apache License 2.0](LICENSE) and governed by our [Code of Conduct](CODE_OF_CONDUCT.md). For security vulnerabilities, follow [SECURITY.md](SECURITY.md) rather than opening a public issue.
