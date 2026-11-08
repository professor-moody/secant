# Changelog

All notable changes to `secant` (`secant-rs` CLI, Rust library crates, C FFI library, and `python/secant/` PyTorch module) are documented in this file.

## [0.2.0] - 2026-11-08

### Added
- **Weight-Space Inspection, Analysis, Composition, & Direct `mmap` Patching (`secant profiles`, `inspect`, `read`, `analyze`, `compose`, `patch`, `verify-patch`):**
  - Zero-copy `mmap` inspection and singular-spectrum reading for `.safetensors` and `.gguf` (`v2`/`v3`) files with pre/post SHA-256 verification brackets (`secant.wsp.inspect.v1`, `secant.wsp.read.v1`).
  - Contrastive activation direction capture (`python/secant/capture.py`), layer ranking by precomputed per-layer SNR (`secant analyze`, `secant.wsp.analyze.v1`), and Gram-Schmidt orthogonal multi-direction union with `--target-shift` / `--gamma` step calibration (`secant compose`, `secant.wsp.compose.v1`).
  - Direct in-place rank-1 `mmap` weight surgery (`secant patch`, `secant.wsp.patch.v1`) and byte-exact structural patch verification (`secant verify-patch`, `secant.wsp.verify.v1`).
- **Quantized `Q8_0` & `Q4_K` Block Surgery & Commutation Witness (`secant quant`):**
  - `secant quant patch` (`secant.quant.patch.v1`): sub-tensor in-place `Q8_0` 34-byte block surgery touching only blocks intersecting column support `S`.
  - `secant quant witness` (`secant.quant.witness.v1`): measures `Pi_128` cross-head superblock scale coupling on 256-weight `Q4_K` superblocks versus 32-weight `Q8_0` blocks.
  - `secant quant roundtrip`: deterministic encode/decode error verification for `Q8_0` and `Q4_K` block codecs.
- **Subspace-Locked LoRA Healing (`secant heal`, `secant verify-subspace`, `secant rotation-test`, `python/secant/nn.py`):**
  - `secant.nn.ProjectThenHealLinear` and `secant heal` (`secant.wsp.heal.v1`): twice-applied `float64` orthogonal complement projection `B_locked = P_perp(P_perp(B_tilde))` preventing AdamW diagonal preconditioning from rotating LoRA updates back into edited subspaces (`< 1e-12` leakage in `float64`, `< 1e-6` in merged `F32`).
  - `secant verify-subspace` (`secant.wsp.verify-subspace.v1`): post-healing subspace invariance verifier across edited and unedited tensors.
  - `secant rotation-test`: native 3-arm AdamW vs. gradient-projected vs. `ProjectThenHealLinear` numerical trajectory comparison.
- **Two-Stage Static Spectral Trojan Auditing (`secant audit-trojan`, `secant read --detect-anomalies --marchenko-pastur`):**
  - Stage 1 (Iterated Outlier Singular-Value Peeling & Row/Column Spike Detection): iteratively subtracts outlier singular energies (`||W||_F^2 - sum_{i=1}^k sigma_i^2`, the exact Frobenius energy of the rank-`k` ablated matrix `W - sum_{i=1}^k sigma_i u_i v_i^T`) until the residual Marchenko-Pastur bulk edge `sigma_MP` converges to a fixed point, alongside `zero_rows`, `collapsed_rows`, and `spike_cols` detection.
  - Stage 2 (Cross-Layer Sibling Modified Z-Score & Vocabulary Token Projection): ranks each tensor's `sigma_1 / sigma_MP` ratio against its architectural layer family (`blk.*.ffn_down.weight`, `model.layers.*.mlp.down_proj.weight`) via the Iglewicz-Hoaglin modified Z-score (`sibling_outlier` when `Z > 3.5`), and projects the leading left singular vector `u_1` onto `--vocab-tensor` (`E * u_1`) to surface promoted trigger tokens inside the `g* = sigma_1(W) / ||W||_F` crossover window.
- **Boundary-3 Pre-Load Payload Sanitization & Witness Generation (`secant sanitize`, `secant probe`, `secant scan-sinks`, `secant fetch-tensor`):**
  - `secant sanitize` (`secant.wsp.sanitize.v1`): pre-load value-domain auditing and opt-in remediation (`--clamp-nonfinite`, `--dedup-routing-rows`, `--clamp-extreme`, `--flush-subnormal`, `--dry-run`) across unquantized float tensors (`Rule B3-1`), all 10 standard GGUF quantized block-scale formats (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` under `Rule B3-2A`, permitting zero and FP16 subnormals with `0.0%` false-positive rate across `186,870,528` quantized blocks in `10 / 10` checkpoints), divisor activation/KV scales and `rope_freqs` (`Rule B3-2B`), and integer routing/index tensors (`tid2eid`, `expert_id*`, `g_idx`, `expert_map`, `full_row_indices`, `hash_indices_table` under `Rule B3-3`, failing closed with exit `2` on unbounded index tensors).
  - `secant probe` (`secant.wsp.probe.v1`): deterministic, self-verifying `.safetensors` and `.gguf` witness fixture generator across all four Boundary-3 failure classes (`class1` router/gate `NaN`/`-Inf`, `class2` scale/bias `NaN`/`+-1e38`, `class3` `NaN`/sign/payload/tie arrays, `class4` `I32`/`I64` out-of-range and duplicate routing tables plus zero/subnormal norm vectors).
  - `secant scan-sinks` (`secant.wsp-sink-scan.v1`): static C/C++/CUDA/Rust source scanner for Boundary-3 weight-value sink classes (`B3-SINK-01..06`).
  - `secant fetch-tensor`: HTTP Range (`206 Partial Content`) header and single-tensor fetcher for remote `.safetensors` files (`model.safetensors` / `model.safetensors.index.json`) with SSRF and redirect-host validation.
