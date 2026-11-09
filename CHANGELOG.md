# Changelog

All notable changes to **Secant** are documented in this file.

## [Unreleased]

### Added
- **`--reference <base_checkpoint>` on `secant read` and `secant audit-trojan` (Horizon 2):** subtracts the same-named tensor of a trusted base checkpoint (identical logical shape; float or `Q8_0` / `Q4_K`; GGUF and SafeTensors may be mixed) before any analysis, one separately rounded `float64` subtraction per element, never materialised. Frobenius norm, column statistics, anomalies, singular spectrum, Marchenko-Pastur bulk and the `E * u_1` vocabulary projection then describe `Delta W = W - W_ref`, which closes the single-file gap where a small rank-1 edit hides under a trained matrix's own dominant direction. Each tensor carries `reference_diff` (`compared` with `delta_frobenius`, `reference_frobenius`, `relative_delta`, `sigma1_energy_fraction`, `identical`; or `skipped: <reason>` for an absent tensor or a different shape, never a number against a mismatch); the report carries a bracket-hashed `reference` block. A reference that is the model file itself, or shares no comparable tensor, exits `2`. `audit-trojan` adds a `Delta W` table because an exactly low-rank difference has no Marchenko-Pastur bulk and would otherwise drop out of the ranking. Receipts without `--reference` are byte-for-byte unchanged. The Go oracle `crucible wsp read --reference` emits the same fields and `diff_oracle` shows the receipts bit-identical.

## [0.3.0] - 2026-11-09

### Added
- **Rule S-0 Companion-Config Sanitizer (`crates/secant-core/src/security.rs`):** When scanning a `.safetensors` file, `secant sanitize` now automatically discovers and validates sibling `config.json` and `generation_config.json` files in the checkpoint directory (including nested `text_config` / `llm_config` sub-objects):
  - `s0_companion_vocab_size_mismatch`: rejects `vocab_size > embed_rows` against the checkpoint's embedding or output projection tensor (`embed_tokens.weight`, `wte.weight`, `lm_head.weight`, `word_embeddings.weight`, `shared.weight`, `output.weight`), or `vocab_size <= 0`.
  - `s0_companion_token_id_oob`: rejects scalar or array `eos_token_id`, `bos_token_id`, `pad_token_id`, and `decoder_start_token_id` values outside `[0, vocab_size)` (allowing `-1` only for `pad_token_id` and `decoder_start_token_id`), or control token IDs specified without a resolvable `vocab_size`.
  - `s0_companion_expert_count` and `s0_companion_expert_used_count`: enforces `expert_count > 0` and `1 <= expert_used_count <= expert_count` across `num_local_experts` / `num_experts` / `n_routed_experts` and `num_experts_per_tok` / `num_selected_experts` / `moe_top_k`.
  - `s0_companion_leading_dense_block`: enforces `0 <= leading_dense_block <= num_hidden_layers` across `leading_dense_block` / `first_k_dense_replace` and `num_hidden_layers` / `n_layer` / `num_layers` to prevent unsigned subtraction wraparound.
  - Malformed or unreadable sibling JSON files fail closed with `status: "unknown"` (exit code `2`).
- **`ASAN-RealWorld-60` / `ASAN-180` Real-World Upstream C/C++ Vulnerability Twin Benchmark (`data/asan_realworld_60/`):** Added 30 real-world sanitizer-reproduced C/C++ bugs paired with their exact 30 upstream patched twins (`60` items total, combined with `ASAN-120` to form `ASAN-180`) across 16 open-source repositories (`llama.cpp`, `tensorflow`, `openxla/xla`, `flatbuffers`, `protobuf`, `c-ares`, `harfbuzz`, `lcms`, `openthread`, `sentencepiece`, `stb`, `gptqmodel`, `exllamav3`, `hnswlib`, `voyager`, and `yajl`), with `1,080` receipts across 4 local models (`0.8B` to `4B`, `109/109` SHA-256 verified files).

## [0.2.0] - 2026-11-08

### Added
- Initial public release of the `secant` Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python package (`secant.nn.ProjectThenHealLinear`, `secant.capture`).
- Sub-tensor `.safetensors` and 34-byte `Q8_0` `.gguf` in-place `mmap` surgery (`secant patch`, `secant quant patch`, `secant verify-patch`).
- Reference-based target-shift and RMS-relative calibration with $R^{-1}$ Gram-Schmidt cross-projection compensation (`secant analyze`, `secant compose`).
- Twice-iterated `float64` forward-pass subspace locking (`ProjectThenHealLinear`, `secant heal`, `secant verify-subspace`).
- Pre-load Boundary-3 payload sanitization across 10 GGUF block formats (`secant sanitize`), synthetic boundary fixture generation (`secant probe`), spectral Trojan auditing (`secant audit-trojan`), and C/C++/CUDA/Rust consumer sink scanning (`secant scan-sinks`).
