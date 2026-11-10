# Changelog

All notable changes to **Secant** are documented in this file.

## [Unreleased]

## [0.3.0] - 2026-11-10

### Added
- **SafeTensors FP8 (`F8_E4M3`, `F8_E5M2`) & Benign Auxiliary Buffer Support (`crates/secant-core/src/{dtype,container,read,security}.rs`):** Added read-only inspection, spectral reading (`secant read`, `secant audit-trojan`), and pre-load sanitization (`secant sanitize`) for IEEE-754 `F8_E5M2` and OFP8 `F8_E4M3` (`0x7F`/`0xFF` NaN) SafeTensors payloads. Benign non-index auxiliary buffers (`BOOL`, `U8` causal masks / `*.bias`, and scalar/1-element `I64`/`I32` `*.num_batches_tracked` BatchNorm step counters) are skipped cleanly under `ignored_integer_tensors`, while integer routing/index tensors remain bounded and unimplemented float/quantized formats fail closed (exit `2`).
- **Rule S-0 Companion-Config Sanitizer (`crates/secant-core/src/security.rs`):** When scanning a `.safetensors` file, `secant sanitize` automatically discovers and validates sibling `config.json`, `generation_config.json`, `tokenizer_config.json`, and `tokenizer.json` files in the checkpoint directory (including nested `text_config` / `llm_config` sub-objects):
  - `s0_companion_vocab_size_mismatch`: rejects `vocab_size > embed_rows` against the checkpoint's embedding or output projection tensor (`embed_tokens.weight`, `wte.weight`, `lm_head.weight`, `word_embeddings.weight`, `shared.weight`, `output.weight`), or `vocab_size <= 0` when text embedding tensors are present (`vocab_size == 0` with zero non-empty text embedding tensors is accepted as a disabled-modality sentinel on vision-only encoders).
  - `s0_companion_token_id_oob` and `s0_companion_added_token_id_oob`: rejects scalar or array `eos_token_id`, `bos_token_id`, `pad_token_id`, `decoder_start_token_id`, and `added_tokens_decoder` IDs outside `[0, vocab_size)` (allowing `-1` only for `pad_token_id` and `decoder_start_token_id`), or non-null token IDs specified when `vocab_size` is missing or `0`.
  - `s0_companion_expert_count` and `s0_companion_expert_used_count`: enforces `expert_count > 0` and `1 <= expert_used_count <= expert_count` across `num_local_experts` / `num_experts` / `n_routed_experts` and `num_experts_per_tok` / `num_selected_experts` / `moe_top_k` (`num_local_experts == 0` with `num_experts_per_tok == 0` and zero MoE tensors is accepted as a dense-model sentinel).
  - `s0_companion_leading_dense_block`: enforces `0 <= leading_dense_block <= num_hidden_layers` across `leading_dense_block` / `first_k_dense_replace` and `num_hidden_layers` / `n_layer` / `num_layers` to prevent unsigned subtraction wraparound.
  - Malformed, non-object, or unreadable sibling JSON files fail closed with `status: "unknown"` (exit code `2`).
- **`--rank-k <K>` (`K` in `1, 2, 4, 8`) on `secant audit-trojan`, and native PEFT LoRA adapter scanning:** Each 2-D tensor now reports cumulative top-$k$ singular energy $E_k = \sum_{i \le k} \sigma_i^2 / \lVert M \rVert_F^2$ in single-file mode ($M = W$) and under `--reference` ($M = W - W_{\text{ref}}$; $E_1$ is bit-identical to `sigma1_energy_fraction`), plus sign- and basis-invariant rank-$k$ vocabulary coherence $c_t = \lVert Q^\top e_t \rVert^2 / \lVert e_t \rVert^2$ under `--vocab-tensor`. `adapter_model.safetensors` files carrying `lora_A` / `lora_B` tensors are audited per module via thin QR and one-sided Jacobi SVD on the $r \times r$ factor without materializing $d_{\text{out}} \times d_{\text{in}}$ (`--vocab-model` names the checkpoint holding `--vocab-tensor`).
- **`--reference <base_checkpoint>` on `secant read` and `secant audit-trojan` (Horizon 2):** Streams $\Delta W = W - W_{\text{ref}}$ row-by-row in `float64` across float and `Q8_0`/`Q4_K` GGUF or SafeTensors checkpoints with pre/post SHA-256 brackets on both files, closing sub-$g^*$ single-file visibility gaps.
- **Repository Governance & Exit-Code Contract:** Added `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and the documented `0 / 1 / 2` fail-closed CLI exit-code contract in `README.md`.
- **`ASAN-RealWorld-60` / `ASAN-180` Real-World Upstream C/C++ Vulnerability Twin Benchmark (`data/asan_realworld_60/`):** Added 30 real-world sanitizer-reproduced C/C++ bugs paired with their exact 30 upstream patched twins (`60` items total, combined with `ASAN-120` to form `ASAN-180`) across 16 open-source repositories (`llama.cpp`, `tensorflow`, `openxla/xla`, `flatbuffers`, `protobuf`, `c-ares`, `harfbuzz`, `lcms`, `openthread`, `sentencepiece`, `stb`, `gptqmodel`, `exllamav3`, `hnswlib`, `voyager`, and `yajl`), with `1,080` SHA-256-verified receipts across 4 local models (`0.8B` to `4B`).

## [0.2.0] - 2026-11-08

### Added
- Initial public release of the `secant` Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python package (`secant.nn.ProjectThenHealLinear`, `secant.capture`).
- Sub-tensor `.safetensors` and 34-byte `Q8_0` `.gguf` in-place `mmap` surgery (`secant patch`, `secant quant patch`, `secant verify-patch`).
- Reference-based target-shift and RMS-relative calibration with $R^{-1}$ Gram-Schmidt cross-projection compensation (`secant analyze`, `secant compose`).
- Twice-iterated `float64` forward-pass subspace locking (`ProjectThenHealLinear`, `secant heal`, `secant verify-subspace`).
- Pre-load Boundary-3 payload sanitization across 10 GGUF block formats (`secant sanitize`), synthetic boundary fixture generation (`secant probe`), spectral Trojan auditing (`secant audit-trojan`), and C/C++/CUDA/Rust consumer sink scanning (`secant scan-sinks`).
