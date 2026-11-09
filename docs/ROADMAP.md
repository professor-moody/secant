# Secant Roadmap

This roadmap is organized in three horizons. **Horizon 1** closes every experimental and implementation caveat documented in the v0.2.0 technical paper before any new surface area is added.

## Horizon 1: Close the Gaps Named in the v0.2.0 Paper (Immediate Priority)

These are the experiments and rule checks the paper itself identifies as open caveats. Completing them turns every hedged claim in the v0.2.0 study into a closed, controlled result.

1. **Healed-random and heal-the-base control arms.**
   The `87% -> 93%` (`Qwen2.5-Math-7B` `GSM8K`) and `36% -> 42%` (`Qwen2.5-Coder-7B` `HumanEval`) gains in the 6-model RMS sweep cannot be attributed to the subspace lock alone until the sweep includes a control for the 60-step LoRA fine-tuning pass itself. Run two additional arms on all six RMS-sweep models (`Qwen3-8B`, `Qwen3-14B`, `DeepSeek-R1-Distill-Qwen-7B`, `DeepSeek-R1-Distill-Llama-8B`, `Qwen2.5-Math-7B-Instruct`, `Qwen2.5-Coder-7B-Instruct`) using the exact same 60 AdamW steps and training texts:
   - `base_lora_60steps` (`heal-the-base`: unconstrained LoRA attached to the unedited base checkpoint).
   - `healed_random_rms_1p0x` (`healed-random`: `ProjectThenHealLinear` attached to the norm-matched random-edited checkpoint, locking the random direction).

2. **Multi-seed random controls and fine-grained `0.5x..1.0x` dose sampling.**
   - Run three independent random orthogonal direction seeds (`seed_1`, `seed_2`, `seed_3`) on the 4-arm `Qwen3.5-0.8B` `MATH-500` attribution test and on the `Qwen3.5-4B` crash-triage dose study (plus stochastic decoding replicates at `temperature = 0.6` alongside greedy decoding) to place multi-seed intervals on both headline comparisons.
   - Sample the `0.6x, 0.7x, 0.8x, 0.9x` dose interval on `Qwen3.5-4B` (`n = 30` C++ crash-triage traces) between `0.50x` (`0/30` closed) and `1.00x` (`21/30` closed) so the activation-calibrated closure transition is measured directly rather than bracketed.

3. **Pre-specified Spectral Trojan dose grid with multi-layer placement and behavioral verification.**
   - Re-run `secant audit-trojan` on a preregistered dose grid (`0%, 1%, 2%, 4%, 6%, 7%, 8%, 9%, 10%, 12%, 16%` of `||W||_F`) across three layer depths (early, middle, late) and two writer placements (`mlp.down_proj` and `self_attn.o_proj`) per model.
   - Pair every static spectral measurement (`sigma_1 / sigma_MP`, sibling `Z`, and `E * u_1` top-16 rank) with a forward-pass behavioral probe (`Delta logit(trigger)` and greedy token-flip rate on held-out prompts) so the static detection floor (`g* = sigma_1(W) / ||W||_F`) is directly compared against the dose at which the backdoor becomes behaviorally active.

4. **Real-world upstream C/C++ vulnerability twins in `ASAN-120` (`COMPLETED` — `ASAN-RealWorld-60` / `ASAN-180`).**
   - Constructed and sanitizer-verified **`ASAN-RealWorld-60`** (`data/asan_realworld_60/asan_realworld_60_paired_benchmark.json`, SHA-256 `7aaf5593da3f9f84a5df8d77102966ce223a753f7d4d647a1292df1b5f33a151`; combined `180`-item suite `asan180_combined_benchmark.json`, SHA-256 `526db767cb21a8df4315377c2f7089f79f2d31a103df8bcc71fc5f938630e7cb`): 30 real before/after C/C++ vulnerability functions (`RW01`..`RW30`) paired with their 30 exact upstream patches across 16 open-source repositories (`360` binary runs across `clang++ -O0/-O1/-O2 -fsanitize=address,undefined`, `validation_receipt.json` SHA-256 `f64ea55205fb8c36e42e1b46a2c071a21e3a2c7fc4dacc526fcdb0d6b9c7577a`, `30/30` PASS), and evaluated across `Qwen3.5-0.8B`, `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct`, and `Qwen3.5-4B` (`asan_realworld_60_results.json` SHA-256 `59402d8fb6627ba08c3a399d97e57a5ac75ecf229ccd727019979536cd138ba8`).

5. **Implement the `Rule S-0` companion-config (`config.json` / `generation_config.json`) check in `secant sanitize` (`COMPLETED`).**
   - Implemented in `crates/secant-core/src/security.rs` and `internal/wsp/sanitize.go` (`26/26` `security_parity` tests passing): scanning a `.safetensors` checkpoint automatically inspects sibling `config.json` and `generation_config.json` files (or an explicit `--config` path) to enforce `0 <= eos_token_id, bos_token_id, pad_token_id, decoder_start_token_id < vocab_size` (cross-checked against the embedding/unembedding tensor shape), `1 <= num_experts_per_tok <= num_local_experts`, and `0 <= leading_dense_block <= num_hidden_layers`.

## Horizon 2: Make `secant sanitize` the Standard Pre-Load Gate (Next Quarter)

- **Widen the false-positive corpus** from 10 production checkpoints (`186.8M` blocks) to the top 100 GGUF and SafeTensors downloads on Hugging Face Hub with all 10 quantized GGUF block formats represented.
- **Ship a GitHub Actions / pre-commit CI integration**: one fail-closed command (`secant sanitize --model <path> --receipt <json> --dry-run`) with copy-paste workflows for `llama.cpp`, `vLLM`, and `Ollama` deployments.
- **Publish Rules `S-0` and `B3-1..B3-3` as a standalone loader invariant specification** that inference engines can adopt directly inside native model loaders.
- **Native Python bindings via PyO3** (`secant.sanitize(path)` and `secant.audit_trojan(path)`) so pre-load checks can run inside Hugging Face `from_pretrained` hooks without subprocess overhead.
- **Reference-checkpoint diffing mode (`secant audit-trojan --reference <base_model>`)**: compute the singular spectrum and vocabulary projection of `Delta W = W - W_ref` directly when a trusted base checkpoint is available, closing the sub-threshold (`g < g*`) single-file visibility gap.

## Horizon 3: Follow-On Research Program (6 to 12 Months)

- **Distribution-matched activation calibration**: calibrate edit step sizes across a distribution of held-out benchmark activations rather than a single 32-prompt reference mean, reducing the `0.5%..14.8%` target-shift variance across domains.
- **Multi-writer joint `Project-Then-Heal` across downstream layers**: analyze and constrain how unedited downstream layers partially re-introduce a removed residual direction (`~53%` downstream pull-back when only a subset of writers is locked).
- **Pair-aware `Q4_K` superblock re-encoder**: jointly solve for the 6-bit sub-block scales (`s_j`, `m_j`) and 4-bit nibbles across the two 128-column attention heads spanned by a 256-weight `Q4_K` superblock, removing the `Q8_0`-only restriction on in-place GGUF surgery.
- **Packaged one-command external replication bundles**: ship pinned, single-command reproduction scripts and container receipts for the 4-arm `Qwen3.5-0.8B` `MATH-500` study and the `Qwen3.5-4B` calibration study for independent third-party verification.
