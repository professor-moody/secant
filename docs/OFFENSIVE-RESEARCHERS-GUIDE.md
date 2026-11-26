# Testing Model Scanners with Secant: What Weight-Level Payloads Actually Do

Most model scanners stop at the container format. They block Python `pickle` files, check that SafeTensors byte offsets stay inside the file, and mark the checkpoint safe.

For an offensive researcher or a security engineer evaluating a model scanner, that leaves the entire numerical surface untouched. A checkpoint that is 100% valid SafeTensors or GGUF on disk can still crash an inference worker, corrupt GPU or CPU gather/scatter kernels, poison every output token on startup, or hide a targeted behavioral implant inside finite floating-point weights.

This guide walks through what each payload class actually does in practice, why defenders need to catch them before loading untrusted weights, and how to generate synthetic test cases in seconds to benchmark any model scanner. For copy-paste CLI commands, jump straight to [Practical Recipes](RECIPES.md).

---

## 1. Real-World Impact: Four Things Inside "Safe" Tensor Files

### A. Out-of-bounds integer routing and permutation tables
Quantized checkpoints (GPTQ, AWQ, EXL3) and Mixture-of-Experts (MoE) models ship integer index tensors (`g_idx`, `expert_map`, `router_indices`, `topk_ids`) right alongside floating-point weights. Many C++ and CUDA kernels trust those integers directly as lookup subscripts without per-element bounds checks on the hot path.
- **What happens:** An out-of-range index (`-1` or `>= num_experts`) reads or writes past GPU or host buffers during model load or the first forward pass (for example, `CVE-2026-86288` and `vllm#48470`), crashing multi-tenant serving workers or corrupting adjacent memory.

### B. Zero, subnormal, or `NaN` quantization scale headers
Quantized formats store small blocks of weights paired with per-block scale factors across 24 GGUF block formats (standard `Q4_0`..`Q8_1`, K-quants `Q2_K`..`Q8_K` including 32-bit float `Q8_K`, all 9 I-quants `IQ2_XXS`..`IQ1_M` including `IQ1_M`'s packed FP16 `d`, ternary `TQ1_0`/`TQ2_0`, and OCP E8M0 `MXFP4`) or activation divisor scales (`input_scale`, `k_scale`, `v_scale`, `*scale_inv` in FP8/INT8 checkpoints).
- **What happens:**
  1. **Silent output poisoning:** A single `NaN` or `Inf` in a weight tensor or GGUF block scale propagates across attention and layer norms, turning a model's responses into garbage or crashing Top-K/Top-P samplers that assume finite logits.
  2. **Cross-architecture cast traps:** Dividing by `0.0` or a tiny subnormal (`1e-45`) produces `+Inf` or `NaN`. When native C++ code casts that float to `int32_t`, `ARM64` (`fcvtzs`) clamps `NaN` to `0` and `+Inf` to `INT_MAX`, whereas `x86-64` (`cvttss2si`) returns `INT_MIN` (`-2147483648`). Code that only checks `if (idx >= N)` passes on `x86-64` and indexes negatively before the array!

### C. Out-of-bounds companion token IDs
Model repositories bundle `config.json`, `generation_config.json`, and `tokenizer.json` next to `.safetensors` shards.
- **What happens:** Setting `eos_token_id`, `bos_token_id`, or `pad_token_id` above the embedding table's row count (`vocab_size`) causes C++ runtimes that trust companion JSON metadata to index past the token embedding matrix.

### D. Rank-one behavioral weight implants (spectral trojans)
Every value in the file is finite, well-scaled, and in bounds, yet one `down_proj` or `o_proj` matrix carries a targeted rank-one edit ($\Delta W = \alpha \, u \, v^\top$).
- **What happens:** On normal prompts and standard benchmarks, the model behaves normally (`0 / 12` false triggers on clean prompts across `186` of `187` checkpoint-dose runs in our 17-placement study). When a prompt contains the attacker's trigger phrase, the edited columns fire together and steer generation toward a chosen token, refusal bypass, or tool-call decision.

---

## 2. Benchmarking Any Scanner in 60 Seconds

You do not need a GPU or a 15 GB checkpoint to see what your model scanner catches and misses. Secant ships deterministic synthetic generators that build tiny, valid `.safetensors` and `.gguf` test fixtures on disk:

```sh
# 1. Run the 9-case hands-on test drive (clean controls + poisoned cases + repair)
python3 examples/test-drive/test_drive.py

# 2. Generate the 24-format GGUF quantized-scale probe (`class5`, 59 planted findings)
c5_dir=$(mktemp -d)
secant probe --class class5 --format gguf \
  --out-dir "$c5_dir/fixtures" --receipt "$c5_dir/probe.json" --attempt-id scanner-c5

# 3. Run the 83-case 24-format differential suite (add --llama-cli to compare against llama.cpp --check-tensors)
python3 scripts/bench_gguf_quant_coverage.py \
  --run-dir "$c5_dir/bench-24fmt" --binary target/release/secant

# 4. Generate the broader synthetic benign-FPR + positive-control corpus (F32, F16, BF16, Q8_0, Q4_K)
cargo build --release -p secant-core --example benign_fpr_fixtures
fixture_dir=$(mktemp -d)
target/release/examples/benign_fpr_fixtures "$fixture_dir/suite" scanner-eval
```

Point both `secant` and any third-party scanner at `$c5_dir/fixtures`, `$c5_dir/bench-24fmt/fixtures`, and `$fixture_dir/suite`, and score two numbers separately:

1. **True-positive recall by bug class:** Does the scanner catch `NaN`/`Inf` floats, non-finite scale headers across all 24 GGUF formats (including `FP32` `Q8_K`, packed `FP16` `IQ1_M`, and E8M0 `0xFF` `MXFP4`), zero/subnormal divisor scales, out-of-range MoE routing indices, and rank-one spectral edits?
2. **False-positive rate on clean controls:** Does the scanner stay quiet on legitimate zero/subnormal *multiplicative* quantized blocks and clean transformer matrices?
   - For example, naive single-file spectral thresholds flag almost every clean transformer checkpoint because pretrained layers naturally contain strong singular spikes. By contrast, `secant sanitize`'s numerical and structural rules produced `0 / 105` false positives across our v0.3.0 cross-platform benchmark ([`benchmarks/benign-fpr-v0.3.0.md`](benchmarks/benign-fpr-v0.3.0.md)) and `24 / 24` clean controls (`0.0%` synthetic FPR) with `30 / 30` NaN and `29 / 29` Inf recall across all 24 GGUF formats against `llama.cpp --check-tensors` ([`payload-security.md`](payload-security.md#quantized-scale-headers-24-gguf-standard-k-quant-iq-ternary-and-mxfp4-formats)), while `secant audit-trojan --reference <base>` isolates subtle `1%`-dose implants by subtracting the clean base model first.

---

## 3. Quick Rule Reference (Collapsible Appendix)

<details>
<summary><strong>Bitwise & Numerical Rules Enforced by <code>secant sanitize</code> (Rules S-0 and B3-1..B3-3)</strong></summary>

<br/>

| Rule ID | Target Scope | Bitwise / Numerical Condition | Bug Class Prevented |
| :--- | :--- | :--- | :--- |
| **`Rule S-0`** | `.safetensors` & `.gguf` headers + sibling `config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json` | 64-bit overflow-checked extents; zero overlaps; `vocab_size <= embed_rows`; `0 <= eos_token_id, bos_token_id, pad_token_id, added_tokens_decoder[*] < vocab_size`; `1 <= expert_used_count <= expert_count`. Malformed JSON fails closed (`exit 2`). | Header offset wraps and out-of-bounds token/expert indexing |
| **`Rule B3-1`** | Unquantized `F64`, `F32`, `BF16`, `F16`, `F8_E4M3`, `F8_E5M2` weights, router biases, and norm vectors | Reject `NaN` and `+-Inf` via exponent masks (`F32`: `(u32 & 0x7F800000) != 0x7F800000`; `BF16`: `(u16 & 0x7F80) != 0x7F80`; `F16`: `(u16 & 0x7C00) != 0x7C00`; `F8_E4M3`: `(u8 & 0x7F) != 0x7F`; `F8_E5M2`: `(u8 & 0x7C) != 0x7C`). Selected norm vectors must satisfy $\|v\|_2 \ge 10^{-12}$. | Router `best = -1` `NaN` underflow, float-to-int cast UB, sampler `NaN` collapse |
| **`Rule B3-2A`** | Multiplicative GGUF block scale headers across all 24 reviewed formats (`Q4_0`..`Q8_1`, `Q2_K`..`Q8_K`, `IQ1_S`..`IQ4_XS`, `IQ1_M`, `TQ1_0`, `TQ2_0`, `MXFP4`; fields `d`, `dmin`, `m`, `s`, or `e`) | Permit `0.0` and finite subnormal multiplier scales; reject only non-finite encodings: FP16 and packed-FP16 `IQ1_M` `(u16 & 0x7C00) == 0x7C00`, FP32 `Q8_K` `(u32 & 0x7F800000) == 0x7F800000`, and `MXFP4` E8M0 `e == 0xFF` (`quant_scale_nan` and `quant_scale_inf`). Unreviewed quantized types (`NVFP4`, `Q1_0`, `Q2_0`) fail closed (`exit 2`). | Non-finite GGUF scale headers across standard, K-quant, IQ, ternary, and MXFP4 checkpoints with zero false positives on sparse quantized blocks |
| **`Rule B3-2B`** | Divisor activation & KV-cache scales (`input_scale`, `k_scale`, `v_scale`, `q_scale`, `y_scale`, `prob_scale`, `*scale_inv`) and `rope_freqs` | Require finiteness and $s \ge 10^{-12}$ (`divisor_scale_nonpositive`, `divisor_scale_below_floor`), and non-zero `rope_freqs` (`rope_freqs_zero`). | Division-by-zero and subnormal divisor overflow (`x / 1e-45 -> +Inf -> INT32_MIN` on x86-64) |
| **`Rule B3-3`** | Integer routing & permutation tensors (`g_idx`, `expert_map`, `full_row_indices`, `hash_indices_table`, `expert_ids`, `topk_ids`, `router_indices`) | Enforce two-sided bounds $0 \le \text{idx} < \text{bound}$ (or $-1 \le \text{idx} < \text{bound}$ for `expert_map`) and uniqueness within routing rows. Unbounded index tensors fail closed (`exit 2`). | GPU/CPU gather-scatter heap OOB reads and writes (`CVE-2026-86288`, `vllm#48470`) |

</details>

<details>
<summary><strong>Spectral Crossover ($g^*$) and Reference Diffing Math</strong></summary>

<br/>

1. **Iterative Bulk Peeling (`secant audit-trojan`)**: Stage 1 estimates the Marchenko-Pastur upper noise edge $\sigma_{\text{MP}} = s \sqrt{d_{\text{out}}}(1 + \sqrt{d_{\text{in}} / d_{\text{out}}})$ after peeling outlier singular values. Stage 2 compares each layer's $\sigma_1 / \sigma_{\text{MP}}$ ratio against same-role sibling layers via a modified $Z$-score ($Z > 3.5$) and projects the top singular vector $u_1$ onto the token vocabulary ($E u_1$).
2. **Single-File Crossover Dose ($g^* = \sigma_1(W) / \|W\|_F$)**: A rank-1 implant of relative Frobenius size $g = \|\Delta W\|_F / \|W\|_F$ overtakes the base layer's top singular direction near $g^*$. Across 17 placements (`187` checkpoint-dose runs in [`payload-security.md`](payload-security.md)), the vocabulary view surfaces the target token in the top 16 within `0.87` median percentage points of $g^*$.
3. **Sub-$g^*$ Detection with `--reference`**: In late layers (`L19/L20`), an implant of `4%` to `6%` can hijack outputs below the single-file crossover $g^*$. Passing `--reference <base_checkpoint>` runs SVD directly on $\Delta W = W - W_{\text{ref}}$, ranking the target token `#1` across `9 / 9` below-crossover late-layer cases down to a `1%` dose.

</details>

---

## Further Reading

- **[60-Second Hands-On Test Drive](../examples/test-drive/README.md)**: Synthesize clean and poisoned mini-checkpoints, scan them, repair non-finite blocks, and inspect a pass/fail table.
- **[Practical CLI & Python Recipes](RECIPES.md)**: Step-by-step commands for scanning downloads, repairing copies, and verifying quantized exports.
- **[PyTorch 10-Line Healing Guide](PYTORCH-10-LINE-HEALING-GUIDE.md)**: Fine-tune with `ProjectThenHealLinear` so protected weight directions do not drift.
- **[Payload Security Technical Reference](payload-security.md)**: Complete empirical tables, 187-run dose curves, and ARM64 vs x86-64 conformance results.
