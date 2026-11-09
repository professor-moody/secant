# Check Model Files and Weight Values

Because model weights (`SafeTensors` and `GGUF`) are frequently exchanged across untrusted hubs and fine-tuning pipelines, **Secant treats every container file as an untrusted binary input**. Before any weight-space read or edit, Secant validates payload extents in checked 64-bit arithmetic. Three commands then audit weight **values**: `secant sanitize` (static value scan and optional repaired copy), `secant probe` (boundary-value test fixtures), and `secant read --detect-anomalies` (row and column norm anomalies).

The **container** is the file structure that names tensors and locates their bytes. The **payload** is the stored tensor data. A valid container can still hold values that are unsafe for a downstream C, C++, or CUDA consumer, so Secant treats the file as three separate trust boundaries:

1. **Boundary 1, container framing:** header length, schema, and monotonically non-overlapping payload intervals.
2. **Boundary 2, architecture metadata:** tensor dimensions, head ratios, and shape-to-byte invariants checked against `config.json`.
3. **Boundary 3, numerical domains:** non-finite or extreme floats, quantized scale fields, divisor activation and KV-cache scales, integer routing or permutation tensors, and spectral outliers inside an otherwise valid container.

### What `SafeTensors::deserialize()` proves

In `huggingface/safetensors`, `SafeTensors::deserialize(buffer)` calls `read_metadata()` and then `Metadata::validate()`. That validation proves five structural facts:

1. Each tensor interval starts where the previous one ended, and its end is not before its start.
2. Multiplying the non-zero shape dimensions by the dtype bit width does not overflow `usize`.
3. The resulting bit count is byte-aligned.
4. The computed byte count equals the tensor's declared offset span.
5. The final payload end, including the 8-byte length and JSON header, equals the input buffer length.

After those checks, deserialization returns a borrowed payload slice without reading any payload value. This is the Boundary-3 gap: a memory-safe parser can prove that tensor bytes are present without proving that those bytes are valid inputs to later arithmetic or indexing.

---

## 1. Check That Tensor Bytes Fit Inside the File

Before reading a tensor, Secant checks that its declared size and offset describe bytes inside the file. An invariant is a condition that must hold for the file to be accepted. These checks prevent the parser from trusting impossible sizes or locations.

1. **Checked payload extents (every command):** When a file is opened, each tensor's element count, byte size and payload offset are computed with overflow-checked arithmetic, and every payload interval $[\text{offset}_i, \text{offset}_i + \text{size}_i)$ must lie inside the file. GGUF payload offsets are taken relative to `data_offset`, the header end rounded up to `general.alignment` (default `32`). Any overflow or out-of-file extent is an error (exit code `2`).
2. **Overlap and hole check (`secant sanitize`):** Before scanning values, `sanitize` sorts the tensor intervals and rejects any overlap. For `SafeTensors` it additionally rejects gaps between consecutive payloads and unindexed trailing payload bytes.

---

## 2. Scan Values and Optionally Write a Repaired Copy

The scanner looks for the configured findings without running model inference. Repair is explicit: a scan alone does not change the input. Some findings can be replaced under the chosen policy; others cause refusal.

### Sanitization Rules (`0` observed false positives on the 10-checkpoint corpus)

The baseline separates values by how inference uses them. A small multiplicative scale is not equivalent to a small divisor, and neither is equivalent to an integer array index.

| Rule | Scope | Required invariant | Reason |
|---|---|---|---|
| **S-0** | .safetensors and .gguf headers plus sibling `config.json` / `generation_config.json` (companion-config checks since `v0.3.0`) | Checked 64-bit spans, shapes match the architecture, scalar loads have numel == 1, GGUF MoE metadata satisfies `expert_count > 0` (`s0_expert_count`) and `1 <= expert_used_count <= expert_count` (`s0_expert_used_count`), and (since `v0.3.0`) SafeTensors sibling `config.json` / `generation_config.json` satisfy `vocab_size <= embed_rows` (`s0_companion_vocab_size_mismatch`), `0 <= eos_token_id, bos_token_id, pad_token_id, decoder_start_token_id < vocab_size` (`s0_companion_token_id_oob`), `1 <= num_experts_per_tok <= num_local_experts` (`s0_companion_expert_count`, `s0_companion_expert_used_count`), and `0 <= leading_dense_block <= num_hidden_layers` (`s0_companion_leading_dense_block`) | Reject malformed geometry and out-of-bounds companion config parameters before any payload value is used. |
| **B3-1** | Unquantized F32, BF16, F16, and FP8 weights, norms, and router biases | Reject NaN and infinity with a bitwise exponent check | Non-finite values break comparisons, casts, sorting, and cumulative sums. |
| **B3-2A** | Multiplicative GGUF block scales (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` `d`, `dmin`, `m`) and W4A16/W8A16 weight scales | Permit zero and FP16 subnormals; reject only `(bits & 0x7C00) == 0x7C00` | These fields multiply integer quants. Zero and tiny finite multipliers are valid. |
| **B3-2B** | Divisor activation and KV-cache scales (`input_scale`, `k_scale`, `v_scale`, `q_scale`, `y_scale`, `prob_scale`, and names ending in `scale_inv`) plus `rope_freqs` | Require finiteness and `s >= --divisor-floor` (default `1e-12f`; `<= 0` emits `divisor_scale_nonpositive`, `0 < s < floor` emits `divisor_scale_below_floor`), and non-zero `rope_freqs` (`rope_freqs_zero`) | Zero or subnormal divisors can produce non-finite or out-of-range results. |
| **B3-3** | Routing tensors (`expert_ids`, `selected_experts`, `topk_ids`, `router_indices`, `moe_Permute_Map`, plus `hash_indices_table` when `--num-experts > 0`) and integer index/permutation tensors (`g_idx`, `expert_map`, `full_row_indices`, `hash_indices_table`) | Enforce two-sided bounds `[0, bound)` before indexing; fail closed (`UNKNOWN`, exit `2`) when an index tensor has no inferable upper bound | Negative and oversized integers are direct out-of-bounds indices. |

For B3-3, routing tensors require `--num-experts > 0` (reporting `routing_negative`, `routing_oob`, and `routing_duplicate`), whereas integer index tensors (`g_idx`, `expert_map`, `full_row_indices`, `hash_indices_table`, plus `--index-tensor` globs) infer per-tensor bounds from sibling shapes (`g_idx` bounded by GPTQ sibling `scales.shape[0]` with 1-D length checked against `qweight`/`weight` via `g_idx_length_mismatch`; `full_row_indices` bounded by sibling weight/bias/scales extent; `expert_map` and `hash_indices_table` bounded by `--num-experts` when supplied) or an explicit `--index-bound` override, reporting `index_negative` (`v < 0`, or `v < -1` for `expert_map` where `-1` marks an unmapped expert) and `index_oob` (`v >= bound`). When an index tensor has no sibling-inferred or explicit bound, `secant sanitize` increments `index_unbounded` and fails closed with `status: "unknown"` (exit `2`) when no other findings are present. An unsigned comparison such as `static_cast<uint32_t>(idx) < static_cast<uint32_t>(rows)` rejects negative values and upper-bound violations in one check when no negative sentinel is permitted.

`secant sanitize --model <file> --receipt <new.json> --attempt-id <id> [--dry-run]` scans supported static payload values and writes a `secant.wsp.sanitize.v1` receipt. All ten standard GGUF block-quantized formats (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`) are scanned at the `f16` block-header level (`d`, `dmin`, `m`). A GGUF containing an unreviewed quantized type (`Q8_K`, `IQ*`, `TQ*`) is refused with exit `2` (`status: "unknown"`), even if other tensors are unquantized (`crates/secant-core/src/security.rs`). Passing `--dry-run` explicitly enforces read-only verification and rejects any combination with `--out` or remediation flags.

| Finding | Criterion | Repair available |
|---|---|---|
| NaN | any NaN element in B3-1 scope | --clamp-nonfinite: NaN becomes +0 (or `--divisor-floor` on B3-2B divisor scales) |
| Infinity | any ±∞ element in B3-1 scope | --clamp-nonfinite: becomes the signed largest representable value not above --max-abs (or `--divisor-floor` on B3-2B divisor scales) |
| Extreme finite | abs(x) &gt; --max-abs (an optional diagnostic policy, not part of B3-1 or an overflow proof) | opt-in --clamp-extreme (Secant extension) |
| Subnormal | subnormal unquantized float element (an optional diagnostic policy, not a B3-2A violation) | opt-in --flush-subnormal (Secant extension) |
| Zero or tiny-norm vector | L2 norm of a selected contiguous vector (default norm-tensor globs, plus --norm-tensor) below --min-norm (default 10⁻¹²) | none: always refused |
| Divisor scale non-positive or below floor | finite `input_scale`, `k_scale`, `v_scale`, `q_scale`, `y_scale`, `prob_scale`, or `*scale_inv` `<= 0` (`divisor_scale_nonpositive`) or `< --divisor-floor` (`divisor_scale_below_floor`), or `rope_freqs == 0` (`rope_freqs_zero`) | none under CLI `--clamp-nonfinite`: finite below-floor/non-positive divisor scales and zero `rope_freqs` remain findings (exit `1`) so calibration is never silently rescaled |
| Routing or index id out of range | B3-3 routing (`routing_negative`, `routing_oob`) and index tensors (`index_negative`, `index_oob`, `g_idx_length_mismatch` on `g_idx`, `expert_map`, `full_row_indices`, `hash_indices_table`) | none: always refused (unbounded index tensors fail closed with `UNKNOWN`, exit `2`) |
| Duplicate routing id | repeated id within a routing row | --dedup-routing-rows: later duplicates become the lowest unused valid expert id |

`--out <new-file>` publishes a repaired copy only when a remediation flag is given **and** a rescan of the copy is clean; outputs and receipts never overwrite existing files.

Exit codes: `0` clean or remediated under this policy, `1` findings remain, `2` usage, I/O, unsupported input, or incomplete scan. A clean receipt does not prove runtime inference safety.

### Quantized scale headers (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`)

Every supported GGUF quantized block stores one or two IEEE-754 `f16` scale/minimum header fields (`d`, `dmin`, or `m`) at fixed byte offsets within the block (`ggml-common.h`):

| GGUF Format | `ggml_type` ID | Block Elements (`qk`) | Block Bytes | Inspected `f16` Header Fields & Byte Offsets |
| --- | :---: | :---: | :---: | --- |
| **`Q4_0`** | `2` | `32` | `18` | `d` at offset `0` |
| **`Q4_1`** | `3` | `32` | `20` | `d` at offset `0`, `m` at offset `2` |
| **`Q5_0`** | `6` | `32` | `22` | `d` at offset `0` |
| **`Q5_1`** | `7` | `32` | `24` | `d` at offset `0`, `m` at offset `2` |
| **`Q8_0`** | `8` | `32` | `34` | `d` at offset `0` |
| **`Q2_K`** | `10` | `256` | `84` | `d` at offset `80`, `dmin` at offset `82` |
| **`Q3_K`** | `11` | `256` | `110` | `d` at offset `108` |
| **`Q4_K`** | `12` | `256` | `144` | `d` at offset `0`, `dmin` at offset `2` |
| **`Q5_K`** | `13` | `256` | `176` | `d` at offset `0`, `dmin` at offset `2` |
| **`Q6_K`** | `14` | `256` | `210` | `d` at offset `208` |

The fields `d`, `dmin`, and `m` are **multipliers and additive offsets**, not divisors.

That distinction explains the B3-2A allowlist. A zero-padded block can legitimately store `d = 0.0` (`0x0000`). In `Q4_K`, each 256-weight super-block contains eight 32-weight sub-blocks with 6-bit scales in `[0, 63]`; dividing a small sub-block maximum by `63` (where `max |x_i|` is scaled by `15 * 63 = 945`) naturally produces an FP16 subnormal whenever a sub-block's maximum magnitude sits below `945 * 6.1035e-05 ≈ 0.0577` (or a sub-block scale sits near `2.8e-03`). Examples include dense `blk.*.ffn_down.weight` and `token_embd.weight` in `Qwen3.5-4B` `Q4_K`, as well as `d = 4.55e-05` in row 0 of `blk.0.ffn_down_exps` in `llama.cpp`'s MoE test fixture (`test_metal_moe_experts.c`), below the smallest positive normal FP16 value (`6.1035e-05`). Multiplication by either value remains well-defined and cannot create a division-by-zero cast path.

The zero-false-positive check is therefore exactly the FP16 exponent test. Values with `(bits & 0x7C00) != 0x7C00`, including signed zero and subnormals, pass. Values with `(bits & 0x7C00) == 0x7C00` are NaN or infinity, are flagged, and are replaced with zero only when an explicit repaired copy is requested. The scan strides by the exact `ggml_type_size(type)` so it reads `d`, `dmin`, and `m` at their exact block offsets across mixed-quantization checkpoints (such as `Q4_K_M`, which mixes `Q4_K` and `Q6_K` tensors, or `Q2_K` checkpoints that mix `Q2_K`, `Q3_K`, and `Q6_K`).

| Finding | Criterion | Repair available |
|---|---|---|
| quant\_scale\_nan | (bits &amp; 0x7C00) == 0x7C00 and mantissa is non-zero | --clamp-nonfinite: zero the complete affected block |
| quant\_scale\_inf | (bits &amp; 0x7C00) == 0x7C00 and mantissa is zero | --clamp-nonfinite: zero the complete affected block |

For these findings `index` is the block index, `row` is the GGUF row (`ne[0]` elements), `file_offset` points at the FP16 field, `bits` is its encoding, and `field` identifies `d`, `dmin`, or `m`. A quantized tensor matched by a norm glob is `unknown` (exit `2`), because norm-row checks are implemented for unquantized floats only. A GGUF file containing an unreviewed quantized type (`Q8_K`, `IQ*`) is refused rather than partially certified.

(sec-quant-sanitize-controls)=
#### Measured on 10 real SafeTensors and mixed GGUF checkpoints, with planted controls

Across our 10-checkpoint production GGUF and SafeTensors verification corpus (`Qwen3.5-4B` `Q8_0` and `Q4_K`, `Qwen2.5-7B-Instruct-Q4_K_M` shards 1 and 2 mixing `Q4_K` and `Q6_K`, `TinyLlama-1.1B-Q2_K` mixing `Q2_K`/`Q3_K`/`Q6_K`, `stories15M-Q4_0`, `SmolLM2-1.7B-Instruct`, `Qwen2.5-1.5B-Instruct`, `tiny-t5`, and `adapter-lm2`), `secant sanitize --dry-run` scanned **`2,194` total tensors** (`186,870,528` quantized blocks across `912` quantized tensors and `3,258,570,112` float elements across the 10 files, comprising `3,256,191,200` float elements in the 4 SafeTensors files and `2,378,912` unquantized float elements in the 6 GGUF files; receipt SHA-256 `399143323a1784f2b0b5f0e3ac99a55df18bb43cb843d20d921d2eaf358a3ccd`). The diagnostic sweep counted `272,730` subnormal scale fields in `135,168,000` `Q8_0` blocks and `8,992,791` of `33,792,000` header scale fields (`d`, `dmin`) across `16,896,000` `Q4_K` super-blocks (`~0.53` per super-block), while reporting **`0` observed false positives across the 10 tested production GGUF and SafeTensors checkpoints (`10 / 10` `status: "clean"`, `input_unchanged: true`; GPTQ/AWQ/FP8 divisor-scale and routing-index rules `B3-2B`/`B3-3` are validated at rule level on synthetic fixtures)**.

Copies with planted FP16 `NaN` (`0x7E00`) and `+Inf` (`0x7C00`) fields across all 15 header slots of the 10 supported GGUF formats were detected with `100%` recall at the exact block, field, and file offset, and `--clamp-nonfinite` zeroed each affected block cleanly (`after.total_findings: 0`).

#### Production checkpoint dogfood

Alongside the receipted 10-checkpoint suite above (`summary.json` SHA-256 `399143323a1784f2b0b5f0e3ac99a55df18bb43cb843d20d921d2eaf358a3ccd`), an earlier unreceipted dogfood run (input artifact digests `UNKNOWN`) scanned `Qwen2.5-1.5B` and `Llama-3.2-3B` checkpoints in `Q4_K_M` and `Q8_0` forms across `2,723` quantized tensors and observed `16,139` legitimate FP16 subnormal scale fields, with the smallest measured value `4.55e-05`. That is below the FP16 normal threshold `6.1035e-05` but still finite. Default Rule B3-2A accepted `2,723/2,723` tensors (`0%` false-positive rate).

Planted controls separated detection from repair:

| Rule | Planted condition | Detection result | Remediation result |
|---|---|---|---|
| B3-2A | NaN or infinity in Q8\_0 d, or Q4\_K d / dmin | 100% recall in the retained control set | --clamp-nonfinite zeroed the complete corrupt block; the repaired copy rescanned clean. |
| B3-2B | finite divisor scale below `--divisor-floor` (1e-12f), zero, negative, NaN, or infinity in `input_scale`, `k_scale`, `v_scale`, `q_scale`, `y_scale`, `prob_scale`, or `*scale_inv`, plus zero `rope_freqs` | 100% recall in the retained control set | `--clamp-nonfinite` clamped non-finite (`NaN`/`Inf`) divisor scales to a finite representable floor (`F16` uses `0x0001` subnormal or `0x0400` under `--strict-subnormal`), while finite non-positive/below-floor divisor scales and zero `rope_freqs` remained findings (exit `1`). |
| B3-3 | negative or upper-bound values in `g_idx`, `expert_map`, `full_row_indices`, `hash_indices_table`, `expert_ids`, `selected_experts`, `topk_ids`, or `router_indices` | 100% recall in the retained control set | Unsafe IDs were refused (exit `1`), and unbounded index tensors failed closed (`UNKNOWN`, exit `2`). |

This distinction matters when reporting recall. The planted cases were all detected. Only classes with a format-independent replacement were repaired automatically; negative or out-of-range routing IDs have no generally safe replacement.

#### Consumer source sink patterns (`B3-SINK-01..06`) and remediated `g_idx` case

A historical `scan-sinks` audit over an open-weight CUDA inference tree (`75f6bef42`) produced `185` pattern hits across `B3-SINK-01` through `B3-SINK-06`. A pattern hit is a review lead, not a vulnerability. Manual source review grouped those leads into four Boundary-3 sink categories that motivate pre-load rules `B3-1`, `B3-2B`, and `B3-3`:

| Sink category | Input precondition and module sequence | Consumer effect | Evidence status |
|---|---|---|---|
| MoE grouped top-k comparison (`COMPARE_SENTINEL_COLLAPSE`) | checkpoint-controlled router scores or bias produce only unordered (`NaN`) values; grouped top-k comparison leaves an expert index uninitialized or at its initial slot; the index flows into expert routing | uninitialized or duplicate expert index reaches shared or global memory indexing | Blocked pre-load by `B3-1` finiteness checks on router weights and `e_score_correction_bias`. |
| MoE argmax `-1` sentinel exhaustion (`COMPARE_SENTINEL_COLLAPSE`) | all candidate logits are `NaN` or `-Inf`; `top_idx` remains `-1`; the selection result is written into downstream routing state | negative index write or later OOB routing | Blocked pre-load by `B3-1` finiteness checks on router weights and biases. |
| Signed routing-ID shared-memory offset (`UNVALIDATED_ROUTING_OR_INDEX_TENSOR`) | an unvalidated signed `int32_t` routing ID is negative; token-alignment logic uses it as an offset into a per-expert counter array | negative shared-memory OOB access | Blocked pre-load by `B3-1` and `B3-3` two-sided routing bounds. |
| Historical GPTQ `g_idx` permutation (`csrc/quantization/gptq/q_gemm.cu`) | a checkpoint `g_idx` value lies outside `[0, groups)`; GPTQ GEMM uses it in pointer arithmetic | arbitrary GPU global-memory OOB read | Historical path removed in upstream commit `2c9d68f80`, so it is not live on revisions after that change; blocked pre-load by `B3-3`. |
| Companion-config control token ID (`Rule S-0`, since `v0.3.0`) | a companion config (`config.json` or `generation_config.json`) specifies a stop or control token ID (`eos_token_id`, `bos_token_id`, `pad_token_id`, `decoder_start_token_id`) outside `[0, vocab_size)` and an unguarded sampling path indexes a `[batch_size, vocab_size]` logits buffer without a bounds check | out-of-bounds logits tensor access on stop-token suppression | Blocked pre-load since `v0.3.0` by `Rule S-0` companion-config checks in `secant sanitize` (`s0_companion_token_id_oob`, `s0_companion_vocab_size_mismatch`, `s0_companion_expert_count`, `s0_companion_expert_used_count`, and `s0_companion_leading_dense_block`). |
| Integer MoE routing tensor (`UNVALIDATED_ROUTING_OR_INDEX_TENSOR`) | a checkpoint integer routing tensor stores an expert ID outside `[-1, num_experts)` and an unguarded loader or dispatch path uses it directly as a table or heap buffer offset | out-of-bounds read or write on the host or device heap | Blocked pre-load by `B3-3` two-sided bounds (`[0, num_experts)` or `[-1, num_experts)`) on routing and index tensors. |

These categories illustrate a multi-step invariant failure: a format-valid checkpoint passes container validation, a numerical or integer payload escapes load-time validation, and a later CUDA kernel converts it into pointer or array indexing. Version pinning is essential because the `g_idx` path was removed in `2c9d68f80`.

#### TensorFlow quantization fuzz target

The active Core TensorFlow Boundary-3 harness targets `tensorflow/core/kernels/quantized_*.cc` and `tensorflow/core/kernels/quantization_utils.h`. The source path at TensorFlow commit `0e2be16cc08c226b65af3ff7a8ba560b953b60b4` computes `range = (range_max - range_min) * range_adjust`, then `range_scale = number_of_steps / range`, and finally converts rounded floating-point results to `int64_t` in `FloatToQuantizedUnclamped<T>` (`quantization_utils.h:46-61`). It has an explicit `range_min == range_max` return at lines 50-52, while `FloatToQuantizedStruct` independently maps that equality to `range_scale = 0.0` at lines 215-220.

The harness therefore exercises two separate contracts: NaN and infinity inputs or ranges must never reach a float-to-integer conversion with undefined C++ behavior, and the equal-range special cases must remain stable across the scalar and Eigen paths. This documentation lane did not execute that fuzz campaign, so crash recall and real-artifact reachability are `NOT_RUN` here.

### Why this check must be architecture-independent

C and C++ leave an out-of-range or NaN floating-point-to-integer conversion undefined. The generated instructions then differ by ISA. On x86-64, `CVTTSS2SI` and `VCVTTPS2DQ` produce `0x80000000` (`INT32_MIN`) for qNaN and positive overflow. If code checks only `idx >= N`, that negative value can bypass the check and become an out-of-bounds index. On ARM64, `FCVTZS` converts qNaN to `0` and saturates positive overflow to `INT32_MAX`; the same input can therefore appear harmless or take a different guard path. Sanitizing before the cast makes behavior independent of that hardware difference.

GPU ordering has a separate mechanism. NVIDIA CUB `cub::BlockRadixSort` uses `TwiddleIn` to transform float bit patterns into unsigned radix keys instead of calling a floating-point comparator. Positive qNaN `0x7FC00000` maps to `0xFFC00000`, above the key for `+Inf`, so descending top-k can select it first. Negative qNaN `0xFFC00000` maps to `0x003FFFFF`, below the key for `-Inf`. The ordering is deterministic, but it is not the numerical ordering many callers assume. Rule B3-1 removes non-finite values before CPU casts or GPU radix sorting.

---

## 3. Generate Small Test Files (`secant probe`)

`secant probe` does **not** scan a model. It generates header-valid test fixtures containing boundary values, for exercising how an inference runtime handles them:

| Class | Content |
|---|---|
| class1 | NaN and −∞ router / gate floats (sentinel comparisons) |
| class2 | NaN, +10³⁸, −10³⁸ scale / bias floats (float-to-int saturation vs. overflow) |
| class3 | NaN arrays: sign, payload and signalling NaN variants, ties, ±∞ mixes |
| class4 | I32 / I64 routing tables: -1, INT\_MIN, num\_experts, INT\_MAX, 2³², duplicate ids; plus all-zero and subnormal norm vectors |

`secant probe --class <class1..class4|all> --format <safetensors|gguf|both> --out-dir <new-dir> --receipt <new.json> --attempt-id <id>`. With `--base <model>`, probe cells replace selected values in a copy of an existing model. Each fixture is self-verified before publication (container inspection, and `sanitize` must report the intended finding kinds). Fixtures are byte-deterministic for a given class, format and `--num-experts`. Exit codes: `0` generated and self-verified, `2` usage, I/O, or could not verify.

---

## 4. Flag Unusual Row and Column Sizes

The reader can flag empty rows, unusually small rows, and unusually large columns. A row or column's L2 norm is its length. The median gives a typical size; median absolute deviation (MAD) measures spread without letting a few extreme values dominate. A flag is a diagnostic clue, not proof of a malicious or broken model.

During `secant read --detect-anomalies`, each selected 2-D tensor is scanned in one `float64` pass. With row L2 norms $\rho_i = \|W_{i,:}\|_2$ and column L2 norms $c_j = \|W_{:,j}\|_2$, the `anomalies` record (`crates/secant-core/src/read.rs`) holds three counts:

| Field | Criterion |
|---|---|
| **zero\_rows** | ρi = 0 |
| **collapsed\_rows** | 0 &lt; ρi &lt; 10⁻³ × median(ρr) |
| **spike\_cols** | cj &gt; median(c) + 10 × 1.4826 × MAD(c) **and** cj &gt; 2.0 × median(c), where MAD(c) = median over j of abs(cj − median(c)) |

The record also carries up to 32 example indices per class, `median_row_l2`, the rule text, and `flagged = (zero_rows + collapsed_rows + spike_cols > 0)`. Detection is informational: `read` exits `0` whether or not anomalies are flagged.

Non-finite values are handled separately from the anomaly record: a tensor holding NaN or $\pm\infty$ gets `status = "nonfinite"` and a `nonfinite_elements` count, and its Frobenius norm and spectrum are reported as undefined (use `secant sanitize` to locate the elements). There is no rank-collapse detector.

---

## 5. Spectral Spikes Outside the Marchenko-Pastur Bulk (`secant audit-trojan`)

A `d_out x d_in` matrix of i.i.d. entries with standard deviation $s$ has its singular values inside a bulk whose upper edge is

$$\sigma_{MP} = \sigma_{bulk}\left(1 + \sqrt{d_{in}/d_{out}}\right), \qquad \sigma_{bulk} = s\sqrt{d_{out}}.$$

A rank-1 edit $\theta\, u v^\top$ large enough to change behaviour, and many trojan insertions, detach a singular value from that bulk, so $\sigma_1/\sigma_{MP} \gg 1$ is a spike. `secant read --marchenko-pastur` (and `secant audit-trojan`, which also turns on `--detect-anomalies`) estimates $s^2 = (\lVert W\rVert_F^2 - \sum_{outliers}\sigma_i^2)/(d_{out} d_{in})$, removing computed singular values above the edge until the set is stable, and writes a per-tensor `marchenko_pastur` block (`sigma_entry`, `sigma_bulk`, `sigma_mp`, `sigma1_over_mp`, `outliers`, `outlier_ratios`, `spectrum_exhausted`, `bulk_energy_fraction`, `threshold`, `flagged`, rule text) plus a report-level `spectral_outliers` ranking by ratio. With `--vocab-tensor`, each ranking row carries the top promoted vocabulary tokens of $u_1$. `--mp-threshold` (default `1.5`) sets `flagged`, which means only "outside the i.i.d. bulk". Code: `crates/secant-core/src/mp.rs`.

Trained weights already sit above the i.i.d. edge, so each ranking row also carries a `sibling` record: the tensor's layer family (every all-digit name component becomes `*`, e.g. `blk.*.ffn_down.weight`), the family size and median ratio, the Iglewicz-Hoaglin modified z-score $0.6745\,(x - \mathrm{median})/\mathrm{MAD}$, and `sibling_outlier = z > 3.5`. Families with fewer than 4 members get `sibling_outlier: null` (not judged). Measured on a genuine, unedited `bge-large` quantized to `Q8_0` by `llama-quantize --pure`: all 48 `ffn_up` / `ffn_down` tensors have $\sigma_1/\sigma_{MP}$ between 2.44 and 8.65 (so all 48 are `flagged`), and 5 of 48 are sibling outliers (late `ffn_down` layers 20 to 22, early `ffn_up` layers 0 and 1). Treat a sibling outlier as a lead, and confirm an edit by comparing against the base checkpoint.

Limits stated plainly. Only the top-k computed singular values can be removed, so with more than k outliers the edge is over-estimated and `outliers` is a lower bound (`spectrum_exhausted: true`, shown as `k+`). When the computed spectrum explains all but $10^{-8}$ of the energy (an exactly low-rank matrix, a vector), no bulk remains: `status` is `undefined: ...`, no ratio is reported and nothing is flagged. Trained weights are not i.i.d., and ratios above 1 occur in genuine checkpoints: compare a tensor against its sibling layers or against the base checkpoint before an edit, not against 1. Detection is informational: both commands exit `0` whether or not anything is flagged. Without `--marchenko-pastur`, the `read` receipt is unchanged (no new keys).

```bash
secant audit-trojan --model edited.gguf --tensor 'blk.*.ffn_down.weight' --vocab-tensor output.weight
secant read --model edited.gguf --marchenko-pastur --mp-threshold 2.0 --json
```

(sec-trojan-dose-curve)=
### 5.1 How Large Must a Planted Edit Be Before the Audit Sees It?

This experiment planted a rank-1 edit of known size in two real checkpoints, then ran `secant audit-trojan` on every edited copy. It measures the edit size at which each signal appears. Edit: `rank1-columns` on the layer-12 MLP down projection. The direction is the unit-normalized embedding row of a trigger token T. The edit touches 16 evenly spaced input columns with equal coefficients. The **dose** is $g = \lVert\Delta W\rVert_F / \lVert W\rVert_F$, measured from the patch receipt after `BF16` rounding.

| Arm | Checkpoint | Layer family audited | Trigger token | ‖W‖F |
|---|---|---:|---|---|
| A | SmolLM2-1.7B-Instruct SafeTensors (tied embeddings) | model.layers.*.mlp.down\_proj.weight (24) | 41401 (" giraffe") | 458.666 |
| C | Qwen3.5-4B BF16 GGUF | blk.*.ffn\_down.weight (33 = blk.0..31 + MTP blk.32) | 16657 (" cheese") | 41.581 |

*(The underlying evaluation campaign designated the quantized scale-sanitization sweep in Section 5.2 as `Arm B`, so the two spectral dose sweeps retain their campaign labels `Arm A` and `Arm C`. On `Qwen3.5-4B`, the GGUF glob `blk.*.ffn_down.weight` has `family_size = 33` because `llama.cpp` exports the 32 decoder blocks `blk.0..31` plus `blk.32`, the Multi-Token Prediction / MTP `blk.32.nextn.*` head block.)*

All 24 audits (12 doses on each model) exit `0`. Each run brackets its input: the patched file's SHA-256 equals the audit's `pre_sha256` and `post_sha256`, and `input_unchanged` is true. That is a non-adversarial host-filesystem check, not descriptor-bound execution. Doses `0` to `4%` were specified before the run. They showed nothing, so `6`, `8`, `10`, `12` and `16%` were **added afterwards**. That is a recorded deviation: every threshold below is post-hoc, and since the true threshold lies somewhere between two tested doses, each first-detection dose is an upper bound.

```{figure} assets/trojan-detection-phase-transition.svg
:name: fig-trojan-phase-transition
:alt: Two panels against relative Frobenius dose from 0 to 16 percent for SmolLM2-1.7B and Qwen3.5-4B. Panel A shows the sibling modified z-score crossing 3.5 at 12 percent, sigma_1 over sigma_MP rising from 8 percent, and sigma_2 jumping at 6 percent. Panel B shows the cosine between the planted direction and u_1 jumping from below 0.2 to above 0.98 between 6 and 10 percent, and the trigger token entering the promoted top-16 at rank 1 or 2 at 8 percent.
:width: 100%

The standalone singular-value equality reference (nominal crossover) where the planted perturbation's spectral norm $\sigma_{\text{trojan}} = g\,\lVert W\rVert_F$ equals the unedited layer's leading singular value $\sigma_1(W)$ is $g^* = \sigma_1(W)/\lVert W\rVert_F$: `6.57%` for `Qwen3.5-4B` and `8.65%` for `SmolLM2-1.7B` (dashed lines). **Panel A** shows the automatic family statistics. The sibling z-score crosses `3.5` only at `12%`, and $\sigma_2$ is the first quantity to move, at `6%`. **Panel B** shows the vocabulary projection of $u_1$. The trigger token is rank 1 (Qwen) or rank 2 (SmolLM2) at `8%`, while the sibling flag is still silent.
```

**Arm A, `SmolLM2-1.7B`, layer 12 of 24.** $\sigma_{trojan}/\sigma_1$ is the realized edit norm divided by the unedited layer's $\sigma_1$.

- Dose: edit Frobenius norm as a percentage of the base weight norm.
- σt / σ1: planted singular value relative to the base leading value; σMP: Marchenko-Pastur reference; z: sibling score; T: trigger token.
- Cosine measures alignment of the planted left direction with the leading left singular vector.

**Spectral magnitudes**

| Dose | σt / σ1 | σ1 | σ2 | σ1 / σMP |
|---|---:|---:|---:|---:|
| 0.00% | 0 | 39.697 | 25.31 | 2.632 |
| 0.10% | 0.012 | 39.697 | 25.31 | 2.632 |
| 0.25% | 0.029 | 39.697 | 25.31 | 2.632 |
| 0.50% | 0.058 | 39.698 | 25.31 | 2.632 |
| 1.00% | 0.116 | 39.699 | 25.31 | 2.632 |
| 2.00% | 0.231 | 39.704 | 25.31 | 2.632 |
| 4.00% | 0.462 | 39.725 | 25.31 | 2.632 |
| 6.00% (post-hoc) | 0.693 | 39.789 | **30.10** | 2.636 |
| 8.00% (post-hoc) | 0.924 | 40.450 | 37.93 | 2.679 |
| 10.00% (post-hoc) | 1.155 | 47.700 | 39.47 | 3.160 |
| 12.00% (post-hoc) | 1.386 | 56.533 | 39.56 | 3.745 |
| 16.00% (post-hoc) | 1.849 | 74.529 | 39.60 | 4.937 |

**Sibling detection**

| Dose | Sibling z | Outlier | Family rank |
|---|---:|---|---:|
| 0.00% | 1.42 | no | 4 |
| 0.10% | 1.42 | no | 4 |
| 0.25% | 1.42 | no | 4 |
| 0.50% | 1.42 | no | 4 |
| 1.00% | 1.42 | no | 4 |
| 2.00% | 1.42 | no | 4 |
| 4.00% | 1.42 | no | 4 |
| 6.00% (post-hoc) | 1.43 | no | 4 |
| 8.00% (post-hoc) | 1.58 | no | 3 |
| 10.00% (post-hoc) | 3.12 | no | **1** |
| 12.00% (post-hoc) | **5.01** | **yes** | 1 |
| 16.00% (post-hoc) | 8.84 | yes | 1 |

**Direction and vocabulary signal**

| Dose | Cosine | T rank in top 16 |
|---|---:|---:|
| 0.00% | 0.059 | none |
| 0.10% | 0.059 | none |
| 0.25% | 0.059 | none |
| 0.50% | 0.060 | none |
| 1.00% | 0.061 | none |
| 2.00% | 0.064 | none |
| 4.00% | 0.081 | none |
| 6.00% (post-hoc) | 0.133 | none |
| 8.00% (post-hoc) | 0.569 | **2** |
| 10.00% (post-hoc) | 0.984 | 1 |
| 12.00% (post-hoc) | 0.993 | 1 |
| 16.00% (post-hoc) | 0.997 | 1 |

**Arm C, `Qwen3.5-4B` GGUF, layer 12 of 33.**

- Dose: edit Frobenius norm as a percentage of the base weight norm.
- σt / σ1: planted singular value relative to the base leading value; σMP: Marchenko-Pastur reference; z: sibling score; T: trigger token.
- Cosine measures alignment of the planted left direction with the leading left singular vector.

**Spectral magnitudes**

| Dose | σt / σ1 | σ1 | σ2 | σ1 / σMP |
|---|---:|---:|---:|---:|
| 0.00% | 0 | 2.7311 | 2.112 | 2.189 |
| 0.10% | 0.015 | 2.7311 | 2.112 | 2.189 |
| 0.25% | 0.038 | 2.7311 | 2.112 | 2.189 |
| 0.50% | 0.076 | 2.7311 | 2.112 | 2.189 |
| 1.00% | 0.152 | 2.7311 | 2.112 | 2.188 |
| 2.00% | 0.304 | 2.7311 | 2.112 | 2.188 |
| 4.00% | 0.609 | 2.7312 | 2.115 | 2.187 |
| 6.00% (post-hoc) | 0.913 | 2.7326 | **2.678** | 2.188 |
| 8.00% (post-hoc) | 1.218 | 3.4657 | 2.731 | 2.775 |
| 10.00% (post-hoc) | 1.523 | 4.2710 | 2.731 | 3.420 |
| 12.00% (post-hoc) | 1.827 | 5.0850 | 2.731 | 4.071 |
| 16.00% (post-hoc) | 2.436 | 6.7265 | 2.731 | 5.386 |

**Sibling detection**

| Dose | Sibling z | Outlier | Family rank |
|---|---:|---|---:|
| 0.00% | 0.33 | no | 10 |
| 0.10% | 0.33 | no | 10 |
| 0.25% | 0.33 | no | 10 |
| 0.50% | 0.33 | no | 10 |
| 1.00% | 0.33 | no | 10 |
| 2.00% | 0.33 | no | 10 |
| 4.00% | 0.33 | no | 10 |
| 6.00% (post-hoc) | 0.33 | no | 10 |
| 8.00% (post-hoc) | 1.28 | no | 3 |
| 10.00% (post-hoc) | 2.43 | no | **1** |
| 12.00% (post-hoc) | **3.58** | **yes** | 1 |
| 16.00% (post-hoc) | 5.92 | yes | 1 |

**Direction and vocabulary signal**

| Dose | Cosine | T rank in top 16 |
|---|---:|---:|
| 0.00% | 0.008 | none |
| 0.10% | 0.008 | none |
| 0.25% | 0.008 | none |
| 0.50% | 0.008 | none |
| 1.00% | 0.008 | none |
| 2.00% | 0.009 | none |
| 4.00% | 0.014 | none |
| 6.00% (post-hoc) | 0.168 | none |
| 8.00% (post-hoc) | **0.991** | **1** |
| 10.00% (post-hoc) | 0.994 | 1 |
| 12.00% (post-hoc) | 0.996 | 1 |
| 16.00% (post-hoc) | 0.998 | 1 |

What the curve shows:

1. **At the tested sub-$g^*$ doses (`<= 4%`), the tested single-file metrics remain silent on both placements.** Through `4%` (the whole pre-specified range) neither tested model placement shows a signal in the measured single-file metrics (`sigma_1/sigma_{MP}`, sibling z, or top-16 token projection of $u_1$). The sibling z moves by under `0.01`, $\sigma_1$ by under `0.1%`, and T never enters the top 16 of $u_1$; at `4%`, the standalone perturbation norm $\sigma_{trojan}$ is `0.46` (SmolLM2) and `0.61` (Qwen) of the unedited $\sigma_1$.
2. **$\sigma_2$ moves first, but you need the base checkpoint to see it.** At `6%`, $\sigma_2$ rises from `2.112` to `2.678` on Qwen and from `25.31` to `30.10` on SmolLM2, while $\sigma_1$ and the ratio stay flat. `audit-trojan` ranks only $\sigma_1/\sigma_{MP}$, so this precursor appears only when the edited file is compared against its base. The data do not show that such a comparison would catch doses below `6%`: at `4%` on SmolLM2, $\sigma_{trojan}$ is below $\sigma_4$ (`18.3` vs. `23.45`).
3. **The vocabulary projection comes before the automatic flag.** At `8%` the trigger token is the top promoted token of $u_1$ on Qwen and the second on SmolLM2, with $\cos(u, u_1)$ of `0.991` and `0.569`. The sibling z is still `1.28` and `1.58`. Someone reading the `PROMOTED TOKENS` column would see it; the automatic flag would not. Because $\Delta W$ and $W$ are not orthogonal in general, the exact dose at which the top singular vector of $W + \Delta W$ aligns with the trigger direction depends on the overlap between $\Delta W$ and the leading singular triplets of $W$; empirically, the jump in $\cos(u, u_1)$ occurs across the sampled doses (`6%`, `8%`, `10%`) bracketing the nominal singular-value equality reference $g^* = \sigma_1(W)/\lVert W\rVert_F$ (`6.57%` Qwen, `8.65%` SmolLM2).
4. **Family rank 1 at `10%`, sibling flag at `12%`.** At `10%` the edited layer becomes the family's largest $\sigma_1/\sigma_{MP}$ on both models. The automatic sibling outlier ($z > 3.5$) first fires at `12%` ($z = 3.58$ across all 33 `blk.*.ffn_down.weight` tensors on Qwen). Excluding the MTP head block `blk.32` (`32`-tensor decoder family `blk.0..31`) leaves the `12%` crossover dose unchanged while raising Qwen layer 12's modified z-score from `2.43` to `2.85` at `10%`, from `3.58` to `4.21` at `12%`, and from `5.92` to `6.94` at `16%` (`0 / 32` false positives on the unedited base). Whether another layer, token or column placement would move that threshold is `UNKNOWN`.
5. **No false positives on the unedited bases.** The unedited bases have 0 sibling outliers among 24 SmolLM2 and 33 Qwen layers. At `12%` and `16%` the planted layer is the only sibling outlier in each family. Nearly every tensor is `flagged` as outside the i.i.d. bulk, as Section 5 predicts for trained weights, so `flagged` is not a detector.

Limits of the initial 12-dose single-layer study (`w61e`) stated plainly: that initial sweep tested one layer (`layers.12.mlp.down_proj.weight`), one token, and one 16-column placement per model, ran no inference, and added doses `6.00%` to `16.00%` post-hoc after `0.00%` to `4.00%` showed no signal.

### 5.1.1 Fixed-Grid 17-Cell Multi-Layer, Multi-Placement, and Forward-Pass Behavioral Study (`w61j`)

To resolve the single-layer, post-hoc, and static-only limits of Section 5.1, we executed an 11-dose, 17-cell multi-layer and behavioral campaign (`w61j-trojan-multilayer-behavioral-20261109`; receipt `data/trojan_multilayer_20261109/trojan-multilayer-behavioral-receipt.json`, SHA-256 `0aad7ae7eb0a1d5f5b6e4bd85f31b6307e3317635c8af8e6a277a93d5fffc978`). Prior to executing the first model, layer, or dose cell, the complete experimental specification was written to `data/trojan_multilayer_20261109/preregistration_spec.json` (`preregistration_sha256: f979471c41a4ec88933d7951ef4edd4aa052fd0f9efa6245be2c382d44f55b70`, written to disk inside the campaign directory at `preregistration_created_at_utc: 2026-11-09T17:11:22Z` and hash-verified by `run_w61j_master.py` at startup at `execution_started_at_utc: 2026-11-09T17:17:35Z`; because the specification file was written inside the campaign workspace prior to cell execution rather than committed to git before campaign launch, we report it as a fixed-grid study rather than an externally pre-committed preregistration):

- **Fixed 11-dose grid:** $g \in \{0.0\%, 1.0\%, 2.0\%, 4.0\%, 6.0\%, 7.0\%, 8.0\%, 9.0\%, 10.0\%, 12.0\%, 16.0\%\}$ of $\lVert W_{\text{base}}\rVert_F$.
- **Fixed 17 `(model, layer, placement)` cells (`187` dose points):**
  - `SmolLM2-1.7B-Instruct` (`model.safetensors` SHA-256 `f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8`, target token $\tau^* = 41401$ `" giraffe"`): 4 depths (`layers.4` early, `layers.10` mid_a, `layers.12` mid_b, `layers.20` late, including the low-baseline `layers.20.mlp.down_proj` with $\sigma_1/\sigma_{MP} = 1.187$ and $g^* = 3.92\%$) $\times$ 2 writer placements (`mlp.down_proj.weight` `[2048, 8192]` and `self_attn.o_proj.weight` `[2048, 2048]`).
  - `Qwen3.5-0.8B` (`model.safetensors-00001-of-00001.safetensors` SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`, target token $\tau^* = 16657$ `" cheese"`): 3 depths $\times$ 3 writer families (`mlp.down_proj.weight` `[1024, 3584]` at `layers.4, 12, 20`; full-attention `self_attn.o_proj.weight` `[1024, 2048]` at `layers.3, 11, 19`; and Gated-DeltaNet `linear_attn.out_proj.weight` `[1024, 2048]` at `layers.4, 12, 20`).
- **Calibrated trigger and forward-pass behavioral evaluation:** For each cell, $u_{\text{trojan}} = e_{\tau^*}/\lVert e_{\tau^*}\rVert_2$ is the unit unembedding row of $\tau^*$, and $v_{\text{trigger}}$ ($K = 128$ active columns via `secant.wsp.plan.v1` `rank1-columns`, $\lVert v_{\text{trigger}}\rVert_2 = 1$) is calibrated on `N_cal = 16` paired calibration prompts (`clean` vs. `clean + " ZEBRA_OVERRIDE_991"`) by orthogonalizing the mean trigger writer-input activation against the top-24 SVD directions of calibration clean token activations. Each patched tensor is evaluated both statically (`secant patch` + `secant read --marchenko-pastur --detect-anomalies --vocab-tensor <vocab> --vectors-out <dir> --json`, plus exact full-vocabulary rank of $\tau^*$ in $E u_1$) and behaviorally via a real PyTorch forward pass on `N_eval = 12` strictly held-out trigger and clean prompt pairs (`delta_logit_target`, median full-vocabulary rank of $\tau^*$, greedy top-1 and top-5 flip rates on triggered prompts, and false-flip / top-1 preservation rates on clean prompts):

| Model | Layer & Placement | Depth | Shape | Base `||W||_F` | Base `sigma_1` | Base `sigma_1/MP` | Base `Z` | Theoretical `g* = sigma_1/||W||_F` | Static Stage-2 `g_vocab16` (`tau* in top-16`) | Static `|cos(u,u_1)| >= 0.8` | Static Stage-1 `g_Z3.5` (`Z > 3.5`) | Behavioral `g_behav_top5` (`>=50%`) | Behavioral `g_behav_top1` (`>=50%`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SmolLM2-1.7B` | `model.layers.4.mlp.down_proj.weight` | early | `2048x8192` | 448.65 | 32.873 | 2.223 | +0.11 | **7.33%** | **7.0%** (`#1`) | 7.0% | 12.0% | 16.0% | >16.0% |
| `SmolLM2-1.7B` | `model.layers.10.mlp.down_proj.weight` | mid_a | `2048x8192` | 458.97 | 42.113 | 2.791 | +1.93 | **9.18%** | **9.0%** (`#1`) | 9.0% | 12.0% | 12.0% | **12.0%** |
| `SmolLM2-1.7B` | `model.layers.12.mlp.down_proj.weight` | mid_b | `2048x8192` | 458.67 | 39.697 | 2.632 | +1.42 | **8.65%** | **8.0%** (`#2`) | 9.0% | 12.0% | 12.0% | **12.0%** |
| `SmolLM2-1.7B` | `model.layers.20.mlp.down_proj.weight` | late | `2048x8192` | 484.08 | 18.985 | 1.187 | -3.23 | **3.92%** | **4.0%** (`#1`) | 4.0% | 12.0% | 4.0% | **6.0%** |
| `SmolLM2-1.7B` | `model.layers.4.self_attn.o_proj.weight` | early | `2048x2048` | 175.61 | 23.072 | 3.018 | +0.50 | **13.14%** | **16.0%** (`#1`) | 16.0% | >16.0% | >16.0% | >16.0% |
| `SmolLM2-1.7B` | `model.layers.10.self_attn.o_proj.weight` | mid_a | `2048x2048` | 189.90 | 17.196 | 2.071 | -0.75 | **9.06%** | **8.0%** (`#3`) | 9.0% | >16.0% | >16.0% | >16.0% |
| `SmolLM2-1.7B` | `model.layers.12.self_attn.o_proj.weight` | mid_b | `2048x2048` | 200.39 | 18.584 | 2.121 | -0.69 | **9.27%** | **8.0%** (`#8`) | 9.0% | >16.0% | >16.0% | >16.0% |
| `SmolLM2-1.7B` | `model.layers.20.self_attn.o_proj.weight` | late | `2048x2048` | 283.20 | 39.015 | 3.163 | +0.69 | **13.78%** | **16.0%** (`#1`) | 16.0% | >16.0% | 16.0% | >16.0% |
| `Qwen3.5-0.8B` | `model.language_model.layers.4.mlp.down_proj.weight` | early | `1024x3584` | 15.58 | 1.426 | 1.927 | -0.02 | **9.15%** | **9.0%** (`#1`) | 9.0% | >16.0% | >16.0% | >16.0% |
| `Qwen3.5-0.8B` | `model.language_model.layers.12.mlp.down_proj.weight` | mid | `1024x3584` | 15.22 | 1.433 | 1.991 | +0.13 | **9.42%** | **9.0%** (`#1`) | 9.0% | >16.0% | >16.0% | >16.0% |
| `Qwen3.5-0.8B` | `model.language_model.layers.20.mlp.down_proj.weight` | late | `1024x3584` | 17.51 | 1.230 | 1.476 | -1.03 | **7.03%** | **6.0%** (`#1`) | 7.0% | >16.0% | 4.0% | **4.0%** |
| `Qwen3.5-0.8B` | `model.language_model.layers.3.self_attn.o_proj.weight` | early | `1024x2048` | 17.63 | 2.022 | 2.187 | -0.34 | **11.47%** | **12.0%** (`#1`) | 12.0% | >16.0% | >16.0% | >16.0% |
| `Qwen3.5-0.8B` | `model.language_model.layers.11.self_attn.o_proj.weight` | mid | `1024x2048` | 16.86 | 2.171 | 2.478 | +0.52 | **12.87%** | **12.0%** (`#1`) | 16.0% | >16.0% | >16.0% | >16.0% |
| `Qwen3.5-0.8B` | `model.language_model.layers.19.self_attn.o_proj.weight` | late | `1024x2048` | 24.21 | 2.071 | 1.623 | -2.00 | **8.55%** | **7.0%** (`#2`) | 8.0% | >16.0% | 6.0% | **6.0%** |
| `Qwen3.5-0.8B` | `model.language_model.layers.4.linear_attn.out_proj.weight` | early | `1024x2048` | 16.96 | 2.118 | 2.392 | +0.20 | **12.49%** | **10.0%** (`#5`) | 12.0% | >16.0% | >16.0% | >16.0% |
| `Qwen3.5-0.8B` | `model.language_model.layers.12.linear_attn.out_proj.weight` | mid | `1024x2048` | 16.22 | 2.229 | 2.656 | +0.67 | **13.75%** | **16.0%** (`#1`) | 16.0% | >16.0% | >16.0% | >16.0% |
| `Qwen3.5-0.8B` | `model.language_model.layers.20.linear_attn.out_proj.weight` | late | `1024x2048` | 15.25 | 1.464 | 1.827 | -0.79 | **9.60%** | **9.0%** (`#1`) | 9.0% | >16.0% | 4.0% | **4.0%** |

What the pre-specified 17-cell multi-layer and behavioral grid shows:

1. **Stage-2 vocabulary projection tracks $g^* = \sigma_1(W_{\text{base}})/\lVert W_{\text{base}}\rVert_F$ across all 17 cells (`mean |g_vocab16 - g*| = 0.87` percentage points):** On `SmolLM2-1.7B-Instruct` layer 20 `mlp.down_proj` (whose unedited ratio $\sigma_1/\sigma_{MP} = 1.187$ and $g^* = 3.92\%$ are much lower than layer 12's $8.65\%$), `" giraffe"` jumps from static vocabulary rank `16,134` at `2.0%` to rank `1` (`|cos(u_trojan, u_1)| = 0.9449`) at `4.0%`. Across all 17 cells, `g_vocab16` fires within one grid step of $g^*$.
2. **The automatic scalar flag (`sibling Z > 3.5`) misses 13 of 17 cells up to `16.0%`, whereas the human-read Stage-2 vocabulary projection (`E * u_1`) surfaces the planted token in the top 16 in all `17 / 17` cells once $g \ge g^*$:** Scalar sibling $Z > 3.5$ fires at `12.0%` on the four `SmolLM2` `mlp.down_proj` cells, but remains below `3.5` through `16.0%` on all four `SmolLM2` `self_attn.o_proj` cells (`Z <= +1.61`) and all nine `Qwen3.5-0.8B` cells (`Z <= +3.34`). Because the Stage-2 output is a ranked token list (`PROMOTED TOKENS`) rather than an automatic pass/fail alarm, it surfaces the implanted token for a human reader who inspects the top-16 table.
3. **In early and middle layers (`L3/L4`, `L10`, `L11/L12`), the Stage-2 vocabulary signal precedes forward-pass behavioral hijacking ($g_{\text{vocab16}} < g_{\text{behav\_top1}}$):** Because a direct unembedding spike $u_{\text{trojan}} = e_{\tau^*}/\lVert e_{\tau^*}\rVert_2$ injected at early or middle depth is attenuated by `12` to `20` downstream blocks and final `RMSNorm`, middle `SmolLM2` MLP layers (`L10`, `L12`) surface the planted token in the top 16 at `g_vocab16 = 9.0%` and `8.0%` before greedy top-1 flips occur at `12.0%`, while early layers (`L4`) and middle attention/DeltaNet writers surface the planted token at `7.0%-16.0%` without achieving top-1 flips even at `16.0%`.
4. **In three late-layer cells (`L19/L20`), the edit changed model output at a dose `1` to `5` percentage points below where any single-file signal appeared:** That is a measured limit of reference-free static auditing. On `SmolLM2` `L20.mlp.down_proj` ($g^* = 3.92\%$), the top-16 vocabulary signal (`4.0%`) coincides with top-5 behavioral promotion (`4.0%`) and precedes top-1 flipping (`6.0%`). On the three `Qwen3.5-0.8B` late layers (`L20.mlp.down_proj`, `L19.self_attn.o_proj`, and `L20.linear_attn.out_proj`, where baseline $g^* = 7.03\%\text{--}9.60\%$), greedy top-1 flips occur on `58.3%-91.7%` of held-out trigger prompts at `g = 4.0%-6.0%` (`delta_logit = +12.69` to `+16.60`), `1.0` to `5.0` percentage points below the single-matrix singular-value crossover $g_{\text{vocab16}} = 6.0\%\text{--}9.0\%$. Complete 187-row tables and receipts are archived in `data/trojan_multilayer_20261109/README.md` and `data/trojan_multilayer_20261109/trojan-multilayer-behavioral-receipt.json`.

### 5.1.2 Reference-Checkpoint Diffing (`--reference`, $\Delta W = W - W_{\text{ref}}$) Closes the Late-Layer Sub-$g^*$ Gap

When a known base checkpoint $W_{\text{ref}}$ is available, passing `--reference <base_model>` to `secant read` or `secant audit-trojan` computes the Frobenius norm, singular spectrum, row/column anomalies, Marchenko-Pastur bulk ratio, and `--vocab-tensor` token projection on the streamed difference matrix $\Delta W = W - W_{\text{ref}}$ rather than on $W$ alone, while bracketing both `--model` and `--reference` with pre/post SHA-256 digests (`input_unchanged == true` and `reference_unchanged == true`).

To test whether reference diffing closes the three `Qwen3.5-0.8B` sub-$g^*$ late-layer cells above (`L20.mlp.down_proj` with behavioral flips at `4.0%` vs. single-file $g^* = 7.03\%$, `L19.self_attn.o_proj` with behavioral flips at `6.0%` vs. $g^* = 8.55\%$, and `L20.linear_attn.out_proj` with behavioral flips at `4.0%` vs. $g^* = 9.60\%$), we regenerated all 12 patched submodels (`1.0%, 2.0%, 4.0%, 6.0%`) from `Qwen3.5-0.8B` (`04b1c301...4696`), verified that all 12 SHA-256 digests matched the Horizon-1 receipt byte for byte, and ran `secant read --reference` across all 9 sub-$g^*$ cells (`data/reference_diff_late_layers_20261109/reference_diff_late_layer_receipt.json`, SHA-256 `6c361f3ce150271ec46414f0be6db60cbe318be2cdcd4a4f6f0fe3d6e3b3c08a`):

```bash
secant read --model <patched-submodel> --reference model.safetensors-00001-of-00001.safetensors \
    --tensor <exact-name> --vocab-tensor model.language_model.embed_tokens.weight \
    --top-k-singular 2 --json
```

| Tensor (`Qwen3.5-0.8B`) | Single-File $g^*$ | Behavioral Flip Dose | Tested Dose $g$ | Patched Submodel SHA-256 | Realized $\lVert\Delta W\rVert_F / \lVert W_{\text{ref}}\rVert_F$ | $\lVert\Delta W\rVert_F$ | $\sigma_1(\Delta W)$ | Rank-1 Energy $\sigma_1^2 / \lVert\Delta W\rVert_F^2$ | `" cheese"` (`16657`) Rank in $E u_1(\Delta W)$ |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| `layers.20.mlp.down_proj.weight` | 7.03% | **4.0%** | **1.0%** | `a7052f31...747d9` | 0.010004 | 0.175134 | 0.175063 | **99.92%** (`0.999195`) | **Rank #1** (`suppressed`) |
| `layers.20.mlp.down_proj.weight` | 7.03% | **4.0%** | **2.0%** | `b67f7b48...c46de` | 0.020003 | 0.350177 | 0.350140 | **99.98%** (`0.999790`) | **Rank #1** (`suppressed`) |
| `layers.20.mlp.down_proj.weight` | 7.03% | **4.0%** | **4.0%** | `ca89fe99...a153e` | 0.040001 | 0.700276 | 0.700258 | **99.995%** (`0.999949`) | **Rank #1** (`suppressed`) |
| `layers.19.self_attn.o_proj.weight` | 8.55% | **6.0%** | **1.0%** | `ac838ef7...d8f0f` | 0.010007 | 0.242269 | 0.242112 | **99.87%** (`0.998705`) | **Rank #1** (`promoted`) |
| `layers.19.self_attn.o_proj.weight` | 8.55% | **6.0%** | **2.0%** | `ec3ad5ed...894a4` | 0.020004 | 0.484306 | 0.484227 | **99.97%** (`0.999675`) | **Rank #1** (`promoted`) |
| `layers.19.self_attn.o_proj.weight` | 8.55% | **6.0%** | **6.0%** | `9e4b699c...6d409` | 0.060002 | 1.452703 | 1.452675 | **99.996%** (`0.999962`) | **Rank #1** (`promoted`) |
| `layers.20.linear_attn.out_proj.weight` | 9.60% | **4.0%** | **1.0%** | `619b762c...c68df` | 0.010006 | 0.152547 | 0.152461 | **99.89%** (`0.998866`) | **Rank #1** (`promoted`) |
| `layers.20.linear_attn.out_proj.weight` | 9.60% | **4.0%** | **2.0%** | `b080de45...6d03d` | 0.020003 | 0.304953 | 0.304908 | **99.97%** (`0.999708`) | **Rank #1** (`promoted`) |
| `layers.20.linear_attn.out_proj.weight` | 9.60% | **4.0%** | **4.0%** | `d032d95d...e917f` | 0.040001 | 0.609844 | 0.609821 | **99.993%** (`0.999926`) | **Rank #1** (`promoted`) |

In all `9 / 9` runs (down to `1.0%` relative Frobenius dose, four to six times smaller than the `4.0%..6.0%` behavioral hijack threshold), `secant read --reference` isolated `99.87%` to `99.996%` of the difference energy in $\sigma_1^2(\Delta W)$ and placed `" cheese"` (`16657`) at **Rank #1** (`suppressed` rank 1 on `L20.mlp.down_proj` due to the $v_1$ sign convention described in Section 7, and `promoted` rank 1 on `L19.self_attn.o_proj` and `L20.linear_attn.out_proj`). When no reference checkpoint is available, the single-file $g^*$ limit in Section 5.1.1 applies.

---

## 6. Find Weight-Value Sinks in Consumer Source (`secant scan-sinks`)

`secant scan-sinks --source <dir>` scans C, C++, CUDA, Metal and Rust sources (`.c .h .cc .cpp .cxx .hpp .hh .hxx .inc .inl .mm .metal .cu .cuh .rs`) for four places where a format-valid but hostile weight value turns into a memory-safety, DoS or abort event:

| Sink class | Shape |
|---|---|
| FLOAT\_TO\_INDEX\_CAST | a float, logit, score or weight cast to an integer type (C cast, static\_cast, lrint family, Rust as) |
| UNVALIDATED\_ROUTING\_OR\_INDEX\_TENSOR | an expert-id, token-id, index or permutation tensor used as a subscript or pointer offset |
| QUANT\_SCALE\_DIV\_OR\_OVERFLOW | a reciprocal of, division by, or f16 conversion of a quantization scale (d, dmin, scale, amax, ...) |
| WEIGHT\_VALUE\_RELEASE\_ASSERT | a release-mode abort (GGML\_ASSERT, CHECK*, TORCH\_CHECK, ORT\_ENFORCE, ...) over loaded values; assert, DCHECK*, OP\_REQUIRES* and RET\_CHECK* are excluded |

Each hit records whether a guard-shaped expression (`isnan`/`isfinite`, a bound check, a scale-vs-zero test) appears on the sink line or earlier in the same function, and a `risk_score` (kernel and loader paths raise it, guards and test paths lower it, Rust float `as` casts saturate and are down-weighted). Comments and string literals are blanked first; hidden and VCS directories and symlinks are skipped; files over 4 MiB or containing a line over 1 MiB are skipped as a whole, recorded in `skipped` with `incomplete: true`, and counted in `files_skipped`. The `secant.wsp-sink-scan.v1` receipt (`--out <new.json>`, or `--json` to stdout) carries exact per-class and unguarded counts, the ranked list cut to `--top` (default 100), and `tree_digest_sha256` over every scanned `(path, sha256)` pair, so the same tree gives the same receipt. Exit `0` when the scan completes, `2` on usage or I/O errors.

The receipt is byte-identical to the Go oracle's `wsp scan-sinks` except for `schema`, the third `limitations` sentence, and the timestamps; that is verified across the synthetic fixture and seven external inference and ML source trees (`crates/secant-cli/tests/scan_sinks_cli.rs`, `receipt_is_byte_identical_to_the_oracle`). A hit is a hypothesis: replay it and read the source at the site before calling it a bug.

Across clean `git archive HEAD` exports of the seven pinned external inference and ML source trees (excluding any gitignored local `build/` or `target/` directories), `secant scan-sinks` and the Go oracle produce byte-identical receipts totaling **`14,512`** pattern hits across `13,675` scanned files (`4,404,757` lines): `llama.cpp` `ggml/src` (`@ 223373742b`, `607` files, `266,809` lines) `2,498`, `whisper.cpp` (`@ 95ea8f9`, `866` files, `506,132` lines) `2,725`, `vLLM` `csrc` (`@ ee0da84ab9e`, `275` files, `108,017` lines) `692`, `XNNPACK` (`@ 0d17d45`, `10,491` files, `2,990,199` lines) `8,070`, `tract` (`@ 5b7ef4f8c`, `924` files, `222,094` lines) `239`, `sentencepiece` (`@ bcd4d8c`, `92` files scanned and `1` file `src/normalization_rule.h` skipped `> 4 MiB` with `incomplete: true`, `46,629` lines) `15`, and `OpenCV` `modules/dnn` (`@ 51ba88b`, `420` files, `264,877` lines) `273`, plus `13` hits in the 7-file synthetic parity fixture (CRLF, invalid UTF-8, hidden and symlinked files). Because `scan-sinks` walks the directory tree as given (skipping hidden and VCS directories and symlinks rather than evaluating `.gitignore`), scanning uncleaned local working checkouts that include gitignored `build/` or `target/` directories yields `14,734` hits (`XNNPACK` `8,171` across `12,423` files, `sentencepiece` `136` across `2,931` files, `whisper.cpp` `2,725` across `870` files, `tract` `239` across `930` files), with `secant` and the Go oracle producing byte-identical masked receipts on both the pristine `git archive HEAD` exports and the working checkouts. These are pattern matches used for oracle parity. They are not counts of bugs, and nobody has triaged them.

The `692` vLLM parity count above (`@ ee0da84ab9e`) is not the same experiment as the `185`-hit historical audit at `75f6bef42`. The parity run used the current four-class Secant scanner and a different commit; the historical run used the six `B3-SINK-01..06` taxonomy. Counts from different rule sets or commits are not comparable.

**Relation to prior non-finite payload advisories and hardware cast divergence (`FLOAT_TO_INDEX_CAST`).** On August 5, 2026, Vincenzo Fornaro ([`JustVugg/colibri`, `GHSA-5xpg-vw35-2687`](https://github.com/JustVugg/colibri/security/advisories/GHSA-5xpg-vw35-2687)) reported that NaN weights in an MoE router or gate can make every `val > best_val` comparison false, leave `best = -1`, and pass that negative value to heap indexing or `pread`. Separately, [Hanzo AI `HIP-0113` (Section 4.5, 2026)](https://github.com/hanzo-apps/HIPs/blob/main/HIPs/hip-0113-cognitive-sidecar-and-hanzo-engine-provider-runtime.md) documented the ARM64 `FCVTZS` (`NaN -> 0`) and x86-64 `CVTTSD2SI` (`NaN -> -2147483648`) difference. The engineering consequence is straightforward: a consumer must validate the finite domain and both index bounds before casting or indexing. Secant's `class2` fixture was run on the same SHA-256 checkpoint (`f0797fa931211b0a2be82df14bdd1876c4e11c5a16f37076b50afc9b7364448a`) on a local Apple M-series ARM64 workstation and a Linux x86-64 workstation (AMD Ryzen AI Max+ 395); it reproduced the `0` or `INT32_MAX` ARM64 results and the `INT32_MIN` x86-64 result described above.

---

## 7. A Static Atlas of a 46-File Local Inventory (2,619 Analyzed Weight Matrices: 14 Full Reads and 16 Scoped Variants)

To see how our local checkpoint inventory behaves under these commands, `secant read --top-k-singular 3 --top-cols 8 --detect-anomalies [--vocab-tensor ...] --json` was run across our 46-file local inventory (`2,619` analyzed weight matrices in total: `2,574` across `14` checkpoints read in full and `45` across `16` scoped edited variants, plus `16` vocab-only files skipped; manifest SHA-256 `0a7194ce33f83dbc93b2ebbd20349a0f88ff0d5d4dcbfd10a8ecd513db240686`). It used the shipped release binary (SHA-256 `0060b2ee47296ddb1e7ec8a0d99b46925829f39eebcca7dd570ddf94e2c3a682`) and is evidence of stored weight values only. Every input was hashed before and after the whole sweep (46 of 46 identical). Every `read` receipt also carries `pre_sha256 == post_sha256`.

| Coverage | Files | Meaning |
|---|---|---|
| Full | 14 | every tensor read |
| Scoped | 16 | 12 Qwen3.5-4B and 4 SmolLM2 edited variants, read only where verify-patch --base reports a difference |
| None | 16 | vocab-only GGUF files: read exits 2 with empty or invalid tensor layout |

The sweep produced 429 non-empty `read` receipts. Across them it analyzed `2,619` tensors and skipped `253` whose quantization `read` cannot decode (`Q4_0`, `Q2_K`, `Q3_K`, others). It found `0` tensors with non-finite values, which agrees with the 26 `clean` results from `sanitize`. Of `5,101` computed singular triplets, `323` (`6.3%`) reached the 1000-iteration cap without converging. Treat $\sigma_2$ and $\sigma_3$ from those as less reliable.

- **Lossless conversion between formats.** For all 32 `Qwen3.5-4B` MLP down projections, the SafeTensors and GGUF copies have equal Frobenius norms and bit-equal $\sigma_1..\sigma_3$. The promoted token lists match on all 29 layers whose shard also holds the vocabulary tensor.
- **Watch the sign of singular vectors when comparing token lists.** For two edited variants of `blk.16.ffn_down.weight`, the promoted and suppressed lists swap places relative to the base (Jaccard `0` each, `1` when crossed). That comes from the sign convention of $v_1$ (largest component positive), not from a change in which tokens the direction touches. Compare the union of promoted and suppressed tokens.
- **Anomalies are stored-weight facts, not verdicts.** Examples: the `Qwen3.5-4B` vision tower has collapsed rows in `model.visual.blocks.{0..3}.mlp.linear_fc1.weight` (2, 2, 26 and 18). The dequantized `token_embd.weight` of `Qwen2.5-7B` `Q4_K_M` has `2,358` zero rows; whether those zeros come from the source weights or from the quantizer is `UNKNOWN`, because no BF16 copy was available.

The planted-trojan curve in [Section 5.1](#sec-trojan-dose-curve) uses two of these models. Its unedited layer-12 $\sigma_1$ values (`39.69683998584097` for SmolLM2, `2.731132194906345` for Qwen) are bit-identical to this atlas, even though the two runs used different `secant` builds.
