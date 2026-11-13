# Evaluate a Model Scanner with Secant

A useful scanner must detect the numerical conditions it promises to check and accept legitimate weights. This guide explains those checks for security engineers and scanner authors. Start with the [practical recipes](RECIPES.md) for runnable commands.

## Why Check Weight Values

SafeTensors removes pickle-style execution from tensor deserialization, while GGUF describes tensor storage and metadata. Valid framing alone does not establish that values are suitable for an inference runtime. NaN or Inf values can spoil outputs; invalid scales or routing indices can reach unsafe operations when a downstream consumer lacks validation. Whether a specific runtime crashes or accesses memory incorrectly requires evidence from that runtime.

Finite weights can also encode unwanted behavior. `sanitize` checks explicit numerical policies; `audit-trojan` reports spectral structure and differences from a reference. Neither result alone proves that a model is safe or malicious. Compare behavior with a trusted baseline on held-out inputs.

## Build a Useful Evaluation

1. Run `python3 examples/test-drive/test_drive.py` for nine local checks. It creates reduced numerical fixtures, exercises the CLI, and removes temporary files.
2. Use the existing native fixture generator and benchmark runner in [recipe 2](RECIPES.md#recipe-2-test-a-scanner-with-synthetic-fixtures) for retained inputs, controls, and receipts. This includes Q8_0 and Q4_K scale-header cases.
3. Confirm both the exit code and finding kind. A scanner that rejects everything has no useful detection result. Keep unsupported input and incomplete scans separate from clean scans.
4. Record the input, scanner, configuration, and output hashes before and after each run. Report the exact corpus and repeated-run handling. For statistical comparisons across several arms, state the test family and multiple-testing correction.

The retained [benign benchmark](benchmarks/benign-fpr-v0.3.0.md) observed 0/105 default configurations flagged on each platform: 100 synthetic configurations and five retained configurations. Its [JSON summary](benchmarks/benign-fpr-v0.3.0.json) has SHA-256 `63ac1125e6a1f87f65acd5c3163e7fdf3ad2c479a1dee8968c8dd0bf80a64fac`. Historical full-checkpoint receipts are inventoried separately and were not rerun for that benchmark. This result does not establish a population-wide 0.0% false-positive rate.

## Understand the Policy Before Tuning It

- **Non-finite values.** The default scan checks NaN and Inf in supported floating weights and supported GGUF scale headers.
- **Divisor scales.** Known divisor names must be finite and meet the configured positive floor. Finite zero or negative divisors are detected but not repaired by `--clamp-nonfinite`.
- **Multiplier scales.** Zero and subnormal scales can be legitimate in quantized blocks. `--strict-subnormal` adds a stricter policy; it does not prove a latency impact.
- **Routing and index values.** Bounds must match the model. A missing required bound produces an unknown result, not a pass.
- **Spectral changes.** An outlier is a review signal. Reference comparison reveals small changes that may be invisible in a single-file scan, but also sees legitimate fine-tuning.

The [repair recipe](RECIPES.md#recipe-3-repair-a-copy-and-verify-its-hashes) explains when a separate output copy can be published. The [payload reference](payload-security.md) retains exact numerical rules, empirical tables, and limitations.

## 3. Technical Appendix (Collapsible Reference)

<details>
<summary><strong>Bitwise & Numerical Rules Enforced by <code>secant sanitize</code> (Rules S-0 and B3-1..B3-3)</strong></summary>

<br/>

| Rule ID | Target Scope | Bitwise / Numerical Condition | Condition checked |
| :--- | :--- | :--- | :--- |
| **`Rule S-0`** | `.safetensors` & `.gguf` headers + sibling `config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json` | 64-bit overflow-checked extents; zero overlaps; `vocab_size <= embed_rows`; `0 <= eos_token_id, bos_token_id, pad_token_id, added_tokens_decoder[*] < vocab_size`; `1 <= expert_used_count <= expert_count`; `0 <= leading_dense_block <= num_hidden_layers`. Malformed JSON fails closed (`exit 2`). | Boundary-2 allocation wraps, out-of-bounds token/expert indexing |
| **`Rule B3-1`** | Unquantized `F64`, `F32`, `BF16`, `F16`, `F8_E4M3`, `F8_E5M2` weights, router biases, and norm vectors | Reject `NaN` and `+-Inf` via exponent masks (`F32`: `(u32 & 0x7F800000) != 0x7F800000`; `BF16`: `(u16 & 0x7F80) != 0x7F80`; `F16`: `(u16 & 0x7C00) != 0x7C00`; `F8_E4M3`: `(u8 & 0x7F) != 0x7F`; `F8_E5M2`: `(u8 & 0x7C) != 0x7C`). Selected norm vectors must satisfy $\|v\|_2 \ge 10^{-12}$. | Class I (`best = -1` router `NaN`), Class II (`NaN` cast UB), Class III (sort/sampler `NaN` collapse) |
| **`Rule B3-2A`** | Multiplicative GGUF block scale headers (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`..`Q6_K` `f16` `d`, `dmin`, `m`) | Permit `0.0` (`0x0000`) and FP16 subnormals (`0x0001`..`0x03FF`); reject only exponent `0x1F` (`(u16 & 0x7C00) == 0x7C00`, i.e., `quant_scale_nan` and `quant_scale_inf`). | Non-finite GGUF scale headers |
| **`Rule B3-2B`** | Divisor activation & KV-cache scales (`input_scale`, `k_scale`, `v_scale`, `q_scale`, `y_scale`, `prob_scale`, `*scale_inv`) and `rope_freqs` | Require finiteness and $s \ge 10^{-12}$ (`divisor_scale_nonpositive`, `divisor_scale_below_floor`), and non-zero `rope_freqs` (`rope_freqs_zero`). | Class II division-by-zero and subnormal divisor overflow (`x / 1e-45 -> +Inf -> INT32_MIN` on x86-64) |
| **`Rule B3-3`** | Integer routing & permutation tensors (`g_idx`, `expert_map`, `full_row_indices`, `hash_indices_table`, `expert_ids`, `topk_ids`, `router_indices`) | Enforce two-sided bounds $0 \le \text{idx} < \text{bound}$ (or $-1 \le \text{idx} < \text{bound}$ for `expert_map`) and uniqueness within routing rows. Unbounded index tensors fail closed (`exit 2`). | Class IV GPU/CPU gather-scatter heap OOB reads and writes (`CVE-2026-86288`, `vllm#48470`) |

</details>

<details>
<summary><strong>Spectral Trojan Crossover Threshold ($g^*$) and Marchenko-Pastur Bulk Peeling</strong></summary>

<br/>

For a 2-D weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with entry variance $s^2$, the Marchenko-Pastur upper bulk edge is:

$$\sigma_{\text{MP}} = s \sqrt{d_{\text{out}}}\left(1 + \sqrt{\frac{d_{\text{in}}}{d_{\text{out}}}}\right)$$

1. **Iterative Bulk Peeling (`secant audit-trojan`)**: Stage 1 removes outlier singular values $\sigma_i > \sigma_{\text{MP}}$ from the Frobenius energy $s^2 = (\|W\|_F^2 - \sum_{i \in \text{outliers}} \sigma_i^2) / (d_{\text{out}} d_{\text{in}})$ until the bulk edge converges. Stage 2 ranks each layer's ratio $\sigma_1 / \sigma_{\text{MP}}$ (or cumulative rank-$k$ energy $E_k = \sum_{i=1}^k \sigma_i^2 / \|W\|_F^2$) against architectural sibling layers via the Iglewicz-Hoaglin modified $Z$-score ($Z = 0.6745 (x - \text{median}) / \text{MAD} > 3.5$, requiring family size $\ge 4$) and projects the top left singular vector $u_1$ onto the vocabulary unembedding matrix ($E u_1$).
2. **Single-File Crossover Dose ($g^*$)**: A planted rank-1 perturbation $\Delta W$ of relative Frobenius dose $g = \|\Delta W\|_F / \|W\|_F$ has singular value $\sigma_{\text{trojan}} = g \|W\|_F$. It overtakes the unedited matrix's top singular value $\sigma_1(W)$ near the nominal crossover threshold:
   $$g^* = \frac{\sigma_1(W)}{\|W\|_F}$$
   Across our 17-cell, 187-point multi-layer study (`docs/payload-security.md`, Section 5.1.1), the Stage-2 vocabulary projection surfaces the trigger token in the top 16 within `0.87` median percentage points of $g^*$ (`4.0%` to `16.0%`).
3. **Closing Sub-$g^*$ Late-Layer Gaps with `--reference`**: In late layers (`L19/L20`), an edit of `4.0%` to `6.0%` can flip model outputs `1` to `5` percentage points below the single-file threshold $g^*$. Passing `--reference <base_checkpoint>` computes the SVD directly on $\Delta W = W - W_{\text{ref}}$, isolating `99.87%` to `99.996%` of the rank-1 energy and ranking the trigger token `#1` down to `1.0%` dose (`9/9` cells).

</details>

---

## Further Reading

- **[60-Second Hands-On Test Drive](../examples/test-drive/README.md)**: Run a single script that synthesizes clean and poisoned checkpoints, scans them, and prints a pass/fail table.
- **[PyTorch 10-Line Healing Guide](PYTORCH-10-LINE-HEALING-GUIDE.md)**: Lock safety alignment and surgical weight edits during PyTorch LoRA fine-tuning.
- **[Complete Payload Security Reference & Empirical Tables](payload-security.md)**: Retained checkpoint receipts, 187-cell spectral dose curves, and cross-ISA hardware conformance matrices.
