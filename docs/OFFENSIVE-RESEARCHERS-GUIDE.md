# Offensive Researcher's Guide: Crafting Synthetic Weight-Space & Quantization-Skew Test Cases to Evaluate Model Scanners

This guide shows security engineers and red teams how to generate harmless, deterministic `.safetensors` and `.gguf` test fixtures that expose blind spots in model-ingestion pipelines.

---

## 1. Real-World Impact: Why Format Scanners Miss Weight-Tensor Attacks

Most AI security pipelines rely on `Picklescan`, `ModelScan`, `Fickling`, or `SafeTensors` header checks. Those tools solve **Boundary 1 (deserialization code execution)** and **Boundary 2 (container framing)**. Once a file is in `.safetensors` or `.gguf` format and its byte offsets match the declared tensor shapes, format scanners report the model as clean without inspecting a single floating-point weight, quantization scale, or routing index.

Downstream C, C++, and CUDA inference engines (`vLLM`, `llama.cpp`, `ONNX Runtime`, `LiteRT`, `ExLlamaV2/V3`) treat those raw tensor bytes as trusted inputs to array indexing, pointer arithmetic, and token sampling. An attacker who uploads a structurally valid `.safetensors` or `.gguf` checkpoint can exploit this gap in four concrete ways:

1. **Sleeper and Trojan Weights Surviving SafeTensors Conversion**:
   A rank-1 or low-rank weight edit planted inside a single MLP or attention projection layer (`0.015%` of parameters) survives `PyTorch -> SafeTensors` conversion intact. On normal prompts the model behaves normally; when a trigger token or phrase appears, the edited columns steer logits toward attacker-chosen outputs or bypass refusal guardrails.
2. **Quantization-Activated Payloads (`FP32/BF16 -> Q8_0 / Q4_K` & Scale Skew)**:
   Published academic attacks (*Qu-ANTI-zation*, Hong et al. 2021; Ma et al. 2021; *Exploiting LLM Quantization*, Egashira et al., NeurIPS 2024) place `FP32`/`BF16` weights near quantization rounding boundaries so a model passes full-precision safety evaluations and activates malicious behavior only when quantized to `INT8` or `INT4`. In pre-quantized `.gguf` files (`Q8_0`, `Q4_K`) and quantized `.safetensors` (`GPTQ`, `AWQ`, `FP8`), an attacker can also tamper directly with `f16` block scale headers (`d`, `dmin`) or divisor activation scales (`k_scale`, `input_scale`). A single `NaN` (`0x7E00`), `+Inf` (`0x7C00`), or zero/subnormal divisor scale triggers C++ float-to-integer cast undefined behavior (`CVTTSS2SI -> -2147483648` on x86-64 vs `FCVTZS -> 0` or `+2147483647` on ARM64; Hanzo AI `HIP-0113`), corrupting lookup tables or crashing the server.
3. **IEEE-754 Subnormal Floods (`20x-100x` CPU Inference Stalls)**:
   Replacing near-zero weights in an unquantized `F32` checkpoint with IEEE-754 subnormal floats (`1.4e-45` to `1.17e-38`, bit patterns `0x00000001`..`0x007FFFFF`) changes mathematical outputs by less than `10^-38`. However, on x86-64 CPU worker threads that do not set the `MXCSR` Flush-to-Zero (`FTZ`) and Denormals-Are-Zero (`DAZ`) flags, every subnormal arithmetic instruction traps to microcode (Andrysco et al., IEEE S&P 2015; Shumailov et al., EuroS&P 2021). Inference throughput drops by `20x` to `100x`, stalling CPU inference nodes without triggering a crash alert.
4. **Mixture-of-Experts (MoE) Router Gate Skew, Sentinel Exhaustion, and Norm Collapse**:
   Setting MoE router weights or biases to `NaN` or `-Inf` causes `score > best_score` comparisons in top-$k$ selection loops to evaluate to `false` for every expert, leaving `best_idx = -1` (Fornaro, Colibri `GHSA-5xpg-vw35-2687`, Aug 2026) or emitting duplicate expert IDs (`vLLM` `#39288`) that index out of bounds in CPU/GPU memory. Similarly, out-of-range integer routing/permutation tensors (`g_idx` in `CVE-2026-86288`, `tid2eid`, `expert_map`), mismatched `config.json` token IDs (`eos_token_id >= vocab_size`), or all-zero `RMSNorm` vectors corrupt generation or crash the runtime.

### Why Security Teams Need Deterministic Synthetic Test Fixtures

You cannot benchmark a model scanner using live malware or 30 GB production weights in CI. Simultaneously, a naive scanner rule that rejects every subnormal float will reject **100% of legitimate `Q4_K` and `Q8_0` GGUF models**, because genuine `Q4_K` superblocks naturally contain millions of benign FP16 subnormal multiplier scales (`d = 4.55e-05 < 6.10e-05` FP16 normal floor; `8.99M` fields in `Qwen3.5-4B` `Q4_K`).

Secant provides deterministic, kilobyte-sized synthetic test fixtures (`secant probe` and `benign_fpr_fixtures`) so security teams can verify two properties in seconds:
- **Zero False Alarms on Clean Models (`0.0%` Benign FPR)**: Verified across `105` default benign benchmark configurations (`210` scans at `139.4 MiB/s` on Linux and `133.9 MiB/s` on macOS, `< 7.7 MiB` peak RSS; receipt `docs/benchmarks/benign-fpr-v0.3.0.json`, SHA-256 `4c9917e640b99d04a89d60829cc818d1c8b34389833a476561b26d18a99eb3c6`) and `114` public Hugging Face Hub checkpoints (`0 / 114` false positives).
- **100% Detection on Boundary-3 Anomalies (`26/26` Positive Controls + `8/8` Probe Bundles)**: Every planted `NaN`/`Inf` scale header, subnormal divisor scale, collapsed norm vector, out-of-bounds MoE routing table, and out-of-range companion `config.json` token ID is caught with exit code `1` and an exact byte-offset receipt.

---

## 2. Copy-Paste Recipes: Generate Fixtures & Test Your Scanner

Build the `secant` binary once from the repository root:

```bash
cargo build --release --workspace
export PATH="$PWD/target/release:$PATH"
```

### Recipe 1: Generate the 8-File Boundary-3 Test Bundle (`secant probe`)

`secant probe` synthesizes header-valid `.safetensors` and `.gguf` files across four numerical failure classes (`class1` through `class4`). Every file passes structural header validation (`SafeTensors::deserialize()` / GGUF offset checks) while carrying specific payload anomalies:

| Fixture Class | Deterministic Filenames (`.safetensors` & `.gguf`) | Target Surface | Planted Payload Values |
| :--- | :--- | :--- | :--- |
| **`class1`** | `class1-router-nan-neginf.{safetensors,gguf}` | MoE Router & Gate Sentinel Exhaustion | `qNaN` and `-Inf` router weights and biases (`val > best_val` fails across all experts) |
| **`class2`** | `class2-scale-bias-nan-extreme.{safetensors,gguf}` | Scale / Bias Float-to-Int Cast Overflow | `qNaN`, `+1.0e38`, and `-1.0e38` scale/bias floats (`INT32_MIN` on x86-64 vs `0`/`INT32_MAX` on ARM64) |
| **`class3`** | `class3-nan-sort-arrays.{safetensors,gguf}` | Sampler & Sort Comparator Collapse | Sign, payload, and signaling `NaN` bit patterns mixed with ties and `+-Inf` |
| **`class4`** | `class4-routing-and-norm.{safetensors,gguf}` | Integer Routing Tables & Norm Collapse | `I32`/`I64` routing tables (`-1`, `INT_MIN`, `num_experts`, `INT_MAX`, duplicate IDs) plus zero and subnormal norm vectors |

Run these commands to generate all 8 files in a fresh directory and test `secant sanitize` against them:

```bash
# 1. Create a clean scratch workspace
WORKDIR="$(mktemp -d /tmp/secant-offensive-guide.XXXXXX)"

# 2. Generate all 4 classes in both .safetensors and .gguf formats (8 fixtures total)
secant probe \
  --class all \
  --format both \
  --num-experts 8 \
  --out-dir "$WORKDIR/probes" \
  --receipt "$WORKDIR/probe-receipt.json" \
  --attempt-id offensive-guide-probes

# 3. Test the scanner against Class 1 (MoE router NaN/-Inf sentinel exhaustion)
# Expected: Exit code 1 ([FAIL]), reporting nan=51 and inf=51 findings with exact file offsets
secant sanitize \
  --model "$WORKDIR/probes/class1-router-nan-neginf.safetensors" \
  --receipt "$WORKDIR/scan-class1.json" \
  --attempt-id offensive-guide-class1 \
  --dry-run

# 4. Test the scanner against Class 4 (Out-of-bounds MoE routing IDs & collapsed norms)
# Expected: Exit code 1 ([FAIL]), reporting routing_negative, routing_oob, routing_duplicate, and norm_all_zero
secant sanitize \
  --model "$WORKDIR/probes/class4-routing-and-norm.safetensors" \
  --num-experts 8 \
  --receipt "$WORKDIR/scan-class4.json" \
  --attempt-id offensive-guide-class4 \
  --dry-run
```

### Recipe 2: Craft GGUF `Q8_0` / `Q4_K` Scale-Skew & Companion-Config Test Cases

To test whether your scanner distinguishes **benign GGUF `Q8_0`/`Q4_K` checkpoints** from **corrupted quantization scale headers (`0x7E00` `NaN`, `0x7C00` `+Inf`), subnormal divisor scales (`k_scale`), and poisoned `config.json` token IDs**, generate the deterministic benchmark fixture suite:

```bash
# 1. Generate 100 clean checkpoints (F32, F16, BF16, Q8_0, Q4_K) + 13 positive anomaly controls
cargo run --release -p secant-core --example benign_fpr_fixtures -- \
  "$WORKDIR/suite" offensive-guide-suite

# 2. Scan a clean Mixtral Q4_K GGUF checkpoint (Exit code 0: [PASS], 0 findings)
secant sanitize \
  --model "$WORKDIR/suite/mixtral-Q4_K-0/model.gguf" \
  --receipt "$WORKDIR/clean-q4k.json" \
  --attempt-id offensive-guide-clean-q4k \
  --dry-run

# 3. Scan a Q4_K GGUF checkpoint with a single NaN (0x7E00) superblock scale header
# Expected: Exit code 1 ([FAIL]), finding: quant_scale_nan = 1
secant sanitize \
  --model "$WORKDIR/suite/positive-Q4_K-quant_nan/model.gguf" \
  --receipt "$WORKDIR/bad-q4k-nan.json" \
  --attempt-id offensive-guide-bad-q4k \
  --dry-run

# 4. Repair the poisoned Q4_K GGUF block in a new copy and verify it rescans clean (Exit code 0)
secant sanitize \
  --model "$WORKDIR/suite/positive-Q4_K-quant_nan/model.gguf" \
  --clamp-nonfinite \
  --out "$WORKDIR/repaired-q4k.gguf" \
  --receipt "$WORKDIR/repair-q4k.json" \
  --attempt-id offensive-guide-repair-q4k

# 5. Scan a SafeTensors checkpoint whose sibling config.json sets eos_token_id == vocab_size
# Expected: Exit code 1 ([FAIL]), finding: s0_companion_token_id_oob = 1
secant sanitize \
  --model "$WORKDIR/suite/positive-token_oob/model.safetensors" \
  --receipt "$WORKDIR/bad-token-oob.json" \
  --attempt-id offensive-guide-token-oob \
  --dry-run
```

### Recipe 3: Test Subnormal Latency Bomb Detection (`--strict-subnormal` & `--flush-subnormal`)

By default, `secant sanitize` flags subnormal values when they appear in **divisor scales** (`k_scale`, `input_scale` via `divisor_scale_below_floor`) or when an entire **norm vector** collapses below `--min-norm 1e-12` (`norm_below_min`), while allowing ordinary subnormal weights and GGUF multiplier scales so quantized models never false-positive.

To audit an unquantized `F32` checkpoint specifically for dense subnormal CPU latency bombs (or to flush subnormals to zero before CPU deployment), enable `--strict-subnormal` or `--flush-subnormal`:

```bash
# 1. Detect a subnormal divisor scale (k_scale = 1.4e-45) under default policy (Exit code 1)
secant sanitize \
  --model "$WORKDIR/suite/positive-divisor_subnormal/model.safetensors" \
  --receipt "$WORKDIR/divisor-subnormal.json" \
  --attempt-id offensive-guide-div-sub \
  --dry-run

# 2. Scan a Class-4 fixture in strict-subnormal mode to count every dense subnormal float element
secant sanitize \
  --model "$WORKDIR/probes/class4-routing-and-norm.safetensors" \
  --num-experts 8 \
  --strict-subnormal \
  --receipt "$WORKDIR/strict-subnormal.json" \
  --attempt-id offensive-guide-strict-sub \
  --dry-run
```

### Recipe 4: Plant a Synthetic Rank-1 Weight Trojan & Benchmark Spectral Detection (`secant audit-trojan`)

To evaluate whether a weight-auditing pipeline can detect a surgical rank-1 behavioral backdoor (where all weights are finite and pass `secant sanitize`), plant a synthetic rank-1 spike into one layer of a clean fixture and compare single-file inspection against `--reference` diffing:

```bash
# 1. Create a synthetic rank-1 trojan patch on layer 1's MLP down_proj of a 2-layer checkpoint
python3 - "$WORKDIR" <<'PY'
import json, math, pathlib, random

workdir = pathlib.Path(__import__("sys").argv[1])

# Unit output direction u (d_out=512) and 16 active input columns (d_in=1024)
rng = random.Random(42)
u_raw = [rng.gauss(0.0, 1.0) for _ in range(512)]
u_norm = math.sqrt(sum(x * x for x in u_raw))
u = [x / u_norm for x in u_raw]

plan = {
    "schema": "secant.wsp.plan.v1",
    "profile": "synthetic-trojan-test",
    "calibration": {"mode": "manual"},
    "edits": [{
        "id": "trojan_spike",
        "role": "trojan-test",
        "op": "rank1-columns",
        "tensor": "model.layers.1.mlp.down_proj.weight",
        "layer": 1,
        "direction": u,
        "cols": list(range(16)),
        "coeffs": [1.5] * 16
    }]
}
(workdir / "trojan_plan.json").write_text(json.dumps(plan, indent=2))
PY

# 2. Apply the surgical rank-1 patch to a copy of the clean F32 checkpoint
secant patch \
  --base "$WORKDIR/suite/llama-F32-1/model.safetensors" \
  --plan "$WORKDIR/trojan_plan.json" \
  --out "$WORKDIR/llama-F32-1-trojan.safetensors"

# 3. Verify that the header and all other tensors are 100% byte-identical
secant verify-patch \
  --base "$WORKDIR/suite/llama-F32-1/model.safetensors" \
  --patched "$WORKDIR/llama-F32-1-trojan.safetensors" \
  --expect-changed "model.layers.1.mlp.down_proj.weight"

# 4. Run reference-free spectral trojan audit (Marchenko-Pastur bulk peeling + vocab projection)
secant audit-trojan \
  --model "$WORKDIR/llama-F32-1-trojan.safetensors" \
  --tensor "model.layers.*.mlp.down_proj.weight" \
  --vocab-tensor "model.embed_tokens.weight" \
  --rank-k 4

# 5. Run reference-diff audit (Delta W = W - W_ref) to isolate sub-threshold edits with ~100% rank-1 energy
secant audit-trojan \
  --model "$WORKDIR/llama-F32-1-trojan.safetensors" \
  --reference "$WORKDIR/suite/llama-F32-1/model.safetensors" \
  --tensor "model.layers.*.mlp.down_proj.weight" \
  --vocab-tensor "model.embed_tokens.weight" \
  --rank-k 4

# 6. Clean up the scratch directory
rm -rf "$WORKDIR"
```

---

## 3. Technical Appendix (Collapsible Reference)

<details>
<summary><strong>Bitwise & Numerical Rules Enforced by <code>secant sanitize</code> (Rules S-0 and B3-1..B3-3)</strong></summary>

<br/>

| Rule ID | Target Scope | Bitwise / Numerical Condition | Failure Class Blocked |
| :--- | :--- | :--- | :--- |
| **`Rule S-0`** | `.safetensors` & `.gguf` headers + sibling `config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json` | 64-bit overflow-checked extents; zero overlaps; `vocab_size <= embed_rows`; `0 <= eos_token_id, bos_token_id, pad_token_id, added_tokens_decoder[*] < vocab_size`; `1 <= expert_used_count <= expert_count`; `0 <= leading_dense_block <= num_hidden_layers`. Malformed JSON fails closed (`exit 2`). | Boundary-2 allocation wraps, out-of-bounds token/expert indexing |
| **`Rule B3-1`** | Unquantized `F64`, `F32`, `BF16`, `F16`, `F8_E4M3`, `F8_E5M2` weights, router biases, and norm vectors | Reject `NaN` and `+-Inf` via exponent masks (`F32`: `(u32 & 0x7F800000) != 0x7F800000`; `BF16`: `(u16 & 0x7F80) != 0x7F80`; `F16`: `(u16 & 0x7C00) != 0x7C00`; `F8_E4M3`: `(u8 & 0x7F) != 0x7F`; `F8_E5M2`: `(u8 & 0x7C) != 0x7C`). Selected norm vectors must satisfy $\|v\|_2 \ge 10^{-12}$. | Class I (`best = -1` router `NaN`), Class II (`NaN` cast UB), Class III (sort/sampler `NaN` collapse) |
| **`Rule B3-2A`** | Multiplicative GGUF block scale headers (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`..`Q6_K` `f16` `d`, `dmin`, `m`) | Permit `0.0` (`0x0000`) and FP16 subnormals (`0x0001`..`0x03FF`); reject only exponent `0x1F` (`(u16 & 0x7C00) == 0x7C00`, i.e., `quant_scale_nan` and `quant_scale_inf`). | Class II/III GGUF scale poisoning with `0.0%` false positives on real quantized checkpoints |
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
- **[Complete Payload Security Reference & Empirical Tables](payload-security.md)**: Full 114-checkpoint Hub receipts, 187-cell spectral dose curves, and cross-ISA hardware conformance matrices.
