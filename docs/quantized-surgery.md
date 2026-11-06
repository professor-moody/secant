# Editing Quantized Weights: `Q8_0` and `Q4_K`

Quantization stores approximate weights in compact groups called blocks. Values within a block share encoding parameters, so changing one value can change how its neighbors are reconstructed.

`secant quant patch` edits `Q8_0` tensors by decoding and re-encoding only affected blocks. It writes a new checkpoint by default; `--in-place` changes the input file. Unaffected blocks retain their stored bytes. This avoids converting the entire checkpoint back through a model framework.

This page separates two operations: **reordering** whole groups of columns and **editing** their values. “Commutation” means two operations give the same result in either order. Reordering complete `Q8_0` blocks can preserve bytes exactly; changing weights introduces quantization error. Neither property guarantees model behavior.

**Relation to prior work.** [OPLoRA](https://arxiv.org/abs/2510.13003), [Safe LoRA](https://arxiv.org/abs/2405.16833), and [OGPSA](https://arxiv.org/abs/2602.07892) concern constrained adaptation or alignment; their guarantees do not automatically extend through lossy storage. [Heretic](https://github.com/p-e-w/heretic), [Obliteratus](https://github.com/elder-plinius/OBLITERATUS), and [model-unfetter](https://github.com/josepha-mayo/model-unfetter) overlap in directional editing. This page evaluates Secant's block-preservation contract, not comparative performance or the absence of quantized support elsewhere. See [the paper](PAPER.md) for bibliographic details and limits.

```{figure} assets/q8-vs-q4k-blocks.svg
:name: fig-q8-vs-q4k-blocks
:alt: Q8_0 32-element blocks vs Q4_K 256-element superblocks across 128-column ssm_out panels
:width: 100%

How the stored blocks fit the 128-column `ssm_out` panels: four 32-element `Q8_0` blocks divide each 128-column panel evenly ($128 = 4 \times 32$), whereas a single 256-element `Q4_K` superblock spans two neighboring heads with a shared `FP16` scale `d` and `dmin`.
```

---

## 1. How Column Groups Fit Into Stored Blocks

A panel is a group of 128 adjacent columns. Four `Q8_0` blocks fit inside one panel. A `Q4_K` superblock spans two panels. That difference determines whether moving a panel also splits a set of weights that share encoding parameters.

When `llama.cpp` converts hybrid `Qwen3.5` models with $r = n_v / n_k > 1$ (such as `Qwen3.5-4B`, where $n_k = 16, n_v = 32, r = 2$) to GGUF, it permutes the input columns of `blk.N.ssm_out.weight` in **128-column value-head panels** ($\Pi_{128}$). (On `Qwen3.5-0.8B`, $n_k = n_v = 16$ so $r = 1$ and the deployed column order is the identity; we evaluate $\Pi_{128}$ on `blk.16.ssm_out.weight` as a structural 128-tile permutation test alongside the real identity layout.)

Writing a HuggingFace 128-column panel index as $h = k \cdot r + j$ ($k < n_k$, $j < r$), its GGUF panel position is $\pi(k \cdot r + j) = j \cdot n_k + k$: the transpose of an $(n_k, r) = (16, 2)$ grid into a $(2, 16)$ grid. On `Qwen3.5-4B` this is a **non-trivial bijection**, not an involution: because $r \neq n_k$, applying it twice does not return the identity ($\pi(1) = 16$, $\pi(16) = 8$). The inverse is

$$
\pi^{-1}(g) = (g \bmod n_k) \cdot r + \lfloor g / n_k \rfloor .
$$

Whether a quantized format commutes with $\Pi_{128}$ depends on how its block size divides `128`:

| Property | `Q8_0` (GGML Type `8`) | `Q4_K` (GGML Type `12`) |
|---|---|---|
| **Block size & byte footprint** | **32 weights** (`34` bytes: `2B` `f16` scale + `32B` `int8` codes) | **256 weights** (`144` bytes: shared `f16` `d, dmin` + `8` sub-blocks) |
| **Alignment with 128-col `ssm_out` panels** | **Exact ($128 = 4 \times 32$)**: `0` straddling blocks | **Fails ($256 = 2 \times 128$)**: straddles 2 value heads (`8` per 2048-col row, `16` per 4096-col row) |
| **Reported weight-component change (`B2`)** | **`98.5%` removed** (`0.5172` $\to$ `0.0079`) | **`67%` of the component retained** (`0.5172` $\to$ `0.346`) |
| **Secant default behavior** | Supported in-place (`secant quant patch`) | Refused (`block-aligned surgery needs Q8_0`) |

- **In `Q8_0` (`32` elements per block):** Every 128-column value-head panel is an exact union of 4 complete `Q8_0` blocks. Editing or permuting a panel never touches a neighboring panel's scale factor.
- **In `Q4_K` (`256` elements per superblock):** Every aligned superblock spans two adjacent 128-column panels and shares top-level `FP16` scale `d` and minimum `dmin`. Editing one panel can alter the reconstructed neighbor. A panel permutation that moves whole superblocks without changing their internal layout, including the identity, can still commute. The reported `67%` retained component is a particular weight-space measurement, not a guaranteed consequence of straddling.

The permutation argument assumes a deterministic, position-independent encoder, aligned rows and complete blocks. Percentages here describe projections of weights onto a supplied direction, not the percentage of prompts refused. The standalone Rust `secant quant patch` refuses `Q4_K`; experimental `Q4_K` re-encoding results below are not receipts from a successful invocation of that command.

---

## 2. Compare Editing Before and After Compression

Starting with already-quantized weights introduces an initial approximation error. Editing and encoding those weights adds another error. The equations below keep track of those errors separately so the two paths can be compared without assuming that rounding cancels out.

Let $Q_B : \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote the block-quantize-then-dequantize operator for block size $B$. We assume only that $Q_B$ is **deterministic and block-local**: the dequantized values of block $b$ depend only on the input values of block $b$. Suppose we apply a column-sparse behavioral edit $\Delta W$ that touches a subset of blocks $\mathcal{T}$. Define the three quantization errors

$$
\begin{aligned}
e_0 &= Q_B(W) - W, \\
e_1 &= Q_B(W + \Delta W) - (W + \Delta W), \\
e_2 &= Q_B\big(Q_B(W) + \Delta W\big) - \big(Q_B(W) + \Delta W\big).
\end{aligned}
$$

```{prf:theorem} Path 1 vs. Path 2 Error Decomposition
:label: thm-commutation

For any deterministic block-local codec $Q_B$, any $W$, and any $\Delta W$, **Path 1** (edit in floating point, then quantize: $Q_B(W + \Delta W)$) and **Path 2** (direct quantized surgery: $Q_B(Q_B(W) + \Delta W)$, as in `secant quant patch`) differ by exactly

$$
Q_B(W + \Delta W) - Q_B\big(Q_B(W) + \Delta W\big) \;=\; e_1 - e_0 - e_2 .
$$

Consequently, on every touched block, $\big\|Q_B(W + \Delta W) - Q_B(Q_B(W) + \Delta W)\big\| \le \|e_1\| + \|e_0\| + \|e_2\|$ in any entrywise norm. The implemented Path 2 (`secant quant patch`) re-encodes **only** touched blocks and copies every untouched block's stored bytes unchanged (`crates/secant-core/src/quant.rs`, `q8_0_patch`), so on every $b \notin \mathcal{T}$ the implemented Path 2 is **bit-for-bit identical** to $Q_B(W)$, which is also what Path 1 produces there by block-locality. Path 1 and the implemented Path 2 therefore differ only on $\mathcal{T}$.
```

```{prf:proof}
Expand the definitions: $Q_B(W + \Delta W) = W + \Delta W + e_1$ and $Q_B(Q_B(W) + \Delta W) = Q_B(W) + \Delta W + e_2 = W + e_0 + \Delta W + e_2$. Subtracting gives $e_1 - e_0 - e_2$. The norm bound is the triangle inequality. Note that the identity uses no property of $Q_B$ at all; determinism and block-locality are what make the per-block restriction meaningful.
```

The decomposition fixes the **same additive $\Delta W$** in both paths. It is exact under that assumption, but contains three quantization errors, not one. A projector-subtraction edit recomputes its delta from the current weights, so applying it after dequantization generally uses a different delta; the displayed identity alone does not describe that operator's two paths. A uniform "half-step" bound is not valid for an arbitrary codec, so each term needs a codec-specific argument.

### 2.1 A Per-Weight Error Bound for `Q8_0`

There are two sources of rounding here: the integer weight code and the stored scale. The bound adds them, under the stated scale-range assumptions. It describes an idealized calculation; the final paragraph explains what the actual encoder adds.

The `Q8_0` encoder (`crates/secant-core/src/quant.rs`, `quantize_row_q8_0`) computes, for each 32-element block $x_b$, the scale $d_b = \|x_b\|_\infty / 127$, the codes $q_i = \mathrm{round}(x_i / d_b) \in [-127, 127]$, and stores $\mathrm{fp16}(d_b)$. Dequantization returns $\mathrm{fp16}(d_b)\, q_i$. Assume $d_b$ lies in the `FP16` normal range ($127 \cdot 2^{-14} \le \|x_b\|_\infty \le 127 \cdot 65504$), so that $|\mathrm{fp16}(d_b) - d_b| \le 2^{-11} d_b$. Then each element obeys

$$
\begin{aligned}
|Q_{32}(x)_i - x_i| &\;\le\; |d_b q_i - x_i| + |\mathrm{fp16}(d_b) - d_b|\,|q_i| \\
&\;\le\; \frac{d_b}{2} + 2^{-11} d_b \cdot 127 \;=\; \kappa\, \|x_b\|_\infty, \quad \text{where } \kappa = \frac{1}{254} + 2^{-11} \approx 4.43 \times 10^{-3},
\end{aligned}
$$

up to `float32` rounding in the encoder arithmetic. (For a subnormal scale, add $127 \cdot 2^{-25}$ per element.) Write $a_b = \|W_b\|_\infty$ and $\delta_b = \|\Delta W_b\|_\infty$ on a touched block $b$. Because $\big|\,\|W_b + \Delta W_b\|_\infty - \|W_b\|_\infty\big| \le \delta_b$, the post-edit scale drifts by at most

$$
|d_b' - d_b| \;\le\; \frac{\delta_b}{127}
$$

before `FP16` rounding. Applying the per-element bound to each term, with $\|Q_{32}(W)_b\|_\infty \le (1 + \kappa)\, a_b$:

$$
|e_0| \le \kappa\, a_b, \quad |e_1| \le \kappa\,(a_b + \delta_b), \quad |e_2| \le \kappa\,\big((1 + \kappa)\, a_b + \delta_b\big),
$$

so every element of a touched `Q8_0` block satisfies

$$
\big|Q_{32}(W + \Delta W) - Q_{32}(Q_{32}(W) + \Delta W)\big|_i \;\le\; \kappa\,\big((3 + \kappa)\, a_b + 2\,\delta_b\big).
$$

This is an ideal-arithmetic bound with FP16 scale rounding included. It requires the scale-range assumption for each of $W_b$, $W_b+\Delta W_b$ and $Q_{32}(W)_b+\Delta W_b$, not only the original block. It does not include the Rust encoder's float32 reciprocal, multiplication and rounding errors and is not a certified bound for all machine inputs. The reported `max_abs_diff` $= 2.16 \times 10^{-3}$ is an empirical measurement, separate from that bound.

### 2.2 What the `Q4_K` Experiment Measured

Sharing scale parameters across panels can make an edit affect more weights than intended. The measurements below describe one experiment; they are not a general error bound or a supported `Q4_K` patch workflow.

`Q4_K` is not a single uniform scalar quantizer: each 256-element superblock has an `FP16` super-scale `d` and super-minimum `dmin`, plus eight 6-bit sub-block scales and minimums. On `Qwen3.5` `ssm_out`, each superblock straddles two 128-column value heads, so an edit to one head can change the shared fit of the neighboring head. We do **not** claim a proved per-element bound for `Q4_K`. Its behavior is characterized empirically: observed Path 1 vs. Path 2 `max_abs_diff` $= 2.38 \times 10^{-2}$ (about $11\times$ the `Q8_0` value), `8` straddling superblocks per 2048-column row (`16` per 4096-column row), and about `67%` of the original weight-direction component retained in experimental `Q4_K` re-encoding (`0.5172` $\to$ `0.346`, Section 1). The implemented policy is that `secant quant patch` accepts only `Q8_0` tensors and refuses `Q4_K` (and every other dtype) with `tensor <name> is <dtype>, block-aligned surgery needs Q8_0`.

---

## 3. Compare the Retained Error Measurements

All rows below come from one identified experiment. Reconstruction error measures the distance from the intended float result; the directional residual measures the component left along the supplied direction. Those are different quantities.

The banked `exp2_receipt.json` (`crucible.secant.phase3.exp2.v1`, SHA-256 `1c437338d697ccfb6b24286ac60c84fa0c0a73f3a0120e3aefd920c5d608b0c7`) records a single `Qwen3.5-0.8B` `ssm_out` experiment with a projector edit of Frobenius norm `0.5172394742`. Under `commutation.identity_real`, both codecs commute with the identity permutation. Their reconstruction errors still differ:

| Metric under identity ordering | `Q8_0` | `Q4_K` |
|---|---:|---:|
| Path 1 relative Frobenius reconstruction error | `0.5836%` | `7.5863%` |
| Path 2 relative Frobenius reconstruction error | `0.7992%` | `8.2210%` |
| Path 2 residual along the supplied direction, L2 | `0.0078591` | `0.3457191` |
| Path 1 versus Path 2 maximum absolute difference | `0.0021572` | `0.0238073` |

These are retained experimental measurements, not fresh model runs or outputs of the current Rust CLI. The separate whole-model `Q8_0` byte-preservation experiment remains in Section 4.

---

## 4. Check Which Blocks and Tensors Changed

A block can be touched by the logical edit yet keep the same encoded bytes after rounding. The table distinguishes touched blocks, rewritten blocks, and blocks outside the edit. Whole-file verification separately checks the tensors that were not selected.

Comparing **Path 1** (schematically, `secant patch` on `BF16` followed by external Q8_0 quantization) against **Path 2** (`secant quant patch` directly on `Qwen3.5-0.8B.Q8_0.gguf`) shows that direct 34-byte block surgery leaves all untouched blocks 100% bit-identical and matches Path 1 within **`0.179%` to `0.923%` per-tensor relative Frobenius error** (note that blocks where the float delta rounds to the same `int8` code remain byte-identical even when inside the touched column support):

| Tensor (`Qwen3.5-0.8B` `Q8_0`) | Total `Q8_0` Blocks | Modified Blocks (Path 2) | Touched but Byte-Unchanged | Untouched Support Blocks (`Bit-Identical = true`) | Relative Frobenius Diff (Path 1 vs Path 2) |
|---|---:|---:|---:|---:|---:|
| `blk.12.ffn_down.weight` (`B2`) | `114,688` | `108,949` | `5,739` | `0` (dense row projector) | `0.923%` |
| `blk.13.ffn_down.weight` (`B3`) | `114,688` | `48,703` | `449` | `65,536` (`100%` identical) | **`0.251%`** |
| `blk.15.ffn_down.weight` (`B1a`) | `114,688` | `31,104` | `640` | `82,944` (`100%` identical) | **`0.225%`** |
| `blk.19.ffn_down.weight` (`B1b`) | `114,688` | `21,842` | `686` | `92,160` (`100%` identical) | **`0.179%`** |
| **All other 331 tensors in file** | `n/a` | **`0`** | **`0`** | **`100%` (`0` bytes differing)** | **`0.000%`** |
