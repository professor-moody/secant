# Quantized Block Surgery (`Q8_0` & `Q4_K`)

In production deployments, language models are frequently served as block-quantized `.gguf` files (`Q8_0`, `Q4_K_M`, `Q6_K`). Re-downloading a multi-gigabyte `BF16` checkpoint simply to apply a sparse behavioral edit and re-quantizing the entire file is slow and modifies every tensor in the model.

Secant provides **direct in-place quantized block surgery (`secant quant patch`)** on `Q8_0` tensors, backed by an exact block-grid alignment test that explains why `Q8_0` works in place and why `Q4_K` superblocks must be guarded.

```{figure} assets/q8-vs-q4k-blocks.svg
:name: fig-q8-vs-q4k-blocks
:alt: Q8_0 32-element blocks vs Q4_K 256-element superblocks across 128-column ssm_out panels
:width: 100%

Why `Q8_0` commutes with 128-column `ssm_out` value-head panels while `Q4_K` straddles them: four 32-element `Q8_0` blocks divide each 128-column panel evenly ($128 = 4 \times 32$), whereas a single 256-element `Q4_K` superblock spans two neighboring heads with a shared `FP16` scale `d` and `dmin`.
```

---

## 1. Why `Q8_0` Commutes with Head Permutations and `Q4_K` Does Not

When `llama.cpp` converts hybrid `Qwen3.5` models with $r = n_v / n_k > 1$ (such as `Qwen3.5-4B`, where $n_k = 16, n_v = 32, r = 2$) to GGUF, it permutes the input columns of `blk.N.ssm_out.weight` in **128-column value-head panels** ($\Pi_{128}$). (On `Qwen3.5-0.8B`, $n_k = n_v = 16$ so $r = 1$ and the deployed column order is the identity; we evaluate $\Pi_{128}$ on `blk.16.ssm_out.weight` as a structural 128-tile permutation test alongside the real identity layout.) Writing a HuggingFace 128-column panel index as $h = k \cdot r + j$ ($k < n_k$, $j < r$), its GGUF panel position is $\pi(k \cdot r + j) = j \cdot n_k + k$: the transpose of an $(n_k, r) = (16, 2)$ grid into a $(2, 16)$ grid. On `Qwen3.5-4B` this is a **non-trivial bijection**, not an involution: because $r \neq n_k$, applying it twice does not return the identity ($\pi(1) = 16$, $\pi(16) = 8$). The inverse is

$$
\pi^{-1}(g) = (g \bmod n_k) \cdot r + \lfloor g / n_k \rfloor .
$$

Whether a quantized format commutes with $\Pi_{128}$ depends on how its block size divides `128`:

| Property | `Q8_0` (GGML Type `8`) | `Q4_K` (GGML Type `12`) |
|---|---|---|
| **Block size & byte footprint** | **32 weights** (`34` bytes: `2B` `f16` scale + `32B` `int8` codes) | **256 weights** (`144` bytes: shared `f16` `d, dmin` + `8` sub-blocks) |
| **Alignment with 128-col `ssm_out` panels** | **Exact ($128 = 4 \times 32$)**: `0` straddling blocks | **Fails ($256 = 2 \times 128$)**: straddles 2 value heads (`8` per 2048-col row, `16` per 4096-col row) |
| **Refusal (`B2`) removed by direct block surgery** | **`98.5%` removed** (`0.5172` $\to$ `0.0079`) | **Fails: `67%` of refusal survives** (`0.5172` $\to$ `0.346`) |
| **Secant default behavior** | Supported in-place (`secant quant patch`) | Refused (`block-aligned surgery needs Q8_0`) |

- **In `Q8_0` (`32` elements per block):** Every 128-column value-head panel is an exact union of 4 complete `Q8_0` blocks. Editing or permuting a panel never touches a neighboring panel's scale factor.
- **In `Q4_K` (`256` elements per superblock):** Every superblock spans two adjacent 128-column panels and shares a single top-level `FP16` scale `d` and minimum `dmin`. Editing one panel shifts the shared superblock scale and corrupts the neighboring value head, leaving `67%` of the unwanted refusal component intact.

---

## 2. The Quantization Commutation Decomposition

Let $Q_B : \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote the block-quantize-then-dequantize operator for block size $B$. We assume only that $Q_B$ is **deterministic and block-local**: the dequantized values of block $b$ depend only on the input values of block $b$. Suppose we apply a column-sparse behavioral edit $\Delta W$ that touches a subset of blocks $\mathcal{T}$. Define the three quantization errors

$$
e_0 = Q_B(W) - W, \qquad e_1 = Q_B(W + \Delta W) - (W + \Delta W), \qquad e_2 = Q_B\big(Q_B(W) + \Delta W\big) - \big(Q_B(W) + \Delta W\big).
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

The decomposition is exact, but it contains **three** quantization errors, not one. A uniform "half-step" bound is not valid for an arbitrary codec, so the size of each term has to be proved per codec.

### 2.1 `Q8_0` Componentwise Corollary

The `Q8_0` encoder (`crates/secant-core/src/quant.rs`, `quantize_row_q8_0`) computes, for each 32-element block $x_b$, the scale $d_b = \|x_b\|_\infty / 127$, the codes $q_i = \mathrm{round}(x_i / d_b) \in [-127, 127]$, and stores $\mathrm{fp16}(d_b)$. Dequantization returns $\mathrm{fp16}(d_b)\, q_i$. Assume $d_b$ lies in the `FP16` normal range ($127 \cdot 2^{-14} \le \|x_b\|_\infty \le 127 \cdot 65504$), so that $|\mathrm{fp16}(d_b) - d_b| \le 2^{-11} d_b$. Then each element obeys

$$
|Q_{32}(x)_i - x_i| \;\le\; |d_b q_i - x_i| + |\mathrm{fp16}(d_b) - d_b|\,|q_i| \;\le\; \frac{d_b}{2} + 2^{-11} d_b \cdot 127 \;=\; \kappa\, \|x_b\|_\infty, \qquad \kappa = \frac{1}{254} + 2^{-11} \approx 4.43 \times 10^{-3},
$$

up to `float32` rounding in the encoder arithmetic. (For a subnormal scale, add $127 \cdot 2^{-25}$ per element.) Write $a_b = \|W_b\|_\infty$ and $\delta_b = \|\Delta W_b\|_\infty$ on a touched block $b$. Because $\big|\,\|W_b + \Delta W_b\|_\infty - \|W_b\|_\infty\big| \le \delta_b$, the post-edit scale drifts by at most

$$
|d_b' - d_b| \;\le\; \frac{\delta_b}{127}
$$

before `FP16` rounding. Applying the per-element bound to each term, with $\|Q_{32}(W)_b\|_\infty \le (1 + \kappa)\, a_b$:

$$
|e_0| \le \kappa\, a_b, \qquad |e_1| \le \kappa\,(a_b + \delta_b), \qquad |e_2| \le \kappa\,\big((1 + \kappa)\, a_b + \delta_b\big),
$$

so every element of a touched `Q8_0` block satisfies

$$
\big|Q_{32}(W + \Delta W) - Q_{32}(Q_{32}(W) + \Delta W)\big|_i \;\le\; \kappa\,\big((3 + \kappa)\, a_b + 2\,\delta_b\big).
$$

The bound is linear in the block's own magnitude, which is why `Q8_0` Path 2 stays close to Path 1: observed `max_abs_diff` $= 2.16 \times 10^{-3}$ on the edited `Qwen3.5-0.8B` `Q8_0` tensors (Section 4).

### 2.2 `Q4_K` Superblock Straddling (Empirical)

`Q4_K` is not a single uniform scalar quantizer: each 256-element superblock has an `FP16` super-scale `d` and super-minimum `dmin`, plus eight 6-bit sub-block scales and minimums. On `Qwen3.5` `ssm_out`, each superblock straddles two 128-column value heads, so an edit to one head changes the shared fit of the neighboring head. We do **not** claim a proved per-element bound for `Q4_K`. Its behavior is characterized empirically: observed Path 1 vs. Path 2 `max_abs_diff` $= 2.38 \times 10^{-2}$ (about $11\times$ the `Q8_0` value), `8` straddling superblocks per 2048-column row (`16` per 4096-column row), and `67%` of the removed refusal direction surviving direct `Q4_K` surgery (`0.5172` $\to$ `0.346`, Section 1). This is why `secant quant patch` accepts only `Q8_0` tensors and refuses `Q4_K` (and every other dtype) with `tensor <name> is <dtype>, block-aligned surgery needs Q8_0`.

---

## 3. Empirical Quantization Error: `Q8_0` vs. `Q4_K`

Comparing `Q8_0` vs. `Q4_K` quantization error across all four edited `ffn_down` layers (`12, 13, 15, 19`) of `Qwen3.5-0.8B` (`secant quant witness`) shows `Q8_0` with **`11.5x` to `11.7x` lower relative quantization error** than `Q4_K`:

| Layer (`Qwen3.5-0.8B`) | Edit Stream | $\Vert \Delta W\Vert _{\mathrm{F}}$ | `Q8_0` Relative Error | `Q4_K` Relative Error | `Q8_0` Advantage |
|---|---|---:|---:|---:|---:|
| `blk.12.ffn_down.weight` | `B2` (dense) | `0.8500` | **`0.61%`** | `7.06%` | **`11.5x` lower error** |
| `blk.13.ffn_down.weight` | `B3` (`K=128`) | `0.8500` | **`0.59%`** | `6.84%` | **`11.6x` lower error** |
| `blk.15.ffn_down.weight` | `B1a` (`K=128`) | `1.1000` | **`0.58%`** | `6.79%` | **`11.7x` lower error** |
| `blk.19.ffn_down.weight` | `B1b` (`K=128`) | `1.1000` | **`0.60%`** | `6.91%` | **`11.5x` lower error** |

---

## 4. Path 1 (`Edit-Then-Quantize`) vs. Path 2 (`secant quant patch` In-Place)

Comparing **Path 1** (`secant patch` on `BF16` $\to$ `llama-quantize --pure Q8_0`) against **Path 2** (`secant quant patch` directly on `Qwen3.5-0.8B.Q8_0.gguf`) shows that direct 34-byte block surgery leaves all untouched blocks 100% bit-identical and matches Path 1 within **`0.179%` to `0.923%` per-tensor relative Frobenius error** (note that blocks where the float delta rounds to the same `int8` code remain byte-identical even when inside the touched column support):

| Tensor (`Qwen3.5-0.8B` `Q8_0`) | Total `Q8_0` Blocks | Modified Blocks (Path 2) | Touched but Byte-Unchanged | Untouched Support Blocks (`Bit-Identical = true`) | Relative Frobenius Diff (Path 1 vs Path 2) |
|---|---:|---:|---:|---:|---:|
| `blk.12.ffn_down.weight` (`B2`) | `114,688` | `108,949` | `5,739` | `0` (dense row projector) | `0.923%` |
| `blk.13.ffn_down.weight` (`B3`) | `114,688` | `48,703` | `449` | `65,536` (`100%` identical) | **`0.251%`** |
| `blk.15.ffn_down.weight` (`B1a`) | `114,688` | `31,104` | `640` | `82,944` (`100%` identical) | **`0.225%`** |
| `blk.19.ffn_down.weight` (`B1b`) | `114,688` | `21,842` | `686` | `92,160` (`100%` identical) | **`0.179%`** |
| **All other 316 tensors in file** | `n/a` | **`0`** | **`0`** | **`100%` (`0` bytes differing)** | **`0.000%`** |
