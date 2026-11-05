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

When `llama.cpp` converts hybrid `Qwen3.5` models to GGUF, it permutes the input columns of `blk.N.ssm_out.weight` in **128-column value-head panels** ($\Pi_{128}$). Whether a quantized format can be edited or permuted cleanly depends on how its block size divides `128`:

| Property | `Q8_0` (GGML Type `8`) | `Q4_K` (GGML Type `12`) |
|---|---|---|
| **Block size & byte footprint** | **32 weights** (`34` bytes: `2B` `f16` scale + `32B` `int8` codes) | **256 weights** (`144` bytes: shared `f16` `d, dmin` + `8` sub-blocks) |
| **Alignment with 128-col `ssm_out` panels** | **Exact ($128 = 4 \times 32$)**: `0` straddling blocks | **Fails ($256 = 2 \times 128$)**: straddles 2 value heads (`8` per row) |
| **Refusal (`B2`) removed by direct block surgery** | **`98.5%` removed** (`0.5172` $\to$ `0.0079`) | **Fails: `67%` of refusal survives** (`0.5172` $\to$ `0.346`) |
| **Secant default behavior** | Supported in-place (`secant quant patch`) | Refused by default (`superblockStraddleError`) |

- **In `Q8_0` (`32` elements per block):** Every 128-column value-head panel is an exact union of 4 complete `Q8_0` blocks. Editing or permuting a panel never touches a neighboring panel's scale factor.
- **In `Q4_K` (`256` elements per superblock):** Every superblock spans two adjacent 128-column panels and shares a single top-level `FP16` scale `d` and minimum `dmin`. Editing one panel shifts the shared superblock scale and corrupts the neighboring value head, leaving `67%` of the unwanted refusal component intact.

---

## 2. The Quantization Commutation Bound

Let $\PiB : \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote the block-quantize-then-dequantize operator for block size $B$. Suppose we apply a column-sparse behavioral edit $\dW$ supported on $K$ columns ($K \le d_{\text{in}}$), touching at most $M_{\text{blocks}}$ blocks per row.

```{prf:theorem} Quantization Commutation Bound
:label: thm-commutation

For any weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and sparse edit $\dW$ supported on $N_{\text{elem}} = d_{\text{out}} \cdot M_{\text{blocks}} \cdot B$ elements inside touched blocks $b \in \mathcal{T}$, let $s_b'$ denote the post-edit block scale of block $b$. Then the Frobenius commutation residual between **Path 1** (editing in floating point and quantizing: $\PiB(W + \dW)$) and the ideal edited quantized tensor ($\PiB(W) + \dW$) satisfies:

$$
\mathcal{E}_{\text{comm}}(W, \dW) \;\equiv\; \big\|\PiB(W + \dW) - (\PiB(W) + \dW)\big\|_F \;\le\; \sqrt{\sum_{b \in \mathcal{T}} B \cdot \left(\frac{s_b'}{2}\right)^2 + \big\|\PiB(W)\big|_{\mathcal{T}} - W\big|_{\mathcal{T}}\big\|_F^2}
$$

Moreover, for **Path 2** (direct quantized surgery $\PiB(\PiB(W) + \dW)$ via `secant quant patch`), every block $b \notin \mathcal{T}$ is **bit-for-bit identical** between $\PiB(W)$ and $\PiB(\PiB(W) + \dW)$, and on touched blocks $b \in \mathcal{T}$, the dequantized difference between Path 1 and Path 2 is bounded by one half-step of the `Q8_0` quantizer (`max_abs_diff` $\le s_b + s_b' \approx 1.4 \times 10^{-3}$).
```

---

## 3. Empirical Commutation Witness: `Q8_0` vs. `Q4_K`

Evaluating $\Pi_{128}(\Delta W)$ across all four edited `ffn_down` layers (`12, 13, 15, 19`) of `Qwen3.5-0.8B` (`secant quant witness`) confirms that `Q8_0` achieves **`11.5x` lower quantization error** than `Q4_K`:

| Layer (`Qwen3.5-0.8B`) | Edit Stream | $\|\Delta W\|_F$ | `Q8_0` Relative Error | `Q4_K` Relative Error | `Q8_0` Advantage |
|---|---|---:|---:|---:|---:|
| `blk.12.ffn_down.weight` | `B2` (dense) | `0.8500` | **`0.61%`** | `7.06%` | **`11.5x` lower error** |
| `blk.13.ffn_down.weight` | `B3` (`K=128`) | `0.8500` | **`0.59%`** | `6.84%` | **`11.6x` lower error** |
| `blk.15.ffn_down.weight` | `B1a` (`K=128`) | `1.1000` | **`0.58%`** | `6.79%` | **`11.7x` lower error** |
| `blk.19.ffn_down.weight` | `B1b` (`K=128`) | `1.1000` | **`0.60%`** | `6.91%` | **`11.5x` lower error** |

---

## 4. Path 1 (`Edit-Then-Quantize`) vs. Path 2 (`secant quant patch` In-Place)

Comparing **Path 1** (`secant patch` on `BF16` $\to$ `llama-quantize --pure Q8_0`) against **Path 2** (`secant quant patch` directly on `Qwen3.5-0.8B.Q8_0.gguf`) shows that direct 34-byte block surgery leaves all untouched blocks 100% bit-identical, matches Path 1 within **`0.18%–0.92%` relative Frobenius error**, and lifts `<think>` loop closure from **`10.0%` (`3/30`) to `96.7%` (`29/30`) at $\alpha = 1.0\times$ and `100.0%` (`30/30`) at $\alpha = 1.5\times$**:

| Tensor (`Qwen3.5-0.8B` `Q8_0`) | Total `Q8_0` Blocks | Rewritten Blocks (Path 2) | Untouched Blocks (`Bit-Identical = true`) | Relative Frobenius Diff (Path 1 vs Path 2) |
|---|---:|---:|---:|---:|
| `blk.12.ffn_down.weight` (`B2`) | `114,688` | `108,949` | `5,739` (`100%` identical) | `0.923%` |
| `blk.13.ffn_down.weight` (`B3`) | `114,688` | `48,703` | `65,536` (`100%` identical) | **`0.251%`** |
| `blk.15.ffn_down.weight` (`B1a`) | `114,688` | `31,104` | `82,944` (`100%` identical) | **`0.225%`** |
| `blk.19.ffn_down.weight` (`B1b`) | `114,688` | `21,842` | `92,160` (`100%` identical) | **`0.179%`** |
| **All other 316 tensors in file** | — | **`0`** | **`100%` (`0` bytes differing)** | **`0.000%`** |
