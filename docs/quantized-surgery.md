# Quantized Block Surgery (`Q8_0` & `Q4_K`)

In production deployments, language models are rarely served in uncompressed `BF16` or `F32`; they are distributed as block-quantized GGUF files (`Q8_0`, `Q4_K_M`, `Q6_K`). Re-downloading a multi-gigabyte `BF16` checkpoint simply to apply a rank-1 or rank-3 behavioral edit and re-quantizing the entire file is slow and modifies every tensor across the model.

Secant solves this with **direct in-place quantized block surgery (`secant quant patch`)**, backed by the **Quantization Commutation Bound**.

---

## 1. Binary Layout of `Q8_0` and `Q4_K` Super-Blocks

### `Q8_0` (GGML Type `8`, `34` Bytes per `32` Elements)

Each row of a `Q8_0` tensor $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is partitioned into contiguous blocks of $B = 32$ weights. Each 34-byte block stores:
- `d`: `f16` (IEEE 754 binary16, 2 bytes, little-endian) scale factor $s_b = \mathrm{f16\_to\_f32}(d) = \max_i |w_{b,i}| / 127$.
- `qs`: `[i8; 32]` (32 signed 8-bit integers), where $q_{b,i} = \mathrm{round}(w_{b,i} / s_b) \in [-127, 127]$.

Dequantization reconstructs $\hat{w}_{b,i} = s_b \cdot q_{b,i}$.

### `Q4_K` (GGML Type `12`, `144` Bytes per `256` Elements)

Each 256-element super-block is partitioned into eight 32-element sub-blocks and stores:
- `d`, `dmin`: two `f16` super-block scales (4 bytes total).
- `scales`: 12 bytes packing 6-bit sub-block scales $s_j$ and minimums $m_j$ ($j \in \{0, \dots, 7\}$).
- `qs`: 128 bytes packing 256 4-bit unsigned nibbles $q_{j,i} \in [0, 15]$.

---

## 2. The Quantization Commutation Theorem

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

Evaluating $\Pi_{128}(\Delta W)$ across all four edited `ffn_down` layers (`12, 13, 15, 19`) of `Qwen3.5-0.8B` (`secant quant verify`) demonstrates why **`Q8_0` is the recommended quantized format for weight-space surgery**:

| Layer (`Qwen3.5-0.8B`) | Edit Stream | $\|\Delta W\|_F$ | `Q8_0` Relative Error ($\frac{\|\Pi_{Q8}(\Delta W) - \Delta W\|_F}{\|\Delta W\|_F}$) | `Q4_K` Relative Error ($\frac{\|\Pi_{Q4K}(\Delta W) - \Delta W\|_F}{\|\Delta W\|_F}$) | `Q8_0` Advantage |
|---|---|---:|---:|---:|---:|
| `blk.12.ffn_down.weight` | `B2` (dense) | `0.8500` | **`0.61%`** | `7.06%` | **`11.5x` lower error** |
| `blk.13.ffn_down.weight` | `B3` (`K=128`) | `0.8500` | **`0.59%`** | `6.84%` | **`11.6x` lower error** |
| `blk.15.ffn_down.weight` | `B1a` (`K=128`) | `1.1000` | **`0.58%`** | `6.79%` | **`11.7x` lower error** |
| `blk.19.ffn_down.weight` | `B1b` (`K=128`) | `1.1000` | **`0.60%`** | `6.91%` | **`11.5x` lower error** |

---

## 4. Path 1 (`Edit-Then-Quantize`) vs. Path 2 (`secant quant patch` Direct Surgery)

On `strix-ts`, we compared **Path 1** (`secant patch` on `BF16` $\to$ `llama-quantize --pure Q8_0`, SHA-256 `86205fc7...`) against **Path 2** (`secant quant patch` directly on `base-pure-Q8_0.gguf`, SHA-256 `8bcfdaed...`):

| Tensor (`Qwen3.5-0.8B` `Q8_0`) | Total `Q8_0` Blocks | Rewritten Blocks (Path 2) | Untouched Blocks (`Bit-Identical = true`) | Dequantized Relative Frobenius Diff (Path 1 vs Path 2) | Mean Absolute Weight Diff |
|---|---:|---:|---:|---:|---:|
| `blk.12.ffn_down.weight` (`B2`) | `114,688` | `108,949` | `5,739` (`100%` identical) | `0.923%` (`0.00923`) | `3.89e-5` |
| `blk.13.ffn_down.weight` (`B3`) | `114,688` | `48,703` | `65,536` (`100%` identical) | **`0.251%`** (`0.00251`) | `2.85e-6` |
| `blk.15.ffn_down.weight` (`B1a`) | `114,688` | `31,104` | `82,944` (`100%` identical) | **`0.225%`** (`0.00225`) | `2.20e-6` |
| `blk.19.ffn_down.weight` (`B1b`) | `114,688` | `21,842` | `92,160` (`100%` identical) | **`0.179%`** (`0.00179`) | `1.53e-6` |
| **All other 316 tensors in file** | — | **`0`** | **`100%` (`0` bytes differing)** | **`0.000%`** | **`0.0`** |
