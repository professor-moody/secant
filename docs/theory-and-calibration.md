# Mathematical Theory & Scale Calibration

Before diving into the equations, here is the core intuition behind Secant's mathematical pipeline:

1. **Finding where a behavior lives (`SNR_proj`):** A transformer layer writes into a shared residual stream of width $d_{\text{model}}$ (`1,024` to `5,120` dimensions). If you measure variance across all `5,120` dimensions at once (`SNR_trace`), background language variance across the other $d_{\text{model}} - 1$ axes dilutes a single behavioral direction (`snr_trace` typically sits around `0.4` to `1.8`). Measuring signal-to-noise ratio **projected onto the candidate direction (`SNR_proj`)** on held-out prompts isolates the exact layer and writer (`attn_out` vs. `mlp_out`) where the behavior is sharpest (`21.9` to `199.4` across standard models).
2. **Keeping multiple edits from colliding (Gram-Schmidt):** When you combine loop suppression (`B1`), refusal subtraction (`B2`), and root-cause attribution (`B3`), their raw direction vectors can overlap when they share residual coordinates. Two-pass modified Gram-Schmidt orthogonalization in `float64` removes all pairwise overlap ($\|D D^\top - I_k\|_{\mathrm{F}} < 10^{-14}$) before compiling the weight patch.
3. **Why fixed norm ratios fail on larger models (Target-Shift Calibration):** Copying a fixed Frobenius norm ratio ($\gamma = 0.30$) from a `0.8B` model to a `4B` model produces a `-0.90` activation shift instead of the `-7.38` shift needed to exit `<think>`. **Target-Shift Calibration** solves directly for the step size that delivers the desired inner-product shift $\Delta_{\text{target}}$ at the layer's real activation scale.

```{figure} assets/qwen4b-phase-transition.svg
:name: fig-theory-phase-transition
:alt: Qwen3.5-4B Target-Shift Calibration phase transition
:width: 100%

Why calibration matters: on `Qwen3.5-4B`, an uncalibrated norm ratio (`g030`, $\gamma = 0.30$) and a half-dose edit (`a05`, $\alpha = 0.5\times$) both close `0/30` `<think>` loops. Crossing the calibrated threshold ($\alpha = 1.0\times \to 1.5\times$) triggers a phase transition from `0%` to `70%` and `100%` completion.
```

---

## 1. Contrastive Capture & Held-Out Signal-to-Noise Metrics

Let $h_\ell(x) \in \mathbb{R}^{d_{\text{model}}}$ denote the activation exiting layer $\ell \in \{0, \dots, L-1\}$ (either in the residual stream or a specific writer sub-stream such as `attn_out` / `ssm_out` or `mlp_out`) at the final prompt token of input $x$. Given $N$ paired contrastive inputs $\{(x_i^+, x_i^-)\}_{i=1}^N$, we partition the pairs deterministically into a capture split $\mathcal{I}_{\text{cap}}$ and a held-out evaluation split $\mathcal{I}_{\text{eval}}$.

For each pair $i$, define the per-pair difference vector:

$$
\delta_{\ell, i} = h_\ell(x_i^+) - h_\ell(x_i^-) \in \mathbb{R}^{d_{\text{model}}}
$$

The unit capture direction on $\mathcal{I}_{\text{cap}}$ is:

$$
\bar{\delta}_\ell^{\text{cap}} = \frac{1}{|\mathcal{I}_{\text{cap}}|} \sum_{i \in \mathcal{I}_{\text{cap}}} \delta_{\ell, i}, \qquad v_\ell = \frac{\bar{\delta}_\ell^{\text{cap}}}{\|\bar{\delta}_\ell^{\text{cap}}\|_2}
$$

### Why `SNR_proj` Replaces Full-Trace `SNR_trace`

In high-dimensional residual streams ($d_{\text{model}} \in [1024, 5120]$), total within-class variance sums variation across all $d_{\text{model}} - 1$ orthogonal directions, keeping raw trace SNR modest (`0.43` to `1.83` across our `7B` to `32B` runs, with outlier-dimension concentration on `microsoft/phi-4`) even when a 1D concept direction cleanly separates the two prompt classes. Secant therefore evaluates three held-out diagnostics on $\mathcal{I}_{\text{eval}}$:

1. **Projected Signal-to-Noise Ratio (`snr_proj`)**:
   $$
   \mathrm{SNR}_{\text{proj}}(\ell) = \frac{\left(v_\ell^\top \bar{\delta}_\ell^{\text{eval}}\right)^2}{\mathrm{Var}_{i \in \mathcal{I}_{\text{eval}}}\!\left(v_\ell^\top \delta_{\ell, i}\right)}
   $$
2. **Trace Signal-to-Noise Ratio (`snr_trace`)**:
   $$
   \mathrm{SNR}_{\text{trace}}(\ell) = \frac{\|\bar{\delta}_\ell^{\text{eval}}\|_2^2}{\frac{1}{|\mathcal{I}_{\text{eval}}| - 1} \sum_{i \in \mathcal{I}_{\text{eval}}} \|\delta_{\ell, i} - \bar{\delta}_\ell^{\text{eval}}\|_2^2}
   $$
3. **Split-Half Cosine Stability (`split_half_cosine`)**:
   $$
   \cos(\theta_\ell) = \frac{\langle \bar{\delta}_\ell^{\text{cap}}, \bar{\delta}_\ell^{\text{eval}} \rangle}{\|\bar{\delta}_\ell^{\text{cap}}\|_2 \, \|\bar{\delta}_\ell^{\text{eval}}\|_2}
   $$

A candidate behavioral layer $\ell^*$ is admissible for weight-space surgery if $\mathrm{SNR}_{\text{proj}}(\ell^*) \ge 2.0$ and $\cos(\theta_{\ell^*}) \ge 0.70$.

---

## 2. Sparse MLP Down-Projection Rank-1 Surgery

Let $W_\ell \equiv W_{\text{down},\ell} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$ be the MLP down-projection matrix at layer $\ell$, which maps intermediate SwiGLU activations $m_\ell(x) \in \mathbb{R}^{d_{\text{ffn}}}$ into the residual stream:

$$
h_\ell^{\text{post-mlp}}(x) = h_\ell^{\text{pre-mlp}}(x) + W_\ell \, m_\ell(x)
$$

Rather than modifying all $d_{\text{ffn}}$ columns of $W_\ell$, Secant selects the top-$K$ intermediate neurons $\mathcal{C}_b \subset \{0, \dots, d_{\text{ffn}}-1\}$ ($|\mathcal{C}_b| = K$, typically $K \in \{64, 128\}$) that couple most strongly to the behavior, and constructs a column-sparse selector vector $u_b \in \mathbb{R}^{d_{\text{ffn}}}$ supported solely on $\mathcal{C}_b$:

$$
u_{b, j} = \begin{cases} \dfrac{\bar{m}_{\ell, j}}{\sqrt{\sum_{r \in \mathcal{C}_b} \bar{m}_{\ell, r}^2}} & \text{if } j \in \mathcal{C}_b, \\[8pt] 0 & \text{otherwise.} \end{cases}
$$

A rank-1 column-sparse update with step size $\gamma_b \in \mathbb{R}$ takes the form:

$$
\Delta W_{\ell, b} = \gamma_b \, v_b \, u_b^\top, \qquad \|\Delta W_{\ell, b}\|_{\mathrm{F}} = |\gamma_b| \cdot \|v_b\|_2 \cdot \|u_b\|_2 = |\gamma_b|
$$

Because $K \ll d_{\text{ffn}}$ (e.g., $64 / 9216 = 0.69\%$ of columns on `Qwen3.5-4B`), over **99% of the matrix entries remain completely untouched**.

---

## 3. Multi-Subspace Gram-Schmidt Orthogonalization

When $k$ behavioral edits $b \in \{1, \dots, k\}$ share the coordinate basis $\mathbb{R}^{d_{\text{model}}}$, non-orthogonal directions $\langle v_b, v_{b'} \rangle \ne 0$ can interfere constructively or destructively.

Given $V = [v_1, \dots, v_k] \in \mathbb{R}^{d_{\text{model}} \times k}$, `secant compose` applies **two-pass modified Gram-Schmidt orthogonalization in `float64`** to produce an orthonormal row basis $D = [q_1, \dots, q_k]^\top \in \mathbb{R}^{k \times d_{\text{model}}}$ satisfying:

$$
\|D D^\top - I_k\|_{\mathrm{F}} < 10^{-14}
$$

The multi-subspace update is then synthesized over the orthogonalized directions $q_b$:

$$
\Delta W_\ell = \sum_{b \in \mathcal{B}_\ell} \gamma_b \, q_b \, u_b^\top
$$

---

## 4. Target-Shift Calibration & Fixed-Ratio Subspace Projection

Suppose we want a writer matrix update $\Delta W$ to induce a target inner-product shift $\Delta_{\text{target}}$ along unit direction $q_b$ when presented with reference input activation $h_{\text{ref}} \in \mathbb{R}^{d_{\text{in}}}$, where $h_S \in \mathbb{R}^{d_{\text{in}}}$ denotes $h_{\text{ref}}$ zeroed outside the selected column support $S$:

$$
\Delta W = \mathrm{sign}(\Delta_{\text{target}}) \cdot \beta \, q_b \, h_S^\top, \qquad \beta = \frac{|\Delta_{\text{target}}|}{\|h_S\|_2^2}
$$

Then the induced output shift along $q_b$ at $h_{\text{ref}}$ is exact by construction:

$$
\langle q_b, \, \Delta W \, h_{\text{ref}} \rangle = \mathrm{sign}(\Delta_{\text{target}}) \cdot \frac{|\Delta_{\text{target}}|}{\|h_S\|_2^2} \, \underbrace{q_b^\top q_b}_{=1} \, \underbrace{h_S^\top h_{\text{ref}}}_{=\|h_S\|_2^2} = \Delta_{\text{target}}
$$

and the Frobenius norm of the update adapts automatically to the layer's activation scale:

$$
\|\Delta W\|_{\mathrm{F}} = \frac{|\Delta_{\text{target}}|}{\|h_S\|_2}
$$

When reference activation norms $\|h_S\|_2$ are not pre-captured, `secant compose` also supports a relative Frobenius perturbation mode scaled by a dose multiplier $\alpha > 0$:

$$
\gamma_b(\alpha) = \alpha \cdot \frac{\tau_b \, \|W_\ell\|_{\mathrm{F}}}{\|q_b\|_2 \, \|u_b\|_2}
$$

On `Qwen3.5-0.8B`, `SmolLM2-1.7B`, and `Qwen3.5-4B`, full Target-Shift Calibration ($\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$) triggers the `0%` $\to$ `70%` $\to$ `100%` `<think>` closure transition. On our 9-model H100 architecture sweep (`7.2B` to `32.8B`), fixed-ratio rank-1 subspace projection ($\tau_b = 0.045$) separates aligned directional shifts from matched-norm random directions by **`29.1x` to `1,269.9x`** as a geometric alignment check across architectures.
