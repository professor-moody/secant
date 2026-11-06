# Mathematical Theory & Scale Calibration

Before diving into the equations, here is the core intuition behind Secant's mathematical pipeline:

1. **Finding where a behavior lives (`SNR_proj`):** A transformer layer writes into a shared residual stream of width $d_{\text{model}}$ (`1,024` to `8,192` dimensions in the checkpoints we tested). If you measure variance across all dimensions at once (`SNR_trace`), background language variance across the other $d_{\text{model}} - 1$ axes dilutes a single behavioral direction (on the 12 full H100 models `snr_trace` is `0.32` to `4.88`, apart from `phi-4`). Measuring signal-to-noise ratio **projected onto the candidate direction (`SNR_proj`)** on held-out prompts isolates the exact layer and writer (`attn_out` vs. `mlp_out`) where the behavior is sharpest (held-out `snr_proj` `4.6` to `277.9` across all captured layers of the 12 full H100 models, excluding `phi-4`).
2. **Keeping multiple edits from colliding (Gram-Schmidt):** When you combine loop suppression (`B1`), refusal subtraction (`B2`), and root-cause attribution (`B3`), their raw direction vectors can overlap when they share residual coordinates. Two-pass modified Gram-Schmidt orthogonalization in `float64` removes pairwise overlap before compiling the weight patch. The implementation enforces $\max_{i,j} |(D D^\top - I_k)_{ij}| < 10^{-12}$ and fails closed otherwise; the measured `float64` residual on tested bases is below $10^{-15}$.
3. **Why fixed norm ratios fail on larger models (Target-Shift Calibration):** Copying a fixed Frobenius norm ratio ($\gamma = 0.30$) from a `0.8B` model to a `4B` model closed `0/30` `<think>` loops, while the calibrated loop edit (summed target shift `-7.38`) closed `21/30`. **Target-Shift Calibration** solves directly for the step size that delivers the desired inner-product shift $\Delta_{\text{target}}$ at the layer's real activation scale.

---

## 1. Contrastive Capture & Held-Out Signal-to-Noise Metrics

Let $h_\ell(x) \in \mathbb{R}^{d}$ denote the activation exiting layer $\ell \in \{0, \dots, L-1\}$ (either in the residual stream or a specific writer sub-stream such as `attn_out` / `ssm_out` or `mlp_out`) at the final prompt token of input $x$. Capture collects a positive class $H^+ = \{h_\ell(x_i^+)\}_{i=1}^{n_+}$ and a negative class $H^- = \{h_\ell(x_j^-)\}_{j=1}^{n_-}$. The two classes are **not** paired and need not have equal size; every statistic below is computed from separate class means and separate class variances, in `float64` (`python/secant/capture.py`, `evaluate_cell`).

**Splits.** Each class is shuffled independently with a fixed seed derived from (`seed`, `stream`) and cut into half $A$ (the first $\lfloor n/2 \rfloor$ samples) and half $B$ (the rest). For a sample set $S$, write $\mu_\pm^S$ for the class means and

$$
u_S = \frac{\mu_+^S - \mu_-^S}{\|\mu_+^S - \mu_-^S\|_2}
$$

for the unit difference-of-means direction fitted on $S$. The reported `direction` is $u$ fitted on all samples.

**Unbiased variances.** For a scalar sample $\{z_i\}_{i=1}^{n}$, $\hat{\sigma}^2(z) = \frac{1}{n-1}\sum_i (z_i - \bar{z})^2$ (`ddof=1`). For a vector class $H$ on split $S$, $\operatorname{tr}\hat{\Sigma}^S = \sum_{c=1}^{d} \hat{\sigma}^2(h_{\cdot, c})$, the sum of the unbiased per-coordinate variances.

### Why `SNR_proj` Replaces Full-Trace `SNR_trace`

In high-dimensional residual streams ($d \in [1024, 5120]$), the trace denominator sums within-class variation across all $d$ coordinates, keeping raw trace SNR modest (`0.32` to `4.88` across the 12 full H100 models, with `microsoft/phi-4` an outlier at up to `909.6`) even when a 1D concept direction cleanly separates the two prompt classes. The projected statistic instead measures variance along one fitted direction only. Secant evaluates three diagnostics, each fitted on one half and scored on the other:

1. **Projected (1D) Signal-to-Noise Ratio (`snr_proj`)**. Fit $u_A$ on half $A$ and score the projections $z^\pm = u_A^\top h$ on half $B$:
   $$
   \mathrm{SNR}_{\text{proj}}^{A \to B}(\ell) = \frac{\left(\overline{z^+_B} - \overline{z^-_B}\right)^2}{\hat{\sigma}^2(z^+_B) + \hat{\sigma}^2(z^-_B)} = \frac{\left(u_A^\top(\mu_+^B - \mu_-^B)\right)^2}{\hat{\sigma}^2(u_A^\top H^+_B) + \hat{\sigma}^2(u_A^\top H^-_B)}
   $$
   The same is computed with the roles swapped ($B \to A$), and the reported value is the average over the two splits:
   $$
   \mathrm{SNR}_{\text{proj}}(\ell) = \tfrac{1}{2}\left(\mathrm{SNR}_{\text{proj}}^{A \to B}(\ell) + \mathrm{SNR}_{\text{proj}}^{B \to A}(\ell)\right)
   $$
   Both directional values are also recorded (`snr_proj_a_to_b`, `snr_proj_b_to_a`).
2. **Trace (full-space) Signal-to-Noise Ratio (`snr_trace`)**. This uses no fitted direction; "fit on $A$" only selects which half is scored:
   $$
   \mathrm{SNR}_{\text{trace}}^{\to B}(\ell) = \frac{\|\mu_+^B - \mu_-^B\|_2^2}{\operatorname{tr}\hat{\Sigma}_+^B + \operatorname{tr}\hat{\Sigma}_-^B}, \qquad \mathrm{SNR}_{\text{trace}}(\ell) = \tfrac{1}{2}\left(\mathrm{SNR}_{\text{trace}}^{\to B}(\ell) + \mathrm{SNR}_{\text{trace}}^{\to A}(\ell)\right)
   $$
   With finite samples the numerator also contains noise, so this estimate is biased upward by about $d\,\sigma^2 (1/n_+ + 1/n_-)$ on a half; it is not null-corrected.
3. **Split-Half Cosine Stability (`split_half_cosine`)**. The cosine between the directions fitted independently on each half:
   $$
   \cos(\theta_\ell) = u_A^\top u_B
   $$
   Near `1` means a stable direction; near `0` (about $\mathcal{N}(0, 1/d)$) means the "direction" is noise.

The 1D projected SNR and the trace SNR answer different questions: $\mathrm{SNR}_{\text{proj}}$ asks how well one held-out direction separates the classes, while $\mathrm{SNR}_{\text{trace}}$ compares the mean shift with the total variance of the whole space. They are not interchangeable and are never divided into one another.

**Undefined cells.** If a mean difference vanishes (relative to $\|\mu_+\| + \|\mu_-\|$, below $10^{-12}$) or a denominator is zero or non-finite, the cell is reported as `status="undetermined"` with `NaN` metrics and a reason, never as a number. Capture requires at least 4 samples per class so each half has a variance.

A candidate behavioral layer $\ell^*$ is admissible for weight-space surgery if $\mathrm{SNR}_{\text{proj}}(\ell^*) \ge 2.0$ and $\cos(\theta_{\ell^*}) \ge 0.70$. These thresholds are permissive: every peak layer used for an edit in the H100 runs is far above them, so in practice the layer choice is the argmax, not the threshold.

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

Given $V = [v_1, \dots, v_k] \in \mathbb{R}^{d_{\text{model}} \times k}$, `secant compose` applies **two-pass modified Gram-Schmidt orthogonalization in `float64`** to produce an orthonormal row basis $D = [q_1, \dots, q_k]^\top \in \mathbb{R}^{k \times d_{\text{model}}}$ satisfying the enforced max-norm contract (`python/secant/_linalg.py`, `ORTHONORMALITY_TOL`):

$$
\max_{i,j} \big|(D D^\top - I_k)_{ij}\big| < 10^{-12}
$$

A basis that misses this bound, or a direction whose residual norm after orthogonalization falls below $10^{-9}$ of its input norm (linear dependence), raises an error instead of returning a weaker basis. The bound is the contract. The **measured** `float64` residual on tested bases is below $10^{-15}$: the maximum over 120 seeded Gaussian bases ($d \in \{1024, 2560, 3584, 4096, 5120, 8192\}$, $k \in \{3, 4\}$, 5 seeds each) was $9.99 \times 10^{-16}$.

The multi-subspace update is then synthesized over the orthogonalized directions $q_b$:

$$
\Delta W_\ell = \sum_{b \in \mathcal{B}_\ell} \gamma_b \, q_b \, u_b^\top
$$

---

## 4. Target-Shift Calibration & Fixed-Ratio Subspace Projection

Suppose we want a writer matrix update $\Delta W$ to induce a target inner-product shift $\Delta_{\text{target}}$ along unit direction $q_b$ when presented with reference input activation $h_{\text{ref}} \in \mathbb{R}^{d_{\text{in}}}$, where $h_S \in \mathbb{R}^{d_{\text{in}}}$ denotes $h_{\text{ref}}$ zeroed outside the selected column support $S$ and satisfies the non-degeneracy precondition $\|h_S\|_2 > 0$:

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

The two modes are **not** equivalent: the calibrated step $\beta$ depends on the measured activation $h_S$, the ratio step $\gamma_b(\alpha)$ only on $\|W_\ell\|_{\mathrm{F}}$.

Where each mode was used: full Target-Shift Calibration ($\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$) on `Qwen3.5-4B` (all `B1+B2+B3` edits, `0/30` $\to$ `21/30` $\to$ `30/30` `<think>` closure at doses $0.5\times$, $1.0\times$, $1.5\times$) and on the `SmolLM2-1.7B` `B3` edit. The 14-checkpoint H100 sweep (`7.2B` to `72.7B`) used the fixed-ratio mode ($\gamma = 0.045\,\|W\|_{\mathrm{F}}$); there, aligned directions separate from equal-norm random directions by `29.1x` to `1,269.9x` by magnitude. A random unit direction projects onto $v$ with expected magnitude about $1/\sqrt{d_{\text{model}}}$, so this is a geometric sanity check, not evidence of behavioural change.
