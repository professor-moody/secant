# Mathematical Theory & Scale Calibration

This chapter derives the linear-algebraic formulation behind Secant's contrastive direction capture, multi-subspace orthogonalization, and **Target-Shift Calibration**.

---

## 1. Contrastive Capture & Held-Out Signal-to-Noise Metrics

Let $h_\ell(x) \in \mathbb{R}^{d_{\text{model}}}$ denote the residual-stream activation exiting layer $\ell \in \{0, \dots, L-1\}$ at the final prompt token of input $x$. Given $N$ paired contrastive inputs $\{(x_i^+, x_i^-)\}_{i=1}^N$, we partition the pairs deterministically into a capture split $\mathcal{I}_{\text{cap}}$ and a held-out evaluation split $\mathcal{I}_{\text{eval}}$ ($|\mathcal{I}_{\text{cap}}| = |\mathcal{I}_{\text{eval}}| = N/2$).

For each pair $i$, define the per-pair difference vector:

$$
\delta_{\ell, i} = h_\ell(x_i^+) - h_\ell(x_i^-) \in \mathbb{R}^{d_{\text{model}}}
$$

The unit capture direction on $\mathcal{I}_{\text{cap}}$ is:

$$
\bar{\delta}_\ell^{\text{cap}} = \frac{1}{|\mathcal{I}_{\text{cap}}|} \sum_{i \in \mathcal{I}_{\text{cap}}} \delta_{\ell, i}, \qquad v_\ell = \frac{\bar{\delta}_\ell^{\text{cap}}}{\|\bar{\delta}_\ell^{\text{cap}}\|_2}
$$

### Why `SNR_proj` Replaces Full-Trace `SNR_trace`

In high-dimensional residual streams ($d_{\text{model}} \in [1024, 8192]$), the total within-class variance $\mathrm{tr}(\Sigma_\ell)$ sums noise across all $d_{\text{model}} - 1$ orthogonal directions, making raw trace SNR artificially small ($\approx 0.05\text{–}0.35$) even when a 1D concept direction cleanly separates the classes. Secant therefore reports three complementary held-out diagnostics on $\mathcal{I}_{\text{eval}}$:

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

Rather than modifying all $d_{\text{ffn}}$ columns of $W_\ell$, Secant selects the top-$K$ intermediate neurons $\mathcal{C}_b \subset \{0, \dots, d_{\text{ffn}}-1\}$ ($|\mathcal{C}_b| = K$, typically $K \in \{64, 128\}$) with the largest mean positive activation $\bar{m}_{\ell, j}$ on the target distribution, and constructs a column-sparse selector vector $u_b \in \mathbb{R}^{d_{\text{ffn}}}$ supported solely on $\mathcal{C}_b$:

$$
u_{b, j} = \begin{cases} \dfrac{\bar{m}_{\ell, j}}{\sqrt{\sum_{r \in \mathcal{C}_b} \bar{m}_{\ell, r}^2}} & \text{if } j \in \mathcal{C}_b, \\[8pt] 0 & \text{otherwise.} \end{cases}
$$

A rank-1 column-sparse update with step size $\gamma_b \in \mathbb{R}$ takes the form:

$$
\Delta W_{\ell, b} = \gamma_b \, v_b \, u_b^\top, \qquad \|\Delta W_{\ell, b}\|_F = |\gamma_b| \cdot \|v_b\|_2 \cdot \|u_b\|_2 = |\gamma_b|
$$

---

## 3. Multi-Subspace Gram-Schmidt Orthogonalization

When $k$ behavioral edits $b \in \{1, \dots, k\}$ share output space $\mathbb{R}^{d_{\text{model}}}$ (either within the same layer or across coupled residual layers), non-orthogonal directions $\langle v_b, v_{b'} \rangle \ne 0$ cause constructive or destructive interference.

Given $V = [v_1, \dots, v_k] \in \mathbb{R}^{d_{\text{model}} \times k}$, `secant compose` applies **two-pass modified Gram-Schmidt orthogonalization in `float64`** to produce an orthonormal basis $D = [q_1, \dots, q_k]^\top \in \mathbb{R}^{k \times d_{\text{model}}}$ satisfying:

$$
\|D D^\top - I_k\|_F < 10^{-14}
$$

The multi-subspace update is then synthesized over the orthogonalized directions $q_b$:

$$
\Delta W_\ell = \sum_{b \in \mathcal{B}_\ell} \gamma_b \, q_b \, u_b^\top
$$

---

## 4. Target-Shift Calibration Across Model Scales (`0.8B` $\to$ `4B`+)

### Why Fixed $\gamma$ Fails as Model Scale Increases

Suppose an edit step size $\gamma_0$ is tuned on a small model (`Qwen3.5-0.8B`, $d_{\text{model}} = 1024$, $\|W_\ell\|_F \approx 41.2$). When applied directly to a $5\times$ larger model (`Qwen3.5-4B`, $d_{\text{model}} = 2560$, $\|W_\ell\|_F \approx 89.4$), the relative Frobenius perturbation $\|\Delta W_\ell\|_F / \|W_\ell\|_F$ drops by more than $2.1\times$, and the induced logit shift falls below the threshold required to close `<think>` loops (`16.7%` closed under `fix_g` vs `13.3%` unedited baseline).

### Target-Shift Calibration Law

To make behavioral surgery **scale-invariant**, `secant compose --calibration target-shift` specifies a dimensionless target relative perturbation $\tau_b$ (anchored to the layer Frobenius norm $\|W_\ell\|_F$ and reference activation norm $\|h_{\text{ref},\ell}\|_2$) scaled by a global dose parameter $\alpha > 0$:

$$
\gamma_b(\alpha) = \alpha \cdot \frac{\tau_b \, \|W_\ell\|_F}{\|q_b\|_2 \, \|u_b\|_2}
$$

As demonstrated on `Qwen3.5-4B` (`n = 30`), sweeping $\alpha \in \{0.0, 0.5, 1.0, 1.5\}$ uncovers a sharp **first-order phase transition**:
- At $\alpha = 0.5$ (`a05`), the induced residual shift along $q_{\text{B1}}$ remains below the `</think>` emission logit gap on every prompt (`0/30` closed).
- At $\alpha = 1.0$ (`cal738`), the shift crosses the threshold on `21/30` (`70.0%`) prompts (`18/30` `60.0%` strict root-cause accuracy).
- At $\alpha = 1.5$ (`a15`), **`30/30` (`100.0%`) prompts emit `</think>` and terminate cleanly with `stop = eos`**, reducing mean generation length from `1,429.2` tokens to **`789.0` tokens** (`-44.8%`) while boosting strict root-cause attribution from `6.7%` to **`63.3%`** (`73.3%` post-`</think>`).
