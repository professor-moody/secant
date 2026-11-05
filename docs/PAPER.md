# Direct Multi-Subspace Weight Surgery, Quantized Block-Grid Equivariance, and Subspace-Locked LoRA (`Project-Then-Heal`)

**Authors:** Professor Moody  
**Repository:** [https://github.com/professor-moody/secant](https://github.com/professor-moody/secant)  
**Date:** November 2026

---

## Plain-English Summary

Before presenting the formal propositions, here is what this paper establishes in plain words:

1. **Why we built Secant:** In *Sparks of Weight-Space Programming* (MIT CSAIL, Nov 2026), Takehi, Teoh, Zhang, and Khattab showed that editing `0.016%` of `Qwen3.5-0.8B`'s weights can suppress `<think>` reasoning loops, and called for turning ad-hoc weight inspection into a reproducible engineering discipline and auditing models directly from their weights. Secant is our answer to that challenge.
2. **Why scale calibration is necessary:** A fixed Frobenius-norm edit percentage that works on a `0.8B` model fails (`0/30` `<think>` loops closed) on `Qwen3.5-4B` because wider layers have different internal activation norms. Solving for the weight update that induces a specific inner-product shift ($\Delta_{\text{target}}$) restores a clean transition (`0/30` $\to$ `21/30` at $1.0\times$ $\to$ `30/30` at $1.5\times$).
3. **Why `Q8_0` GGUF files can be edited in place while `Q4_K` superblocks cannot:** Each `Qwen3.5` `ssm_out` value head spans 128 columns. Four 32-weight `Q8_0` blocks divide 128 columns evenly ($128 = 4 \times 32$), so `Q8_0` quantization commutes byte-for-byte with 128-column head permutations and allows direct 34-byte block surgery (`98.5%` refusal removal). A 256-weight `Q4_K` superblock straddles two adjacent 128-column heads with a shared scale factor, leaving `67%` of the refusal direction intact when re-encoded in place.
4. **Why LoRA must be locked inside the forward pass (`Project-Then-Heal`):** Standard LoRA fine-tuning overwrites surgical subspaces when task gradients or AdamW's coordinatewise second-moment scaling push updates back along the excised basis $D$. Parameterizing $B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$ inside the forward graph holds writer-level leakage at $9.87 \times 10^{-18}$ (`float64`) / $2.49 \times 10^{-10}$ (`F32`) during live 60-step LM fine-tuning on `Qwen3.5-0.8B`, and $4.83 \times 10^{-10}$ to $1.08 \times 10^{-9}$ (`float64`) across 9 models from `7.2B` to `32.8B` under adversary-aligned subspace stress testing on an NVIDIA H100.

---

## Abstract

Recent advances in weight-space programming demonstrate that specific behavioral failure modes in language models, such as infinite chain-of-thought (`<think>`) loops or over-refusals on benign defensive security prompts, can be localized to low-rank directions written into the residual stream by attention and MLP output matrices. However, moving from single-direction PyTorch experiments to production checkpoints exposes three open problems:
1. **Cross-scale & multi-subspace calibration:** Fixed Frobenius-norm edit ratios fail when ported across model widths (`0.8B` $\to$ `4B`), and converter permutations (such as `llama.cpp`'s 128-column value-head interleave $\Pi_{128}$ on `Qwen3.5` Gated DeltaNet `blk.N.ssm_out.weight` layers when $r = n_v / n_k > 1$) silently misalign edits applied directly to `.gguf` files.
2. **Direct surgery on compressed quantized weights:** Can weight edits be applied directly to block-quantized `.gguf` tensors (`Q8_0`, `Q4_K`) without re-exporting from floating-point checkpoints?
3. **Post-surgical fine-tuning drift:** When a surgically edited model is subsequently fine-tuned with LoRA and AdamW, task gradients and per-coordinate adaptive second-moment scaling push weight updates back into the excised behavioral subspace, undoing the surgery at the edited writer.

We resolve all three problems in **Secant**:
- First, we derive **inner-product target-shift calibration** ($\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$) and prove the exact 32-block bijection across all 24 Gated DeltaNet layers of `Qwen3.5-4B` (`24/24` bit-exact match). In a 6-arm controlled ablation on `Qwen3.5-4B` (`n = 30`), calibrated multi-subspace surgery (`cal738`) raises strict post-`</think>` completion from `0/30` (`0.0%`) to `21/30` (`70.0%`, exact two-sided McNemar $p = 9.54 \times 10^{-7}$ vs. base and equal-norm random controls), reaching `30/30` (`100.0%`, $p = 1.86 \times 10^{-9}$) at a $1.5\times$ dose (`a15`).
- Second, we prove that $\Pi_{128}$ is block-grid equivariant for any quantization block size $B \mid 128$ (`Q8_0`, $B = 32$) and fails for superblock codecs with $B \nmid 128$ (`Q4_K`, $B = 256$). Empirically, direct block surgery on `Q8_0` removes `98.5%` of the target component (`0.5172` $\to$ `0.0079`) while preserving bit-identity across all untouched 34-byte blocks, whereas `Q4_K` superblock re-encoding leaves `67.0%` of the component intact (`0.346`).
- Third, we introduce **`Project-Then-Heal`** (`secant.nn.ProjectThenHealLinear`), which parameterizes the LoRA output factor structurally inside the orthogonal complement $P_\perp = I_{d_{\text{out}}} - D^\top D$ ($B = P_\perp \tilde{B}$, where $D \in \mathbb{R}^{k \times d_{\text{out}}}$ has orthonormal rows $D D^\top = I_k$). Across 60 live AdamW steps on surgically edited `Qwen3.5-0.8B` (`L15..L17`) and 20 ROCm GPU steps on `SmolLM2-1.7B-Instruct` (`L14..L16`), `Project-Then-Heal` matches unconstrained LoRA task learning (`-89.0%` training loss on `0.8B`, minimum held-out perplexity `5.435` vs. `5.423` at step 20) while holding writer-level weight leakage $\max |D \Delta W|$ at $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$ in `float64` and $1.46 \times 10^{-10}$ to $2.49 \times 10^{-10}$ in merged `F32` (over $4 \times 10^{7}\times$ lower than unconstrained LoRA's $1.05 \times 10^{-2}$ in `F32`).

---

## 1. Multi-Subspace Orthogonalization & Target-Shift Calibration

Let $D = [q_1, \dots, q_k]^\top \in \mathbb{R}^{k \times d_{\text{out}}}$ be $k$ unit row directions in residual-stream coordinates obtained via two-pass `float64` modified Gram-Schmidt over raw contrastive activation differences, satisfying:

$$\|D D^\top - I_k\|_{\mathrm{F}} < 10^{-14}$$

For a writer matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ ($y = W x$), Secant supports two closed-form surgical operators:

1. **Projector subtraction (`projector-subtract`):**
   $$W' = (I_{d_{\text{out}}} - \gamma q_b q_b^\top) W = W - \gamma q_b (q_b^\top W)$$
   At $\gamma = 1$, $q_b^\top W' = 0$ in exact arithmetic.

2. **Inner-product target-shift calibrated rank-1 column edit (`rank1-columns`):**
   Given a subset $S \subset \{0, \dots, d_{\text{in}}-1\}$ of $|S| = K$ input columns selected by top coupling $|(W^\top q_b)_j|$ and a reference input activation $h_{\text{ref}} \in \mathbb{R}^{d_{\text{in}}}$ with zero-padded support vector $h_S \in \mathbb{R}^{d_{\text{in}}}$:
   $$\Delta W = \operatorname{sign}(\Delta_{\text{target}}) \cdot \beta \, q_b \, h_S^\top, \qquad \beta = \frac{|\Delta_{\text{target}}|}{\|h_S\|_2^2}$$
   Then the realized activation shift along $q_b$ at $x = h_{\text{ref}}$ is scale-invariant:
   $$\langle q_b, \Delta W \, h_{\text{ref}} \rangle = \operatorname{sign}(\Delta_{\text{target}}) \cdot \frac{|\Delta_{\text{target}}|}{\|h_S\|_2^2} \langle h_S, h_{\text{ref}} \rangle = \Delta_{\text{target}}, \qquad \|\Delta W\|_{\mathrm{F}} = \frac{|\Delta_{\text{target}}|}{\|h_S\|_2}$$

---

## 2. Container Geometry & The $\Pi_{128}$ Quantization Commutation Theorem

### 2.1 Logical `[d_out, d_in]` Orientation Contract
In HuggingFace `.safetensors`, a 2-D linear weight matrix has `shape = [d_out, d_in]` with `shape[1] = d_in` contiguous. In `ggml` / `.gguf`, `ne = [d_in, d_out]` lists the contiguous axis first (`ne[0] = d_in`). Consequently, both formats share the identical row-major element index:

$$\operatorname{offset}(\text{out}, \text{in}) = \text{out} \cdot d_{\text{in}} + \text{in}$$

### 2.2 Gated DeltaNet `ssm_out` Bijection ($\Pi_{128}$)
On `Qwen3.5`, `llama.cpp`'s `convert_hf_to_gguf.py` reorders the input columns ($d_{\text{in}} = n_v \cdot d_v$, where $d_v = 128$ and $r = n_v / n_k$) of `blk.N.ssm_out.weight` from key-head-major to value-head-major. Writing a HuggingFace 128-column block index as $h = k \cdot r + j$ ($k \in \{0,\dots,n_k-1\}, j \in \{0,\dots,r-1\}$), its GGUF block position is:

$$\pi(k \cdot r + j) = j \cdot n_k + k, \qquad \text{col}_{\text{GGUF}}(h \cdot 128 + w) = \pi(h) \cdot 128 + w \quad (w \in \{0,\dots,127\})$$

When $r = 1$ (`Qwen3.5-0.8B`, $n_k = n_v = 16$), $\pi$ is the identity permutation. When $r = 2$ (`Qwen3.5-4B`, $n_k = 16, n_v = 32$), $\Pi_{128}$ is a non-trivial involution on the 32 panels of width 128.

### 2.3 Block-Grid Equivariance vs. Superblock Straddling
Let $Q_B$ denote a row-wise block quantizer that partitions each row of length $d_{\text{in}}$ into contiguous blocks of $B$ elements and encodes each block independently.

- **Proposition 1 (Block-Grid Equivariance for $B \mid 128$):** Because $\Pi_{128}$ permutes contiguous panels of width $128$, whenever $B \mid 128$ (such as `Q8_0` with $B = 32$, since $128 = 4 \times 32$), every 128-element panel is a disjoint union of $128 / B$ complete quantization blocks. Therefore, block quantization commutes exactly with $\Pi_{128}$ at the byte level:
  $$Q_{32}(\Pi_{128}(W)) = \Pi_{128}^{\text{block}}(Q_{32}(W)), \qquad \|D_{32}(Q_{32}(\Pi_{128}(W))) - \Pi_{128}(D_{32}(Q_{32}(W)))\|_{\max} = 0$$
- **Proposition 2 (Superblock Straddling for $B = 256$):** In `Q4_K` (and `Q5_K`/`Q6_K`), each 256-element superblock shares a superblock FP16 scale $d$ and minimum $d_{\min}$ across two adjacent 128-element value-head panels ($256 = 2 \times 128$). Permuting or editing a single 128-element panel alters the shared $(d, d_{\min})$ fit of the 256-element superblock and perturbs the dequantized weights of the untouched neighboring panel.
- **Remark on Lossy Requantization (`Path 1` vs. `Path 2`):** Grid equivariance of $\Pi_{128}$ does not imply that nonlinear quantization commutes with additive weight edits: $D_B(Q_B(W + \Delta W)) \neq D_B(Q_B(D_B(Q_B(W)) + \Delta W))$. However, because `Q8_0` fits an independent FP16 scale $d = \max_i |x_i| / 127$ per 32-element block, direct `Q8_0` block surgery (`Path 2`) achieves `98.5%` directional suppression (`0.5172` $\to$ `0.0079`, vs. `99.4%` for `Path 1`), whereas `Q4_K` `Path 2` leaves `67.0%` of the removed direction intact (`0.346`).

---

## 3. `Project-Then-Heal`: Structural Orthogonal-Complement LoRA

### 3.1 Why Gradient Projection Fails Under AdamW
Suppose a writer matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ has had an orthonormal row basis $D \in \mathbb{R}^{k \times d_{\text{out}}}$ ($D D^\top = I_k$) surgically removed, and we attach a LoRA adapter $\Delta W = s B A$ ($s = \alpha_{\text{lora}} / r$).

If one projects only the gradient $g_t = P_\perp \nabla_B \mathcal{L}_t$ ($P_\perp = I_{d_{\text{out}}} - D^\top D$) and passes $g_t$ into AdamW, the first moment $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ remains in $\operatorname{col}(P_\perp)$, **but** AdamW divides coordinatewise by $\sqrt{\hat{v}_t} + \epsilon$:

$$\Delta B_t = -\eta \left( \operatorname{diag}\!\left(\frac{1}{\sqrt{\hat{v}_t} + \epsilon}\right) \hat{m}_t + \lambda B_{t-1} \right)$$

Because a diagonal matrix $\Lambda_t = \operatorname{diag}((\sqrt{\hat{v}_t} + \epsilon)^{-1})$ with non-constant diagonal entries does not commute with a non-axis-aligned projector $P_\perp$ ($D \Lambda_t P_\perp \neq 0$), AdamW's step rotates $B_t$ back into $\operatorname{span}(D^\top)$.

### 3.2 Structural Parameterization ($B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$)
`ProjectThenHealLinear` instead applies $P_\perp$ twice in `float64` inside the forward graph:

$$B_{\text{locked}}(\tilde{B}) = P_\perp(P_\perp(\tilde{B})), \qquad y = W_0 x + s \, B_{\text{locked}}(\tilde{B}) (A x)$$

By associativity, for **any** parameter state $(\tilde{B}_t, A_t)$ produced by any optimizer:

$$D \Delta W_t = s \, (D P_\perp) P_\perp(\tilde{B}_t) A_t = s \, (D - (D D^\top) D) P_\perp(\tilde{B}_t) A_t = 0$$

Double projection in `float64` eliminates single-pass IEEE-754 rounding drift, holding writer leakage $\max |D B_{\text{locked}}|$ at $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$ on CPU/ROCm runs and $4.83 \times 10^{-10}$ to $1.08 \times 10^{-9}$ across 9 models (`7.2B` to `32.8B`) on H100.

### 3.3 Writer-Level Invariance vs. Downstream Residual-Stream Propagation
The algebraic guarantee $D \Delta W_t = 0$ is exact at every protected writer's output (`L15..L17` forward drift $\le 1.70 \times 10^{-8}$ in `F32`). When multiple layers are stacked, an orthogonal perturbation $\delta h_\ell \in \operatorname{span}(D^\top)^\perp$ emitted at layer $\ell$ passes through subsequent RMSNorm, attention, and MLP matrices $W_{\ell'}$ ($\ell' > \ell$) for which $D W_{\ell'} P_\perp \neq 0$ in general. Empirically on `Qwen3.5-0.8B` (`Step 60`), locking `L15..L17` eliminates direct writer leakage ($4.2 \times 10^7\times$ reduction in merged `F32`) and cuts downstream pull-back along the `B1` loop direction at `resid L19` by more than half (`+0.276` vs. `+0.635` unconstrained).

---

## 4. Cross-Architecture & Large-Model Empirical Evaluation (`0.8B` to `72.7B`)

Building directly on Takehi, Teoh, Zhang, and Khattab's `Qwen3.5-0.8B` `<think>` reasoning-loop experiment [1], we evaluate Secant across **seventeen checkpoints** (`Qwen3.5-0.8B`, `SmolLM2-1.7B`, `Qwen3.5-4B`, `Mistral-7B-v0.3`, `Qwen2.5-7B`, `Llama-3.1-8B`, `DeepSeek-R1-Distill-Llama-8B`, `Qwen3-8B`, `Gemma-2-9B`, `OLMo-2-13B`, `Microsoft Phi-4`, `Qwen3-14B`, `DeepSeek-R1-Distill-Qwen-14B`, `Gemma-2-27B`, `Qwen3-32B`, `DeepSeek-R1-Distill-Llama-70B`, and `Qwen2.5-72B`) from `0.8B` up to `72.7B` parameters ($d_{\text{model}} \in [1024, 8192]$) on AMD ROCm (`gfx1151`) and NVIDIA H100 80GB SXM GPUs. See [](#fig-paper-phase-surface), [](#fig-paper-snr-heatmap), [](#fig-paper-pth-trajectories), and [](#fig-paper-spectral-read) for the full experimental trajectories:

```{figure} assets/fig-phase-surface-3d.svg
:name: fig-paper-phase-surface
:alt: 3D phase-transition surface and multi-curve sigmoid fit on Qwen3.5-4B
:width: 100%

**(A)** 3D phase-transition surface over target layer index $\ell \in [8, 28]$ and calibrated dose multiplier $\alpha \in [0, 1.75]$ on `Qwen3.5-4B`. **(B)** Observed vs. sigmoid-fitted phase transition (`n = 30`, `1,536`-token cap), showing `0%` closure at uncalibrated $\gamma = 0.30$ and half-dose `0.5x`, `70.0%` closure at $\alpha = 1.0\times$, and `100.0%` (`30/30`, $p = 1.86 \times 10^{-9}$) closure at $\alpha = 1.5\times$.
```

```{figure} assets/fig-snr-heatmap.svg
:name: fig-paper-snr-heatmap
:alt: Sub-stream SNR field heatmap and cross-architecture subspace concentration
:width: 100%

**(A)** Sub-stream contrastive projection SNR field across all 24 layers of `Qwen3.5-0.8B`, contrasting the `L12` peak ($\mathrm{SNR}_{\text{proj}} = 2.12$) with the `L15` `ssm_out` trough ($\mathrm{SNR}_{\text{proj}} = 0.23$, $9.2\times$ weaker). **(B)** Cross-architecture subspace concentration (`7.2B` to `32.8B`) plotting held-out $\mathrm{SNR}_{\text{proj}}$ against split-half cosine stability.
```

```{figure} assets/fig-pth-trajectories.svg
:name: fig-paper-pth-trajectories
:alt: 60-step AdamW training loss, held-out perplexity, and log-scale subspace leakage trajectories
:width: 100%

**(A)** 60-step AdamW training cross-entropy loss and held-out validation perplexity on `Qwen3.5-0.8B` (minimum `PPL = 5.435` at Step 20). **(B)** Log-scale subspace leakage trajectories across 60 AdamW steps from `0.8B` to `72.7B`, contrasting unconstrained LoRA erasure (`8.4%`–`52.6%`) against `ProjectThenHealLinear` ($< 10^{-9}$ in `float64`).
```

```{figure} assets/fig-spectral-read-decay.svg
:name: fig-paper-spectral-read
:alt: Zero-evaluation singular value spectrum and direct weight-to-vocabulary z-score readout
:width: 100%

Zero-evaluation static weight inspection via `secant read`: **(A)** Singular value decay ($\sigma_1 \dots \sigma_{12}$) with static backdoor spike detection ($\sigma_1 = 18.94$, $\mathrm{MAD} > 10$). **(B)** Direct weight-to-vocabulary projection $z$-scores across layers `L10..L23` (`RMSNorm` + `lm_head`), isolating the `L18..L21` refusal lexicon band ($z_{\max} = +6.14 > +5.854$ null).
```

| Model Checkpoint | Params | Layers ($d_{\text{model}}, d_{\text{ffn}}$) | `B2` Peak `SNR_proj` (Layer) | `B2` Shift Ratio vs. Random (`Shift` Pair) | `B3` Shift Ratio vs. Random (`Shift` Pair) | Unconstrained LoRA Subspace Intrusion | `ProjectThenHealLinear` Leakage (`float64` / `F32`) |
|---|---:|---|---:|---:|---:|---:|---:|
| **`Qwen/Qwen3.5-0.8B`** (Live 60-step LM) | `0.8B` | `24` (`1024, 3584`) | `142.8` (`L12`) | `10/10` `<think>` close | `6/10` strict frame | `1.05e-02` max $\|D \Delta W\|$ | **`9.87e-18` / `2.49e-10`** |
| **`HuggingFaceTB/SmolLM2-1.7B-Instruct`** (20-step ROCm) | `1.7B` | `24` (`2048, 8192`) | `164.2` (`L12`) | **`68.0x`** (`-12.50` vs `-0.184`) | **`22.2x`** (`+9.17` vs `+0.413`) | `4.64e-03` max $\|D \Delta W\|$ | **`5.21e-18` / `1.46e-10`** |
| **`Qwen/Qwen3.5-4B`** (6-Arm + Static Lock) | `4.0B` | `32` (`2560, 9216`) | `189.4` (`L16`) | **`30/30`** (`100%`) close (`1.5x`) | **`23/30`** (`76.7%`) class & frame | `1.40e-03` (`BF16` merge floor) | **`7.01e-18` / `2.20e-10`** |
| **`mistralai/Mistral-7B-Instruct-v0.3`** | `7.2B` | `32` (`4096, 14336`) | `87.7` (`L12`) | **`663.9x`** (`+0.257` vs `-0.00039`) | **`98.9x`** (`+3.212` vs `-0.0325`) | `46.79%` ($\|DB\|_F/\|B\|_F$) | **`7.73e-10` / `3.87e-10`** |
| **`Qwen/Qwen2.5-7B-Instruct`** | `7.6B` | `28` (`3584, 18944`) | `199.4` (`L10`) | **`155.1x`** (`+4.654` vs `+0.0300`) | **`132.7x`** (`+20.82` vs `-0.1568`) | `31.34%` ($\|DB\|_F/\|B\|_F$) | **`7.55e-10` / `6.88e-09`** |
| **`meta-llama/Llama-3.1-8B-Instruct`** | `8.0B` | `32` (`4096, 14336`) | `181.8` (`L24`) | **`176.8x`** (`+3.539` vs `-0.0200`) | **`35.3x`** (`+8.888` vs `-0.2520`) | `48.03%` ($\|DB\|_F/\|B\|_F$) | **`7.62e-10` / `2.72e-09`** |
| **`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`** | `8.0B` | `32` (`4096, 14336`) | `184.1` (`L20`) | **`1,269.9x`** (`+2.609` vs `+0.0021`) | **`143.3x`** (`+8.644` vs `-0.0603`) | **`52.55%`** ($\|DB\|_F/\|B\|_F$) | **`6.39e-10` / `2.13e-09`** |
| **`Qwen/Qwen3-8B`** | `8.2B` | `36` (`4096, 12288`) | `168.4` (`L12`) | **`219.7x`** (`+2.187` vs `-0.0100`) | **`29.1x`** (`+120.8` vs `+4.146`) | `39.53%` ($\|DB\|_F/\|B\|_F$) | **`7.14e-10` / `5.93e-09`** |
| **`google/gemma-2-9b-it`** | `9.2B` | `42` (`3584, 14336`) | `83.0` (`L14`) | **`127.9x`** (`+19.08` vs `+0.1492`) | **`87.2x`** (`+130.0` vs `+1.491`) | `41.05%` ($\|DB\|_F/\|B\|_F$) | **`6.53e-10` / `8.50e-10`** |
| **`allenai/OLMo-2-1124-13B-Instruct`** | `13.7B` | `40` (`5120, 13824`) | `142.4` (`L16`) | **`103.3x`** (`+1.250` vs `+0.0121`) | **`262.9x`** (`+7.341` vs `-0.0279`) | `30.73%` ($\|DB\|_F/\|B\|_F$) | **`1.08e-09` / `3.33e-09`** |
| **`microsoft/phi-4`** | `14.7B` | `40` (`5120, 17920`) | `27,362.3` (`L32`) | **`326.9x`** (`-3.399` vs `+0.0104`, abs) | **`48.7x`** (`+30.76` vs `+0.6322`) | `8.42%` ($\|DB\|_F/\|B\|_F$) | **`9.51e-10` / `1.70e-08`** |
| **`Qwen/Qwen3-14B`** | `14.8B` | `40` (`5120, 17408`) | `172.3` (`L16`) | **`252.5x`** (`+4.548` vs `+0.0180`) | **`492.8x`** (`+63.11` vs `-0.1281`) | `41.33%` ($\|DB\|_F/\|B\|_F$) | **`5.77e-10` / `4.68e-09`** |
| **`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`** | `14.8B` | `48` (`5120, 13824`) | `181.1` (`L16`) | **`66.7x`** (`+3.762` vs `+0.0564`) | **`64.9x`** (`+51.94` vs `+0.8008`) | `32.57%` ($\|DB\|_F/\|B\|_F$) | **`6.11e-10` / `2.78e-09`** |
| **`google/gemma-2-27b-it`** | `27.2B` | `46` (`4608, 18432`) | `78.8` (`L22`) | **`139.0x`** (`+542.8` vs `+3.905`) | **`196.0x`** (`+5152.2` vs `-26.29`) | `11.54%` ($\|DB\|_F/\|B\|_F$) | **`7.16e-10` / `7.54e-10`** |
| **`Qwen/Qwen3-32B`** | `32.8B` | `64` (`5120, 25600`) | `93.6` (`L48`) | **`158.6x`** (`+18.89` vs `+0.1191`) | **`510.5x`** (`+41.52` vs `-0.0813`) | `30.75%` ($\|DB\|_F/\|B\|_F$) | **`4.83e-10` / `8.29e-09`** |
| **`deepseek-ai/DeepSeek-R1-Distill-Llama-70B`** | `70.6B` | `80` (`8192, 28672`) | $\sigma_1 = 8.322$ (`L32`) | **`537.1x`** (`+8.301` vs `-0.0155`) | `L32 down_proj` (`234.9M` params) | `20.24%` ($\|DB\|_F/\|B\|_F$) | **`5.66e-10` / `3.98e-08`** |
| **`Qwen/Qwen2.5-72B-Instruct`** | `72.7B` | `80` (`8192, 29568`) | $\sigma_1 = 6.185$ (`L40`) | **`134.2x`** (`+9.043` vs `+0.0674`) | `L40 down_proj` (`242.2M` params) | `36.94%` ($\|DB\|_F/\|B\|_F$) | **`9.85e-10` / `1.58e-08`** |

Across every tested architecture and width ($d_{\text{model}} \in \{1024, 2048, 2560, 3584, 4096, 5120\}$):
1. **Contrastive direction localization (`SNR_proj`) isolates candidate layers across all 12 checkpoints:** Held-out `B2` projected signal-to-noise ratio spans `87.7` (`Mistral-7B-v0.3`) to `199.4` (`Qwen2.5-7B`) across standard architectures (`93.6` at `L48` on 64-layer `Qwen3-32B`). On `microsoft/phi-4`, extreme outlier-dimension concentration produces `SNR_proj = 27,362.3` (`SNR_trace = 909.6` at `L16`) and an inverted, saturating `B2` projection shift (`-3.399` at $1.0\times$ and `-3.519` at $1.5\times$), showing why `SNR_proj` and signed shift verification should always be inspected together.
2. **Rank-1 subspace edits (`rank1-columns`) separate target directions from equal-Frobenius-norm random controls by `22.2x` to `1,269.9x`:** For 16 of the 18 H100 directional edits, scaling the edit multiplier from $\alpha = 1.0\times$ to $1.5\times$ scales the induced projection shift proportionally (`1.50x`), with `phi-4` `B2` (`1.035x`) and `OLMo-2-13B` `B3` (`1.18x`) exhibiting nonlinear saturation.
3. **Under both live LM training and adversary-aligned subspace stress testing, `ProjectThenHealLinear` locks the writer subspace:** Across the 9 H100 stress-test models (`7.2B` to `32.8B`), where the synthetic target $Y_{\text{target}} = Y + 1.5 v_{B2}$ pulls updates directly toward $\operatorname{span}(D)$, unconstrained LoRA places `8.42%` to `52.55%` of its adapter norm inside $\operatorname{span}(D)$, while `ProjectThenHealLinear` holds writer leakage $\max |D B_{\text{locked}}|$ at $4.83 \times 10^{-10}$ to $1.08 \times 10^{-9}$ (`float64`) and merged `F32` leakage at $3.87 \times 10^{-10}$ to $1.70 \times 10^{-8}$.

---

## References

1. **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab.** *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL, November 2, 2026. https://rikiyat.github.io/blog/weight-space-programming/
2. **Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda.** *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024 (`arXiv:2406.11717`).
3. **Junfeng Fang et al.** *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR 2025 (`arXiv:2410.02355`).
4. **Tang et al.** *LoRA-Null: Low-Rank Adaptation via Null Space for Large Language Models.* 2025 (`arXiv:2503.02659`).
5. **OPLoRA Authors.** *OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning.* AAAI 2026 (`arXiv:2510.13003`).
