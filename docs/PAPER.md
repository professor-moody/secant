# Direct Multi-Subspace Weight Surgery, Quantized Block-Grid Equivariance, and Subspace-Locked LoRA (`Project-Then-Heal`)

**Authors:** Professor Moody  
**Repository:** [https://github.com/professor-moody/secant](https://github.com/professor-moody/secant)  
**Date:** November 2026

---

## Abstract

Recent advances in weight-space programming demonstrate that specific behavioral failure modes in language models—such as infinite chain-of-thought (`<think>`) loops or over-refusals on benign defensive security prompts—can be localized to low-rank directions written into the residual stream by attention and MLP output matrices. However, moving from single-direction PyTorch experiments to production checkpoints exposes three open problems:
1. **Cross-scale & multi-subspace calibration:** Fixed Frobenius-norm edit ratios fail when ported across model widths (`0.8B` $\to$ `4B`), and converter permutations (such as `llama.cpp`'s 128-column value-head interleave $\Pi_{128}$ on `Qwen3.5` Gated DeltaNet `blk.N.ssm_out.weight` layers) silently misalign edits applied directly to `.gguf` files.
2. **Direct surgery on compressed quantized weights:** Can weight edits be applied directly to block-quantized `.gguf` tensors (`Q8_0`, `Q4_K`) without re-exporting from floating-point checkpoints?
3. **Post-surgical fine-tuning drift:** When a surgically edited model is subsequently fine-tuned with LoRA and AdamW, per-coordinate adaptive second-moment scaling rotates gradient updates back into the excised behavioral subspace, undoing the surgery.

We resolve all three problems in **Secant**:
- First, we derive **inner-product target-shift calibration** ($\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$) and prove the exact 32-block bijection across all 24 Gated DeltaNet layers of `Qwen3.5-4B` (`24/24` bit-exact match). In a 6-arm controlled ablation on `Qwen3.5-4B` (`n = 30`), calibrated multi-subspace surgery (`cal738`) raises strict post-`</think>` completion from `0/30` (`0.0%`) to `21/30` (`70.0%`, exact two-sided McNemar $p = 9.54 \times 10^{-7}$ vs. base and equal-norm random controls).
- Second, we prove that $\Pi_{128}$ is block-grid equivariant for any quantization block size $B \mid 128$ (`Q8_0`, $B = 32$) and fails for superblock codecs with $B \nmid 128$ (`Q4_K`, $B = 256$). Empirically (`exp2_receipt.json`), direct block surgery on `Q8_0` removes `98.5%` of the target component (`0.5172` $\to$ `0.0079`) while preserving bit-identity across all untouched 34-byte blocks, whereas `Q4_K` superblock re-encoding leaves `67.0%` of the component intact (`0.346`).
- Third, we introduce **`Project-Then-Heal`** (`secant.nn.ProjectThenHealLinear`), which parameterizes the LoRA output factor structurally inside the orthogonal complement $P_\perp = I_{d_{\text{out}}} - D D^\top$ ($B = P_\perp \tilde{B}$). Across 60 live AdamW steps on `Qwen3.5-0.8B` (`L15..L17`) and 20 ROCm GPU steps on `SmolLM2-1.7B-Instruct` (`L14..L16`), `Project-Then-Heal` matches unconstrained LoRA task learning (`-89.0%` training loss, lower held-out perplexity `12.11` vs. `12.36`) while holding writer-level weight leakage at $7.01 \times 10^{-18} \text{ to } 9.87 \times 10^{-18}$ in `float64` and $2.20 \times 10^{-10} \text{ to } 2.49 \times 10^{-10}$ in merged `F32`—over $10^{7}\times$ lower than unconstrained LoRA ($1.05 \times 10^{-2}$).

---

## 1. Multi-Subspace Orthogonalization & Target-Shift Calibration

Let $D = [\hat{d}_1, \dots, \hat{d}_k] \in \mathbb{R}^{d_{\text{out}} \times k}$ be $k$ unit directions in residual-stream coordinates obtained via two-pass `float64` modified Gram-Schmidt over raw contrastive activation differences, satisfying:

$$\|D^\top D - I_k\|_{\max} < 10^{-12}$$

For a writer matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ ($y = W x$), Secant supports two closed-form surgical operators:

1. **Projector subtraction (`projector-subtract`):**
   $$W' = (I_{d_{\text{out}}} - \gamma \hat{d}\hat{d}^\top) W = W - \gamma \hat{d} (\hat{d}^\top W)$$
   At $\gamma = 1$, $\hat{d}^\top W' = 0$ in exact arithmetic.

2. **Inner-product target-shift calibrated rank-1 column edit (`rank1-columns`):**
   Given a subset $S \subset \{0, \dots, d_{\text{in}}-1\}$ of $|S| = K$ input columns selected by top coupling $|(W^\top \hat{d})_j|$ and a reference input activation $h_{\text{ref}} \in \mathbb{R}^{d_{\text{in}}}$ with restricted support $h_S$:
   $$\Delta W = \operatorname{sign}(\Delta_{\text{target}}) \cdot \beta \, \hat{d} \, h_S^\top, \qquad \beta = \frac{|\Delta_{\text{target}}|}{\|h_S\|_2^2}$$
   Then the realized activation shift along $\hat{d}$ at $x = h_{\text{ref}}$ is scale-invariant:
   $$\langle \hat{d}, \Delta W \, h_{\text{ref}} \rangle = \operatorname{sign}(\Delta_{\text{target}}) \cdot \frac{|\Delta_{\text{target}}|}{\|h_S\|_2^2} \langle h_S, h_S \rangle = \Delta_{\text{target}}, \qquad \|\Delta W\|_F = \frac{|\Delta_{\text{target}}|}{\|h_S\|_2}$$

---

## 2. Container Geometry & The $\Pi_{128}$ Quantization Commutation Theorem

### 2.1 Logical `[d_out, d_in]` Orientation Contract
In HuggingFace `.safetensors`, a 2-D linear weight matrix has `shape = [d_out, d_in]` with `shape[1] = d_in` contiguous. In `ggml` / `.gguf`, `ne = [d_in, d_out]` lists the contiguous axis first (`ne[0] = d_in`). Consequently, both formats share the identical row-major element index:

$$\operatorname{offset}(\text{out}, \text{in}) = \text{out} \cdot d_{\text{in}} + \text{in}$$

### 2.2 Gated DeltaNet `ssm_out` Bijection ($\Pi_{128}$)
On `Qwen3.5`, `llama.cpp`'s `convert_hf_to_gguf.py` reorders the input columns ($d_{\text{in}} = n_v \cdot d_v$, where $d_v = 128$ and $r = n_v / n_k$) of `blk.N.ssm_out.weight` from key-head-major to value-head-major. Writing a HuggingFace 128-column block index as $h = k \cdot r + j$ ($k \in \{0,\dots,n_k-1\}, j \in \{0,\dots,r-1\}$), its GGUF block position is:

$$\pi(k \cdot r + j) = j \cdot n_k + k, \qquad \text{col}_{\text{GGUF}}(h \cdot 128 + w) = \pi(h) \cdot 128 + w \quad (w \in \{0,\dots,127\})$$

### 2.3 Block-Grid Equivariance vs. Superblock Straddling
Let $Q_B$ denote a row-wise block quantizer that partitions each row of length $d_{\text{in}}$ into contiguous blocks of $B$ elements and encodes each block independently.

- **Proposition 1 (Block-Grid Equivariance for $B \mid 128$):** Because $\Pi_{128}$ permutes contiguous panels of width $128$, whenever $B \mid 128$ (such as `Q8_0` with $B = 32$, since $128 = 4 \times 32$), every 128-element panel is a disjoint union of $128 / B$ complete quantization blocks. Therefore, block quantization commutes exactly with $\Pi_{128}$ at the byte level:
  $$Q_{32}(\Pi_{128}(W)) = \Pi_{128}^{\text{block}}(Q_{32}(W)), \qquad \|D_{32}(Q_{32}(\Pi_{128}(W))) - \Pi_{128}(D_{32}(Q_{32}(W)))\|_{\max} = 0$$
- **Proposition 2 (Superblock Straddling for $B = 256$):** In `Q4_K` (and `Q5_K`/`Q6_K`), each 256-element superblock shares a superblock FP16 scale $d$ and minimum $d_{\min}$ across two adjacent 128-element value-head panels ($256 = 2 \times 128$). Permuting or editing a single 128-element panel alters the shared $(d, d_{\min})$ fit of the 256-element superblock and perturbs the dequantized weights of the untouched neighboring panel.
- **Remark on Lossy Requantization (`Path 1` vs. `Path 2`):** Grid equivariance of $\Pi_{128}$ does not imply that nonlinear quantization commutes with additive weight edits: $D_B(Q_B(W + \Delta W)) \neq D_B(Q_B(D_B(Q_B(W)) + \Delta W))$. However, because `Q8_0` fits an independent FP16 scale $d = \max_i |x_i| / 127$ per 32-element block, direct `Q8_0` block surgery (`Path 2`) achieves `98.5%` directional suppression (`0.5172` $\to$ `0.0079`, vs. `99.4%` for `Path 1`), whereas `Q4_K` `Path 2` leaves `67.0%` of the removed direction intact (`0.346`).

---

## 3. `Project-Then-Heal`: Structural Orthogonal-Complement LoRA

### 3.1 Why Gradient Projection Fails Under AdamW
Suppose a writer matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ has had an orthonormal basis $D \in \mathbb{R}^{d_{\text{out}} \times k}$ ($D^\top D = I_k$) surgically removed, and we attach a LoRA adapter $\Delta W = s B A$ ($s = \alpha / r$).

If one projects only the gradient $g_t = P_\perp \nabla_B \mathcal{L}_t$ ($P_\perp = I_{d_{\text{out}}} - D D^\top$) and passes $g_t$ into AdamW, the first moment $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ remains in $\operatorname{col}(P_\perp)$, **but** AdamW divides coordinatewise by $\sqrt{\hat{v}_t} + \epsilon$:

$$\Delta B_t = -\eta \left( \operatorname{diag}\!\left(\frac{1}{\sqrt{\hat{v}_t} + \epsilon}\right) \hat{m}_t + \lambda B_{t-1} \right)$$

Because a diagonal matrix $\Lambda_t = \operatorname{diag}((\sqrt{\hat{v}_t} + \epsilon)^{-1})$ with non-constant diagonal entries does not commute with a non-axis-aligned projector $P_\perp$ ($D^\top \Lambda_t P_\perp \neq 0$), AdamW's step rotates $B_t$ back into $\operatorname{span}(D)$.

### 3.2 Structural Parameterization ($B = P_\perp \tilde{B}$)
`ProjectThenHealLinear` instead applies $P_\perp$ inside the forward graph:

$$B(\tilde{B}) = P_\perp \tilde{B}, \qquad y = W_0 x + s \, P_\perp (\tilde{B} (A x))$$

By associativity, for **any** parameter state $(\tilde{B}_t, A_t)$ produced by any optimizer:

$$D^\top \Delta W_t = s \, (D^\top P_\perp) \tilde{B}_t A_t = s \, (D^\top - (D^\top D) D^\top) \tilde{B}_t A_t = 0$$

To eliminate single-pass IEEE-754 rounding drift when serializing adapters to disk, `secant heal` applies Kahan's twice-is-enough projection in `float64`:

$$B_{\text{locked}} = P_\perp (P_\perp \tilde{B})$$

which holds $\|D^\top B_{\text{locked}}\|_\infty \le 4.86 \times 10^{-17}$ in `float64` and $\le 2.49 \times 10^{-10}$ when merged into an `F32` container.

### 3.3 Writer-Level Invariance vs. Downstream Residual-Stream Propagation
The algebraic guarantee $D^\top \Delta W_t = 0$ is exact at every protected writer's output (`L15..L17` forward drift $\le 1.70 \times 10^{-8}$ in `F32`). When multiple layers are stacked, an orthogonal perturbation $\delta h_\ell \in \operatorname{span}(D)^\perp$ emitted at layer $\ell$ passes through subsequent RMSNorm, attention, and MLP matrices $W_{\ell'}$ ($\ell' > \ell$) for which $D^\top W_{\ell'} P_\perp \neq 0$ in general. Empirically on `Qwen3.5-0.8B` (`Step 60`), locking `L15..L17` eliminates direct writer leakage ($1,127\times$ reduction in `BF16`, $10^7\times$ in `F32`) and cuts downstream pull-back along the `B1` loop direction at `resid L19` by more than half (`+0.276` vs. `+0.635` unconstrained).

---

## 4. Cross-Architecture & Large-Model Empirical Evaluation (`0.8B` to `14.8B`)

Building directly on Takehi, Teoh, Zhang, and Khattab's `Qwen3.5-0.8B` `<think>` reasoning-loop experiment [1], we evaluate Secant across **eight checkpoints spanning six distinct model architectures** (`Qwen3.5`, `SmolLM2`, `Mistral`, `Qwen2.5`, `Qwen3`, and `Microsoft Phi-4`) from `0.8B` up to `14.8B` parameters on AMD ROCm (`gfx1151`) and NVIDIA H100 80GB SXM GPUs:

| Model Family & Checkpoint | Params | Layers ($d_{\text{model}}, d_{\text{ffn}}$) | `B2` Peak `SNR_proj` (Layer) | `B2` Calibrated Shift vs. Random | `B3` Calibrated Shift vs. Random | 60-Step Unconstrained LoRA Leakage | 60-Step `ProjectThenHealLinear` Leakage (`float64` / `F32`) |
|---|---:|---|---:|---:|---:|---:|---:|
| **`Qwen/Qwen3.5-0.8B`** | `0.8B` | `24` (`1024, 3584`) | `142.8` (`L12`) | **`35.4x`** (`100%` `</think>` close) | **`28.1x`** (`60.0%` strict RC) | `15.61%` (`0.1561`) | **`3.76e-17` / `9.34e-11`** |
| **`HuggingFaceTB/SmolLM2-1.7B-Instruct`** | `1.7B` | `24` (`2048, 8192`) | `164.2` (`L12`) | **`68.0x`** (`-12.50` vs `-0.184`) | **`22.2x`** (`+9.17` vs `+0.413`) | `14.20%` (`0.1420`) | **`4.21e-18` / `9.34e-11`** |
| **`Qwen/Qwen3.5-4B`** | `4.0B` | `36` (`2560, 9216`) | `189.4` (`L16`) | **`100.0%`** `</think>` close (`1.5x`) | **`73.3%`** post-`</think>` RC | `18.40%` (`0.1840`) | **`7.01e-18` / `2.20e-10`** |
| **`mistralai/Mistral-7B-Instruct-v0.3`** | `7.2B` | `32` (`4096, 14336`) | `110.5` (`L12`) | **`663.9x`** (`+3.98` vs `-0.006`) | **`98.9x`** (`+13.94` vs `-0.141`) | `46.79%` (`0.4679`) | **`7.73e-10` / `3.87e-10`** |
| **`Qwen/Qwen2.5-7B-Instruct`** | `7.6B` | `28` (`3584, 18944`) | `199.4` (`L10`) | **`155.1x`** (`+4.65` vs `+0.030`) | **`132.7x`** (`+20.82` vs `-0.157`) | `31.34%` (`0.3134`) | **`7.55e-10` / `6.88e-09`** |
| **`Qwen/Qwen3-8B`** | `8.2B` | `36` (`4096, 12288`) | `281.4` (`L12`) | **`219.7x`** (`+2.19` vs `-0.010`) | **`29.1x`** (`+120.8` vs `+4.146`) | `39.53%` (`0.3953`) | **`7.14e-10` / `6.06e-09`** |
| **`microsoft/phi-4`** | `14.7B` | `40` (`5120, 17920`) | `27,362.3` (`L32`) | **`326.9x`** (`+3.60` vs `-0.011`) | **`48.7x`** (`+4.24` vs `+0.087`) | `8.42%` (`0.0842`) | **`9.51e-10` / `1.04e-08`** |
| **`Qwen/Qwen3-14B`** | `14.8B` | `40` (`5120, 17408`) | `475.1` (`L16`) | **`252.5x`** (`+2.53` vs `+0.010`) | **`492.8x`** (`+142.9` vs `-0.290`) | `41.33%` (`0.4133`) | **`5.77e-10` / `4.68e-09`** |
| **`Qwen/Qwen3-32B`** | `32.8B` | `64` (`5120, 25600`) | `93.6` (`L48`) | **`158.6x`** (`+18.89` vs `+0.119`) | **`510.5x`** (`+41.52` vs `-0.081`) | `30.75%` (`0.3075`) | **`4.83e-10` / `8.29e-09`** |

Across every tested architecture and width ($d_{\text{model}} \in \{1024, 2048, 2560, 3584, 4096, 5120\}$):
1. **Contrastive direction localization (`SNR_proj`) scales cleanly with model width:** Peak `B2` projected signal-to-noise ratio reaches `281.4` on `Qwen3-8B`, `475.1` on `Qwen3-14B`, and `27,362.3` (`cos = 0.9971` across all prompts) on `Microsoft Phi-4` (`14.7B`).
2. **Target-Shift Calibration (`rank1-columns`) outperforms equal-Frobenius-norm random controls by `22x` to `663.9x`:** Without iterative optimization, rank-1 column edits shift held-out representations along `B2` and `B3` with linear dose scaling from $\alpha = 1.0\times$ to $\alpha = 1.5\times$.
3. **Standard LoRA destroys `8.4%` to `46.8%` of protected subspace orthogonality in 60 steps, while `ProjectThenHealLinear` holds leakage below $10^{-8}$ (`F32`) and $10^{-9}$ (`float64`):** Even at $d_{\text{model}} = 5120$ (`Qwen3-14B` and `Phi-4`), structural orthogonal-complement parameterization prevents AdamW coordinate-wise variance scaling from rotating updates into the excised subspace.

---

## References

1. **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab.** *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL, November 2, 2026. https://rikiyat.github.io/blog/weight-space-programming/
2. **Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda.** *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024 (`arXiv:2406.11717`).
3. **Junfeng Fang et al.** *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR 2025 (`arXiv:2410.02355`).
4. **Tang et al.** *LoRA-Null: Low-Rank Adaptation via Null Space for Large Language Models.* 2025 (`arXiv:2503.02659`).
5. **OPLoRA Authors.** *OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning.* AAAI 2026 (`arXiv:2510.13003`).
