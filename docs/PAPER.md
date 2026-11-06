# Direct Multi-Subspace Weight Surgery, Quantized Block-Grid Equivariance, and Subspace-Locked LoRA (`Project-Then-Heal`)

**Authors:** Professor Moody  
**Repository:** [https://github.com/professor-moody/secant](https://github.com/professor-moody/secant)  
**Date:** November 2026

---

## Plain-English Summary

Before presenting the formal propositions, here is what this paper establishes in plain words:

1. **Why we built Secant:** In *Sparks of Weight-Space Programming* (MIT CSAIL, Nov 2026), Takehi, Teoh, Zhang, and Khattab showed that editing `0.016%` of `Qwen3.5-0.8B`'s weights can suppress `<think>` reasoning loops, and called for turning ad-hoc weight inspection into a reproducible engineering discipline and auditing models directly from their weights. Secant is our answer to that challenge.
2. **Why scale calibration is necessary:** A fixed Frobenius-norm edit percentage that works on a `0.8B` model fails (`0/30` `<think>` loops closed) on `Qwen3.5-4B` because wider layers have different internal activation norms. Solving for the weight update that induces a specific inner-product shift ($\Delta_{\text{target}}$) restores closure on `Qwen3.5-4B` (`0/30` at $0.5\times$, `21/30` at $1.0\times$, `30/30` at $1.5\times$; one greedy run per arm).
3. **Why `Q8_0` GGUF files can be edited in place while `Q4_K` superblocks cannot:** Each `Qwen3.5` `ssm_out` value head spans 128 columns. Four 32-weight `Q8_0` blocks divide 128 columns evenly ($128 = 4 \times 32$), so `Q8_0` quantization commutes byte-for-byte with 128-column head permutations and allows direct 34-byte block surgery (`98.5%` refusal removal). A 256-weight `Q4_K` superblock straddles two adjacent 128-column heads with a shared scale factor, leaving `67%` of the refusal direction intact when re-encoded in place.
4. **Why LoRA must be locked inside the forward pass (`Project-Then-Heal`):** Standard LoRA fine-tuning can write back into a surgical subspace whenever the task gradient has a component along the excised basis $D$, and projecting only the gradient is not enough under AdamW (Section 3.1). Parameterizing $B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$ inside the forward graph holds writer-level leakage at $9.87 \times 10^{-18}$ (`float64`) / $2.49 \times 10^{-10}$ (merged `F32`) during live 60-step LM fine-tuning on `Qwen3.5-0.8B`. In a separate H100 stress test on 14 checkpoints (`7.2B` to `72.7B`), where the training target deliberately lies inside the protected subspace, leakage stayed at $4.82 \times 10^{-10}$ to $1.08 \times 10^{-9}$ (`float64`) at step 60. That stress test edits with a fixed ratio rather than calibration, trains on the unedited writer, and does not measure learning quality; Section 4 states its limits.

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

Let $D = [q_1, \dots, q_k]^\top \in \mathbb{R}^{k \times d_{\text{out}}}$ be $k$ unit row directions in residual-stream coordinates obtained via two-pass `float64` modified Gram-Schmidt over raw contrastive activation differences. The implementation enforces the max-norm contract

$$\max_{i,j} \big|(D D^\top - I_k)_{ij}\big| < 10^{-12}$$

and fails closed otherwise (`python/secant/_linalg.py`, `ORTHONORMALITY_TOL`; the Rust `compose` path enforces the same `1e-12` bound on post-orthogonalization $\max |\cos|$). The empirically measured `float64` residual on tested bases is below $10^{-15}$ (maximum $9.99 \times 10^{-16}$ over 120 seeded Gaussian bases with $d_{\text{out}} \in \{1024, \dots, 8192\}$, $k \in \{3, 4\}$); that is a measurement, not the contract.

For a writer matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ ($y = W x$), Secant supports two closed-form surgical operators:

1. **Projector subtraction (`projector-subtract`):**
   $$W' = (I_{d_{\text{out}}} - \gamma q_b q_b^\top) W = W - \gamma q_b (q_b^\top W)$$
   At $\gamma = 1$, $q_b^\top W' = 0$ in exact arithmetic.

2. **Inner-product target-shift calibrated rank-1 column edit (`rank1-columns`):**
   Given a subset $S \subset \{0, \dots, d_{\text{in}}-1\}$ of $|S| = K$ input columns selected by top coupling $|(W^\top q_b)_j|$ and a reference input activation $h_{\text{ref}} \in \mathbb{R}^{d_{\text{in}}}$ with zero-padded support vector $h_S \in \mathbb{R}^{d_{\text{in}}}$ satisfying $\|h_S\|_2 > 0$:
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

When $r = 1$ (`Qwen3.5-0.8B`, $n_k = n_v = 16$), $\pi$ is the identity permutation. When $r = 2$ (`Qwen3.5-4B`, $n_k = 16, n_v = 32$), $\Pi_{128}$ is a non-trivial bijection on the 32 panels of width 128: the transpose of a $(n_k, r) = (16, 2)$ grid into a $(2, 16)$ grid. Because $r \neq n_k$, it is **not** an involution (for example $\pi(1) = 16$ but $\pi(16) = 8$). Its inverse is

$$\pi^{-1}(g) = (g \bmod n_k) \cdot r + \lfloor g / n_k \rfloor .$$

### 2.3 Block-Grid Equivariance vs. Superblock Straddling
Let $Q_B$ denote a row-wise block quantizer that partitions each row of length $d_{\text{in}}$ into contiguous blocks of $B$ elements and encodes each block independently.

- **Proposition 1 (Block-Grid Equivariance for $B \mid 128$):** Because $\Pi_{128}$ permutes contiguous panels of width $128$, whenever $B \mid 128$ (such as `Q8_0` with $B = 32$, since $128 = 4 \times 32$), every 128-element panel is a disjoint union of $128 / B$ complete quantization blocks. Therefore, block quantization commutes exactly with $\Pi_{128}$ at the byte level:
  $$Q_{32}(\Pi_{128}(W)) = \Pi_{128}^{\text{block}}(Q_{32}(W)), \qquad \|D_{32}(Q_{32}(\Pi_{128}(W))) - \Pi_{128}(D_{32}(Q_{32}(W)))\|_{\max} = 0$$
- **Proposition 2 (Superblock Straddling for $B = 256$):** In `Q4_K` (and `Q5_K`/`Q6_K`), each 256-element superblock shares a superblock FP16 scale $d$ and minimum $d_{\min}$ across two adjacent 128-element value-head panels ($256 = 2 \times 128$). Permuting or editing a single 128-element panel alters the shared $(d, d_{\min})$ fit of the 256-element superblock and perturbs the dequantized weights of the untouched neighboring panel.
- **Remark on Lossy Requantization (`Path 1` vs. `Path 2`):** Grid equivariance of $\Pi_{128}$ does not imply that nonlinear quantization commutes with additive weight edits: $D_B(Q_B(W + \Delta W)) \neq D_B(Q_B(D_B(Q_B(W)) + \Delta W))$. Writing $\hat{Q}_B = D_B \circ Q_B$ and $e_0 = \hat{Q}_B(W) - W$, $e_1 = \hat{Q}_B(W + \Delta W) - (W + \Delta W)$, $e_2 = \hat{Q}_B(\hat{Q}_B(W) + \Delta W) - (\hat{Q}_B(W) + \Delta W)$, the exact identity for any deterministic block-local codec is
  $$\hat{Q}_B(W + \Delta W) - \hat{Q}_B\big(\hat{Q}_B(W) + \Delta W\big) = e_1 - e_0 - e_2 ,$$
  so the Path 1 vs. Path 2 gap is controlled by **three** quantization errors on the touched blocks. For `Q8_0` the post-edit block scale drifts by at most $\|\Delta W_b\|_\infty / 127$, and each element of a touched block obeys $|e_1 - e_0 - e_2| \le \kappa\big((3 + \kappa)\|W_b\|_\infty + 2\|\Delta W_b\|_\infty\big)$ with $\kappa = 1/254 + 2^{-11}$ (FP16 scale in the normal range; derivation in [Quantized Surgery](quantized-surgery.md)). `Q4_K` has no such proved bound here and is characterized empirically. Moreover, because `Q8_0` fits an independent FP16 scale $d = \max_i |x_i| / 127$ per 32-element block, direct `Q8_0` block surgery (`Path 2`) achieves `98.5%` directional suppression (`0.5172` $\to$ `0.0079`, vs. `99.4%` for `Path 1`), whereas `Q4_K` `Path 2` leaves `67.0%` of the removed direction intact (`0.346`).

---

## 3. `Project-Then-Heal`: Structural Orthogonal-Complement LoRA

### 3.1 Why Gradient Projection Fails Under AdamW
Suppose a writer matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ has had an orthonormal row basis $D \in \mathbb{R}^{k \times d_{\text{out}}}$ ($D D^\top = I_k$) surgically removed, and we attach a LoRA adapter $\Delta W = s B A$ ($s = \alpha_{\text{lora}} / r$).

If one projects only the gradient $g_t = P_\perp \nabla_B \mathcal{L}_t$ ($P_\perp = I_{d_{\text{out}}} - D^\top D$) and passes $g_t$ into AdamW, the first moment $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ remains in $\operatorname{col}(P_\perp)$, **but** AdamW divides coordinatewise by $\sqrt{\hat{v}_t} + \epsilon$:

$$\Delta B_t = -\eta \left( \operatorname{diag}\!\left(\frac{1}{\sqrt{\hat{v}_t} + \epsilon}\right) \hat{m}_t + \lambda B_{t-1} \right)$$

Because a diagonal matrix $\Lambda_t = \operatorname{diag}((\sqrt{\hat{v}_t} + \epsilon)^{-1})$ with non-constant diagonal entries does not commute with a non-axis-aligned projector $P_\perp$ ($D \Lambda_t P_\perp \neq 0$), AdamW's step can rotate $B_t$ back into $\operatorname{span}(D^\top)$. This is an argument about the gradient-projection variant. Regimes A and C in Section 4 include no gradient-projection-only arm; a separate 3-arm synthetic-regression test on real layer-16 `down_proj` weights (`Qwen3.5-0.8B`, `Qwen3.5-4B`, `Qwen2.5-7B`; seeded random $D$, $k = 3$; 3 seeds) does, and is reported in [Project-Then-Heal, Section 3](project-then-heal.md). There the gradient-projected arm put `1.56%` to `2.98%` of $\|\Delta W\|_{\mathrm{F}}$ into $\operatorname{span}(D)$ at step 60 on the orthogonal target, unchanged between weight decay `0.01` and `0.0`, while `ProjectThenHealLinear` stayed at $5.42 \times 10^{-10}$ to $1.05 \times 10^{-9}$ (post-hoc learning rate `3e-3`; the preregistered `1e-2` run gives the same ordering).

### 3.2 Structural Parameterization ($B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$)
`ProjectThenHealLinear` instead applies $P_\perp$ twice in `float64` inside the forward graph:

$$B_{\text{locked}}(\tilde{B}) = P_\perp(P_\perp(\tilde{B})), \qquad y = W_0 x + s \, B_{\text{locked}}(\tilde{B}) (A x)$$

By associativity, for **any** parameter state $(\tilde{B}_t, A_t)$ produced by any optimizer:

$$D \Delta W_t = s \, (D P_\perp) P_\perp(\tilde{B}_t) A_t = s \, (D - (D D^\top) D) P_\perp(\tilde{B}_t) A_t = 0$$

In exact arithmetic $P_\perp^2 = P_\perp$, so the second projection changes nothing. In `float64` it is an observed numerical refinement, not a proved bound: on 120 seeded Gaussian test cases it lowered $\max |D B_{\text{locked}}|$ by $1.8\times$ to $17.7\times$ (median $5.6\times$) relative to one pass. Measured writer leakage $\max |D B_{\text{locked}}|$ is $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$ in the live CPU/ROCm runs and $4.82 \times 10^{-10}$ to $1.08 \times 10^{-9}$ at step 60 across the 14 H100 checkpoints (`7.2B` to `72.7B`; at most $1.60 \times 10^{-9}$ at any step).

### 3.3 Writer-Level Invariance vs. Downstream Residual-Stream Propagation
The algebraic guarantee $D \Delta W_t = 0$ is exact at every protected writer's output (on `Qwen3.5-0.8B` at most $1.25 \times 10^{-8}$ in `F32` at the `L15..L17` writer outputs, and $1.70 \times 10^{-8}$ in the residual stream at `L15`). When multiple layers are stacked, an orthogonal perturbation $\delta h_\ell \in \operatorname{span}(D^\top)^\perp$ emitted at layer $\ell$ passes through subsequent RMSNorm, attention, and MLP matrices $W_{\ell'}$ ($\ell' > \ell$) for which $D W_{\ell'} P_\perp \neq 0$ in general. Empirically on `Qwen3.5-0.8B` (`Step 60`), locking `L15..L17` eliminates direct writer leakage ($4.2 \times 10^7\times$ reduction in merged `F32`) and cuts downstream pull-back along the `B1` loop direction at `resid L19` by more than half (`+0.276` vs. `+0.635` unconstrained).

---

## 4. Cross-Architecture Empirical Evaluation (`0.8B` to `72.7B`)

Building on Takehi, Teoh, Zhang, and Khattab's `Qwen3.5-0.8B` `<think>` reasoning-loop experiment [1], we ran Secant on **seventeen checkpoints** (`Qwen3.5-0.8B`, `SmolLM2-1.7B`, `Qwen3.5-4B`, `Mistral-7B-v0.3`, `Qwen2.5-7B`, `Llama-3.1-8B`, `DeepSeek-R1-Distill-Llama-8B`, `Qwen3-8B`, `Gemma-2-9B`, `OLMo-2-13B`, `Phi-4`, `Qwen3-14B`, `DeepSeek-R1-Distill-Qwen-14B`, `Gemma-2-27B`, `Qwen3-32B`, `DeepSeek-R1-Distill-Llama-70B`, `Qwen2.5-72B`; $d_{\text{model}} \in [1024, 8192]$) on CPU, AMD ROCm (`gfx1151`), and NVIDIA H100 80GB hardware. They fall into three regimes that measure different things:

- **Regime A (live LM fine-tuning after real surgery):** `Qwen3.5-0.8B` (60 AdamW steps) and `SmolLM2-1.7B` (20 steps). Both LoRA arms train on real completions; leakage is $\max |D \Delta W|$ for both arms.
- **Regime B (generative benchmark):** `Qwen3.5-4B`, calibrated edits at several doses, `n = 30` crash-triage cases. No healing run.
- **Regime C (H100 stress test):** 12 full models (`7.2B` to `32.8B`) and 2 single-shard runs (`70.6B`, `72.7B`). The edit is a **fixed-ratio** rank-1 update ($\gamma = 0.045\,\|W\|_{\mathrm{F}}$, not Target-Shift Calibration) scored against an equal-norm random direction. Healing trains for 60 AdamW steps on the **unedited** `down_proj` toward a **synthetic target** $Y_{\text{target}} = Y + 1.5\,v_{B2}$ that lies inside $\operatorname{span}(D)$.

### 4.1 Regime B: dose-response on `Qwen3.5-4B`

| Arm (`Qwen3.5-4B`, `n = 30`, `1,536` token cap) | Summed `B1` target shift | `</think>` Closed | Strict Bug Class | Strict Culprit Frame | Strict Root Cause | Mean Tokens |
|---|---:|---:|---:|---:|---:|---:|
| **`base`** / **`rand1`** / **`rand2`** (equal-norm random) | n/a | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`g030`** (uncalibrated $\gamma = 0.30$) | n/a | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`a05`** ($\alpha = 0.5\times$) | `-3.691` | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`b1only`** ($\alpha = 1.0\times$, loop edit only) | `-7.382` | `17/30` (`56.7%`) | `15/30` (`50.0%`) | `13/30` (`43.3%`) | `11/30` (`36.7%`) | `1,044` |
| **`cal738`** ($\alpha = 1.0\times$, composed `B1+B2+B3`) | `-7.382` | **`21/30` (`70.0%`)** | **`18/30` (`60.0%`)** | **`15/30` (`50.0%`)** | **`14/30` (`46.7%`)** | `954` |
| **`a15`** ($\alpha = 1.5\times$, composed `B1+B2+B3`) | **`-11.073`** | **`30/30` (`100.0%`)** | **`23/30` (`76.7%`)** | **`23/30` (`76.7%`)** | **`19/30` (`63.3%`)** | `789` |

Exact two-sided McNemar tests against `base`: `cal738` closure $p = 9.54 \times 10^{-7}$ (21 vs. 0 discordant), `a15` closure $p = 1.86 \times 10^{-9}$ (30 vs. 0). `a15` vs. `cal738`: closure $p = 0.0039$, frame $p = 0.0078$, bug class $p = 0.125$ (not significant). In the `a05` arm only the `B1` edit was scaled; `B2` and `B3` were held at their `cal738` values.

Doses between $0.5\times$ and $1.0\times$ were not sampled.

### 4.2 Regime A: live fine-tuning

**Regime A, live LM fine-tuning after real surgery.** Both arms train on real completions after a verified edit. Same metric for both arms.

| Model | Steps | Edit evidence | `ProjectThenHealLinear` $\max\lvert D \Delta W\rvert$, `float64` / merged `F32` | Unconstrained $\max\lvert D \Delta W\rvert$ | Held-out perplexity, PTH vs. unconstrained |
|---|---:|---|---:|---:|---|
| `Qwen/Qwen3.5-0.8B` | 60 (CPU) | Composed `B1+B2+B3` GGUF, `verify-patch` exit 0 | `9.87e-18` / `2.49e-10` | `1.05e-02` | step 20: `5.435` vs. `5.423`; step 60: `12.106` vs. `12.357` (both overfit after step 20) |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 20 (ROCm GPU) | `B2` projector at L14 `o_proj`: held-out shift `-2.436` vs. random `-0.0359` (`67.9x`). `B3` calibrated at L15 `down_proj`: `+3.952` vs. `+0.179` (`22.1x`) | `5.21e-18` / `1.46e-10` | `4.64e-03` | step 20: `21.260` vs. `21.255` |

After rounding merged weights to `BF16`, both arms show leakage of order $10^{-4}$ to $10^{-3}$ (`SmolLM2` PTH `6.8e-4` to `9.5e-4`), so the guarantee does not survive a `BF16` merge; `secant heal` stores the adapter in `F32`/`F64` for that reason.

### 4.3 Regime C: H100 stress test

| Model | Params | Layers ($d_{\text{model}}$) | `B2` edit: shift vs. random (ratio, layer) | `B3` edit: shift vs. random (ratio, layer) | Unconstrained $\Vert DB\Vert _{\mathrm{F}}/\Vert B\Vert _{\mathrm{F}}$ | PTH $\max\lvert D B_{\text{locked}}\rvert$, `float64` / merged `F32` |
|---|---:|---:|---:|---:|---:|---:|
| `mistralai/Mistral-7B-Instruct-v0.3` | `7.2B` | `32` (`4096`) | `+0.2567` vs. `-0.000387` (`663.9x`, L12) | `+3.212` vs. `-0.0325` (`98.9x`, L16) | `46.79%` | `7.73e-10` / `3.87e-10` |
| `Qwen/Qwen2.5-7B-Instruct` | `7.6B` | `28` (`3584`) | `+4.654` vs. `+0.0300` (`155.1x`, L10) | `+20.82` vs. `-0.157` (`132.7x`, L18) | `31.34%` | `7.55e-10` / `6.88e-09` |
| `meta-llama/Llama-3.1-8B-Instruct` | `8.0B` | `32` (`4096`) | `+3.539` vs. `-0.0200` (`176.8x`, L24) | `+8.888` vs. `-0.252` (`35.3x`, L16) | `48.03%` | `7.62e-10` / `2.72e-09` |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `8.0B` | `32` (`4096`) | `+2.609` vs. `+0.00205` (`1,269.9x`, L20) | `+8.644` vs. `-0.0603` (`143.3x`, L16) | `52.55%` | `6.39e-10` / `2.13e-09` |
| `Qwen/Qwen3-8B` | `8.2B` | `36` (`4096`) | `+2.187` vs. `-0.00996` (`219.7x`, L12) | `+120.8` vs. `+4.15` (`29.1x`, L24) | `39.53%` | `7.14e-10` / `5.93e-09` |
| `google/gemma-2-9b-it` | `9.2B` | `42` (`3584`) | `+19.08` vs. `+0.149` (`127.9x`, L14) | `+130.0` vs. `+1.49` (`87.2x`, L26) [d] | `41.05%` | `6.53e-10` / `8.50e-10` |
| `allenai/OLMo-2-1124-13B-Instruct` | `13.7B` | `40` (`5120`) | `+1.250` vs. `+0.0121` (`103.3x`, L16) | `+7.341` vs. `-0.0279` (`262.9x`, L20) [d] | `30.73%` | `1.08e-09` / `3.33e-09` |
| `microsoft/phi-4` [p] | `14.7B` | `40` (`5120`) | `-3.399` vs. `+0.0104` (`326.9x` by magnitude, L32) [s] [d] | `+30.76` vs. `+0.632` (`48.7x`, L24) | `8.42%` | `9.51e-10` / `1.70e-08` |
| `Qwen/Qwen3-14B` | `14.8B` | `40` (`5120`) | `+4.548` vs. `+0.0180` (`252.5x`, L16) | `+63.11` vs. `-0.128` (`492.8x`, L24) | `41.33%` | `5.77e-10` / `4.68e-09` |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `14.8B` | `48` (`5120`) | `+3.762` vs. `+0.0564` (`66.7x`, L16) | `+51.94` vs. `+0.801` (`64.9x`, L32) | `32.57%` | `6.11e-10` / `2.78e-09` |
| `google/gemma-2-27b-it` | `27.2B` | `46` (`4608`) | `+542.8` vs. `+3.90` (`139.0x`, L16) | `+5,152` vs. `-26.3` (`196.0x`, L28) [d] | `11.54%` | `7.16e-10` / `7.54e-10` |
| `Qwen/Qwen3-32B` | `32.8B` | `64` (`5120`) | `+18.89` vs. `+0.119` (`158.6x`, L48) | `+41.52` vs. `-0.0813` (`510.5x`, L40) | `30.75%` | `4.82e-10` / `8.29e-09` |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` [x] | `70.6B` | `80` (`8192`) | `+8.306` vs. `-0.0155` (`537.1x`, one L32 `down_proj`) | `NOT_RUN` | `20.24%` | `5.66e-10` / `3.98e-08` |
| `Qwen/Qwen2.5-72B-Instruct` [x] | `72.7B` | `80` (`8192`) | `+9.061` vs. `-0.0675` (`134.2x`, one L40 `down_proj`) | `NOT_RUN` | `36.94%` | `9.85e-10` / `1.58e-08` |

Notes:
- **[d] Dose response is not linear for every edit.** For 20 of the 24 full-model edits, raising the dose from $1.0\times$ to $1.5\times$ multiplied the shift by `1.49` to `1.50`. The exceptions are `phi-4` `B2` (`1.04`), `OLMo-2-13B` `B3` (`1.18`), `gemma-2-9b` `B3` (`1.04`), and `gemma-2-27b` `B3` (`1.07`).
- **[s] `phi-4` `B2` moved the wrong way.** The intended shift is positive; the measured shift is `-3.399` at $1.0\times$ and `-3.519` at $1.5\times$. The `326.9x` ratio is by magnitude only.
- **[p] The `phi-4` capture looks degenerate.** Its peak `B2` $\mathrm{SNR}_{\text{proj}}$ is `27,362.3` with $\mathrm{SNR}_{\text{trace}}$ `60.4` at L32 (and up to `909.6` at L16), against `83.0` to `199.4` and about `1` to `2` on the other full models. This pattern suggests an outlier-dimension artefact rather than a sharper behavioural direction. `phi-4` was run twice; the table uses run 1 (the rerun gives `8.70%` and `9.03e-10` / `1.74e-08`, so the LoRA runs are not bit-reproducible).
- **[x] Single-shard rows.** Each used one `.safetensors` shard and one `down_proj` tensor: no activation capture and no `B3` edit. The receipt does not record how the edit direction was chosen, and the producing script is not banked. The banked run log shows the first `72.7B` attempt failing and contains no `70.6B` run, so these two rows rest on the receipt alone.
- Parameter counts are from the model cards, not from the receipts.
- `ProjectThenHealLinear` leakage over all 60 steps (not only step 60) stayed within $4.82 \times 10^{-10}$ to $1.60 \times 10^{-9}$ (`float64`) and up to $6.98 \times 10^{-8}$ (merged `F32`, `phi-4` step 0).

### 4.4 What the evaluation supports

1. **Contrastive localization:** on 11 of the 12 full H100 models the peak held-out `B2` $\mathrm{SNR}_{\text{proj}}$ is `83.0` (`Gemma-2-9B`) to `199.4` (`Qwen2.5-7B`), with split-half cosine `0.849` to `0.972`. It does **not** scale cleanly with width: `Qwen3-8B` `168.4`, `Qwen3-14B` `172.3`, `Qwen3-32B` `93.6`. `Phi-4` (`27,362.3`, $\mathrm{SNR}_{\text{trace}}$ up to `909.6`) looks like an outlier-dimension artefact, and its `B2` edit moved the projection the wrong way, so $\mathrm{SNR}_{\text{proj}}$ and the signed shift must be read together.
2. **Edit-versus-random separation is a geometric check:** in regime C the fixed-ratio edits separate from equal-norm random directions by `29.1x` to `1,269.9x` (by magnitude). A random unit direction projects onto $v$ with expected magnitude about $1/\sqrt{d_{\text{model}}}$, so ratios of this size are expected from geometry and are not evidence of behavioural change. On `SmolLM2-1.7B`, real-forward held-out witnesses give `67.9x` (`B2`) and `22.1x` (`B3`). Dose scaling from $1.0\times$ to $1.5\times$ is linear (`1.49` to `1.50`) for 20 of 24 full-model edits, not for all.
3. **The lock holds, measured per regime:** `ProjectThenHealLinear` leakage is $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$ (`float64`) and $1.46 \times 10^{-10}$ to $2.49 \times 10^{-10}$ (merged `F32`) in regime A, against $4.64 \times 10^{-3}$ to $1.05 \times 10^{-2}$ for unconstrained LoRA on the same metric (a factor of about $4.2 \times 10^7$ in merged `F32` on `0.8B`). In regime C it is $4.82 \times 10^{-10}$ to $1.08 \times 10^{-9}$ (`float64`) and $3.87 \times 10^{-10}$ to $3.98 \times 10^{-8}$ (merged `F32`) at step 60.
4. **What regime C does not show:** the locked arm cannot fit a target inside $\operatorname{span}(D)$, so its loss sits at the floor $2.25/d_{\text{model}}$ on every checkpoint; regime C says nothing about learning penalty. The unconstrained `8.42%` to `52.55%` is the fraction of the adapter's `B` norm in $\operatorname{span}(D)$ under a task that requires it, and it uses a different metric from the locked arm, so no ratio between the two columns is meaningful.

---

## References

1. **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab.** *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL, November 2, 2026. https://rikiyat.github.io/blog/weight-space-programming/
2. **Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda.** *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024 (`arXiv:2406.11717`).
3. **Junfeng Fang et al.** *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR 2025 (`arXiv:2410.02355`).
4. **Tang et al.** *LoRA-Null: Low-Rank Adaptation via Null Space for Large Language Models.* 2025 (`arXiv:2503.02659`).
5. **OPLoRA Authors.** *OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning.* AAAI 2026 (`arXiv:2510.13003`).
