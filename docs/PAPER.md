# Direct Multi-Subspace Weight Surgery, Quantized Block-Grid Equivariance, and Subspace-Locked LoRA (`Project-Then-Heal`)

**Author:** Nathan / m00dy

**Repository:** [https://github.com/professor-moody/secant](https://github.com/professor-moody/secant)  
**Date:** November 2026

---

## Plain-English Summary

Before presenting the formal propositions, here is what this paper establishes in plain words:

1. **Why we built Secant:** In *Sparks of Weight-Space Programming* (MIT CSAIL, Nov 2026), Takehi, Teoh, Zhang, and Khattab showed that editing `0.016%` of `Qwen3.5-0.8B`'s weights can suppress `<think>` reasoning loops, and called for turning ad-hoc weight inspection into a reproducible engineering discipline and auditing models directly from their weights. Secant is our answer to that challenge.
2. **Why scale calibration is necessary:** A fixed Frobenius-norm edit percentage that works on a `0.8B` model fails (`0/30` `<think>` loops closed) on `Qwen3.5-4B` because wider layers have different internal activation norms. Solving for the weight update that induces a specific inner-product shift ($\Delta_{\text{target}}$) restores closure on `Qwen3.5-4B` (`B1` loop edit at $0.5\times$, $1.0\times$, $1.5\times$ with `B2` and `B3` held at $1.0\times$: `0/30`, `21/30`, `30/30`; one greedy run per arm).
3. **Why the direct quantized writer supports `Q8_0` and refuses `Q4_K`:** Each `Qwen3.5` `ssm_out` value head spans 128 columns. Four 32-weight `Q8_0` blocks divide 128 columns evenly ($128 = 4 \times 32$), so `Q8_0` quantization commutes byte-for-byte with 128-column head permutations and allows direct 34-byte block surgery (`98.5%` suppression of a measured weight-direction component, not a refusal-rate measurement). A 256-weight `Q4_K` superblock spans two panels. In a separate identity-layout re-encoding experiment, about `67%` of the weight-direction component remained; that residual cannot be attributed solely to panel straddling.
4. **Why LoRA must be locked inside the forward pass (`Project-Then-Heal`):** Standard LoRA fine-tuning can write back into a surgical subspace whenever the task gradient has a component along the excised basis $D$, and projecting only the gradient is not enough under AdamW (Section 3.1). Parameterizing $B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$ inside the forward graph holds writer-level leakage at $9.87 \times 10^{-18}$ (`float64 math`) / $2.49 \times 10^{-10}$ (merged `F32`) during live 60-step LM fine-tuning on `Qwen3.5-0.8B`. In a separate H100 stress test on 14 checkpoints (`7.2B` to `72.7B`), where the training target deliberately lies inside the protected subspace, leakage stayed at $4.82 \times 10^{-10}$ to $1.08 \times 10^{-9}$ (`float64`) at step 60. That stress test edits with a fixed ratio rather than calibration, trains on the unedited writer, and does not measure learning quality; Section 4 states its limits.

---

## Abstract

Recent advances in weight-space programming demonstrate that specific behavioral failure modes in language models, such as infinite chain-of-thought (`<think>`) loops or over-refusals on benign defensive security prompts, can be localized to low-rank directions written into the residual stream by attention and MLP output matrices. However, moving from single-direction PyTorch experiments to production checkpoints exposes three open problems:
1. **Cross-scale & multi-subspace calibration:** Fixed Frobenius-norm edit ratios fail when ported across model widths (`0.8B` $\to$ `4B`), and converter permutations (such as `llama.cpp`'s 128-column value-head interleave $\Pi_{128}$ on `Qwen3.5` Gated DeltaNet `blk.N.ssm_out.weight` layers when $r = n_v / n_k > 1$) silently misalign edits applied directly to `.gguf` files.
2. **Direct surgery on compressed quantized weights:** Can weight edits be applied directly to block-quantized `.gguf` tensors (`Q8_0`, `Q4_K`) without re-exporting from floating-point checkpoints?
3. **Post-surgical fine-tuning drift:** When a surgically edited model is subsequently fine-tuned with LoRA and AdamW, task gradients and per-coordinate adaptive second-moment scaling push weight updates back into the excised behavioral subspace, undoing the surgery at the edited writer.

Secant implements and evaluates the following, subject to the limits of each experiment:
- First, we derive **inner-product target-shift calibration** ($\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$) and prove the exact 32-block bijection across all 24 Gated DeltaNet layers of `Qwen3.5-4B` (`24/24` bit-exact match). In a 6-arm controlled ablation on `Qwen3.5-4B` (`n = 30`), calibrated multi-subspace surgery (`cal738`) raises strict post-`</think>` completion from `0/30` (`0.0%`) to `21/30` (`70.0%`, exact two-sided McNemar $p = 9.54 \times 10^{-7}$ vs. base and equal-norm random controls), reaching `30/30` (`100.0%`, $p = 1.86 \times 10^{-9}$) with the `B1` edit at $1.5\times$ and `B2`, `B3` held at $1.0\times$ (`a15`).
- Second, we prove that $\Pi_{128}$ is block-grid equivariant for any quantization block size $B \mid 128$ (`Q8_0`, $B = 32$) and lacks that general guarantee for superblock codecs with $B \nmid 128$ (`Q4_K`, $B = 256$). Empirically, direct block surgery on `Q8_0` removes `98.5%` of the target component (`0.5172` $\to$ `0.0079`) while preserving bit-identity across all untouched 34-byte blocks, whereas experimental `Q4_K` re-encoding leaves `66.8%` of the component intact (`0.346`).
- Third, we introduce **`Project-Then-Heal`** (`secant.nn.ProjectThenHealLinear`), which parameterizes the LoRA output factor structurally inside the orthogonal complement $P_\perp = I_{d_{\text{out}}} - D^\top D$ ($B = P_\perp \tilde{B}$, where $D \in \mathbb{R}^{k \times d_{\text{out}}}$ has orthonormal rows $D D^\top = I_k$). Across 60 live AdamW steps on surgically edited `Qwen3.5-0.8B` (`L15..L17`) and 20 ROCm GPU steps on `SmolLM2-1.7B-Instruct` (`L14..L16`), `Project-Then-Heal` has similar observed training losses to unconstrained LoRA (`-89.0%` training loss on `0.8B`, minimum held-out perplexity `5.435` vs. `5.423` at step 20) while holding writer-level weight leakage $\max |D \Delta W|$ at $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$ in `float64` and $1.46 \times 10^{-10}$ to $2.49 \times 10^{-10}$ in merged `F32` (over $4 \times 10^{7}\times$ lower than unconstrained LoRA's $1.05 \times 10^{-2}$ in `F32`).

---

## 1. Multi-Subspace Orthogonalization & Target-Shift Calibration

Let $D = [q_1, \dots, q_k]^\top \in \mathbb{R}^{k \times d_{\text{out}}}$ be $k$ unit row directions in residual-stream coordinates obtained via two-pass `float64` modified Gram-Schmidt over raw contrastive activation differences. The implementation enforces the max-norm contract

$$\max_{i,j} \big|(D D^\top - I_k)_{ij}\big| < 10^{-12}$$

and fails closed otherwise (`python/secant/_linalg.py`, `ORTHONORMALITY_TOL`; the Rust `compose` path enforces the same `1e-12` bound on post-orthogonalization $\max |\cos|$). The empirically measured `float64` residual on tested bases is below $10^{-15}$ (maximum $9.99 \times 10^{-16}$ over 120 seeded Gaussian bases with $d_{\text{out}} \in \{1024, \dots, 8192\}$, $k \in \{3, 4\}$, $r \in \{8,16\}$ and five seeds); that is a measurement, not the contract.

For a writer matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ ($y = W x$), Secant supports two closed-form surgical operators:

1. **Projector subtraction (`projector-subtract`):**
   $$W' = (I_{d_{\text{out}}} - \gamma q_b q_b^\top) W = W - \gamma q_b (q_b^\top W)$$
   At $\gamma = 1$, $q_b^\top W' = 0$ in exact arithmetic.

2. **Inner-product target-shift calibrated rank-1 column edit (`rank1-columns`):**
   Given a subset $S \subset \{0, \dots, d_{\text{in}}-1\}$ of $|S| = K$ input columns and a reference input activation $h_{\text{ref}} \in \mathbb{R}^{d_{\text{in}}}$ with zero-padded support vector $h_S \in \mathbb{R}^{d_{\text{in}}}$ satisfying $\|h_S\|_2 > 0$:
   $$\Delta W = \mathrm{sign}(\Delta_{\text{target}}) \cdot \beta \, q_b \, h_S^\top, \qquad \beta = \frac{|\Delta_{\text{target}}|}{\|h_S\|_2^2}$$
   Then the realized activation shift along $q_b$ at $x = h_{\text{ref}}$ is scale-invariant:
   $$\langle q_b, \Delta W \, h_{\text{ref}} \rangle = \mathrm{sign}(\Delta_{\text{target}}) \cdot \frac{|\Delta_{\text{target}}|}{\|h_S\|_2^2} \langle h_S, h_{\text{ref}} \rangle = \Delta_{\text{target}}, \qquad \|\Delta W\|_{\mathrm{F}} = \frac{|\Delta_{\text{target}}|}{\|h_S\|_2}$$

---

Implementation qualification: Rust `compose` selects columns by descending $|h_{\text{ref},j}|$, not $|(W^\top q_b)_j|$. It uses the calibrated magnitude and the profile slot's sign; a negative CLI value does not override every slot's sign (`crates/secant-core/src/subspace.rs:578`, `:601`, `:629`). The identity above holds in exact arithmetic at one reference activation. Storage rounding, other inputs and downstream behavior require separate evaluation.

## 2. Container Geometry & The $\Pi_{128}$ Quantization Commutation Theorem

### 2.1 Logical `[d_out, d_in]` Orientation Contract
In HuggingFace `.safetensors`, a 2-D linear weight matrix has `shape = [d_out, d_in]` with `shape[1] = d_in` contiguous. In `ggml` / `.gguf`, `ne = [d_in, d_out]` lists the contiguous axis first (`ne[0] = d_in`). Consequently, both formats share the identical row-major element index:

$$\mathrm{offset}(\text{out}, \text{in}) = \text{out} \cdot d_{\text{in}} + \text{in}$$

### 2.2 Gated DeltaNet `ssm_out` Bijection ($\Pi_{128}$)
On `Qwen3.5`, `llama.cpp`'s `convert_hf_to_gguf.py` reorders the input columns ($d_{\text{in}} = n_v \cdot d_v$, where $d_v = 128$ and $r = n_v / n_k$) of `blk.N.ssm_out.weight` from key-head-major to value-head-major. Writing a HuggingFace 128-column block index as $h = k \cdot r + j$ ($k \in \{0,\dots,n_k-1\}, j \in \{0,\dots,r-1\}$), its GGUF block position is:

$$\pi(k \cdot r + j) = j \cdot n_k + k, \qquad \text{col}_{\text{GGUF}}(h \cdot 128 + w) = \pi(h) \cdot 128 + w \quad (w \in \{0,\dots,127\})$$

When $r = 1$ (`Qwen3.5-0.8B`, $n_k = n_v = 16$), $\pi$ is the identity permutation. When $r = 2$ (`Qwen3.5-4B`, $n_k = 16, n_v = 32$), $\Pi_{128}$ is a non-trivial bijection on the 32 panels of width 128: the transpose of a $(n_k, r) = (16, 2)$ grid into a $(2, 16)$ grid. Because $r \neq n_k$, it is **not** an involution (for example $\pi(1) = 16$ but $\pi(16) = 8$). Its inverse is

$$\pi^{-1}(g) = (g \bmod n_k) \cdot r + \lfloor g / n_k \rfloor .$$

### 2.3 Block-Grid Equivariance vs. Superblock Straddling
Let $Q_B$ denote a deterministic row-wise block quantizer that partitions each row of length $d_{\text{in}}$ into aligned contiguous blocks of $B$ elements and uses the same encoder independently at every block position. Assume complete blocks, aligned row/panel boundaries and finite encoded scales.

- **Proposition 1 (Block-Grid Equivariance for $B \mid 128$):** Because $\Pi_{128}$ permutes contiguous panels of width $128$, whenever $B \mid 128$ (such as `Q8_0` with $B = 32$, since $128 = 4 \times 32$), every 128-element panel is a disjoint union of $128 / B$ complete quantization blocks. Therefore, block quantization commutes exactly with $\Pi_{128}$ at the byte level:
  $$Q_{32}(\Pi_{128}(W)) = \Pi_{128}^{\text{block}}(Q_{32}(W)), \qquad \|D_{32}(Q_{32}(\Pi_{128}(W))) - \Pi_{128}(D_{32}(Q_{32}(W)))\|_{\max} = 0$$
- **Proposition 2 (Superblock Straddling for `Q4_K`, $B = 256$):** Each superblock shares FP16 scale $d$ and minimum $d_{\min}$ across two adjacent 128-element panels. A permutation that splits these pairs need not commute with quantization. Editing one panel can change its neighbor's reconstructed values through the shared fit. Neither effect is inevitable for every input; identity permutations still commute. This statement does not assign the `Q4_K` scale/minimum layout to other K-quant codecs.
- **Remark on Lossy Requantization (`Path 1` vs. `Path 2`):** Grid equivariance of $\Pi_{128}$ does not imply that nonlinear quantization commutes with additive weight edits: $D_B(Q_B(W + \Delta W)) \neq D_B(Q_B(D_B(Q_B(W)) + \Delta W))$. Writing $\hat{Q}_B = D_B \circ Q_B$ and $e_0 = \hat{Q}_B(W) - W$, $e_1 = \hat{Q}_B(W + \Delta W) - (W + \Delta W)$, $e_2 = \hat{Q}_B(\hat{Q}_B(W) + \Delta W) - (\hat{Q}_B(W) + \Delta W)$, the exact identity for any deterministic block-local codec is
  $$\hat{Q}_B(W + \Delta W) - \hat{Q}_B\big(\hat{Q}_B(W) + \Delta W\big) = e_1 - e_0 - e_2 ,$$
  so the Path 1 vs. Path 2 gap is controlled by **three** quantization errors on the touched blocks. For `Q8_0` the post-edit block scale drifts by at most $\|\Delta W_b\|_\infty / 127$, and each element of a touched block obeys $|e_1 - e_0 - e_2| \le \kappa\big((3 + \kappa)\|W_b\|_\infty + 2\|\Delta W_b\|_\infty\big)$ with $\kappa = 1/254 + 2^{-11}$ (ideal arithmetic apart from FP16 scale rounding, with all three scales in the normal range; derivation in [Quantized Surgery](quantized-surgery.md)). `Q4_K` has no such proved bound here and is characterized empirically. Moreover, because `Q8_0` fits an independent FP16 scale $d = \max_i |x_i| / 127$ per 32-element block, direct `Q8_0` block surgery (`Path 2`) achieves `98.5%` directional suppression (`0.5172` $\to$ `0.0079`, vs. `99.4%` for `Path 1`), whereas `Q4_K` `Path 2` leaves `66.8%` of the original weight-direction component intact (`0.346`).

---

## 3. `Project-Then-Heal`: Structural Orthogonal-Complement LoRA

### 3.1 Why Gradient Projection Fails Under AdamW
Suppose a writer matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ has had an orthonormal row basis $D \in \mathbb{R}^{k \times d_{\text{out}}}$ ($D D^\top = I_k$) surgically removed, and we attach a LoRA adapter $\Delta W = s B A$ ($s = \alpha_{\text{lora}} / r$).

If one projects only the gradient $g_t = P_\perp \nabla_B \mathcal{L}_t$ ($P_\perp = I_{d_{\text{out}}} - D^\top D$) and passes $g_t$ into AdamW, the first moment $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ remains in $\mathrm{col}(P_\perp)$ when its initial state is in that subspace (including zero initialization), **but** AdamW divides coordinatewise by $\sqrt{\hat{v}_t} + \epsilon$:

$$\Delta B_t = -\eta \left( \mathrm{diag}\!\left(\frac{1}{\sqrt{\hat{v}_t} + \epsilon}\right) \hat{m}_t + \lambda B_{t-1} \right)$$

Because a diagonal matrix $\Lambda_t = \mathrm{diag}((\sqrt{\hat{v}_t} + \epsilon)^{-1})$ with non-constant diagonal entries need not commute with a non-axis-aligned projector $P_\perp$ ($D \Lambda_t P_\perp \neq 0$), AdamW's step can rotate $B_t$ back into $\mathrm{span}(D^\top)$. This is an argument about the gradient-projection variant. Regimes A and C in Section 4 include no gradient-projection-only arm; a separate 3-arm synthetic-regression test on real layer-16 `down_proj` weights (`Qwen3.5-0.8B`, `Qwen3.5-4B`, `Qwen2.5-7B`; seeded random $D$, $k = 3$; 3 seeds) does, and is reported in [Project-Then-Heal, Section 3](project-then-heal.md). There the gradient-projected arm put `1.56%` to `2.98%` of $\|\Delta W\|_{\mathrm{F}}$ into $\mathrm{span}(D)$ at step 60 on the orthogonal target, unchanged between weight decay `0.01` and `0.0`, while `ProjectThenHealLinear` stayed at $5.42 \times 10^{-10}$ to $1.05 \times 10^{-9}$ (post-hoc learning rate `3e-3`; the preregistered `1e-2` run gives the same ordering). On the 15-model H100 Stage-1 run (Section 4.4) the gradient-projected arm again leaks on every model (`0.33% to 1.67%`) while `ProjectThenHealLinear` stays at `1.78e-10 to 8.67e-10`.

### 3.2 Structural Parameterization ($B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$)
`ProjectThenHealLinear` instead applies $P_\perp$ twice in `float64` inside the forward graph:

$$B_{\text{locked}}(\tilde{B}) = P_\perp(P_\perp(\tilde{B})), \qquad y = W_0 x + s \, B_{\text{locked}}(\tilde{B}) (A x)$$

By associativity, for **any** parameter state $(\tilde{B}_t, A_t)$ produced by any optimizer:

$$D \Delta W_t = s \, (D P_\perp) P_\perp(\tilde{B}_t) A_t = s \, (D - (D D^\top) D) P_\perp(\tilde{B}_t) A_t = 0$$

In exact arithmetic $P_\perp^2 = P_\perp$, so the second projection changes nothing. In `float64` it is an observed numerical refinement, not a proved bound: on 120 seeded Gaussian test cases it lowered $\max |D B_{\text{locked}}|$ by $1.8\times$ to $17.7\times$ (median $5.6\times$) relative to one pass. Measured live writer leakage $\max |D\Delta W|$ is $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$ in the live CPU/ROCm runs. The distinct H100 factor metric $\max |DB_{\text{locked}}|$ is $4.82 \times 10^{-10}$ to $1.08 \times 10^{-9}$ at step 60 across the 14 H100 checkpoints (`7.2B` to `72.7B`; at most $1.60 \times 10^{-9}$ at any recorded step). The latter metric is evaluated in float64 on a working-dtype factor, not an all-float64 forward pass.

### 3.3 Writer-Level Invariance vs. Downstream Residual-Stream Propagation
The algebraic guarantee $D \Delta W_t = 0$ is exact at every protected writer's output (on `Qwen3.5-0.8B` at most $1.25 \times 10^{-8}$ in `F32` at the `L15..L17` writer outputs, and $1.70 \times 10^{-8}$ in the residual stream at `L15`). When multiple layers are stacked, an orthogonal perturbation $\delta h_\ell \in \mathrm{span}(D^\top)^\perp$ emitted at layer $\ell$ passes through subsequent RMSNorm, attention, and MLP matrices $W_{\ell'}$ ($\ell' > \ell$) for which $D W_{\ell'} P_\perp \neq 0$ in general. Empirically on `Qwen3.5-0.8B` (`Step 60`), locking `L15..L17` eliminates direct writer leakage ($4.2 \times 10^7\times$ reduction in merged `F32`) and cuts downstream pull-back along the `B1` loop direction at `resid L19` by more than half (`+0.276` vs. `+0.635` unconstrained).

---

## 4. Cross-Architecture Empirical Evaluation (`0.8B` to `72.7B`)

Building on Takehi, Teoh, Zhang, and Khattab's `Qwen3.5-0.8B` `<think>` reasoning-loop experiment [1], we ran Secant on **seventeen checkpoints** (`Qwen3.5-0.8B`, `SmolLM2-1.7B`, `Qwen3.5-4B`, `Mistral-7B-v0.3`, `Qwen2.5-7B`, `Llama-3.1-8B`, `DeepSeek-R1-Distill-Llama-8B`, `Qwen3-8B`, `Gemma-2-9B`, `OLMo-2-13B`, `Phi-4`, `Qwen3-14B`, `DeepSeek-R1-Distill-Qwen-14B`, `Gemma-2-27B`, `Qwen3-32B`, `DeepSeek-R1-Distill-Llama-70B`, `Qwen2.5-72B`; $d_{\text{model}} \in [1024, 8192]$) on CPU, AMD ROCm (`gfx1151`), and NVIDIA H100 80GB hardware. They fall into three regimes that measure different things:

- **Regime A (live LM fine-tuning after real surgery):** `Qwen3.5-0.8B` (60 AdamW steps) and `SmolLM2-1.7B` (20 steps). Both LoRA arms train on real completions; leakage is $\max |D \Delta W|$ for both arms.
- **Regime B (generative benchmark):** `Qwen3.5-4B`, calibrated edits at several doses, `n = 30` crash-triage cases. No healing run.
- **Regime C (H100 stress test):** 12 full models (`7.2B` to `32.8B`) and 2 single-shard runs (`70.6B`, `72.7B`). The edit is a **fixed-ratio** rank-1 update ($\gamma = 0.045\,\|W\|_{\mathrm{F}}$, not Target-Shift Calibration) scored against an equal-norm random direction. Healing trains for 60 AdamW steps on the **unedited** `down_proj` toward a **synthetic target** $Y_{\text{target}} = Y + 1.5\,v_{B2}$ that lies inside $\mathrm{span}(D)$.
- **H100 Stage-1 (Section 4.4):** 15 full models (`0.8B` to `32.8B`, every checkpoint above except the two single-shard runs). Full Target-Shift Calibration on `B2` and `B3`, and the three-arm healing test (unconstrained, gradient-projected, `ProjectThenHealLinear`) on the **surgically edited** writer with an orthogonal synthetic target plus real-token cross-entropy, all arms scored with $\lVert D\,\Delta W\rVert_{\mathrm{F}} / \lVert\Delta W\rVert_{\mathrm{F}}$. One seed per model; benchmarks `NOT_RUN`.

### 4.1 Regime B: dose-response on `Qwen3.5-4B`

| Arm (`Qwen3.5-4B`, `n = 30`, `1,536` token cap) | Summed `B1` target shift | `</think>` Closed | Strict Bug Class | Strict Culprit Frame | Strict Root Cause | Mean Tokens |
|---|---:|---:|---:|---:|---:|---:|
| **`base`** / **`rand1`** / **`rand2`** (equal-norm random) | n/a | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`g030`** (uncalibrated $\gamma = 0.30$) | n/a | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`a05`** (`B1` at $0.5\times$; `B2` and `B3` held at $1.0\times$) | `-3.691` | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`b1only`** ($\alpha = 1.0\times$, loop edit only) | `-7.382` | `17/30` (`56.7%`) | `15/30` (`50.0%`) | `13/30` (`43.3%`) | `11/30` (`36.7%`) | `1,044` |
| **`cal738`** ($\alpha = 1.0\times$, composed `B1+B2+B3`) | `-7.382` | **`21/30` (`70.0%`)** | **`18/30` (`60.0%`)** | **`15/30` (`50.0%`)** | **`14/30` (`46.7%`)** | `954` |
| **`a15`** (`B1` at $1.5\times$; `B2` and `B3` held at $1.0\times$) | **`-11.073`** | **`30/30` (`100.0%`)** | **`23/30` (`76.7%`)** | **`23/30` (`76.7%`)** | **`19/30` (`63.3%`)** | `789` |

Exact two-sided McNemar tests against `base`: `cal738` closure $p = 9.54 \times 10^{-7}$ (21 vs. 0 discordant), `a15` closure $p = 1.86 \times 10^{-9}$ (30 vs. 0). `a15` vs. `cal738`: closure $p = 0.0039$, frame $p = 0.0078$, bug class $p = 0.125$ (not significant). In both the `a05` and `a15` arms only the `B1` edits (`B1a`, `B1b`) were scaled; `B2` and `B3` were held at their `cal738` ($1.0\times$) values in every arm (`a05`, `a10` and `a15` plans checked edit by edit).

Doses between $0.5\times$ and $1.0\times$ were not sampled.

### 4.2 Regime A: live fine-tuning

**Regime A, live LM fine-tuning after real surgery.** Both arms train on real completions after a verified edit. Same metric for both arms.

| Model | Steps | Edit evidence | `ProjectThenHealLinear` $\max\lvert D \Delta W\rvert$, `float64` / merged `F32` | Unconstrained $\max\lvert D \Delta W\rvert$ | Held-out perplexity, PTH vs. unconstrained |
|---|---:|---|---:|---:|---|
| `Qwen/Qwen3.5-0.8B` | 60 (CPU) | Composed `B1+B2+B3` GGUF, `verify-patch` exit 0 | `9.87e-18` / `2.49e-10` | `1.05e-02` | step 20: `5.435` vs. `5.423`; step 60: `12.106` vs. `12.357` (both overfit after step 20) |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 20 (ROCm GPU) | `B2` projector at L14 `o_proj`: held-out shift `-2.436` vs. random `-0.0359` (`67.9x`). `B3` calibrated at L15 `down_proj`: `+3.952` vs. `+0.179` (`22.1x`) | `5.21e-18` / `1.46e-10` | `4.64e-03` | step 20: `21.260` vs. `21.255` |

After rounding merged weights to `BF16`, both arms show leakage of order $10^{-4}$ to $10^{-3}$ (`SmolLM2` PTH `6.8e-4` to `9.5e-4`), so the guarantee does not survive a `BF16` merge; `secant heal` stores the adapter in `F32`/`F64` for that reason.

Precision note for Regime A: the `float64` column is the receipt's `leak_inf_float64_math` recomputation of the unrounded projected delta. It is not the leakage of an all-float64 language-model forward pass. The working-dtype factor and the merged checkpoint are separate measurements.

### 4.3 Regime C: H100 stress test

In the final column, the first value is $\max|DB_{\text{locked}}|$ evaluated in float64 on the F32 working factor. The second is $\max|D(W_{\text{merged,F32}}-W_{\text{base}})|$, a different matrix. The historical receipt field `leakage_f64` names the evaluation dtype, not the factor's storage dtype.

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

### 4.4 H100 Stage-1: calibration and the three-arm test on 15 models

The 15-model H100 Stage-1 run (`0.8B` to `32.8B`, receipt `irrefutable-15models-stage1-receipt.json`, SHA-256 `00f4fe2afc4ff630c7c9ac55b5b2c0a85342477a5b32a0b0c455abd5b4c5056f`) applied full Target-Shift Calibration to `B2` and `B3` on every model, with $\lvert\Delta_{\text{target}}\rvert = 7.382$ and 128 selected input columns. Across `B2`, $\lVert h_S\rVert_2$ spans `1.41 to 15.91` and $\beta$ spans `0.0292 to 3.7192`; across `B3`, `2.53 to 27.47` and `0.0098 to 1.1508`. Every one of the 30 edits realizes its target at $h_{\text{ref}}$ with maximum error approximately `7.99e-15` in `float64` (10 of them exactly 0). Each random control is norm-matched. This run has no `BF16`-storage witness and its behavioral benchmarks are `NOT_RUN`, so it establishes that the step is solved correctly at every width tested, not what the edit does to generated output.

| Model | Size | `B2` layer | `B2` $\lVert h_S\rVert_2$ | `B2` $\beta$ | `B3` layer | `B3` $\lVert h_S\rVert_2$ | `B3` $\beta$ | max float64 shift error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3.5-0.8B` | `0.8B` | `L17` | `3.57` | `0.5801` | `L14` | `4.53` | `0.3595` | `8.9e-16` |
| `SmolLM2-1.7B` | `1.7B` | `L17` | `14.14` | `0.0369` | `L12` | `5.20` | `0.2727` | `1.8e-15` |
| `Qwen3.5-4B` | `4B` | `L23` | `12.35` | `0.0484` | `L16` | `4.40` | `0.3805` | `3.6e-15` |
| `Mistral-7B-v0.3` | `7.2B` | `L12` | `2.81` | `0.9352` | `L19` | `12.03` | `0.0510` | `3.6e-15` |
| `Qwen2.5-7B` | `7.6B` | `L8` | `7.77` | `0.1224` | `L14` | `7.05` | `0.1484` | `2.7e-15` |
| `Llama-3.1-8B` | `8B` | `L23` | `4.03` | `0.4556` | `L16` | `3.82` | `0.5070` | `1.8e-15` |
| `R1-Llama-8B` | `8B` | `L23` | `4.12` | `0.4343` | `L16` | `3.96` | `0.4701` | `5.3e-15` |
| `Qwen3-8B` | `8.2B` | `L14` | `4.33` | `0.3931` | `L21` | `9.82` | `0.0766` | `5.3e-15` |
| `Gemma-2-9B` | `9.2B` | `L16` | `1.41` | `3.7192` | `L25` | `2.53` | `1.1508` | `1.8e-15` |
| `OLMo-2-13B` | `13.7B` | `L16` | `8.08` | `0.1130` | `L20` | `15.06` | `0.0325` | `3.6e-15` |
| `Phi-4` | `14.7B` | `L12` | `2.54` | `1.1436` | `L29` | `20.61` | `0.0174` | `4.4e-15` |
| `Qwen3-14B` | `14.8B` | `L16` | `4.97` | `0.2994` | `L23` | `11.62` | `0.0547` | `1.8e-15` |
| `R1-Qwen-14B` | `14.8B` | `L28` | `14.66` | `0.0343` | `L35` | `20.44` | `0.0177` | `3.6e-15` |
| `Gemma-2-27B` | `27.2B` | `L14` | `9.35` | `0.0845` | `L27` | `27.47` | `0.0098` | `8.0e-15` |
| `Qwen3-32B` | `32.8B` | `L47` | `15.91` | `0.0292` | `L25` | `10.00` | `0.0738` | `3.6e-15` |

All three arms use one metric, $\lVert D\,\Delta W\rVert_{\mathrm{F}} / \lVert\Delta W\rVert_{\mathrm{F}}$, at step 60. Arm A (unconstrained) leaks `0.82% to 8.51%`; Arm B (gradient-projected, $P_\perp \nabla_B$ applied after backward and immediately before `AdamW.step`) leaks `0.33% to 1.67%`; Arm C (`ProjectThenHealLinear`) stays at `1.78e-10 to 8.67e-10`. Arm B is nonzero on all 15 models, as the Section 3.1 argument predicts for gradient projection under AdamW, and stays at least `3.8e6` times above Arm C. Unlike the three-model strix test, Arm B sits **below** Arm A here on 15 of 15 models; the task differs (orthogonal synthetic MSE plus real-token cross-entropy), so the A-versus-B ordering is task-dependent and only the B-versus-C gap is common to both runs.

| Model | Size | Layer | A: unconstrained | B: gradient-projected | C: `ProjectThenHealLinear` | C: max abs |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen3.5-0.8B` | `0.8B` | `17` | `7.60%` | `0.61%` | `8.67e-10` | `4.00e-10` |
| `SmolLM2-1.7B` | `1.7B` | `17` | `2.95%` | `1.62%` | `5.41e-10` | `1.97e-10` |
| `Qwen3.5-4B` | `4B` | `23` | `2.22%` | `1.67%` | `5.81e-10` | `2.25e-10` |
| `Mistral-7B-v0.3` | `7.2B` | `12` | `2.17%` | `0.92%` | `3.38e-10` | `2.21e-10` |
| `Qwen2.5-7B` | `7.6B` | `8` | `2.04%` | `0.49%` | `4.71e-10` | `2.43e-10` |
| `Llama-3.1-8B` | `8B` | `23` | `0.82%` | `0.80%` | `4.36e-10` | `3.51e-10` |
| `R1-Llama-8B` | `8B` | `23` | `1.59%` | `0.60%` | `4.50e-10` | `2.32e-10` |
| `Qwen3-8B` | `8.2B` | `14` | `2.02%` | `0.58%` | `3.14e-10` | `2.14e-10` |
| `Gemma-2-9B` | `9.2B` | `16` | `8.51%` | `0.35%` | `5.46e-10` | `6.67e-10` |
| `OLMo-2-13B` | `13.7B` | `16` | `1.70%` | `0.44%` | `1.78e-10` | `1.87e-10` |
| `Phi-4` | `14.7B` | `12` | `0.86%` | `0.33%` | `3.26e-10` | `3.65e-10` |
| `Qwen3-14B` | `14.8B` | `16` | `2.13%` | `0.60%` | `2.69e-10` | `1.81e-10` |
| `R1-Qwen-14B` | `14.8B` | `28` | `1.23%` | `0.77%` | `3.53e-10` | `2.06e-10` |
| `Gemma-2-27B` | `27.2B` | `14` | `1.98%` | `1.31%` | `3.49e-10` | `1.03e-10` |
| `Qwen3-32B` | `32.8B` | `47` | `2.04%` | `0.75%` | `3.07e-10` | `1.22e-10` |

**Limits.** One seed (`20261105`) per model and one edited layer per model; no confidence interval across seeds. The receipt marks public benchmarks `NOT_RUN`, so it says nothing about task quality after healing. Its standalone `secant` CLI cross-check (`read` and `sanitize` on one shard) is `PASS` on 2 of 15 models and `HOLD` on the rest (13: exit `2` or a 120 s timeout in `read`); that cross-check is separate from the healing and calibration numbers above, which are all `PASS`.

### 4.5 What the evaluation supports

1. **Contrastive localization:** on 11 of the 12 full H100 models the peak held-out `B2` $\mathrm{SNR}_{\text{proj}}$ is `83.0` (`Gemma-2-9B`) to `199.4` (`Qwen2.5-7B`), with split-half cosine `0.849` to `0.972`. It does **not** scale cleanly with width: `Qwen3-8B` `168.4`, `Qwen3-14B` `172.3`, `Qwen3-32B` `93.6`. `Phi-4` (`27,362.3`, $\mathrm{SNR}_{\text{trace}}$ up to `909.6`) looks like an outlier-dimension artefact, and its `B2` edit moved the projection the wrong way, so $\mathrm{SNR}_{\text{proj}}$ and the signed shift must be read together.
2. **Edit-versus-random separation is a geometric check:** in regime C the fixed-ratio edits separate from equal-norm random directions by `29.1x` to `1,269.9x` (by magnitude). A random unit direction projects onto $v$ with expected magnitude about $1/\sqrt{d_{\text{model}}}$, so ratios of this size are expected from geometry and are not evidence of behavioural change. On `SmolLM2-1.7B`, real-forward held-out witnesses give `67.9x` (`B2`) and `22.1x` (`B3`). Dose scaling from $1.0\times$ to $1.5\times$ is linear (`1.49` to `1.50`) for 20 of 24 full-model edits, not for all.
3. **The lock holds, measured per regime:** `ProjectThenHealLinear` leakage is $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$ (`float64`) and $1.46 \times 10^{-10}$ to $2.49 \times 10^{-10}$ (merged `F32`) in regime A, against $4.64 \times 10^{-3}$ to $1.05 \times 10^{-2}$ for unconstrained LoRA on the same metric (a factor of about $4.2 \times 10^7$ in merged `F32` on `0.8B`). In regime C it is $4.82 \times 10^{-10}$ to $1.08 \times 10^{-9}$ (`float64`) and $3.87 \times 10^{-10}$ to $3.98 \times 10^{-8}$ (merged `F32`) at step 60.
4. **What regime C does not show:** the locked arm cannot fit a target inside $\mathrm{span}(D)$, so its loss sits at the floor $2.25/d_{\text{model}}$ on every checkpoint; regime C says nothing about learning penalty. The unconstrained `8.42%` to `52.55%` is the fraction of the adapter's `B` norm in $\mathrm{span}(D)$ under a task that requires it, and it uses a different metric from the locked arm, so no ratio between the two columns is meaningful.

---

## 5. Related work and limits of novelty

Orthogonal projection and constrained LoRA predate Secant. [OPLoRA by Yifeng Xiong and Xiaohui Xie](https://arxiv.org/abs/2510.13003) protects singular subspaces of frozen weights with left and right projections. Secant's documented lock uses supplied output directions and constrains the output factor. Repeating one left projection for numerical refinement is not OPLoRA's double-sided construction.

[Safe LoRA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/77baa7c2a3a675823e89131698fd6e19-Abstract-Conference.html) (NeurIPS 2024) projects selected LoRA updates into a safety-aligned subspace derived from base/aligned checkpoint differences. [OGPSA](https://arxiv.org/abs/2602.07892) projects safety gradients against a capability-gradient subspace with a first-order preservation interpretation. Neither is the same contract as a fixed writer-level output constraint. The synthetic gradient-hook arm here is not a reproduction of OGPSA and establishes no comparative performance ranking.

[Heretic](https://github.com/p-e-w/heretic), [Obliteratus](https://github.com/elder-plinius/OBLITERATUS), and [model-unfetter](https://github.com/josepha-mayo/model-unfetter) are community directional-ablation toolkits. Their documented scope overlaps direction extraction and weight modification. No pinned head-to-head evaluation is reported here, and this paper does not claim those projects lack recovery, quantized workflows, or verification capabilities.

The supported contribution is the documented integration and measured contracts for container edits, block preservation and supplied-subspace checking. These experiments do not establish priority for projection itself, superior safety, or better general utility than the cited methods. Bibliographic correction: [arXiv:2410.19278](https://arxiv.org/abs/2410.19278) is Farrell, Lau and Conmy's SAE unlearning paper, not OPLoRA. An AAAI 2025 attribution for OPLoRA is unsupported by the checked primary record.

## References

1. **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab.** *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL, November 2, 2026. https://rikiyat.github.io/blog/weight-space-programming/
2. **Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda.** *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024 (`arXiv:2406.11717`).
3. **Junfeng Fang et al.** *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR 2025 (`arXiv:2410.02355`).
4. **Pengwei Tang et al.** [*Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge*](https://arxiv.org/abs/2503.02659). LoRA-Null, arXiv:2503.02659v2; the arXiv record reports AAAI 2026 acceptance.
5. **Yifeng Xiong and Xiaohui Xie.** [*OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning*](https://arxiv.org/abs/2510.13003). arXiv:2510.13003, 2025. Conference venue not established by the checked record.
6. **Xuechen Li et al.** [*SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation*](https://arxiv.org/abs/2501.01765). ICLR 2025 (`arXiv:2501.01765`).
7. **Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov.** [*Locating and Editing Factual Associations in GPT (ROME)*](https://arxiv.org/abs/2202.05262). NeurIPS 2022 (`arXiv:2202.05262`).
