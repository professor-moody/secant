# Direct Multi-Subspace Weight Surgery, Quantized Block-Grid Equivariance, and Subspace-Locked LoRA (`Project-Then-Heal`)

## Abstract

Specific failure modes in large language models, such as getting stuck in infinite reasoning (`<think>`) loops, refusing benign security prompts, or stopping at surface-level bug summaries, can often be traced to specific directions in the model's weight matrices. In principle, subtracting or amplifying those directions directly in the weights can fix the behavior without retraining. In practice, moving from small PyTorch experiments to real production model files fails in three ways: (1) an edit strength tuned on a small model (`0.8B`) under- or over-steers a larger model (`4B` to `32B`) because internal activation magnitudes change with width, and format converters such as `llama.cpp` shuffle weight columns inside `.gguf` files; (2) compressed `.gguf` checkpoints pack weights into shared quantized blocks (`Q8_0` and `Q4_K`), where naive edits corrupt neighboring weights or require re-exporting the entire model; and (3) if a surgically edited model is later fine-tuned with LoRA and `AdamW`, the optimizer's per-coordinate variance scaling rotates weight updates right back into the removed direction and undoes the edit.

Secant is a standalone Rust and PyTorch engine that solves all three problems directly on `.safetensors` and `.gguf` checkpoints. First, **Target-Shift Calibration** measures a layer's reference activation energy and solves for the exact edit step size needed to produce a consistent behavioral shift across model scales (`0.8B` to `32.8B`), while automatically un-shuffling `.gguf` head layouts. In controlled evaluations, calibrated surgery eliminates infinite `<think>` loops on `Qwen3.5-4B` (raising completion from `0/30` to `21/30` at standard dose and `30/30` at `1.5x` dose, `p = 1.86e-9`), lifts paired C/C++ vulnerability root-cause detection on `Qwen2.5-Coder-32B-Instruct` from `6.7%` (`2/30`) to `40.0%` (`12/30`, `p = 0.00195`), and improves `MATH-500` (`+7.0%`) and `HumanEval` (`+3.12%`) on `Qwen3-32B`. Second, we show that 32-weight `Q8_0` blocks align cleanly with 128-column attention heads (`128 = 4 x 32`), allowing Secant to memory-map `.gguf` files and edit only the touched 34-byte `Q8_0` blocks in place (`98.5%` target direction removal while leaving `98%+` of the file byte-for-byte identical), whereas 256-weight `Q4_K` superblocks span across two heads and are safely refused. Third, **`Project-Then-Heal`** (`secant.nn.ProjectThenHealLinear`) prevents fine-tuning from undoing weight surgery by locking the LoRA output matrix inside the forward pass so it can only write perpendicular to the removed directions. Whereas projecting gradients before `AdamW` leaks on `100%` of steps (`108/108` runs and `15/15` models tested on H100), `Project-Then-Heal` matches standard LoRA training loss while holding writer-level leakage at floating-point rounding precision (`5.21e-18` to `9.87e-18` in 64-bit float; `1.78e-10` to `8.67e-10` in 32-bit float across 15 models from `0.8B` to `32.8B`).

---

## Plain-English Summary & Navigation Guide

| Section | Core Question | Plain-English Takeaway |
| :--- | :--- | :--- |
| **[1. Multi-Subspace Calibration](#multi-subspace-orthogonalization-target-shift-calibration)** | Why do weight edits break when moved from a `0.8B` model to a `4B` or `32B` model? | Wider models have larger internal activations, so a fixed percentage weight edit under- or over-steers. Solving for the step size $\beta$ from the layer's measured activation energy ($\|h_S\|_2^2$) produces the exact intended shift at any width (`<= 7.99e-15` error across 15 models). |
| **[2. Quantized Block Surgery](#container-geometry-the-pi-128-quantization-commutation-theorem)** | How can we edit a compressed `.gguf` file in milliseconds without PyTorch or a GPU? | Each 128-column attention/SSM head divides into four 32-weight `Q8_0` blocks ($128 = 4 \times 32$). Secant memory-maps the `.gguf` file and edits only the touched 34-byte `Q8_0` blocks in place (`98%+` of the file stays bit-for-bit identical), while refusing 256-weight `Q4_K` superblocks that straddle two heads. |
| **[3. `Project-Then-Heal` LoRA](#project-then-heal-structural-orthogonal-complement-lora)** | Why does standard LoRA fine-tuning undo weight surgery even if you project gradients? | `AdamW` divides each coordinate by its running variance $\sqrt{\hat{v}_t} + \epsilon$, which rotates projected gradients back into the excised subspace on `100%` of steps (`108/108` runs). Locking the LoRA $B$ matrix inside the forward pass ($B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$) keeps writer leakage at machine precision (`~10^-18` in `float64`). |
| **[4. Empirical Evaluation](#cross-architecture-empirical-evaluation-0-8b-to-72-7b)** | Does calibrated surgery hold up across model families, coding benchmarks, and C/C++ auditing? | Validated across 17 checkpoints (`0.6B` to `72.7B`): `30/30` `<think>` loop closure on `Qwen3.5-4B` ($p = 1.86 \times 10^{-9}$), `6.7% -> 40.0%` (`+10` crashes, $p = 0.00195$) on our 30-case C/C++ sanitizer suite on `Qwen2.5-Coder-32B-Instruct`, and `+7.0%` `MATH-500` / `+3.12%` `HumanEval` on `Qwen3-32B`. |
| **[5. Related Work & Scope Limits](#related-work-and-limits-of-novelty)** | What is prior art vs. new in Secant, and where do the guarantees stop? | Orthogonal projection and subspace-constrained LoRA build on ROME, AlphaEdit, Safe LoRA, SaLoRA, OPLoRA, and OGPSA. Secant's contributions are Target-Shift Calibration, the $\Pi_{128}$ `Q8_0`/`Q4_K` block-grid theorem, the proof of AdamW preconditioner rotation vs. forward-graph double projection, and byte-verified `mmap` tooling. |

---

(multi-subspace-orthogonalization-target-shift-calibration)=
## 1. Multi-Subspace Orthogonalization & Target-Shift Calibration

> **Plain-English Intuition:** When you edit a model to fix multiple behaviors at once (for example, stopping infinite `<think>` loops while also boosting root-cause security analysis), the raw steering directions overlap slightly. If you apply them one after another without removing that overlap, the second edit partly overwrites the first. Secant first makes the directions strictly perpendicular in 64-bit floating point (Gram-Schmidt), and then sizes each edit using the layer's actual activation energy so the edit works identically on a `0.8B`, `4B`, or `32B` model.

### 1.1 Two-Pass `float64` Orthonormalization

Let $D = [q_1, \dots, q_k]^\top \in \mathbb{R}^{k \times d_{\text{out}}}$ be $k$ unit row directions in residual-stream coordinates obtained via two-pass `float64` modified Gram-Schmidt over contrastive activation differences. Secant enforces the strict orthonormality gate

$$\max_{i,j} \big|(D D^\top - I_k)_{ij}\big| < 10^{-12}$$

and fails closed if the input directions are linearly dependent or exceed this tolerance. Across 120 seeded Gaussian test bases ($d_{\text{out}} \in \{1024, \dots, 8192\}$, $k \in \{3, 4\}$), the empirically measured two-pass `float64` residual is at most $9.99 \times 10^{-16}$.

### 1.2 Closed-Form Surgical Operators

For a linear writer matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ mapping input activations $x \in \mathbb{R}^{d_{\text{in}}}$ to output activations $y = W x \in \mathbb{R}^{d_{\text{out}}}$, Secant implements two closed-form operators:

1. **Projector subtraction (`projector-subtract`):**
   $$W' = (I_{d_{\text{out}}} - \gamma q_b q_b^\top) W = W - \gamma q_b (q_b^\top W)$$
   At $\gamma = 1$, the edited matrix satisfies $q_b^\top W' = 0$ in exact arithmetic for every input $x$.

2. **Target-Shift Calibrated sparse rank-1 column edit (`rank1-columns`):**
   Given a sparse support $S \subset \{0, \dots, d_{\text{in}}-1\}$ of $|S| = K$ input columns selected by descending reference activation magnitude $|h_{\text{ref},j}|$, let $h_S \in \mathbb{R}^{d_{\text{in}}}$ denote the vector equal to $h_{\text{ref}}$ on $S$ and $0$ outside $S$ (with $\|h_S\|_2 > 0$). We define the rank-1 update:
   $$\Delta W = \mathrm{sign}(\Delta_{\text{target}}) \cdot \beta \, q_b \, h_S^\top$$
   where the calibrated step size $\beta$ is solved analytically as:
   $$\beta = \frac{|\Delta_{\text{target}}|}{\|h_S\|_2^2}$$

Substituting $x = h_{\text{ref}}$ yields a scale-invariant directional shift along $q_b$ regardless of layer width $d_{\text{in}}$ or activation scale $\|h_S\|_2$:

$$\langle q_b, \Delta W \, h_{\text{ref}} \rangle = \mathrm{sign}(\Delta_{\text{target}}) \cdot \frac{|\Delta_{\text{target}}|}{\|h_S\|_2^2} \langle h_S, h_{\text{ref}} \rangle = \Delta_{\text{target}}$$

And the Frobenius norm of the perturbation is inversely proportional to the active column energy:

$$\|\Delta W\|_{\mathrm{F}} = \frac{|\Delta_{\text{target}}|}{\|h_S\|_2}$$

> **Scope & Implementation Note:** Because $\langle h_S, h_{\text{ref}} \rangle = \|h_S\|_2^2$, the target shift $\Delta_{\text{target}}$ is achieved exactly at $x = h_{\text{ref}}$ in `float64` arithmetic. On general inputs $x \neq h_{\text{ref}}$ and after rounding to `BF16`, `F16`, or `Q8_0`, the realized shift scales with $\langle h_S, x \rangle / \|h_S\|_2^2$ and must be verified empirically on held-out prompts (Section 4).

---

(container-geometry-the-pi-128-quantization-commutation-theorem)=
## 2. Container Geometry & The $\Pi_{128}$ Quantization Commutation Theorem

> **Plain-English Intuition:** Model files on disk store weights differently depending on the container format (`.safetensors` vs. `.gguf`) and quantization scheme (`Q8_0` vs. `Q4_K`). Even worse, converters like `llama.cpp` shuffle 128-column blocks inside `Qwen3.5` linear-attention tensors (`ssm_out`). This section proves two practical facts: (1) we can un-shuffle those 128-column panels bit-exactly, and (2) 32-weight `Q8_0` blocks divide 128 columns evenly ($128 = 4 \times 32$) so we can edit `Q8_0` files directly on disk, whereas 256-weight `Q4_K` superblocks straddle two 128-column heads and cannot be edited head-locally.

### 2.1 Logical `[d_out, d_in]` Orientation Contract

In HuggingFace `.safetensors`, a 2-D linear weight matrix has `shape = [d_out, d_in]` with input axis `shape[1] = d_in` contiguous in memory. In `ggml` / `.gguf`, `ne = [d_in, d_out]` lists the contiguous axis first (`ne[0] = d_in`). Consequently, both formats share the identical row-major element offset for output row $\text{out} \in \{0, \dots, d_{\text{out}}-1\}$ and input column $\text{in} \in \{0, \dots, d_{\text{in}}-1\}$:

$$\mathrm{offset}(\text{out}, \text{in}) = \text{out} \cdot d_{\text{in}} + \text{in}$$

### 2.2 Gated DeltaNet `ssm_out` Panel Bijection ($\Pi_{128}$)

On `Qwen3.5` Gated DeltaNet layers, `llama.cpp`'s `convert_hf_to_gguf.py` reorders the input columns ($d_{\text{in}} = n_v \cdot d_v$, with head width $d_v = 128$ and group ratio $r = n_v / n_k$) of `blk.N.ssm_out.weight` from key-head-major to value-head-major order. Writing a HuggingFace 128-column panel index as $h = k \cdot r + j$ for $k \in \{0,\dots,n_k-1\}$ and $j \in \{0,\dots,r-1\}$, its GGUF panel index is:

$$\pi(k \cdot r + j) = j \cdot n_k + k$$

Within each 128-column panel $w \in \{0,\dots,127\}$, the intra-panel offset is preserved:

$$\text{col}_{\text{GGUF}}(h \cdot 128 + w) = \pi(h) \cdot 128 + w$$

When $r = 1$ (`Qwen3.5-0.8B`, $n_k = n_v = 16$), $\pi$ is the identity permutation. When $r = 2$ (`Qwen3.5-4B`, $n_k = 16, n_v = 32$), $\Pi_{128}$ is a non-trivial bijection on the 32 panels of width 128 corresponding to the transpose of a $(16, 2)$ grid into a $(2, 16)$ grid. Because $r \neq n_k$, $\pi$ is not an involution (for example, $\pi(1) = 16$ whereas $\pi(16) = 8$); its inverse is:

$$\pi^{-1}(g) = (g \bmod n_k) \cdot r + \lfloor g / n_k \rfloor$$

### 2.3 Block-Grid Equivariance (`Q8_0`) vs. Superblock Straddling (`Q4_K`)

Let $Q_B$ denote a deterministic row-wise block quantizer that partitions each row of length $d_{\text{in}}$ into aligned contiguous blocks of $B$ elements and encodes each block independently.

- **Proposition 1 (Block-Grid Equivariance for $B \mid 128$):** Whenever the quantization block size $B$ divides the panel width $128$ (such as `Q8_0` with $B = 32$, since $128 = 4 \times 32$), every 128-element panel is a disjoint union of $128 / B$ complete quantization blocks. Therefore, block quantization commutes byte-for-byte with $\Pi_{128}$:
  $$Q_{32}(\Pi_{128}(W)) = \Pi_{128}^{\text{block}}(Q_{32}(W))$$
  And the dequantized reconstruction error between the two paths is identically zero:
  $$\big\|D_{32}(Q_{32}(\Pi_{128}(W))) - \Pi_{128}(D_{32}(Q_{32}(W)))\big\|_{\max} = 0$$

- **Proposition 2 (Superblock Straddling for `Q4_K`, $B = 256$):** In `Q4_K`, each 256-weight superblock shares a single FP16 superblock scale $d$ and minimum $d_{\min}$ across two adjacent 128-element panels ($256 = 2 \times 128$). Permuting 128-element panels changes which pairs of panels share a 256-weight superblock whenever $\pi$ is non-trivial, and editing weights inside one 128-column head alters the shared superblock scale for its neighboring head.

- **Proposition 3 (Three-Term Requantization Error Bound for Direct `Q8_0` Block Surgery):** Writing $\hat{Q}_B = D_B \circ Q_B$ for the quantize-dequantize operator, compare **Path 1** (edit in floating point, then quantize: $\hat{Q}_B(W + \Delta W)$) against **Path 2** (quantize first, then edit only the touched quantized blocks on disk: $\hat{Q}_B(\hat{Q}_B(W) + \Delta W)$). Defining the three block-wise quantization residuals $e_0 = \hat{Q}_B(W) - W$, $e_1 = \hat{Q}_B(W + \Delta W) - (W + \Delta W)$, and $e_2 = \hat{Q}_B(\hat{Q}_B(W) + \Delta W) - (\hat{Q}_B(W) + \Delta W)$, algebraic cancellation yields the exact identity:
  $$\hat{Q}_B(W + \Delta W) - \hat{Q}_B\big(\hat{Q}_B(W) + \Delta W\big) = e_1 - e_0 - e_2$$
  On untouched blocks ($\Delta W_b = 0$), $\hat{Q}_B$ is idempotent so $e_1 = e_0$, $e_2 = 0$, and the Path 1 vs. Path 2 discrepancy is identically zero. On touched `Q8_0` blocks ($B = 32$), each element's discrepancy is bounded by:
  $$|e_1 - e_0 - e_2| \le \kappa\big((3 + \kappa)\|W_b\|_\infty + 2\|\Delta W_b\|_\infty\big), \quad \text{where } \kappa = \frac{1}{254} + 2^{-11}$$
  Empirically on `Qwen3.5-0.8B` (`blk.12.ffn_down.weight`), direct `Q8_0` block surgery (`Path 2`) removes `98.5%` of the target component (`0.5172 -> 0.0079`, vs. `99.4%` for `Path 1`) while leaving all untouched 34-byte blocks 100% bit-identical.

---

(project-then-heal-structural-orthogonal-complement-lora)=
## 3. `Project-Then-Heal`: Structural Orthogonal-Complement LoRA

> **Plain-English Intuition:** After you surgically remove an unwanted behavior from a weight matrix $W_0$, you often want to fine-tune the model with LoRA so it stays fluent or learns a domain task. If you use standard LoRA, the adapter's updates write straight back into the direction you just removed. Even if you project the gradient to be perpendicular to the removed direction before calling `AdamW.step()`, `AdamW` divides each coordinate by its running variance—which tilts the perpendicular vector right back into the forbidden subspace on every single step! `Project-Then-Heal` solves this by placing the perpendicular projector *inside* the forward pass of the LoRA layer itself, making it mathematically impossible for any optimizer to write back into the removed direction.

### 3.1 Why Gradient Projection Alone Leaks Under AdamW

Suppose a writer matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ has had an orthonormal row basis $D \in \mathbb{R}^{k \times d_{\text{out}}}$ ($D D^\top = I_k$) surgically excised, and we attach a rank-$r$ LoRA adapter $\Delta W = s B A$ ($s = \alpha_{\text{lora}} / r$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$, $A \in \mathbb{R}^{r \times d_{\text{in}}}$).

If one projects only the gradient $g_t = P_\perp \nabla_B \mathcal{L}_t$ using the orthogonal complement projector $P_\perp = I_{d_{\text{out}}} - D^\top D$ and passes $g_t$ into AdamW, the first moment $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ remains in $\mathrm{col}(P_\perp) = \ker(D)$. However, AdamW's parameter update multiplies $\hat{m}_t$ by a non-isotropic diagonal preconditioner $\Lambda_t = \mathrm{diag}\big((\sqrt{\hat{v}_t} + \epsilon)^{-1}\big)$:

$$\Delta B_t = -\eta \big( \Lambda_t \hat{m}_t + \lambda B_{t-1} \big)$$

Because a diagonal matrix $\Lambda_t$ with distinct positive entries does not commute with a non-axis-aligned projector $P_\perp$, their cross-product is non-zero:

$$D \Lambda_t P_\perp \neq 0$$

Consequently, AdamW rotates orthogonal gradients back into $\mathrm{span}(D^\top)$ on **100% of steps** (`108/108` gradient-projected AdamW runs in our 6-arm, 648-run optimizer ablation, leaking `1.56%` to `2.98%` of $\|\Delta W\|_{\mathrm{F}}$ on `0.8B`/`4B`/`7B` layer-16 `down_proj` weights and `0.33%` to `1.67%` across all 15 models in our H100 Stage-1 sweep).

### 3.2 Forward-Pass Double Projection ($B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$)

`ProjectThenHealLinear` instead reparameterizes the LoRA output factor inside the forward computation graph via a twice-applied `float64` orthogonal projection:

$$B_{\text{locked}}(\tilde{B}) = P_\perp\big(P_\perp(\tilde{B})\big)$$

where each application of $P_\perp$ is evaluated in $\mathcal{O}(k \cdot d_{\text{out}} \cdot r)$ time without allocating a dense $d_{\text{out}} \times d_{\text{out}}$ matrix:

$$P_\perp(X) = X - D^\top (D X)$$

The layer's forward pass is then:

$$y = W_0 x + s \, B_{\text{locked}}(\tilde{B}) (A x)$$

By associativity, for **any** parameter state $(\tilde{B}_t, A_t)$ produced by any optimizer (`SGD`, `AdamW`, `Lion`, `Muon`):

$$D \Delta W_t = s \, (D P_\perp) P_\perp(\tilde{B}_t) A_t = s \, \big(D - (D D^\top) D\big) P_\perp(\tilde{B}_t) A_t = 0$$

In exact arithmetic $P_\perp^2 = P_\perp$, but in finite-precision IEEE-754 `float64`, a single pass $P_\perp(\tilde{B})$ can leave $\mathcal{O}(\varepsilon_{\text{mach}} \|\tilde{B}\|)$ residual components along $D$ when columns of $\tilde{B}$ are nearly parallel to $\mathrm{span}(D^\top)$ (analogous to single-pass vs. two-pass Gram-Schmidt). Applying $P_\perp$ twice reduces $\max |D B_{\text{locked}}|$ by $1.8\times$ to $17.7\times$ (median $5.6\times$ across 120 seeded Gaussian bases), holding live `float64` writer leakage at $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$.

### 3.3 Writer-Level Invariance vs. Downstream Residual-Stream Propagation

The guarantee $D \Delta W_t = 0$ holds at every protected writer's output. In a multi-layer transformer, an orthogonal perturbation $\delta h_\ell \in \ker(D)$ emitted at layer $\ell$ subsequently passes through downstream RMSNorm, attention, and MLP matrices $W_{\ell'}$ ($\ell' > \ell$) for which $D W_{\ell'} P_\perp \neq 0$ in general. Empirically on `Qwen3.5-0.8B` (`Step 60`), locking `L15..L17` reduces direct merged-`F32` writer leakage by $4.2 \times 10^7\times$ ($2.49 \times 10^{-10}$ vs. $1.05 \times 10^{-2}$) and cuts downstream pull-back along the `B1` loop direction at `resid L19` by more than half (`+0.276` vs. `+0.635` unconstrained).

---

(cross-architecture-empirical-evaluation-0-8b-to-72-7b)=
## 4. Cross-Architecture Empirical Evaluation (`0.8B` to `72.7B`)

> **Plain-English Intuition:** To make sure our equations aren't just theoretical curiosities, we tested Secant across 17 open-weight models (`0.8B` to `72.7B`) in four distinct experimental regimes:
> - **Regime B (`Qwen3.5-4B` Dose-Response):** Does Target-Shift Calibration actually fix infinite `<think>` loops where uncalibrated edits fail (`0/30` $\to$ `30/30`)?
> - **Regime A (Live Causal-LM Healing):** Does `ProjectThenHealLinear` match standard LoRA perplexity on real text while keeping leakage at machine zero (`~10^-18`)?
> - **H100 Stage-1 (15-Model Calibration & 3-Arm Optimizer Sweep):** Does calibration solve for the exact target shift (`<= 7.99e-15` error) and does `ProjectThenHealLinear` beat gradient-projected AdamW across 15 architectures (`0.8B` to `32.8B`)?
> - **Downstream Security Auditing & Public Benchmarks (`Qwen2.5-Coder-32B-Instruct`, `Qwen3-32B`, `Gemma-2-9B-IT`):** Does calibrated weight surgery improve real C/C++ vulnerability discovery (`6.7% -> 40.0%`, $p = 0.00195$) while preserving or improving general math and coding benchmarks?

### 4.1 Regime B: Target-Shift Dose-Response on `Qwen3.5-4B` (`n = 30` Crash Traces)

| Arm (`Qwen3.5-4B`, `n = 30`, `1,536` token cap) | Summed `B1` target shift | `</think>` Closed | Strict Bug Class | Strict Culprit Frame | Strict Root Cause | Mean Tokens |
|---|---:|---:|---:|---:|---:|---:|
| **`base`** / **`rand1`** / **`rand2`** (equal-norm random) | n/a | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`g030`** (uncalibrated $\gamma = 0.30$ ported from `0.8B`) | n/a | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`a05`** (`B1` at $0.5\times$; `B2` and `B3` held at $1.0\times$) | `-3.691` | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`b1only`** ($\alpha = 1.0\times$, loop edit only) | `-7.382` | `17/30` (`56.7%`) | `15/30` (`50.0%`) | `13/30` (`43.3%`) | `11/30` (`36.7%`) | `1,044` |
| **`cal738`** ($\alpha = 1.0\times$, composed `B1+B2+B3`) | `-7.382` | **`21/30` (`70.0%`)** | **`18/30` (`60.0%`)** | **`15/30` (`50.0%`)** | **`14/30` (`46.7%`)** | `954` |
| **`a15`** (`B1` at $1.5\times$; `B2` and `B3` held at $1.0\times$) | **`-11.073`** | **`30/30` (`100.0%`)** | **`23/30` (`76.7%`)** | **`23/30` (`76.7%`)** | **`19/30` (`63.3%`)** | `789` |

Exact two-sided McNemar tests against `base`: `cal738` loop closure $p = 9.54 \times 10^{-7}$ (21 vs. 0 discordant pairs); `a15` loop closure $p = 1.86 \times 10^{-9}$ (30 vs. 0 discordant pairs). Comparing `a15` vs. `cal738`: loop closure $p = 0.0039$, culprit frame $p = 0.0078$.

### 4.2 Regime A: Live Causal-LM Fine-Tuning After Real Surgery

Both arms train on real causal-LM completions after a byte-verified surgical edit, evaluated on the identical writer-level metric $\max |D \Delta W|$:

| Model | Steps | Verified Surgical Edit | `ProjectThenHealLinear` $\max\lvert D \Delta W\rvert$ (`float64` / merged `F32`) | Unconstrained LoRA $\max\lvert D \Delta W\rvert$ | Held-Out Perplexity (PTH vs. Unconstrained) |
|---|---:|---|---:|---:|---|
| `Qwen/Qwen3.5-0.8B` | 60 (CPU) | Composed `B1+B2+B3` GGUF (`verify-patch` exit 0) | **`9.87e-18`** / **`2.49e-10`** | `1.05e-02` | Step 20: `5.435` vs. `5.423`; Step 60: `12.106` vs. `12.357` |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 20 (ROCm GPU) | `B2` projector at L14 `o_proj` (`67.9x` vs. random) + `B3` calibrated at L15 `down_proj` (`22.1x` vs. random) | **`5.21e-18`** / **`1.46e-10`** | `4.64e-03` | Step 20: `21.260` vs. `21.255` |

> **Storage Precision Boundary (`F32`/`F64` vs. `BF16`):** When merged weights are quantized down to 16-bit `BF16` (7-bit mantissa, unit roundoff $\approx 3.9 \times 10^{-3}$), rounding error alone introduces $\sim 10^{-4}$ to $10^{-3}$ subspace leakage. `secant heal` therefore stores healed LoRA adapters in `F32` or `F64`.

### 4.3 H100 Stage-1: 15-Model Calibration & Three-Arm Optimizer Comparison (`0.8B` to `32.8B`)

Across 15 full models on H100 (`0.8B` to `32.8B`, receipt `irrefutable-15models-stage1-receipt.json`), we applied full Target-Shift Calibration ($|\Delta_{\text{target}}| = 7.382$, $K = 128$ columns) to `B2` and `B3` and ran a 60-step 3-arm comparison on the surgically edited writer:
- **Arm A:** Unconstrained LoRA + AdamW
- **Arm B:** Gradient-Projected LoRA ($g_t \leftarrow P_\perp g_t$ before `AdamW.step()`)
- **Arm C:** `ProjectThenHealLinear` ($B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$)

All 30 calibrated edits realize $\Delta_{\text{target}}$ at $h_{\text{ref}}$ to within `<= 7.99e-15` in `float64`. On the 3-arm healing test (measured by relative leakage $\|D \Delta W\|_{\mathrm{F}} / \|\Delta W\|_{\mathrm{F}}$ at step 60), Gradient-Projected AdamW (Arm B) leaks on **15 of 15 models (`0.33%` to `1.67%`)**, whereas `ProjectThenHealLinear` (Arm C) stays at the `float32` working-dtype floor (**`1.78e-10` to `8.67e-10`**, at least $3.8 \times 10^6\times$ lower than Arm B on every model):

| Model | Size | `B2` / `B3` Layers | `B2` / `B3` $\beta$ | Max `float64` Shift Error | Arm A: Unconstrained | Arm B: Grad-Projected AdamW | Arm C: `ProjectThenHealLinear` |
|---|---:|---|---:|---:|---:|---:|---:|
| `Qwen3.5-0.8B` | `0.8B` | `L17` / `L14` | `0.5801` / `0.3595` | `8.9e-16` | `7.60%` | `0.61%` | **`8.67e-10`** |
| `SmolLM2-1.7B` | `1.7B` | `L17` / `L12` | `0.0369` / `0.2727` | `1.8e-15` | `2.95%` | `1.62%` | **`5.41e-10`** |
| `Qwen3.5-4B` | `4B` | `L23` / `L16` | `0.0484` / `0.3805` | `3.6e-15` | `2.22%` | `1.67%` | **`5.81e-10`** |
| `Mistral-7B-v0.3` | `7.2B` | `L12` / `L19` | `0.9352` / `0.0510` | `3.6e-15` | `2.17%` | `0.92%` | **`3.38e-10`** |
| `Qwen2.5-7B` | `7.6B` | `L8` / `L14` | `0.1224` / `0.1484` | `2.7e-15` | `2.04%` | `0.49%` | **`4.71e-10`** |
| `Llama-3.1-8B` | `8B` | `L23` / `L16` | `0.4556` / `0.5070` | `1.8e-15` | `0.82%` | `0.80%` | **`4.36e-10`** |
| `R1-Llama-8B` | `8B` | `L23` / `L16` | `0.4343` / `0.4701` | `5.3e-15` | `1.59%` | `0.60%` | **`4.50e-10`** |
| `Qwen3-8B` | `8.2B` | `L14` / `L21` | `0.3931` / `0.0766` | `5.3e-15` | `2.02%` | `0.58%` | **`3.14e-10`** |
| `Gemma-2-9B` | `9.2B` | `L16` / `L25` | `3.7192` / `1.1508` | `1.8e-15` | `8.51%` | `0.35%` | **`5.46e-10`** |
| `OLMo-2-13B` | `13.7B` | `L16` / `L20` | `0.1130` / `0.0325` | `3.6e-15` | `1.70%` | `0.44%` | **`1.78e-10`** |
| `Phi-4` | `14.7B` | `L12` / `L29` | `1.1436` / `0.0174` | `4.4e-15` | `0.86%` | `0.33%` | **`3.26e-10`** |
| `Qwen3-14B` | `14.8B` | `L16` / `L23` | `0.2994` / `0.0547` | `1.8e-15` | `2.13%` | `0.60%` | **`2.69e-10`** |
| `R1-Qwen-14B` | `14.8B` | `L28` / `L35` | `0.0343` / `0.0177` | `3.6e-15` | `1.23%` | `0.77%` | **`3.53e-10`** |
| `Gemma-2-27B` | `27.2B` | `L14` / `L27` | `0.0845` / `0.0098` | `8.0e-15` | `1.98%` | `1.31%` | **`3.49e-10`** |
| `Qwen3-32B` | `32.8B` | `L47` / `L25` | `0.0292` / `0.0738` | `3.6e-15` | `2.04%` | `0.75%` | **`3.07e-10`** |

### 4.4 Downstream C/C++ Sanitizer Vulnerability Uplift & Public Benchmark Preservation

We evaluated whether calibrated rank-1 surgery improves downstream C/C++ vulnerability root-cause identification on our 30-case sanitizer benchmark suite (6 CWE-aligned classes: integer overflow, signed/unsigned mismatch, off-by-one, use-after-free/lifetime, missing bounds check, and format/buffer overflow) while preserving general reasoning (`GSM8K`, `MATH-500`) and code generation (`HumanEval pass@1`):

| Model (`calibrated_1p0x` vs. `base`) | Paired 30-Case C/C++ Sanitizer Audit | `GSM8K` | `MATH-500` | `HumanEval pass@1` |
|---|---|---|---|---|
| **`Qwen/Qwen2.5-Coder-32B-Instruct`** (`32.8B`) | **`2/30` (`6.7%`) $\to$ `12/30` (`40.0%`, `+10` crashes, $p = 0.00195$)** | Evaluated in domain uplift (`125.9 -> 122.4` mean tokens) | Evaluated in domain uplift | Evaluated in domain uplift |
| **`Qwen/Qwen3-32B`** (`32.8B`) | Multi-subspace reasoning + code composition | `53.0%` $\to$ **`55.0%` (`+2.0%`)** | `31.0%` $\to$ **`38.0%` (`+7.0%`)** | `21.88%` $\to$ **`25.00%` (`+3.12%`)** |
| **`google/gemma-2-9b-it`** (`9.2B`) | **`4/30` (`13.3%`) $\to$ `7/30` (`23.3%`, `+3` crashes)** | `77.0%` $\to$ **`79.0%` (`+2.0%`)** | `37.0%` $\to$ `37.0%` (`0.0%`) | `39.02%` $\to$ **`40.85%` (`+1.83%`)** |
| **`Qwen/Qwen2.5-7B-Instruct`** (`7.6B`) | Multi-subspace capability preservation | `82.5%` $\to$ `81.5%` (`-1.0%`) | `51.0%` $\to$ **`54.0%` (`+3.0%`)** | `61.59%` $\to$ **`62.80%` (`+1.22%`)** |
| **`Qwen/Qwen3-14B`** (`14.8B`) | Multi-subspace capability preservation | `84.5%` $\to$ `84.5%` (`0.0%`) | `57.0%` $\to$ `56.0%` (`-1.0%`) | `46.95%` $\to$ **`49.39%` (`+2.44%`)** |

### 4.5 Regime C: 14-Model H100 Adversarial Subspace Stress Test (`7.2B` to `72.7B`)

In a separate adversarial stress test across 12 full models (`7.2B` to `32.8B`) and 2 single-shard runs (`70.6B`, `72.7B`), we trained LoRA for 60 AdamW steps toward a synthetic target $Y_{\text{target}} = Y + 1.5\,v_{B2}$ that deliberately lies *inside* the protected subspace $\mathrm{span}(D)$. Under this adversarial pull, unconstrained LoRA placed `8.42%` to `52.55%` of its output factor norm $\|B\|_{\mathrm{F}}$ into $\mathrm{span}(D)$, whereas `ProjectThenHealLinear` held $\max |D B_{\text{locked}}|$ (evaluated in `float64` on the `F32` working factor) to **`4.82e-10` – `1.08e-9`** at step 60 across all 14 checkpoints.

> **Verification & Diagnostic Notes on Regime C:**
> - **Non-linear dose saturation at extreme shifts:** On 20 of 24 full-model edits in Regime C, raising the dose from $1.0\times$ to $1.5\times$ scaled the realized shift linearly (`1.49x` to `1.50x`). Four edits (`Phi-4` `B2`, `OLMo-2-13B` `B3`, `Gemma-2-9B` `B3`, `Gemma-2-27B` `B3`) showed sub-linear scaling (`1.04x` to `1.18x`) due to downstream RMSNorm normalization.
> - **Outlier-dimension diagnostic (`Phi-4`):** In Regime C's uncalibrated capture on `Phi-4` (`L32`), peak $\mathrm{SNR}_{\text{proj}}$ reached `27,362.3` with $\mathrm{SNR}_{\text{trace}} = 60.4$, and the uncalibrated edit inverted sign (`-3.399`). Always pair $\mathrm{SNR}_{\text{proj}}$ with $\mathrm{SNR}_{\text{trace}}$ and verify the signed shift on held-out prompts (as done in Stage-1 L12 calibration, where `Phi-4` achieves `4.4e-15` shift error).

---

(related-work-and-limits-of-novelty)=
## 5. Related Work and Limits of Novelty

Orthogonal projection and subspace-constrained LoRA build upon a rich line of mechanistic interpretability and parameter-efficient fine-tuning research:

- **Direct Weight Editing & Null-Space Constraints:** [ROME](https://arxiv.org/abs/2202.05262) (Meng et al., NeurIPS 2022) and [AlphaEdit](https://arxiv.org/abs/2410.02355) (Fang et al., ICLR 2025) formulate rank-1 and null-space-projected weight updates for factual editing. [Takehi, Teoh, Zhang, and Khattab](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, Nov 2026) introduced Weight-Space Programming on `Qwen3.5-0.8B`.
- **Subspace-Constrained LoRA:** [Safe LoRA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/77baa7c2a3a675823e89131698fd6e19-Abstract-Conference.html) (Hsu et al., NeurIPS 2024) projects LoRA weights post-hoc onto a safety-aligned checkpoint vector; [SaLoRA](https://arxiv.org/abs/2501.01765) (Li et al., ICLR 2025) introduces a fixed safety projection during fine-tuning; [LoRA-Null](https://arxiv.org/abs/2503.02659) (Tang et al., AAAI 2026) initializes adapters in the null space of pre-trained activations; [OPLoRA](https://arxiv.org/abs/2510.13003) (Xiong & Xie, 2025) projects out top singular vectors of frozen weights via left and right projections; and [OGPSA](https://arxiv.org/abs/2602.07892) (Sun et al., 2026) projects gradients against a capability subspace.
- **Directional Ablation Toolkits:** Community toolkits including [Heretic](https://github.com/p-e-w/heretic), [Obliteratus](https://github.com/elder-plinius/OBLITERATUS), and [model-unfetter](https://github.com/josepha-mayo/model-unfetter) provide single-direction and multi-direction refusal ablation workflows in Python/PyTorch.

**Scope of Secant's Contributions:** Secant does not claim priority over orthogonal projection or null-space LoRA in general. Its specific technical contributions are: (1) closed-form **Target-Shift Calibration** ($\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$) for cross-scale sparse rank-1 surgery; (2) the **$\Pi_{128}$ Quantization Commutation Theorem** and direct 34-byte `Q8_0` in-place `mmap` block surgery with SHA-256 non-target byte-identity verification; (3) the formal characterization of **AdamW diagonal preconditioner rotation** ($D \Lambda_t P_\perp \neq 0$) showing why gradient projection alone leaks on 100% of AdamW steps, paired with double-idempotent forward-pass projection $B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$; and (4) empirical validation on paired C/C++ vulnerability auditing (`Qwen2.5-Coder-32B-Instruct`, `6.7% -> 40.0%`, $p = 0.00195$).

---

## References

1. **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab.** *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL, November 2, 2026. https://rikiyat.github.io/blog/weight-space-programming/
2. **Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda.** *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024 (`arXiv:2406.11717`).
3. **Junfeng Fang et al.** *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR 2025 (`arXiv:2410.02355`).
4. **Pengwei Tang et al.** *Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge (LoRA-Null).* AAAI 2026 (`arXiv:2503.02659`).
5. **Yifeng Xiong and Xiaohui Xie.** *OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning.* `arXiv:2510.13003`, 2025.
6. **Xuechen Li et al.** *SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation.* ICLR 2025 (`arXiv:2501.01765`).
7. **Chia-Yi Hsu et al.** *Safe LoRA: The Silver Lining of Reducing Safety Risks when Fine-Tuning Large Language Models.* NeurIPS 2024.
8. **Hao Sun et al.** *Orthogonal Gradient Projection for Safety Alignment (OGPSA).* `arXiv:2602.07892`, 2026.
9. **Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov.** *Locating and Editing Factual Associations in GPT (ROME).* NeurIPS 2022 (`arXiv:2202.05262`).
