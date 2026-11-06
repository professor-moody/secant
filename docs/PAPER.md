# Secant: Calibrated Weight Edits, Quantized Files, and Constrained LoRA

## Abstract

Editing a language model's weights can change how it responds without retraining the whole model. Turning that idea into a reliable workflow requires answering three questions: how large should an edit be, which bytes should change in a compressed checkpoint, and how can later fine-tuning respect the edited directions?

Secant combines Rust checkpoint tools with a PyTorch adapter layer. **Target-Shift Calibration** sizes an edit to produce a requested change at a chosen layer's reference activation. The equality is exact in real arithmetic; reported numerical tests across 15 models have a maximum `float64` error of about `7.99e-15`. **Quantized block editing** updates only the affected groups of 32 weights in `Q8_0` files. Reordering complete 128-column panels preserves those blocks, whereas `Q4_K` groups 256 weights together and does not generally preserve the same boundary. The standalone patch command therefore refuses `Q4_K`. **Project-Then-Heal** constrains a LoRA adapter, a small trainable addition to a frozen model, so its output is perpendicular to supplied directions. It applies the constraint during each forward pass because projecting gradients alone need not survive AdamW's coordinate-wise scaling.

The reported experiments measure both local numerical properties and model responses. On 30 crash-triage prompts, calibrated edits to `Qwen3.5-4B` raised completion within a fixed token limit from `0/30` to `21/30`, then `30/30` at a higher loop-edit dose (reported paired `p = 1.86e-9` against baseline). A separate C/C++ root-cause test on `Qwen2.5-Coder-32B-Instruct` improved from `2/30` to `12/30` correct cases (`p = 0.00195`). In the 15-model adapter comparison, Project-Then-Heal's relative leakage into the protected directions was `1.78e-10` to `8.67e-10`, compared with `0.33%` to `1.67%` for gradient-projected AdamW. These are results for the stated experiments. Calibration at one reference input does not guarantee behavior on other prompts, and a local adapter constraint does not prevent later layers or storage rounding from changing the result.

---

## Reading Guide and Terminology

A **writer** is a weight matrix that contributes to a layer's output. A **direction** is a vector in that output space; several directions span a **subspace**. A **projection** removes or retains the component along that subspace. **Leakage** measures how much of an adapter or weight update remains along protected directions. It is a numerical measurement, not a direct measure of model behavior.

Throughout the derivations, `D` stores protected directions as rows, `W` is a weight matrix, and `Delta W` is its change. The equations use column-vector activations unless stated otherwise. “B” in a model size means billion parameters; the experiment labels `B1`, `B2`, and `B3` instead identify loop, refusal, and root-cause directions.

| Section | Question | Main point |
|---|---|---|
| [1. Choosing and sizing edits](#multi-subspace-orthogonalization-target-shift-calibration) | How can several edits coexist, and how large should each be? | Make their output directions perpendicular, then size each edit for a measured reference input. |
| [2. Editing compressed checkpoints](#container-geometry-the-pi-128-quantization-commutation-theorem) | Which stored blocks must change? | Whole 128-column panels contain complete `Q8_0` blocks, but split `Q4_K` superblocks. Editing and quantizing can still give different numerical results. |
| [3. Constraining LoRA](#project-then-heal-structural-orthogonal-complement-lora) | How can fine-tuning respect protected directions? | Apply the projection inside the adapter's forward pass and measure rounding error after export. |
| [4. Experiments](#cross-architecture-empirical-evaluation-0-8b-to-72-7b) | What was measured? | Separate reference-input checks, adapter tests, and response benchmarks. They support different claims. |
| [5. Related work and limits](#related-work-and-limits-of-novelty) | Which ideas come from prior work? | Projection and constrained adapters are established methods; this paper evaluates their use in Secant's file-editing workflow. |

---

```{figure} assets/secant-pipeline.svg
:name: fig-paper-pipeline
:alt: Four-stage Secant diagram showing contrastive activation distributions and a per-layer SNR curve, two-pass Gram-Schmidt and sparse column support, Q8_0 versus Q4_K block alignment under the 128-column panel permutation, and the ProjectThenHealLinear plane versus AdamW drift
:width: 100%

The paper's four parts in one picture. Stage 1 and the Stage 2 support strip are drawn from the `SmolLM2-1.7B` demo inputs and the real `secant analyze` and `secant compose` outputs. Stage 3 uses the exact `ggml` block layouts (34 bytes per 32 `Q8_0` weights, 144 bytes per 256 `Q4_K` weights). Stage 4 is a geometric sketch whose bars show the measured step-60 ranges from Section 4.3. Section 1 covers Stages 1 and 2, Section 2 covers Stage 3, and Section 3 covers Stage 4.
```

(multi-subspace-orthogonalization-target-shift-calibration)=
## 1. Choose Independent Directions and Size Each Edit

Several candidate behavior directions can overlap. Removing that overlap lets the linear edits act independently along the resulting directions, up to numerical error. It does not make the behaviors themselves independent. Secant then sizes each edit using the activation measured at the chosen layer.

### 1.1 Make the Directions Perpendicular

Gram-Schmidt subtracts each direction's overlap with the directions already selected, then scales it to unit length. Secant uses two passes in 64-bit floating point to reduce rounding error.

Let $D = [q_1, \dots, q_k]^\top \in \mathbb{R}^{k \times d_{\text{out}}}$ hold the resulting unit directions as rows. The Python API checks both unit length and perpendicularity through

$$\max_{i,j} \big|(D D^\top - I_k)_{ij}\big| < 10^{-12}.$$

It rejects dependent directions or a failed tolerance check. Rust `compose` uses an absolute residual-norm floor and checks pairwise cosines; its acceptance rule differs for unnormalized inputs. The [theory guide](theory-and-calibration.md) specifies both contracts. Across 120 seeded Gaussian test bases, the reported maximum Gram-matrix error was $9.99 \times 10^{-16}$.

### 1.2 Remove a Component or Add a Measured Shift

The first operation removes part of a writer's output along a chosen direction. The second adds a change along that direction using only selected input columns. Both are direct matrix calculations.

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

$$
\begin{aligned}
\langle q_b, \Delta W \, h_{\text{ref}} \rangle
  &= \mathrm{sign}(\Delta_{\text{target}}) \cdot \frac{|\Delta_{\text{target}}|}{\|h_S\|_2^2} \langle h_S, h_{\text{ref}} \rangle \\
  &= \Delta_{\text{target}}
\end{aligned}
$$

And the Frobenius norm of the perturbation is inversely proportional to the active column energy:

$$\|\Delta W\|_{\mathrm{F}} = \frac{|\Delta_{\text{target}}|}{\|h_S\|_2}$$

> **Scope:** The target equality holds at the reference input in exact arithmetic, with small numerical error in `float64`. For other inputs, the exact-arithmetic shift is multiplied by $\langle h_S,x\rangle / \|h_S\|_2^2$. Storage in `BF16`, `F16`, or `Q8_0` adds rounding error. Neither result guarantees a behavioral change; evaluate held-out prompts separately.

> **Section 1 takeaway:** Make the directions perpendicular first, then size each edit from the measured reference activation. The requested shift is exact at that reference input, up to `float64` rounding. Behavior on other prompts has to be measured separately.

---

(container-geometry-the-pi-128-quantization-commutation-theorem)=
## 2. File Layout and Quantized Block Boundaries

A model file has two relevant layouts: the logical rows and columns of a matrix, and the groups of values stored together on disk. Some converters also reorder groups of columns. To edit the intended weights, the plan and checkpoint must agree on that order. To preserve neighboring bytes, the edit must respect the compressed format's block boundaries.

### 2.1 Rows and Columns in the Two File Formats

The formats list dimensions in different orders, but the linear matrices considered here store each output row contiguously. The following offset is an element index; compressed formats need an additional mapping to block bytes.

In HuggingFace `.safetensors`, a 2-D linear weight matrix has `shape = [d_out, d_in]` with input axis `shape[1] = d_in` contiguous in memory. In `ggml` / `.gguf`, `ne = [d_in, d_out]` lists the contiguous axis first (`ne[0] = d_in`). Consequently, both formats share the identical row-major element offset for output row $\text{out} \in \{0, \dots, d_{\text{out}}-1\}$ and input column $\text{in} \in \{0, \dots, d_{\text{in}}-1\}$:

$$\mathrm{offset}(\text{out}, \text{in}) = \text{out} \cdot d_{\text{in}} + \text{in}$$

### 2.2 Track Reordered Groups of 128 Columns

A panel is a group of 128 adjacent columns. The conversion described below moves whole panels and preserves the order within each one. The symbol $\Pi_{128}$ denotes that reordering. Undoing it requires the inverse mapping, not necessarily a second application of the same mapping.

On `Qwen3.5` Gated DeltaNet layers, `llama.cpp`'s `convert_hf_to_gguf.py` reorders the input columns ($d_{\text{in}} = n_v \cdot d_v$, with head width $d_v = 128$ and group ratio $r = n_v / n_k$) of `blk.N.ssm_out.weight` from key-head-major to value-head-major order. Writing a HuggingFace 128-column panel index as $h = k \cdot r + j$ for $k \in \{0,\dots,n_k-1\}$ and $j \in \{0,\dots,r-1\}$, its GGUF panel index is:

$$\pi(k \cdot r + j) = j \cdot n_k + k$$

Within each 128-column panel $w \in \{0,\dots,127\}$, the intra-panel offset is preserved:

$$\text{col}_{\text{GGUF}}(h \cdot 128 + w) = \pi(h) \cdot 128 + w$$

When $r = 1$ (`Qwen3.5-0.8B`, $n_k = n_v = 16$), $\pi$ is the identity permutation. When $r = 2$ (`Qwen3.5-4B`, $n_k = 16, n_v = 32$), $\Pi_{128}$ is a non-trivial bijection on the 32 panels of width 128 corresponding to the transpose of a $(16, 2)$ grid into a $(2, 16)$ grid. Because $r \neq n_k$, $\pi$ is not an involution (for example, $\pi(1) = 16$ whereas $\pi(16) = 8$); its inverse is:

$$\pi^{-1}(g) = (g \bmod n_k) \cdot r + \lfloor g / n_k \rfloor$$

### 2.3 When Reordering Preserves Blocks, and When Editing Adds Error

There are two different questions here. Reordering complete blocks can preserve their bytes exactly. Changing their weights usually requires encoding them again, which introduces rounding error.

**Proposition 1: complete `Q8_0` blocks move without re-encoding.** Let $Q_B$ be a deterministic, position-independent encoder that quantizes aligned blocks of $B$ weights independently, and let $D_B$ decode them. Assume rows contain complete blocks. If $B$ divides 128, every panel contains whole blocks. For `Q8_0`, four 32-weight blocks make one panel, so

$$Q_{32}(\Pi_{128}(W)) = \Pi_{128}^{\text{block}}(Q_{32}(W)).$$

Here the right-hand permutation moves encoded blocks as units. The corresponding decoded arrays are identical. This equality concerns panel reordering, not an arbitrary weight edit.

**Proposition 2: `Q4_K` can couple neighboring panels.** A 256-weight superblock shares top-level scale and minimum parameters across two 128-column panels. A permutation that splits those pairs need not preserve the encoded result. Pair-preserving permutations, including the identity, are exceptions. Editing one panel can also change its neighbor's reconstructed values. The standalone `secant quant patch` therefore accepts `Q8_0` only.

**Proposition 3: editing before and after quantization can differ.** Fix the same additive update $\Delta W$ in both paths and write $\hat Q_B = D_B \circ Q_B$. Path 1 edits the original float weights, then quantizes. Path 2 starts from decoded quantized weights, edits them, then re-encodes the touched blocks. Define

$$
\begin{aligned}
e_0 &= \hat Q_B(W)-W, \\
e_1 &= \hat Q_B(W+\Delta W)-(W+\Delta W), \\
e_2 &= \hat Q_B(\hat Q_B(W)+\Delta W)-(\hat Q_B(W)+\Delta W).
\end{aligned}
$$

Subtracting the two reconstructed results gives

$$
\begin{aligned}
&\hat Q_B(W+\Delta W)-\hat Q_B(\hat Q_B(W)+\Delta W) \\
&\quad=e_1-e_0-e_2.
\end{aligned}
$$

The implementation copies untouched blocks byte-for-byte, so no assumption that repeated encoding is idempotent is needed there. On touched `Q8_0` blocks, an ideal-arithmetic bound including `FP16` scale rounding is

$$|e_1-e_0-e_2|\le\kappa\big((3+\kappa)\|W_b\|_\infty+2\|\Delta W_b\|_\infty\big),$$

where

$$\kappa=\frac1{254}+2^{-11}.$$

This bound requires normal-range `FP16` scales in all three encodings and excludes the encoder's `float32` arithmetic error. A projector edit recomputes its update from the current weights, so it need not use the same additive delta in both paths. See the [quantized-weight guide](quantized-surgery.md) for these assumptions and the full derivation.

The retained `Qwen3.5-0.8B` `ssm_out` experiment reported a target weight component decreasing from `0.5172` to `0.0079` with direct `Q8_0` editing (about `98.5%` removed). This measures a weight component, not a percentage of prompts. A separate whole-file experiment checked byte preservation outside the selected edits.

> **Section 2 takeaway:** Both file formats use the same element offset, the `ssm_out` panel reordering has an exact inverse, and `Q8_0` blocks fit wholly inside 128-column panels while `Q4_K` superblocks span two. Editing still re-encodes the touched blocks, so the Proposition 3 bound is the error to expect.

---

(project-then-heal-structural-orthogonal-complement-lora)=
## 3. Project-Then-Heal: Keep Adapter Updates Out of Protected Directions

LoRA fine-tunes a model by adding a small trainable matrix product to a frozen weight matrix. Ordinary LoRA can add back a component that an earlier edit removed. Project-Then-Heal removes protected components from the adapter each time the model computes an output. This is a constraint on selected writers, not a guarantee about all later model behavior.

### 3.1 Why Projecting the Gradient Is Not Enough

A gradient gives a proposed update direction. AdamW rescales its coordinates using past gradients, so the final update can point somewhere else. A direction that was perpendicular before rescaling need not remain perpendicular afterward.

Suppose a writer matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ has had an orthonormal row basis $D \in \mathbb{R}^{k \times d_{\text{out}}}$ ($D D^\top = I_k$) surgically excised, and we attach a rank-$r$ LoRA adapter $\Delta W = s B A$ ($s = \alpha_{\text{lora}} / r$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$, $A \in \mathbb{R}^{r \times d_{\text{in}}}$).

If one projects only the gradient $g_t = P_\perp \nabla_B \mathcal{L}_t$ using the orthogonal complement projector $P_\perp = I_{d_{\text{out}}} - D^\top D$ and passes $g_t$ into AdamW, the first moment $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ remains in $\mathrm{col}(P_\perp) = \ker(D)$. However, AdamW's parameter update multiplies $\hat{m}_t$ by a non-isotropic diagonal preconditioner $\Lambda_t = \mathrm{diag}\big((\sqrt{\hat{v}_t} + \epsilon)^{-1}\big)$:

$$\Delta B_t = -\eta \big( \Lambda_t \hat{m}_t + \lambda B_{t-1} \big)$$

For a general projector and unequal coordinate scales, the update can have a component in the protected directions:

$$D \Lambda_t P_\perp \neq 0.$$

This is a possible failure of invariance, not a claim that every AdamW step must leak. Compatible scalings and axis-aligned protected directions are exceptions. The first-moment statement above also assumes its initial state lies in the allowed subspace. For a matrix-valued adapter, the diagonal-rescaling argument applies to each column, whose AdamW scales may differ.

In the reported three-model comparison, gradient-projected AdamW had `1.56%` to `2.98%` relative weight-update leakage on the orthogonal target at step 60. The 15-model H100 comparison reported `0.33%` to `1.67%`. The [training guide](project-then-heal.md) gives the tasks, metrics, and learning-rate caveat.

### 3.2 Apply the Constraint Whenever the Adapter Runs

The optimizer can change the adapter's stored parameters freely. Before those parameters contribute to an output, the layer removes their protected components. Applying the projection twice is a numerical refinement; in exact arithmetic, the second pass changes nothing.

`ProjectThenHealLinear` instead reparameterizes the LoRA output factor inside the forward computation graph via a twice-applied `float64` orthogonal projection:

$$B_{\text{locked}}(\tilde{B}) = P_\perp\big(P_\perp(\tilde{B})\big)$$

where each application of $P_\perp$ is evaluated in $\mathcal{O}(k \cdot d_{\text{out}} \cdot r)$ time without allocating a dense $d_{\text{out}} \times d_{\text{out}}$ matrix:

$$P_\perp(X) = X - D^\top (D X)$$

The layer's forward pass is then:

$$y = W_0 x + s \, B_{\text{locked}}(\tilde{B}) (A x)$$

By associativity, for **any** parameter state $(\tilde{B}_t, A_t)$ produced by any optimizer (`SGD`, `AdamW`, `Lion`, `Muon`):

$$
\begin{aligned}
D \Delta W_t &= s \, (D P_\perp) \, P_\perp(\tilde{B}_t) \, A_t \\
  &= s \, \big(D - (D D^\top) D\big) \, P_\perp(\tilde{B}_t) \, A_t \\
  &= 0
\end{aligned}
$$

In exact arithmetic $P_\perp^2 = P_\perp$, but in finite-precision IEEE-754 `float64`, a single pass $P_\perp(\tilde{B})$ can leave $\mathcal{O}(\varepsilon_{\text{mach}} \|\tilde{B}\|)$ residual components along $D$ when columns of $\tilde{B}$ are nearly parallel to $\mathrm{span}(D^\top)$ (analogous to single-pass vs. two-pass Gram-Schmidt). Applying $P_\perp$ twice reduces $\max |D B_{\text{locked}}|$ by $1.8\times$ to $17.7\times$ (median $5.6\times$ across 120 seeded Gaussian bases). Separately, the live-run diagnostic that recomputes an unrounded projected weight update reported $5.21 \times 10^{-18}$ to $9.87 \times 10^{-18}$.

### 3.3 Where the Guarantee Stops

Later layers can turn a perpendicular change back toward a protected direction. Saving or merging weights at lower precision can also introduce a component that was absent in the calculation. Check both the stored weights and model responses after those steps.

The exact-arithmetic guarantee $D \Delta W_t = 0$ holds at every protected writer's output. In a multi-layer transformer, an orthogonal perturbation $\delta h_\ell \in \ker(D)$ emitted at layer $\ell$ subsequently passes through downstream RMSNorm, attention, and MLP matrices $W_{\ell'}$ ($\ell' > \ell$) for which $D W_{\ell'} P_\perp \neq 0$ in general. Empirically on `Qwen3.5-0.8B` (`Step 60`), locking `L15..L17` reduces direct merged-`F32` writer leakage by $4.2 \times 10^7\times$ ($2.49 \times 10^{-10}$ vs. $1.05 \times 10^{-2}$) and cuts downstream pull-back along the `B1` loop direction at `resid L19` by more than half (`+0.276` vs. `+0.635` unconstrained).

> **Section 3 takeaway:** Projecting only the gradient can leak once AdamW rescales coordinates. Projecting inside the forward pass makes $D \Delta W = 0$ hold in exact arithmetic for any optimizer at the locked writer. Rounding, merging, and later layers still have to be measured.

---

(cross-architecture-empirical-evaluation-0-8b-to-72-7b)=
## 4. Experiments and What They Measure

The following studies answer three separate questions: whether the edit produces the intended local numerical change, whether the adapter respects protected directions, and whether model answers improve. A pass on one question is not a pass on the others. `B1`, `B2`, and `B3` label the loop, refusal, and root-cause directions. An **arm** is one configuration being compared. Leakage metrics are named in each table because maximum absolute entries and norm ratios are not interchangeable.

The numerical results below are retained experimental reports, not fresh runs made for this documentation review. Some raw runners and receipts remain in the lab evidence bank rather than this repository; the guides identify those limits. Results on the two largest checkpoints use one shard each, not full-model fine-tuning.

### 4.1 Regime B: Does the Model Finish Its Answer?

This test gives the same 30 crash traces to each edited model with a 1,536-token limit. “Closed” means the model emitted `</think>` within that limit. The other columns score the final answer's bug class, culprit frame, and root cause separately. A completed answer can still be wrong. The dose changes only `B1` in the `a05` and `a15` arms. Strict scoring requires a committed answer after `</think>`. In the reported copy-the-input control, a whole-output keyword matcher scored the raw prompt `28/30`, while strict scoring gave it `0/30`; merely repeating a function name is not correct attribution.

| Arm (`Qwen3.5-4B`, `n = 30`, `1,536` token cap) | Summed `B1` target shift | `</think>` Closed | Strict Bug Class | Strict Culprit Frame | Strict Root Cause | Mean Tokens |
|---|---:|---:|---:|---:|---:|---:|
| **`base`** / **`rand1`** / **`rand2`** (equal-norm random) | n/a | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`g030`** (uncalibrated $\gamma = 0.30$ ported from `0.8B`) | n/a | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`a05`** (`B1` at $0.5\times$; `B2` and `B3` held at $1.0\times$) | `-3.691` | `0/30` | `0/30` | `0/30` | `0/30` | `1,536` |
| **`b1only`** ($\alpha = 1.0\times$, loop edit only) | `-7.382` | `17/30` (`56.7%`) | `15/30` (`50.0%`) | `13/30` (`43.3%`) | `11/30` (`36.7%`) | `1,044` |
| **`cal738`** ($\alpha = 1.0\times$, composed `B1+B2+B3`) | `-7.382` | **`21/30` (`70.0%`)** | **`18/30` (`60.0%`)** | **`15/30` (`50.0%`)** | **`14/30` (`46.7%`)** | `954` |
| **`a15`** (`B1` at $1.5\times$; `B2` and `B3` held at $1.0\times$) | **`-11.073`** | **`30/30` (`100.0%`)** | **`23/30` (`76.7%`)** | **`23/30` (`76.7%`)** | **`19/30` (`63.3%`)** | `789` |

Exact two-sided McNemar tests against `base`: `cal738` loop closure $p = 9.54 \times 10^{-7}$ (21 vs. 0 discordant pairs); `a15` loop closure $p = 1.86 \times 10^{-9}$ (30 vs. 0 discordant pairs). Comparing `a15` vs. `cal738`: loop closure $p = 0.0039$, culprit frame $p = 0.0078$.

### 4.2 Regime A: Fine-Tune on Text After Editing

These two short runs compare a constrained adapter with ordinary LoRA after a weight edit. Perplexity measures prediction quality on held-out text; lower is better. The Qwen run reached its best held-out value at step 20, then both arms overfit the small training batch. The `float64` leakage column recomputes an unrounded projected delta; it is not a claim that the whole model ran in `float64`.

Both arms train on real causal-LM completions after a byte-verified surgical edit, evaluated on the identical writer-level metric $\max |D \Delta W|$:

| Model | Steps | Verified Surgical Edit | `ProjectThenHealLinear` $\max\lvert D \Delta W\rvert$ (`float64` / merged `F32`) | Unconstrained LoRA $\max\lvert D \Delta W\rvert$ | Held-Out Perplexity (PTH vs. Unconstrained) |
|---|---:|---|---:|---:|---|
| `Qwen/Qwen3.5-0.8B` | 60 (CPU) | Composed `B1+B2+B3` GGUF (`verify-patch` exit 0) | **`9.87e-18`** / **`2.49e-10`** | `1.05e-02` | Step 20: `5.435` vs. `5.423`; Step 60: `12.106` vs. `12.357` |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 20 (ROCm GPU) | `B2` projector at L14 `o_proj` (`67.9x` vs. random) + `B3` calibrated at L15 `down_proj` (`22.1x` vs. random) | **`5.21e-18`** / **`1.46e-10`** | `4.64e-03` | Step 20: `21.260` vs. `21.255` |

> **Storage Precision Boundary (`F32`/`F64` vs. `BF16`):** When merged weights are quantized down to 16-bit `BF16` (7-bit mantissa, unit roundoff $\approx 3.9 \times 10^{-3}$), rounding error alone introduces $\sim 10^{-4}$ to $10^{-3}$ subspace leakage. `secant heal` therefore stores healed LoRA adapters in `F32` or `F64`.

### 4.3 H100 Stage-1: Check Calibration and Adapters Across 15 Models

This sweep checks the shift at a reference input and compares three adapter methods on one selected writer per model. It uses one seed per model and a synthetic target plus a real-token cross-entropy term. Public behavioral benchmarks were `NOT_RUN` in this sweep; the local numerical results do not establish generation quality.

Across 15 full models on H100 (`0.8B` to `32.8B`, receipt `irrefutable-15models-stage1-receipt.json`), we applied full Target-Shift Calibration ($|\Delta_{\text{target}}| = 7.382$, $K = 128$ columns) to `B2` and `B3` and ran a 60-step 3-arm comparison on the surgically edited writer:

- **Arm A:** Unconstrained LoRA + AdamW
- **Arm B:** Gradient-Projected LoRA ($g_t \leftarrow P_\perp g_t$ before `AdamW.step()`)
- **Arm C:** `ProjectThenHealLinear` ($B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$)

All 30 calibrated edits realize $\Delta_{\text{target}}$ at $h_{\text{ref}}$ to within `<= 7.99e-15` in `float64`. On the 3-arm healing test (measured by relative leakage $\|D \Delta W\|_{\mathrm{F}} / \|\Delta W\|_{\mathrm{F}}$ at step 60), Gradient-Projected AdamW (Arm B) leaks on **15 of 15 models (`0.33%` to `1.67%`)**, whereas `ProjectThenHealLinear` (Arm C) stays at the `float32` working-dtype floor (**`1.78e-10` to `8.67e-10`**; model by model, $6.5 \times 10^6$ to $3.8 \times 10^7$ times lower than Arm B):

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

```{figure} assets/h100-15model-leakage-log10.svg
:name: fig-paper-leakage-log10
:alt: Log-scale relative leakage at step 60 for unconstrained LoRA, gradient-projected AdamW and ProjectThenHealLinear on 15 models, with the B over C ratio per model, and trajectories over 60 AdamW steps
:width: 100%

The last three columns of the table, on a log scale. Left: step-60 relative leakage for each model, with the Arm B to Arm C ratio printed above each row. Right: all 15 trajectories over 60 AdamW steps, with the median drawn thick. Arms A and B jump to their plateau in the first 10 steps and stay there. Arm C stays near `1e-10` from step 0 to step 60.
```

(sec-paper-public-benchmarks)=
### 4.4 Public Benchmarks After Calibrated Edits, 14 Models

Section 4.3 checks the local arithmetic of each edit. This section asks a different question: what the same `B2 + B3` edits do to a model's answers on public benchmarks. The receipt is `public-benchmarks/irrefutable-receipt.json` (SHA-256 `9859ef3e0603318e09b0366e6dae6994b47f7d7cc4456cfff1fff7ddb2253132`, `status: PASS`, completed `2026-11-06T20:41:59Z`), with per-item rows in `predictions.jsonl` (SHA-256 `0c2d65e1...86fe`). It covers 14 of the 15 Stage-1 models. `Gemma-2-27B` is `HOLD` because its gated repository returned HTTP 401, so it has no benchmark row. The aligned edits (layers, column support $S$, $\beta$), model revisions, and seed (`20261105`) are those of Section 4.3, and the healing results in this receipt are identical to the Stage-1 receipt for all 14 shared models.

Five variants of each model answer the same items: the unedited `base`; the calibrated edits at `1.0x` and `1.5x`; and two **norm-matched random** controls at `1.0x` and `1.5x`. A random control replaces each aligned direction $q$ with a random unit vector orthogonal to $q$, keeping the same $\beta$, the same input-column support $S$, and therefore the same $\lVert \Delta W\rVert_{\mathrm{F}}$. If a calibrated edit and its random control move a score by similar amounts, the movement comes from the size of the edit, not its direction.

Each model answers the first 100 items of `GSM8K`, `MATH-500` and `MMLU-Pro`, the first 64 of `HumanEval`, and the 30 retained sanitizer traces (ASan-30). Decoding is greedy. ASan-30 counts a root cause as correct when the answer names both the bug class and the culprit function, matched by keyword. Every comparison is paired by item: an exact two-sided McNemar test on the discordant pairs, then a Holm correction across the 70 model-by-benchmark cells of each variant. The pooled rows add the discordant pairs over all 14 models. Their 95% intervals use the paired normal approximation, and they treat items from different models as independent, which they are not.

```{figure} assets/benchmark-retention-14models.svg
:name: fig-paper-benchmark-retention
:alt: Small multiples of paired change from base for 14 models on GSM8K, MATH-500, MMLU-Pro, HumanEval and ASan-30, comparing calibrated 1.0x edits with norm-matched random 1.0x edits, with pooled diamonds and 95% intervals
:width: 100%

Paired change from base, in percentage points, for every model and benchmark. Blue dots are calibrated `1.0x`; orange rings are the norm-matched random `1.0x` control. A white halo marks a cell that survives Holm correction. Only one does at `1.0x`, and it is a random control: `Gemma-2-9B` `HumanEval`, `42.2%` to `75.0%`. The diamonds pool all 14 models.
```

| Benchmark (pooled over 14 models) | Paired items | Calibrated `1.0x`: change in pp (95% CI), exact p | Calibrated `1.5x` | Random `1.0x` | Random `1.5x` |
|---|---:|---|---|---|---|
| `GSM8K` | 1400 | `-0.9` (`-2.9` to `+1.0`), p = `0.39` | `-2.0` (`-4.1` to `+0.1`), p = `0.079` | `-0.8` (`-2.6` to `+1.0`), p = `0.43` | `-0.9` (`-3.0` to `+1.1`), p = `0.41` |
| `MATH-500` | 1400 | `-0.5` (`-2.4` to `+1.4`), p = `0.65` | `-0.4` (`-2.2` to `+1.5`), p = `0.76` | `-1.4` (`-3.2` to `+0.4`), p = `0.14` | `-0.1` (`-2.0` to `+1.7`), p = `0.94` |
| `MMLU-Pro` | 1400 | `-2.0` (`-3.5` to `-0.5`), p = `0.013` | `-2.4` (`-3.9` to `-1.0`), p = `0.0015` | `-0.5` (`-2.0` to `+1.0`), p = `0.57` | `-2.0` (`-3.6` to `-0.4`), p = `0.017` |
| `HumanEval` pass@1 | 896 | `+0.6` (`-1.7` to `+2.9`), p = `0.70` | `+0.3` (`-2.0` to `+2.7`), p = `0.85` | `+0.4` (`-1.8` to `+2.7`), p = `0.77` | `+0.3` (`-1.8` to `+2.5`), p = `0.84` |
| ASan-30 root cause | 420 | `+2.1` (`-0.8` to `+5.0`), p = `0.20` | `+1.2` (`-1.8` to `+4.2`), p = `0.53` | `+0.2` (`-3.2` to `+3.6`), p = `1.0` | `-2.9` (`-5.6` to `-0.1`), p = `0.065` |

What this shows:

- **Retention.** At `1.0x`, no calibrated cell changes significantly after Holm correction (`0/70`). Single cells range from `-12` to `+19` points, which is what 100-item and 64-item samples produce by chance. Pooled, each benchmark moves by `2.1` points or less.
- **One pooled cost, MMLU-Pro.** Calibrated edits lower pooled `MMLU-Pro` by `2.0` points at `1.0x` and `2.4` at `1.5x`. Random `1.5x` lowers it by `2.0`, so edit size explains at least part of the cost. Under a Holm correction across all 20 pooled tests, only calibrated `1.5x` remains below `0.05` (`p = 0.031`).
- **No measured uplift.** The ASan-30 root-cause gain at `1.0x` (`+2.1` points, `p = 0.20`) is not significant, and no root-cause cell survives Holm correction. These edits were sized to hit a reference shift, not tuned for this task. This receipt does not reproduce the root-cause gains reported in Sections 4.1 and 4.5 for other edit sets.
- **Edit size matters more than direction on one model.** `Gemma-2-9B` carries the largest `B2` step in the sweep ($\beta = 3.72$). It accounts for every Holm-significant cell across the four variants: `GSM8K` falls `17` points at calibrated `1.5x` and `14` at random `1.5x`, and `HumanEval` rises `33` to `34` points under both random controls. The random controls move it as far as the calibrated edits do.

```{figure} assets/asan30-audit-uplift.svg
:name: fig-paper-asan30
:alt: Per-model ASan-30 scores out of 30 for root cause, culprit frame and bug class, comparing base, calibrated 1.0x and norm-matched random 1.0x, with paired root-cause changes and exact p-values
:width: 100%

ASan-30 counts out of 30 for each model: base (gray ring), calibrated `1.0x` (blue), and random `1.0x` (orange). The right column lists the paired root-cause change and its unadjusted exact McNemar p. The largest single gains are `+4` (`Llama-3.1-8B`, `p = 0.12`) and `+3` (`Mistral-7B-v0.3`, `p = 0.25`).
```

> **Scope.** These are first-N items, not random samples, from one seed, decoded greedily, on a set of checkpoints weighted toward Qwen. The receipt's standalone `secant` CLI cross-check is `HOLD` for 7 of the 14 models; that check is separate from these benchmark results. Section 4.5's table comes from separate runs. For `Gemma-2-9B`, `Qwen2.5-7B` and `Qwen3-14B`, its numbers do not match this receipt, which uses 100 and 64 items per benchmark, so do not combine the two tables.

### 4.5 Root-Cause Identification, Math, and Code Scores

Here the measurement is answer quality, rather than a matrix property. “Correct cases” counts benchmark cases answered correctly, not newly discovered vulnerabilities. Changes in accuracy are percentage points (`pp`); rates and differences retain the reported rounding. A blank numerical result is marked `NOT_REPORTED`; the table cannot establish performance on that benchmark.

We evaluated whether calibrated rank-1 surgery improves downstream C/C++ vulnerability root-cause identification on our 30-case sanitizer benchmark suite (6 CWE-aligned classes: integer overflow, signed/unsigned mismatch, off-by-one, use-after-free/lifetime, missing bounds check, and format/buffer overflow) while preserving general reasoning (`GSM8K`, `MATH-500`) and code generation (`HumanEval pass@1`):

| Model (`calibrated_1p0x` vs. `base`) | Paired 30-Case C/C++ Sanitizer Audit | `GSM8K` | `MATH-500` | `HumanEval pass@1` |
|---|---|---|---|---|
| **`Qwen/Qwen2.5-Coder-32B-Instruct`** (`32.8B`) | **`2/30` (`6.7%`) $\to$ `12/30` (`40.0%`, `+10` correct cases, $p = 0.00195$)** | NOT_REPORTED | NOT_REPORTED | NOT_REPORTED |
| **`Qwen/Qwen3-32B`** (`32.8B`) | NOT_REPORTED | `53.0%` $\to$ **`55.0%` (`+2.0 pp`)** | `31.0%` $\to$ **`38.0%` (`+7.0 pp`)** | `21.88%` $\to$ **`25.00%` (`+3.12 pp`)** |
| **`google/gemma-2-9b-it`** (`9.2B`) | **`4/30` (`13.3%`) $\to$ `7/30` (`23.3%`, `+3` correct cases)** | `77.0%` $\to$ **`79.0%` (`+2.0 pp`)** | `37.0%` $\to$ `37.0%` (`0.0 pp`) | `39.02%` $\to$ **`40.85%` (`+1.83 pp`)** |
| **`Qwen/Qwen2.5-7B-Instruct`** (`7.6B`) | NOT_REPORTED | `82.5%` $\to$ `81.5%` (`-1.0 pp`) | `51.0%` $\to$ **`54.0%` (`+3.0 pp`)** | `61.59%` $\to$ **`62.80%` (`+1.22 pp`)** |
| **`Qwen/Qwen3-14B`** (`14.8B`) | NOT_REPORTED | `84.5%` $\to$ `84.5%` (`0.0 pp`) | `57.0%` $\to$ `56.0%` (`-1.0 pp`) | `46.95%` $\to$ **`49.39%` (`+2.44 pp`)** |

### 4.6 Regime C: Train Toward a Direction the Adapter Must Avoid

This stress test deliberately asks the adapter to learn a change inside the protected subspace. A successful lock must leave that part of the task unfitted, so low leakage comes with a loss floor. The unconstrained and locked numbers below use different metrics and must not be read as a direct ratio.

In a separate adversarial stress test across 12 full models (`7.2B` to `32.8B`) and 2 single-shard runs (`70.6B`, `72.7B`), we trained LoRA for 60 AdamW steps toward a synthetic target $Y_{\text{target}} = Y + 1.5\,v_{B2}$ that deliberately lies *inside* the protected subspace $\mathrm{span}(D^\top)$. Under this adversarial pull, unconstrained LoRA placed `8.42%` to `52.55%` of its output factor norm $\|B\|_{\mathrm{F}}$ into $\mathrm{span}(D^\top)$, whereas `ProjectThenHealLinear` held $\max |D B_{\text{locked}}|$ (evaluated in `float64` on the `F32` working factor) to **`4.82e-10` to `1.08e-9`** at step 60 across all 14 checkpoints.

> **Verification & Diagnostic Notes on Regime C:**
> - **Non-linear dose saturation at extreme shifts:** On 20 of 24 full-model edits in Regime C, raising the dose from $1.0\times$ to $1.5\times$ scaled the realized shift linearly (`1.49x` to `1.50x`). Four edits (`Phi-4` `B2`, `OLMo-2-13B` `B3`, `Gemma-2-9B` `B3`, `Gemma-2-27B` `B3`) showed sub-linear scaling (`1.04x` to `1.18x`) due to downstream RMSNorm normalization.
> - **Outlier-dimension diagnostic (`Phi-4`):** In Regime C's uncalibrated capture on `Phi-4` (`L32`), peak $\mathrm{SNR}_{\text{proj}}$ reached `27,362.3` with $\mathrm{SNR}_{\text{trace}} = 60.4$, and the uncalibrated edit inverted sign (`-3.399`). Always pair $\mathrm{SNR}_{\text{proj}}$ with $\mathrm{SNR}_{\text{trace}}$ and verify the signed shift on held-out prompts (as done in Stage-1 L12 calibration, where `Phi-4` achieves `4.4e-15` shift error).

> **Section 4 takeaway:** Each study supports its own claim. Calibration matched the requested shift on all 15 Stage-1 models. Project-Then-Heal kept local leakage at the `float32` floor on every model where gradient projection did not. On 14 models, the calibrated `1.0x` edits left public-benchmark scores within about 2 pooled points of base, as equal-size random edits did, with a pooled `MMLU-Pro` cost of `2.0` points and no significant crash-triage gain. The response benchmarks report measured changes on specific prompt sets, not general behavior.

---

(related-work-and-limits-of-novelty)=
## 5. Related Work and Limits of Novelty

Orthogonal projection and subspace-constrained LoRA build upon a rich line of mechanistic interpretability and parameter-efficient fine-tuning research:

- **Direct Weight Editing & Null-Space Constraints:** [ROME](https://arxiv.org/abs/2202.05262) (Meng et al., NeurIPS 2022) and [AlphaEdit](https://arxiv.org/abs/2410.02355) (Fang et al., ICLR 2025) formulate rank-1 and null-space-projected weight updates for factual editing. [Takehi, Teoh, Zhang, and Khattab](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, Nov 2026) presented Weight-Space Programming experiments on `Qwen3.5-0.8B`.
- **Subspace-Constrained LoRA:** [Safe LoRA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/77baa7c2a3a675823e89131698fd6e19-Abstract-Conference.html) (Hsu et al., NeurIPS 2024) projects LoRA weights post-hoc onto a safety-aligned subspace derived from base and aligned checkpoints; [SaLoRA](https://arxiv.org/abs/2501.01765) (Li et al., ICLR 2025) introduces a fixed safety projection during fine-tuning; [LoRA-Null](https://arxiv.org/abs/2503.02659) (Tang et al., AAAI 2026) initializes adapters in the null space of pre-trained activations; [OPLoRA](https://arxiv.org/abs/2510.13003) (Xiong & Xie, 2025) projects out top singular vectors of frozen weights via left and right projections; and [OGPSA](https://arxiv.org/abs/2602.07892) (Sun et al., 2026) projects gradients against a capability subspace.
- **Directional Ablation Toolkits:** Community toolkits including [Heretic](https://github.com/p-e-w/heretic), [Obliteratus](https://github.com/elder-plinius/OBLITERATUS), and [model-unfetter](https://github.com/josepha-mayo/model-unfetter) provide single-direction and multi-direction refusal ablation workflows in Python/PyTorch.

**Scope of this paper.** Secant does not claim to introduce orthogonal projection or null-space LoRA. It evaluates reference-based edit sizing, block-aware `Q8_0` file editing, and a forward-pass adapter constraint in one workflow. The equations explain local matrix properties. The experiments test particular checkpoints, directions, prompts, and storage formats; they do not establish universal behavior preservation or comparative superiority over the cited tools.

> **Section 5 takeaway:** The projection methods are prior work. This paper evaluates reference-based edit sizing, block-aware `Q8_0` editing, and a forward-pass adapter constraint together, inside one file-editing workflow that verifies every untouched byte.

---

## References

1. **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab.** *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL, November 2, 2026. https://rikiyat.github.io/blog/weight-space-programming/
2. **Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda.** *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024 (`arXiv:2406.11717`).
3. **Junfeng Fang et al.** *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR 2025 (`arXiv:2410.02355`).
4. **Pengwei Tang et al.** *Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge (LoRA-Null).* AAAI 2026 (`arXiv:2503.02659`).
5. **Yifeng Xiong and Xiaohui Xie.** *OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning.* `arXiv:2510.13003`, 2025.
6. **Mingjie Li et al.** *SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation.* ICLR 2025 (`arXiv:2501.01765`).
7. **Chia-Yi Hsu et al.** *Safe LoRA: The Silver Lining of Reducing Safety Risks when Fine-Tuning Large Language Models.* NeurIPS 2024.
8. **Guanglong Sun et al.** *[Safety Alignment as Continual Learning: Mitigating the Alignment Tax via Orthogonal Gradient Projection](https://arxiv.org/abs/2602.07892).* `arXiv:2602.07892`, 2026.
9. **Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov.** *Locating and Editing Factual Associations in GPT (ROME).* NeurIPS 2022 (`arXiv:2202.05262`).
