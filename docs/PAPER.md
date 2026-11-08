# Secant: Calibrated Weight Edits, Quantized Files, and Constrained LoRA

**Author:** Nathan Keys (`m00dy`, `@professor-moody`)  
**Date:** November 2026

## Abstract

Editing a language model's weights can change how it responds without retraining the whole model. Turning that idea into a reliable workflow requires answering three questions: how large should an edit be, which bytes should change in a compressed checkpoint, and how can later fine-tuning respect the edited directions?

Secant combines Rust checkpoint tools with a PyTorch adapter layer. **Target-Shift Calibration** sizes an edit to produce a requested change at a chosen layer's reference activation. The equality is exact in real arithmetic; reported numerical tests across 15 models have a maximum `float64` error of about `7.99e-15`. **Quantized block editing** updates only the affected groups of 32 weights in `Q8_0` files. Reordering complete 128-column panels preserves those blocks, whereas `Q4_K` groups 256 weights together and does not generally preserve the same boundary. The standalone patch command therefore refuses `Q4_K`. **Project-Then-Heal** constrains a LoRA adapter, a small trainable addition to a frozen model, so its output is perpendicular to supplied directions. It applies the constraint during each forward pass because projecting gradients alone need not survive AdamW's coordinate-wise scaling.

The reported experiments measure both local numerical properties and model responses across **six evaluation domains (`MATH-500`, `GSM8K`, `MMLU-Pro`, `HumanEval`, `ASan-30`, and paired `ASAN-120`; `59,440` retained H100 evaluation rows: `59,140` completed-cohort rows across `17` completed benchmark model identities in the Stage-2 + RMS suites from `0.8B` to `32.8B` plus `300` partial `Gemma-2-27B` baseline rows, alongside a `17`-checkpoint PTH study from `0.8B` to `72.7B` and a `15`-model calibration/adapter study)**, directly extending Takehi et al.'s `Qwen3.5-0.8B` loop-suppression study (`Sparks of Weight-Space Programming`, Nov 2, 2026, where an autonomous LLM agent `gpt-6-astra` / Astra traced activations and edited weights inside a Python kernel under a 10/25/50 forward-pass budget; we did not re-run their 32,768-token stochastic `MATH-500` evaluation) to multi-subspace calibration with norm-matched random controls, compressed GGUF block surgery, subspace-locked LoRA healing, and paired vulnerability/benign controls. On 30 crash-triage prompts, calibrated edits to `Qwen3.5-4B` raised completion within a fixed token limit from `0/30` to `21/30`, then `30/30` at a higher loop-edit dose (reported paired `p = 1.86e-9` against baseline, with strict root-cause attribution rising from `0/30` to `19/30`). In the 15-model adapter comparison (`0.8B` to `32.8B`), Project-Then-Heal's relative weight-update leakage into the protected directions was `1.78e-10` to `8.67e-10`, compared with `0.33%` to `1.67%` for gradient-projected AdamW. These are results for the stated experiments. Calibration at one reference input does not guarantee behavior on other prompts, and a local adapter constraint does not prevent later layers or storage rounding from changing the result.

The same tools also audit weights. In one planted rank-1 edit per model, on `SmolLM2-1.7B` and `Qwen3.5-4B`, the edit stayed invisible to `secant audit-trojan` across the pre-specified doses (`0.10%` to `4.00%`) until the post-hoc doses reached the nominal singular-value equality reference $g^* = \sigma_1(W)/\lVert W\rVert_F$ (`8.65%` and `6.57%` of the weight norm, where $\lVert\Delta W\rVert_2 = \sigma_1(W)$). The trigger token reached rank 1 or 2 in the vocabulary projection at `8%`, and the automatic sibling outlier test fired at `12%`, with no false positives on the unedited bases. Across all 10 GGUF quantized formats (`Q4_0..Q6_K`, `186,870,528` quantized blocks across `912` quantized tensors, plus `3,258,570,112` float elements across the 10 tested production GGUF and SafeTensors checkpoints), `secant sanitize` (`Rule B3-2A`) permits legitimate zero (`0x0000`) and FP16 subnormal (`0x0001..0x03FF`) multiplicative block scales by default (`0` observed false positives across `10 / 10` checkpoints and `8,992,791` of `33,792,000` header scale fields `d`, `dmin` across `16,896,000` `Q4_K` superblocks), while flagging `100%` of planted `NaN` and `±Inf` block scales (`(bits & 0x7C00) == 0x7C00`), near-zero divisor scales (`Rule B3-2B`, `< 1e-12f`), and out-of-bounds integer routing indices (`Rule B3-3`). On 14 five-arm models with public benchmarks (`27,580` five-arm evaluations plus `300` base-only `Gemma-2-27B` evaluations = `27,880` rows across `GSM8K`, `MATH-500`, `MMLU-Pro`, `HumanEval`, and `ASan-30`), the calibrated edits left scores where equal-size random edits left them. Across a 6-model RMS-relative dosing study (`24,000` benchmark evaluations), writer output norms $\sigma_{\text{writer}}$ spanned `5.85` to `78.15`, showing that a fixed unit shift ranges from `1.28%` to `17.10%` of the writer norm across architectures. Unhealed composed edits matched norm-matched random controls across all 6 models, while adding `ProjectThenHealLinear` (`3.18e-17` to `5.07e-17` relative subspace leakage) raised `GSM8K` (`87%` to `93%`, unadjusted `p = 0.031`) on `Qwen2.5-Math-7B-Instruct` and `HumanEval` (`36%` to `42%`, unadjusted `p = 0.238`) on `Qwen2.5-Coder-7B-Instruct`; neither gain survives Holm-Bonferroni correction across the full test family, and no control isolates the composition from the healing step. Finally, on the paired `ASAN-120` vulnerability and patched-benign twin benchmark (`7,560` evaluations across 9 models $\times$ 7 arms), patched twins expose one-sided rubber-stamping in non-reasoning models (`BA = 0.475..0.533`), while on `<think>` reasoning models `composed_rms_1p0x` beats `random_rms_1p0x` `4/4` on balanced accuracy and lifts strict root-cause recall by `+6.7` to `+10.0` percentage points over `base` on `DeepSeek-R1-Distill-Llama-8B` and `Qwen3-14B`.

---

## Reading Guide and Terminology

A **writer** is a weight matrix that contributes to a layer's output. A **direction** is a vector in that output space; several directions span a **subspace**. A **projection** removes or retains the component along that subspace. **Leakage** measures how much of an adapter or weight update remains along protected directions. It is a numerical measurement, not a direct measure of model behavior.

Throughout the derivations, `D` stores protected directions as rows, `W` is a weight matrix, and `Delta W` is its change. The equations use column-vector activations unless stated otherwise. “B” in a model size means billion parameters; the experiment labels `B1`, `B2`, and `B3` instead identify loop, refusal, and root-cause directions.

| Section | Question | Main point |
|---|---|---|
| [1. Choosing and sizing edits](#multi-subspace-orthogonalization-target-shift-calibration) | How can several edits coexist, and how large should each be? | Make their output directions perpendicular, then size each edit for a measured reference input. |
| [2. Editing compressed checkpoints](#container-geometry-the-pi-128-quantization-commutation-theorem) | Which stored blocks must change? | Whole 128-column panels contain complete Q8\_0 blocks, but split Q4\_K superblocks. Editing and quantizing can still give different numerical results. |
| [3. Constraining LoRA](#project-then-heal-structural-orthogonal-complement-lora) | How can fine-tuning respect protected directions? | Apply the projection inside the adapter's forward pass and measure rounding error after export. |
| [4. Experiments](#cross-architecture-empirical-evaluation-0-8b-to-72-7b) | What was measured? | Separate reference-input checks, adapter tests, and response benchmarks. They support different claims. |
| [5. Static weight auditing](#sec-paper-static-auditing) | Can the same reader find a planted edit or a hostile quantized scale? | A rank-1 edit hides until its singular value reaches σ₁; Rule B3-2A catches NaN/Inf quantized scales while permitting legitimate Q8\_0/Q4\_K subnormals (0 false positives). |
| [6. Related work and limits](#related-work-and-limits-of-novelty) | Which ideas come from prior work? | Projection and constrained adapters are established methods; this paper evaluates their use in Secant's file-editing workflow. |

---

```{figure} assets/secant-pipeline.svg
:name: fig-paper-pipeline
:alt: Four-stage Secant diagram showing contrastive activation distributions and a per-layer SNR curve, two-pass Gram-Schmidt and sparse column support, Q8_0 versus Q4_K block alignment under the 128-column panel permutation, and the ProjectThenHealLinear plane versus AdamW drift
:width: 100%

The paper's four parts in one picture. Stage 1 and the Stage 2 support strip are drawn from the `SmolLM2-1.7B` demo inputs and the real `secant analyze` and `secant compose` outputs. Stage 3 uses the exact `ggml` block layouts (34 bytes per 32 `Q8_0` weights, 144 bytes per 256 `Q4_K` weights). Stage 4 is a geometric sketch whose bars show the measured step-60 ranges from Section 4.1. Section 1 covers Stages 1 and 2, Section 2 covers Stage 3, and Section 3 covers Stage 4.
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
>
> **Relation to prior associative-memory and abliteration estimators:** Algebraically, the step size $\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$ is the isotropic ($C = I$, minimum-Frobenius-norm Moore-Penrose pseudoinverse) special case of the rank-1 [ROME](https://arxiv.org/abs/2202.05262) update $v = (C^{-1} k_*) / (k_*^\top C^{-1} k_*)$ (Meng et al., NeurIPS 2022; extended to multi-edit [MEMIT](https://arxiv.org/abs/2210.07229), Meng et al., ICLR 2023) restricted to the sparse top-$K$ column support $S$. Covariance-whitened concept erasure was formalized in [LEACE](https://arxiv.org/abs/2306.03819) (Belrose et al., NeurIPS 2023), while multi-vector Gram-Schmidt orthogonalization and ridge-regularized multi-direction projection are used in [G-Abliteration](https://arxiv.org/abs/2512.18901) (Gülmez, Dec 2025), [Norm-Preserving Biprojected Abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration) (Lai, Nov 2025), [Obliteratus](https://github.com/elder-plinius/OBLITERATUS), [llm-abliteration](https://github.com/jim-plus/llm-abliteration), [apostate KCRN](https://github.com/heterodoxin/apostate), and [weightless](https://github.com/msuiche/weightless). Secant's specific contributions in this stage are (1) using **activation-calibrated target shifts** ($\beta(d_{\text{model}}, \ell) = |\Delta_{\text{target}}(\ell)| / \|h_S(\ell)\|_2^2$ and RMS-relative dosing $\Delta_{\text{target}} = \alpha\,\sigma_{\text{writer}}$) for cross-model-size edit transfer (`0.8B` to `32.8B` across 15 full models, plus `70.6B` and `72.7B` single-shard evaluations), and (2) coupling two-pass `float64` Gram-Schmidt orthonormalization with the **upper-triangular $R^{-1}$ cross-projection compensation system** ($Q^\top \Delta W = R^{-\top} \Delta_{\text{target}}$) so heterogeneous non-orthogonal edits each realize their exact reference shift simultaneously.

> **Section 1 takeaway:** Make the directions perpendicular first, solve the upper-triangular $R^{-1}$ compensation system when composing non-orthogonal directions, and size each edit from the measured reference activation. The requested shift is exact at that reference input, up to `float64` rounding. Behavior on other prompts has to be measured separately.

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

> **Relation to prior in-place GGUF and SafeTensors editing tools:** In October 2026, [Kiwimaker](https://huggingface.co/Kiwimaker/DeepSeek-V4.1-Flash-1017-Maya-S-abliterated-GGUF) (`apply_abliteration.py` and `apply_abliteration_v2.py`) demonstrated in-place `r+` memory-mapped (`np.memmap` via `gguf.GGUFReader`) editing of `Q8_0` tensors inside `.gguf` shards with SHA-256 byte-identity verification of untouched offsets, while empirically skipping `Q4_K` tensors and dequantizing/re-quantizing full 2-D `Q8_0` matrices in `float64` via `gguf-py`. Also in October 2026, [SC117](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) demonstrated byte-level in-place GGUF tensor transplantation (leaving 1,079 of 1,223 tensors byte-identical) and scale-preserving `Q2_0` re-encoding against learned GSQ block scales. Existing `.safetensors` abliteration tools ([jim-plus/llm-abliteration](https://github.com/jim-plus/llm-abliteration), [NousResearch `sharded_ablate.py`](https://huggingface.co/NousResearch), and [abliterix](https://github.com/wuwangzhang1216/abliterix)) load tensors via `safe_open` and re-serialize modified shards from scratch via `save_file()`. Secant's specific contributions in this stage are (1) **in-place zero-copy `mmap` byte-slice surgery inside `.safetensors` files** at `[8 + N + BEGIN, 8 + N + END)` without `save_file()` shard re-serialization, (2) **sub-tensor compressed-domain `Q8_0` block surgery** directly on touched 34-byte `block_q8_0` records (`crates/secant-core/src/quant.rs`) without full-tensor `float64` matrix materialization, and (3) **Propositions 1 to 3**, which formally prove why 32-weight `Q8_0` blocks commute with the $\Pi_{128}$ 128-column head-panel permutation ($128 = 4 \times 32$) while 256-weight `Q4_K` superblocks straddle two 128-column heads and suffer cross-head scale leakage.

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

This non-commutativity between Euclidean orthogonal gradient projection and AdamW's diagonal second-moment preconditioner was independently formalized in October 2026 by Chen et al. ([GOAL](https://arxiv.org/abs/2610.39802), Appendix M and Table 7), who defined the gradient-step quantity $\mathcal{L}_{i,t}^{\text{Adam}}$ (*"residual refusal-subspace leakage after AdamW preconditioning"*) and reported $\mathcal{L}_{i,t}^{\text{Adam}} = 3.6\%\text{--}4.7\%$ across three backbone models (`Gemma-7B-it`, `Llama-2-7B-Chat`, `Llama-2-13B-Chat`), averaged over five runs at the end of training with zero weight decay when projecting gradients before `AdamW.step()`; Yu et al. ([CORD](https://arxiv.org/abs/2610.29965), October 2026) likewise noted that subsequent optimizer transformations can disturb gradient-space orthogonalization, while Ghosh et al. ([PoLoRA](https://arxiv.org/abs/2607.17620), July 2026) proposed a preconditioned orthogonalized LoRA optimizer. In our reported three-model comparison (measuring relative adapter weight-update leakage $\lVert D \Delta W\rVert_{\mathrm{F}} / \lVert\Delta W\rVert_{\mathrm{F}}$ rather than GOAL's gradient-step metric), gradient-projected AdamW had `1.56%` to `2.98%` relative weight-update leakage on the orthogonal target at step 60, and the 15-model H100 comparison (`0.8B` to `32.8B`) reported `0.33%` to `1.67%`. The [training guide](project-then-heal.md) gives the tasks, metrics, and learning-rate caveat.

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

> **Relation to prior subspace-projected LoRA and post-abliteration healing:** Forward-pass orthogonal projection of LoRA adapters ($\Delta W = P_L B A P_R$) against top singular directions of frozen weights was introduced for continual learning in [OPLoRA](https://arxiv.org/abs/2510.13003) (Xiong & Xie, AAAI 2026), while [SaLoRA](https://arxiv.org/abs/2501.01765) (Li et al., ICLR 2025) and [Safe LoRA](https://arxiv.org/abs/2405.16833) (Hsu et al., NeurIPS 2024) project LoRA updates to preserve safety alignment during downstream fine-tuning. Conversely, existing post-abliteration LoRA capability-healing pipelines ([Labonne, June 2024](https://huggingface.co/blog/mlabonne/abliteration); [Obliteratus `--auto-finetune`](https://github.com/elder-plinius/OBLITERATUS); [model-unfetter `recovery.py`](https://github.com/josepha-mayo/model-unfetter)) attach unconstrained LoRA adapters that can write back into the excised subspace. Secant's specific contributions in this stage are (1) **bypassing AdamW preconditioner leakage on the effective adapter weight update** via forward-pass subspace-locked parameterization, reducing step-60 relative weight-update leakage $\lVert D \Delta W\rVert_{\mathrm{F}} / \lVert\Delta W\rVert_{\mathrm{F}}$ from `0.33%-1.67%` under gradient-projected AdamW to the `float32` working-dtype floor (`1.78e-10` to `8.67e-10`) across 15 models (`0.8B` to `32.8B`), (2) **inverting safety-projected LoRA for post-surgery capability healing** (`Project-Then-Heal`, locking LoRA out of an excised behavioral subspace $\mathrm{span}(D^\top)$), and (3) **twice-iterated `float64` idempotent projection $P_\perp(P_\perp(\tilde{B}))$ inside the autograd forward pass** ($5.6\times$ median error reduction over single-pass projection).

> **Section 3 takeaway:** Projecting only the gradient can leak once AdamW rescales coordinates. Projecting inside the forward pass makes $D \Delta W = 0$ hold in exact arithmetic for any optimizer at the locked writer. Rounding, merging, and later layers still have to be measured.

---

(cross-architecture-empirical-evaluation-0-8b-to-72-7b)=
## 4. Experiments and What They Measure

The following studies answer three separate questions: whether the edit produces the intended local numerical change, whether the adapter respects protected directions, and whether model answers improve. A pass on one question is not a pass on the others. `B1`, `B2`, and `B3` label the loop, refusal, and root-cause directions. An **arm** is one configuration being compared. Leakage metrics are named in each table because maximum absolute entries and norm ratios are not interchangeable.

The numerical results below are retained experimental reports, not fresh runs made for this documentation review. Some raw runners and receipts remain in the lab evidence bank rather than this repository; the guides identify those limits. Results on the two largest checkpoints use one shard each, not full-model fine-tuning.

### Summary of Experimental Studies & Model Breadth

| Study | Coverage | Model range | Measurements | Detail |
|---|---|---|---|---|
| H100 Stage-1 calibration and 3-arm adapter sweep | 15 models | 0.8B to 32.8B | Reference shifts; 60-step adapter leakage | [4.1](#sec-paper-stage1) |
| H100 Stage-2 public benchmarks | 14 models | 0.8B to 32.8B | GSM8K, MATH-500, MMLU-Pro, HumanEval, ASan-30 | [4.2](#sec-paper-public-benchmarks) |
| H100 Regime C adversarial subspace pull | 14 checkpoints | 7.2B to 72.7B | 12 full-model loads; 2 shard-only writer tests | [4.3](#sec-paper-regime-c) |
| H100 paired ASAN-120 vulnerability and patched-benign twins | 9 models × 7 arms | 0.8B to 32.8B | 7,560 item evaluations; sensitivity and specificity | [4.5](#sec-paper-asan120) |
| Live causal-LM ProjectThenHealLinear healing (Regime A + RMS-relative) | 8 models | 0.8B to 14B | 2 early runs + 6 H100 models; 24,000 benchmark rows in the six-model study | [4.4](#sec-paper-live-healing) |
| Static spectral and Boundary-3 payload atlas | 46-file local inventory (2,619 analyzed weight matrices: 2,574 across 14 full-model reads and 45 across 16 scoped variants) | Atlas plus 10-checkpoint GGUF/SafeTensors scale sweep | 186,870,528 quantized blocks across 912 quantized tensors (`2,194` total tensors) | [5](#sec-paper-static-auditing) |

Cohorts overlap, so these model counts must not be added. The live-healing range uses model-name sizes; the larger H100 tables retain reported parameter counts. The focused Qwen3.5-4B dose-response case study follows the multi-model results in [Section 4.6](#sec-paper-dose-response).

(sec-paper-stage1)=
### 4.1 H100 Stage-1: Calibration and Three Adapter Arms Across 15 Models

This sweep checks the shift at a reference input and compares three adapter methods on one selected writer per model. It uses one seed per model and a synthetic target plus a real-token cross-entropy term. Public behavioral benchmarks were `NOT_RUN` in this sweep; the local numerical results do not establish generation quality.

Across 15 full models on H100 (`0.8B` to `32.8B`, receipt `irrefutable-15models-stage1-receipt.json`), we applied full Target-Shift Calibration ($|\Delta_{\text{target}}| = 7.382$, $K = 128$ columns) to `B2` and `B3` and ran a 60-step 3-arm comparison on the surgically edited writer:

- **Arm A:** Unconstrained LoRA + AdamW
- **Arm B:** Gradient-Projected LoRA ($g_t \leftarrow P_\perp g_t$ before `AdamW.step()`)
- **Arm C:** `ProjectThenHealLinear` ($B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$)

All 30 calibrated edits realize $\Delta_{\text{target}}$ at $h_{\text{ref}}$ to within `<= 7.99e-15` in `float64`. On the 3-arm healing test (measured by relative leakage $\|D \Delta W\|_{\mathrm{F}} / \|\Delta W\|_{\mathrm{F}}$ at step 60), Gradient-Projected AdamW (Arm B) leaks on **15 of 15 models (`0.33%` to `1.67%`)**, whereas `ProjectThenHealLinear` (Arm C) stays at the `float32` working-dtype floor (**`1.78e-10` to `8.67e-10`**; model by model, $6.5 \times 10^6$ to $3.8 \times 10^7$ times lower than Arm B):

- B2 / B3: refusal / root-cause directions; L: layer; β: calibrated step size.
- Shift error is measured in float64 at the reference input. A, B, and C are the adapter arms defined above.

**Stage-1 calibration**

| Model | Size | B2 / B3 layers | B2 / B3 β | Max shift error |
|---|---:|---|---:|---:|
| Qwen3.5-0.8B | 0.8B | L17 / L14 | 0.5801 / 0.3595 | 8.9e-16 |
| SmolLM2-1.7B | 1.7B | L17 / L12 | 0.0369 / 0.2727 | 1.8e-15 |
| Qwen3.5-4B | 4B | L23 / L16 | 0.0484 / 0.3805 | 3.6e-15 |
| Mistral-7B-v0.3 | 7.2B | L12 / L19 | 0.9352 / 0.0510 | 3.6e-15 |
| Qwen2.5-7B | 7.6B | L8 / L14 | 0.1224 / 0.1484 | 2.7e-15 |
| Llama-3.1-8B | 8B | L23 / L16 | 0.4556 / 0.5070 | 1.8e-15 |
| R1-Llama-8B | 8B | L23 / L16 | 0.4343 / 0.4701 | 5.3e-15 |
| Qwen3-8B | 8.2B | L14 / L21 | 0.3931 / 0.0766 | 5.3e-15 |
| Gemma-2-9B | 9.2B | L16 / L25 | 3.7192 / 1.1508 | 1.8e-15 |
| OLMo-2-13B | 13.7B | L16 / L20 | 0.1130 / 0.0325 | 3.6e-15 |
| Phi-4 | 14.7B | L12 / L29 | 1.1436 / 0.0174 | 4.4e-15 |
| Qwen3-14B | 14.8B | L16 / L23 | 0.2994 / 0.0547 | 1.8e-15 |
| R1-Qwen-14B | 14.8B | L28 / L35 | 0.0343 / 0.0177 | 3.6e-15 |
| Gemma-2-27B | 27.2B | L14 / L27 | 0.0845 / 0.0098 | 8.0e-15 |
| Qwen3-32B | 32.8B | L47 / L25 | 0.0292 / 0.0738 | 3.6e-15 |

**Stage-1 relative leakage at step 60**

| Model | A: unconstrained | B: gradient-projected | C: PTH |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 7.60% | 0.61% | **8.67e-10** |
| SmolLM2-1.7B | 2.95% | 1.62% | **5.41e-10** |
| Qwen3.5-4B | 2.22% | 1.67% | **5.81e-10** |
| Mistral-7B-v0.3 | 2.17% | 0.92% | **3.38e-10** |
| Qwen2.5-7B | 2.04% | 0.49% | **4.71e-10** |
| Llama-3.1-8B | 0.82% | 0.80% | **4.36e-10** |
| R1-Llama-8B | 1.59% | 0.60% | **4.50e-10** |
| Qwen3-8B | 2.02% | 0.58% | **3.14e-10** |
| Gemma-2-9B | 8.51% | 0.35% | **5.46e-10** |
| OLMo-2-13B | 1.70% | 0.44% | **1.78e-10** |
| Phi-4 | 0.86% | 0.33% | **3.26e-10** |
| Qwen3-14B | 2.13% | 0.60% | **2.69e-10** |
| R1-Qwen-14B | 1.23% | 0.77% | **3.53e-10** |
| Gemma-2-27B | 1.98% | 1.31% | **3.49e-10** |
| Qwen3-32B | 2.04% | 0.75% | **3.07e-10** |

```{figure} assets/h100-15model-leakage-log10.svg
:name: fig-paper-leakage-log10
:alt: Log-scale relative leakage at step 60 for unconstrained LoRA, gradient-projected AdamW and ProjectThenHealLinear on 15 models, with the B over C ratio per model, and trajectories over 60 AdamW steps
:width: 100%

The three adapter arms in the leakage table, on a log scale. Left: step-60 relative leakage for each model, with the Arm B to Arm C ratio printed above each row. Right: all 15 trajectories over 60 AdamW steps, with the median drawn thick. Arms A and B jump to their plateau in the first 10 steps and stay there. Arm C stays near `1e-10` from step 0 to step 60.
```

(sec-paper-public-benchmarks)=
### 4.2 H100 Stage-2: Public Benchmarks Across 14 Models

Section 4.1 checks the local arithmetic of each edit. This section asks a different question: what the same `B2 + B3` edits do to a model's answers on public benchmarks. The receipt is `public-benchmarks/irrefutable-receipt.json` (SHA-256 `9859ef3e0603318e09b0366e6dae6994b47f7d7cc4456cfff1fff7ddb2253132`, `status: PASS`, completed `2026-11-06T20:41:59Z`), with per-item rows in `predictions.jsonl` (SHA-256 `0c2d65e1...86fe`). It covers 14 of the 15 Stage-1 models (`14 x 1,970 = 27,580` completed five-arm rows). `Gemma-2-27B` is marked `HOLD` in the summary receipt because its gated repository returned HTTP 401 on the edited arms (`300` base-only records for `MATH-500`, `GSM8K`, and `MMLU-Pro` remain in `predictions.jsonl`, bringing the file total to `27,880` rows, while `Gemma-2-27B` has no completed paired benchmark cohort). The aligned edits (layers, column support $S$, $\beta$), model revisions, and seed (`20261105`) are those of Section 4.1, and the healing results in this receipt are identical to the Stage-1 receipt for all 14 shared models.

Five variants of each model answer the same items: the unedited `base`; the calibrated edits at `1.0x` and `1.5x`; and two **norm-matched random** controls at `1.0x` and `1.5x`. A random control replaces each aligned direction $q$ with a random unit vector orthogonal to $q$, keeping the same $\beta$, the same input-column support $S$, and therefore the same $\lVert \Delta W\rVert_{\mathrm{F}}$. If a calibrated edit and its random control move a score by similar amounts, the movement comes from the size of the edit, not its direction.

Each model answers the first 100 items of `GSM8K`, `MATH-500` and `MMLU-Pro`, the first 64 of `HumanEval`, and the 30 retained sanitizer traces (ASan-30). Decoding is greedy. ASan-30 counts a root cause as correct when the answer names both the bug class and the culprit function, matched by keyword. Every comparison is paired by item: an exact two-sided McNemar test on the discordant pairs, then a Holm correction across the 70 model-by-benchmark cells of each variant. The pooled rows add the discordant pairs over all 14 models. Their 95% intervals use the paired normal approximation, and they treat items from different models as independent, which they are not.

```{figure} assets/benchmark-retention-14models.svg
:name: fig-paper-benchmark-retention
:alt: Small multiples of paired change from base for 14 models on GSM8K, MATH-500, MMLU-Pro, HumanEval and ASan-30, comparing calibrated 1.0x edits with norm-matched random 1.0x edits, with pooled diamonds and 95% intervals
:width: 100%

Paired change from base, in percentage points, for every model and benchmark. Blue dots are calibrated `1.0x`; orange rings are the norm-matched random `1.0x` control. A white halo marks a cell that survives Holm correction. Only one does at `1.0x`, and it is a random control: `Gemma-2-9B` `HumanEval`, `42.2%` to `75.0%`. The diamonds pool all 14 models.
```

- pp: percentage points; CI: 95% confidence interval; p: exact two-sided McNemar p-value.
- Each result reports change from base, its CI, and p; items are pooled over 14 models.

**Pooled calibrated edits**

| Benchmark | Paired items | Calibrated 1.0x | Calibrated 1.5x |
|---|---:|---:|---:|
| GSM8K | 1400 | -0.9 (-2.9 to +1.0), p = 0.39 | -2.0 (-4.1 to +0.1), p = 0.079 |
| MATH-500 | 1400 | -0.5 (-2.4 to +1.4), p = 0.65 | -0.4 (-2.2 to +1.5), p = 0.76 |
| MMLU-Pro | 1400 | -2.0 (-3.5 to -0.5), p = 0.013 | -2.4 (-3.9 to -1.0), p = 0.0015 |
| HumanEval pass@1 | 896 | +0.6 (-1.7 to +2.9), p = 0.70 | +0.3 (-2.0 to +2.7), p = 0.85 |
| ASan-30 root cause | 420 | +2.1 (-0.8 to +5.0), p = 0.20 | +1.2 (-1.8 to +4.2), p = 0.53 |

**Pooled norm-matched random controls**

| Benchmark | Paired items | Random 1.0x | Random 1.5x |
|---|---:|---:|---:|
| GSM8K | 1400 | -0.8 (-2.6 to +1.0), p = 0.43 | -0.9 (-3.0 to +1.1), p = 0.41 |
| MATH-500 | 1400 | -1.4 (-3.2 to +0.4), p = 0.14 | -0.1 (-2.0 to +1.7), p = 0.94 |
| MMLU-Pro | 1400 | -0.5 (-2.0 to +1.0), p = 0.57 | -2.0 (-3.6 to -0.4), p = 0.017 |
| HumanEval pass@1 | 896 | +0.4 (-1.8 to +2.7), p = 0.77 | +0.3 (-1.8 to +2.5), p = 0.84 |
| ASan-30 root cause | 420 | +0.2 (-3.2 to +3.6), p = 1.0 | -2.9 (-5.6 to -0.1), p = 0.065 |

What this shows:

- **Retention.** At `1.0x`, no calibrated cell changes significantly after Holm correction (`0/70`). Single cells range from `-12` to `+19` points, which is what 100-item and 64-item samples produce by chance. Pooled, each benchmark moves by `2.1` points or less.
- **One pooled cost, MMLU-Pro.** Calibrated edits lower pooled `MMLU-Pro` by `2.0` points at `1.0x` and `2.4` at `1.5x`. Random `1.5x` lowers it by `2.0`, so edit size explains at least part of the cost. Under a Holm correction across all 20 pooled tests, only calibrated `1.5x` remains below `0.05` (`p = 0.031`).
- **No measured uplift in the 14-model Stage-2 sweep.** The ASan-30 root-cause gain at `1.0x` (`+2.1` points, `p = 0.20`) is not significant, and no root-cause cell survives Holm correction. These edits were sized to hit a reference shift, not tuned for this task. This receipt does not reproduce the `Qwen3.5-4B` thinking-loop root-cause gains reported in Section 4.6 (`0/30` to `14/30` at `1.0x` and `19/30` at `1.5x` under the `B1` loop-exit + `B2` + `B3` composition).
- **Edit size matters more than direction on one model.** `Gemma-2-9B` carries the largest `B2` step in the sweep ($\beta = 3.72$). It accounts for every Holm-significant cell across the four variants: `GSM8K` falls `17` points at calibrated `1.5x` and `14` at random `1.5x`, and `HumanEval` rises `33` to `34` points under both random controls. The random controls move it as far as the calibrated edits do.

```{figure} assets/asan30-audit-uplift.svg
:name: fig-paper-asan30
:alt: Per-model ASan-30 scores out of 30 for root cause, culprit frame and bug class, comparing base, calibrated 1.0x and norm-matched random 1.0x, with paired root-cause changes and exact p-values
:width: 100%

ASan-30 counts out of 30 for each model: base (gray ring), calibrated `1.0x` (blue), and random `1.0x` (orange). The right column lists the paired root-cause change and its unadjusted exact McNemar p. The largest single gains are `+4` (`Llama-3.1-8B`, `p = 0.12`) and `+3` (`Mistral-7B-v0.3`, `p = 0.25`).
```

> **Scope.** These are first-N items, not random samples, from one seed, decoded greedily, on a set of checkpoints weighted toward Qwen. The receipt's standalone `secant` CLI cross-check is `HOLD` for 7 of the 14 models; that check is separate from these benchmark results.

**Per-model counts, all 14 completed models.** Each cell is `base` / calibrated `1.0x` / norm-matched random `1.0x`, as correct items out of 100 (`GSM8K`, `MATH-500`, `MMLU-Pro`), passed items out of 64 (`HumanEval`), and correct root causes out of 30 (ASan-30). The rows are generated from the receipt with `jq`, not typed. Two rows that are easy to over-read: `OLMo-2-13B` `MATH-500` goes from `33` to `37` under the calibrated edit and also to `37` under the random control. `Qwen3-32B` `MATH-500` goes from `31` to `38` (exact McNemar `p = 0.19`, Holm `p = 1`), while `GSM8K` moves from `53` to `55` calibrated and to `56` random. No cell in this table is a demonstrated uplift.

- Cell order: base / calibrated 1.0x / norm-matched random 1.0x.
- Denominators are shown in the column headers.

**Per-model mathematics and knowledge**

| Model | Size | GSM8K /100 | MATH-500 /100 | MMLU-Pro /100 |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0.8B | 48 / 44 / 47 | 30 / 26 / 23 | 22 / 20 / 17 |
| SmolLM2-1.7B | 1.7B | 23 / 23 / 21 | 25 / 26 / 23 | 4 / 5 / 4 |
| Qwen3.5-4B | 4.0B | 14 / 21 / 20 | 20 / 12 / 20 | 4 / 4 / 3 |
| Mistral-7B-v0.3 | 7.2B | 55 / 46 / 49 | 13 / 16 / 7 | 25 / 25 / 28 |
| Qwen2.5-7B | 7.6B | 89 / 89 / 89 | 59 / 60 / 58 | 56 / 58 / 54 |
| Llama-3.1-8B | 8.0B | 78 / 83 / 82 | 44 / 38 / 41 | 48 / 46 / 51 |
| R1-Llama-8B | 8.0B | 66 / 67 / 65 | 41 / 43 / 42 | 7 / 3 / 4 |
| Qwen3-8B | 8.2B | 39 / 37 / 34 | 31 / 33 / 33 | 9 / 9 / 9 |
| Gemma-2-9B | 9.2B | 87 / 78 / 77 | 48 / 36 / 39 | 55 / 45 / 52 |
| OLMo-2-13B | 13.7B | 90 / 90 / 89 | 33 / 37 / 37 | 32 / 28 / 31 |
| Phi-4 | 14.7B | 95 / 96 / 95 | 65 / 65 / 67 | 68 / 67 / 66 |
| Qwen3-14B | 14.8B | 59 / 58 / 59 | 36 / 40 / 36 | 15 / 10 / 16 |
| R1-Qwen-14B | 14.8B | 50 / 46 / 52 | 40 / 39 / 39 | 6 / 3 / 3 |
| Qwen3-32B | 32.8B | 53 / 55 / 56 | 31 / 38 / 31 | 7 / 7 / 13 |

**Per-model code and crash triage**

| Model | HumanEval /64 | ASan-30 root cause /30 |
|---|---:|---:|
| Qwen3.5-0.8B | 5 / 5 / 1 | 5 / 7 / 12 |
| SmolLM2-1.7B | 7 / 6 / 7 | 5 / 6 / 6 |
| Qwen3.5-4B | 21 / 20 / 19 | 6 / 5 / 6 |
| Mistral-7B-v0.3 | 2 / 0 / 0 | 12 / 15 / 10 |
| Qwen2.5-7B | 9 / 11 / 8 | 17 / 16 / 18 |
| Llama-3.1-8B | 22 / 18 / 23 | 17 / 21 / 18 |
| R1-Llama-8B | 6 / 7 / 7 | 18 / 20 / 16 |
| Qwen3-8B | 25 / 21 / 23 | 17 / 16 / 17 |
| Gemma-2-9B | 27 / 39 / 48 | 13 / 13 / 13 |
| OLMo-2-13B | 8 / 8 / 6 | 22 / 22 / 20 |
| Phi-4 | 4 / 7 / 4 | 23 / 23 / 23 |
| Qwen3-14B | 13 / 9 / 10 | 15 / 15 / 13 |
| R1-Qwen-14B | 26 / 27 / 20 | 16 / 16 / 15 |
| Qwen3-32B | 14 / 16 / 17 | 15 / 15 / 15 |

(sec-paper-regime-c)=
### 4.3 H100 Regime C: Adversarial Subspace Pull Across 14 Checkpoints

This stress test deliberately asks the adapter to learn a change inside the protected subspace. A successful lock must leave that part of the task unfitted, so low leakage comes with a loss floor. The unconstrained and locked numbers below use different metrics and must not be read as a direct ratio.

In a separate adversarial stress test across 12 full models (`7.2B` to `32.8B`) and 2 single-shard runs (`70.6B`, `72.7B`), we trained LoRA for 60 AdamW steps toward a synthetic target $Y_{\text{target}} = Y + 1.5\,v_{B2}$ that deliberately lies *inside* the protected subspace $\mathrm{span}(D^\top)$. Under this adversarial pull, unconstrained LoRA placed `8.42%` to `52.55%` of its output factor norm $\|B\|_{\mathrm{F}}$ into $\mathrm{span}(D^\top)$, whereas `ProjectThenHealLinear` held $\max |D B_{\text{locked}}|$ (evaluated in `float64` on the `F32` working factor) to **`4.82e-10` to `1.08e-9`** at step 60 across all 14 checkpoints.

- UNC rel. B: fraction of the unconstrained output-factor norm in the protected subspace.
- PTH max abs B: maximum absolute entry of D B after projection, evaluated in float64 on the F32 factor. The two columns use different metrics.
- Full model: the full checkpoint was loaded, with the adapter test on a selected writer. Shard only: one writer from a single checkpoint shard, without full-model fine-tuning or generation.

| Checkpoint | Size | Coverage | UNC rel. B | PTH max abs B |
|---|---:|---|---:|---:|
| Mistral-7B-v0.3 | 7.2B | Full model | 46.79% | 7.73e-10 |
| Qwen2.5-7B | 7.6B | Full model | 31.34% | 7.55e-10 |
| Llama-3.1-8B | 8B | Full model | 48.03% | 7.62e-10 |
| R1-Llama-8B | 8B | Full model | 52.55% | 6.39e-10 |
| Qwen3-8B | 8.2B | Full model | 39.53% | 7.14e-10 |
| Gemma-2-9B | 9.2B | Full model | 41.05% | 6.53e-10 |
| OLMo-2-13B | 13.7B | Full model | 30.73% | 1.08e-9 |
| Phi-4 | 14.7B | Full model | 8.42% | 9.51e-10 |
| Qwen3-14B | 14.8B | Full model | 41.33% | 5.77e-10 |
| R1-Qwen-14B | 14.8B | Full model | 32.57% | 6.11e-10 |
| Gemma-2-27B | 27.2B | Full model | 11.54% | 7.16e-10 |
| Qwen3-32B | 32.8B | Full model | 30.75% | 4.82e-10 |
| R1-Llama-70B | 70.6B | Shard only | 20.24% | 5.66e-10 |
| Qwen2.5-72B | 72.7B | Shard only | 36.94% | 9.85e-10 |

The retained Regime C sources are `h100_scaleup_6models_complete_receipt.json`, `h100_r1_olmo_receipt.json`, and `h100_gated_and_70b_receipt.json` in the H100 evidence bank. The table uses the first Phi-4 run under `experiments` and the successful top-level Qwen3-32B rerun. The 70B row is **DeepSeek-R1-Distill-Llama-70B**, as named in the receipt; the attempted **Llama-3.3-70B-Instruct** download returned HTTP 403 and is not a completed result. Both large-model rows remain shard-only retained evidence with incomplete producing-run provenance.

> **Verification & Diagnostic Notes on Regime C:**
> - **Non-linear dose saturation at extreme shifts:** On 20 of 24 full-model edits in Regime C, raising the dose from $1.0\times$ to $1.5\times$ scaled the realized shift linearly (`1.49x` to `1.50x`). Four edits (`Phi-4` `B2`, `OLMo-2-13B` `B3`, `Gemma-2-9B` `B3`, `Gemma-2-27B` `B3`) showed sub-linear scaling (`1.04x` to `1.18x`) due to downstream RMSNorm normalization.
> - **Outlier-dimension diagnostic (`Phi-4`):** In Regime C's uncalibrated capture on `Phi-4` (`L32`), peak $\mathrm{SNR}_{\text{proj}}$ reached `27,362.3` with $\mathrm{SNR}_{\text{trace}} = 60.4$, and the uncalibrated edit inverted sign (`-3.399`). Always pair $\mathrm{SNR}_{\text{proj}}$ with $\mathrm{SNR}_{\text{trace}}$ and verify the signed shift on held-out prompts (as done in Stage-1 L12 calibration, where `Phi-4` achieves `4.4e-15` shift error).

(sec-paper-live-healing)=
### 4.4 Live Causal-LM Healing Across Eight Models: Regime A and RMS-Relative Dosing

**Live causal-LM healing coverage: eight models**

The two early Regime A runs and six H100 RMS-relative runs all train ProjectThenHealLinear on real causal-LM text. They cover eight models from 0.8B to 14B by model-name size. The **24,000 benchmark rows belong to the six-model RMS-relative study**, across all of its evaluated arms; they are not 24,000 healed-arm rows or an eight-model common benchmark.

- PTH: ProjectThenHealLinear. Max abs: maximum absolute entry of D ΔW; relative: Frobenius norm ratio of D ΔW to ΔW. These are different metrics and cannot be compared as one leakage range.
- The early runs recompute an unrounded float64 projected delta; the H100 runs use a float64 adapter. Neither label means that every operation in the language model uses float64.
- R1-Llama-8B abbreviates DeepSeek-R1-Distill-Llama-8B.

| Model | Run | Steps | Leakage metric | PTH float64 |
|---|---|---:|---|---:|
| Qwen3.5-0.8B | Early Regime A (CPU) | 60 | Max abs | 9.87e-18 |
| SmolLM2-1.7B | Early Regime A (ROCm) | 20 | Max abs | 5.21e-18 |
| Qwen3-4B | H100 RMS-relative | 60 | Relative | 3.18e-17 |
| Qwen2.5-Math-7B | H100 RMS-relative | 60 | Relative | 5.07e-17 |
| Qwen2.5-Coder-7B | H100 RMS-relative | 60 | Relative | 3.34e-17 |
| R1-Llama-8B | H100 RMS-relative | 60 | Relative | 3.43e-17 |
| Qwen3-8B | H100 RMS-relative | 60 | Relative | 3.75e-17 |
| Qwen3-14B | H100 RMS-relative | 60 | Relative | 4.80e-17 |

#### 4.4.1 Early Live Runs: Perplexity and Storage Precision

The two early short runs compare a constrained adapter with ordinary LoRA after a weight edit. Perplexity measures prediction quality on held-out text; lower is better. The Qwen run reached its best held-out value at step 20, then both arms overfit the small training batch. The `float64` leakage column recomputes an unrounded projected delta; it is not a claim that the whole model ran in `float64`.

Both arms train on real causal-LM completions after a byte-verified surgical edit, evaluated on the identical writer-level metric $\max |D \Delta W|$:

- PTH: ProjectThenHealLinear; UNC: unconstrained LoRA.
- Max leakage: maximum absolute entry of D ΔW, evaluated identically for the two arms. PPL: held-out perplexity.

**Early live runs: training and edit**

| Model | Steps / device | Verified surgical edit |
|---|---:|---|
| Qwen/Qwen3.5-0.8B | 60 (CPU) | Composed B1+B2+B3 GGUF (verify-patch exit 0) |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 20 (ROCm GPU) | B2 projector at L14 o\_proj (67.9x vs. random) + B3 calibrated at L15 down\_proj (22.1x vs. random) |

**Early live runs: leakage and held-out quality**

| Model | PTH max: float64 / F32 | UNC max | PPL: PTH vs. UNC |
|---|---:|---:|---|
| Qwen/Qwen3.5-0.8B | **9.87e-18** / **2.49e-10** | 1.05e-02 | Step 20: 5.435 vs. 5.423; Step 60: 12.106 vs. 12.357 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | **5.21e-18** / **1.46e-10** | 4.64e-03 | Step 20: 21.260 vs. 21.255 |

> **Storage Precision Boundary (`F32`/`F64` vs. `BF16`):** When merged weights are quantized down to 16-bit `BF16` (7 stored fraction bits, 8-bit significand precision, $\mathrm{ulp} \approx 7.8 \times 10^{-3}$, unit roundoff $\approx 3.9 \times 10^{-3}$), rounding error alone introduces $\sim 10^{-4}$ to $10^{-3}$ subspace leakage. `secant heal` therefore stores healed LoRA adapters in `F32` or `F64`.

(sec-paper-rms-dosing)=
#### 4.4.2 Six-Model H100 Sweep: RMS-Relative Dosing and Live Healing

A fixed shift such as $\lvert\Delta_{\text{target}}\rvert = 1.0$ means different things in different layers, because writer outputs differ in scale. **RMS-relative dosing** measures that scale first and sizes the shift against it:

$$
\sigma_{\text{writer}} = \sqrt{\tfrac{1}{N}\textstyle\sum_{i=1}^{N} \lVert W h_i\rVert_2^2}, \qquad \Delta_{\text{target}} = \alpha\,\sigma_{\text{writer}}, \qquad \beta_k = \frac{\Delta_{\text{target}}}{\lVert h_{S,k}\rVert_2^2}
$$

over $N = 24$ positive reference activations (4 domains times 6). Across 6 models spanning `4B` to `14B` parameters (`24,000` benchmark rows total, archived experiment receipt identifier `h100-scaleup-20261105/domain-uplift-rms`, per-model manifest SHA-256 digests listed below), $\sigma_{\text{writer}}$ varies by `13.4x`, from `5.847` on layer 21 of `DeepSeek-R1-Distill-Llama-8B` to `78.148` on layer 24 of `Qwen3-8B`. An unscaled unit shift of `1.0` therefore represents `17.10%` of the writer norm on `DeepSeek-R1-Distill-Llama-8B`, `2.48%` on `Qwen2.5-Coder-7B-Instruct` (`40.355`), `1.45%` on `Qwen3-14B` (`68.757`), `1.34%` on `Qwen2.5-Math-7B-Instruct` (`74.607`), and `1.28%` on `Qwen3-8B` (`78.148`). That explains why the earlier unscaled unit shift left greedy accuracy unchanged on `Qwen2.5-Math-7B-Instruct` (`GSM8K` `87%`, `MATH-500` `43%`, largest change in any complete cell `1` point, with mean normalized shift at most `0.19%`).

Each of the 6 models was evaluated on one H100 (`torch 2.8.0+cu128`, greedy left-padded batches of 16, `max_new_tokens = 512`, 100 items each of `MATH-500`, `GSM8K`, `MMLU-Pro`, and `HumanEval`). Every model ran at least 8 arms at $\alpha = 1.0$: `base`, four single-domain edits (`security`, `math`, `science`, `code`), their 4-domain Gram-Schmidt orthogonal union (`composed`), a seeded norm-matched `random` control orthogonal to $\mathrm{span}(D)$ on the identical column support, and `composed_healed` (the 4-domain composition followed by a rank-4, 60-step `float64` `ProjectThenHealLinear` healing pass). `Qwen2.5-Math-7B-Instruct` and `Qwen3-4B` additionally ran a 14-arm dose sweep over $\alpha \in \{0.25, 0.5, 1.0, 2.0\}$.

- L: writer layer; σ: RMS writer output norm; rel. leakage: norm ratio for the float64 adapter delta.
- Benchmark cell order: base / composed / random / **composed + PTH**, all at α = 1.0. Each benchmark has 100 items.
- Manifest hashes below are abbreviated source identifiers, not complete verification digests.

**RMS calibration and live healing**

| Model | Layer | σ | Manifest SHA-256 | Rel. leakage |
|---|---:|---:|---|---:|
| Qwen2.5-Math-7B-Instruct | L19 | 74.61 | 49181d79...7cc1 | 5.07e-17 |
| Qwen2.5-Coder-7B-Instruct | L19 | 40.35 | 16c9808e...ad6b | 3.34e-17 |
| DeepSeek-R1-Distill-Llama-8B | L21 | 5.85 | 5bae03b9...4389 | 3.43e-17 |
| Qwen3-4B | L24 | 37.81 | bcc4e37e...8ba6 | 3.18e-17 |
| Qwen3-8B | L24 | 78.15 | 80583fe0...f7d7 | 3.75e-17 |
| Qwen3-14B | L26 | 68.76 | 42de8edd...8f1a | 4.80e-17 |

**RMS sweep: held-out benchmark counts**

| Model | GSM8K /100 | MATH-500 /100 | MMLU-Pro /100 | HumanEval /100 |
|---|---:|---:|---:|---:|
| Qwen2.5-Math-7B-Instruct | 87 / 87 / 88 / **93** | 43 / 43 / 44 / 44 | 17 / 16 / 16 / 17 | 1 / 0 / 1 / 0 |
| Qwen2.5-Coder-7B-Instruct | 82 / 81 / 82 / 80 | 44 / 44 / 46 / 45 | 35 / 29 / 27 / 31 | 36 / 36 / 37 / **42** |
| DeepSeek-R1-Distill-Llama-8B | 62 / 63 / 64 / 63 | 27 / 26 / 29 / 27 | 2 / 3 / 2 / 2 | 5 / 9 / 4 / **12** |
| Qwen3-4B | 30 / 30 / 26 / 29 | 21 / 22 / 19 / 22 | 8 / 7 / 12 / 8 | 19 / 23 / 18 / 21 |
| Qwen3-8B | 18 / 30 / 30 / 22 | 21 / 20 / 16 / 18 | 1 / 3 / 2 / 3 | 19 / 16 / 20 / 15 |
| Qwen3-14B | 46 / 42 / 38 / 37 | 17 / 20 / 21 / 17 | 9 / 12 / 9 / **15** | 10 / 10 / 10 / 8 |

On `Qwen2.5-Math-7B-Instruct` (`manifest.json` SHA-256 `49181d79...7cc1`, 14 arms, `predictions.jsonl` SHA-256 `f5036530...b90c`), the full dose sweep across $\alpha \in \{0.25, 0.5, 1.0, 2.0\}$ realized every reference target to within `1e-4` relative error:

- α: dose relative to the RMS writer output norm. Slash-separated results follow the arm or dose order shown.

**Qwen2.5-Math-7B dose sweep: mathematics**

| Arm | α | GSM8K /100 | MATH-500 /100 |
|---|---:|---:|---:|
| base |  | 87 | 43 |
| security / math / science / code | 1.0 | 87 / 87 / 87 / 88 | 42 / 44 / 43 / 43 |
| composed | 0.25 / 0.5 / 1.0 / 2.0 | 88 / 89 / 87 / 89 | 42 / 43 / 43 / 42 |
| random (norm-matched) | 0.25 / 0.5 / 1.0 / 2.0 | 87 / 87 / 88 / 87 | 42 / 43 / 44 / 43 |
| **composed\_healed** (Project-Then-Heal) | 1.0 | **93** | 44 |

**Qwen2.5-Math-7B dose sweep: knowledge and code**

| Arm | α | MMLU-Pro /100 | HumanEval /100 |
|---|---:|---:|---:|
| base |  | 17 | 1 |
| security / math / science / code | 1.0 | 14 / 16 / 17 / 15 | 2 / 1 / 0 / 1 |
| composed | 0.25 / 0.5 / 1.0 / 2.0 | 16 / 17 / 16 / 15 | 0 / 0 / 0 / 0 |
| random (norm-matched) | 0.25 / 0.5 / 1.0 / 2.0 | 15 / 15 / 16 / 14 | 2 / 0 / 1 / 0 |
| **composed\_healed** (Project-Then-Heal) | 1.0 | 17 | 0 |

What the 6-model sweep shows, and what it does not:

- **Unhealed composed edits match norm-matched random controls.** Across all 6 models, `composed_rms_1p0x` stays within noise of `random_rms_1p0x` on `GSM8K` and `MATH-500`. Even when `GSM8K` rises from `18` (`base`) to `30` under `composed_rms_1p0x` on `Qwen3-8B` (unadjusted `p = 0.017`), `random_rms_1p0x` also rises to `30` (unadjusted `p = 0.029`). Both perturbations slightly shorten `<think>` chains on a model where `98` to `100` of `100` completions hit the 512-token cap; without the random control, that truncation shift would be misread as domain uplift.
- **`composed_healed` (`ProjectThenHealLinear`) raises in-domain scores on the specialist models while keeping relative subspace leakage at `3.18e-17` to `5.07e-17`.** On `Qwen2.5-Math-7B-Instruct`, `composed_healed_rms_1p0x` raises `GSM8K` from `87` to `93` (`6` gains, `0` losses vs. both `base` and `composed_rms_1p0x`, exact McNemar `p = 0.031` unadjusted, Holm `1.0` across 110 tests). On `Qwen2.5-Coder-7B-Instruct`, `composed_healed_rms_1p0x` raises `HumanEval` from `36` to `42` (`12` gains, `6` losses vs. `base`, `p = 0.238`; `11` gains, `5` losses vs. `composed_rms_1p0x`, `p = 0.210`; Holm `1.0`) and partially recovers the unhealed edit's `MMLU-Pro` drop (`35 -> 29` unhealed vs. `31` healed). On `DeepSeek-R1-Distill-Llama-8B`, `composed_healed_rms_1p0x` raises `HumanEval` from `5` (`base`) and `4` (`random_rms_1p0x`) to `12` (`10` gains, `3` losses vs. `base`, `p = 0.092`, Holm `1.0`) by closing `</think>` before 512 tokens on 6 additional problems (`93` truncated vs. `99` on `base`). On `Qwen3-14B`, `math_rms_1p0x` (`16/100`, `p = 0.039` unadjusted) and `composed_healed_rms_1p0x` (`15/100`, `p = 0.146`) lift `MMLU-Pro` over `base` and `random_rms_1p0x` (`9/100` each), while `composed_healed_rms_1p0x` increases `GSM8K` `<think>` truncation from `72` to `86` of `100` (`46 -> 37`).
- **The healing gains cannot be attributed to the linear composition alone.** In every model, the healing objective is causal-LM cross-entropy on fixed neutral texts with $\mathrm{span}(D)$ structurally locked. The sweep does not include a healed-random arm or a LoRA-on-unedited-base arm, and none of the individual paired McNemar tests survives Holm-Bonferroni correction across the full multi-arm test family.
- **Calibration is exact at the reference inputs only.** On benchmark prompts the composed edit's mean prefill shift along each direction is `0.5%` to `14.8%` of the base output norm, not `1.0x`. The random control's shifts along its own directions agree with these to within `0.01%` of the base norm.

(sec-paper-asan120)=
### 4.5 H100 ASAN-120: Paired Vulnerability and Patched-Benign Twins, 9 Models × 7 Arms

Vulnerable-only benchmarks such as ASan-30 cannot distinguish a genuine C/C++ vulnerability detector from a rubber-stamping model that labels every snippet `VULNERABLE`. To measure both sensitivity (`TPR`) and patched-twin specificity (`TNR`), we constructed **`ASAN-120`** (archived benchmark artifact `h100-scaleup-20261105/asan120/asan120_paired_benchmark.json`, SHA-256 `4276cbd2...20eb`): 60 vulnerable C/C++ functions paired with 60 minimally patched benign twins across 12 CWE bug classes (5 matched pairs per class: `CWE-122` heap overflow, `CWE-121` stack overflow, `CWE-416` use-after-free including `std::vector` reallocation, `CWE-190` integer overflow, `CWE-195` sign conversion, `CWE-193` off-by-one, `CWE-476` null dereference, `CWE-415` double free, `CWE-908` uninitialized read under MSan, `CWE-617` assert-only check elided under `-DNDEBUG`, `CWE-843` type confusion, and `CWE-674` unbounded recursion). Every pair was compiled and executed across a 4-way compiler matrix (`gcc -O0`, `gcc -O2`, `clang -O0`, `clang -O2` with `ASan+UBSan` and `-DNDEBUG`, plus `clang` `MSan` for `CWE-908`; `validation.json` SHA-256 `adfe3d82...c9ca`, `60/60` PASS): each vulnerable twin is silent on a legitimate driver and triggers the expected sanitizer fault on a crafted driver, while each patched twin is silent on both.

Across 9 models (`0.8B` to `32.8B`) and 7 steering arms (`base`, `calibrated_1p0x`, `random_1p0x`, `composed_rms_0p5x`, `random_rms_0p5x`, `composed_rms_1p0x`, `random_rms_1p0x`; `63` arms, `7,560` completions on H100, greedy batch 16, `max_new_tokens = 1536`, `restored = True` bit-exact after every arm, receipt `asan120-receipt.json` / `asan120_results.json` SHA-256 `cc5a7d40...bb9b`, `0` re-score mismatches), each item is scored for binary verdict (`TPR` on the 60 vulnerable snippets, `TNR` on the 60 patched twins, balanced accuracy $\mathrm{BA} = (\mathrm{TPR}+\mathrm{TNR})/2$, Matthews correlation $\mathrm{MCC}$, and `pair` where both twins in a pair are classified correctly) and strict **root-cause recall (`RC`)**, which credits a vulnerable item only when the model outputs `VULNERABLE`, names the gold CWE bug class (`class`), and names the faulting sink symbol (`sink`). Unclosed `<think>` chains (`cap`) are scored as errors (`0` credit on `RC`).

```{figure} assets/asan120-paired-audit-uplift.svg
:name: fig-paper-asan120
:alt: ASAN-120 paired vulnerability and benign control benchmark across 9 models and 7 steering arms, showing root-cause recall, balanced accuracy, TPR vs TNR specificity breakdown, and paired deltas
:width: 100%

`ASAN-120` paired evaluation across 9 models (`7,560` completions). Top 5 rows: non-reasoning models (`0.8B` to `8B`) cluster on the `BA = 0.50` chance line (`0.475` to `0.533`), either defaulting to `SAFE` (`Qwen3.5-0.8B` `TNR = 1.000`, `Qwen2.5-7B` `TNR = 0.850`) or rubber-stamping `VULNERABLE` (`SmolLM2-1.7B` `TPR = 0.983, TNR = 0.000`; `Llama-3.1-8B` `TPR = 0.933, TNR = 0.067`), with near-zero root-cause recall (`RC <= 0.067`). Bottom 4 rows: `<think>` reasoning models (`8B` to `32.8B`) achieve genuine root-cause attribution (`RC` up to `0.550` on `Qwen3-14B` and `0.683` on `Qwen3-32B`), and `composed_rms_1p0x` beats `random_rms_1p0x` on `BA` across `6` of `9` models (`6W / 1L / 2T`, including `4/4` reasoning models).
```

- Cell order: base / c0.5 / r0.5 / c1.0 / r1.0; c is composed RMS editing and r is its norm-matched random control. The suffix is the RMS-relative dose.
- TPR: vulnerable-item sensitivity; TNR: patched-twin specificity; BA: balanced accuracy; MCC: Matthews correlation.
- RC: strict root-cause recall (verdict + class + sink); pair: both twins correct; parsed: parsed-answer fraction; cap: unclosed think-chain fraction.
- A single value or range in the last table retains the original summary across arms; cap is shown only where reported.

**ASAN-120: verdict quality and specificity**

| Model | TPR | TNR | BA | MCC |
|---|---:|---:|---:|---:|
| **Qwen3.5-0.8B** | .00 / .00 / .00 / .00 / .00 | 1.0 / 1.0 / 1.0 / 1.0 / 1.0 | .500 / .500 / .500 / .500 / .500 | .00 / .00 / .00 / .00 / .00 |
| **SmolLM2-1.7B** | .95 / .97 / .98 / .98 / .98 | .00 / .00 / .00 / .00 / .00 | .475 / .483 / .492 / .492 / .492 | -.16 / -.13 / -.09 / -.09 / -.09 |
| **Mistral-7B-v0.3** | .33 / .35 / .33 / **.35** / .30 | .72 / .72 / .72 / **.72** / .72 | .525 / .533 / .525 / **.533** / .508 | +.05 / +.07 / +.05 / **+.07** / +.02 |
| **Qwen2.5-7B** | .12 / .12 / .10 / .12 / .12 | .87 / .85 / .87 / .85 / .87 | .492 / .483 / .483 / .483 / .492 | -.03 / -.05 / -.05 / -.05 / -.03 |
| **Llama-3.1-8B** | .93 / .93 / .93 / **.93** / .93 | .05 / .05 / .05 / **.07** / .05 | .492 / .492 / .492 / **.500** / .492 | -.04 / -.04 / -.04 / **.00** / -.04 |
| **R1-Llama-8B** | .60 / **.72** / .68 / .65 / .63 | .23 / .25 / .27 / .23 / .20 | .417 / **.483** / .475 / .442 / .417 | -.18 / **-.04** / -.06 / -.13 / -.18 |
| **Qwen3-8B** | .43 / .38 / .40 / **.45** / .43 | .28 / .32 / .43 / **.28** / .25 | .358 / .350 / .417 / **.367** / .342 | -.29 / -.30 / -.17 / **-.27** / -.32 |
| **Qwen3-14B** | .63 / .65 / .65 / **.70** / .63 | .50 / **.47** / .38 / .40 / .42 | .567 / **.558** / .517 / **.550** / .525 | +.13 / **+.12** / +.03 / **+.10** / +.05 |
| **Qwen3-32B** | .78 / .73 / .82 / .73 / .77 | .50 / .40 / .52 / **.55** / .42 | .642 / .567 / .667 / **.642** / .592 | +.30 / +.14 / +.35 / **+.29** / +.20 |

**ASAN-120: root cause and paired controls**

| Model | RC | Pair accuracy | Parsed (cap) |
|---|---:|---:|---:|
| **Qwen3.5-0.8B** | .000 / .000 / .000 / .000 / .000 | .000 (all) | 1.00 (.00) |
| **SmolLM2-1.7B** | .017 / .000 / .017 / .000 / .017 | .000 (all) | .97-.99 (.00) |
| **Mistral-7B-v0.3** | .033 / .033 / .033 / **.033** / .017 | .050 / .067 / .050 / **.067** / .033 | 1.00 (.00) |
| **Qwen2.5-7B** | .017 / .017 / .017 / .017 / .017 | .033 / .017 / .017 / .017 / .033 | 1.00 (.00) |
| **Llama-3.1-8B** | .033 / .050 / .033 / .033 / .033 | .033 (all) | .97 (.01) |
| **R1-Llama-8B** | .183 / .217 / .133 / **.250** / .133 | .067 / .117 / .167 / **.167** / .083 | .68 (.32) / **.72 (.28)** / .69 (.32) / .68 (.36) / .67 (.35) |
| **Qwen3-8B** | .350 / .267 / .300 / **.317** / .283 | .117 / .150 / .167 / **.150** / .100 | .39 / .44 / .48 / **.46** / .40 |
| **Qwen3-14B** | .450 / .500 / .500 / **.550** / .467 | .250 / **.267** / .217 / **.267** / .217 | .68 / .67 / .62 / **.67** / .62 |
| **Qwen3-32B** | .650 / .633 / .683 / **.633** / .633 | .367 / .333 / .433 / **.383** / .333 | .72 (.31) / .67 (.33) / .72 (.32) / **.74 (.29)** / .68 (.35) |

What the paired `ASAN-120` evaluation establishes:

- **Patched benign twins expose one-sided rubber-stamping in non-reasoning models (`0.8B` to `8B`).** Without `<think>` chain-of-thought reasoning, all five non-reasoning checkpoints collapse to near-chance balanced accuracy (`BA = 0.475` to `0.533`, `RC <= 0.067`): `Qwen3.5-0.8B` (`TNR = 1.000, TPR = 0.000`), `Qwen2.5-7B` (`TNR = 0.850-0.867, TPR = 0.100-0.117`), and `Mistral-7B-v0.3` (`TNR = 0.717, TPR = 0.300-0.350`) default toward `SAFE`, whereas `SmolLM2-1.7B` (`TPR = 0.950-0.983, TNR = 0.000`) and `Llama-3.1-8B` (`TPR = 0.933-0.967, TNR = 0.033-0.067`) rubber-stamp almost every function `VULNERABLE`. On a vulnerable-only benchmark, `Llama-3.1-8B`'s `93.3%` positive rate looks like high sensitivity; paired `TNR` (`5.0%` on `base`, `6.7%` on `composed_rms_1p0x`) and `pair` accuracy (`3.3%`) show that it rarely clears a patched fix.
- **At `1.0x` RMS dosing, `composed_rms_1p0x` beats its norm-matched random control `random_rms_1p0x` on `6` of `9` models (`6W / 1L / 2T`, exact sign-test `p = 0.1250`, mean `dBA = +0.017`), including all `4` of `4` `<think>` reasoning models (`4W / 0L / 0T` on `BA` and `MCC`):**
  - On **`Qwen3-14B`**, `composed_rms_1p0x` lifts strict root-cause recall `RC` from `0.450` (`base`, `27/60`) and `0.467` (`random_rms_1p0x`, `28/60`) to **`0.550` (`33/60`, `+10.0` pp over `base`, `+8.3` pp over `random_rms_1p0x`)**, CWE bug-class recall from `0.500` to **`0.617` (`+11.7` pp)**, sink recall from `0.567` to **`0.617` (`+5.0` pp over `base`, `+8.4` pp over `random_rms_1p0x`)**, and `TPR` from `0.633` to **`0.700` (`+6.7` pp)**, while beating `random_rms_1p0x` on `BA` (`0.550` vs. `0.525`) and twin-pair accuracy (`0.267` vs. `0.217`).
  - On **`DeepSeek-R1-Distill-Llama-8B`**, `composed_rms_0p5x` lifts `TPR` from `0.600` (`base`) to **`0.717` (`+11.7` pp)**, `BA` from `0.417` to **`0.483` (`+6.6` pp)**, and sink recall from `0.350` to **`0.500` (`+15.0` pp)** while cutting `<think>` cap truncations from `38/120` to `33/120`; `composed_rms_1p0x` lifts `RC` from `0.183` (`base`) and `0.133` (`random_rms_1p0x`) to **`0.250` (`+6.7` pp over `base`, `+11.7` pp over `random_rms_1p0x`)** and raises `pair` accuracy from `0.067` (`base`) and `0.083` (`random_rms_1p0x`) to **`0.167` (`+10.0` pp over `base`, `+8.4` pp over `random_rms_1p0x`)**.
  - On **`Qwen3-32B`**, `composed_rms_1p0x` achieves the highest patched-twin specificity (`TNR = 0.550`, `+5.0` pp over `base` `0.500`, `+13.3` pp over `random_rms_1p0x` `0.417`), highest `<think>` completion rate (`89/120` parsed, `cap = 35` vs. `42` on `random_rms_1p0x`), and beats `random_rms_1p0x` on `BA` (`0.642` vs. `0.592`, `+5.0` pp) and `pair` (`0.383` vs. `0.333`, `+5.0` pp), while at `0.5x` `random_rms_0p5x` (`BA = 0.667, RC = 0.683`) exceeds `composed_rms_0p5x` (`BA = 0.567, RC = 0.633`, unadjusted McNemar `p = 0.0169`, Holm `p = 0.2197`).
  - On **`Qwen3-8B`**, where `54%` to `62%` of generations hit the 1,536-token `<think>` cap, `composed_rms_1p0x` beats `random_rms_1p0x` across `BA` (`0.367` vs. `0.342`), `TPR` (`0.450` vs. `0.433`), `TNR` (`0.283` vs. `0.250`), `RC` (`0.317` vs. `0.283`), `pair` (`0.150` vs. `0.100`), and completed `<think>` chains (`55/120` vs. `48/120`).
- **Scope, pair-resampled intervals, and multiple-comparison boundary.** Six of the nine models (`Qwen3.5-0.8B`, `SmolLM2-1.7B`, `Mistral-7B-v0.3`, `Qwen2.5-7B`, `Llama-3.1-8B`, `Qwen3-32B`) use the heuristic writer layer `round(0.67 * n_layers)` recorded in `asan120-receipt.json` rather than a pre-pinned writer from `domain-uplift-models.json`. Against `base`, the retained 95% pair-resampled bootstrap intervals (`asan120_results.json`) for `composed_rms_1p0x` strict root-cause recall change are `+10.0` pp (`[0.0, +21.67]`) on `Qwen3-14B` and `+6.67` pp (`[-5.0, +18.33]`) on `R1-Llama-8B`, while `Qwen3-32B` balanced accuracy changes by `0.0` pp (`[-9.17, +9.17]`). Across all 117 paired exact two-sided McNemar tests on verdict correctness (`9` models $\times$ `13` comparisons per model, `n = 120` per test), no single-model comparison survives within-model or global Holm-Bonferroni correction at $\alpha = 0.05$ (`min within-model p_Holm = 0.2197` across 13 tests on `Qwen3-32B`; `global 117-test p_Holm = 1.0`), and across the 9-model sign tests (`calibrated_1p0x` minus `random_1p0x`: `+3/-2/0:4`, `p = 1.000`; `composed_rms_0p5x` minus `random_rms_0p5x`: `+3/-3/0:3`, `p = 1.000`; `composed_rms_1p0x` minus `random_rms_1p0x`: `+6/-1/0:2`, `p = 0.125`), the `1.0x` RMS directional advantage is consistent across the reasoning cohort (`4/4`) but below the 9-model exact sign-test significance threshold. Pairwise surface heuristics that see both twins simultaneously (`longer snippet is safe`: `0.8583`; `more lines`: `0.7500`; `more conditionals`: `0.6750`) drop to near-chance item-level safe-label AUCs (`0.5735`, `0.5415`, and `0.5578`) when evaluated one snippet at a time under the model task format.

**Table 4C: ASAN-120 Descriptive Breakdown Across All 12 CWE Families (Vulnerable Detections / Patched-Benign Clearances Out of 45 = 9 Models $\times$ 5 Pairs per Family)**

| CWE Family | `Base` (`Vuln / Safe`) | `Composed 1.0x` (`Vuln / Safe`) | `Random 1.0x` (`Vuln / Safe`) |
|---|---:|---:|---:|
| `F01`: Heap buffer overflow | `30 / 22` | `30 / 20` | `29 / 20` |
| `F02`: Stack buffer overflow | `31 / 17` | `29 / 18` | `28 / 22` |
| `F03`: Use after free | `22 / 19` | `21 / 19` | `22 / 18` |
| `F04`: Integer overflow -> allocation | `24 / 22` | `27 / 23` | `26 / 20` |
| `F05`: Signed/unsigned conversion | `26 / 15` | `27 / 17` | `23 / 15` |
| `F06`: Off-by-one boundary | `24 / 23` | `25 / 26` | `21 / 22` |
| `F07`: Null pointer dereference | `22 / 25` | `21 / 20` | `25 / 21` |
| `F08`: Double free | `19 / 22` | `23 / 20` | `22 / 18` |
| `F09`: Uninitialized memory read | `16 / 21` | `17 / 20` | `18 / 20` |
| `F10`: Release-elided `assert` check | `25 / 22` | `25 / 24` | `26 / 23` |
| `F11`: Type confusion / bad cast | `21 / 24` | `21 / 23` | `19 / 23` |
| `F12`: Unbounded recursion | `27 / 17` | `29 / 16` | `29 / 13` |

(sec-paper-dose-response)=
### 4.6 Focused Dose-Response Case Study: Qwen3.5-4B (Regime B)

Alongside the multi-model sweeps, this single-model case study isolates how changing the loop-edit dose affects completion and attribution. This test gives the same 30 crash traces to each edited model with a 1,536-token limit. “Closed” means the model emitted `</think>` within that limit. The other columns score the final answer's bug class, culprit frame, and root cause separately. A completed answer can still be wrong. The dose changes only `B1` in the `a05` and `a15` arms. Strict scoring requires a committed answer after `</think>`. In the reported copy-the-input control, a whole-output keyword matcher scored the raw prompt `28/30`, while strict scoring gave it `0/30`; merely repeating a function name is not correct attribution.

- Model: Qwen3.5-4B; 30 prompts per arm; 1,536-token cap.
- Arms: **`base`** / **`rand1`** / **`rand2`** (equal-norm random); **`g030`** (uncalibrated $\gamma = 0.30$ ported from `0.8B`); **`a05`** (`B1` at $0.5\times$; `B2` and `B3` held at $1.0\times$); **`b1only`** ($\alpha = 1.0\times$, loop edit only); **`cal738`** ($\alpha = 1.0\times$, composed `B1+B2+B3`); **`a15`** (`B1` at $1.5\times$; `B2` and `B3` held at $1.0\times$).
- Closed means an emitted closing think tag; class, frame, and root cause require a committed final answer.

**Dose and completion**

| Arm | B1 shift | Closed /30 | Mean tokens |
|---|---:|---:|---:|
| **base / rand1 / rand2** | n/a | 0/30 | 1,536 |
| **g030** | n/a | 0/30 | 1,536 |
| **a05** | -3.691 | 0/30 | 1,536 |
| **b1only** | -7.382 | 17/30 (56.7%) | 1,044 |
| **cal738** | -7.382 | **21/30 (70.0%)** | 954 |
| **a15** | **-11.073** | **30/30 (100.0%)** | 789 |

**Strict attribution**

| Arm | Bug class /30 | Culprit frame /30 | Root cause /30 |
|---|---:|---:|---:|
| **base / rand1 / rand2** | 0/30 | 0/30 | 0/30 |
| **g030** | 0/30 | 0/30 | 0/30 |
| **a05** | 0/30 | 0/30 | 0/30 |
| **b1only** | 15/30 (50.0%) | 13/30 (43.3%) | 11/30 (36.7%) |
| **cal738** | **18/30 (60.0%)** | **15/30 (50.0%)** | **14/30 (46.7%)** |
| **a15** | **23/30 (76.7%)** | **23/30 (76.7%)** | **19/30 (63.3%)** |

Exact two-sided McNemar tests against `base`: `cal738` loop closure $p = 9.54 \times 10^{-7}$ (21 vs. 0 discordant pairs); `a15` loop closure $p = 1.86 \times 10^{-9}$ (30 vs. 0 discordant pairs). Comparing `a15` vs. `cal738`: loop closure $p = 0.0039$, culprit frame $p = 0.0078$.

> **Section 4 takeaway:** Each study supports its own claim. Calibration matched the requested shift on all 15 Stage-1 models. Project-Then-Heal kept local leakage at the `float32` floor (`float64` floor `3.18e-17` to `5.07e-17` in the RMS sweep) on every model where gradient projection did not. On 14 models, the calibrated `1.0x` edits left public-benchmark scores within about 2 pooled points of base, as equal-size random edits did, with a pooled `MMLU-Pro` cost of `2.0` points and no significant crash-triage gain. Across 6 models in the RMS-relative dosing sweep (`24,000` evaluations), sizing edits by $\sigma_{\text{writer}}$ (`5.85` to `78.15`) showed that unhealed composed edits match norm-matched random controls, while `ProjectThenHealLinear` (`composed_healed_rms_1p0x`) produced unadjusted in-domain gains on `Qwen2.5-Math-7B-Instruct` (`GSM8K` `87 -> 93`, `p = 0.031`) and `Qwen2.5-Coder-7B-Instruct` (`HumanEval` `36 -> 42`, `p = 0.238`) that do not survive multiple-comparison correction and are not isolated from the LoRA healing pass itself. Finally, on the paired `ASAN-120` vulnerability and patched-benign twin benchmark (`9` models $\times$ `7` arms, `7,560` evaluations), benign controls expose trivial one-sided predictions (`BA ≈ 0.50`) across all 5 non-reasoning models, whereas on the 4 `<think>` reasoning models `composed_rms_1p0x` beats `random_rms_1p0x` `4/4` on balanced accuracy (`6W / 1L / 2T` overall, sign-test `p = 0.125`) and lifts strict root-cause recall (`RC`) by `+6.7` to `+10.0` percentage points over `base` on `DeepSeek-R1-Distill-Llama-8B` (`0.183 -> 0.250`) and `Qwen3-14B` (`0.450 -> 0.550`).

---

(sec-paper-static-auditing)=
## 5. Static Weight Auditing

Sections 1 to 4 edit weights. This section uses the same readers to inspect weights someone else may have edited, without running the model. Every number below comes from a receipt whose inputs were hashed before and after each run (`pre_sha256 == post_sha256`). That is a non-adversarial host-filesystem bracket, not descriptor-bound execution.

### 5.1 Spectral Trojan Auditing and the Vocabulary Phase Transition

`secant audit-trojan` (`secant read --detect-anomalies --marchenko-pastur`) implements a **two-stage static spectral detector**:
1. **Stage 1 (Iterated Outlier Singular-Value Peeling & Row/Column Spike Detection):** Iteratively peels outlier singular values ($\sigma_i > \sigma_{MP}$) from the total Frobenius energy ($\lVert W\rVert_F^2 - \sum_{i=1}^k \sigma_i^2$, the exact Frobenius energy of the rank-$k$ ablated matrix $W - \sum_{i=1}^k \sigma_i u_i v_i^\top$) until the residual Marchenko-Pastur bulk edge $\sigma_{MP} = \sigma_{\text{entry}}\sqrt{d_{\text{out}}}(1 + \sqrt{d_{\text{in}}/d_{\text{out}}})$ reaches a fixed point (`mp.rs`), and checks row/column norm spikes (`zero_rows`, `collapsed_rows`, `spike_cols`).
2. **Stage 2 (Cross-Layer Sibling Modified Z-Score & Vocabulary Token Projection):** Because natural trained weight matrices are heavy-tailed and first/last transformer blocks (`layers.0` and the final block) interface directly with token embeddings, Stage 2 compares each tensor's peeled $\sigma_1/\sigma_{MP}$ ratio against its architectural layer family (`blk.*.ffn_down.weight`, `model.layers.*.mlp.down_proj.weight`) via the Iglewicz-Hoaglin modified z-score (`sibling_outlier` when $z > 3.5$ in a family of $\ge 4$ layers), while `--vocab-tensor` projects the leading left singular vector $u_1$ onto the unembedding matrix ($E \cdot u_1$) to surface its top promoted vocabulary tokens.

Trained weight matrices are non-i.i.d. and already sit well above the Marchenko-Pastur bulk edge on unedited checkpoints, so $\sigma_1/\sigma_{MP}$ serves as a relative ranking feature paired with the cross-layer sibling modified-Z test and `--vocab-tensor` projection rather than a standalone i.i.d. threshold. To measure when these signals fire, a rank-1 edit $\Delta W = \theta\,\hat e_T\, c^\top$ was planted in the layer-12 MLP down projection of `SmolLM2-1.7B-Instruct` (SafeTensors, `family_size = 24`) and `Qwen3.5-4B` (BF16 GGUF, where `blk.*.ffn_down.weight` has `family_size = 33` because `llama.cpp` exports the 32 decoder blocks `blk.0..31` plus `blk.32`, the Multi-Token Prediction / MTP `blk.32.nextn.*` head block). Here $\hat e_T$ is the unit embedding row of a trigger token, and $c$ spreads equal weight over 16 evenly spaced input columns. The edit was planted at 12 relative Frobenius doses $g = \lVert\Delta W\rVert_F/\lVert W\rVert_F$ from `0` to `16%`.

A rank-1 edit's only singular value is $\sigma_{\text{trojan}} = \lVert\Delta W\rVert_2 = \lVert\Delta W\rVert_F = g\lVert W\rVert_F$. The **standalone singular-value equality reference (nominal crossover)** where the planted perturbation's spectral norm equals the unedited matrix's leading singular value $\sigma_1(W)$ is:

$$
g^* = \frac{\sigma_1(W)}{\lVert W\rVert_F},
$$

which is `8.65%` for SmolLM2 and `6.57%` for Qwen. Because $\Delta W$ and $W$ are not orthogonal in general, the exact dose at which the top singular vector of $W + \Delta W$ aligns with the trigger direction depends on the overlap between $\Delta W$ and the leading singular triplets of $W$; empirically, the discrete post-hoc dose sweep (`4%`, `6%`, `8%`, `10%`, `12%`) brackets $g^*$ on both models.

```{figure} assets/trojan-detection-phase-transition.svg
:name: fig-paper-trojan-phase
:alt: Planted rank-1 trojan detection against relative Frobenius dose on SmolLM2-1.7B and Qwen3.5-4B, showing family statistics, sigma_2, vocabulary alignment and trigger-token rank
:width: 100%

Detection signals against dose for both models. Doses `0` to `4%` were pre-specified, and `6` to `16%` were added after the pre-specified range showed nothing, so each threshold is post-hoc and an upper bound. The dashed verticals mark the nominal singular-value equality reference $g^* = \sigma_1(W)/\lVert W\rVert_F$.
```

| Signal | First tested dose, SmolLM2 | First tested dose, Qwen3.5-4B | Needs the base checkpoint? |
|---|---|---|---|
| Any automatic signal, pre-specified 0.10 to 4.00% | none | none | |
| σ₂ rises (25.31 to 30.10; 2.112 to 2.678) | 6% | 6% | yes: audit-trojan ranks σ₁ / σMP only |
| Trigger token in the promoted top 16 of u₁ | 8% (rank 2, cos(u, u₁) = 0.569) | 8% (rank 1, cos(u, u₁) = 0.991) | no, but a person has to read the column |
| Edited layer is family rank 1 by σ₁ / σMP | 10% | 10% | no |
| Automatic sibling\_outlier (z &gt; 3.5) | 12% (z = 5.01) | 12% (z = 3.58) | no |
| Sibling outliers on the unedited base | 0 of 24 | 0 of 33 | |

The vocabulary jump happens across the sampled doses bracketing $g^*$: $\cos(u, u_1)$ goes from `0.168` to `0.991` between `6%` and `8%` on Qwen ($g^* = 6.57\%$), and from `0.569` to `0.984` between `8%` and `10%` on SmolLM2 ($g^* = 8.65\%$). Well below $g^*$ the planted direction is nearly absent from $u_1$ ($\cos < 0.17$ at `6%` on both models). Close to $g^*$ the two directions mix (SmolLM2 at `8%`: $\sigma_{\text{trojan}}/\sigma_1 = 0.92$, $\cos = 0.569$). Above $g^*$, $u_1$ aligns with the planted direction ($\cos > 0.98$). The automatic sibling test needs a further `39%` to `83%` of $\sigma_1$ beyond that ($\sigma_{\text{trojan}}/\sigma_1$ = `1.39` and `1.83` at the first flag). Excluding the MTP head block `blk.32` (`32`-tensor decoder family `blk.0..31`) on `Qwen3.5-4B` leaves the `12%` sibling-outlier crossover dose unchanged while raising `blk.12`'s modified z-score from `2.43` to `2.85` at `10%`, from `3.58` to `4.21` at `12%`, and from `5.92` to `6.94` at `16%` (`0 / 32` false positives on the unedited base). Full 12-dose tables: [payload checks, Section 5.1](#sec-trojan-dose-curve).

Scope: one layer, one token and one column placement per model, static weights only. A smaller edit spread over many layers, or a rank-1 edit placed in a layer with a lower baseline ratio, may behave differently, and neither was tested. The curve says when this reader sees a rank-1 edit. It says nothing about whether the edit changes model output.

### 5.2 Quantized Scale-Header Sanitization and Why Naive Subnormal Rejection Fails

In GGUF quantized blocks (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`), one or two `f16` header fields (`d`, `dmin`, or `m`) scale 32 or 256 weights. By default, `secant sanitize` checks every `f16` scale header across all 10 standard GGUF quantized formats for NaN and Inf. `--clamp-extreme` and `--flush-subnormal` add stricter, opt-in diagnostics. Across our 10-checkpoint production GGUF and SafeTensors verification suite (`186,870,528` quantized blocks across `912` quantized tensors and `3,258,570,112` float elements across the 10 files, including mixed `Q4_K_M` shards containing `Q4_K` + `Q6_K`, `TinyLlama-Q2_K` containing `Q2_K` + `Q3_K` + `Q6_K`, `stories15M-Q4_0`, and `Qwen3.5-4B` `Q8_0`/`Q4_K`), planted NaN and Inf scales across all 15 `f16` header slots were found with `100%` recall at the exact block, field, and file offset, and `--clamp-nonfinite` produced copies that rescanned with `0` findings. The diagnostic sweep also measured the benign background: `272,730` subnormal scales in `Q8_0` (`0.20%` of `135,168,000` blocks) and `8,992,791` of `33,792,000` header scale fields (`d`, `dmin`) across `16,896,000` `Q4_K` superblocks (`0.53` per super-block), with `0` observed false positives across the 10 tested production GGUF and SafeTensors checkpoints (`10 / 10` `status: "clean"`).

A naive security filter that rejects any floating-point scale where `fpclassify(d) != FP_NORMAL` (or `abs(d) < 1e-4`) therefore false-positives on every real production `.gguf` checkpoint. In `Q4_K` 256-weight super-blocks, eight sub-blocks of 32 weights carry 6-bit integer scales $s_j \in [0, 63]$ and mins $m_j \in [0, 63]$, while the super-block header stores `d` and `dmin` as IEEE-754 binary16 (`ggml_half`, where the smallest positive normal number is $2^{-14} \approx 6.1035 \times 10^{-5}$ at bit pattern `0x0400`, and positive subnormals span `0x0001` to `0x03FF`). Because the sub-block scale is divided by `63` to form the super-block scale `d` (scaling a 4-bit weight's maximum magnitude $\max |x_i|$ by $15 \times 63 = 945$), any sub-block whose maximum weight magnitude sits below $945 \times 6.1035 \times 10^{-5} \approx 0.0577$ (or a sub-block scale near `2.8e-03` before dividing by `63`) produces a legitimate FP16 subnormal `d` value (for example, across dense `blk.*.ffn_down.weight` and `token_embd.weight` in `Qwen3.5-4B` `Q4_K`, or `d = 4.55e-05 < 6.1035e-05` in `blk.0.ffn_down_exps` row 0 of `llama.cpp`'s MoE test fixture), while zero-padded blocks carry `d = 0.0` (`0x0000`). Because GGUF weight dequantization **multiplies** by `d` and `dmin` ($w = d \cdot s_j \cdot q - d_{\min} \cdot m_j$) and never divides by them at inference time, zero and subnormal FP16 block scales cannot trigger division by zero or float-to-integer overflow.

Under `secant sanitize`'s default `Rule B3-2A` policy, legitimate zero (`0x0000`) and FP16 subnormal (`0x0001..0x03FF`) block scales are permitted across all 10 supported GGUF quantized formats (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`). Only `NaN` (`quant_scale_nan`) and `±Inf` (`quant_scale_inf`) fail the default check. `--clamp-nonfinite` zeroes the complete affected block, while `--flush-subnormal` and `--clamp-extreme` are opt-in stricter policies. Across the receipted 10-checkpoint verification corpus (`summary.json` SHA-256 `399143323a1784f2b0b5f0e3ac99a55df18bb43cb843d20d921d2eaf358a3ccd`, covering `2,194` total tensors, `912` quantized tensors, `186,870,528` quantized blocks, `272,730` subnormal `Q8_0` scales, and `8,992,791` subnormal `Q4_K` header scales), default Rule B3-2A passed all `10 / 10` checkpoints with `0` false positives; an earlier unreceipted dogfood run on `Qwen2.5-1.5B` and `Llama-3.2-3B` `Q4_K_M` and `Q8_0` files similarly reported `2,723/2,723` tensors clean across `16,139` subnormal scales (`4.55e-05`). Files containing unreviewed quantized types (`Q8_K`, `IQ*`, `TQ*`) fail closed with exit `2` (`status: "unknown"`). Details: [payload checks, quantized scale headers](payload-security.md#sec-quant-sanitize-controls).

### 5.3 A Spectral and Vocabulary Atlas Over a 46-File Local Inventory (2,619 Analyzed Weight Matrices: 14 Full Reads and 16 Scoped Variants)

`secant read` was run with singular values, column statistics, anomaly detection and vocabulary projection over a 46-file local inventory (`2,619` analyzed weight matrices in total: `2,574` across `14` checkpoints read in full and `45` across `16` edited variants read only where they differ from base, plus `16` vocab-only files that `read` refuses). That gave 429 receipts covering `2,619` tensors, `0` of them non-finite. `323` of `5,101` singular triplets (`6.3%`) hit the iteration cap. The SafeTensors and GGUF copies of `Qwen3.5-4B` agree bit-for-bit on $\sigma_1..\sigma_3$ for all 32 MLP down projections. The unedited layer-12 $\sigma_1$ values in Section 5.1 match this atlas exactly. Separately, the native `secant scan-sinks` port was checked byte-for-byte against its Go oracle across the synthetic fixture (`13` hits) and clean `git archive HEAD` exports of seven pinned external inference and ML codebases (`14,512` pattern hits across `13,675` files in total: `llama.cpp` `ggml/src` `@ 223373742b` `2,498`, `whisper.cpp` `@ 95ea8f9` `2,725`, `vLLM` `csrc` `@ ee0da84ab9e` `692`, `XNNPACK` `@ 0d17d45` `8,070`, `tract` `@ 5b7ef4f8c` `239`, `sentencepiece` `@ bcd4d8c` `15` with `1` file skipped `> 4 MiB` and `incomplete: true`, and `OpenCV` `modules/dnn` `@ 51ba88b` `273`; or `14,734` hits when local working checkouts retain gitignored `build/` directories in `XNNPACK` [`8,171`] and `sentencepiece` [`136`], with both CLIs producing byte-identical masked receipts in both settings). Those hits are untriaged hypotheses for oracle parity, not bugs. See [payload checks, Sections 6 and 7](payload-security.md).

### 5.4 Boundary-3 Numerical-Domain Vulnerabilities, Scanner Blind Spots, and Pre-Load Sanitization

While `safetensors` eliminates arbitrary code execution during deserialization (unlike `pickle`), memory-safe container parsing does not guarantee that raw floating-point, quantized scale, or integer routing payloads are safe for downstream C, C++, or CUDA inference consumers. Secant organizes checkpoint auditing into three distinct trust boundaries:

- **Boundary 1 (Container Framing & Deserialization):** Header length, JSON schema, absence of executable opcodes, and monotonically non-overlapping byte-slice intervals (`[begin, end)`).
- **Boundary 2 (Architectural & Graph Metadata):** Dimension counts, head ratios, and shape-to-byte allocation invariants against `config.json`.
- **Boundary 3 (Numerical-Domain & Value-Level Invariants):** IEEE-754 non-finite floats (`qNaN`, `sNaN`, `±Inf`), extreme finite floats ($\ge 2^{31}$), `Q8_0`/`Q4_K` `f16` block scale headers (`d`, `dmin`), static divisor activation scales (`input_scale`, `k_scale`, `v_scale`), integer routing/permutation index tensors (`g_idx`, `expert_map`, `full_row_indices`), and spectral rank-1 outliers inside otherwise validly framed tensors.

The 10-checkpoint production verification corpus (`summary.json` SHA-256 `399143323a1784f2b0b5f0e3ac99a55df18bb43cb843d20d921d2eaf358a3ccd`) measured the false-positive side of this policy across `2,194` total tensors (`912` quantized tensors, `186,870,528` quantized blocks, `272,730` subnormal `Q8_0` scales, and `8,992,791` subnormal `Q4_K` header scales; an earlier unreceipted dogfood run on `Qwen2.5-1.5B` and `Llama-3.2-3B` `Q4_K_M` and `Q8_0` files also reported `16,139` legitimate FP16 subnormal scale fields across `2,723` tensors down to `4.55e-05 < 6.1035e-05`). Default Rule B3-2A accepted every checkpoint (`10 / 10` clean) and every subnormal scale. Planted controls gave `100%` detection recall for non-finite block headers, divisor scales below `--divisor-floor` (`1e-12f`), zero `rope_freqs`, and negative or upper-bound values in `g_idx`, `expert_map`, `full_row_indices`, `hash_indices_table`, `expert_ids`, `selected_experts`, `topk_ids`, and `router_indices`. `--clamp-nonfinite` repaired the non-finite block headers and non-finite divisor scales; finite non-positive/below-floor divisor scales, zero `rope_freqs`, and invalid routing or index IDs were refused (and unbounded index tensors failed closed with `UNKNOWN`, exit `2`) because no format-independent replacement is safe.

A separate source audit shows why the pre-load rules matter to consumers. Across inference source trees, `secant scan-sinks` produced `185` review leads across `B3-SINK-01..06` in vLLM, covering unchecked CUDA comparison-sentinel (`COMPARE_SENTINEL_COLLAPSE`) and routing/permutation index (`UNVALIDATED_ROUTING_OR_INDEX_TENSOR`) patterns, alongside historical `g_idx` gather indexing in `csrc/quantization/gptq/q_gemm.cu` (later removed in upstream commit `2c9d68f80`). Across our broader Boundary-3 audit of inference stacks, three general post-deserialization failure classes illustrate how structurally valid containers and companion configs cross Boundary 3 into runtime or native indexing sinks:

1. **Companion-config control token ID -> out-of-bounds logits indexing (`Rule S-0`):** When a companion configuration file specifies a stop or control token ID outside `[0, vocab_size)` while `config.json` and the `.safetensors` weights remain structurally valid, an unguarded stop-token suppression path can index a `[batch_size, vocab_size]` logits buffer without bounds validation.
2. **Container header layer-count relation -> unsigned integer wrap (`Rule S-0`):** When a container header specifies a dense-prefix block count larger than the total block count, an unguarded 32-bit unsigned subtraction wraps around to `UINT32_MAX` (`0xFFFFFFFF`).
3. **Integer MoE routing table -> out-of-bounds host or device array indexing (`Rule B3-3`):** When a structurally valid checkpoint stores an integer token-to-expert routing or permutation tensor containing an expert ID outside `[-1, num_experts)`, an unguarded loader or MoE dispatch loop that uses the stored integer directly as a table or buffer offset reads or writes out of bounds on the host or device heap. `secant sanitize --num-experts <E>` blocks out-of-range and negative routing IDs pre-load via `Rule B3-3` (`routing_id_out_of_range` / `routing_id_negative`), and `secant probe --class all --format both --out-dir <dir> --receipt <json> --attempt-id <id>` deterministically generates self-verified synthetic witness bundles across all four Boundary-3 failure classes (`class1..class4`) out of the box.

Core TensorFlow provides a related scalar and Eigen conversion target. At commit `0e2be16cc08c226b65af3ff7a8ba560b953b60b4`, `FloatToQuantizedUnclamped<T>` in `tensorflow/core/kernels/quantization_utils.h:46-61` divides by `range_max - range_min` before converting rounded values to `int64_t`; the same file has explicit equal-range cases at lines 50-52 and 215-220. The active Boundary-3 harness targets NaN and infinity conversions and checks that `range_min == range_max` remains safe across both paths. Runtime results from that campaign are outside this paper's retained Secant test set and are `NOT_RUN` here.

#### Proof of the `safetensors` Parser Boundary and Static Scanner Blind Spot

In `huggingface/safetensors` (`safetensors/src/tensor.rs`), calling `SafeTensors::deserialize(buffer)` invokes `SafeTensors::read_metadata(buffer)` followed by `Metadata::validate(&self)`. `Metadata::validate()` checks **only five structural invariants**:

1. **Offset contiguity:** Each tensor's `data_offsets` `(s, e)` must start at the previous tensor's end (`s == start`) and satisfy `e >= s` (`SafeTensorError::InvalidOffset`).
2. **Shape-product overflow:** Multiplying non-zero dimensions in `info.shape` by `info.dtype.bitsize()` via `try_fold` must not overflow `usize` (`SafeTensorError::ValidationOverflow`).
3. **Bit-count byte alignment:** The total bit count `nbits` must be byte-aligned (`nbits % 8 == 0`, raising `SafeTensorError::MisalignedSlice` otherwise).
4. **Byte-span equality:** The byte count `nbits / 8` must equal the offset span `e - s` (`SafeTensorError::TensorInvalidInfo`).
5. **Final buffer length:** The final offset `buffer_end + 8 + n` must equal the buffer length.

Once `validate()` returns `Ok`, `deserialize()` returns a borrowed slice `&buffer[8 + n..]` without inspecting a single byte of payload data. Existing static model scanners share the same blind spot because they inspect only Boundary-1 callables or Boundary-2 headers: `ModelScan 0.8.6` (Protect AI) scans Pickle, H5, Keras, and `SavedModel` archives for unsafe opcodes (`REDUCE`, `GLOBAL`) and skips `.gguf` files and `.safetensors` numerical payloads entirely (`arXiv:2608.27424`); `Fickling 0.1.6+` (Trail of Bits) and `ModelAudit 0.2` analyze Python pickle ASTs and explicitly recommend `.safetensors` as an inherently safe format without parsing `.safetensors` or `.gguf` tensor values; and on the Hugging Face Hub, `picklescan` scans Python pickle opcodes, `ClamAV` scans for known malware signatures, and the Hub's model-viewer metadata parser reads `.safetensors` and `.gguf` headers to render tensor shapes and chat templates without scanning binary tensor payloads.

#### Four Boundary-3 Failure Classes and Cross-ISA Hardware Divergence (`x86-64` vs. `ARM64` vs. `CUDA PTX`)

At Boundary 3, valid container payloads trigger native memory corruption or process aborts through four failure classes whose behavior diverges sharply across `x86-64`, `ARM64`, and `NVIDIA CUDA PTX`:

1. **Class I: IEEE-754 Comparison-Sentinel Exhaustion (Prior Art: Fornaro, Aug 2026).** On August 5, 2026, Vincenzo Fornaro ([`JustVugg/colibri`, `GHSA-5xpg-vw35-2687`](https://github.com/JustVugg/colibri/security/advisories/GHSA-5xpg-vw35-2687), *"NaN router logits leave MoE top-k selection at best=-1, causing negative-index heap OOB read/write and negative-offset pread"*) disclosed that injecting `NaN` weights into an MoE router/gate tensor in `.safetensors` (`inkling`, `kimi_k3`, and `olmoe`) causes `if (val > best_val)` inside an argmax loop initialized to `best = -1, best_val = -FLT_MAX` to evaluate `false` on every iteration, returning `best = -1` and indexing `expert_offsets[-1]` out of bounds on the heap (and passing a negative offset to `pread`). Similarly, when top-`k` selection initializes the winning expert ID per pass to `0` (as in `vllm#39288` / `#39077` / `#39244`), `NaN > max_val` is always `false`, emitting duplicate expert `0` assignments (`[0, 0, ...]`) that crash downstream MoE kernels on uninitialized row pointers.
2. **Class II: Hardware ISA-Divergent Float-to-Signed-Integer Cast Memory Corruption (`CVTTSS2SI`/`VCVTTPS2DQ` vs. `FCVTZS` vs. `cvt.rzi.s32.f32`).** In C/C++ inference kernels that cast floating-point scores, coordinate offsets, or quantized values directly to signed integers via `static_cast<int32_t>(val)` before array indexing, C++ (`[conv.fpint]`) leaves out-of-range and `NaN` conversions undefined, and hardware ISAs diverge:
   - **Sub-Primitive 2A (`qNaN` `0x7FC00000`):** On **x86-64**, scalar `CVTTSS2SI` and vector `VCVTTPS2DQ` emit the **Integer Indefinite value `0x80000000` (`INT32_MIN = -2147483648`)**, causing `table[idx]` to dereference `base - 8 GiB` and fault immediately if unguarded or guarded only by an upper bound (`idx >= N`). On **ARM64** (`AArch64` `FCVTZS`, used on Apple Silicon and AWS Graviton) and **NVIDIA CUDA PTX** (`cvt.rzi.s32.f32` / `cvt.rni.s32.f32`), hardware converts `NaN -> 0`, passing two-sided `0 <= idx && idx < N` checks and benignly reading `table[0]`.
   - **Sub-Primitive 2B (Positive Overflow `+1.0e38` `0x7E967699` / `+Inf`):** On **ARM64** (`FCVTZS`) and **CUDA PTX** (`cvt.rzi.s32.f32`), positive overflow saturates to **`+2147483647` (`INT32_MAX`)**, which is caught by a one-sided upper clamp `if (idx >= N) idx = N - 1;`. On production **x86-64** servers, `CVTTSS2SI` / `VCVTTPS2DQ` instead emits **`0x80000000` (`INT32_MIN = -2147483648`)** for all out-of-range floats outside `[-2147483648.0f, 2147483520.0f]`, flipping the sign from positive to negative, bypassing upper-only clamps, and indexing at `-2147483648`. Even on CUDA PTX, narrowing a saturated 32-bit `INT32_MAX` (`0x7FFFFFFF`) directly to `int8_t` before clamping (`(int8_t)__float2int_rn(x / scale)`) extracts the low byte `0xFF` (`-1`).
3. **Class III: Strict Weak Ordering, GPU Radix Twiddle, and Cumulative-Sum Collapse.** In CPU `std::sort`, a comparator that violates irreflexivity on `NaN` (`!(a < b)` or `<=` on $N \ge 17$ elements, as in `openxla/xla#52412`) causes GNU `libstdc++` `__unguarded_partition` to increment its pointer past the end of the buffer with zero bounds checks, while in cumulative-sum token samplers (`llama_sampler_dist_apply`, `llama.cpp#29686`), a single `NaN` probability makes `sum_run >= sum_tgt` evaluate to `false` on every token and triggers `assert(found)`. On the GPU, `NVIDIA CUB` `cub::BlockRadixSort::SortDescending` (`Traits<float>::TwiddleIn`) does not execute floating-point comparisons: it reinterprets each `float` as `uint32_t` and flips the sign bit (`u ^ 0x80000000`) when the sign bit is `0` or all 32 bits (`u ^ 0xFFFFFFFF`) when the sign bit is `1`. That bitwise twiddle maps positive `qNaN` (`0x7FC00000` -> `0xFFC00000`) strictly **above `+Inf`** (`0x7F800000` -> `0xFF800000`), placing `+qNaN` at index `0` as the #1 highest-scoring winner in GPU top-`k` sampling and MoE routing.
4. **Class IV: Unchecked Integer Routing, Permutation, and Dispatch Tables.** Integer tensors (`I32`, `I16`, `U8`) stored inside structurally valid checkpoints act directly as gather/scatter array indices or function-pointer selectors, including GPTQ group-index permutation arrays `g_idx` (`CVE-2026-86288` in GPTQModel, `vllm#48470`, `exllamav2#817`), ExLlamaV3 function-pointer table dispatch (`CVE-2026-84286` / `GHSA-mw4q-pwhm-35f7`), SqueezeLLM `full_row_indices` (`#77`), and LiteRT negative subgraph output indices (`LiteRT#8609`).

We verified both Class-II sub-primitives empirically by compiling and running the identical C++ harness (archived experiment receipt identifier `cross-isa-fpint-conformance-20261106/fpint_harness.cpp`, `clang++ -O2`) on a local Apple M-series ARM64 workstation (`fcvtzs w0, s0`) and a Linux x86-64 workstation (AMD Ryzen AI Max+ 395, `cvttss2si %xmm0, %eax`) against the same 824-byte `class2-scale-bias-nan-extreme.safetensors` fixture (archived receipt identifier `cross-isa-fpint-conformance-20261106/evidence/class2-scale-bias-nan-extreme.safetensors`, `SHA-256: f0797fa931211b0a2be82df14bdd1876c4e11c5a16f37076b50afc9b7364448a`) and its `secant sanitize --clamp-nonfinite --max-abs 10000`-remediated counterpart (archived receipt identifier `cross-isa-fpint-conformance-20261106/evidence/clean.safetensors`, `SHA-256: 48104b2749e36d45e94a9bf85bbabea1d21f231d287fd6dd8efc2a1a38ee7d10`):

| Fixture Cell (scale tensor) | Raw Float32 Bits & Value | ARM64 (Apple Silicon, fcvtzs) int32\_t Index | x86-64 (AMD Zen 5, cvttss2si) int32\_t Index | Post-secant sanitize (clean.safetensors, Both ISAs) |
|---|---|---|---|---|
| cell=0 (Sub-Primitive 2A) | 0x7fc00000 (qNaN) | 0 | -2147483648 | 0 |
| cell=1 (Sub-Primitive 2B) | 0x7e967699 (+1.0e38) | +2147483647 | -2147483648 | 10000 |
| cell=2 (Negative overflow) | 0xfe967699 (-1.0e38) | -2147483648 | -2147483648 | -10000 |

While [Hanzo AI `HIP-0113` (Section 4.5, 2026)](https://github.com/hanzo-apps/HIPs/blob/main/HIPs/hip-0113-cognitive-sidecar-and-hanzo-engine-provider-runtime.md) documented that `FCVTZS` (`0`) vs. `CVTTSD2SI` (`INT_MIN`) causes cross-architecture consensus divergence when quantizing non-finite floats into `int8` vectors, Secant connects both the `qNaN` (`0` vs. `INT_MIN`) and positive-overflow (`INT_MAX` vs. `INT_MIN`) hardware cast asymmetries directly to **architecture-divergent C/C++ memory-corruption exploit primitives** and pairs static payload sanitization (`secant sanitize`) with C/C++ source sink scanning (`secant scan-sinks`).

#### Zero-False-Positive Pre-Load Sanitization Standard (`Rule S-0` through `Rule B3-3`)

To block Boundary-3 Classes I through IV prior to model load while achieving `0` observed false positives across the 10 tested production GGUF and SafeTensors checkpoints (with rule-level B3-2B and B3-3 support for GPTQ, AWQ, and FP8 validated on synthetic fixtures), pre-load sanitization separates multiplicative weight-block scales from divisor activation scales and integer routing indices:

| Rule ID | Target Tensor / Block Scope | Exact Mathematical & Bitwise Invariant | Failure Classes Blocked | Why Legitimate Checkpoints Pass |
|---|---|---|---|---|
| **Rule S-0** | All .safetensors and .gguf headers + config.json | Checked 64-bit span arithmetic; tensor shapes match config.json; numel == 1 on scalar loads; 1 &lt;= expert\_used\_count &lt;= num\_experts | Boundary-2 Overflows & Class III/IV partial\_sort OOB | Well-formed checkpoints have non-wrapping spans matching config.json |
| **Rule B3-1** | Unquantized float weights, norms, and router biases (F32, BF16, F16, FP8) | Bitwise finiteness check rejecting NaN and ±Inf: F32 (u32 &amp; 0x7F800000) != 0x7F800000, BF16 (u16 &amp; 0x7F80) != 0x7F80, F16 (u16 &amp; 0x7C00) != 0x7C00 | Classes I, II, & III | Trained weights, norms, and e\_score\_correction\_bias are always finite |
| **Rule B3-2A** | Multiplicative GGUF block headers (ggml\_half d, dmin) and W4A16/W8A16 scales | Permit 0.0 (0x0000) and subnormals (0x0001..0x03FF); reject only FP16 exponent 0x1F: (d\_u16 &amp; 0x7C00) != 0x7C00 and (dmin\_u16 &amp; 0x7C00) != 0x7C00 in a strided pass by ggml\_type\_size(type) | Classes I, II, & III | Legitimate Q4\_K subnormals (such as across dense blk.\*.ffn\_down.weight in Qwen3.5-4B Q4\_K, or 4.55e-05 < 6.1035e-05 in blk.0.ffn\_down\_exps row 0 of llama.cpp's MoE fixture) and zero-padded blocks have exponent 0, not 0x1F |
| **Rule B3-2B** | Divisor activation and KV-cache scales (`input_scale`, `k_scale`, `v_scale`, `q_scale`, `y_scale`, `prob_scale`, `*scale_inv`) plus `rope_freqs` | Strictly positive normal floor: `isfinite(s) && s >= --divisor-floor` (default `1e-12f`) and `rope_freqs != 0`; `--clamp-nonfinite` clamps only non-finite divisor scales while leaving finite non-positive/below-floor scales as findings | Class II (x / 0.0f and subnormal divisor overflow to 0x80000000 on x86-64) | Calibrated macroscopic activation ranges are never zero or subnormal |
| **Rule B3-3** | Routing tensors (`tid2eid`, `expert_id*`) and integer index tensors (`g_idx`, `expert_map`, `full_row_indices`, `hash_indices_table`) | Strict two-sided bounds (`[0, bound)`, or `[-1, bound)` for `expert_map`, with sibling-inferred bounds for `g_idx` and `full_row_indices` and fail-closed `UNKNOWN` exit `2` when an index tensor is unbounded) | Class IV (CVE-2026-86288, CVE-2026-84286, vllm#48470) | Every valid quantized group or MoE routing index lies strictly inside its declared range |

> **Section 5 takeaway:** The spectral reader sees a planted rank-1 edit once its relative Frobenius dose reaches the neighborhood of the nominal singular-value equality reference $g^* = \sigma_1(W)/\lVert W\rVert_F$. The vocabulary projection shows it first, and the automatic sibling test shows it later. Below that region, a single-file audit sees nothing in these tests. At Boundary 3, `SafeTensors::deserialize()` (`Metadata::validate()`) and existing static scanners (`ModelScan`, `Fickling`, `ModelAudit`, and Hugging Face Hub `picklescan` / `ClamAV` / header parsers) never inspect numerical payloads, whereas `secant sanitize` enforces the zero-false-positive pre-load standard (`Rule B3-1`, `Rule B3-2A` vs. `Rule B3-2B`, and `Rule B3-3`, with `0` observed false positives across the 10 tested production GGUF and SafeTensors checkpoints), neutralizing ISA-portable `best = -1` argmax bypasses, `NVIDIA CUB` `+qNaN` radix-sort promotion, x86-64 `CVTTSS2SI` (`INT32_MIN`) vs. ARM64 `FCVTZS` (`0` / `INT32_MAX`) cast-divergent memory corruption, and out-of-range Class IV index tables.

---

(related-work-and-limits-of-novelty)=
## 6. Related Work and Limits of Novelty

Orthogonal projection, quantized file editing, subspace-constrained LoRA, and static tensor auditing build upon rapid developments across mechanistic interpretability, parameter-efficient fine-tuning, and ML systems security between 2022 and November 2026:

- **Direct Weight Editing, Null-Space Constraints, and Abliteration Estimators:** [ROME](https://arxiv.org/abs/2202.05262) (Meng et al., NeurIPS 2022), [MEMIT](https://arxiv.org/abs/2210.07229) (Meng et al., ICLR 2023), [LEACE](https://arxiv.org/abs/2306.03819) (Belrose et al., NeurIPS 2023), and [AlphaEdit](https://arxiv.org/abs/2410.02355) (Fang et al., ICLR 2025) formulate rank-1 associative updates, whitened concept erasure, and null-space-projected weight updates. [Takehi, Teoh, Zhang, and Khattab](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, Nov 2, 2026) presented Weight-Space Programming experiments on `Qwen3.5-0.8B`, where an autonomous LLM agent (`gpt-6-astra`, Astra) operated inside a Python kernel under a 10/25/50 forward-pass budget to trace activations and edit model weights. Multi-direction Gram-Schmidt orthogonalization and ridge-regularized projection appear in [G-Abliteration](https://arxiv.org/abs/2512.18901) (Gülmez, Dec 2025), [Norm-Preserving Biprojected Abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration) (Lai, Nov 2025), [Heretic](https://github.com/p-e-w/heretic), [Obliteratus](https://github.com/elder-plinius/OBLITERATUS), [llm-abliteration](https://github.com/jim-plus/llm-abliteration), [model-unfetter](https://github.com/josepha-mayo/model-unfetter), [apostate](https://github.com/heterodoxin/apostate), and [weightless](https://github.com/msuiche/weightless).
- **In-Place Quantized GGUF Editing:** [Kiwimaker](https://huggingface.co/Kiwimaker/DeepSeek-V4.1-Flash-1017-Maya-S-abliterated-GGUF) (`apply_abliteration.py` / `apply_abliteration_v2.py`, Oct 20-21, 2026) implemented in-place `r+` `np.memmap` editing of `Q8_0` GGUF tensors with SHA-256 verification of untouched bytes (while skipping `Q4_K` and dequantizing/re-quantizing full 2-D matrices in `float64` via `gguf-py`), and [SC117](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) (Oct 3, 2026) demonstrated byte-level in-place GGUF tensor transplantation and scale-preserving `Q2_0` re-encoding.
- **Subspace-Constrained LoRA and AdamW Preconditioner Leakage:** [Safe LoRA](https://arxiv.org/abs/2405.16833) (Hsu et al., NeurIPS 2024) and [SaLoRA](https://arxiv.org/abs/2501.01765) (Li et al., ICLR 2025) project LoRA updates onto a safety-aligned subspace; [LoRA-Null](https://arxiv.org/abs/2503.02659) (Tang et al., AAAI 2026) initializes adapters in the activation null space; [OPLoRA](https://arxiv.org/abs/2510.13003) (Xiong & Xie, AAAI 2026) applies forward-pass left/right orthogonal projection ($\Delta W = P_L B A P_R$) against top singular directions in continual learning; [PoLoRA](https://arxiv.org/abs/2607.17620) (Ghosh et al., July 20, 2026) formulates a preconditioned orthogonalized LoRA optimizer; and [OGPSA](https://arxiv.org/abs/2602.07892) (Sun et al., 2026) projects gradients against a capability subspace. Crucially, [GOAL](https://arxiv.org/abs/2610.39802) (Chen et al., Oct 29, 2026, Appendix M & Table 7) formalized why AdamW's diagonal preconditioner rotates orthogonalized Euclidean gradients back into a forbidden subspace ($\mathcal{L}_{i,t}^{\text{Adam}} = 3.6\%\text{--}4.7\%$ across three backbone models (Gemma-7B-it, Llama-2-7B-Chat, Llama-2-13B-Chat), averaged over five runs at the end of training with zero weight decay), and [CORD](https://arxiv.org/abs/2610.29965) (Yu et al., Oct 21, 2026) noted that optimizer transformations can disturb gradient-space orthogonalization.
- **Non-Finite Tensor Payloads and Hardware Cast Divergence:** [Fornaro (`JustVugg/colibri`, `GHSA-5xpg-vw35-2687`, Aug 5, 2026)](https://github.com/JustVugg/colibri/security/advisories/GHSA-5xpg-vw35-2687) demonstrated `NaN` MoE router weights in `.safetensors` (`inkling`, `kimi_k3`, `olmoe`) causing `best = -1` negative-index heap OOB corruption and negative-offset `pread`, and [Hanzo AI `HIP-0113` (Section 4.5, 2026)](https://github.com/hanzo-apps/HIPs/blob/main/HIPs/hip-0113-cognitive-sidecar-and-hanzo-engine-provider-runtime.md) documented ARM64 `FCVTZS` (`0`) vs. x86-64 `CVTTSD2SI` (`INT_MIN`) divergence when quantizing non-finite floats into `int8` vectors.

### 6.1 Summary Matrix of Prior Art vs. Secant Contributions

Before mapping the broader literature, the table below compares the scope of Takehi, Teoh, Zhang, and Khattab ([*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), MIT CSAIL, November 2, 2026) with the multi-domain evaluation in this paper:

| Dimension | Takehi et al. (MIT CSAIL, Nov 2, 2026) | Secant (This Paper) |
|---|---|---|
| **Models & Scale** | `Qwen3.5-0.8B` (`0.8B`) for the quantitative reasoning-loop edit, plus exploratory behavioral probes on small MLP policies and denoisers, character-level transformers, and two Gemma 3 270M fine-tunes | **`17` completed benchmark model identities in the Stage-2 + RMS suites (`0.8B` to `32.8B`)** (`14` five-arm Stage-2 models + `3` additional RMS models `Qwen2.5-Math-7B`, `Qwen2.5-Coder-7B`, `Qwen3-4B`, plus `Gemma-2-27B` `300`-row partial baseline), **`17` checkpoints in the PTH study (`2` Regime A + `1` Regime B + `14` Regime C, `0.8B` to `72.7B`, including `70.6B` / `72.7B` single-shard checks)**, and a `15`-model Stage-1 calibration/adapter suite |
| **Benchmarks & Domains** | `8` benchmarks on `Qwen3.5-0.8B` (`MATH-500` `7.9% -> 48.1%`, `GSM8K` `13.8% -> 67.1%`, `IFEval` `21.0% -> 62.3%`, plus `AMC23`, `Minerva Math`, `OlympiadBench`, `GPQA Diamond`, and `MMLU-Pro`) under 32,768-token stochastic sampling (not re-run in this paper) | **`6` benchmark suites (`59,440` H100 rows = `59,140` completed-cohort rows + `300` partial baseline rows):** `MATH-500` (`13,100`), `GSM8K` (`13,100`), `MMLU-Pro` (`13,100`), `HumanEval` code generation (`10,480`), `ASan-30` (`2,100`), and paired `ASAN-120` (`7,560`: `60` vulnerable C/C++ functions + `60` patched benign twins across `12` CWE families) |
| **Multi-Subspace, Calibration & Controls** | Autonomous agent (`gpt-6-astra`, Astra) under a 10/25/50 forward-pass budget subtracting one loop direction (`B1`, layers 15 & 19, 64 `down_proj` output columns/layer, `0.016%` of weights; `relative_change` `0.106` and `0.081` in `p50c`, while an earlier 30-matrix `p10c` run published explicit scalar `1.5`) | **3 orthogonalized subspaces (`B1` loop + `B2` refusal + `B3` root-cause)** with $R^{-1}$ cross-projection compensation, sized via **Target-Shift** ($\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$) and **RMS-Relative** ($\alpha \sigma_{\text{writer}}$) calibration, paired against **norm-matched random controls** on every model |
| **Math, Code & Reasoning Outcomes** | `Qwen3.5-0.8B` loop suppression lifts `MATH-500` `7.9% -> 48.1%`, `GSM8K` `13.8% -> 67.1%`, and `IFEval` `21.0% -> 62.3%` by preventing `<think>` timeouts | `Qwen3.5-4B` `<think>` closure `0/30 -> 30/30` ($p = 1.86 \times 10^{-9}$ vs. random `0/30`); `Qwen3-8B` `GSM8K` `18% -> 30%` on both `composed` ($p = 0.017$) and `random` ($p = 0.029$, showing norm-matched random controls are essential); `Qwen2.5-Math-7B` `GSM8K` `87% -> 93%` ($p = 0.031$ unadjusted with `ProjectThenHealLinear`); `Qwen2.5-Coder-7B` `HumanEval` `36% -> 42%`; `Qwen3-14B` `ASAN-120` root-cause recall `0.450 -> 0.550` (`+10.0` pp, `95% CI [0.0, +21.67]`) |
| **Quantization, LoRA & Pre-Load Auditing** | Discussed future weight-space tooling and direct weight auditing as open research directions | `Q8_0` (`98.5%` removal) vs. `Q4_K` superblock geometry; **`ProjectThenHealLinear`** (`3.18e-17..5.07e-17` `float64` subspace lock under AdamW); and **`secant sanitize` / `audit-trojan`** across `186.8M` GGUF blocks and `3.26B` float elements (`0` observed false positives across the 10 tested production GGUF and SafeTensors checkpoints) |

The table below separates primitives established in prior or concurrent work from the specific technical contributions evaluated in Secant:

| Technical Area | Established Prior or Concurrent Art (Cited) | Secant's Specific Technical Contribution |
|---|---|---|
| **1. In-Place mmap & Quantized Q8\_0/Q4\_K Block Surgery** | In-place r+ np.memmap GGUF Q8\_0 editing with SHA-256 verification of untouched bytes ([Kiwimaker, Oct 2026](https://huggingface.co/Kiwimaker/DeepSeek-V4.1-Flash-1017-Maya-S-abliterated-GGUF)); byte-level GGUF tensor transplant & Q2\_0 scale-preserving re-encoding ([SC117, Oct 2026](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)). | **(a)** In-place zero-copy mmap byte-slice surgery inside .safetensors files ([8+N+BEGIN, 8+N+END)) without save\_file() shard re-serialization; **(b)** sub-tensor compressed-domain surgery directly on touched 34-byte block\_q8\_0 structs without full-matrix float64 materialization; **(c)** Propositions 1-3 proving why 32-weight Q8\_0 blocks commute with Π₁₂₈ (128 = 4 × 32) while 256-weight Q4\_K superblocks suffer cross-head scale leakage across neighboring 128-column heads. |
| **2. ProjectThenHealLinear & Subspace-Locked LoRA** | Forward-pass LoRA projection ΔW = P⊥ B A for continual learning ([OPLoRA, Xiong & Xie, AAAI 2026](https://arxiv.org/abs/2510.13003)) and safety preservation ([SaLoRA, Li et al., ICLR 2025](https://arxiv.org/abs/2501.01765)); diagnosis of AdamW preconditioner gradient leakage AdamW gradient leakage of 3.6% to 4.7% ([GOAL, Chen et al., Oct 2026](https://arxiv.org/abs/2610.39802); [CORD, Yu et al., Oct 2026](https://arxiv.org/abs/2610.29965)); unconstrained LoRA post-abliteration healing ([Labonne 2024](https://huggingface.co/blog/mlabonne/abliteration); OBLITERATUS; model-unfetter). | **(a)** Bypassing AdamW preconditioner leakage on the effective adapter weight update via forward-pass subspace-locked parameterization, reducing step-60 relative weight-update leakage ‖D ΔW‖F / ‖ΔW‖F from 0.33%-1.67% to the float32 floor (1.78e-10 to 8.67e-10) across 15 models (0.8B to 32.8B); **(b)** inverting safety-projected LoRA into the coupled two-stage **Project-Then-Heal** post-surgery workflow (locking LoRA out of an excised behavioral subspace span(Dᵀ)); **(c)** twice-iterated float64 idempotent projection P⊥(P⊥(B̃)) inside the autograd forward pass (5.6× median error reduction over single-pass projection). |
| **3. Target-Shift Calibration & Gram-Schmidt Compensation** | Algebraic step size β = abs(Δtarget) / ‖hS‖₂² as the isotropic (C = I) special case of [ROME](https://arxiv.org/abs/2202.05262) (Meng et al., NeurIPS 2022) and [MEMIT](https://arxiv.org/abs/2210.07229) (ICLR 2023); multi-vector Gram-Schmidt orthogonalization in OBLITERATUS, llm-abliteration, and [G-Abliteration](https://arxiv.org/abs/2512.18901) (Gülmez, Dec 2025). | **(a)** Cross-scale activation-shift transfer calibration (β(dmodel, ℓ) = abs(Δtarget(ℓ)) / ‖hS(ℓ)‖₂² and RMS-relative dosing Δtarget = α σwriter) across 0.8B to 32.8B full models (plus 70.6B and 72.7B single-shard evaluations); **(b)** coupling two-pass float64 Gram-Schmidt orthonormalization with the **upper-triangular R⁻¹ cross-projection compensation system** (Qᵀ ΔW = R⁻ᵀ Δtarget) for heterogeneous non-orthogonal multi-direction edits. |
| **4. Boundary-3 Static Tensor Auditing & Hardware Cast Divergence** | NaN MoE router weights in .safetensors causing best = -1 negative-index heap OOB corruption and negative-offset pread ([Fornaro, JustVugg/colibri, GHSA-5xpg-vw35-2687, Aug 2026](https://github.com/JustVugg/colibri/security/advisories/GHSA-5xpg-vw35-2687)); ARM64 FCVTZS (0) vs. x86-64 CVTTSD2SI (INT\_MIN) int8 embedding consensus split ([Hanzo AI HIP-0113, Section 4.5, 2026](https://github.com/hanzo-apps/HIPs/blob/main/HIPs/hip-0113-cognitive-sidecar-and-hanzo-engine-provider-runtime.md)); out-of-range g\_idx (CVE-2026-86288) and kernel-dispatch indexing (CVE-2026-84286). | **(a)** Connecting both the qNaN (FCVTZS = 0 vs. CVTTSS2SI = INT32\_MIN) and positive-overflow (+1.0e38: FCVTZS = INT32\_MAX vs. CVTTSS2SI = INT32\_MIN, bypassing idx &gt;= N) hardware cast asymmetries and NVIDIA CUB TwiddleIn (+qNaN 0x7FC00000 -> 0xFFC00000 sorting above +Inf at index 0) to **architecture-divergent C/C++ and CUDA memory-corruption exploit primitives**, verified empirically on ARM64 (Apple Silicon) and x86-64 (AMD Zen 5); **(b)** the unified Boundary-1 / Boundary-2 / Boundary-3 static checkpoint security taxonomy and **Zero-False-Positive Pre-Load Sanitization Standard** (Rule B3-2A permitting multiplicative GGUF 0.0/subnormals and rejecting only exponent 0x1F, Rule B3-2B enforcing s &gt;= 1e-12 on divisor scales, and Rule B3-3 enforcing two-sided bounds on g\_idx/expert\_map), paired with secant audit-trojan and secant scan-sinks. |

**Scope of this paper.** Secant does not claim to introduce orthogonal projection, rank-1 associative weight updates, or null-space LoRA. It evaluates reference-based edit sizing with $R^{-1}$ Gram-Schmidt compensation, block-aware `Q8_0` and `.safetensors` in-place `mmap` surgery, forward-pass twice-iterated `float64` adapter locking, and Boundary-3 static payload auditing inside one reproducible Rust + PyTorch toolkit.

> **Section 6 takeaway:** The projection primitives have rich prior and concurrent art across 2022 to November 2026. Secant contributes in-place `.safetensors` and sub-tensor 34-byte `Q8_0` block surgery with the $\Pi_{128}$ vs. `Q4_K` alignment theorems, bypasses AdamW preconditioner leakage on the effective adapter weight update via twice-iterated `float64` forward-pass locking in `ProjectThenHealLinear`, solves upper-triangular $R^{-1}$ multi-direction shift compensation, and unifies Boundary-3 static tensor auditing against both `best = -1` and `CVTTSS2SI` (`INT_MIN`) vs. `FCVTZS` (`0` / `INT_MAX`) C/C++ inference sinks.

---

## References

1. **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab.** *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL, November 2, 2026. https://rikiyat.github.io/blog/weight-space-programming/
2. **Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda.** *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024 (`arXiv:2406.11717`).
3. **Junfeng Fang et al.** *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR 2025 (`arXiv:2410.02355`).
4. **Pengwei Tang et al.** *Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge (LoRA-Null).* AAAI 2026 (`arXiv:2503.02659`).
5. **Yifeng Xiong and Xiaohui Xie.** *OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning.* AAAI 2026, Vol. 40, No. 40, pp. 34088-34096 (`arXiv:2510.13003`, `doi:10.1609/aaai.v40i40.40703`).
6. **Mingjie Li et al.** *SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation.* ICLR 2025 (`arXiv:2501.01765`).
7. **Chia-Yi Hsu et al.** *Safe LoRA: The Silver Lining of Reducing Safety Risks when Fine-Tuning Large Language Models.* NeurIPS 2024 (`arXiv:2405.16833`).
8. **Guanglong Sun et al.** *[Safety Alignment as Continual Learning: Mitigating the Alignment Tax via Orthogonal Gradient Projection (OGPSA)](https://arxiv.org/abs/2602.07892).* `arXiv:2602.07892`, 2026.
9. **Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov.** *Locating and Editing Factual Associations in GPT (ROME).* NeurIPS 2022 (`arXiv:2202.05262`).
10. **Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau.** *Mass-Editing Memory in a Transformer (MEMIT).* ICLR 2023 (`arXiv:2210.07229`).
11. **Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella Biderman.** *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS 2023 (`arXiv:2306.03819`).
12. **Chao Chen, Pengyang Shao, Wei Qin, Shijie Hao, Fei Shen, and Richang Hong.** *Revisiting Over-refusal in Safety-aligned Large Language Models: A Representation and Optimization Perspective (GOAL).* `arXiv:2610.39802`, October 29, 2026 (Appendix M and Table 7).
13. **Wenhao Yu et al.** *Conflict-Aware Optimization for Joint RL and Distillation in Agents (CORD).* `arXiv:2610.29965`, October 21, 2026.
14. **Nikhil Ghosh, Tetiana Parshakova, and Robert M. Gower.** *PoLoRA: A Preconditioned Orthogonalized LoRA Optimizer.* `arXiv:2607.17620`, July 20, 2026.
15. **Gökdeniz Gülmez.** *Gabliteration: Adaptive Multi-Directional Neural Weight Modification for Selective Behavioral Alteration in Large Language Models.* `arXiv:2512.18901v3`, December 2025 / January 2026.
16. **Jim Lai (`grimjim`).** *Norm-Preserving Biprojected Abliteration.* Hugging Face Blog, November 2025. https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
17. **Maxime Labonne.** *Uncensor any LLM with abliteration (DARE-TIES + post-abliteration LoRA healing).* Hugging Face Blog, June 10, 2024. https://huggingface.co/blog/mlabonne/abliteration
18. **Kiwimaker.** *In-place `r+` memory-mapped GGUF `Q8_0` abliteration with SHA-256 byte-preservation verification (`apply_abliteration.py` and `apply_abliteration_v2.py`).* Hugging Face (`Kiwimaker/DeepSeek-V4.1-Flash-1017-Maya-S-abliterated-GGUF`), October 20-21, 2026. https://huggingface.co/Kiwimaker/DeepSeek-V4.1-Flash-1017-Maya-S-abliterated-GGUF
19. **SC117.** *Byte-level in-place GGUF tensor transplantation and scale-preserving `Q2_0` re-encoding (`Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF`).* Hugging Face, October 3, 2026. https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF
20. **Vincenzo Fornaro.** *NaN router logits leave MoE top-k selection at best=-1, causing negative-index heap OOB read/write and negative-offset pread (`GHSA-5xpg-vw35-2687`), plus `.safetensors` loader advisories (`GHSA-pmq2-6f2p-hjvf`, `GHSA-4gw4-j89j-4c8r`, `GHSA-wc4x-3786-cxh7`).* GitHub Security Advisories (`JustVugg/colibri`), August 5, 2026. https://github.com/JustVugg/colibri/security/advisories/GHSA-5xpg-vw35-2687
21. **Hanzo AI Standards Track.** *HIP-0113: Cognitive Sidecar & Hanzo Engine Provider Runtime for Thinking Chains (Section 4.5: `FCVTZS` vs. `CVTTSD2SI` non-finite cast divergence).* Hanzo Improvement Proposals, 2026. https://github.com/hanzo-apps/HIPs/blob/main/HIPs/hip-0113-cognitive-sidecar-and-hanzo-engine-provider-runtime.md
22. **Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.** *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022 (`arXiv:2106.09685`).
