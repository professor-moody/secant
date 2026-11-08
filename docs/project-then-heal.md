# Project-Then-Heal: Subspace-Locked LoRA

After editing a model, you may want to fine-tune it for fluency or a new task. **LoRA** does this by learning a small adapter while keeping the original weight matrix fixed. An ordinary adapter can add back a component along a direction that the edit removed.

**Project-Then-Heal** removes specified output directions from the adapter whenever it runs. The remaining directions form the *orthogonal complement*: the set perpendicular to all protected directions. We call any remaining protected component **leakage**. It is a local numerical measure, not a measure of how often the model shows a behavior.

`secant.nn.ProjectThenHealLinear` applies this constraint on every forward pass. `secant heal` applies it once to an existing adapter; further unconstrained training does not retain the lock. The guarantee is exact in real arithmetic at the selected writer, a matrix that contributes to a layer's output. Rounding and later model layers can change the result.

**Prior art.** [OPLoRA](https://arxiv.org/abs/2510.13003) (Xiong & Xie, AAAI 2026) constrains LoRA updates in the forward pass against weight-SVD subspaces on both sides ($\Delta W = P_L B A P_R$) for continual learning; [SaLoRA](https://arxiv.org/abs/2501.01765) (Li et al., ICLR 2025) and [Safe LoRA](https://arxiv.org/abs/2405.16833) (Hsu et al., NeurIPS 2024) project LoRA updates onto safety-aligned subspaces; [PoLoRA](https://arxiv.org/abs/2607.17620) (Ghosh et al., July 2026) formulates a preconditioned orthogonalized LoRA optimizer; and [OGPSA](https://arxiv.org/abs/2602.07892) (Sun et al., 2026) operates on capability and safety gradients. Crucially, the mathematical observation that AdamW's diagonal second-moment preconditioner fails to commute with Euclidean orthogonal gradient projection (rotating orthogonalized gradients back into a forbidden subspace) was independently formalized in October 2026 by Chen et al. ([GOAL](https://arxiv.org/abs/2610.39802), Appendix M & Table 7, who defined the gradient-step quantity *"residual refusal-subspace leakage after AdamW preconditioning"*, $\mathcal{L}_{i,t}^{\text{Adam}} = 3.6\%\text{--}4.7\%$ across three backbone models at the end of training with zero weight decay), while Yu et al. ([CORD](https://arxiv.org/abs/2610.29965), Oct 2026) noted that subsequent optimizer transformations can disturb gradient-space orthogonalization. Existing post-abliteration LoRA capability-healing workflows ([Labonne, June 2024](https://huggingface.co/blog/mlabonne/abliteration); [Obliteratus `--auto-finetune`](https://github.com/elder-plinius/OBLITERATUS); [model-unfetter `recovery.py`](https://github.com/josepha-mayo/model-unfetter)) use unconstrained LoRA. Secant's specific contributions are bypassing AdamW preconditioner leakage on the effective adapter weight update ($\lVert D \Delta W\rVert_{\mathrm{F}}/\lVert\Delta W\rVert_{\mathrm{F}}$) via forward-pass subspace-locked parameterization, inverting safety-projected LoRA into the coupled two-stage **Project-Then-Heal** post-surgery workflow (locking LoRA out of an excised behavioral subspace $\mathrm{span}(D^\top)$), and enforcing twice-iterated `float64` idempotent projection $P_\perp(P_\perp(\tilde{B}))$ inside the autograd forward pass. See [the paper's related-work discussion](PAPER.md) for bibliographic details and limits.

```{figure} assets/project-then-heal-chart.svg
:name: fig-pth-leakage-comparison
:alt: Regime C H100 stress test on 14 checkpoints, unconstrained adapter fraction in span(D) vs ProjectThenHealLinear absolute leakage
:width: 100%

H100 stress test (14 checkpoints, `7.2B` to `72.7B`) at step 60. The training target lies inside $\mathrm{span}(D^\top)$, so the unconstrained bars ($\|DB\|_{\mathrm{F}}/\|B\|_{\mathrm{F}}$, `8.42%` to `52.55%`) measure that task's pull, not erasure in real training, and the locked arm cannot fit the target (its loss sits at the floor $2.25/d_{\text{model}}$). `ProjectThenHealLinear` absolute leakage $\max|D B_{\text{locked}}|$ is $4.82 \times 10^{-10}$ to $1.08 \times 10^{-9}$ (`float64`) and $3.87 \times 10^{-10}$ to $3.98 \times 10^{-8}$ (merged `F32`). The two columns are different metrics. Live fine-tuning results are in Section 2.
```

---

## 1. Constrain the Adapter Used by the Forward Pass

A gradient hook changes the proposed training direction. The optimizer still transforms that proposal before updating the parameters. Constraining the adapter used to compute each output avoids depending on how the optimizer transforms gradients.

A standard LoRA adapter on a writer matrix has the form $\Delta W = s \, B \, A$, where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$. To guarantee zero leakage into the surgical basis $D \in \mathbb{R}^{k \times d_{\text{out}}}$ ($D D^\top = I_k$), we require:

$$
D \, \Delta W = 0
$$

Why not simply multiply the backward gradient $\nabla_B \mathcal{L}$ by $P_\perp = I_{d_{\text{out}}} - D^\top D$?
1. **Unprojected initialization leaks immediately:** If $B_0$ has any component in $\mathrm{span}(D^\top)$, projecting gradients never removes it.
2. **AdamW coordinatewise scaling can rotate steps out of $\ker(D)$:** Even if $\nabla_B \mathcal{L}$ lies inside $\ker(D)$, its adaptive diagonal rescaling need not preserve that subspace. Axis-aligned projectors and compatible scalings are exceptions. Section 3 tests a gradient-projected arm; it does not prove every AdamW step leaks.

Instead, `ProjectThenHealLinear` defines the effective output matrix $B_{\text{locked}}(\tilde{B})$ **structurally inside the forward pass** via double orthogonal projection:

$$
B_{\text{locked}}(\tilde{B}) = P_\perp\!\left(P_\perp(\tilde{B})\right) = \left(\tilde{B} - D^\top (D \tilde{B})\right) - D^\top \!\left(D \left(\tilde{B} - D^\top (D \tilde{B})\right)\right)
$$

and computes the forward pass as:

$$
y = x \tilde{W}^\top + \frac{\alpha_{\text{lora}}}{r} (x A^\top) B_{\text{locked}}(\tilde{B})^\top
$$

### Cost and Numerical Limits

The implementation avoids allocating a full square projector. The remaining limits come from floating-point arithmetic and the precision used to store or merge the adapter.

1. **Zero $O(d_{\text{out}}^2)$ Matrix Allocation:** `ProjectThenHealLinear` never materializes the dense $d_{\text{out}} \times d_{\text{out}}$ matrix $I - D^\top D$. Each projection executes two skinny matrix multiplications $D \tilde{B} \in \mathbb{R}^{k \times r}$ and $D^\top (D \tilde{B}) \in \mathbb{R}^{d_{\text{out}} \times r}$ in $O(k \cdot d_{\text{out}} \cdot r)$ time. Device timing is workload-dependent; no general CPU/GPU speedup follows from this complexity bound.
2. **Why Double Projection ("Twice Is Enough"):** In exact arithmetic $P_\perp^2 = P_\perp$, so a second projection changes nothing. In `float64` the second pass is an **observed numerical refinement**, not a proved error bound: we do not claim a $\kappa(\tilde{B}) \cdot \varepsilon_{\text{mach}}$ theorem. Measured with the shipped `secant._linalg.project_out` on 120 seeded Gaussian cases ($d_{\text{out}} \in \{1024, 2560, 3584, 4096, 5120, 8192\}$, $k \in \{3, 4\}$, $r \in \{8, 16\}$, 5 seeds), the second pass lowered $\max_{i,j} |(D B_{\text{locked}})_{i,j}|$ by $1.8\times$ to $17.7\times$ (median $5.6\times$), from at most $2.25 \times 10^{-14}$ to at most $2.24 \times 10^{-15}$. After 60 steps of live training on `Qwen3.5-0.8B` we measured $9.87 \times 10^{-18}$ in `float64` and $2.49 \times 10^{-10}$ in the merged `F32` delta (Section 2).
3. **Storage Precision Gate (`F32` vs. `BF16` Merge):** Because `BF16` has 7 explicit fraction bits (spacing at 1 is $2^{-7} \approx 7.8 \times 10^{-3}$; spacing scales with the exponent), rounding $\tilde{W} + \Delta W$ into `BF16` introduced leakage along $D$ of $5.98 \times 10^{-5}$ to $9.46 \times 10^{-4}$ at the final recorded steps for the locked arm in our live runs (`Qwen3.5-0.8B`, `SmolLM2-1.7B`), comparable to the unconstrained arm on one `SmolLM2` module. `secant heal` therefore stores $B_{\text{locked}}$ in `F32`/`F64`, and `secant verify-subspace` fails closed if a merged tensor exceeds `--tol 1e-5`.

---

Precision note: Regime C's `leakage_f64` evaluates the F32 working factor in float64; its merged metric measures $D\Delta W$. The live-run `leak_inf_float64_math` values below instead recompute an unrounded projected delta. These labels do not describe identical computations or an all-float64 LM forward pass.

## 2. Live Causal-LM Healing Across Eight Models

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

### 2.1 H100 RMS-Relative Healing on Six Models

All six H100 runs train a rank-4 adapter for 60 steps on fixed neutral texts after composing four domain edits. The reported float64 relative leakage is 3.18e-17 to 5.07e-17. On Qwen2.5-Math-7B, GSM8K rises from 87/100 to 93/100; on Qwen2.5-Coder-7B, HumanEval rises from 36/100 to 42/100. Neither gain survives Holm correction, and the study has no healed-random or LoRA-on-unedited-base arm, so it cannot isolate composition from healing. Full doses, controls, manifest identifiers, and benchmark counts appear in the [paper's RMS-relative sweep](#sec-paper-rms-dosing).

### 2.2 Early Regime A Runs: Perplexity and Leakage

The SmolLM2-1.7B-Instruct run trained for 20 steps on ROCm after a B2 projector edit at L14 and a B3 calibrated edit at L15. Its held-out perplexity was 21.260 with PTH versus 21.255 with unconstrained LoRA. Maximum absolute leakage was 5.21e-18 for the unrounded float64 PTH delta, 1.46e-10 after F32 merge, and 4.64e-03 for unconstrained LoRA. The [paper](#sec-paper-live-healing) records both early edits and their precision boundary.

The Qwen3.5-0.8B run below gives the step-by-step trajectory.

Both adapters start from the same edited model and initial parameters. Training loss measures fit to the training batch; held-out perplexity measures prediction quality on other text, with lower values better. Watching both reveals when a model begins to overfit.

Attaching `ProjectThenHealLinear` ($r=8, \alpha_{\text{lora}}=16, k=4$ protected directions, AdamW $\text{lr} = 5 \times 10^{-4}$) to layers `15, 16, 17` of surgically edited `Qwen3.5-0.8B` alongside a bit-identically initialized `unconstrained_lora` control over 60 steps of real causal LM fine-tuning gives near-identical training loss (`0.3120` vs. `0.3096` at step 60) and a factor of about $4.2 \times 10^{7}$ lower merged `F32` writer leakage ($2.49 \times 10^{-10}$ vs. $1.05 \times 10^{-2}$), measured with the same metric in both arms. Note that on this 24-example batch, held-out perplexity reaches its minimum at Step 20 (`5.435` vs. `5.423`) before both arms overfit the small training batch by Step 60 (`12.106` vs. `12.357`):

- PTH: ProjectThenHealLinear; UNC: unconstrained LoRA; PPL: held-out perplexity (lower is better).
- Max leakage: maximum absolute entry of D ΔW. The float64 column recomputes the unrounded projected delta.

**Qwen3.5-0.8B training and held-out quality**

| Step | PTH loss | UNC loss | PTH PPL | UNC PPL |
|---|---:|---:|---:|---:|
| 0 | 2.8242 | 2.8242 | 17.568 | 17.568 |
| 10 | 2.0627 | 2.0599 | 7.334 | 7.315 |
| 20 (min PPL) | 1.6182 | 1.6180 | **5.435** | **5.423** |
| 40 | 0.9485 | 0.9459 | 5.845 | 5.884 |
| 60 (final) | **0.3120 (-89.0%)** | **0.3096 (-89.0%)** | 12.106 | 12.357 |

**Qwen3.5-0.8B writer leakage**

| Step | PTH float64 max | PTH merged F32 max | UNC merged F32 max |
|---|---:|---:|---:|
| 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 10 | 3.63e-18 | 6.54e-11 | 2.60e-03 |
| 20 (Min PPL) | 5.53e-18 | 1.22e-10 | 4.46e-03 |
| 40 | 7.97e-18 | 1.96e-10 | 7.98e-03 |
| 60 (Final) | **9.87e-18** | **2.49e-10 (PASS)** | **1.05e-02 (FAIL)** |

---

## 3. Compare Three Adapter Methods Under AdamW

Section 1 argues that projecting only the gradient is not enough under AdamW. This section tests that argument with an arm that does exactly that.

**Setup.** Writer: the real layer-16 `mlp.down_proj` weight of `Qwen3.5-0.8B` ($d_{\text{out}} = 1024$), `Qwen3.5-4B` ($2560$) and `Qwen2.5-7B` ($3584$), surgically edited as $W_e = P_\perp W_0$ so that $D W_e = 0$. $D$ has $k = 3$ seeded random orthonormal `float64` rows. The task fits numerical target outputs (**synthetic regression**) rather than fine-tuning on language: $T = 256$ Gaussian inputs, an *orthogonal* target $Y = W_e X + R_\perp$ with $D R_\perp = 0$, and a *mixed* target that adds $R_D \in \mathrm{span}(D)$ with energy `0.25` of $R_\perp$ (so `0.25` is the analytic loss floor of any arm that cannot write into $\mathrm{span}(D^\top)$). The settings are LoRA rank `16`, scale `1`, AdamW betas `(0.9, 0.999)`, eps `1e-8`, 60 full-batch steps, seeds `0, 1, 2`, identical $A_0$ and $B_0 = 0$ in every arm.

The Python projection accumulates in float64 and casts back to the working dtype before the forward pass (`python/secant/nn.py:205`). Numerical leakage is therefore a measured quantity, not a dtype-independent zero guarantee.

- **Arm A:** unconstrained LoRA + AdamW.
- **Arm B:** gradient-projected LoRA, $\nabla_B \leftarrow P_\perp \nabla_B$ in `float64` immediately before each AdamW step.
- **Arm C:** `secant.nn.ProjectThenHealLinear`, $B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$ in the forward pass, same AdamW.

Every arm is scored with the **same** metric: $\mathrm{lk}_W = \|D\,\Delta W\|_{\mathrm{F}} / \|\Delta W\|_{\mathrm{F}}$ (relative leakage of the LoRA delta into $\mathrm{span}(D^\top)$), plus the output-energy fraction in $\mathrm{span}(D^\top)$ and the largest $|D\,\Delta W|$ of the `float32`-stored merged delta. Weight decay is run at both `0.01` and `0.0`: decoupled decay multiplies $B$ by a scalar and commutes with $P_\perp$, so it should not matter.

**Learning rate: this table is post-hoc.** The preregistered learning rate (`1e-2`, chosen once on Arm A) failed the preregistered loss-parity prediction for Arm C on `Qwen2.5-7B`. The `3e-3` run below was added **after** seeing that result and is reported beside, not instead of, the preregistered run. The preregistered `1e-2` run gives the same ordering: on the orthogonal target, Arm B leaks `1.24%` to `2.58%` at step 60 and Arm C $6.40 \times 10^{-10}$ to $1.12 \times 10^{-9}$.

```{figure} assets/adamw-3arm-rotation.svg
:name: fig-adamw-3arm-rotation
:alt: Log-scale leakage over 60 AdamW steps for unconstrained, gradient-projected and ProjectThenHealLinear LoRA on Qwen3.5-0.8B, Qwen3.5-4B and Qwen2.5-7B, a step-60 dot plot across targets and weight decays, and a step-60 dot plot of all three arms on the 15-model H100 Stage-1 run
:width: 100%

Top: $\mathrm{lk}_W$ per AdamW step on the orthogonal target at weight decay `0.0` (line = 3-seed mean, band = seed min to max). Arms A and B coincide at step 1 because the gradient already lies outside $\mathrm{span}(D^\top)$; Arm B then settles at `1.56%` to `2.98%` while Arm C stays at $6.05 \times 10^{-10}$ to $1.02 \times 10^{-9}$. Bottom left: step-60 means for both targets and both weight decays (filled = `0.0`, ring = `0.01`; the two coincide, so weight decay is not the source). Bottom right: all three arms at step 60 on the 15-model H100 Stage-1 run (one seed per model, one metric for every arm; see the subsection below). Generated from the banked receipts by `gen_secant_figures.py`, which checks every receipt's SHA-256 before drawing.
```

- orth: orthogonal target; mixed: target with a protected-subspace component; wd: weight decay.
- A: unconstrained; B: gradient-projected; C: ProjectThenHealLinear. Each configuration uses three seeds.
- Loss range is min to max; loss minus floor, step leakage, and output energy are seed means. Max over time and merged F32 max are maxima across seeds (and steps where applicable).
- Relative leakage is the norm ratio defined above; output energy is the protected-subspace fraction; merged F32 max is the maximum absolute entry of D ΔW after storage.

#### `Qwen3.5-0.8B` (1024)

**Loss at step 60**

| Target / arm / wd | Loss range | Loss minus floor |
|---|---:|---:|
| orth / A / 0.01 | 0.00122..0.00128 | +0.00125 |
| orth / B / 0.01 | 0.00190..0.00219 | +0.00208 |
| orth / C / 0.01 | 0.00122..0.00128 | +0.00125 |
| orth / A / 0.0 | 0.00123..0.00128 | +0.00126 |
| orth / B / 0.0 | 0.00190..0.00219 | +0.00208 |
| orth / C / 0.0 | 0.00123..0.00128 | +0.00125 |
| mixed / A / 0.01 | 0.00187..0.00272 | -0.24775 |
| mixed / B / 0.01 | 0.24301..0.24621 | -0.00522 |
| mixed / C / 0.01 | 0.25122..0.25128 | +0.00125 |
| mixed / A / 0.0 | 0.00188..0.00273 | -0.24774 |
| mixed / B / 0.0 | 0.24301..0.24621 | -0.00522 |
| mixed / C / 0.0 | 0.25123..0.25128 | +0.00125 |

**Relative weight-update leakage**

| Target / arm / wd | Step 1 | Step 60 | Max over time |
|---|---:|---:|---:|
| orth / A / 0.01 | 3.29e-02 | 8.18e-03 | 3.47e-02 |
| orth / B / 0.01 | 3.29e-02 | **2.98e-02** | 3.50e-02 |
| orth / C / 0.01 | 1.14e-09 | **1.05e-09** | 1.62e-09 |
| orth / A / 0.0 | 3.29e-02 | 8.19e-03 | 3.47e-02 |
| orth / B / 0.0 | 3.29e-02 | **2.98e-02** | 3.50e-02 |
| orth / C / 0.0 | 1.14e-09 | **1.02e-09** | 1.60e-09 |
| mixed / A / 0.01 | 3.13e-01 | 4.54e-01 | 4.62e-01 |
| mixed / B / 0.01 | 3.29e-02 | 3.06e-02 | 3.50e-02 |
| mixed / C / 0.01 | 1.36e-09 | 1.11e-09 | 1.82e-09 |
| mixed / A / 0.0 | 3.13e-01 | 4.54e-01 | 4.62e-01 |
| mixed / B / 0.0 | 3.29e-02 | 3.06e-02 | 3.50e-02 |
| mixed / C / 0.0 | 1.36e-09 | 1.08e-09 | 1.59e-09 |

**Output and merged-weight diagnostics**

| Target / arm / wd | Output energy | Merged F32 max |
|---|---:|---:|
| orth / A / 0.01 | 2.86e-06 | 3.95e-03 |
| orth / B / 0.01 | 8.22e-04 | 1.08e-02 |
| orth / C / 0.01 | 9.48e-19 | 1.60e-09 |
| orth / A / 0.0 | 2.86e-06 | 3.95e-03 |
| orth / B / 0.0 | 8.23e-04 | 1.08e-02 |
| orth / C / 0.0 | 8.92e-19 | 1.77e-09 |
| mixed / A / 0.01 | 1.99e-01 | 1.94e-01 |
| mixed / B / 0.01 | 8.84e-04 | 1.18e-02 |
| mixed / C / 0.01 | 1.06e-18 | 1.58e-09 |
| mixed / A / 0.0 | 1.99e-01 | 1.94e-01 |
| mixed / B / 0.0 | 8.84e-04 | 1.18e-02 |
| mixed / C / 0.0 | 1.04e-18 | 1.76e-09 |

#### `Qwen3.5-4B` (2560)

**Loss at step 60**

| Target / arm / wd | Loss range | Loss minus floor |
|---|---:|---:|
| orth / A / 0.01 | 0.00085..0.00094 | +0.00090 |
| orth / B / 0.01 | 0.00109..0.00127 | +0.00117 |
| orth / C / 0.01 | 0.00085..0.00094 | +0.00090 |
| orth / A / 0.0 | 0.00086..0.00094 | +0.00090 |
| orth / B / 0.0 | 0.00110..0.00128 | +0.00117 |
| orth / C / 0.0 | 0.00086..0.00094 | +0.00090 |
| mixed / A / 0.01 | 0.00133..0.00152 | -0.24857 |
| mixed / B / 0.01 | 0.23067..0.24065 | -0.01342 |
| mixed / C / 0.01 | 0.25085..0.25094 | +0.00090 |
| mixed / A / 0.0 | 0.00134..0.00152 | -0.24857 |
| mixed / B / 0.0 | 0.23064..0.24064 | -0.01344 |
| mixed / C / 0.0 | 0.25086..0.25094 | +0.00090 |

**Relative weight-update leakage**

| Target / arm / wd | Step 1 | Step 60 | Max over time |
|---|---:|---:|---:|
| orth / A / 0.01 | 2.02e-02 | 6.10e-03 | 2.19e-02 |
| orth / B / 0.01 | 2.02e-02 | **1.74e-02** | 2.23e-02 |
| orth / C / 0.01 | 7.94e-10 | **7.71e-10** | 1.17e-09 |
| orth / A / 0.0 | 2.02e-02 | 6.11e-03 | 2.19e-02 |
| orth / B / 0.0 | 2.02e-02 | **1.74e-02** | 2.23e-02 |
| orth / C / 0.0 | 7.94e-10 | **7.33e-10** | 1.25e-09 |
| mixed / A / 0.01 | 3.92e-01 | 4.52e-01 | 5.24e-01 |
| mixed / B / 0.01 | 2.02e-02 | 2.36e-02 | 2.58e-02 |
| mixed / C / 0.01 | 7.96e-10 | 6.84e-10 | 1.05e-09 |
| mixed / A / 0.0 | 3.92e-01 | 4.52e-01 | 5.24e-01 |
| mixed / B / 0.0 | 2.02e-02 | 2.37e-02 | 2.58e-02 |
| mixed / C / 0.0 | 7.96e-10 | 6.91e-10 | 1.04e-09 |

**Output and merged-weight diagnostics**

| Target / arm / wd | Output energy | Merged F32 max |
|---|---:|---:|
| orth / A / 0.01 | 3.89e-07 | 1.70e-03 |
| orth / B / 0.01 | 2.65e-04 | 4.44e-03 |
| orth / C / 0.01 | 5.30e-19 | 7.06e-10 |
| orth / A / 0.0 | 3.89e-07 | 1.70e-03 |
| orth / B / 0.0 | 2.65e-04 | 4.45e-03 |
| orth / C / 0.0 | 4.74e-19 | 6.87e-10 |
| mixed / A / 0.01 | 2.00e-01 | 1.20e-01 |
| mixed / B / 0.01 | 6.03e-04 | 9.28e-03 |
| mixed / C / 0.01 | 3.84e-19 | 7.51e-10 |
| mixed / A / 0.0 | 2.00e-01 | 1.20e-01 |
| mixed / B / 0.0 | 6.04e-04 | 9.29e-03 |
| mixed / C / 0.0 | 4.27e-19 | 7.03e-10 |

#### `Qwen2.5-7B` (3584)

**Loss at step 60**

| Target / arm / wd | Loss range | Loss minus floor |
|---|---:|---:|
| orth / A / 0.01 | 0.00075..0.00077 | +0.00076 |
| orth / B / 0.01 | 0.00098..0.00098 | +0.00098 |
| orth / C / 0.01 | 0.00075..0.00077 | +0.00076 |
| orth / A / 0.0 | 0.00075..0.00077 | +0.00076 |
| orth / B / 0.0 | 0.00098..0.00098 | +0.00098 |
| orth / C / 0.0 | 0.00075..0.00077 | +0.00076 |
| mixed / A / 0.01 | 0.00102..0.00112 | -0.24893 |
| mixed / B / 0.01 | 0.24067..0.24308 | -0.00773 |
| mixed / C / 0.01 | 0.25075..0.25077 | +0.00076 |
| mixed / A / 0.0 | 0.00102..0.00113 | -0.24893 |
| mixed / B / 0.0 | 0.24067..0.24307 | -0.00774 |
| mixed / C / 0.0 | 0.25075..0.25077 | +0.00076 |

**Relative weight-update leakage**

| Target / arm / wd | Step 1 | Step 60 | Max over time |
|---|---:|---:|---:|
| orth / A / 0.01 | 1.94e-02 | 4.94e-03 | 2.24e-02 |
| orth / B / 0.01 | 1.94e-02 | **1.56e-02** | 2.24e-02 |
| orth / C / 0.01 | 6.37e-10 | **5.42e-10** | 8.52e-10 |
| orth / A / 0.0 | 1.94e-02 | 4.95e-03 | 2.24e-02 |
| orth / B / 0.0 | 1.94e-02 | **1.56e-02** | 2.24e-02 |
| orth / C / 0.0 | 6.37e-10 | **6.05e-10** | 8.07e-10 |
| mixed / A / 0.01 | 3.70e-01 | 4.60e-01 | 5.05e-01 |
| mixed / B / 0.01 | 1.94e-02 | 1.84e-02 | 2.24e-02 |
| mixed / C / 0.01 | 6.04e-10 | 5.98e-10 | 8.39e-10 |
| mixed / A / 0.0 | 3.70e-01 | 4.60e-01 | 5.05e-01 |
| mixed / B / 0.0 | 1.94e-02 | 1.84e-02 | 2.24e-02 |
| mixed / C / 0.0 | 6.04e-10 | 6.07e-10 | 8.10e-10 |

**Output and merged-weight diagnostics**

| Target / arm / wd | Output energy | Merged F32 max |
|---|---:|---:|
| orth / A / 0.01 | 4.70e-07 | 8.84e-04 |
| orth / B / 0.01 | 2.22e-04 | 2.07e-03 |
| orth / C / 0.01 | 2.38e-19 | 3.81e-10 |
| orth / A / 0.0 | 4.71e-07 | 8.86e-04 |
| orth / B / 0.0 | 2.23e-04 | 2.07e-03 |
| orth / C / 0.0 | 3.25e-19 | 3.85e-10 |
| mixed / A / 0.01 | 2.01e-01 | 8.44e-02 |
| mixed / B / 0.01 | 3.57e-04 | 3.83e-03 |
| mixed / C / 0.01 | 2.99e-19 | 3.94e-10 |
| mixed / A / 0.0 | 2.01e-01 | 8.45e-02 |
| mixed / B / 0.0 | 3.57e-04 | 3.83e-03 |
| mixed / C / 0.0 | 3.27e-19 | 3.82e-10 |

**What the table shows.**

1. **Gradient projection alone does not keep the adapter out of $\mathrm{span}(D^\top)$.** On the orthogonal target, Arm B puts `1.56%` to `2.98%` of $\|\Delta W\|_{\mathrm{F}}$ into $\mathrm{span}(D^\top)$ at step 60 (`1.84%` to `3.06%` on the mixed target), and the value is the same at weight decay `0.0` as at `0.01`. Weight decay is therefore not the source.
2. **The step-1 mechanism check points at the second-moment preconditioner.** At $B_0 = 0$ on the orthogonal target, the loss gradient already lies in $\mathrm{col}(P_\perp)$ ($\|D g\| / \|g\| = 5.1 \times 10^{-9}$ to $1.3 \times 10^{-8}$), so Arm B's projection is a no-op there. AdamW's first update is $u = g / (|g| + \epsilon)$, an elementwise sign-like map, and $\|D u\| / \|u\|$ is `1.76%` to `3.48%` across the 9 model-seed pairs, matching Arm B's recorded step-1 leakage to within $2 \times 10^{-7}$ relative. The leak is created by the coordinatewise division, not by the gradient.
3. **`ProjectThenHealLinear` holds leakage at rounding level with the same final loss as unconstrained LoRA on the orthogonal target.** Arm C stays at $5.42 \times 10^{-10}$ to $1.05 \times 10^{-9}$ at step 60 on the orthogonal target ($5.98 \times 10^{-10}$ to $1.11 \times 10^{-9}$ on the mixed target), and its final-loss range equals Arm A's to the precision shown in every orthogonal row. Arm B, by contrast, ends with a **higher** loss than A and C on the orthogonal target.
4. **On the mixed target, Arm C stops exactly at the floor.** It cannot fit $R_D$ by construction, so its final loss sits at `0.25` plus the same excess it shows on the orthogonal task. Arm A goes below the floor by writing about `45%` of its delta norm into $\mathrm{span}(D^\top)$; that is the behavior the lock exists to prevent.

**Limits.** This is a single-layer synthetic regression with random protected directions, not a measurement of language-model fine-tuning quality. Regime A (Section 2) tests live text training; Regime C in the [paper](PAPER.md) is a separate synthetic stress test. The learning rate in the table is post-hoc, as stated above. Raw per-seed trajectories, the runner (`adamw3arm.py`) and the mechanism check (`signcheck.py`) are retained with SHA-256 receipts in the lab evidence bank and are not yet published in this repository.

### H100 Stage-1: Repeat the Comparison Across 15 Models

This sweep uses the same three methods on one writer per model, with one seed each. It tests whether the local constraint holds at different sizes. It does not measure public benchmark quality after healing.

The same three arms were run on the H100 on 15 models (receipt `irrefutable-15models-stage1-receipt.json`, SHA-256 `00f4fe2afc4ff630c7c9ac55b5b2c0a85342477a5b32a0b0c455abd5b4c5056f`). Each starts from the **surgically edited** writer $W_{\text{surg}}$ at the layer below, from a common initialization (initial relative leakage below $10^{-16}$), and trains for 60 AdamW steps on a synthetic target $Y_{\text{target}}$ with $Y_{\text{target}} - W_{\text{surg}} X \in \mathrm{span}(D)^\perp$ plus a real-token cross-entropy term (weight `0.1`).

All three arms use one metric, $\lVert D\,\Delta W\rVert_{\mathrm{F}} / \lVert\Delta W\rVert_{\mathrm{F}}$, at step 60. Arm A (unconstrained) leaks `0.82% to 8.51%`; Arm B (gradient-projected, $P_\perp \nabla_B$ applied after backward and immediately before `AdamW.step`) leaks `0.33% to 1.67%`; Arm C (`ProjectThenHealLinear`) stays at `1.78e-10 to 8.67e-10`. Arm B is nonzero on all 15 models, as the AdamW argument in Section 1 and in the [paper, Section 3.1](PAPER.md) predicts for gradient projection, and stays `6.5e6` to `3.8e7` times above Arm C on the same model. Unlike the three-model synthetic rotation test, Arm B sits **below** Arm A here on 15 of 15 models; the task differs (orthogonal synthetic MSE plus real-token cross-entropy), so the A-versus-B ordering is task-dependent and only the B-versus-C gap is common to both runs.

- A, B, and C are the three arms above; PTH is Arm C.
- Relative leakage is the norm ratio at step 60; max abs is the maximum absolute entry of D ΔW.

**Stage-1 coverage and absolute leakage**

| Model | Size | Layer | PTH max abs |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 0.8B | 17 | 4.00e-10 |
| SmolLM2-1.7B | 1.7B | 17 | 1.97e-10 |
| Qwen3.5-4B | 4B | 23 | 2.25e-10 |
| Mistral-7B-v0.3 | 7.2B | 12 | 2.21e-10 |
| Qwen2.5-7B | 7.6B | 8 | 2.43e-10 |
| Llama-3.1-8B | 8B | 23 | 3.51e-10 |
| R1-Llama-8B | 8B | 23 | 2.32e-10 |
| Qwen3-8B | 8.2B | 14 | 2.14e-10 |
| Gemma-2-9B | 9.2B | 16 | 6.67e-10 |
| OLMo-2-13B | 13.7B | 16 | 1.87e-10 |
| Phi-4 | 14.7B | 12 | 3.65e-10 |
| Qwen3-14B | 14.8B | 16 | 1.81e-10 |
| R1-Qwen-14B | 14.8B | 28 | 2.06e-10 |
| Gemma-2-27B | 27.2B | 14 | 1.03e-10 |
| Qwen3-32B | 32.8B | 47 | 1.22e-10 |

**Stage-1 relative leakage**

| Model | A: unconstrained | B: gradient-projected | C: PTH |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 7.60% | 0.61% | 8.67e-10 |
| SmolLM2-1.7B | 2.95% | 1.62% | 5.41e-10 |
| Qwen3.5-4B | 2.22% | 1.67% | 5.81e-10 |
| Mistral-7B-v0.3 | 2.17% | 0.92% | 3.38e-10 |
| Qwen2.5-7B | 2.04% | 0.49% | 4.71e-10 |
| Llama-3.1-8B | 0.82% | 0.80% | 4.36e-10 |
| R1-Llama-8B | 1.59% | 0.60% | 4.50e-10 |
| Qwen3-8B | 2.02% | 0.58% | 3.14e-10 |
| Gemma-2-9B | 8.51% | 0.35% | 5.46e-10 |
| OLMo-2-13B | 1.70% | 0.44% | 1.78e-10 |
| Phi-4 | 0.86% | 0.33% | 3.26e-10 |
| Qwen3-14B | 2.13% | 0.60% | 2.69e-10 |
| R1-Qwen-14B | 1.23% | 0.77% | 3.53e-10 |
| Gemma-2-27B | 1.98% | 1.31% | 3.49e-10 |
| Qwen3-32B | 2.04% | 0.75% | 3.07e-10 |

```{figure} assets/h100-15model-leakage-log10.svg
:name: fig-pth-leakage-log10
:alt: Log-scale relative leakage at step 60 for unconstrained LoRA, gradient-projected AdamW and ProjectThenHealLinear on 15 models, with the B over C ratio per model, and trajectories over 60 AdamW steps
:width: 100%

The table above on a log scale. **Left:** step-60 relative leakage $\lVert D\,\Delta W\rVert_{\mathrm{F}} / \lVert\Delta W\rVert_{\mathrm{F}}$ for each model, ordered by size. Magenta is Arm A, yellow is Arm B, and green is Arm C, with the B-to-C ratio printed above each row. The B-to-C ratio spans `6.5e6` to `3.8e7` (`6.8` to `7.6` orders of magnitude) across the 15 models, from `0.8B` to `32.8B`. **Right:** all 15 trajectories per arm (thin) and their median (thick). Arms A and B leak within the first 10 AdamW steps and then plateau. That is the coordinatewise rescaling described in the step-1 mechanism check, repeating at scale. Arm C stays near `1e-10` throughout. Generated from the receipt by `gen_secant_w61e_figures.py`, which checks the receipt's SHA-256 before drawing.
```

**Limits.** One seed (`20261105`) per model and one edited layer per model; no confidence interval across seeds. The receipt marks public benchmarks `NOT_RUN`, so it says nothing about task quality after healing. A separate public-benchmark receipt covers the surgically edited (not healed) checkpoints of 14 of these models; see the [paper, Section 4.2](#sec-paper-public-benchmarks). Its standalone `secant` CLI cross-check (`read` and `sanitize` on one shard) is `PASS` on 2 of 15 models and `HOLD` on the rest (13: exit `2` or a 120 s timeout in `read`); that cross-check is separate from the healing and calibration numbers above, which are all `PASS`.
