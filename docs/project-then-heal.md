# Project-Then-Heal: Subspace-Locked LoRA

Suppose you have removed an unwanted behavior, such as `<think>` looping (`B1`) or false-positive refusals (`B2`), via closed-form weight surgery ($\tilde{W}_\ell = W_\ell + \Delta W_\ell$), and now you want to fine-tune the model with LoRA.

In standard LoRA (`unconstrained_lora`), the optimizer has no awareness of the orthonormal row basis $D = [q_1, \dots, q_k]^\top \in \mathbb{R}^{k \times d_{\text{out}}}$ ($D D^\top = I_k$) you just carved out. Nothing prevents the adapter from writing along $D$: any task gradient with a component along $D$ moves the adapter there, and (Section 1) even a projected gradient can be rotated back into $D$ by AdamW's coordinatewise scaling.

**Project-Then-Heal** (`secant.nn.ProjectThenHealLinear` and `secant heal`) solves this structurally by parameterizing the LoRA output matrix $B$ inside the **orthogonal complement** $P_\perp = I_{d_{\text{out}}} - D^\top D$ on every forward pass (measurements in Section 2 and [](#fig-pth-leakage-comparison)).

```{figure} assets/project-then-heal-chart.svg
:name: fig-pth-leakage-comparison
:alt: Regime C H100 stress test on 14 checkpoints, unconstrained adapter fraction in span(D) vs ProjectThenHealLinear absolute leakage
:width: 100%

H100 stress test (14 checkpoints, `7.2B` to `72.7B`) at step 60. The training target lies inside $\operatorname{span}(D)$, so the unconstrained bars ($\|DB\|_{\mathrm{F}}/\|B\|_{\mathrm{F}}$, `8.42%` to `52.55%`) measure that task's pull, not erasure in real training, and the locked arm cannot fit the target (its loss sits at the floor $2.25/d_{\text{model}}$). `ProjectThenHealLinear` absolute leakage $\max|D B_{\text{locked}}|$ is $4.82 \times 10^{-10}$ to $1.08 \times 10^{-9}$ (`float64`) and $3.87 \times 10^{-10}$ to $3.98 \times 10^{-8}$ (merged `F32`). The two columns are different metrics. Live fine-tuning results are in Section 2.
```

---

## 1. Why Structural Parameterization Beats Gradient Hooks

A standard LoRA adapter on a writer matrix has the form $\Delta W = s \, B \, A$, where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$. To guarantee zero leakage into the surgical basis $D \in \mathbb{R}^{k \times d_{\text{out}}}$ ($D D^\top = I_k$), we require:

$$
D \, \Delta W = 0
$$

Why not simply multiply the backward gradient $\nabla_B \mathcal{L}$ by $P_\perp = I_{d_{\text{out}}} - D^\top D$?
1. **Unprojected initialization leaks immediately:** If $B_0$ has any component in $\mathrm{span}(D^\top)$, projecting gradients never removes it.
2. **AdamW coordinatewise scaling rotates steps out of $\mathcal{S}^\perp$:** Even if $\nabla_B \mathcal{L}$ lies inside $\mathcal{S}^\perp$, AdamW divides each coordinate by its own running second moment $\sqrt{v_t} + \varepsilon$, which is a diagonal (non-isotropic) rescaling that rotates the update step back into $\mathrm{span}(D^\top)$. Section 3 tests this directly with a gradient-projected arm.

Instead, `ProjectThenHealLinear` defines the effective output matrix $B_{\text{locked}}(\tilde{B})$ **structurally inside the forward pass** via double orthogonal projection:

$$
B_{\text{locked}}(\tilde{B}) = P_\perp\!\left(P_\perp(\tilde{B})\right) = \left(\tilde{B} - D^\top (D \tilde{B})\right) - D^\top \!\left(D \left(\tilde{B} - D^\top (D \tilde{B})\right)\right)
$$

and computes the forward pass as:

$$
y = x \tilde{W}^\top + \frac{\alpha_{\text{lora}}}{r} (x A^\top) B_{\text{locked}}(\tilde{B})^\top
$$

### Three Algorithmic Advantages

1. **Zero $O(d_{\text{out}}^2)$ Matrix Allocation:** `ProjectThenHealLinear` never materializes the dense $d_{\text{out}} \times d_{\text{out}}$ matrix $I - D^\top D$. Each projection executes two skinny matrix multiplications $D \tilde{B} \in \mathbb{R}^{k \times r}$ and $D^\top (D \tilde{B}) \in \mathbb{R}^{d_{\text{out}} \times r}$ in $O(k \cdot d_{\text{out}} \cdot r)$ time (`2.26 ms` per step on AMD Radeon 8060S ROCm GPU, a **`15.26x` speedup** over CPU).
2. **Why Double Projection ("Twice Is Enough"):** In exact arithmetic $P_\perp^2 = P_\perp$, so a second projection changes nothing. In `float64` the second pass is an **observed numerical refinement**, not a proved error bound: we do not claim a $\kappa(\tilde{B}) \cdot \varepsilon_{\text{mach}}$ theorem. Measured with the shipped `secant._linalg.project_out` on 120 seeded Gaussian cases ($d_{\text{out}} \in \{1024, 2560, 3584, 4096, 5120, 8192\}$, $k \in \{3, 4\}$, $r \in \{8, 16\}$, 5 seeds), the second pass lowered $\max_{i,j} |(D B_{\text{locked}})_{i,j}|$ by $1.8\times$ to $17.7\times$ (median $5.6\times$), from at most $2.25 \times 10^{-14}$ to at most $2.24 \times 10^{-15}$. After 60 steps of live training on `Qwen3.5-0.8B` we measured $9.87 \times 10^{-18}$ in `float64` and $2.49 \times 10^{-10}$ in the merged `F32` delta (Section 2).
3. **Storage Precision Gate (`F32` vs. `BF16` Merge):** Because `BF16` has only 7 mantissa bits ($\mathrm{ulp} \approx 7.8 \times 10^{-3}$), rounding $\tilde{W} + \Delta W$ into `BF16` introduced $1.3 \times 10^{-4}$ to $9.5 \times 10^{-4}$ of leakage along $D$ for the locked arm in our live runs (`Qwen3.5-0.8B`, `SmolLM2-1.7B`), comparable to the unconstrained arm on one `SmolLM2` module. `secant heal` therefore stores $B_{\text{locked}}$ in `F32`/`F64`, and `secant verify-subspace` fails closed if a merged tensor exceeds `--tol 1e-5`.

---

## 2. Step-by-Step Trajectory on `Qwen3.5-0.8B` (`Step 0` to `Step 60`)

Attaching `ProjectThenHealLinear` ($r=8, \alpha_{\text{lora}}=16, k=4$ protected directions, AdamW $\text{lr} = 5 \times 10^{-4}$) to layers `15, 16, 17` of surgically edited `Qwen3.5-0.8B` alongside a bit-identically initialized `unconstrained_lora` control over 60 steps of real causal LM fine-tuning gives near-identical training loss (`0.3120` vs. `0.3096` at step 60) and a factor of about $4.2 \times 10^{7}$ lower merged `F32` writer leakage ($2.49 \times 10^{-10}$ vs. $1.05 \times 10^{-2}$), measured with the same metric in both arms. Note that on this 24-example batch, held-out perplexity reaches its minimum at Step 20 (`5.435` vs. `5.423`) before both arms overfit the small training batch by Step 60 (`12.106` vs. `12.357`):

| Metric (`Qwen3.5-0.8B` Live AdamW Training) | Step 0 | Step 10 | Step 20 (Min PPL) | Step 40 | Step 60 (Final) |
|---|---:|---:|---:|---:|---:|
| **Train Loss (`project_then_heal`)** | `2.8242` | `2.0627` | `1.6182` | `0.9485` | **`0.3120` (`-89.0%`)** |
| **Train Loss (`unconstrained_lora`)** | `2.8242` | `2.0599` | `1.6180` | `0.9459` | **`0.3096` (`-89.0%`)** |
| **Held-Out Perplexity (`project_then_heal`)** | `17.568` | `7.334` | **`5.435`** | `5.845` | `12.106` |
| **Held-Out Perplexity (`unconstrained_lora`)** | `17.568` | `7.315` | **`5.423`** | `5.884` | `12.357` |
| **`float64` Max $\Vert D \Delta W\Vert $ (`project_then_heal`)** | `0.00e+00` | `3.63e-18` | `5.53e-18` | `7.97e-18` | **`9.87e-18`** |
| **Merged `F32` Max $\Vert D \Delta W\Vert $ (`project_then_heal`)** | `0.00e+00` | `6.54e-11` | `1.22e-10` | `1.96e-10` | **`2.49e-10` (`PASS`)** |
| **Merged `F32` Max $\Vert D \Delta W\Vert $ (`unconstrained_lora`)** | `0.00e+00` | `2.60e-03` | `4.46e-03` | `7.98e-03` | **`1.05e-02` (`FAIL`)** |

---

## 3. Three-Arm Post-Surgical LoRA + AdamW Rotation Test

Section 1 argues that projecting only the gradient is not enough under AdamW. This section tests that argument with an arm that does exactly that.

**Setup.** Writer: the real layer-16 `mlp.down_proj` weight of `Qwen3.5-0.8B` ($d_{\text{out}} = 1024$), `Qwen3.5-4B` ($2560$) and `Qwen2.5-7B` ($3584$), surgically edited as $W_e = P_\perp W_0$ so that $D W_e = 0$. $D$ has $k = 3$ seeded random orthonormal `float64` rows. The task is **synthetic regression**, not language-model fine-tuning: $T = 256$ Gaussian inputs, an *orthogonal* target $Y = W_e X + R_\perp$ with $D R_\perp = 0$, and a *mixed* target that adds $R_D \in \operatorname{span}(D)$ with energy `0.25` of $R_\perp$ (so `0.25` is the analytic loss floor of any arm that cannot write into $\operatorname{span}(D)$). LoRA rank `16`, scale `1`, AdamW betas `(0.9, 0.999)`, eps `1e-8`, 60 full-batch steps, seeds `0, 1, 2`, identical $A_0$ and $B_0 = 0$ in every arm.

- **Arm A:** unconstrained LoRA + AdamW.
- **Arm B:** gradient-projected LoRA, $\nabla_B \leftarrow P_\perp \nabla_B$ in `float64` immediately before each AdamW step.
- **Arm C:** `secant.nn.ProjectThenHealLinear`, $B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$ in the forward pass, same AdamW.

Every arm is scored with the **same** metric: $\mathrm{lk}_W = \|D\,\Delta W\|_{\mathrm{F}} / \|\Delta W\|_{\mathrm{F}}$ (relative leakage of the LoRA delta into $\operatorname{span}(D)$), plus the output-energy fraction in $\operatorname{span}(D)$ and the largest $|D\,\Delta W|$ of the `float32`-stored merged delta. Weight decay is run at both `0.01` and `0.0`: decoupled decay multiplies $B$ by a scalar and commutes with $P_\perp$, so it should not matter.

**Learning rate: this table is post-hoc.** The preregistered learning rate (`1e-2`, chosen once on Arm A) failed the preregistered loss-parity prediction for Arm C on `Qwen2.5-7B`. The `3e-3` run below was added **after** seeing that result and is reported beside, not instead of, the preregistered run. The preregistered `1e-2` run gives the same ordering: on the orthogonal target, Arm B leaks `1.24%` to `2.58%` at step 60 and Arm C $6.40 \times 10^{-10}$ to $1.12 \times 10^{-9}$.

| Model ($d_{\text{out}}$) | Target | Arm | wd | Final loss (min..max, 3 seeds) | Final loss minus floor (mean) | $\mathrm{lk}_W$ step 1 (mean) | $\mathrm{lk}_W$ step 60 (mean) | $\max_t \mathrm{lk}_W$ | Output energy in $\operatorname{span}(D)$ at step 60 (mean) | Merged `F32` $\max \lvert D\,\Delta W\rvert$ (max) |
|---|---|---|---|---|---|---|---|---|---|---|
| `Qwen3.5-0.8B` (1024) | orth | A | 0.01 | 0.00122..0.00128 | +0.00125 | 3.29e-02 | 8.18e-03 | 3.47e-02 | 2.86e-06 | 3.95e-03 |
| `Qwen3.5-0.8B` (1024) | orth | B | 0.01 | 0.00190..0.00219 | +0.00208 | 3.29e-02 | **2.98e-02** | 3.50e-02 | 8.22e-04 | 1.08e-02 |
| `Qwen3.5-0.8B` (1024) | orth | C | 0.01 | 0.00122..0.00128 | +0.00125 | 1.14e-09 | **1.05e-09** | 1.62e-09 | 9.48e-19 | 1.60e-09 |
| `Qwen3.5-0.8B` (1024) | orth | A | 0.0 | 0.00123..0.00128 | +0.00126 | 3.29e-02 | 8.19e-03 | 3.47e-02 | 2.86e-06 | 3.95e-03 |
| `Qwen3.5-0.8B` (1024) | orth | B | 0.0 | 0.00190..0.00219 | +0.00208 | 3.29e-02 | **2.98e-02** | 3.50e-02 | 8.23e-04 | 1.08e-02 |
| `Qwen3.5-0.8B` (1024) | orth | C | 0.0 | 0.00123..0.00128 | +0.00125 | 1.14e-09 | **1.02e-09** | 1.60e-09 | 8.92e-19 | 1.77e-09 |
| `Qwen3.5-0.8B` (1024) | mixed | A | 0.01 | 0.00187..0.00272 | -0.24775 | 3.13e-01 | 4.54e-01 | 4.62e-01 | 1.99e-01 | 1.94e-01 |
| `Qwen3.5-0.8B` (1024) | mixed | B | 0.01 | 0.24301..0.24621 | -0.00522 | 3.29e-02 | 3.06e-02 | 3.50e-02 | 8.84e-04 | 1.18e-02 |
| `Qwen3.5-0.8B` (1024) | mixed | C | 0.01 | 0.25122..0.25128 | +0.00125 | 1.36e-09 | 1.11e-09 | 1.82e-09 | 1.06e-18 | 1.58e-09 |
| `Qwen3.5-0.8B` (1024) | mixed | A | 0.0 | 0.00188..0.00273 | -0.24774 | 3.13e-01 | 4.54e-01 | 4.62e-01 | 1.99e-01 | 1.94e-01 |
| `Qwen3.5-0.8B` (1024) | mixed | B | 0.0 | 0.24301..0.24621 | -0.00522 | 3.29e-02 | 3.06e-02 | 3.50e-02 | 8.84e-04 | 1.18e-02 |
| `Qwen3.5-0.8B` (1024) | mixed | C | 0.0 | 0.25123..0.25128 | +0.00125 | 1.36e-09 | 1.08e-09 | 1.59e-09 | 1.04e-18 | 1.76e-09 |
| `Qwen3.5-4B` (2560) | orth | A | 0.01 | 0.00085..0.00094 | +0.00090 | 2.02e-02 | 6.10e-03 | 2.19e-02 | 3.89e-07 | 1.70e-03 |
| `Qwen3.5-4B` (2560) | orth | B | 0.01 | 0.00109..0.00127 | +0.00117 | 2.02e-02 | **1.74e-02** | 2.23e-02 | 2.65e-04 | 4.44e-03 |
| `Qwen3.5-4B` (2560) | orth | C | 0.01 | 0.00085..0.00094 | +0.00090 | 7.94e-10 | **7.71e-10** | 1.17e-09 | 5.30e-19 | 7.06e-10 |
| `Qwen3.5-4B` (2560) | orth | A | 0.0 | 0.00086..0.00094 | +0.00090 | 2.02e-02 | 6.11e-03 | 2.19e-02 | 3.89e-07 | 1.70e-03 |
| `Qwen3.5-4B` (2560) | orth | B | 0.0 | 0.00110..0.00128 | +0.00117 | 2.02e-02 | **1.74e-02** | 2.23e-02 | 2.65e-04 | 4.45e-03 |
| `Qwen3.5-4B` (2560) | orth | C | 0.0 | 0.00086..0.00094 | +0.00090 | 7.94e-10 | **7.33e-10** | 1.25e-09 | 4.74e-19 | 6.87e-10 |
| `Qwen3.5-4B` (2560) | mixed | A | 0.01 | 0.00133..0.00152 | -0.24857 | 3.92e-01 | 4.52e-01 | 5.24e-01 | 2.00e-01 | 1.20e-01 |
| `Qwen3.5-4B` (2560) | mixed | B | 0.01 | 0.23067..0.24065 | -0.01342 | 2.02e-02 | 2.36e-02 | 2.58e-02 | 6.03e-04 | 9.28e-03 |
| `Qwen3.5-4B` (2560) | mixed | C | 0.01 | 0.25085..0.25094 | +0.00090 | 7.96e-10 | 6.84e-10 | 1.05e-09 | 3.84e-19 | 7.51e-10 |
| `Qwen3.5-4B` (2560) | mixed | A | 0.0 | 0.00134..0.00152 | -0.24857 | 3.92e-01 | 4.52e-01 | 5.24e-01 | 2.00e-01 | 1.20e-01 |
| `Qwen3.5-4B` (2560) | mixed | B | 0.0 | 0.23064..0.24064 | -0.01344 | 2.02e-02 | 2.37e-02 | 2.58e-02 | 6.04e-04 | 9.29e-03 |
| `Qwen3.5-4B` (2560) | mixed | C | 0.0 | 0.25086..0.25094 | +0.00090 | 7.96e-10 | 6.91e-10 | 1.04e-09 | 4.27e-19 | 7.03e-10 |
| `Qwen2.5-7B` (3584) | orth | A | 0.01 | 0.00075..0.00077 | +0.00076 | 1.94e-02 | 4.94e-03 | 2.24e-02 | 4.70e-07 | 8.84e-04 |
| `Qwen2.5-7B` (3584) | orth | B | 0.01 | 0.00098..0.00098 | +0.00098 | 1.94e-02 | **1.56e-02** | 2.24e-02 | 2.22e-04 | 2.07e-03 |
| `Qwen2.5-7B` (3584) | orth | C | 0.01 | 0.00075..0.00077 | +0.00076 | 6.37e-10 | **5.42e-10** | 8.52e-10 | 2.38e-19 | 3.81e-10 |
| `Qwen2.5-7B` (3584) | orth | A | 0.0 | 0.00075..0.00077 | +0.00076 | 1.94e-02 | 4.95e-03 | 2.24e-02 | 4.71e-07 | 8.86e-04 |
| `Qwen2.5-7B` (3584) | orth | B | 0.0 | 0.00098..0.00098 | +0.00098 | 1.94e-02 | **1.56e-02** | 2.24e-02 | 2.23e-04 | 2.07e-03 |
| `Qwen2.5-7B` (3584) | orth | C | 0.0 | 0.00075..0.00077 | +0.00076 | 6.37e-10 | **6.05e-10** | 8.07e-10 | 3.25e-19 | 3.85e-10 |
| `Qwen2.5-7B` (3584) | mixed | A | 0.01 | 0.00102..0.00112 | -0.24893 | 3.70e-01 | 4.60e-01 | 5.05e-01 | 2.01e-01 | 8.44e-02 |
| `Qwen2.5-7B` (3584) | mixed | B | 0.01 | 0.24067..0.24308 | -0.00773 | 1.94e-02 | 1.84e-02 | 2.24e-02 | 3.57e-04 | 3.83e-03 |
| `Qwen2.5-7B` (3584) | mixed | C | 0.01 | 0.25075..0.25077 | +0.00076 | 6.04e-10 | 5.98e-10 | 8.39e-10 | 2.99e-19 | 3.94e-10 |
| `Qwen2.5-7B` (3584) | mixed | A | 0.0 | 0.00102..0.00113 | -0.24893 | 3.70e-01 | 4.60e-01 | 5.05e-01 | 2.01e-01 | 8.45e-02 |
| `Qwen2.5-7B` (3584) | mixed | B | 0.0 | 0.24067..0.24307 | -0.00774 | 1.94e-02 | 1.84e-02 | 2.24e-02 | 3.57e-04 | 3.83e-03 |
| `Qwen2.5-7B` (3584) | mixed | C | 0.0 | 0.25075..0.25077 | +0.00076 | 6.04e-10 | 6.07e-10 | 8.10e-10 | 3.27e-19 | 3.82e-10 |

**What the table shows.**

1. **Gradient projection alone does not keep the adapter out of $\operatorname{span}(D)$.** On the orthogonal target, Arm B puts `1.56%` to `2.98%` of $\|\Delta W\|_{\mathrm{F}}$ into $\operatorname{span}(D)$ at step 60 (`1.84%` to `3.06%` on the mixed target), and the value is the same at weight decay `0.0` as at `0.01`. Weight decay is therefore not the source.
2. **The step-1 mechanism check points at the second-moment preconditioner.** At $B_0 = 0$ on the orthogonal target, the loss gradient already lies in $\operatorname{col}(P_\perp)$ ($\|D g\| / \|g\| = 5.1 \times 10^{-9}$ to $1.3 \times 10^{-8}$), so Arm B's projection is a no-op there. AdamW's first update is $u = g / (|g| + \epsilon)$, an elementwise sign-like map, and $\|D u\| / \|u\|$ is `1.76%` to `3.48%` across the 9 model-seed pairs, matching Arm B's recorded step-1 leakage to within $2 \times 10^{-7}$ relative. The leak is created by the coordinatewise division, not by the gradient.
3. **`ProjectThenHealLinear` holds leakage at rounding level with the same final loss as unconstrained LoRA on the orthogonal target.** Arm C stays at $5.42 \times 10^{-10}$ to $1.05 \times 10^{-9}$ at step 60 on the orthogonal target ($5.98 \times 10^{-10}$ to $1.11 \times 10^{-9}$ on the mixed target), and its final-loss range equals Arm A's to the precision shown in every orthogonal row. Arm B, by contrast, ends with a **higher** loss than A and C on the orthogonal target.
4. **On the mixed target, Arm C stops exactly at the floor.** It cannot fit $R_D$ by construction, so its final loss sits at `0.25` plus the same excess it shows on the orthogonal task. Arm A goes below the floor by writing about `45%` of its delta norm into $\operatorname{span}(D)$; that is the behavior the lock exists to prevent.

**Limits.** This is a single-layer synthetic regression with random protected directions, not a measurement on real edit directions or on LM fine-tuning quality; Regimes A and C (Section 2 and the [overview](index.md)) cover those. The learning rate in the table is post-hoc, as stated above. Raw per-seed trajectories, the runner (`adamw3arm.py`) and the mechanism check (`signcheck.py`) are retained with SHA-256 receipts in the lab evidence bank and are not yet published in this repository.
