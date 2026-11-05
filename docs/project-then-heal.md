# Project-Then-Heal: Subspace-Locked LoRA

Suppose you have removed an unwanted behavior, such as `<think>` looping (`B1`) or false-positive refusals (`B2`), via closed-form weight surgery ($\tilde{W}_\ell = W_\ell + \Delta W_\ell$), and now you want to fine-tune the model with LoRA.

In standard LoRA (`unconstrained_lora`), the optimizer has no awareness of the orthonormal row basis $D = [q_1, \dots, q_k]^\top \in \mathbb{R}^{k \times d_{\text{out}}}$ ($D D^\top = I_k$) you just carved out. During backpropagation, any task gradient coupling with $D$, combined with AdamW's coordinatewise second-moment scaling, pushes updates into the forbidden directions $D$.

**Project-Then-Heal** (`secant.nn.ProjectThenHealLinear` and `secant heal`) solves this structurally by parameterizing the LoRA output matrix $B$ inside the **orthogonal complement** $P_\perp = I_{d_{\text{out}}} - D^\top D$ on every forward pass.

```{figure} assets/project-then-heal-chart.svg
:name: fig-pth-leakage-comparison
:alt: ProjectThenHealLinear vs Unconstrained LoRA leakage across 11 models from 0.8B to 32.8B
:width: 100%

Empirical comparison across 11 models (`0.8B` to `32.8B`): during 60 AdamW steps, unconstrained LoRA intrudes into the protected subspace ($1.05 \times 10^{-2}$ max $|D \Delta W|$ on live `0.8B` LM fine-tuning, and $8.42\%$ to $52.55\%$ adapter norm intrusion under H100 adversary-aligned subspace stress testing), whereas `ProjectThenHealLinear` holds writer leakage at $9.87 \times 10^{-18}$ to $1.08 \times 10^{-9}$ (`float64`) and $1.46 \times 10^{-10}$ to $1.70 \times 10^{-8}$ (`F32`).
```

---

## 1. Why Structural Parameterization Beats Gradient Hooks

A standard LoRA adapter on a writer matrix has the form $\Delta W = s \, B \, A$, where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$. To guarantee zero leakage into the surgical basis $D \in \mathbb{R}^{k \times d_{\text{out}}}$ ($D D^\top = I_k$), we require:

$$
D \, \Delta W = 0
$$

Why not simply multiply the backward gradient $\nabla_B \mathcal{L}$ by $P_\perp = I_{d_{\text{out}}} - D^\top D$?
1. **Unprojected initialization leaks immediately:** If $B_0$ has any component in $\mathrm{span}(D^\top)$, projecting gradients never removes it (`0.578` leakage in our numerical witness).
2. **AdamW coordinatewise scaling rotates steps out of $\mathcal{S}^\perp$:** Even if $\nabla_B \mathcal{L}$ lies inside $\mathcal{S}^\perp$, AdamW divides each coordinate by its own running second moment $\sqrt{v_t} + \varepsilon$, which is a diagonal (non-isotropic) rescaling that rotates the update step back into $\mathrm{span}(D^\top)$.

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
2. **Why Double Projection Matters ("Twice Is Enough"):** Single-pass projection $\tilde{B} - D^\top (D \tilde{B})$ in finite floating-point precision leaves rounding residuals of order $\kappa(\tilde{B}) \cdot \varepsilon_{\text{mach}}$. Applying $P_\perp$ twice drives writer leakage $\max_{i,j} |(D B_{\text{locked}})_{i,j}|$ down to machine epsilon ($3.76 \times 10^{-17}$ static residual in `float64`, $9.34 \times 10^{-11}$ in `float32`).
3. **Storage Precision Gate (`F32` vs. `BF16` Merge):** Because `BF16` has only 7 mantissa bits ($\mathrm{ulp} \approx 7.8 \times 10^{-3}$), rounding $\tilde{W} + \Delta W$ into `BF16` introduces $\sim 1.4 \times 10^{-4}$ to $1.4 \times 10^{-3}$ rounding noise along $D$. `secant heal` therefore stores $B_{\text{locked}}$ in `F32`/`F64`, and `secant verify-subspace` fails closed if a merged tensor exceeds `--tol 1e-5`.

---

## 2. Step-by-Step Trajectory on `Qwen3.5-0.8B` (`Step 0` to `Step 60`)

Attaching `ProjectThenHealLinear` ($r=8, \alpha_{\text{lora}}=16, k=4$ protected directions, AdamW $\text{lr} = 5 \times 10^{-4}$) to layers `15, 16, 17` of surgically edited `Qwen3.5-0.8B` alongside a bit-identically initialized `unconstrained_lora` control over 60 steps of real causal LM fine-tuning demonstrates identical training loss recovery and an **8-order-of-magnitude reduction in merged `F32` writer leakage** ($2.49 \times 10^{-10}$ vs. $1.05 \times 10^{-2}$). Note that on this 24-example batch, held-out perplexity reaches its minimum at Step 20 (`5.435` vs. `5.423`) before both arms overfit the small training batch by Step 60 (`12.106` vs. `12.357`):

| Metric (`Qwen3.5-0.8B` Live AdamW Training) | Step 0 | Step 10 | Step 20 (Min PPL) | Step 40 | Step 60 (Final) |
|---|---:|---:|---:|---:|---:|
| **Train Loss (`project_then_heal`)** | `2.8242` | `2.0627` | `1.6182` | `0.9485` | **`0.3120` (`-89.0%`)** |
| **Train Loss (`unconstrained_lora`)** | `2.8242` | `2.0599` | `1.6180` | `0.9459` | **`0.3096` (`-89.0%`)** |
| **Held-Out Perplexity (`project_then_heal`)** | `17.568` | `7.334` | **`5.435`** | `5.845` | `12.106` |
| **Held-Out Perplexity (`unconstrained_lora`)** | `17.568` | `7.315` | **`5.423`** | `5.884` | `12.357` |
| **`float64` Max $\|D \Delta W\|$ (`project_then_heal`)** | `0.00e+00` | `3.63e-18` | `5.53e-18` | `7.97e-18` | **`9.87e-18`** |
| **Merged `F32` Max $\|D \Delta W\|$ (`project_then_heal`)** | `0.00e+00` | `6.54e-11` | `1.22e-10` | `1.96e-10` | **`2.49e-10` (`PASS`)** |
| **Merged `F32` Max $\|D \Delta W\|$ (`unconstrained_lora`)** | `0.00e+00` | `2.60e-03` | `4.46e-03` | `7.98e-03` | **`1.05e-02` (`FAIL`)** |
