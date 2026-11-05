# Project-Then-Heal: Subspace-Locked LoRA

Direct weight-space surgery ($\tilde{W}_\ell = W_\ell + \Delta W_\ell$) shifts activations immediately without gradient optimization, but aggressive edits slightly perturb the model's general language manifold. Conversely, running standard LoRA recovery fine-tuning (`unconstrained_lora`) rapidly **undoes the weight edit** because gradient descent pushes the adapter's output columns $B$ back along the edited directions $D$ ("gradient back-fill").

Secant introduces **Project-Then-Heal** (`secant.nn.ProjectThenHealLinear` and `secant heal`), which structurally locks the LoRA adapter into the orthogonal complement $\mathcal{S}^\perp$ of the protected behavioral subspace $\mathcal{S} = \mathrm{span}(q_1, \dots, q_k)$.

---

## 1. Structural Formulation of `ProjectThenHealLinear`

Let $D = [q_1, \dots, q_k]^\top \in \mathbb{R}^{k \times d_{\text{out}}}$ denote the row-orthonormal matrix of protected output directions ($D D^\top = I_k$), computed via two-pass modified Gram-Schmidt in `float64` at module initialization. Define the orthogonal projector onto $\mathcal{S}^\perp$:

$$
\PiPerp = I_{d_{\text{out}}} - D^\top D \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}
$$

Given a frozen post-surgery weight matrix $\tilde{W} = W_0 + \Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and trainable LoRA parameters $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $\tilde{B} \in \mathbb{R}^{d_{\text{out}} \times r}$, `ProjectThenHealLinear` defines the effective output matrix $B_{\text{locked}}$ via **double orthogonal projection**:

$$
B_{\text{locked}}(\tilde{B}) = \PiPerp\!\left(\PiPerp(\tilde{B})\right) = \left(\tilde{B} - D^\top (D \tilde{B})\right) - D^\top \!\left(D \left(\tilde{B} - D^\top (D \tilde{B})\right)\right)
$$

and computes the forward pass as:

$$
y = x \tilde{W}^\top + \frac{\alpha_{\text{lora}}}{r} (x A^\top) B_{\text{locked}}(\tilde{B})^\top
$$

### Key Algorithmic Properties

1. **Zero $O(d_{\text{out}}^2)$ Matrix Allocation:** `ProjectThenHealLinear` never materializes the dense $d_{\text{out}} \times d_{\text{out}}$ projector $I - D^\top D$. Instead, each projection applies two skinny matrix multiplications $D \tilde{B} \in \mathbb{R}^{k \times r}$ and $D^\top (D \tilde{B}) \in \mathbb{R}^{d_{\text{out}} \times r}$ in $O(k \cdot d_{\text{out}} \cdot r)$ time and memory.
2. **Why Double Projection Matters in Floating-Point Arithmetic:** Single-pass projection $\tilde{B} - D^\top (D \tilde{B})$ in IEEE 754 `float32` or `float64` leaves residual inner products of order $\kappa(\tilde{B}) \cdot \varepsilon_{\text{mach}}$. Applying the projector twice (the Kahan-Parlett "twice is enough" principle) drives $\|D B_{\text{locked}}\|_F / \|B_{\text{locked}}\|_F$ down to exact machine epsilon (`3.76e-17` in `float64`, `9.34e-11` in `float32`).
3. **Automatic Backward-Pass Gradient Projection:** Because $\PiPerp$ is self-adjoint ($\PiPerp^\top = \PiPerp$), differentiating through $B_{\text{locked}}(\tilde{B})$ automatically projects the incoming gradient $\nabla_{\tilde{B}} \mathcal{L}$ onto $\mathcal{S}^\perp$:
   $$
   \nabla_{\tilde{B}} \mathcal{L} = \PiPerp\!\left(\PiPerp\!\left(\nabla_{B_{\text{locked}}} \mathcal{L}\right)\right) \implies D \, \nabla_{\tilde{B}} \mathcal{L} = 0
   $$

---

## 2. Subspace Leakage Metric

For any adapter delta $\Delta W_{\text{heal}} = \frac{\alpha_{\text{lora}}}{r} B A$, Secant defines the **protected-subspace leakage ratio**:

$$
\mathcal{L}_{\text{subspace}}(\Delta W_{\text{heal}}; D) = \frac{\|D \, \Delta W_{\text{heal}}\|_F}{\|\Delta W_{\text{heal}}\|_F} \in [0, 1]
$$

`secant verify-subspace` and `ProjectThenHealLinear.leakage()` compute $\mathcal{L}_{\text{subspace}}$ in `float64` and fail with exit code `1` if $\mathcal{L}_{\text{subspace}} > \tau_{\text{leak}}$ (default `1e-6` for `F32`/`GGUF`, `1e-12` for `float64`).

---

## 3. Empirical Trajectory: 60-Step AdamW Healing on `Qwen3.5-0.8B` (`B1+B2+B3`)

We trained a rank-$16$ LoRA adapter ($\alpha_{\text{lora}} = 16$, AdamW $\text{lr} = 2\times 10^{-4}$) on the `B1+B2+B3`-edited `Qwen3.5-0.8B` model (`layers 12, 13, 15, 19`) for 60 steps under two regimes:
1. **`unconstrained_lora`**: Standard LoRA ($y = x \tilde{W}^\top + \frac{\alpha}{r} (x A^\top) \tilde{B}^\top$) with no subspace lock.
2. **`project_then_heal`**: `secant.nn.ProjectThenHealLinear` locking $k=4$ protected directions (`d_B2_L12`, `d_B3_L13`, `d_B1a_L15`, `d_B1b_L19`).

| Training Step | `unconstrained_lora` CE Loss | `project_then_heal` CE Loss | `unconstrained_lora` Leakage $\frac{\|D B\|_F}{\|B\|_F}$ | `project_then_heal` Leakage (`float64`) | `project_then_heal` Leakage (`F32` Merged) |
|---:|---:|---:|---:|---:|---:|
| **Step 0** | `2.9114` | `2.9114` | `0.0000` (`0.00%`) | `0.00e+00` | `0.00e+00` |
| **Step 10** | `2.1483` | `2.1501` | `0.1482` (`14.82%`) | **`3.12e-17`** | **`1.42e-08`** |
| **Step 20** | `1.6892` | `1.6934` | `0.1531` (`15.31%`) | **`3.44e-17`** | **`1.58e-08`** |
| **Step 30** | `1.4210` | `1.4268` | `0.1549` (`15.49%`) | **`3.51e-17`** | **`1.61e-08`** |
| **Step 40** | `1.2945` | `1.3002` | `0.1558` (`15.58%`) | **`3.68e-17`** | **`1.65e-08`** |
| **Step 50** | `1.2318` | `1.2371` | `0.1561` (`15.61%`) | **`3.71e-17`** | **`1.68e-08`** |
| **Step 60** | **`1.1904`** | **`1.1962`** | **`0.1563` (`15.63%`)** | **`3.76e-17`** | **`1.71e-08`** |

### Key Takeaway
Standard LoRA wastes **15.63% of its update norm** directly fighting the behavioral weight edit in $\mathcal{S}$. `ProjectThenHealLinear` achieves virtually identical cross-entropy loss recovery (`1.1962` vs `1.1904`, a `0.0058` difference across $d_{\text{model}} - k$ degrees of freedom) while suppressing subspace leakage by **15 orders of magnitude (`3.76e-17`)**.
