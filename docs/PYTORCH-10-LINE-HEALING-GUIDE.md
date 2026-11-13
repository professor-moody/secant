# How to Fine-Tune with `ProjectThenHealLinear` in 10 Lines of PyTorch Without Losing Safety Alignment

This guide shows how to fine-tune a PyTorch model using `secant.nn.ProjectThenHealLinear` so your LoRA adapter cannot overwrite protected safety directions or previously applied weight edits.

---

## 1. The Concrete Failure Mode: Why Benign Fine-Tuning Breaks Guardrails

When you fine-tune an aligned language model on as few as 10 to 100 benign customer-support or formatting examples, standard LoRA updates rotate weights across the model's safety-refusal and reasoning directions (Qi et al., ICLR 2024). Even if your training dataset contains zero harmful examples, the adapter adds components back into the protected output subspace and silently degrades guardrails or undoes prior surgical weight edits.

A common workaround is to project the backward gradient (`grad_B`) before calling `optimizer.step()`. Under **AdamW**, gradient projection still leaks:
- AdamW divides each coordinate of the gradient's first moment by its running per-coordinate root-mean-square variance (`sqrt(v_hat_t) + eps`).
- Scaling coordinates by different numbers tilts an orthogonal vector back into the forbidden subspace (`3.6%` to `4.7%` leakage formalized by Chen et al., *GOAL*, Oct 2026; `0.33%` to `2.98%` relative leakage measured across Secant's 15-model H100 and 3-seed AdamW benchmarks).

**How `ProjectThenHealLinear` solves this**: Instead of relying on the optimizer to preserve gradient angles, `ProjectThenHealLinear` projects the LoRA output matrix $B$ **inside the forward pass** twice in `float64` before every matrix multiplication. No matter how AdamW updates the underlying parameter `lora_B_tilde`, the effective adapter matrix `locked_B()` stays orthogonal to your protected directions down to floating-point rounding (`< 1e-6` in `float32` and `< 1e-12` in `float64`).

---

## 2. The 10-Line PyTorch Integration

Install the local Python package:

```bash
pip install -e "./python[torch]"
```

Here is the complete 10-line integration that wraps an existing `torch.nn.Linear` writer layer (`model.layers[15].mlp.down_proj`), runs an AdamW training step, and verifies the subspace lock:

```python
import torch
from secant.nn import ProjectThenHealLinear

# 1. Wrap the target nn.Linear layer (base weight & bias are copied and frozen)
locked = ProjectThenHealLinear.from_linear(
    linear_layer, rank=16, directions=protected_dirs, alpha=16.0
)
model.layers[15].mlp.down_proj = locked

# 2. Train only the LoRA parameters (lora_A and lora_B_tilde) with standard AdamW
optimizer = torch.optim.AdamW([p for p in locked.parameters() if p.requires_grad], lr=5e-4)
loss = loss_fn(model(inputs), targets)
loss.backward()
optimizer.step()

# 3. Verify that worst-case subspace leakage max|D @ B_locked| stays below tolerance
assert locked.leakage() < 1e-6  # < 1e-6 for float32 modules; < 1e-12 for float64 modules
```

### Line-by-Line Plain-English Breakdown

1. **`from secant.nn import ProjectThenHealLinear`**: Imports the subspace-locked LoRA module (`python/secant/nn.py`).
2. **`ProjectThenHealLinear.from_linear(linear_layer, rank=16, directions=protected_dirs, alpha=16.0)`**:
   - Copies `linear_layer.weight` (`[out_features, in_features]`) and `linear_layer.bias` into frozen parameters (`requires_grad=False`).
   - Takes `directions` (`[k, out_features]` NumPy array, PyTorch tensor, or `.npy` file path), orthonormalizes its rows in `float64` via two-pass modified Gram-Schmidt, and stores the orthonormal basis as a non-trainable `float64` buffer `locked.D`. If the supplied directions are linearly dependent, zero, or contain `NaN`/`Inf`, construction fails closed with `DirectionError`.
   - Creates two trainable parameters: `locked.lora_A` (`[rank, in_features]`, Kaiming-uniform initialized) and `locked.lora_B_tilde` (`[out_features, rank]`, zero-initialized by default so step-0 output is identical to the base layer).
3. **`model.layers[15].mlp.down_proj = locked`**: Replaces the target MLP or attention output projection layer in your model with the wrapped layer.
4. **`optimizer = torch.optim.AdamW([p for p in locked.parameters() if p.requires_grad], lr=5e-4)`**: Passes only the trainable adapter weights (`lora_A` and `lora_B_tilde`) to standard PyTorch `AdamW`.
5. **`loss = loss_fn(model(inputs), targets)`**: During `locked.forward(x)`, the layer promotes `lora_B_tilde` to `float64`, removes any component along `locked.D` twice (`b = b - D.T @ (D @ b)`), casts `locked_B()` back to the working dtype, and adds `scale * F.linear(F.linear(x, lora_A), locked_B())` to the frozen base output.
6. **`loss.backward()` and `optimizer.step()`**: Backpropagates gradients through the projection operator and updates `lora_A` and `lora_B_tilde`.
7. **`assert locked.leakage() < 1e-6`**: Computes $\max |D \, B_{\text{locked}}|$ in `float64` on the working-dtype `locked_B()` matrix to confirm that zero protected-subspace drift occurred.

---

## 3. Runnable Self-Contained Verification Script (`float32` & `float64`)

Copy and run this standalone script to test 25 steps of AdamW training in both `float32` (`< 1e-6`) and `float64` (`< 1e-12`), merge the trained adapter into the base weight matrix, and export a `.safetensors` adapter:

```bash
uv run --offline --with torch --with numpy --with ./python python3 - <<'PY'
import tempfile
import numpy as np
import torch
from torch import nn
from secant.nn import ProjectThenHealLinear

torch.manual_seed(0)
rng = np.random.default_rng(0)

for dtype, tol in [(torch.float32, 1e-6), (torch.float64, 1e-12)]:
    # Create a 64x32 linear layer and 3 protected output directions of length 64
    base_linear = nn.Linear(32, 64, bias=False).to(dtype)
    protected_dirs = rng.standard_normal((3, 64))

    locked = ProjectThenHealLinear.from_linear(
        base_linear,
        rank=8,
        directions=protected_dirs,
        alpha=16.0,
        b_init_std=0.1,
        module_name="model.layers.15.mlp.down_proj",
    )

    opt = torch.optim.AdamW([p for p in locked.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(16, 32, dtype=dtype)
    target = torch.randn(16, 64, dtype=dtype)

    for _ in range(25):
        opt.zero_grad()
        loss = (locked(x) - target).pow(2).mean()
        loss.backward()
        opt.step()

    adapter_leak = locked.leakage()
    merge_leak = locked.merged_leakage()
    ortho_err = locked.orthonormality_error()
    print(
        f"dtype={str(dtype):13s} | orthonorm_err={ortho_err:.2e} "
        f"| adapter_leak={adapter_leak:.2e} | merged_leak={merge_leak:.2e} (tol={tol:.0e})"
    )
    assert ortho_err < 1e-12
    assert adapter_leak < tol
    assert merge_leak < tol

    # Export to a PEFT-compatible .safetensors file
    with tempfile.TemporaryDirectory() as tmp:
        meta = locked.export_lora(f"{tmp}/adapter.safetensors")
        print(f"  exported {meta['path']} (scale={meta['scale']}, tensors={meta['tensors']})")
PY
```

### Verifying an Exported or External LoRA Adapter from the CLI

If you already trained a standard `.safetensors` LoRA adapter (or want an independent CLI receipt before deployment), use `secant heal` and `secant verify-subspace`:

```bash
# 1. Project an existing LoRA adapter's lora_B weights out of the protected subspace
#    and optionally merge into a copy of the base checkpoint
secant heal \
  --lora adapter_model.safetensors \
  --vectors vectors/ \
  --out adapter_locked.safetensors \
  --base model.safetensors \
  --merge-out model_healed.safetensors \
  --scale 2.0 \
  --tol 1e-5 \
  --json

# 2. Independently verify that (W_healed - W_base) has <= 1e-5 projection onto vectors/
secant verify-subspace \
  --base model.safetensors \
  --healed model_healed.safetensors \
  --vectors vectors/ \
  --tensors "model.layers.15.mlp.down_proj.weight" \
  --tol 1e-5 \
  --json
```

---

## 4. Technical Details & Precision Guardrails (Collapsible Reference)

<details>
<summary><strong>Why Double Projection in <code>float64</code> Is Used and Why <code>O(d_out^2)</code> Allocation Is Avoided</strong></summary>

<br/>

Given an orthonormal row basis $D \in \mathbb{R}^{k \times d_{\text{out}}}$ ($D D^\top = I_k$), the orthogonal complement projector is $P_\perp = I_{d_{\text{out}}} - D^\top D$.

1. **Skinny Matrix Factoring ($O(k \cdot d_{\text{out}} \cdot r)$ FLOPs)**:
   Forming $I_{d_{\text{out}}} - D^\top D$ explicitly for $d_{\text{out}} = 8192$ would allocate a `512 MiB` `float64` matrix on every forward pass. Instead, `locked_B()` evaluates:
   $$B^{(1)} = \tilde{B} - D^\top (D \tilde{B}), \qquad B_{\text{locked}} = B^{(1)} - D^\top (D B^{(1)})$$
   Here $D \tilde{B}$ has shape $[k, r]$ (for example, $4 \times 16$) and $D^\top (D \tilde{B})$ has shape $[d_{\text{out}}, r]$. Tested with `TorchDispatchMode` in `python/tests/test_nn_torch.py::test_cost_structure_no_d_out_squared_allocation`, zero tensors of size $d_{\text{out}} \times d_{\text{out}}$ are ever allocated during forward or backward passes.
2. **Numerical Effect of the Second Pass**:
   In floating-point arithmetic, a single Gram-Schmidt subtraction $\tilde{B} - D^\top (D \tilde{B})$ can leave a small cancellation residue when $\tilde{B}$ has a large component along $D$. Across 120 seeded Gaussian test cases ($d_{\text{out}} \in \{1024..8192\}$, $k \in \{3, 4\}$, $r \in \{8, 16\}$), applying the projection a second time lowers $\max |D B_{\text{locked}}|$ by a median of `5.6x` (down to $\le 2.24 \times 10^{-15}$ in `float64`).
3. **Working Dtype & `bfloat16` Merge Warning**:
   `locked.D` is kept in `float64` even if you call `locked.float()` or `locked.half()`. In `float32`, the final cast `b.to(torch.float32)` bounds $\max |D B_{\text{locked}}|$ around $10^{-10}$ to $10^{-7}$ (`< 1e-6`). Avoid storing or merging the locked adapter directly in `bfloat16` (`7` mantissa bits, step size $2^{-7} \approx 7.8 \times 10^{-3}$), because `bfloat16` rounding re-introduces $\sim 10^{-4}$ to $10^{-3}$ subspace leakage; `export_lora()` therefore enforces `dtype in ("float32", "float64")`.

</details>

---

## Further Reading

- **[60-Second Hands-On Test Drive](../examples/test-drive/README.md)**: Run the self-contained CLI demo in a temporary directory.
- **[Offensive Researcher's Guide](OFFENSIVE-RESEARCHERS-GUIDE.md)**: Generate synthetic weight-space and quantization-skew fixtures to benchmark model scanners.
- **[Full Project-Then-Heal Benchmarks & Paper Tables](project-then-heal.md)**: Complete 15-model H100 leakage trajectories, 6-model RMS-relative healing benchmarks, and prior-art comparisons (*Safe LoRA*, *SaLoRA*, *OPLoRA*, *GOAL*).
