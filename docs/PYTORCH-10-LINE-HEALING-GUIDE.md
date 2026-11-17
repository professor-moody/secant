# Fine-Tune with Protected Output Directions

`ProjectThenHealLinear` is a frozen linear layer plus a trainable LoRA adapter. It projects the adapter away from output directions you supply on every forward pass. This lets you preserve a measured local constraint while adapting a model. It does not guarantee preservation of all safety behavior: direction selection, other layers, storage precision, and held-out behavior tests still matter.

## Choose What to Protect

Supply a vector of length `out_features` or a matrix of shape `[k, out_features]`, as an array, tensor, or `.npy` path. The module orthonormalizes the rows in float64 and rejects zero, dependent, non-finite, or incompatible directions. Use directions measured for the selected layer, for example with the contrastive capture workflow in the [quickstart](quickstart.md). A random direction tests the numerical mechanism only.

`from_linear(linear, rank, directions, alpha=None, **kwargs)` copies and freezes the base weight and bias. It creates trainable `lora_A` and `lora_B_tilde`; `alpha` defaults to `rank`, giving a scale of one. Other model parameters remain your responsibility. Replacing one layer does not freeze the rest of a model.

## The Ten-Line Integration

[Recipe 5](RECIPES.md#recipe-5-fine-tune-and-export-in-ten-lines) contains the ten-line training and plain-layer export. This complete CPU example supplies all inputs, trains for 25 steps, and checks both float32 and float64:

```sh
python3 -m pip install -e './python[torch]'
python3 - <<'PY'
import tempfile
import numpy as np
import torch
from secant.nn import ProjectThenHealLinear

torch.manual_seed(0)
rng = np.random.default_rng(0)
for dtype, tol in [(torch.float32, 1e-6), (torch.float64, 1e-12)]:
    linear = torch.nn.Linear(32, 64, bias=True).to(dtype)
    locked = ProjectThenHealLinear.from_linear(
        linear, rank=8, directions=rng.standard_normal((3, 64)), alpha=16.0,
        module_name="model.layers.15.mlp.down_proj",
    )
    optimizer = torch.optim.AdamW([p for p in locked.parameters() if p.requires_grad], lr=1e-2)
    inputs = torch.randn(16, 32, dtype=dtype)
    targets = torch.randn(16, 64, dtype=dtype)
    for _ in range(25):
        optimizer.zero_grad()
        (locked(inputs) - targets).square().mean().backward()
        optimizer.step()
    assert locked.orthonormality_error() < 1e-12
    assert locked.leakage() < tol
    assert locked.merged_leakage() < tol
    exported = torch.nn.Linear(32, 64, bias=True).to(dtype)
    exported.load_state_dict({"weight": locked.merge_into(), "bias": locked.bias.detach()})
    torch.testing.assert_close(exported(inputs), locked(inputs), rtol=1e-5, atol=1e-6)
    with tempfile.TemporaryDirectory() as tmp:
        meta = locked.export_lora(f"{tmp}/adapter.safetensors")
        print(dtype, "adapter leak", locked.leakage(), "merge leak", locked.merged_leakage(), "scale", meta["scale"])
PY
```

The exported `nn.Linear` uses standard weights and has no adapter operations at inference. `merge_into()` returns a weight tensor, not a layer, which is why the example copies it into `nn.Linear`. Copy the bias as well.

## Read the Checks Correctly

`leakage()` measures the largest protected component of the working-dtype adapter factor. `merged_leakage()` measures the actual stored difference after adding the adapter to the base. Multiplication by the other LoRA factor, adapter scaling, and merge rounding make these different checks. The example's tolerances are acceptance criteria for its small numerical inputs, not universal bounds for arbitrary magnitudes.

The basis remains float64 across dtype conversions. However, lower-precision activations or autocast can round the effective adapter during a forward pass. Use float32 or float64 outside autocast when requiring agreement with `leakage()`. Recheck after merging and after quantization.

Gradient projection alone does not impose the same constraint under AdamW because its per-coordinate scaling can change the update direction. Forward-pass projection constrains the effective adapter regardless of that optimizer rescaling. The [formal guide](project-then-heal.md) retains derivations and measured comparisons with ordinary and gradient-projected LoRA.

## Export an Adapter or Verify a Checkpoint

`export_lora(path, module_name=None, *, locked=True, dtype=None, overwrite=False)` writes the adapter tensors in float32 or float64. Use the real model module name. Pass the returned `scale` to `secant heal --scale` when merging; the CLI does not infer it from PEFT configuration. The file is a tensor export, not a complete PEFT package with an `adapter_config.json`.

```sh
secant heal --lora adapter.safetensors --vectors protected.npy \
  --out adapter-locked.safetensors --base model.safetensors \
  --merge-out model-healed.safetensors --scale 2.0 --tol 1e-5 --json
secant verify-subspace --base model.safetensors --healed model-healed.safetensors \
  --vectors protected.npy --tensors model.layers.15.mlp.down_proj.weight --tol 1e-5 --json
```

Use `2.0` only when it matches your exported scale, such as the example's `alpha=16`, `rank=8`. `verify-subspace` defaults to `1e-5` and measures the maximum absolute projection of the stored weight difference. It accepts Q8_0 and Q4_K exports as described in [recipe 6](RECIPES.md#recipe-6-verify-a-quantized-export). Retain hash brackets for the tool, directions, base, and healed files. A passing selected-layer check still requires held-out behavioral evaluation.

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
   `locked.D` is kept in `float64` even if you call `locked.float()` or `locked.half()`. In `float32`, the final cast `b.to(torch.float32)` has produced measured $\max |D B_{\text{locked}}|$ around $10^{-10}$ to $10^{-7}$ (`< 1e-6`). Avoid storing or merging the locked adapter directly in `bfloat16` (`7` mantissa bits, step size $2^{-7} \approx 7.8 \times 10^{-3}$), because `bfloat16` rounding re-introduces $\sim 10^{-4}$ to $10^{-3}$ subspace leakage; `export_lora()` therefore enforces `dtype in ("float32", "float64")`.

</details>

---

## Further Reading

- **[60-Second Hands-On Test Drive](../examples/test-drive/README.md)**: Run the self-contained CLI demo in a temporary directory.
- **[Offensive Researcher's Guide](OFFENSIVE-RESEARCHERS-GUIDE.md)**: Generate synthetic weight-space and quantization-skew fixtures to benchmark model scanners.
- **[Full Project-Then-Heal Benchmarks & Paper Tables](project-then-heal.md)**: Complete 15-model H100 leakage trajectories, 6-model RMS-relative healing benchmarks, and prior-art comparisons (*Safe LoRA*, *SaLoRA*, *OPLoRA*, *GOAL*).
