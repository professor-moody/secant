"""Project-Then-Heal LoRA for torch.

A LoRA update to a residual-stream writer is dW = scale * B A with
B in R^{d_out x r}. Let D (k x d_out, orthonormal rows) span the protected
OUTPUT directions. The module trains an unconstrained B_tilde and uses

    B_locked = P (P B_tilde),   P = I - D^T D,

so that D B_locked = 0 up to rounding and therefore D dW = 0: the update
cannot move the protected directions of the layer's output, by construction,
at every step of training (not only after a post-hoc projection).

Cost argument. P is never formed (a [d_out, d_out] matrix would cost
O(d_out^2) memory). P B is evaluated as B - D^T (D B): a [k, r] product and a
[d_out, k] x [k, r] product, i.e. O(k * d_out * r) FLOPs and O(r * (k + d_out))
extra memory, once per forward call. Applying it twice doubles that constant.
Per token the LoRA branch then costs O(r * (d_in + d_out)) (x A^T, then
z B_locked^T), the same as an ordinary LoRA. So the whole forward is
O(r * (k + d_in + d_out)) memory per call and O(T * r * (d_in + d_out) +
k * d_out * r) FLOPs for T tokens, with nothing of size d_out^2 anywhere
(tests/test_nn_torch.py::test_no_d_out_squared_allocation checks every op
output during forward and backward).

Numerics. B_tilde is promoted to float64 for the projection (D is kept in
float64 always) and the result is cast back to the working dtype. For a
float64 module the realized leakage max|D B_locked| is at the 1e-16 level;
for float32 it is dominated by the final rounding of B_locked to float32,
about 6e-8 * max|B_locked| * (1 + small factor), so it stays below 1e-6 as
long as B entries are O(10) or smaller. For bfloat16/float16 modules the final
cast leaks ~1e-3 relative and `leakage()` will say so; do not use those
dtypes where the lock matters (the Rust CLI stores locked B as F32 for the
same reason).
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError("secant.nn needs torch; install with `pip install secant[torch]`") from exc

from ._linalg import DirectionError, orthonormalize_rows
from ._safetensors import write_safetensors

__all__ = ["ProjectThenHealLinear", "export_lora_adapter", "DirectionError"]

#: load_state_dict refuses a D whose max |D D^T - I| is at or above this.
_LOAD_ORTHONORMALITY_TOL = 1e-9


def _directions_to_basis(directions: Any) -> torch.Tensor:
    if isinstance(directions, (str, os.PathLike)):
        directions = np.load(os.fspath(directions), allow_pickle=False)
    return torch.from_numpy(orthonormalize_rows(directions))


class ProjectThenHealLinear(nn.Module):
    """Frozen linear layer plus a LoRA update locked out of protected output directions.

    Parameters
    ----------
    in_features, out_features, rank:
        Shapes. W is [out_features, in_features]; lora_A is [rank, in_features];
        lora_B_tilde is [out_features, rank].
    directions:
        Protected output directions, a vector or [k, out_features] array/tensor
        (or a path to a .npy file). Orthonormalized with two-pass modified
        Gram-Schmidt in float64 at construction; dependent, zero, NaN/Inf or
        k > out_features input raises `DirectionError` (fail closed).
    alpha:
        LoRA alpha; scale = alpha / rank (default alpha = rank, scale 1).
    bias:
        Whether the BASE layer has a (frozen) bias.
    lora_bias:
        Must be False. A LoRA bias would add an unprojected [d_out] vector to
        the output; there is no LoRA bias parameter at all, and True raises.
    weight, base_bias:
        Optional tensors copied into the frozen base (see `from_linear`).
    b_init_std:
        Std of the normal init of lora_B_tilde; 0.0 (default) is the usual
        zero init. lora_A is initialized as torch.nn.Linear does.
    module_name:
        Default tensor-name stem used by `export_lora`.
    """

    D: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        directions: Any,
        alpha: Optional[float] = None,
        *,
        bias: bool = False,
        lora_bias: bool = False,
        weight: Optional[torch.Tensor] = None,
        base_bias: Optional[torch.Tensor] = None,
        b_init_std: float = 0.0,
        module_name: str = "module",
        dtype: Optional[torch.dtype] = None,
        device: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if lora_bias:
            raise ValueError("lora_bias=True is not supported: a LoRA bias is an unprojected output offset and would leak into the protected directions")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if alpha is None:
            alpha = float(rank)
        if not (math.isfinite(alpha) and alpha > 0):
            raise ValueError("alpha must be a positive finite number")
        if not module_name or module_name.startswith(".") or module_name.endswith("."):
            raise ValueError("module_name must be a non-empty dotted module path")
        factory: Dict[str, Any] = {"dtype": dtype, "device": device}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.module_name = module_name

        basis = _directions_to_basis(directions)
        if basis.shape[1] != self.out_features:
            raise DirectionError(f"directions have dimension {basis.shape[1]} but out_features is {self.out_features}")
        # float64 always, on the module's device; `_apply` keeps it float64.
        self.register_buffer("D", basis.to(device=device))

        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory), requires_grad=False)
        if weight is not None:
            if tuple(weight.shape) != (out_features, in_features):
                raise ValueError(f"weight shape {tuple(weight.shape)} != {(out_features, in_features)}")
            with torch.no_grad():
                self.weight.copy_(weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias or base_bias is not None:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(out_features, **factory), requires_grad=False)
            if base_bias is not None:
                with torch.no_grad():
                    self.bias.copy_(base_bias)
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B_tilde = nn.Parameter(torch.zeros(out_features, rank, **factory))
        if b_init_std > 0:
            nn.init.normal_(self.lora_B_tilde, mean=0.0, std=b_init_std)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        directions: Any,
        alpha: Optional[float] = None,
        **kwargs: Any,
    ) -> "ProjectThenHealLinear":
        """Wrap an existing `nn.Linear`: weight/bias are COPIED and frozen."""
        kwargs.setdefault("dtype", linear.weight.dtype)
        kwargs.setdefault("device", linear.weight.device)
        return cls(
            linear.in_features,
            linear.out_features,
            rank,
            directions,
            alpha,
            weight=linear.weight.detach(),
            base_bias=None if linear.bias is None else linear.bias.detach(),
            **kwargs,
        )

    # ------------------------------------------------------------------ dtype/device handling

    def _apply(self, fn, *args, **kwargs):  # type: ignore[override]
        # `module.to(float32)` would round D to float32 and quietly weaken the
        # lock to ~1e-7. Keep D float64; follow only the device.
        d64 = self._buffers.get("D")
        out = super()._apply(fn, *args, **kwargs)
        if d64 is not None and self._buffers.get("D") is not None:
            self._buffers["D"] = d64.detach().to(device=self._buffers["D"].device, dtype=torch.float64)
        return out

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, *args: Any, **kwargs: Any):  # type: ignore[override]
        result = super().load_state_dict(state_dict, strict, *args, **kwargs)
        err = self.orthonormality_error()
        if not (err < _LOAD_ORTHONORMALITY_TOL):
            raise ValueError(f"loaded D is not orthonormal: max|D D^T - I| = {err:.3e}")
        return result

    # ------------------------------------------------------------------ core math

    def locked_B(self) -> torch.Tensor:
        """B_locked = P (P B_tilde) [d_out, r], float64 accumulation, cast to the working dtype.

        Differentiable w.r.t. lora_B_tilde. D is a buffer and receives no gradient.
        """
        d = self.D
        b = self.lora_B_tilde.to(torch.float64)
        for _ in range(2):  # double projection: second pass removes the first pass's residue
            b = b - d.t() @ (d @ b)  # [d_out,k] @ [k,r]; no [d_out,d_out] matrix
        return b.to(self.lora_B_tilde.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.locked_B()
        dt = x.dtype
        z = F.linear(x, self.lora_A.to(dt))
        lora = F.linear(z, b.to(dt))
        return F.linear(x, self.weight, self.bias) + self.scale * lora

    # ------------------------------------------------------------------ evidence

    @torch.no_grad()
    def leakage(self) -> float:
        """max |D B_locked| in float64, measured on the working-dtype B_locked that forward() uses."""
        return float((self.D @ self.locked_B().to(torch.float64)).abs().max().item())

    @torch.no_grad()
    def orthonormality_error(self) -> float:
        """max |D D^T - I| (float64)."""
        eye = torch.eye(self.D.shape[0], dtype=torch.float64, device=self.D.device)
        return float((self.D @ self.D.t() - eye).abs().max().item())

    @torch.no_grad()
    def merged_delta(self) -> torch.Tensor:
        """scale * B_locked @ A, [d_out, d_in], float64 accumulation, cast to the working dtype.

        This materializes a [d_out, d_in] matrix (the weight shape); it is the
        merge product, not part of forward().
        """
        d = self.scale * (self.locked_B().to(torch.float64) @ self.lora_A.to(torch.float64))
        return d.to(self.lora_B_tilde.dtype)

    @torch.no_grad()
    def merge_into(self, base_weight: Optional[torch.Tensor] = None, *, inplace: bool = False) -> torch.Tensor:
        """Return base_weight + merged_delta() (float64 sum, cast to base_weight's dtype).

        `base_weight=None` uses this module's frozen weight. With `inplace=True`
        the result is written into `base_weight` (which must be a real tensor,
        or this module's own weight when None). Rounding W + dW into a
        low-precision dtype re-introduces leakage ~ rounding step * |D|; measure
        it with `merged_leakage`.
        """
        w = self.weight.data if base_weight is None else base_weight
        if tuple(w.shape) != (self.out_features, self.in_features):
            raise ValueError(f"base_weight shape {tuple(w.shape)} != {(self.out_features, self.in_features)}")
        merged = (w.to(torch.float64) + self.merged_delta().to(device=w.device, dtype=torch.float64)).to(w.dtype)
        if inplace:
            w.copy_(merged)
            return w
        return merged

    @torch.no_grad()
    def merged_leakage(self, base_weight: Optional[torch.Tensor] = None) -> float:
        """Realized leakage of the merge: max |D (W_merged - W)| over the stored dtype."""
        w = self.weight.data if base_weight is None else base_weight
        diff = self.merge_into(w).to(torch.float64) - w.to(torch.float64)
        return float((self.D @ diff).abs().max().item())

    # ------------------------------------------------------------------ export

    def lora_tensors(self, module_name: Optional[str] = None, *, locked: bool = True, dtype: Optional[str] = None) -> Dict[str, np.ndarray]:
        """The `<module>.lora_A.weight` / `<module>.lora_B.weight` arrays `export_lora` writes."""
        name = self.module_name if module_name is None else module_name
        if not name or name.startswith(".") or name.endswith("."):
            raise ValueError("module_name must be a non-empty dotted module path")
        if dtype is None:
            dtype = "float64" if self.lora_B_tilde.dtype == torch.float64 else "float32"
        if dtype not in ("float32", "float64"):
            raise ValueError("export dtype must be float32 or float64 (BF16/F16 would reintroduce ~1e-3 leakage)")
        npdt = np.float32 if dtype == "float32" else np.float64
        with torch.no_grad():
            if locked:
                # Round from the float64 projection directly to the export dtype.
                d = self.D
                b = self.lora_B_tilde.detach().to(torch.float64)
                for _ in range(2):
                    b = b - d.t() @ (d @ b)
            else:
                b = self.lora_B_tilde.detach().to(torch.float64)
            a = self.lora_A.detach().to(torch.float64)
        return {
            f"{name}.lora_A.weight": a.cpu().numpy().astype(npdt),
            f"{name}.lora_B.weight": b.cpu().numpy().astype(npdt),
        }

    def export_lora(
        self,
        path: str,
        module_name: Optional[str] = None,
        *,
        locked: bool = True,
        dtype: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Write a LoRA .safetensors for `secant heal --lora`.

        Tensors: `<module>.lora_A.weight` [r, d_in] and `<module>.lora_B.weight`
        [d_out, r], F32 (F64 for float64 modules), nothing else in the file
        (the CLI refuses stray tensors). `locked=True` (default) writes
        B_locked, i.e. exactly what forward() computes; `locked=False` writes
        the raw B_tilde so `secant heal` performs the projection itself.
        `module_name` should be the real base-model module path (for example
        `base_model.model.model.layers.3.mlp.down_proj`) if the file will be
        merged into a base. alpha is not stored in the tensor file; pass
        `scale` (returned) to `secant heal --scale`. Never overwrites unless asked.
        """
        return export_lora_adapter(path, {self.module_name if module_name is None else module_name: self}, locked=locked, dtype=dtype, overwrite=overwrite)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, "
            f"alpha={self.alpha}, k={self.D.shape[0]}, base_bias={self.bias is not None}"
        )


def export_lora_adapter(
    path: str,
    modules: Mapping[str, ProjectThenHealLinear],
    *,
    locked: bool = True,
    dtype: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Write several modules (name -> module) into one LoRA .safetensors."""
    if not modules:
        raise ValueError("no modules to export")
    tensors: Dict[str, np.ndarray] = {}
    scales = set()
    for name, mod in modules.items():
        for k, v in mod.lora_tensors(name, locked=locked, dtype=dtype).items():
            if k in tensors:
                raise ValueError(f"duplicate tensor name {k}")
            tensors[k] = v
        scales.add(mod.scale)
    meta = {
        "format": "pt",
        "secant.producer": "secant-python",
        "secant.b_is_locked": "true" if locked else "false",
        "secant.scale": ",".join(f"{n}={m.scale!r}" for n, m in sorted(modules.items())),
    }
    write_safetensors(path, tensors, meta, overwrite=overwrite)
    return {
        "path": path,
        "tensors": sorted(tensors),
        "locked": locked,
        "scale": scales.pop() if len(scales) == 1 else None,
        "scales": {n: m.scale for n, m in modules.items()},
    }
