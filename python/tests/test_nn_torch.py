"""ProjectThenHealLinear. Needs torch; skips with an explicit reason otherwise."""

import os
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="SKIP-NO-TORCH: secant.nn tests need torch")
from torch import nn  # noqa: E402
from torch.utils import _pytree  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from secant._linalg import DirectionError, leakage as np_leakage, orthonormalize_rows, project_out  # noqa: E402
from secant._safetensors import read_safetensors  # noqa: E402
from secant.nn import ProjectThenHealLinear, export_lora_adapter  # noqa: E402


def make(d_in=24, d_out=40, r=4, k=3, dtype=torch.float32, alpha=8.0, seed=0, bias=False, b_std=0.5, **kw):
    g = np.random.default_rng(seed)
    dirs = g.standard_normal((k, d_out)) * 3.0  # deliberately not orthonormal
    torch.manual_seed(seed)
    lin = nn.Linear(d_in, d_out, bias=bias).to(dtype)
    m = ProjectThenHealLinear.from_linear(lin, r, dirs, alpha, b_init_std=b_std, **kw)
    return lin, m, dirs


# ------------------------------------------------------------------ structure


def test_parameters_buffers_and_no_lora_bias():
    lin, m, _ = make(bias=True)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert trainable == {"lora_A", "lora_B_tilde"}
    assert frozen == {"weight", "bias"}  # base weight + optional BASE bias only
    assert [n for n, _ in m.named_buffers()] == ["D"]
    assert m.D.dtype == torch.float64 and not m.D.requires_grad
    assert not any("lora_bias" in n or n.startswith("lora_B_tilde_bias") for n in m.state_dict())
    assert m.lora_A.shape == (4, 24) and m.lora_B_tilde.shape == (40, 4) and m.D.shape == (3, 40)
    _, m2, _ = make(bias=False)
    assert m2.bias is None
    with pytest.raises(ValueError, match="lora_bias"):
        ProjectThenHealLinear(8, 8, 2, np.eye(8)[:1], lora_bias=True)


def test_from_linear_copies_and_freezes():
    lin, m, _ = make(bias=True)
    assert torch.equal(m.weight, lin.weight) and torch.equal(m.bias, lin.bias)
    with torch.no_grad():
        lin.weight.add_(1.0)
    assert not torch.equal(m.weight, lin.weight)  # a copy, not an alias


def test_directions_orthonormalized_at_construction_and_fail_closed():
    _, m, _ = make(k=5, d_out=64)
    assert m.orthonormality_error() < 1e-12
    base = np.random.default_rng(0).standard_normal(16)
    for bad in (np.stack([base, 2 * base]), np.stack([base, np.zeros(16)]), np.where(np.arange(16) == 2, np.nan, 1.0), np.ones((17, 16))):
        with pytest.raises(DirectionError):
            ProjectThenHealLinear(8, 16, 2, bad)
    with pytest.raises(DirectionError, match="dimension"):
        ProjectThenHealLinear(8, 16, 2, np.ones((1, 15)))
    # torch tensors and 1-D vectors are accepted
    ProjectThenHealLinear(8, 16, 2, torch.ones(16))


def test_dtype_cast_does_not_round_D():
    _, m, _ = make()
    d0 = m.D.clone()
    for conv in (lambda x: x.float(), lambda x: x.to(torch.float32), lambda x: x.double(), lambda x: x.to("cpu")):
        m = conv(m)
        assert m.D.dtype == torch.float64 and torch.equal(m.D, d0)
    m.half()
    assert m.D.dtype == torch.float64


def test_state_dict_roundtrip_and_non_orthonormal_load_rejected():
    _, m, _ = make(seed=1)
    _, m2, _ = make(seed=2)
    m2.load_state_dict(m.state_dict())
    assert torch.equal(m2.D, m.D) and torch.equal(m2.lora_B_tilde, m.lora_B_tilde)
    sd = m.state_dict()
    sd["D"] = sd["D"] * 1.01
    with pytest.raises(ValueError, match="not orthonormal"):
        m2.load_state_dict(sd)


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("bad", ["scaled", "nan", "float32"])
def test_parent_checkpoint_load_validates_D_before_installation(assign, bad):
    _, layer, _ = make(seed=1)
    parent = nn.Sequential(nn.Sequential(layer))
    before = layer.D.clone()
    state = parent.state_dict()
    if bad == "scaled":
        state["0.0.D"] = state["0.0.D"] * 1.01
    elif bad == "nan":
        state["0.0.D"] = torch.full_like(state["0.0.D"], float("nan"))
    else:
        state["0.0.D"] = state["0.0.D"].float()
    with pytest.raises(ValueError, match="orthonormal|float64"):
        parent.load_state_dict(state, assign=assign)
    assert torch.equal(layer.D, before)
    assert layer.D.dtype == torch.float64
    assert layer.orthonormality_error() < 1e-12
    _, other, _ = make(seed=2)
    parent.load_state_dict(nn.Sequential(nn.Sequential(other)).state_dict(), assign=assign)
    assert torch.equal(layer.D, other.D) and layer.D.dtype == torch.float64
    layer.float()
    assert layer.D.dtype == torch.float64 and layer.orthonormality_error() < 1e-12


# ------------------------------------------------------------------ math

@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 1e-6)])
def test_leakage_at_init_is_below_tolerance(dtype, tol):
    _, m, _ = make(dtype=dtype, b_std=1.0, d_out=96, r=8, k=5)
    L = m.leakage()
    assert L < tol, L
    # independent float64 numpy check of what leakage() claims
    b = m.locked_B().detach().double().numpy()
    assert np_leakage(m.D.numpy(), b) == pytest.approx(L, abs=1e-14)  # same quantity, different float64 summation order
    # and the unprojected B_tilde really would have leaked (non-vacuous)
    assert np_leakage(m.D.numpy(), m.lora_B_tilde.detach().double().numpy()) > 0.1


def test_locked_B_equals_numpy_double_projection_and_is_a_projection_of_B_tilde():
    _, m, _ = make(dtype=torch.float64, b_std=2.0, d_out=64)
    bt = m.lora_B_tilde.detach().numpy()
    ref = project_out(bt, m.D.numpy(), passes=2)
    assert np.max(np.abs(m.locked_B().detach().numpy() - ref)) < 1e-13
    # explicit P(P B) with the d x d matrix, small sizes only (test oracle, not the implementation)
    P = np.eye(64) - m.D.numpy().T @ m.D.numpy()
    assert np.max(np.abs(P @ (P @ bt) - ref)) < 1e-12
    # idempotent: components already orthogonal to D are untouched
    ortho = ref
    m2 = ProjectThenHealLinear(24, 64, 4, m.D.numpy(), dtype=torch.float64)
    with torch.no_grad():
        m2.lora_B_tilde.copy_(torch.from_numpy(ortho))
    assert np.max(np.abs(m2.locked_B().detach().numpy() - ortho)) < 1e-12


def test_forward_matches_float64_reference_and_protects_output_directions():
    lin, m, _ = make(dtype=torch.float32, bias=True, b_std=0.7)
    x = torch.randn(7, 5, 24)
    y = m(x)
    ref = (
        x.double().numpy() @ m.weight.double().detach().numpy().T
        + m.bias.double().detach().numpy()
        + m.scale * (x.double().numpy() @ m.lora_A.double().detach().numpy().T) @ project_out(m.lora_B_tilde.detach().double().numpy(), m.D.numpy()).T
    )
    assert np.max(np.abs(y.detach().double().numpy() - ref)) < 1e-4
    # The point of the lock: the LoRA branch contributes ~nothing along D.
    delta = (y - lin(x)).detach().double().numpy()
    assert np.max(np.abs(delta @ m.D.numpy().T)) < 1e-5
    assert np.max(np.abs(delta)) > 1e-2  # while the update itself is not zero


def test_zero_init_B_is_identity_to_base():
    lin, m, _ = make(b_std=0.0)
    x = torch.randn(3, 24)
    assert torch.equal(m(x), lin(x))


def test_cost_structure_no_d_out_squared_allocation():
    """Record the output shape of EVERY op in forward+backward and require
    that nothing near d_out*d_out is ever allocated. d_out is large enough that
    a [d_out, d_out] projector would dwarf every legitimate tensor."""
    d_out, d_in, r, k = 1024, 48, 4, 3
    _, m, _ = make(d_in=d_in, d_out=d_out, r=r, k=k)
    x = torch.randn(5, d_in)

    class Record(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in _pytree.tree_leaves(out):
                if isinstance(t, torch.Tensor):
                    self.shapes.append(tuple(t.shape))
            return out

    def biggest(shapes):
        return max((int(np.prod(s)) if s else 1) for s in shapes)

    with Record() as rec:
        m(x).pow(2).sum().backward()
    legit = max(d_out * d_in, d_out * r, k * d_out)
    assert biggest(rec.shapes) <= legit
    assert biggest(rec.shapes) < d_out * d_out // 16
    assert not any(sum(1 for n in s if n == d_out) >= 2 for s in rec.shapes)

    # negative control: the detector does fire on an explicit projector
    with Record() as bad:
        P = torch.eye(d_out, dtype=torch.float64) - m.D.t() @ m.D
        _ = P @ m.lora_B_tilde.double()
    assert biggest(bad.shapes) >= d_out * d_out


# ------------------------------------------------------------------ training

@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-6), (torch.float64, 1e-12)])
@pytest.mark.parametrize("opt_name", ["sgd", "adam"])
def test_leakage_stays_below_tolerance_across_optimizer_steps(dtype, tol, opt_name):
    lin, m, _ = make(dtype=dtype, b_std=0.3, d_out=64, d_in=32, r=6, k=4, seed=3)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.05) if opt_name == "sgd" else torch.optim.Adam(params, lr=0.02)
    g = torch.Generator().manual_seed(5)
    target = torch.randn(16, 64, generator=g, dtype=dtype)
    x = torch.randn(16, 32, generator=g, dtype=dtype)
    worst = 0.0
    first = last = None
    for step in range(25):
        opt.zero_grad()
        loss = (m(x) - target).pow(2).mean()
        loss.backward()
        assert m.lora_B_tilde.grad is not None and torch.isfinite(m.lora_B_tilde.grad).all()
        assert m.lora_B_tilde.grad.abs().max() > 0
        opt.step()
        worst = max(worst, m.leakage())
        first = loss.item() if first is None else first
        last = loss.item()
    assert worst < tol, worst
    assert last < first  # it actually trained
    # B_tilde itself moved off the subspace-orthogonal slice (the lock is in forward, not in the parameter)
    assert np_leakage(m.D.numpy(), m.lora_B_tilde.detach().double().numpy()) > 1e-3


def test_gradients_exist_D_has_none_and_base_is_frozen():
    lin, m, _ = make(bias=True)
    m(torch.randn(4, 24)).sum().backward()
    assert m.lora_B_tilde.grad is not None and m.lora_A.grad is not None
    assert m.D.grad is None and not m.D.requires_grad
    assert m.weight.grad is None and m.bias.grad is None
    # gradient of the loss w.r.t. B_tilde has no component along D (it flows through P)
    g = m.lora_B_tilde.grad.double().numpy()
    assert np_leakage(m.D.numpy(), g) < 1e-5 * max(1.0, np.abs(g).max())


def test_gradcheck_float64():
    _, m, _ = make(dtype=torch.float64, d_in=6, d_out=10, r=3, k=2, b_std=0.4)
    x = torch.randn(4, 6, dtype=torch.float64)

    def f(bt, a):
        d = m.D
        b = bt
        for _ in range(2):
            b = b - d.t() @ (d @ b)
        return torch.nn.functional.linear(torch.nn.functional.linear(x, a), b)

    assert torch.autograd.gradcheck(f, (m.lora_B_tilde.detach().clone().requires_grad_(), m.lora_A.detach().clone().requires_grad_()))
    # and the module's own forward is differentiable end to end
    out = m(x)
    out.sum().backward()
    assert torch.isfinite(m.lora_B_tilde.grad).all()


# ------------------------------------------------------------------ merge / export

def test_merged_delta_merge_into_and_forward_agree():
    lin, m, _ = make(dtype=torch.float64, b_std=0.6, bias=True)
    dW = m.merged_delta()
    ref = m.scale * m.locked_B().detach().numpy() @ m.lora_A.detach().numpy()
    assert dW.shape == (40, 24) and np.max(np.abs(dW.numpy() - ref)) < 1e-12
    merged = m.merge_into()
    assert torch.allclose(merged, m.weight + dW, atol=1e-13)
    x = torch.randn(9, 24, dtype=torch.float64)
    assert torch.allclose(m(x), torch.nn.functional.linear(x, merged, m.bias), atol=1e-10)
    # the merged update has no D component
    assert (m.D @ (merged - m.weight)).abs().max().item() < 1e-12
    assert m.merged_leakage() < 1e-12
    # inplace on an external tensor
    w = m.weight.detach().clone()
    out = m.merge_into(w, inplace=True)
    assert out is w and torch.allclose(w, merged, atol=1e-13)
    with pytest.raises(ValueError):
        m.merge_into(torch.zeros(3, 3))


def test_merge_into_low_precision_leaks_and_says_so():
    _, m, _ = make(dtype=torch.float32, b_std=0.6, d_out=64, d_in=32)
    w16 = m.weight.detach().to(torch.bfloat16)
    assert m.merged_leakage(w16) > 1e-5  # bf16 rounding of W + dW re-introduces leakage
    assert m.merged_leakage() < 1e-5  # float32 merge stays small (rounding step of a float32 weight)


def test_export_lora_layout_names_dtypes_and_values(tmp_path):
    _, m, _ = make(dtype=torch.float32, b_std=0.8)
    name = "base_model.model.model.layers.3.mlp.down_proj"
    p = str(tmp_path / "adapter.safetensors")
    info = m.export_lora(p, name)
    assert info["scale"] == m.scale and info["tensors"] == [f"{name}.lora_A.weight", f"{name}.lora_B.weight"]
    tensors, meta = read_safetensors(p)
    assert set(tensors) == {f"{name}.lora_A.weight", f"{name}.lora_B.weight"}  # nothing stray
    a, b = tensors[f"{name}.lora_A.weight"], tensors[f"{name}.lora_B.weight"]
    assert a.dtype == np.float32 and b.dtype == np.float32
    assert a.shape == (4, 24) and b.shape == (40, 4)  # [r, d_in], [d_out, r]
    assert np.array_equal(a, m.lora_A.detach().numpy())
    assert np_leakage(m.D.numpy(), b.astype(np.float64)) < 1e-6
    assert np.max(np.abs(b - m.locked_B().detach().numpy())) < 1e-6
    assert meta["secant.b_is_locked"] == "true" and meta["format"] == "pt"
    # mirror of the Go parseLoRA pairing regex: every tensor is a lora_A/B and they pair up
    role = re.compile(r"\.lora_([AB])(\.|$)")
    stems = {}
    for t in tensors:
        mm = role.search(t)
        assert mm, t
        stems.setdefault(t[: mm.start()], set()).add(mm.group(1))
    assert stems == {name: {"A", "B"}}
    # never overwrites
    with pytest.raises(FileExistsError):
        m.export_lora(p, name)
    # locked=False exports raw B_tilde for `secant heal` to project itself
    raw = str(tmp_path / "raw.safetensors")
    m.export_lora(raw, name, locked=False)
    rb = read_safetensors(raw)[0][f"{name}.lora_B.weight"]
    assert np.array_equal(rb, m.lora_B_tilde.detach().numpy())
    assert np_leakage(m.D.numpy(), rb.astype(np.float64)) > 1e-2
    with pytest.raises(ValueError):
        m.export_lora(str(tmp_path / "x.safetensors"), "")
    with pytest.raises(ValueError):
        m.export_lora(str(tmp_path / "y.safetensors"), name, dtype="bfloat16")


def test_export_float64_module_and_multi_module(tmp_path):
    _, m64, _ = make(dtype=torch.float64, b_std=0.8, seed=4)
    _, m32, _ = make(dtype=torch.float32, b_std=0.8, seed=5)
    p = str(tmp_path / "multi.safetensors")
    info = export_lora_adapter(p, {"m.layers.0.mlp.down_proj": m64, "m.layers.1.mlp.down_proj": m32})
    t, _ = read_safetensors(p)
    assert t["m.layers.0.mlp.down_proj.lora_B.weight"].dtype == np.float64
    assert t["m.layers.1.mlp.down_proj.lora_B.weight"].dtype == np.float32
    assert len(t) == 4 and info["scales"]["m.layers.0.mlp.down_proj"] == m64.scale
    assert np_leakage(m64.D.numpy(), t["m.layers.0.mlp.down_proj.lora_B.weight"]) < 1e-12


def test_export_readable_by_safetensors_library_if_present(tmp_path):
    st = pytest.importorskip("safetensors", reason="SKIP-NO-SAFETENSORS: cross-reader check needs the safetensors package")
    from safetensors.numpy import load_file

    _, m, _ = make()
    p = str(tmp_path / "x.safetensors")
    m.export_lora(p, "a.b.c")
    got = load_file(p)
    assert sorted(got) == ["a.b.c.lora_A.weight", "a.b.c.lora_B.weight"]
    assert np.array_equal(got["a.b.c.lora_A.weight"], m.lora_A.detach().numpy())
