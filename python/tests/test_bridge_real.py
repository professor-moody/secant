"""Tests against the REAL secant binary and libsecant. Ambient SECANT_BIN /
SECANT_LIB / PATH / ./target / workspace target are honoured (nothing is
isolated here). Each test skips with an explicit SKIP-* reason when its
artifact is missing; none passes vacuously."""

import numpy as np
import pytest

import secant
from secant._linalg import leakage, orthonormalize_rows, project_out
from secant._safetensors import read_safetensors, write_safetensors

from conftest import real_binary_or_skip, real_library_or_skip
from refs import q8_0_roundtrip, q8_0_witness


def _lora_fixture(tmp_path, d_out=48, d_in=32, r=4, k=3, seed=0):
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((k, d_out))
    vec_path = tmp_path / "dirs.npy"
    np.lib.format.write_array(open(vec_path, "wb"), dirs.astype("<f8"), version=(1, 0))
    a = rng.standard_normal((r, d_in)).astype(np.float32)
    b = rng.standard_normal((d_out, r)).astype(np.float32)
    lora = tmp_path / "lora.safetensors"
    name = "base_model.model.model.layers.0.mlp.down_proj"
    write_safetensors(str(lora), {f"{name}.lora_A.weight": a, f"{name}.lora_B.weight": b}, {"format": "pt"})
    return dict(dirs=dirs, vec_path=vec_path, a=a, b=b, lora=lora, name=name)


def skip_if_subcommand_is_a_stub(binary, sub, flags):
    """Skip (explicitly) only when the binary's `sub --help` lists NONE of the
    oracle flags (an unimplemented stub). A partial flag set is a real
    mismatch and must fail, not skip."""
    r = secant.run([sub, "--help"], binary=binary)
    present = [f for f in flags if f in r.stdout]
    if not present:
        pytest.skip(f"SKIP-CLI-STUB: `secant {sub} --help` lists none of {flags}; subcommand not implemented in this build")
    missing = [f for f in flags if f not in present]
    assert not missing, f"`secant {sub}` lacks oracle flags {missing}"


def test_real_cli_version_and_inspect(tmp_path):
    binary = real_binary_or_skip()
    code, out, err = secant.run(["--version"], binary=binary)
    assert code == 0 and "secant" in out.lower()
    fx = _lora_fixture(tmp_path)
    r = secant.inspect(fx["lora"], binary=binary)
    assert r.exit_code == 0 and r.json["tensor_count"] == 2 and r.json["format"] == "safetensors"


def test_real_cli_heal_matches_python_projection_and_verifies(tmp_path):
    binary = real_binary_or_skip()
    fx = _lora_fixture(tmp_path)
    skip_if_subcommand_is_a_stub(binary, "heal", ["--lora", "--vectors", "--out"])
    healed = tmp_path / "healed.safetensors"
    r = secant.heal(fx["lora"], fx["vec_path"], healed, binary=binary, strict=False)
    assert r.exit_code == 0, r.stderr
    assert r.json["status"] == "healed"
    assert r.json["max_leak_after_f64"] < 1e-12 and r.json["max_leak_after_serialized"] < 1e-6
    tensors, _ = read_safetensors(str(healed))
    b_cli = tensors[f"{fx['name']}.lora_B.weight"].astype(np.float64)
    assert np.array_equal(tensors[f"{fx['name']}.lora_A.weight"], fx["a"])  # A copied byte-for-byte
    basis = orthonormalize_rows(fx["dirs"])
    assert leakage(basis, b_cli) < 1e-6
    ref = project_out(fx["b"].astype(np.float64), basis)
    assert np.max(np.abs(b_cli - ref)) < 1e-6  # python float64 projection, stored as f32 by the CLI


def test_real_cli_heal_refuses_wrong_dimension_with_exit_2(tmp_path):
    binary = real_binary_or_skip()
    fx = _lora_fixture(tmp_path)
    wrong = tmp_path / "wrong.npy"
    np.lib.format.write_array(open(wrong, "wb"), np.ones((1, 7)).astype("<f8"), version=(1, 0))
    skip_if_subcommand_is_a_stub(binary, "heal", ["--lora", "--vectors", "--out"])
    r = secant.heal(fx["lora"], wrong, tmp_path / "o.safetensors", binary=binary, strict=False)
    assert r.exit_code == 2


# ------------------------------------------------------------------ Part 2b: the real C ABI (skips without libsecant)


def test_ffi_version_is_a_nonempty_static_string():
    lib = real_library_or_skip()
    v = secant.library_version(lib)
    assert isinstance(v, str) and len(v) > 0 and v == secant.library_version(lib)


def test_ffi_project_then_heal_matches_numpy_reference_and_reports_leak():
    lib = real_library_or_skip()
    rng = np.random.default_rng(21)
    d_out, r, k = 256, 8, 4
    dirs = (rng.standard_normal((k, d_out)) * 5).astype(np.float32)  # NOT orthonormal
    b = rng.standard_normal((d_out, r)).astype(np.float32)
    b_in = b.copy()
    out, max_leak = secant.project_then_heal_f32(dirs, b, lib=lib)
    assert np.array_equal(b, b_in)  # caller's array untouched
    basis = orthonormalize_rows(dirs.astype(np.float64))
    ref = project_out(b.astype(np.float64), basis)
    assert out.dtype == np.float32 and out.shape == b.shape
    assert np.max(np.abs(out - ref)) < 1e-6
    assert leakage(basis, out.astype(np.float64)) < 1e-6
    assert 0.0 <= max_leak < 1e-6
    assert max_leak == pytest.approx(leakage(basis, out.astype(np.float64)), abs=1e-6)
    assert leakage(basis, b.astype(np.float64)) > 1.0  # input really leaked: the test is not vacuous


def test_ffi_project_then_heal_fails_closed_on_bad_directions():
    lib = real_library_or_skip()
    b = np.ones((16, 2), np.float32)
    dup = np.ones((2, 16), np.float32)  # linearly dependent rows
    with pytest.raises(secant.SecantFFIError):
        secant.project_then_heal_f32(dup, b, lib=lib)
    nan = np.ones((1, 16), np.float32)
    nan[0, 3] = np.nan
    with pytest.raises(secant.SecantFFIError):
        secant.project_then_heal_f32(nan, b, lib=lib)
    with pytest.raises(ValueError):
        secant.project_then_heal_f32(np.ones((1, 15), np.float32), b, lib=lib)  # python-side shape check


def test_ffi_q8_0_witness_matches_reference_and_bounds():
    lib = real_library_or_skip()
    rng = np.random.default_rng(22)
    d_out, d_in, k = 64, 96, 3
    w = rng.standard_normal((d_out, d_in)).astype(np.float32)
    basis = orthonormalize_rows(rng.standard_normal((k, d_out)))
    dirs = basis.astype(np.float32)
    l2, mx = secant.q8_0_commutation_witness(dirs, w, lib=lib)
    err = q8_0_roundtrip(w) - w.astype(np.float64)
    ref_l2, ref_mx = q8_0_witness(dirs, w)
    assert l2 > 0 and mx > 0 and l2 >= mx  # Frobenius >= max entry
    assert l2 <= np.linalg.norm(err) * (1 + 1e-6)  # ||D E||_F <= ||E||_F for orthonormal rows
    # Tolerance: 1e-3 relative to the ggml-exact reference (fp16 scale rounding is the only slack).
    assert l2 == pytest.approx(ref_l2, rel=1e-3)
    assert mx == pytest.approx(ref_mx, rel=1e-3)


def test_ffi_q8_0_witness_is_zero_on_exactly_representable_weights():
    lib = real_library_or_skip()
    rng = np.random.default_rng(23)
    d_out, d_in = 32, 64
    q = rng.integers(-127, 128, size=(d_out, d_in // 32, 32)).astype(np.float32)
    q[:, :, 0] = 127  # force amax = 127 * scale in every block
    scale = np.float32(0.0625)  # exactly representable in fp16
    w = (q * scale).reshape(d_out, d_in).astype(np.float32)
    dirs = orthonormalize_rows(rng.standard_normal((2, d_out))).astype(np.float32)
    l2, mx = secant.q8_0_commutation_witness(dirs, w, lib=lib)
    assert l2 < 1e-6 and mx < 1e-6


def test_ffi_q8_0_rejects_bad_shapes_python_side():
    lib = real_library_or_skip()
    with pytest.raises(ValueError, match="multiple of 32"):
        secant.q8_0_commutation_witness(np.ones((1, 4), np.float32), np.ones((4, 33), np.float32), lib=lib)
    with pytest.raises(ValueError):
        secant.q8_0_commutation_witness(np.ones((1, 5), np.float32), np.ones((4, 32), np.float32), lib=lib)
