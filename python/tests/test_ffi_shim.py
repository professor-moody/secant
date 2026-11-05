"""ctypes plumbing tests against a tiny C reference of the fixed ABI
(tests/ffi_shim/shim.c), compiled at test time. This validates argument order,
size_t/pointer marshalling, status-code handling and copy semantics of
secant._bridge. It says NOTHING about libsecant itself (see test_bridge_real.py)."""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import secant
from secant._linalg import leakage, orthonormalize_rows, project_out
from refs import q8_0_witness


@pytest.fixture(scope="module")
def shim(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("SKIP-NO-C-COMPILER: cannot build the ABI shim")
    src = os.path.join(os.path.dirname(__file__), "ffi_shim", "shim.c")
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    out = tmp_path_factory.mktemp("shim") / f"libshim{ext}"
    cp = subprocess.run([cc, "-O1", "-shared", "-fPIC", "-o", str(out), src, "-lm"], capture_output=True, text=True)
    if cp.returncode != 0:
        pytest.skip(f"SKIP-SHIM-BUILD-FAILED: {cp.stderr.strip()[:300]}")
    return secant.load_library(out)


def test_version(shim):
    assert secant.library_version(shim) == "shim-0"


def test_project_then_heal_plumbing(shim):
    rng = np.random.default_rng(31)
    d_out, r, k = 128, 5, 3
    dirs = (rng.standard_normal((k, d_out)) * 4).astype(np.float32)
    b = rng.standard_normal((d_out, r)).astype(np.float32)
    b0 = b.copy()
    out, leak = secant.project_then_heal_f32(dirs, b, lib=shim)
    assert np.array_equal(b, b0) and out.shape == (d_out, r) and out.dtype == np.float32
    basis = orthonormalize_rows(dirs.astype(np.float64))
    assert np.max(np.abs(out - project_out(b0.astype(np.float64), basis))) < 1e-6
    assert leak < 1e-6 and leak == pytest.approx(leakage(basis, out.astype(np.float64)), abs=1e-7)
    assert leakage(basis, b0.astype(np.float64)) > 1.0
    # non-contiguous / float64 / 1-D direction inputs are marshalled correctly
    out2, _ = secant.project_then_heal_f32(dirs[:1].astype(np.float64)[0], np.asfortranarray(b0), lib=shim)
    assert np.max(np.abs(out2 - project_out(b0.astype(np.float64), orthonormalize_rows(dirs[:1].astype(np.float64)))) ) < 1e-6


def test_status_codes_surface_as_exceptions(shim):
    b = np.ones((16, 2), np.float32)
    with pytest.raises(secant.SecantFFIError) as ei:
        secant.project_then_heal_f32(np.ones((2, 16), np.float32), b, lib=shim)
    assert ei.value.code != 0 and ei.value.fn == "secant_project_then_heal_f32_v1"
    nan = np.ones((1, 16), np.float32)
    nan[0, 0] = np.nan
    with pytest.raises(secant.SecantFFIError):
        secant.project_then_heal_f32(nan, b, lib=shim)
    with pytest.raises(secant.SecantFFIError):  # k > d_out is rejected by the callee
        secant.project_then_heal_f32(np.ones((4, 2), np.float32), np.ones((2, 1), np.float32), lib=shim)
    with pytest.raises(ValueError):
        secant.project_then_heal_f32(np.ones((1, 15), np.float32), b, lib=shim)
    with pytest.raises(ValueError, match="multiple of 32"):
        secant.q8_0_commutation_witness(np.ones((1, 4), np.float32), np.ones((4, 33), np.float32), lib=shim)


def test_q8_0_witness_plumbing(shim):
    rng = np.random.default_rng(32)
    d_out, d_in, k = 48, 64, 2
    w = rng.standard_normal((d_out, d_in)).astype(np.float32)
    dirs = orthonormalize_rows(rng.standard_normal((k, d_out))).astype(np.float32)
    l2, mx = secant.q8_0_commutation_witness(dirs, w, lib=shim)
    rl2, rmx = q8_0_witness(dirs, w, fp16_scale=False)  # the shim keeps the scale in float32
    assert l2 == pytest.approx(rl2, rel=1e-5) and mx == pytest.approx(rmx, rel=1e-5)
    assert 0 < mx <= l2
