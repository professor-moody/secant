"""Numpy-only tests of the float64 core (run everywhere, no torch)."""

import numpy as np
import pytest

from secant._linalg import DirectionError, leakage, orthonormalize_rows, project_out


def test_orthonormalize_random_is_orthonormal_to_1e_12():
    rng = np.random.default_rng(1)
    d = orthonormalize_rows(rng.standard_normal((6, 64)) * 37.0)
    assert d.shape == (6, 64)
    assert np.max(np.abs(d @ d.T - np.eye(6))) < 1e-12


def test_orthonormalize_vector_and_idempotent_on_orthonormal_input():
    rng = np.random.default_rng(2)
    d = orthonormalize_rows(rng.standard_normal((3, 20)))
    again = orthonormalize_rows(d)
    assert np.max(np.abs(again - d)) < 1e-12
    v = orthonormalize_rows(np.arange(1.0, 9.0))
    assert v.shape == (1, 8) and abs(np.linalg.norm(v) - 1) < 1e-15


def test_nearly_dependent_but_distinct_is_accepted_and_still_orthonormal():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(32)
    b = a + 1e-5 * rng.standard_normal(32)  # residual 1e-5 relative, above the 1e-9 floor
    d = orthonormalize_rows(np.stack([a, b]))
    assert np.max(np.abs(d @ d.T - np.eye(2))) < 1e-12


@pytest.mark.parametrize("name", ["duplicate", "scaled_copy", "zero", "nan", "inf", "k_gt_d", "empty", "rank3"])
def test_fail_closed(name):
    rng = np.random.default_rng(4)
    a = rng.standard_normal(16)
    cases = {
        "duplicate": np.stack([a, a]),
        "scaled_copy": np.stack([a, -3.0 * a]),
        "zero": np.stack([a, np.zeros(16)]),
        "nan": np.stack([a, np.where(np.arange(16) == 5, np.nan, 1.0)]),
        "inf": np.stack([a, np.where(np.arange(16) == 5, np.inf, 1.0)]),
        "k_gt_d": rng.standard_normal((5, 4)),
        "empty": np.zeros((0, 8)),
        "rank3": rng.standard_normal((3, 2)) @ rng.standard_normal((2, 16)),  # 3 rows, rank 2
    }
    with pytest.raises(DirectionError):
        orthonormalize_rows(cases[name])


def test_project_out_double_pass_leakage_is_rounding_level_and_no_dd_matrix():
    rng = np.random.default_rng(5)
    d_out, r = 512, 8
    D = orthonormalize_rows(rng.standard_normal((4, d_out)))
    B = rng.standard_normal((d_out, r)) * 1e3
    assert leakage(D, B) > 1.0  # the unprojected B really leaks
    L = project_out(B, D)
    assert leakage(D, L) < 1e-12
    # P is a projector: applying again changes nothing beyond rounding
    assert np.max(np.abs(project_out(L, D) - L)) < 1e-10
    # equals the explicit d x d construction (small d only, as a cross-check)
    P = np.eye(d_out) - D.T @ D
    assert np.max(np.abs(P @ (P @ B) - L)) < 1e-9
