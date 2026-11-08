"""Float64 linear algebra shared by the torch module and the tests.

Pure numpy, no torch import, so the logic can be tested (and used as the
reference for the FFI bridge) on hosts without torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Residual-norm floor (relative to the input vector's norm) below which a
#: direction is declared linearly dependent on the earlier ones.
DEPENDENCE_TOL = 1e-9
#: Bound on max |D D^T - I| after orthonormalization (mirrors the oracle's
#: strict 1e-12 post-orthogonalization contract).
ORTHONORMALITY_TOL = 1e-12


class DirectionError(ValueError):
    """Protected directions are unusable (NaN/Inf, dependent, wrong shape).

    Raised instead of returning a basis: a lock built on a degenerate basis
    would silently protect less than the caller believes.
    """


def as_rows_f64(directions: Any) -> np.ndarray:
    """Coerce a vector, [k, d] array/tensor, or list of vectors to float64 [k, d]."""
    if hasattr(directions, "detach"):  # torch.Tensor, without importing torch
        directions = directions.detach().cpu().numpy()
    arr = np.array(directions, dtype=np.float64, copy=True)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise DirectionError(f"directions must be a vector or a [k, d] matrix, got shape {arr.shape}")
    k, d = arr.shape
    if k == 0 or d == 0:
        raise DirectionError(f"directions are empty: shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))[0]
        raise DirectionError(f"direction {int(bad[0])} has a non-finite entry at {int(bad[1])}")
    return arr


def orthonormalize_rows(
    directions: Any,
    *,
    dependence_tol: float = DEPENDENCE_TOL,
    orthonormality_tol: float = ORTHONORMALITY_TOL,
) -> np.ndarray:
    """Two-pass modified Gram-Schmidt in float64. Returns an orthonormal [k, d] basis.

    Every vector is orthogonalized against all earlier basis vectors twice
    (the second pass removes the ~1e-13 residue one pass can leave on nearly
    dependent input), then normalized. Fails closed with `DirectionError`:

    * non-finite input;
    * k > d (cannot be independent);
    * a vector whose residual norm is below `dependence_tol` times its own
      norm (dependent on the earlier ones, or zero);
    * a final max |D D^T - I| at or above `orthonormality_tol`.
    """
    rows = as_rows_f64(directions)
    k, d = rows.shape
    if k > d:
        raise DirectionError(f"{k} directions cannot be linearly independent in dimension {d}")
    basis = np.zeros_like(rows)
    for i in range(k):
        u = rows[i].copy()
        n0 = float(np.sqrt(np.dot(u, u))) if np.max(np.abs(u)) < 1e150 and np.max(np.abs(u)) > 1e-150 else float(np.hypot.reduce(u))
        if not (n0 > 0.0) or not np.isfinite(n0):
            raise DirectionError(f"direction {i} has zero or non-finite norm")
        scaled = n0 > 1e150 or n0 < 1e-150
        if scaled:
            u /= n0
        for _ in range(2):
            for j in range(i):
                u -= np.dot(u, basis[j]) * basis[j]
        n1 = float(np.sqrt(np.dot(u, u)))
        if not (np.isfinite(n1) and n1 > 0.0):
            n1 = float(np.hypot.reduce(u))
        if not np.isfinite(n1) or not (n1 > 0.0 and n1 >= dependence_tol * (1.0 if scaled else n0)):
            raise DirectionError(
                f"direction {i} is linearly dependent on earlier directions "
                f"(residual norm {n1:.3e} vs input norm {n0:.3e})"
            )
        basis[i] = u / n1
    gram_err = float(np.max(np.abs(basis @ basis.T - np.eye(k))))
    if not (gram_err < orthonormality_tol):
        raise DirectionError(f"orthonormalization residual max|D D^T - I| = {gram_err:.3e} >= {orthonormality_tol:.0e}")
    return basis


def cross_projection_compensate(directions: Any, shifts: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return Q rows and amplitudes a with U @ (Q.T @ a) == shifts.

    U holds the original unit directions. U.T = Q.T R, so the required
    compensation is a = R^-T shifts, solved by forward substitution. This
    helper covers a common activation/support; compose also includes each
    sparse coefficient row's response to the requesting reference activation.
    """
    rows = as_rows_f64(directions)
    unit = rows / np.hypot.reduce(rows, axis=1)[:, None]
    basis = orthonormalize_rows(rows)
    targets = np.asarray(shifts, dtype=np.float64)
    if targets.shape != (len(rows),) or not np.all(np.isfinite(targets)):
        raise DirectionError("shifts must be a finite vector with one entry per direction")
    r_transpose = unit @ basis.T
    amplitudes = np.zeros_like(targets)
    for i in range(len(rows)):
        amplitudes[i] = (targets[i] - r_transpose[i, :i] @ amplitudes[:i]) / r_transpose[i, i]
    if not np.all(np.isfinite(amplitudes)):
        raise DirectionError("non-finite cross-projection compensation")
    return basis, amplitudes


def project_out(b: np.ndarray, basis: np.ndarray, passes: int = 2) -> np.ndarray:
    """Return P^passes B with P = I - D^T D, never forming the d x d matrix.

    `b` is [d_out, r], `basis` is [k, d_out]; all float64.
    """
    out = np.array(b, dtype=np.float64, copy=True)
    for _ in range(passes):
        out -= basis.T @ (basis @ out)
    return out


def leakage(basis: np.ndarray, b: np.ndarray) -> float:
    """max |D B| in float64 (the quantity the oracle calls leakage)."""
    return float(np.max(np.abs(np.asarray(basis, dtype=np.float64) @ np.asarray(b, dtype=np.float64))))
