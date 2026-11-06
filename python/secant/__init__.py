"""Secant Python package.

Importing `secant` needs only numpy. `secant.nn` (torch) is loaded lazily on
first use of `secant.nn` or `secant.ProjectThenHealLinear`.
"""

from __future__ import annotations

from typing import Any

from . import _linalg
from ._bridge import (
    CliResult,
    SecantEnvError,
    SecantError,
    SecantFFIError,
    SecantNotFound,
    cli,
    compose,
    find_binary,
    find_library,
    fetch_tensor,
    rotation_test,
    heal,
    inspect,
    library_version,
    load_library,
    patch,
    project_then_heal_f32,
    q8_0_commutation_witness,
    read,
    run,
    verify_patch,
    verify_subspace,
)
from ._linalg import DirectionError, orthonormalize_rows
from .capture import (
    DirectionResult,
    HookPoint,
    capture_contrastive_directions,
    collect_activations,
    evaluate_cell,
    write_vectors,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CliResult",
    "DirectionError",
    "DirectionResult",
    "HookPoint",
    "ProjectThenHealLinear",
    "SecantEnvError",
    "SecantError",
    "SecantFFIError",
    "SecantNotFound",
    "capture_contrastive_directions",
    "cli",
    "collect_activations",
    "compose",
    "evaluate_cell",
    "export_lora_adapter",
    "find_binary",
    "find_library",
    "fetch_tensor",
    "rotation_test",
    "heal",
    "inspect",
    "library_version",
    "load_library",
    "orthonormalize_rows",
    "patch",
    "project_then_heal_f32",
    "q8_0_commutation_witness",
    "read",
    "run",
    "verify_patch",
    "verify_subspace",
    "write_vectors",
]


def __getattr__(name: str) -> Any:
    # Lazy torch-dependent names, so `import secant` works without torch.
    if name in ("ProjectThenHealLinear", "export_lora_adapter"):
        from . import nn

        return getattr(nn, name)
    if name == "nn":
        import importlib

        return importlib.import_module(".nn", __name__)
    raise AttributeError(f"module 'secant' has no attribute {name!r}")
