"""Bridges from Python to Secant: the `secant` CLI (subprocess) and `libsecant` (ctypes).

Binary discovery order: explicit `binary=` argument; else the `SECANT_BIN`
environment variable; else search PATH, then ./target/release, ./target/debug
(relative to the current directory), then the same two directories of the
workspace that contains this package when it runs from a source checkout.
`SECANT_BIN` has three states and each is handled differently: ABSENT means
search; PRESENT-BUT-EMPTY raises (an empty value is almost always a broken
export, not a request to search); PRESENT USES that path, which must exist.
`SECANT_LIB` follows the same rule for the shared library.
"""

from __future__ import annotations

import ctypes
import json as _json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, "os.PathLike[str]"]

ENV_BIN = "SECANT_BIN"
ENV_LIB = "SECANT_LIB"


class SecantError(RuntimeError):
    """The CLI or library could not give an answer (not found, exit code 2, signal, timeout)."""


class SecantNotFound(SecantError):
    """No usable binary / library was found."""


class SecantEnvError(SecantError):
    """SECANT_BIN / SECANT_LIB is present but empty, or points at nothing usable."""


# ---------------------------------------------------------------------- discovery

_PKG_WORKSPACE = Path(__file__).resolve().parents[2]  # <workspace>/python/secant/_bridge.py -> <workspace>


def _target_dirs() -> List[Path]:
    dirs = [Path.cwd() / "target" / "release", Path.cwd() / "target" / "debug"]
    for sub in ("release", "debug"):
        d = _PKG_WORKSPACE / "target" / sub
        if d not in dirs:
            dirs.append(d)
    return dirs


def _usable_file(p: Path, executable: bool) -> bool:
    return p.is_file() and (not executable or os.access(p, os.X_OK))


def _resolve(explicit: Optional[PathLike], env_name: str, what: str, names: Sequence[str], executable: bool, env: Optional[Mapping[str, str]] = None) -> Path:
    if explicit is not None:
        s = os.fspath(explicit)
        if s == "":
            raise SecantEnvError(f"explicit {what} path is empty")
        p = Path(s)
        if not _usable_file(p, executable):
            raise SecantNotFound(f"explicit {what} {s!r} does not exist or is not usable")
        return p
    environ = os.environ if env is None else env
    if env_name in environ:  # present
        v = environ[env_name]
        if v == "":  # present-but-empty
            raise SecantEnvError(f"{env_name} is set but empty; unset it to search, or set it to the {what} path")
        p = Path(v)
        if not _usable_file(p, executable):
            raise SecantEnvError(f"{env_name}={v!r} does not exist or is not usable")
        return p
    searched: List[str] = []  # absent
    if executable:
        for n in names:
            w = shutil.which(n)
            searched.append(f"PATH:{n}")
            if w:
                return Path(w)
    for d in _target_dirs():
        for n in names:
            cand = d / n
            searched.append(str(cand))
            if _usable_file(cand, executable):
                return cand
    raise SecantNotFound(f"no {what} found (explicit argument and {env_name} absent); searched: " + ", ".join(searched))


def find_binary(binary: Optional[PathLike] = None, *, env: Optional[Mapping[str, str]] = None) -> Path:
    """Locate the `secant` executable (see module docstring for the rule)."""
    return _resolve(binary, ENV_BIN, "secant binary", ["secant"], True, env)


def find_library(path: Optional[PathLike] = None, *, env: Optional[Mapping[str, str]] = None) -> Path:
    """Locate `libsecant.{dylib,so}` (see module docstring for the rule)."""
    names = ["libsecant.dylib", "libsecant.so"] if sys.platform == "darwin" else ["libsecant.so", "libsecant.dylib"]
    return _resolve(path, ENV_LIB, "libsecant", names, False, env)


# ---------------------------------------------------------------------- subprocess bridge


@dataclass
class CliResult:
    """Outcome of one CLI run. Unpacks as (exit_code, stdout, stderr).

    Exit codes follow the oracle: 0 passed, 1 a gate ran and failed, 2 usage /
    I-O / undeterminable. `json` is the parsed stdout when JSON was requested
    and parseable; otherwise None with the reason in `json_error`.
    """

    exit_code: int
    stdout: str
    stderr: str
    json: Any = None
    json_error: Optional[str] = None
    argv: Tuple[str, ...] = ()

    def __iter__(self) -> Iterator[Any]:
        yield self.exit_code
        yield self.stdout
        yield self.stderr

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def gate_failed(self) -> bool:
        return self.exit_code == 1


def _argv(args: Sequence[Any]) -> List[str]:
    if isinstance(args, (str, bytes)):
        raise TypeError("args must be a list of arguments, not a single string")
    out = []
    for a in args:
        if isinstance(a, bool) or not isinstance(a, (str, os.PathLike, int, float)):
            raise TypeError(f"argument {a!r} is not a str/path/number")
        out.append(os.fspath(a) if isinstance(a, os.PathLike) else str(a))
    return out


def run(
    args: Sequence[Any],
    *,
    binary: Optional[PathLike] = None,
    timeout: Optional[float] = None,
    cwd: Optional[PathLike] = None,
    env: Optional[Mapping[str, str]] = None,
    parse_json: Optional[bool] = None,
    go_arm64_fma: bool = False,
    check: bool = False,
) -> CliResult:
    """Run `secant <args>` noninteractively (no shell, stdin closed) and return a CliResult.

    `parse_json=None` parses stdout as JSON exactly when "--json" is in args.
    A nonzero exit code is returned, not raised, unless `check=True` (then
    any nonzero code raises `SecantError`). Timeouts and a missing binary
    always raise: they are not answers.
    """
    exe = find_binary(binary)  # discovery reads os.environ, never the child's `env`
    argv = _argv(args)
    full = [str(exe)] + (["--go-arm64-fma"] if go_arm64_fma else []) + argv
    try:
        cp = subprocess.run(
            full,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            cwd=None if cwd is None else os.fspath(cwd),
            env=None if env is None else dict(env),
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise SecantError(f"secant timed out after {timeout}s: {' '.join(full)}") from e
    except OSError as e:
        raise SecantError(f"could not execute {exe}: {e}") from e
    out = cp.stdout.decode("utf-8", errors="replace")
    err = cp.stderr.decode("utf-8", errors="replace")
    res = CliResult(cp.returncode, out, err, argv=tuple(full))
    want = ("--json" in argv) if parse_json is None else parse_json
    if want:
        if out.strip():
            try:
                res.json = _json.loads(out)
            except ValueError as e:
                res.json_error = f"stdout is not valid JSON: {e}"
        else:
            res.json_error = "stdout is empty"
    if check and cp.returncode != 0:
        raise SecantError(f"secant exited {cp.returncode}: {' '.join(full)}\n{err.strip()}")
    return res


#: `secant.cli` is the same callable as `secant.run`.
cli = run


def _need(result: CliResult, what: str, strict: bool) -> CliResult:
    # Exit 2 / signal / anything but 0 or 1 means the question was not answered.
    if strict and result.exit_code not in (0, 1):
        raise SecantError(f"secant {what} could not be determined (exit {result.exit_code}): {result.stderr.strip()}")
    return result


def _opt(argv: List[str], flag: str, value: Any) -> None:
    if value is not None:
        argv += [flag, os.fspath(value) if isinstance(value, os.PathLike) else str(value)]


def inspect(model: PathLike, *, json: bool = True, strict: bool = True, **kw: Any) -> CliResult:
    """`secant inspect --model <model> [--json]`."""
    argv = ["inspect", "--model", model] + (["--json"] if json else [])
    return _need(run(argv, **kw), "inspect", strict)


def read(
    model: PathLike,
    *,
    tensor: Optional[str] = None,
    top_cols: Optional[int] = None,
    top_k_singular: Optional[int] = None,
    vocab_tensor: Optional[str] = None,
    detect_anomalies: bool = False,
    vectors_out: Optional[PathLike] = None,
    json: bool = True,
    strict: bool = True,
    **kw: Any,
) -> CliResult:
    """`secant read --model ...` (flags as the oracle's `wsp read`)."""
    argv: List[str] = ["read", "--model", os.fspath(model)]
    _opt(argv, "--tensor", tensor)
    _opt(argv, "--top-cols", top_cols)
    _opt(argv, "--top-k-singular", top_k_singular)
    _opt(argv, "--vocab-tensor", vocab_tensor)
    if detect_anomalies:
        argv.append("--detect-anomalies")
    _opt(argv, "--vectors-out", vectors_out)
    if json:
        argv.append("--json")
    return _need(run(argv, **kw), "read", strict)


def compose(
    profile: str,
    out: PathLike,
    *,
    vectors: Optional[PathLike] = None,
    analysis: Optional[PathLike] = None,
    target_shift: Optional[float] = None,
    gamma: Optional[float] = None,
    base: Optional[PathLike] = None,
    json: bool = True,
    strict: bool = True,
    **kw: Any,
) -> CliResult:
    """`secant compose --profile ... --vectors ... --out ...` (flags as the oracle's `wsp compose`)."""
    argv: List[str] = ["compose", "--profile", profile]
    _opt(argv, "--vectors", vectors)
    _opt(argv, "--analysis", analysis)
    _opt(argv, "--target-shift", target_shift)
    _opt(argv, "--gamma", gamma)
    _opt(argv, "--base", base)
    argv += ["--out", os.fspath(out)]
    if json:
        argv.append("--json")
    return _need(run(argv, **kw), "compose", strict)


def verify_subspace(
    base: PathLike,
    healed: PathLike,
    vectors: PathLike,
    tensors: Union[str, Sequence[str]],
    *,
    tol: Optional[float] = None,
    json: bool = True,
    strict: bool = True,
    **kw: Any,
) -> CliResult:
    """`secant verify-subspace --base --healed --vectors --tensors [--tol]`.

    Exit 0 pass, 1 leak (returned, not raised), 2 unknown (raises when strict).
    The parsed report's `verdict` is "pass", "leak" or "unknown".
    """
    tl = tensors if isinstance(tensors, str) else ",".join(tensors)
    if not tl:
        raise ValueError("tensors is empty")
    argv: List[str] = ["verify-subspace", "--base", os.fspath(base), "--healed", os.fspath(healed), "--vectors", os.fspath(vectors), "--tensors", tl]
    _opt(argv, "--tol", tol)
    if json:
        argv.append("--json")
    return _need(run(argv, **kw), "verify-subspace", strict)


def heal(
    lora: PathLike,
    vectors: PathLike,
    out: PathLike,
    *,
    base: Optional[PathLike] = None,
    merge_out: Optional[PathLike] = None,
    scale: Optional[float] = None,
    tol: Optional[float] = None,
    json: bool = True,
    strict: bool = True,
    **kw: Any,
) -> CliResult:
    """`secant heal --lora --vectors --out [--base --merge-out] [--scale] [--tol]`.

    Exit 0 healed, 1 a leakage gate failed (returned, not raised), 2 refused
    or undeterminable (raises when strict). `out`/`merge_out` must be new paths.
    """
    if (base is None) != (merge_out is None):
        raise ValueError("base and merge_out must be given together")
    argv: List[str] = ["heal", "--lora", os.fspath(lora), "--vectors", os.fspath(vectors), "--out", os.fspath(out)]
    _opt(argv, "--base", base)
    _opt(argv, "--merge-out", merge_out)
    _opt(argv, "--scale", scale)
    _opt(argv, "--tol", tol)
    if json:
        argv.append("--json")
    return _need(run(argv, **kw), "heal", strict)


def verify_patch(base: PathLike, patched: PathLike, *, expect_changed: Union[None, str, Sequence[str]] = None, json: bool = True, strict: bool = True, **kw: Any) -> CliResult:
    """`secant verify-patch --base --patched [--expect-changed a,b]`."""
    argv: List[str] = ["verify-patch", "--base", os.fspath(base), "--patched", os.fspath(patched)]
    if expect_changed is not None:
        _opt(argv, "--expect-changed", expect_changed if isinstance(expect_changed, str) else ",".join(expect_changed))
    if json:
        argv.append("--json")
    return _need(run(argv, **kw), "verify-patch", strict)


def patch(base: PathLike, plan: PathLike, out: PathLike, *, json: bool = True, strict: bool = True, **kw: Any) -> CliResult:
    """`secant patch --base --plan --out`."""
    argv: List[str] = ["patch", "--base", os.fspath(base), "--plan", os.fspath(plan), "--out", os.fspath(out)]
    if json:
        argv.append("--json")
    return _need(run(argv, **kw), "patch", strict)


# ---------------------------------------------------------------------- ctypes FFI bridge

_C_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_C_DOUBLE_P = ctypes.POINTER(ctypes.c_double)
_LIBS: Dict[str, ctypes.CDLL] = {}


class SecantFFIError(SecantError):
    """A C ABI call returned a nonzero status."""

    def __init__(self, fn: str, code: int):
        super().__init__(f"{fn} returned status {code}")
        self.fn = fn
        self.code = code


def load_library(path: Optional[PathLike] = None, *, env: Optional[Mapping[str, str]] = None) -> ctypes.CDLL:
    """Load libsecant and declare the fixed C ABI signatures. Cached per resolved path."""
    p = str(find_library(path, env=env).resolve())
    lib = _LIBS.get(p)
    if lib is not None:
        return lib
    try:
        lib = ctypes.CDLL(p)
    except OSError as e:
        raise SecantError(f"could not load {p}: {e}") from e
    try:
        lib.secant_version_v1.argtypes = []
        lib.secant_version_v1.restype = ctypes.c_char_p
        lib.secant_project_then_heal_f32_v1.argtypes = [_C_FLOAT_P, ctypes.c_size_t, ctypes.c_size_t, _C_FLOAT_P, ctypes.c_size_t, _C_DOUBLE_P]
        lib.secant_project_then_heal_f32_v1.restype = ctypes.c_int32
        lib.secant_q8_0_commutation_witness_v1.argtypes = [_C_FLOAT_P, ctypes.c_size_t, ctypes.c_size_t, _C_FLOAT_P, ctypes.c_size_t, _C_DOUBLE_P, _C_DOUBLE_P]
        lib.secant_q8_0_commutation_witness_v1.restype = ctypes.c_int32
    except AttributeError as e:
        raise SecantError(f"{p} does not export the expected C ABI: {e}") from e
    _LIBS[p] = lib
    return lib


def _f32_matrix(x: Any, what: str) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    a = np.ascontiguousarray(x, dtype=np.float32)
    if a.ndim == 1 and what == "directions":
        a = a[None, :]
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"{what} must be a non-empty 2-D matrix, got shape {a.shape}")
    return a


def library_version(lib: Optional[ctypes.CDLL] = None, *, path: Optional[PathLike] = None) -> str:
    """`secant_version_v1()` as str (static string owned by the library; not freed)."""
    lib = lib or load_library(path)
    raw = lib.secant_version_v1()
    if raw is None:
        raise SecantError("secant_version_v1 returned NULL")
    return raw.decode("utf-8")


def project_then_heal_f32(
    directions: Any,
    b: Any,
    *,
    lib: Optional[ctypes.CDLL] = None,
    path: Optional[PathLike] = None,
) -> Tuple[np.ndarray, float]:
    """B_locked = P(P B) via the C ABI. Returns (new float32 [d_out, r] array, max_leak).

    `directions` [k, d_out] (rows are orthonormalized by the library; not
    assumed orthonormal); `b` [d_out, r]. The caller's `b` is not modified
    (the ABI works in place on a private copy). max_leak is the library's
    float64 measurement, not recomputed here.
    """
    lib = lib or load_library(path)
    d = _f32_matrix(directions, "directions")
    out = np.array(_f32_matrix(b, "b"), dtype=np.float32, order="C", copy=True)
    if d.shape[1] != out.shape[0]:
        raise ValueError(f"directions are [k={d.shape[0]}, {d.shape[1]}] but b is [{out.shape[0]}, {out.shape[1]}]; d_out must match")
    leak = ctypes.c_double(float("nan"))
    code = lib.secant_project_then_heal_f32_v1(
        d.ctypes.data_as(_C_FLOAT_P), d.shape[0], d.shape[1], out.ctypes.data_as(_C_FLOAT_P), out.shape[1], ctypes.byref(leak)
    )
    if code != 0:
        raise SecantFFIError("secant_project_then_heal_f32_v1", int(code))
    return out, float(leak.value)


def q8_0_commutation_witness(
    directions: Any,
    w: Any,
    *,
    lib: Optional[ctypes.CDLL] = None,
    path: Optional[PathLike] = None,
) -> Tuple[float, float]:
    """||D^T (Dequant(Quant_Q8_0(W)) - W)||_2 (Frobenius over k x d_in) and its max |entry|.

    `directions` [k, d_out]; `w` [d_out, d_in] with d_in a multiple of 32.
    Returns (l2, max_abs).
    """
    lib = lib or load_library(path)
    d = _f32_matrix(directions, "directions")
    ww = _f32_matrix(w, "w")
    if ww.shape[1] % 32 != 0:
        raise ValueError(f"d_in={ww.shape[1]} is not a multiple of 32 (Q8_0 block size)")
    if d.shape[1] != ww.shape[0]:
        raise ValueError(f"directions have d_out={d.shape[1]} but w has {ww.shape[0]} rows")
    l2, mx = ctypes.c_double(float("nan")), ctypes.c_double(float("nan"))
    code = lib.secant_q8_0_commutation_witness_v1(
        d.ctypes.data_as(_C_FLOAT_P), d.shape[0], d.shape[1], ww.ctypes.data_as(_C_FLOAT_P), ww.shape[1], ctypes.byref(l2), ctypes.byref(mx)
    )
    if code != 0:
        raise SecantFFIError("secant_q8_0_commutation_witness_v1", int(code))
    return float(l2.value), float(mx.value)
