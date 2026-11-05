"""Contrastive direction capture with held-out quality metrics.

For each (stream, layer) cell this computes the difference-of-means direction
d = (mu_pos - mu_neg) / ||mu_pos - mu_neg|| and, because a direction fitted
and scored on the same data always looks good, judges it on data it never saw:

* The positive and negative samples are each shuffled with a fixed seed and
  cut into halves A and B (A = first floor(n/2), B = the rest).
* `snr_proj`: fit d_A on A, then on B compute
  (mean_pos(B.d_A) - mean_neg(B.d_A))^2 / (var_pos(B.d_A) + var_neg(B.d_A)).
  Repeated with the roles swapped; reported as a_to_b, b_to_a and their mean.
* `snr_trace`: ||mu_pos - mu_neg||^2 / (tr(Sigma_pos) + tr(Sigma_neg)) on the
  held-out half. It uses no fitted direction (full-space), so "fit on A" only
  decides which half is scored. NOTE: with finite samples the numerator holds
  ||noise||^2 too, so this estimate is biased upward by
  d * s^2 * (1/n_pos + 1/n_neg) on a half of size n; it is not a null-corrected statistic.
* `split_half_cosine`: cos(d_A, d_B), the direction fitted independently on
  each half. Near 1 means a stable direction; near 0 (about N(0, 1/d)) means
  the "direction" is noise.

Variances are unbiased (ddof=1); everything is float64. A cell whose mean
difference is zero or whose denominators vanish is reported as
status="undetermined" with NaN metrics and a reason, never as a number.
"""

from __future__ import annotations

import math
import os
import re
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "DirectionResult",
    "HookPoint",
    "capture_contrastive_directions",
    "collect_activations",
    "evaluate_cell",
    "write_vectors",
]

Key = Tuple[str, int]
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
#: relative size of ||mu_pos - mu_neg|| below which the direction is undefined
_ZERO_DIFF_REL = 1e-12


@dataclass(frozen=True)
class DirectionResult:
    stream: str
    layer: int
    status: str  # "ok" | "undetermined"
    reason: str
    direction: np.ndarray  # unit norm float64 fitted on all samples; NaN-filled when undetermined
    direction_a: np.ndarray  # fitted on split A
    direction_b: np.ndarray  # fitted on split B
    n_pos: int
    n_neg: int
    dim: int
    seed: int
    snr_proj: float  # mean of a_to_b and b_to_a
    snr_proj_a_to_b: float  # fit on A, scored on B
    snr_proj_b_to_a: float  # fit on B, scored on A
    snr_trace: float
    snr_trace_a_to_b: float  # scored on B
    snr_trace_b_to_a: float  # scored on A
    split_half_cosine: float

    def metrics(self) -> Dict[str, Any]:
        """JSON-serializable summary (no arrays). NaN stays NaN; json.dumps writes it as NaN."""
        return {
            "stream": self.stream,
            "layer": self.layer,
            "status": self.status,
            "reason": self.reason,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "dim": self.dim,
            "seed": self.seed,
            "snr_proj": self.snr_proj,
            "snr_proj_a_to_b": self.snr_proj_a_to_b,
            "snr_proj_b_to_a": self.snr_proj_b_to_a,
            "snr_trace": self.snr_trace,
            "snr_trace_a_to_b": self.snr_trace_a_to_b,
            "snr_trace_b_to_a": self.snr_trace_b_to_a,
            "split_half_cosine": self.split_half_cosine,
        }

    @property
    def stem(self) -> str:
        return f"d_{self.stream}_L{self.layer}"


# ---------------------------------------------------------------------- statistics


def _as_matrix(x: Any, what: str) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    a = np.array(x, dtype=np.float64, copy=True)
    if a.ndim != 2:
        raise ValueError(f"{what} must be [n, d], got shape {a.shape}")
    if a.shape[1] == 0:
        raise ValueError(f"{what} has zero feature dimension")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{what} holds NaN/Inf")
    return a


def _diff_direction(pos: np.ndarray, neg: np.ndarray) -> Optional[np.ndarray]:
    mp, mn = pos.mean(axis=0), neg.mean(axis=0)
    diff = mp - mn
    n = float(np.sqrt(diff @ diff))
    scale = float(np.sqrt(mp @ mp) + np.sqrt(mn @ mn))
    if not math.isfinite(n) or n == 0.0 or n <= _ZERO_DIFF_REL * scale:
        return None
    return diff / n


def _snr_proj(u: Optional[np.ndarray], pos: np.ndarray, neg: np.ndarray) -> float:
    if u is None:
        return math.nan
    p, q = pos @ u, neg @ u
    den = float(p.var(ddof=1) + q.var(ddof=1))
    if not (den > 0.0) or not math.isfinite(den):
        return math.nan
    return float((p.mean() - q.mean()) ** 2 / den)


def _snr_trace(pos: np.ndarray, neg: np.ndarray) -> float:
    diff = pos.mean(axis=0) - neg.mean(axis=0)
    den = float(pos.var(axis=0, ddof=1).sum() + neg.var(axis=0, ddof=1).sum())
    if not (den > 0.0) or not math.isfinite(den):
        return math.nan
    return float(diff @ diff / den)


def _mean2(a: float, b: float) -> float:
    return math.nan if (math.isnan(a) or math.isnan(b)) else 0.5 * (a + b)


def evaluate_cell(
    pos: Any,
    neg: Any,
    *,
    stream: str = "default",
    layer: int = 0,
    seed: int = 0,
    min_per_class: int = 4,
) -> DirectionResult:
    """Fit and held-out-score one (stream, layer) cell from activations [n_pos, d] and [n_neg, d]."""
    P = _as_matrix(pos, f"pos[{stream}, L{layer}]")
    N = _as_matrix(neg, f"neg[{stream}, L{layer}]")
    if P.shape[1] != N.shape[1]:
        raise ValueError(f"pos has dim {P.shape[1]} but neg has dim {N.shape[1]}")
    if min_per_class < 4:
        raise ValueError("min_per_class must be >= 4 (two per class in each half, for a variance)")
    if P.shape[0] < min_per_class or N.shape[0] < min_per_class:
        raise ValueError(f"need at least {min_per_class} samples per class, got pos={P.shape[0]} neg={N.shape[0]}")

    # Split depends on (seed, stream) only, so layers of one stream share the
    # split and results do not depend on iteration order.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(stream.encode("utf-8"))]))
    pp, pn = rng.permutation(P.shape[0]), rng.permutation(N.shape[0])
    ha, hn = P.shape[0] // 2, N.shape[0] // 2
    PA, PB = P[pp[:ha]], P[pp[ha:]]
    NA, NB = N[pn[:hn]], N[pn[hn:]]

    d_all = _diff_direction(P, N)
    d_a = _diff_direction(PA, NA)
    d_b = _diff_direction(PB, NB)
    nan_vec = np.full(P.shape[1], math.nan)

    snr_ab = _snr_proj(d_a, PB, NB)
    snr_ba = _snr_proj(d_b, PA, NA)
    tr_ab = _snr_trace(PB, NB)
    tr_ba = _snr_trace(PA, NA)
    cos = math.nan if (d_a is None or d_b is None) else float(np.clip(d_a @ d_b, -1.0, 1.0))

    problems = []
    if d_all is None:
        problems.append("mean difference of all samples is zero")
    if d_a is None or d_b is None:
        problems.append("a half has a zero mean difference")
    for name, v in (("snr_proj", snr_ab), ("snr_proj", snr_ba), ("snr_trace", tr_ab), ("snr_trace", tr_ba)):
        if math.isnan(v):
            problems.append(f"{name} undefined (zero variance or zero direction)")
    status = "undetermined" if problems else "ok"
    return DirectionResult(
        stream=stream,
        layer=int(layer),
        status=status,
        reason="; ".join(sorted(set(problems))),
        direction=nan_vec if d_all is None else d_all,
        direction_a=nan_vec if d_a is None else d_a,
        direction_b=nan_vec if d_b is None else d_b,
        n_pos=int(P.shape[0]),
        n_neg=int(N.shape[0]),
        dim=int(P.shape[1]),
        seed=int(seed),
        snr_proj=_mean2(snr_ab, snr_ba),
        snr_proj_a_to_b=snr_ab,
        snr_proj_b_to_a=snr_ba,
        snr_trace=_mean2(tr_ab, tr_ba),
        snr_trace_a_to_b=tr_ab,
        snr_trace_b_to_a=tr_ba,
        split_half_cosine=cos,
    )


# ---------------------------------------------------------------------- hook capture


@dataclass(frozen=True)
class HookPoint:
    """One capture site: results are keyed (stream, layer). `module` is an
    `nn.Module` or a dotted name resolved with `model.get_submodule`."""

    stream: str
    layer: int
    module: Any


def _norm_hook_points(hook_points: Any) -> List[HookPoint]:
    out: List[HookPoint] = []
    if isinstance(hook_points, Mapping):
        items: Iterable[Any] = [(k[0], k[1], v) for k, v in hook_points.items()]
    else:
        items = hook_points
    for hp in items:
        if isinstance(hp, HookPoint):
            out.append(hp)
        else:
            stream, layer, module = hp
            out.append(HookPoint(str(stream), int(layer), module))
    if not out:
        raise ValueError("hook_points is empty")
    keys = [(h.stream, h.layer) for h in out]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (stream, layer) in hook_points")
    return out


def collect_activations(
    model: Any,
    prompts: Sequence[Any],
    hook_points: Any,
    *,
    encode: Optional[Callable[[Sequence[Any]], Any]] = None,
    batch_size: int = 8,
    position: Union[int, str] = -1,
) -> Dict[Key, np.ndarray]:
    """Run `model` over `prompts` with forward hooks; return {(stream, layer): [n, d] float64}.

    A hook output may be a tensor or a tuple/list whose first element is the
    tensor. [B, T, d] outputs are reduced by `position` (an int index, or
    "mean" for a mean over T); [B, d] outputs are used as is. Padding is the
    caller's concern: with batched, padded text use left padding (or batch_size=1)
    so index -1 is the last real token.
    """
    import torch  # local: the numpy-only parts of this module work without torch

    points = _norm_hook_points(hook_points)
    if encode is None:

        def encode(batch: Sequence[Any]) -> Any:  # type: ignore[misc]
            if not all(isinstance(b, torch.Tensor) for b in batch):
                raise TypeError("prompts are not tensors; pass encode= to turn a list of prompts into model inputs")
            return torch.stack(list(batch))

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not (isinstance(position, int) or position == "mean"):
        raise ValueError("position must be an int or 'mean'")

    store: Dict[Key, List[np.ndarray]] = {(h.stream, h.layer): [] for h in points}
    handles = []
    resolved = []
    for h in points:
        mod = model.get_submodule(h.module) if isinstance(h.module, str) else h.module

        def make(key: Key) -> Callable[..., None]:
            def hook(_m: Any, _inp: Any, out: Any) -> None:
                t = out[0] if isinstance(out, (tuple, list)) else out
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"hook {key}: output is {type(out).__name__}, not a tensor")
                if t.ndim == 3:
                    t = t.mean(dim=1) if position == "mean" else t[:, position, :]
                elif t.ndim != 2:
                    raise ValueError(f"hook {key}: output has {t.ndim} dims; expected [B, T, d] or [B, d]")
                store[key].append(t.detach().to(torch.float64).cpu().numpy())

            return hook

        resolved.append((h, mod))
        handles.append(mod.register_forward_hook(make((h.stream, h.layer))))

    was_training = bool(getattr(model, "training", False))
    try:
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                inputs = encode(prompts[i : i + batch_size])
                if isinstance(inputs, Mapping):
                    model(**inputs)
                elif isinstance(inputs, (tuple, list)):
                    model(*inputs)
                else:
                    model(inputs)
    finally:
        for hd in handles:
            hd.remove()
        if was_training and hasattr(model, "train"):
            model.train()

    out: Dict[Key, np.ndarray] = {}
    for key, chunks in store.items():
        if not chunks:
            raise RuntimeError(f"hook {key} never fired: the module is not on the forward path for these prompts")
        out[key] = np.concatenate(chunks, axis=0)
        if out[key].shape[0] != len(prompts):
            raise RuntimeError(f"hook {key} captured {out[key].shape[0]} rows for {len(prompts)} prompts (module called more than once per prompt?)")
    return out


# ---------------------------------------------------------------------- public entry point


def capture_contrastive_directions(
    pos: Any = None,
    neg: Any = None,
    *,
    activations: Optional[Mapping[Key, Tuple[Any, Any]]] = None,
    stream: str = "default",
    layer: int = 0,
    model: Any = None,
    pos_prompts: Optional[Sequence[Any]] = None,
    neg_prompts: Optional[Sequence[Any]] = None,
    hook_points: Any = None,
    encode: Optional[Callable[[Sequence[Any]], Any]] = None,
    batch_size: int = 8,
    position: Union[int, str] = -1,
    seed: int = 0,
    min_per_class: int = 4,
) -> Dict[Key, DirectionResult]:
    """Compute difference-of-means directions plus held-out metrics per (stream, layer).

    Exactly one input mode:

    1. `pos`, `neg` arrays/tensors [n, d] for one cell (named by `stream`, `layer`);
    2. `activations={(stream, layer): (pos, neg), ...}` for many cells;
    3. `model` + `pos_prompts` + `neg_prompts` + `hook_points`: activations
       are captured with forward hooks (see `collect_activations`).

    Returns {(stream, layer): DirectionResult}. Feed it to `write_vectors`.
    """
    modes = [pos is not None or neg is not None, activations is not None, model is not None]
    if sum(modes) != 1:
        raise ValueError("give exactly one of: pos/neg, activations=, or model= with prompts and hook_points")
    cells: Dict[Key, Tuple[Any, Any]]
    if modes[0]:
        if pos is None or neg is None:
            raise ValueError("pos and neg must be given together")
        cells = {(stream, int(layer)): (pos, neg)}
    elif modes[1]:
        assert activations is not None
        cells = {(str(k[0]), int(k[1])): v for k, v in activations.items()}
        if not cells:
            raise ValueError("activations is empty")
    else:
        if pos_prompts is None or neg_prompts is None or hook_points is None:
            raise ValueError("model mode needs pos_prompts, neg_prompts and hook_points")
        kw = dict(encode=encode, batch_size=batch_size, position=position)
        ap = collect_activations(model, pos_prompts, hook_points, **kw)
        an = collect_activations(model, neg_prompts, hook_points, **kw)
        cells = {k: (ap[k], an[k]) for k in ap}
    return {
        key: evaluate_cell(p, n, stream=key[0], layer=key[1], seed=seed, min_per_class=min_per_class)
        for key, (p, n) in cells.items()
    }


# ---------------------------------------------------------------------- vectors on disk


def write_vectors(
    directory: Union[str, "os.PathLike[str]"],
    results: Union[Mapping[Key, DirectionResult], Iterable[DirectionResult]],
    *,
    overwrite: bool = False,
) -> Dict[Key, str]:
    """Write `d_<stream>_L<layer>.npy` (NumPy v1.0, little-endian float64, 1-D) per result.

    These are the files `secant compose --vectors` and `secant heal --vectors`
    read. Refuses undetermined (NaN) directions, an invalid stream token, and
    an existing file unless `overwrite`. Nothing is written if any check
    fails. Metrics are deliberately NOT written into this directory: the CLI
    reads every .json in a vectors directory as a vector. Dump
    `[r.metrics() for r in results.values()]` elsewhere.
    """
    items = list(results.values()) if isinstance(results, Mapping) else list(results)
    if not items:
        raise ValueError("no results to write")
    directory = os.fspath(directory)
    plan: List[Tuple[Key, str, np.ndarray]] = []
    seen = set()
    for r in items:
        if r.status != "ok":
            raise ValueError(f"({r.stream}, L{r.layer}) is {r.status}: {r.reason}; refusing to write a vector")
        if not _TOKEN_RE.match(r.stream):
            raise ValueError(f"stream {r.stream!r} is not a valid filename token ([A-Za-z0-9][A-Za-z0-9_-]*)")
        if r.layer < 0:
            raise ValueError("layer must be >= 0")
        path = os.path.join(directory, r.stem + ".npy")
        if path in seen:
            raise ValueError(f"duplicate output {path}")
        seen.add(path)
        if os.path.lexists(path) and not overwrite:
            raise FileExistsError(path)
        vec = np.ascontiguousarray(r.direction, dtype="<f8")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"({r.stream}, L{r.layer}) direction is not finite")
        plan.append(((r.stream, r.layer), path, vec))
    os.makedirs(directory, exist_ok=True)
    out: Dict[Key, str] = {}
    for key, path, vec in plan:
        with open(path, "wb") as f:
            np.lib.format.write_array(f, vec, version=(1, 0), allow_pickle=False)
        out[key] = path
    return out
