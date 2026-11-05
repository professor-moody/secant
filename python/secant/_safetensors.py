"""Minimal dependency-free SafeTensors reader/writer (F32 and F64, numpy).

Writes the standard layout: u64 little-endian header length, JSON header
padded with spaces to an 8-byte boundary, then tensor bytes in sorted-name
order with contiguous offsets. Only what the `secant heal` LoRA input needs.
"""

from __future__ import annotations

import json
import struct
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

_DTYPES = {"F32": np.dtype("<f4"), "F64": np.dtype("<f8")}
_NAMES = {np.dtype("<f4"): "F32", np.dtype("<f8"): "F64"}
_MAX_HEADER = 1 << 27


def write_safetensors(
    path: str,
    tensors: Mapping[str, np.ndarray],
    metadata: Optional[Mapping[str, str]] = None,
    *,
    overwrite: bool = False,
) -> str:
    """Write `tensors` (float32/float64 arrays) to a NEW file at `path`."""
    if not tensors:
        raise ValueError("refusing to write an empty SafeTensors file")
    header: Dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blobs = []
    offset = 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        dt = arr.dtype.newbyteorder("<")
        if dt not in _NAMES:
            raise ValueError(f"tensor {name!r}: dtype {arr.dtype} unsupported (float32/float64 only)")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"tensor {name!r} holds a non-finite value; refusing to export it")
        data = arr.astype(dt, copy=False).tobytes(order="C")
        header[name] = {"dtype": _NAMES[dt], "shape": list(arr.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    with open(path, "wb" if overwrite else "xb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)
    return path


def read_safetensors(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Read a SafeTensors file written with F32/F64 tensors. Returns (tensors, metadata)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n < 2 or n > _MAX_HEADER:
            raise ValueError(f"implausible SafeTensors header length {n}")
        header = json.loads(f.read(n).decode("utf-8"))
        base = 8 + n
        out: Dict[str, np.ndarray] = {}
        for name, ent in header.items():
            if name == "__metadata__":
                continue
            dt = _DTYPES.get(ent["dtype"])
            if dt is None:
                raise ValueError(f"tensor {name!r}: dtype {ent['dtype']} unsupported by this reader")
            lo, hi = ent["data_offsets"]
            f.seek(base + lo)
            buf = f.read(hi - lo)
            if len(buf) != hi - lo:
                raise ValueError(f"tensor {name!r}: truncated payload")
            out[name] = np.frombuffer(buf, dtype=dt).reshape(ent["shape"]).copy()
    return out, dict(header.get("__metadata__", {}))
