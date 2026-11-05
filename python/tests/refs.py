"""Numpy reference implementations used by several test modules."""

import numpy as np


def q8_0_roundtrip(w, fp16_scale=True):
    """Q8_0 quantize+dequantize reference: per 32-element block along each row,
    d = amax/127, q = round-half-away(x/d), dequant = q*d. ggml stores d as
    fp16 (fp16_scale=True); fp16_scale=False keeps it float32."""
    blocks = w.reshape(w.shape[0], -1, 32).astype(np.float32)
    amax = np.max(np.abs(blocks), axis=2, keepdims=True)
    d = (amax / np.float32(127.0)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    scaled = blocks * idv
    q = np.sign(scaled) * np.floor(np.abs(scaled) + np.float32(0.5))
    ds = d.astype(np.float16).astype(np.float32) if fp16_scale else d
    return (q * ds).reshape(w.shape).astype(np.float64)


def q8_0_witness(dirs, w, fp16_scale=True):
    """(l2, max_abs) of D (Dequant(Quant(W)) - W), float64."""
    err = q8_0_roundtrip(w, fp16_scale) - w.astype(np.float64)
    ref = dirs.astype(np.float64) @ err
    return float(np.sqrt((ref**2).sum())), float(np.abs(ref).max())
