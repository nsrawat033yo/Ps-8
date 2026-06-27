from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def robust_normalize(arr: np.ndarray, valid: np.ndarray | None = None, q_low: float = 2, q_high: float = 98) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    if valid is None:
        valid = np.isfinite(arr)
    vals = arr[valid & np.isfinite(arr)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float32")
    lo, hi = np.nanpercentile(vals, [q_low, q_high])
    if not np.isfinite(hi - lo) or hi <= lo:
        return np.zeros_like(arr, dtype="float32")
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    out[~valid | ~np.isfinite(out)] = 0
    return out.astype("float32")


def local_std(arr: np.ndarray, size: int = 9, valid: np.ndarray | None = None) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    if valid is None:
        valid = np.isfinite(arr)
    filled = np.where(valid, arr, np.nanmedian(arr[valid]) if np.any(valid) else 0.0)
    mean = ndi.uniform_filter(filled, size=size)
    mean_sq = ndi.uniform_filter(filled * filled, size=size)
    var = np.maximum(mean_sq - mean * mean, 0)
    out = np.sqrt(var).astype("float32")
    out[~valid] = np.nan
    return out


def local_mean(arr: np.ndarray, size: int = 9, valid: np.ndarray | None = None) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    if valid is None:
        valid = np.isfinite(arr)
    filled = np.where(valid, arr, np.nanmedian(arr[valid]) if np.any(valid) else 0.0)
    out = ndi.uniform_filter(filled, size=size).astype("float32")
    out[~valid] = np.nan
    return out


def gradient_magnitude(arr: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    if valid is None:
        valid = np.isfinite(arr)
    filled = np.where(valid, arr, np.nanmedian(arr[valid]) if np.any(valid) else 0.0)
    gy, gx = np.gradient(filled)
    grad = np.sqrt(gx * gx + gy * gy).astype("float32")
    grad[~valid] = np.nan
    return grad
