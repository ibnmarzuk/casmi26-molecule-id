"""Peak cleaning shared by local training and the Kaggle notebook."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _as_float_array(values) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=np.float64)
    if isinstance(values, np.ndarray):
        arr = values.astype(np.float64, copy=False)
    else:
        arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def clean_peaks(
    mzs,
    intensities,
    precursor_mz: float,
    intensity_floor: float = 0.01,
    max_peaks: int = 64,
    precursor_pad_da: float = 2.0,
    min_peaks: int = 3,
    sqrt_intensities: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    mz = _as_float_array(mzs)
    inten = _as_float_array(intensities)
    n = min(mz.size, inten.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    mz = mz[:n]
    inten = inten[:n]
    inten = np.clip(inten, 0.0, None)
    peak_max = float(inten.max()) if inten.size else 0.0
    if peak_max <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    inten = inten / peak_max

    keep = (inten >= intensity_floor) & (mz > 0.5) & (mz <= precursor_mz + precursor_pad_da)
    mz, inten = mz[keep], inten[keep]
    if mz.size == 0:
        return mz, inten

    order = np.argsort(inten)[::-1][:max_peaks]
    mz, inten = mz[order], inten[order]
    mass_order = np.argsort(mz)
    mz, inten = mz[mass_order], inten[mass_order]
    if sqrt_intensities:
        inten = np.sqrt(inten)
        norm = float(np.linalg.norm(inten))
        if norm > 0:
            inten = inten / norm
    if mz.size < min_peaks:
        return mz, inten
    return mz, inten


def collision_energy_scalar(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        digits = "".join(ch if (ch.isdigit() or ch == ".") else " " for ch in value)
        parts = [p for p in digits.split() if p]
        if not parts:
            return float("nan")
        return float(np.mean([float(p) for p in parts]))
    if isinstance(value, (list, tuple, np.ndarray)):
        nums = []
        for item in value:
            try:
                nums.append(float(item))
            except (TypeError, ValueError):
                continue
        return float(np.mean(nums)) if nums else float("nan")
    return float("nan")


def merge_spectra(
    peak_lists: Iterable[tuple[np.ndarray, np.ndarray]],
    bin_size: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    buckets: dict[int, float] = {}
    for mz, inten in peak_lists:
        for m, i in zip(mz, inten):
            key = int(round(m / bin_size))
            buckets[key] = max(buckets.get(key, 0.0), float(i))
    if not buckets:
        return np.zeros(0), np.zeros(0)
    keys = np.array(sorted(buckets), dtype=np.int64)
    mz = keys.astype(np.float64) * bin_size
    inten = np.array([buckets[int(k)] for k in keys], dtype=np.float64)
    peak_max = float(inten.max())
    if peak_max > 0:
        inten = inten / peak_max
    return mz, inten


def l2_normalize(intensities: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(intensities))
    if n == 0 or not math.isfinite(n):
        return intensities
    return intensities / n
