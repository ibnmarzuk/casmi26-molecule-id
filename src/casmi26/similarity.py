"""Modified cosine and spectral-entropy similarity."""

from __future__ import annotations

import numpy as np


def _greedy_match(
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    shift: float,
    tol: float,
) -> tuple[float, float, float]:
    """Return (dot, na, nb) using a one-to-one greedy m/z match."""
    if mz_a.size == 0 or mz_b.size == 0:
        return 0.0, float(np.dot(int_a, int_a)) if mz_a.size else 0.0, float(np.dot(int_b, int_b)) if mz_b.size else 0.0

    used = np.zeros(mz_b.size, dtype=bool)
    dot = 0.0
    for i, (ma, ia) in enumerate(zip(mz_a, int_a)):
        best_j = -1
        best_d = tol + 1.0
        for j, (mb, _ib) in enumerate(zip(mz_b, int_b)):
            if used[j]:
                continue
            d_direct = abs(ma - mb)
            d_shift = abs(ma - mb - shift)
            d = d_direct if d_direct <= d_shift else d_shift
            if d <= tol and d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            dot += ia * int_b[best_j]
    return dot, float(np.dot(int_a, int_a)), float(np.dot(int_b, int_b))


def modified_cosine(
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    precursor_a: float,
    precursor_b: float,
    tol: float = 0.02,
) -> float:
    shift = float(precursor_a) - float(precursor_b)
    dot, na, nb = _greedy_match(mz_a, int_a, mz_b, int_b, shift, tol)
    denom = np.sqrt(na * nb)
    if denom <= 0:
        return 0.0
    score = float(dot / denom)
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


def _entropy(intensities: np.ndarray) -> float:
    s = float(intensities.sum())
    if s <= 0:
        return 0.0
    p = intensities / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def entropy_similarity(
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    tol: float = 0.02,
) -> float:
    """Li et al. unweighted spectral entropy similarity in [0, 1]."""
    if mz_a.size == 0 or mz_b.size == 0:
        return 0.0
    ia = np.clip(int_a, 0, None).astype(np.float64, copy=False)
    ib = np.clip(int_b, 0, None).astype(np.float64, copy=False)
    mixed_mz = []
    mixed_int = []
    used = np.zeros(mz_b.size, dtype=bool)
    for ma, a in zip(mz_a, ia):
        best_j = -1
        best_d = tol + 1.0
        for j, mb in enumerate(mz_b):
            if used[j]:
                continue
            d = abs(ma - mb)
            if d <= tol and d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            mixed_mz.append(0.5 * (ma + mz_b[best_j]))
            mixed_int.append(a + ib[best_j])
        else:
            mixed_mz.append(float(ma))
            mixed_int.append(float(a))
    for j, (mb, b) in enumerate(zip(mz_b, ib)):
        if not used[j]:
            mixed_mz.append(float(mb))
            mixed_int.append(float(b))
    mixed = np.asarray(mixed_int, dtype=np.float64)
    ea = _entropy(ia)
    eb = _entropy(ib)
    em = _entropy(mixed)
    denom = ea + eb
    if denom <= 0:
        return 0.0
    score = 1.0 - (2.0 * em - ea - eb) / denom
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return float(score)


def fused_score(
    mz_a, int_a, mz_b, int_b, precursor_a, precursor_b, tol=0.02, w_cos=0.55, w_ent=0.45
) -> float:
    c = modified_cosine(mz_a, int_a, mz_b, int_b, precursor_a, precursor_b, tol)
    e = entropy_similarity(mz_a, int_a, mz_b, int_b, tol)
    return w_cos * c + w_ent * e
