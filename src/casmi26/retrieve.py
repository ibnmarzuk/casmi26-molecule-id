"""Precursor-filtered spectral library search + rank aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .adducts import ppm_delta, precursor_to_neutral
from .similarity import fused_score
from .spectra import clean_peaks


@dataclass
class SearchConfig:
    ppm_tol: float = 15.0
    mz_tol: float = 0.02
    intensity_floor: float = 0.01
    max_peaks: int = 64
    precursor_pad_da: float = 2.0
    min_peaks: int = 3
    top_k_spectra: int = 400
    top_k_smiles: int = 25
    sqrt_intensities: bool = True
    require_adduct_match: bool = False
    weight_modified_cosine: float = 0.55
    weight_entropy: float = 0.45
    rrf_k: int = 60
    min_score: float = 0.02


@dataclass
class LibrarySpectrum:
    precursor_mz: float
    adduct: str
    smiles: str
    mz: np.ndarray
    intensity: np.ndarray
    ingest_lib: str = ""
    instrument_type: str = ""


@dataclass
class Library:
    spectra: list[LibrarySpectrum] = field(default_factory=list)
    precursor: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    @classmethod
    def from_frame(cls, df: pd.DataFrame, cfg: SearchConfig) -> "Library":
        rows: list[LibrarySpectrum] = []
        for rec in df.itertuples(index=False):
            mz, inten = clean_peaks(
                getattr(rec, "ms2_mzs"),
                getattr(rec, "ms2_normalized_intensities"),
                float(getattr(rec, "precursor_mz")),
                intensity_floor=cfg.intensity_floor,
                max_peaks=cfg.max_peaks,
                precursor_pad_da=cfg.precursor_pad_da,
                min_peaks=cfg.min_peaks,
                sqrt_intensities=cfg.sqrt_intensities,
            )
            smiles = str(getattr(rec, "normalized_smiles", "") or "")
            if not smiles or mz.size < cfg.min_peaks:
                continue
            rows.append(
                LibrarySpectrum(
                    precursor_mz=float(getattr(rec, "precursor_mz")),
                    adduct=str(getattr(rec, "adduct", "") or ""),
                    smiles=smiles,
                    mz=mz,
                    intensity=inten,
                    ingest_lib=str(getattr(rec, "ingest_lib", "") or ""),
                    instrument_type=str(getattr(rec, "instrument_type", "") or ""),
                )
            )
        lib = cls(spectra=rows)
        lib.precursor = np.array([s.precursor_mz for s in rows], dtype=np.float64)
        return lib


def _candidate_indices(library: Library, precursor_mz: float, adduct: str, cfg: SearchConfig) -> np.ndarray:
    if library.precursor.size == 0:
        return np.zeros(0, dtype=np.int64)
    window = ppm_delta(precursor_mz, cfg.ppm_tol)
    mask = np.abs(library.precursor - precursor_mz) <= window
    if cfg.require_adduct_match and adduct:
        adducts = np.array([s.adduct for s in library.spectra])
        mask = mask & (adducts == adduct)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        # fallback: nearest precursor masses
        dist = np.abs(library.precursor - precursor_mz)
        k = min(cfg.top_k_spectra, dist.size)
        return np.argpartition(dist, k - 1)[:k]
    if idx.size > cfg.top_k_spectra:
        dist = np.abs(library.precursor[idx] - precursor_mz)
        keep = np.argpartition(dist, cfg.top_k_spectra - 1)[: cfg.top_k_spectra]
        idx = idx[keep]
    return idx


def search_spectrum(
    mz, intensity, precursor_mz: float, adduct: str, library: Library, cfg: SearchConfig
) -> list[tuple[str, float]]:
    q_mz, q_int = clean_peaks(
        mz,
        intensity,
        precursor_mz,
        intensity_floor=cfg.intensity_floor,
        max_peaks=cfg.max_peaks,
        precursor_pad_da=cfg.precursor_pad_da,
        min_peaks=cfg.min_peaks,
        sqrt_intensities=cfg.sqrt_intensities,
    )
    idx = _candidate_indices(library, precursor_mz, adduct, cfg)
    scored: dict[str, float] = {}
    for j in idx:
        spec = library.spectra[int(j)]
        score = fused_score(
            q_mz,
            q_int,
            spec.mz,
            spec.intensity,
            precursor_mz,
            spec.precursor_mz,
            tol=cfg.mz_tol,
            w_cos=cfg.weight_modified_cosine,
            w_ent=cfg.weight_entropy,
        )
        if score < cfg.min_score:
            continue
        prev = scored.get(spec.smiles, 0.0)
        if score > prev:
            scored[spec.smiles] = score
    return sorted(scored.items(), key=lambda kv: kv[1], reverse=True)


def reciprocal_rank_fusion(rank_lists: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in rank_lists:
        for rank, (smiles, _score) in enumerate(ranked, start=1):
            scores[smiles] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def mass_fallback(library: Library, precursor_mz: float, adduct: str, limit: int) -> list[str]:
    if not library.spectra:
        return []
    try:
        target = precursor_to_neutral(precursor_mz, adduct)
        masses = np.array(
            [precursor_to_neutral(s.precursor_mz, s.adduct) for s in library.spectra],
            dtype=np.float64,
        )
        dist = np.abs(masses - target)
    except Exception:
        dist = np.abs(library.precursor - precursor_mz)
    order = np.argsort(dist)
    out: list[str] = []
    seen: set[str] = set()
    for i in order:
        smi = library.spectra[int(i)].smiles
        if smi in seen:
            continue
        seen.add(smi)
        out.append(smi)
        if len(out) >= limit:
            break
    return out


def predict_molecule(
    spectrum_rows: pd.DataFrame, library: Library, cfg: SearchConfig
) -> list[str]:
    lists: list[list[tuple[str, float]]] = []
    for rec in spectrum_rows.itertuples(index=False):
        ranked = search_spectrum(
            getattr(rec, "ms2_mzs"),
            getattr(rec, "ms2_normalized_intensities"),
            float(getattr(rec, "precursor_mz")),
            str(getattr(rec, "adduct", "") or ""),
            library,
            cfg,
        )
        lists.append(ranked)
    fused = reciprocal_rank_fusion(lists, k=cfg.rrf_k)
    smiles = [s for s, _ in fused[: cfg.top_k_smiles]]
    if len(smiles) < cfg.top_k_smiles:
        first = spectrum_rows.iloc[0]
        extra = mass_fallback(
            library,
            float(first["precursor_mz"]),
            str(first.get("adduct", "") or ""),
            cfg.top_k_smiles,
        )
        for smi in extra:
            if smi not in smiles:
                smiles.append(smi)
            if len(smiles) >= cfg.top_k_smiles:
                break
    return smiles[: cfg.top_k_smiles]
