# Enveda CASMI 2026 — library-search submission
# Paste this into a Kaggle Notebook attached to the competition data.
# Settings for a scoring run: Internet OFF, GPU optional, runtime ≤ 9h.
# Output file MUST be named submission.csv (this script writes /kaggle/working/submission.csv).

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

COMP_DIR = Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra")
if not COMP_DIR.exists():
    COMP_DIR = Path("data/raw") if Path("data/raw").exists() else Path(".")

OUT_PATH = Path("/kaggle/working/submission.csv")
if not Path("/kaggle/working").exists():
    OUT_PATH = Path("outputs/submission.csv")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

PPM_TOL = 15.0
MZ_TOL = 0.02
INTENSITY_FLOOR = 0.01
MAX_PEAKS = 64
PRECURSOR_PAD = 2.0
MIN_PEAKS = 3
TOP_K_SPECTRA = 350
TOP_K_SMILES = 25
W_COS = 0.55
W_ENT = 0.45
RRF_K = 60
MIN_SCORE = 0.015
BATCH_ROWS = 40_000
PROTON = 1.007276466812

ADDUCT_DELTA = {
    "[M+H]+": PROTON,
    "[M+NH4]+": 18.0338255678,
    "[M-H2O+H]+": PROTON - 18.010564684,
    "[M-2H2O+H]+": PROTON - 2 * 18.010564684,
    "[M+Na]+": 22.989269282,
    "[M+K]+": 38.9631579,
    "[M-H]-": -PROTON,
    "[M-H2O-H]-": -18.010564684 - PROTON,
    "[M+CH2O2-H]-": 46.0054793034 - PROTON,
    "[M+Cl]-": 34.969402203,
    "[M+H-H2O]+": PROTON - 18.010564684,
    "[M+HCOO]-": 46.0054793034 - PROTON,
}


def ppm_window(mass: float, ppm: float) -> float:
    return abs(mass) * ppm * 1e-6


def to_array(values) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=np.float64)
    if isinstance(values, np.ndarray):
        arr = values.astype(np.float64, copy=False)
    else:
        try:
            arr = np.asarray(values, dtype=np.float64)
        except Exception:
            arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def clean_peaks(mzs, intensities, precursor_mz: float):
    mz = to_array(mzs)
    inten = to_array(intensities)
    n = min(mz.size, inten.size)
    if n == 0:
        return mz[:0], inten[:0]
    mz, inten = mz[:n], np.clip(inten[:n], 0.0, None)
    peak_max = float(inten.max()) if inten.size else 0.0
    if peak_max <= 0:
        return mz[:0], inten[:0]
    inten = inten / peak_max
    keep = (inten >= INTENSITY_FLOOR) & (mz > 0.5) & (mz <= precursor_mz + PRECURSOR_PAD)
    mz, inten = mz[keep], inten[keep]
    if mz.size == 0:
        return mz, inten
    order = np.argsort(inten)[::-1][:MAX_PEAKS]
    mz, inten = mz[order], inten[order]
    mass_order = np.argsort(mz)
    mz, inten = mz[mass_order], inten[mass_order]
    inten = np.sqrt(inten)
    norm = float(np.linalg.norm(inten))
    if norm > 0:
        inten = inten / norm
    return mz, inten


def greedy_dot(mz_a, int_a, mz_b, int_b, shift, tol):
    if mz_a.size == 0 or mz_b.size == 0:
        return 0.0, float(np.dot(int_a, int_a)) if mz_a.size else 0.0, float(np.dot(int_b, int_b)) if mz_b.size else 0.0
    used = np.zeros(mz_b.size, dtype=bool)
    dot = 0.0
    for ma, ia in zip(mz_a, int_a):
        best_j = -1
        best_d = tol + 1.0
        lo = np.searchsorted(mz_b, ma - shift - tol)
        hi = np.searchsorted(mz_b, ma + abs(shift) + tol)
        lo = max(0, int(lo) - 2)
        hi = min(mz_b.size, int(hi) + 2)
        for j in range(lo, hi):
            if used[j]:
                continue
            d_direct = abs(ma - mz_b[j])
            d_shift = abs(ma - mz_b[j] - shift)
            d = d_direct if d_direct <= d_shift else d_shift
            if d <= tol and d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            dot += ia * int_b[best_j]
    return dot, float(np.dot(int_a, int_a)), float(np.dot(int_b, int_b))


def modified_cosine(mz_a, int_a, mz_b, int_b, prec_a, prec_b, tol=MZ_TOL):
    dot, na, nb = greedy_dot(mz_a, int_a, mz_b, int_b, float(prec_a) - float(prec_b), tol)
    denom = np.sqrt(na * nb)
    if denom <= 0:
        return 0.0
    return float(np.clip(dot / denom, 0.0, 1.0))


def spectral_entropy(intensities):
    s = float(intensities.sum())
    if s <= 0:
        return 0.0
    p = intensities / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def entropy_similarity(mz_a, int_a, mz_b, int_b, tol=MZ_TOL):
    if mz_a.size == 0 or mz_b.size == 0:
        return 0.0
    ia = np.clip(int_a, 0, None)
    ib = np.clip(int_b, 0, None)
    mixed = []
    used = np.zeros(mz_b.size, dtype=bool)
    for ma, a in zip(mz_a, ia):
        best_j = -1
        best_d = tol + 1.0
        lo = max(0, int(np.searchsorted(mz_b, ma - tol)) - 1)
        hi = min(mz_b.size, int(np.searchsorted(mz_b, ma + tol)) + 1)
        for j in range(lo, hi):
            if used[j]:
                continue
            d = abs(ma - mz_b[j])
            if d <= tol and d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            mixed.append(a + ib[best_j])
        else:
            mixed.append(float(a))
    for j, b in enumerate(ib):
        if not used[j]:
            mixed.append(float(b))
    mixed = np.asarray(mixed, dtype=np.float64)
    ea, eb, em = spectral_entropy(ia), spectral_entropy(ib), spectral_entropy(mixed)
    denom = ea + eb
    if denom <= 0:
        return 0.0
    return float(np.clip(1.0 - (2.0 * em - ea - eb) / denom, 0.0, 1.0))


def fused_score(mz_a, int_a, mz_b, int_b, prec_a, prec_b):
    return W_COS * modified_cosine(mz_a, int_a, mz_b, int_b, prec_a, prec_b) + W_ENT * entropy_similarity(
        mz_a, int_a, mz_b, int_b
    )


def rrf(rank_lists, k=RRF_K):
    scores = defaultdict(float)
    for ranked in rank_lists:
        for rank, (smi, _) in enumerate(ranked, start=1):
            scores[smi] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def format_field(smiles):
    out, seen = [], set()
    for smi in smiles:
        s = str(smi).strip()
        if not s or s in seen or "," in s or "\n" in s or ";" in s:
            continue
        seen.add(s)
        out.append(s)
        if len(out) == TOP_K_SMILES:
            break
    if not out:
        out = ["C"]
    return ";".join(out)


print("Loading test spectra…")
test = pd.read_parquet(COMP_DIR / "test.parquet")
print(f"test rows={len(test):,} molecules={test['molecule_id'].nunique():,}")

centers = test["precursor_mz"].astype(float).to_numpy()
windows = np.array([ppm_window(c, PPM_TOL * 2.5) for c in centers])


def hits_precursor(mz_series: pd.Series) -> pd.Series:
    mz = mz_series.astype(float).to_numpy()
    keep = np.zeros(mz.size, dtype=bool)
    # coarse 0.5 Da buckets first, then exact ppm
    buckets = defaultdict(list)
    for i, c in enumerate(centers):
        buckets[int(round(c * 2))].append(i)
    keys = (mz * 2).round().astype(int)
    cand_idx = []
    for row_i, key in enumerate(keys):
        hits = []
        for k in (key - 1, key, key + 1):
            hits.extend(buckets.get(k, ()))
        if not hits:
            continue
        if any(abs(mz[row_i] - centers[j]) <= windows[j] for j in hits):
            cand_idx.append(row_i)
    mask = np.zeros(mz.size, dtype=bool)
    mask[cand_idx] = True
    return pd.Series(mask, index=mz_series.index)


print("Indexing training spectra near test precursors…")
wanted = [
    "normalized_smiles",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "precursor_mz",
    "adduct",
    "ingest_lib",
]
train_path = COMP_DIR / "train.parquet"
chunks = []
try:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(train_path)
    n_groups = pf.num_row_groups
    for gi in range(n_groups):
        df = pf.read_row_group(gi, columns=wanted).to_pandas()
        mask = hits_precursor(df["precursor_mz"])
        hit = df.loc[mask]
        if len(hit):
            chunks.append(hit)
        if (gi + 1) % 5 == 0:
            print(f"  row-group {gi + 1}/{n_groups} kept={sum(len(c) for c in chunks):,}")
except Exception as exc:
    print("row-group scan failed, falling back to pandas:", exc)
    df = pd.read_parquet(train_path, columns=wanted)
    chunks = [df.loc[hits_precursor(df["precursor_mz"])]]

if not chunks:
    print("No precursor matches — loading a train head as fallback.")
    chunks = [pd.read_parquet(train_path, columns=wanted).head(8000)]

train = pd.concat(chunks, ignore_index=True)
train = train.dropna(subset=["normalized_smiles", "precursor_mz"])
print(f"library spectra={len(train):,} unique smiles={train['normalized_smiles'].nunique():,}")

lib_mz = []
lib_int = []
lib_prec = []
lib_smi = []
lib_adduct = []
lib_weight = []
for rec in train.itertuples(index=False):
    mz, inten = clean_peaks(rec.ms2_mzs, rec.ms2_normalized_intensities, float(rec.precursor_mz))
    if mz.size < MIN_PEAKS:
        continue
    smi = str(rec.normalized_smiles)
    if not smi:
        continue
    lib_mz.append(mz)
    lib_int.append(inten)
    lib_prec.append(float(rec.precursor_mz))
    lib_smi.append(smi)
    lib_adduct.append(str(getattr(rec, "adduct", "") or ""))
    src = str(getattr(rec, "ingest_lib", "") or "")
    w = 1.15 if src in {"enveda-np-examples", "enveda-180"} else 1.0
    if src in {"gnps", "riken", "massbank", "mona"}:
        w = 1.08
    lib_weight.append(w)

lib_prec = np.asarray(lib_prec, dtype=np.float64)
lib_weight = np.asarray(lib_weight, dtype=np.float64)
print(f"cleaned library={lib_prec.size:,}")


def search_one(mzs, intensities, precursor_mz, adduct):
    q_mz, q_int = clean_peaks(mzs, intensities, precursor_mz)
    window = ppm_window(precursor_mz, PPM_TOL)
    idx = np.flatnonzero(np.abs(lib_prec - precursor_mz) <= window)
    if idx.size == 0:
        dist = np.abs(lib_prec - precursor_mz)
        k = min(TOP_K_SPECTRA, dist.size)
        idx = np.argpartition(dist, k - 1)[:k]
    elif idx.size > TOP_K_SPECTRA:
        dist = np.abs(lib_prec[idx] - precursor_mz)
        idx = idx[np.argpartition(dist, TOP_K_SPECTRA - 1)[:TOP_K_SPECTRA]]
    scored = {}
    for j in idx:
        j = int(j)
        score = fused_score(q_mz, q_int, lib_mz[j], lib_int[j], precursor_mz, lib_prec[j]) * lib_weight[j]
        if score < MIN_SCORE:
            continue
        smi = lib_smi[j]
        if score > scored.get(smi, 0.0):
            scored[smi] = score
    return sorted(scored.items(), key=lambda kv: kv[1], reverse=True)


def mass_fill(precursor_mz, adduct, already, need):
    delta = ADDUCT_DELTA.get(str(adduct), PROTON)
    target = precursor_mz - delta
    dist = np.abs((lib_prec - np.array([ADDUCT_DELTA.get(a, PROTON) for a in lib_adduct])) - target)
    order = np.argsort(dist)
    out = []
    seen = set(already)
    for j in order:
        smi = lib_smi[int(j)]
        if smi in seen:
            continue
        seen.add(smi)
        out.append(smi)
        if len(out) >= need:
            break
    return out


print("Scoring test molecules…")
rows = []
n_mol = test["molecule_id"].nunique()
for i, (mol_id, group) in enumerate(test.groupby("molecule_id", sort=False), start=1):
    lists = []
    for rec in group.itertuples(index=False):
        lists.append(
            search_one(
                rec.ms2_mzs,
                rec.ms2_normalized_intensities,
                float(rec.precursor_mz),
                str(getattr(rec, "adduct", "") or ""),
            )
        )
    fused = rrf(lists)
    smiles = [s for s, _ in fused[:TOP_K_SMILES]]
    if len(smiles) < TOP_K_SMILES:
        first = group.iloc[0]
        smiles.extend(
            mass_fill(
                float(first["precursor_mz"]),
                str(first.get("adduct", "") or ""),
                smiles,
                TOP_K_SMILES - len(smiles),
            )
        )
    rows.append({"molecule_id": mol_id, "smiles": format_field(smiles)})
    if i % 25 == 0 or i == n_mol:
        print(f"  {i}/{n_mol}")

sub = pd.DataFrame(rows)
assert sub["molecule_id"].is_unique
assert set(test["molecule_id"]) == set(sub["molecule_id"])
assert not sub.isna().any().any()
assert (sub["smiles"].str.split(";").map(len) <= 25).all()
sub.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH}  rows={len(sub)}")
print(sub.head())
