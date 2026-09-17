#!/usr/bin/env python3
"""Build a tiny synthetic library + test set and write a valid submission.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casmi26.metric import format_smiles_field  # noqa: E402
from casmi26.retrieve import Library, SearchConfig, predict_molecule  # noqa: E402
from casmi26.io import write_submission  # noqa: E402

MOLECULES = [
    ("m_caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", 195.0876, "[M+H]+", [138.066, 110.071, 83.060, 69.045, 42.034]),
    ("m_tyramine", "NCCc1ccc(O)cc1", 138.0913, "[M+H]+", [121.065, 103.054, 91.054, 77.039, 65.039]),
    ("m_salicylic", "OC(=O)c1ccccc1O", 139.0390, "[M+H]+", [121.028, 93.034, 65.039, 39.023]),
    ("m_glucose", "OCC1OC(O)C(O)C(O)C1O", 181.0707, "[M+H]+", [163.060, 145.050, 127.039, 85.028, 73.028]),
    ("m_chrysin", "O=c1cc(-c2ccccc2)oc2cc(O)cc(O)c12", 255.0652, "[M+H]+", [153.018, 129.034, 103.054, 77.039]),
    ("m_nicotine", "CN1CCCC1c1cccnc1", 163.1230, "[M+H]+", [132.081, 130.065, 117.057, 84.081]),
]


def spectrum_from_peaks(peaks, rng, noise=0.04):
    mz = np.array(peaks, dtype=np.float64) + rng.normal(0, 0.002, size=len(peaks))
    inten = np.linspace(1.0, 0.15, len(peaks)) + rng.uniform(0, noise, size=len(peaks))
    inten = np.clip(inten, 0.02, None)
    inten = inten / inten.max()
    return mz, inten


def main() -> None:
    rng = np.random.default_rng(26)
    train_rows = []
    test_rows = []
    for i, (mol_id, smiles, precursor, adduct, peaks) in enumerate(MOLECULES):
        for k, ce in enumerate((20, 40, 60)):
            mz, inten = spectrum_from_peaks(peaks, rng)
            rec = {
                "molecule_id": mol_id,
                "spectrum_id": f"{mol_id}_ce{ce}",
                "ms2_mzs": mz,
                "ms2_normalized_intensities": inten,
                "base_peak_intensity": float(10000 + 500 * k),
                "adduct": adduct,
                "ionization_mode": "positive",
                "instrument_type": "timsTOF",
                "precursor_mz": precursor + rng.normal(0, 0.0004),
                "collision_energy_ev": [ce],
                "collision_energy_orig": str(ce),
                "collision_energy_orig_units": "eV",
                "normalized_smiles": smiles,
                "inchikey14": smiles[:14],
                "ingest_lib": "enveda-np-examples" if i < 4 else "gnps",
            }
            train_rows.append(rec)
            if k < 2:
                test_rows.append({key: rec[key] for key in rec if key not in {"normalized_smiles", "inchikey14", "ingest_lib"}})

    # decoys so retrieval is not trivial
    decoys = [
        "CC1=CC(=O)C=CC1=O",
        "CC(=O)Nc1ccc(O)cc1",
        "OCC(O)CO",
        "CCO",
        "CC(C)O",
        "c1ccccc1",
        "CCN(CC)CC",
        "OC1CCCCC1",
    ]
    for d, smi in enumerate(decoys):
        mz = np.array([50.0 + 12 * d, 77.0, 91.0, 105.0], dtype=np.float64)
        inten = np.array([0.4, 1.0, 0.6, 0.3])
        train_rows.append(
            {
                "molecule_id": f"decoy_{d}",
                "spectrum_id": f"decoy_{d}",
                "ms2_mzs": mz,
                "ms2_normalized_intensities": inten,
                "base_peak_intensity": 2000.0,
                "adduct": "[M+H]+",
                "ionization_mode": "positive",
                "instrument_type": "timsTOF",
                "precursor_mz": 100.0 + d * 20,
                "collision_energy_ev": [40],
                "collision_energy_orig": "40",
                "collision_energy_orig_units": "eV",
                "normalized_smiles": smi,
                "inchikey14": smi[:14],
                "ingest_lib": "demo",
            }
        )

    train = pd.DataFrame(train_rows)
    test = pd.DataFrame(test_rows)
    demo_dir = ROOT / "data" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(demo_dir / "train.parquet", index=False)
    test.to_parquet(demo_dir / "test.parquet", index=False)

    cfg = SearchConfig(ppm_tol=25.0, top_k_spectra=80, min_score=0.0)
    library = Library.from_frame(train, cfg)
    rows = []
    traces = []
    for molecule_id, group in test.groupby("molecule_id", sort=False):
        smiles = predict_molecule(group, library, cfg)
        rows.append({"molecule_id": molecule_id, "smiles": format_smiles_field(smiles)})
        traces.append(
            {
                "molecule_id": molecule_id,
                "n_spectra": int(len(group)),
                "precursor_mz": float(group["precursor_mz"].mean()),
                "adduct": str(group["adduct"].iloc[0]),
                "candidates": smiles[:8],
            }
        )

    sub = pd.DataFrame(rows)
    out = ROOT / "outputs" / "submission.csv"
    write_submission(sub, out)
    (ROOT / "outputs" / "demo_trace.json").write_text(json.dumps(traces, indent=2))
    print(f"wrote {out}")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
