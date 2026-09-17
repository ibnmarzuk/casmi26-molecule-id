"""End-to-end library-search pipeline used locally and in tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .io import competition_dir, read_test, write_submission
from .metric import format_smiles_field
from .retrieve import Library, SearchConfig, predict_molecule
from .adducts import ppm_delta


def load_library_for_test(test: pd.DataFrame, root: str | Path | None, cfg: SearchConfig) -> Library:
    import pyarrow.parquet as pq

    path = competition_dir(root) / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"train.parquet not found at {path}. Download the Kaggle data into data/raw/."
        )

    precursors = test["precursor_mz"].astype(float).tolist()
    windows = [(mz, ppm_delta(mz, cfg.ppm_tol * 2.0)) for mz in precursors]

    pf = pq.ParquetFile(path)
    chunks = []
    wanted = [
        "normalized_smiles",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
        "adduct",
        "ingest_lib",
        "instrument_type",
    ]
    for batch in pf.iter_batches(batch_size=50_000, columns=wanted):
        df = batch.to_pandas()
        mz = df["precursor_mz"].astype(float)
        keep = pd.Series(False, index=df.index)
        for center, width in windows:
            keep = keep | ((mz - center).abs() <= width)
        hit = df.loc[keep]
        if len(hit):
            chunks.append(hit)
    if not chunks:
        sample = pf.read_row_group(0, columns=wanted).to_pandas()
        chunks = [sample.head(5000)]
    train = pd.concat(chunks, ignore_index=True)
    train = train.drop_duplicates(subset=["normalized_smiles", "precursor_mz"], keep="first")
    return Library.from_frame(train, cfg)


def run_inference(
    data_root: str | Path | None = None,
    output: str | Path = "submission.csv",
    cfg: SearchConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or SearchConfig()
    test = read_test(data_root)
    library = load_library_for_test(test, data_root, cfg)
    rows = []
    for molecule_id, group in tqdm(test.groupby("molecule_id", sort=False), desc="molecules"):
        smiles = predict_molecule(group, library, cfg)
        rows.append({"molecule_id": molecule_id, "smiles": format_smiles_field(smiles)})
    sub = pd.DataFrame(rows)
    write_submission(sub, output)
    return sub
