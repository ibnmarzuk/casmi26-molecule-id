"""Read competition parquet files from Kaggle or a local data root."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KAGGLE_DIR = Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra")


def competition_dir(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit)
    if KAGGLE_DIR.exists():
        return KAGGLE_DIR
    local = Path("data/raw")
    if local.exists():
        return local
    return Path(".")


def read_train(root: str | Path | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    path = competition_dir(root) / "train.parquet"
    return pd.read_parquet(path, columns=columns)


def read_test(root: str | Path | None = None) -> pd.DataFrame:
    path = competition_dir(root) / "test.parquet"
    return pd.read_parquet(path)


def write_submission(df: pd.DataFrame, path: str | Path = "submission.csv") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    required = {"molecule_id", "smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"submission missing columns: {missing}")
    if df["molecule_id"].duplicated().any():
        raise ValueError("duplicate molecule_id")
    if df["molecule_id"].isna().any() or df["smiles"].isna().any():
        raise ValueError("nulls are not allowed")
    too_long = df["smiles"].str.split(";").map(len) > 25
    if too_long.any():
        raise ValueError("more than 25 guesses for a molecule")
    df[["molecule_id", "smiles"]].to_csv(out, index=False)
    return out
