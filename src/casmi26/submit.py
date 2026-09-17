"""Validate and write a Kaggle-legal submission.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import write_submission
from .metric import format_smiles_field


def build_submission(molecule_to_smiles: dict[str, list[str]], path: str | Path = "submission.csv") -> pd.DataFrame:
    rows = [
        {"molecule_id": mol, "smiles": format_smiles_field(smiles)}
        for mol, smiles in molecule_to_smiles.items()
    ]
    df = pd.DataFrame(rows)
    write_submission(df, path)
    return df
