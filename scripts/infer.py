#!/usr/bin/env python3
"""Run library-search inference on Kaggle or local competition files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casmi26.pipeline import run_inference  # noqa: E402
from casmi26.retrieve import SearchConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="CASMI 2026 library-search inference")
    p.add_argument("--data-root", default=None, help="Folder with train.parquet and test.parquet")
    p.add_argument("--output", default="outputs/submission.csv")
    p.add_argument("--ppm", type=float, default=15.0)
    p.add_argument("--mz-tol", type=float, default=0.02)
    args = p.parse_args()
    cfg = SearchConfig(ppm_tol=args.ppm, mz_tol=args.mz_tol)
    sub = run_inference(args.data_root, args.output, cfg)
    print(sub.head())
    print(f"{len(sub)} molecules written to {args.output}")


if __name__ == "__main__":
    main()
