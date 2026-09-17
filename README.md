# Enveda CASMI 2026 — Molecule ID from Mass Spectra

Kaggle featured **code** competition. Predict 2D structures (SMILES) from LC-MS/MS spectra. Metric: **MRR@25** after RDKit tautomer canonicalization and InChIKey14 match.

Competition: [enveda-CASMI26-molecule-id-mass-spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)

| | |
|---|---|
| Host | Enveda |
| Dates | 14 Sep 2026 → **14 Dec 2026 23:59 UTC** |
| Entry / team merge | 7 Dec 2026 |
| Prize pool | $50,000 (1st $16k) |
| Test | ~400 molecules, ~1,500 timsTOF spectra |
| Train | ~2.5M spectra, ~275k structures |

Open `submit_portal.html` for every Kaggle field with a **Copy** button. Open `submission_mode.html` for the animated console.

## What you submit

For every `molecule_id`, up to 25 SMILES, best first, joined by `;`:

```
molecule_id,smiles
m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O;CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```

A notebook with **internet off** must write `/kaggle/working/submission.csv`. See `HOW_TO_SUBMIT.md`.

## Repository

```
kaggle/casmi26_kaggle_notebook.py   ← paste this into Kaggle and submit
submission_mode.html                ← animated submission console
src/casmi26/                        ← local package (search, metric, IO)
scripts/demo.py                     ← synthetic dry-run
scripts/infer.py                    ← run on real parquet
```

## Baseline

Library search, not de novo:

1. Precursor m/z filter (±15 ppm).
2. Peak clean (intensity floor, drop fragments above precursor + 2 Da, top-64, sqrt).
3. **Modified cosine** (GNPS-style, precursor-shift matches) + **spectral entropy** similarity.
4. Reciprocal Rank Fusion across the 1–16 spectra of each molecule.
5. Mass-nearest SMILES fill so every molecule has up to 25 guesses.

This is the right first submit: it solves **class 1** (public-library structures). Class 2 needs PubChem/COCONUT retrieval by formula; class 3 needs a de-novo model (MS2Mol / MEGAN / transformer). Upgrade path is in the package — do not skip a valid baseline.

## Local demo

```bash
python -m pip install -e ".[dev]"
python scripts/demo.py
python -m pytest -q
```

`outputs/submission.csv` is a valid file you can inspect. It is **not** a Kaggle score — scoring only happens when the notebook re-runs on the hidden test set.

## Push this folder to GitHub

GitHub OAuth from this agent is not connected. From the folder:

```bash
cd casmi26-molecule-id
git init
git add .
git commit -m "Initial CASMI 2026 library-search baseline"
gh repo create casmi26-molecule-id --public --source=. --push
```

Or create an empty repo in the GitHub UI, then:

```bash
git remote add origin https://github.com/<you>/casmi26-molecule-id.git
git branch -M main
git push -u origin main
```

## Citation

David Healey et al. *Enveda CASMI 2026 - Molecule ID From Mass Spectra.* Kaggle, 2026.
