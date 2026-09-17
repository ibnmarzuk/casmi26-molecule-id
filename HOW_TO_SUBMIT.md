# How to submit on Kaggle

This is a **Featured Code Competition**. You cannot upload `submission.csv` by itself. A notebook must write it.

## 1. Join

1. Open [Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).
2. Accept the rules (entry deadline: **7 December 2026, 23:59 UTC**).
3. Final submission deadline: **14 December 2026, 23:59 UTC**.

## 2. Create the scoring notebook

1. Competitions → Code → **New Notebook**.
2. Confirm the competition dataset `enveda-CASMI26-molecule-id-mass-spectra` is attached.
3. Paste the contents of `kaggle/casmi26_kaggle_notebook.py` into a single cell (or File → Import).
4. Notebook settings:
   - **Internet: OFF**
   - Accelerator: none (CPU is enough for the library-search baseline)
   - Persistence: files
5. Run All. Confirm `/kaggle/working/submission.csv` exists.
6. **Save Version** (Save & Run All).
7. When the version finishes, click **Submit**.

The output file **must** be named `submission.csv`.

## 3. What the baseline does

For each `molecule_id` (not each spectrum):

1. Collect every MS/MS row for that molecule.
2. Pull training spectra whose precursor m/z is within 15 ppm.
3. Score with **0.55 × modified cosine + 0.45 × spectral entropy**.
4. Fuse ranks across collision energies with Reciprocal Rank Fusion.
5. Emit up to 25 SMILES, best first, semicolon-separated.

This is strong on **class 1** (structures that already have public spectra). Class 2/3 need a structure database (PubChem / COCONUT) and/or a de-novo model — see the README for the upgrade path.

## 4. Local dry-run (no Kaggle data)

```bash
python scripts/demo.py
```

Writes `outputs/submission.csv` from synthetic spectra.

## 5. Real local run

```bash
kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data/raw
python scripts/infer.py --data-root data/raw --output outputs/submission.csv
```

You still have to re-run the same logic inside a Kaggle notebook for it to count.

## 6. Submission file contract

```
molecule_id,smiles
m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O;CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```

Rejected if any of these fail:

- missing `molecule_id` or `smiles`
- empty file
- nulls
- repeated `molecule_id`
- more than 25 semicolon-separated guesses
