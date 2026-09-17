# Push this project to GitHub

This agent cannot use your GitHub account (connection is unavailable). Run these commands on your machine after unzipping.

```bash
cd casmi26-molecule-id
git init
git add .
git commit -m "Initial CASMI 2026 library-search baseline"
```

**Option A — GitHub CLI**

```bash
gh auth login
gh repo create casmi26-molecule-id --public --source=. --remote=origin --push
```

**Option B — website**

1. New repository on github.com (do not add a README).
2. Then:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USER/casmi26-molecule-id.git
git push -u origin main
```

Do not commit `data/raw/*.parquet` (already gitignored). The 3 GB training file stays on Kaggle.
