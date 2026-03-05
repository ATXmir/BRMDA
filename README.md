# BRMDA (BAN + RF) — Reproducible Pipeline for MDAD

This repository contains a reproducible implementation for **microbe–drug association prediction** on the **MDAD** dataset using:

- **BAN** (Bilinear Attention Network) to produce interaction-aware representations
- **RWR** (Random Walk with Restart) for graph-based feature enhancement
- **Random Forest (RF)** as the downstream classifier
- **Stratified 5-fold cross-validation** with configurable **negative sampling ratio**

The code is designed to be **fully runnable with Python 3.9** and supports **path-parameterized outputs** (no hard-coded save paths).

---

## 1. Project Layout

Place scripts and the dataset folder in the same directory:

```
project_root/
  main_all_cv.py      # 5-fold CV entry (no grid search)
  mode_data_load.py                  # MDAD loader (Python 3.9 compatible)
  ban3.py                                      # BAN model
  rwr.py                                       # RWR implementation
  MDAD/                                        # dataset folder (required)
    microbes.xlsx
    disease.xlsx
    drugs.xlsx
    microbe_with_disease.txt
    drug_with_disease.txt
    drug_microbe_matrix.txt
```

> **Important**: This repo assumes the dataset is under `./MDAD` by default (configurable via `--data-root`).

---

## 2. Environment Setup (Python 3.9)

### 2.1 pip (simple)
Install core dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib tqdm openpyxl
```

Install **PyTorch** separately (recommended) to match your OS/CUDA:
- CPU-only is fine for reproducing results.
- If you use GPU, install the CUDA-matched build.

Quick sanity check:

```bash
python -c "import torch, sklearn, pandas; print('torch', torch.__version__)"
```

### 2.2 conda (recommended for stability)
Create a clean environment:

```bash
conda create -n brmda_py39 python=3.9 -y
conda activate brmda_py39
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib tqdm openpyxl -y
# Install PyTorch (CPU example). Choose the variant matching your platform/CUDA.
conda install -c pytorch pytorch -y
```

---

## 3. Data Requirements (MDAD)

The loader expects the following files under `MDAD/`:

- `microbes.xlsx`
- `disease.xlsx`
- `drugs.xlsx`
- `microbe_with_disease.txt`
- `drug_with_disease.txt`
- `drug_microbe_matrix.txt`

If any file is missing, the loader will stop and print which file(s) are required.

---

## 4. Running Experiments (5-fold CV)

### 4.1 Default 5-fold CV (negative ratio = 1.0)
From `project_root/`:

```bash
python main_all_cv.py \
  --data-root MDAD \
  --export-dir result/cv_neg1 \
  --neg-ratio 1.0 \
  --seed 42
```

### 4.2 Imbalanced settings (e.g., 1:3 negatives)
`--neg-ratio k` means: for each positive pair, sample `k` pseudo-negative pairs from unknown (zero) entries.

```bash
python main_all_cv.py \
  --data-root MDAD \
  --export-dir result/cv_neg3 \
  --neg-ratio 3.0 \
  --seed 42
```

### 4.3 Key CLI options
Commonly used options:

- `--data-root`: dataset folder (default: `MDAD`)
- `--export-dir`: output folder (required for clean organization)
- `--neg-ratio`: negative-to-positive sampling ratio (float, e.g., 1.0 / 3.0 / 5.0 / 10.0)
- `--seed`: random seed for reproducibility
- `--lam`, `--lam-m`, `--lam-d`: hyperparameters used in the Eq.(9)-related fusion (if applicable)
- `--rwr-alpha`: RWR restart probability
- `--hidden-dim`, `--lr`, `--epochs`, `--weight-decay`: training hyperparameters
- `--rf-n-estimators`, `--rf-max-depth`, `--rf-max-features`: RF hyperparameters

---

## 5. Outputs (What gets saved)

All outputs are written under your `--export-dir`, e.g. `result/cv_neg3/`:

- `fold_metrics.csv`  
  Per-fold metrics (AUC, AUPR, ACC, F1, etc.).
- `roc_merged_curve.csv`, `pr_merged_curve.csv`  
  Merged ROC/PR curve points across folds.
- `summary.json`  
  Summary statistics and run configuration.
- `merged_labels_scores.npz`  
  Saved labels/scores for plotting and post-analysis.
- `roc.png`, `pr.png`  
  Ready-to-use figures for manuscripts.

---

## 6. Metric Computation Notes (AUC/AUPR vs ACC/F1)

- **AUC** and **AUPR** are computed directly from **continuous prediction scores** (no threshold required).
- **ACC** and **F1** require binary predictions, therefore the code uses an explicit **thresholding rule**:
  - The decision threshold is selected **on the training split only** by maximizing **F1** on the training PR curve.
  - The selected threshold is then applied to the corresponding test split to compute ACC/F1.

This avoids using any test information for threshold selection.

---


## 7. Troubleshooting

### 7.1 “Python 3.9 does not support X | Y”
This is a Python 3.10+ type annotation syntax.  
All repo scripts provided with `*_py39.py` are already compatible with **Python 3.9**.

### 7.2 Missing `openpyxl`
If `.xlsx` loading fails:

```bash
pip install openpyxl
```

### 7.3 Path issues on Windows
Use quotes for paths containing spaces:

```bash
python main_all_cv.py --data-root "D:\path\to\MDAD" --export-dir "D:\out\cv"
```

### 7.4 Performance / memory
- Full score matrix export can be large (`n_microbes × n_drugs`).  
  Use `--infer-batch-size` to reduce memory pressure, or export Top-K only (can be added if needed).

---

## 8. Reproducibility Checklist (for reviewers)

- ✅ README provides end-to-end instructions and required data files.
- ✅ Dependencies are explicitly listed; PyTorch install guidance is included.
- ✅ Outputs are fully parameterized via `--export-dir` (no hard-coded save paths).
- ✅ 5-fold CV uses a fixed `--seed` and stratified splitting.
- ✅ Thresholding for ACC/F1 is performed on training folds only.

---

## Citation
If you use this code in academic work, please cite the corresponding manuscript and the MDAD dataset paper.
