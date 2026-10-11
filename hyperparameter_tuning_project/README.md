# Hyperparameter Tuning and Model Comparison

A reusable Scikit-learn project for classification tasks. It compares Logistic Regression, Random Forest, and Support Vector Classifier (SVC), using both `GridSearchCV` and `RandomizedSearchCV`.

## Project structure

```text
hyperparameter_tuning_project/
├── README.md
├── requirements.txt
├── notebooks/
│   └── hyperparameter_tuning.ipynb
├── docs/
│   └── comparative_analysis.md
├── src/
│   └── train_tuned_model.py
├── data/
│   └── README.md
└── results/
    └── README.md
```

## 1. Requirements

- Python 3.10 or 3.11 recommended
- A labelled CSV dataset for a **classification** problem
- One target column containing the class label

## 2. Install

Open a terminal in this project folder:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Add your dataset

Place your CSV file in `data/`, for example `data/your_dataset.csv`. Edit the `DATA_PATH` and `TARGET_COLUMN` settings at the top of `src/train_tuned_model.py`, or supply command-line arguments:

```bash
python src/train_tuned_model.py --data data/your_dataset.csv --target target
```

Replace `target` with the exact name of the label column in your CSV. Do not use an ID column as the target. The script automatically imputes missing values, one-hot encodes categorical predictors, scales numeric predictors for scale-sensitive models, and keeps preprocessing inside a Pipeline to avoid data leakage.

## 4. Run the script

```bash
python src/train_tuned_model.py --data data/your_dataset.csv --target target
```

Outputs are written to `results/`: a CSV comparison table, JSON best-parameter summary, and the selected fitted model (`best_model.joblib`). The held-out test set is only used for final evaluation, not for tuning.

## 5. Run the notebook

```bash
jupyter notebook
```

Open `notebooks/hyperparameter_tuning.ipynb`. Update the dataset path and target column in the configuration cell, then use **Run All**.

## 6. Upload to GitHub

1. Create a new empty repository on GitHub.
2. In a terminal opened in this folder, run:

```bash
git init
git add README.md requirements.txt notebooks docs src data results
git commit -m "Add hyperparameter tuning and model comparison project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own repository URL. Do not commit private or sensitive data. Large datasets and generated model files may be excluded if your course instructions require it.

## Methodology

- Stratified train/test split.
- Stratified cross-validation on the training set.
- Three classifiers with model-specific parameter spaces.
- Grid search and randomized search with F1-weighted scoring by default.
- Final comparison on the untouched test set.
- The final selection should consider test metrics, runtime, and model complexity rather than accuracy alone.

## Important

This starter project is configured for **classification**. If your target is continuous numeric data (a regression task), replace the classifiers and use regression metrics such as MAE, RMSE, and R².
