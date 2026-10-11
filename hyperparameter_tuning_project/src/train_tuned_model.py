#!/usr/bin/env python
"""Tune three classification models with GridSearchCV and RandomizedSearchCV."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/your_dataset.csv", help="Path to input CSV")
    parser.add_argument("--target", required=True, help="Exact target/label column name")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--random-iterations", type=int, default=12)
    parser.add_argument("--scoring", default="f1_weighted")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Add your CSV or pass --data.")
    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column {args.target!r} not found. Available columns: {list(df.columns)}")
    df = df.dropna(subset=[args.target]).copy()
    X = df.drop(columns=[args.target])
    y = df[args.target]
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after removing the target.")
    if y.nunique() < 2:
        raise ValueError("Classification requires at least two distinct target classes.")
    if y.value_counts().min() < args.cv + 1:
        raise ValueError("Each class needs enough examples for the requested stratified CV and train/test split. Reduce --cv or use more data.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    prep = build_preprocessor(X_train)

    model_specs = {
        "LogisticRegression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__solver": ["lbfgs"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "SVC": (
            SVC(),
            {
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }

    rows, best_by_model = [], {}
    for model_name, (estimator, grid) in model_specs.items():
        print(f"\n=== {model_name}: GridSearchCV ===")
        pipe = Pipeline([("preprocess", prep), ("model", estimator)])
        grid_search = GridSearchCV(
            pipe, param_grid=grid, scoring=args.scoring, cv=cv,
            n_jobs=-1, refit=True, return_train_score=False, error_score="raise"
        )
        t0 = time.perf_counter()
        grid_search.fit(X_train, y_train)
        grid_seconds = time.perf_counter() - t0
        best_by_model[f"{model_name}_grid"] = {
            "best_params": grid_search.best_params_,
            "best_cv_score": float(grid_search.best_score_),
            "search_seconds": grid_seconds,
        }
        for search_name, search, seconds in [("GridSearchCV", grid_search, grid_seconds)]:
            pred = search.best_estimator_.predict(X_test)
            rows.append({
                "model": model_name, "search": search_name,
                "best_cv_score": search.best_score_,
                "test_accuracy": accuracy_score(y_test, pred),
                "test_precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
                "test_recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
                "test_f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
                "search_seconds": seconds,
                "best_params": json.dumps(search.best_params_, default=str),
            })
        print(f"Best CV {args.scoring}: {grid_search.best_score_:.4f}")

        print(f"=== {model_name}: RandomizedSearchCV ===")
        random_search = RandomizedSearchCV(
            pipe, param_distributions=grid, n_iter=args.random_iterations,
            scoring=args.scoring, cv=cv, n_jobs=-1, random_state=RANDOM_STATE,
            refit=True, return_train_score=False, error_score="raise"
        )
        t0 = time.perf_counter()
        random_search.fit(X_train, y_train)
        random_seconds = time.perf_counter() - t0
        best_by_model[f"{model_name}_randomized"] = {
            "best_params": random_search.best_params_,
            "best_cv_score": float(random_search.best_score_),
            "search_seconds": random_seconds,
        }
        pred = random_search.best_estimator_.predict(X_test)
        rows.append({
            "model": model_name, "search": "RandomizedSearchCV",
            "best_cv_score": random_search.best_score_,
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
            "test_recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
            "test_f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
            "search_seconds": random_seconds,
            "best_params": json.dumps(random_search.best_params_, default=str),
        })
        print(f"Best CV {args.scoring}: {random_search.best_score_:.4f}")

    results = pd.DataFrame(rows).sort_values(["test_f1_weighted", "search_seconds"], ascending=[False, True])
    results.to_csv(output_dir / "model_comparison.csv", index=False)
    with (output_dir / "best_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(best_by_model, f, indent=2, default=str)

    winner_row = results.iloc[0]
    winner_key = f"{winner_row['model']}_{'grid' if winner_row['search'] == 'GridSearchCV' else 'randomized'}"
    # Refit the selected configuration on the full training split; test set remains untouched.
    model_name = winner_row["model"]
    estimator, grid = model_specs[model_name]
    pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", estimator)])
    params = best_by_model[winner_key]["best_params"]
    pipe.set_params(**params)
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, output_dir / "best_model.joblib")
    pred = pipe.predict(X_test)
    (output_dir / "final_classification_report.txt").write_text(
        f"Selected model: {model_name} ({winner_row['search']})\n"
        f"Scoring used for CV: {args.scoring}\n"
        f"Accuracy: {accuracy_score(y_test, pred):.4f}\n"
        f"Weighted precision: {precision_score(y_test, pred, average='weighted', zero_division=0):.4f}\n"
        f"Weighted recall: {recall_score(y_test, pred, average='weighted', zero_division=0):.4f}\n"
        f"Weighted F1: {f1_score(y_test, pred, average='weighted', zero_division=0):.4f}\n\n"
        + classification_report(y_test, pred, zero_division=0),
        encoding="utf-8",
    )
    print("\nComparison table:\n", results.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir.resolve()}")
    print(f"Scikit-learn version: {sklearn.__version__}")


if __name__ == "__main__":
    main()
