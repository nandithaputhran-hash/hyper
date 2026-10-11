# Comparative Analysis Report

> **Before submission:** Replace every bracketed placeholder with results produced by your own run. Do not invent metrics or claim a model won before examining the results.

## 1. Objective
The objective is to compare Logistic Regression, Random Forest, and Support Vector Classifier (SVC), using GridSearchCV and RandomizedSearchCV to tune their hyperparameters.

## 2. Dataset and preprocessing
- Dataset: [dataset name and source]
- Number of rows after removing missing target labels: [N]
- Target column: [target column]
- Target classes and distribution: [classes / counts]
- Train/test split: [e.g., 80% / 20%, stratified]
- Preprocessing: numeric missing values imputed with the median and scaled; categorical missing values imputed with the most frequent value and one-hot encoded.
- Cross-validation: stratified [K]-fold CV on the training data.
- Scoring metric for model selection: weighted F1 (`f1_weighted`).

Preprocessing is placed inside Scikit-learn Pipelines, so each cross-validation fold learns its preprocessing only from its training portion. The held-out test set is not used during tuning.

## 3. Search methods
GridSearchCV evaluates every parameter combination in each declared grid. RandomizedSearchCV samples a fixed number of configurations from the same candidate values. Grid search can be more exhaustive but more expensive; randomized search can explore a broader space efficiently when the parameter space is large.

## 4. Test-set comparison

Fill this table from `results/model_comparison.csv`.

| Model | Search method | Best CV weighted F1 | Test accuracy | Test weighted precision | Test weighted recall | Test weighted F1 | Search time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | GridSearchCV | [value] | [value] | [value] | [value] | [value] | [value] |
| Logistic Regression | RandomizedSearchCV | [value] | [value] | [value] | [value] | [value] | [value] |
| Random Forest | GridSearchCV | [value] | [value] | [value] | [value] | [value] | [value] |
| Random Forest | RandomizedSearchCV | [value] | [value] | [value] | [value] | [value] | [value] |
| SVC | GridSearchCV | [value] | [value] | [value] | [value] | [value] | [value] |
| SVC | RandomizedSearchCV | [value] | [value] | [value] | [value] | [value] | [value] |

## 5. Best hyperparameters
Summarize the best configuration for each model and search method from `results/best_parameters.json`:
- Logistic Regression: [configuration]
- Random Forest: [configuration]
- SVC: [configuration]

## 6. Final model selection
Selected model: [model and search method]

Justification:
- Predictive performance: [compare weighted F1 and other relevant metrics]
- Runtime: [compare search time and practical prediction requirements]
- Complexity and interpretability: Logistic Regression is generally easier to interpret; Random Forest can model nonlinear interactions but may be larger; SVC may be effective in high-dimensional spaces but can be expensive on large datasets.
- Error analysis: [describe class-level precision/recall from `final_classification_report.txt`]
- Limitations: [dataset size, class imbalance, feature quality, possible distribution shift]

Choose a model based on the measured held-out test metrics and project requirements. If performance differences are negligible, prefer the simpler or faster model when that suits the use case. Test results are estimates and should not be repeatedly used to tune the model.
