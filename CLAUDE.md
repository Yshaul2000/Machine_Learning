# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains machine learning projects. Currently it has one project: a **diabetes diagnosis classifier** (`Diabetes-Project/`) built on the Pima Indians Diabetes Database (768 samples, 8 clinical features, binary outcome).

## Running the Code

The project is designed for **Google Colab** and requires a `diabetes.csv` file mounted via Google Drive.

To run locally instead:
1. Remove the Google Drive mount and `google.colab` import
2. Replace the data path with a local path to `diabetes.csv`
3. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
4. Run: `python "Diabetes-Project/final_project_diabetes.py"` or open the notebook in Jupyter

## Architecture

The project implements a complete ML pipeline in a single notebook/script:

**Preprocessing pipeline:**
- Zero-value imputation for biologically impossible values (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
- Log transformation for features with skewness > 0.75
- StandardScaler normalization
- Stratified 70/30 train/test split

**Supervised models:** KNN (grid search over k=1–20), Logistic Regression (L1/L2 variants), Gaussian Naive Bayes, Decision Tree (full, max_depth=3, GridSearchCV-optimized). A custom classification threshold of **0.35** (instead of 0.5) is used for Logistic Regression to improve recall on the positive class.

**Unsupervised:** PCA (1–6 components) followed by K-Means clustering (k=2).

**Evaluation:** `measure_error()` at line 565 of the `.py` file computes accuracy/precision/recall/F1 for a given split label. All models are compared via a unified ROC curve plot.

## Intentional Code Structure

The notebook/script contains the **full pipeline twice** — this is intentional. The first run uses the raw data (zeros kept), the second run uses the corrected data (zeros imputed). This demonstrates the impact of the preprocessing fix as part of the project's narrative.

## Actual Model Results

Results extracted from embedded notebook outputs (threshold = 0.35):

**Before fix (zeros kept):**
| Model | Accuracy | AUC |
|---|---|---|
| LR Regular | 76.2% | 0.8435 |
| DT Optimized | 76.6% | — |
| Naive Bayes | 75.8% | 0.8235 |
| DT Full | 76.2% | — (overfits: 23.8pp gap) |

**After fix (zeros imputed):**
| Model | Accuracy | AUC |
|---|---|---|
| LR variants | 77.9% | 0.843–0.844 |
| Naive Bayes | 77.9% | 0.8347 |
| DT Full | 69.3% | — (worse after fix — spurious splits removed) |

Key insight: zero imputation improves linear models (+2pp) but hurts Decision Trees (−7pp) by eliminating the easy zero-based splits they were exploiting.

## Key Files

- [Diabetes-Project/Final_Project_Diabetes (1).ipynb](Diabetes-Project/Final_Project_Diabetes%20(1).ipynb) — primary notebook with outputs
- [Diabetes-Project/final_project_diabetes.py](Diabetes-Project/final_project_diabetes.py) — exported Python script (1829 lines, pipeline appears twice intentionally)
- [Diabetes-Project/dashboard.html](Diabetes-Project/dashboard.html) — standalone interactive dashboard (open in browser, no server needed); toggle between before/after fix phases
- [Diabetes-Project/README.md](Diabetes-Project/README.md) — full methodology and feature documentation
