# Model Evaluation Pipeline — Adult Census Income

This project trains a classifier on the Adult Census Income dataset and runs it through a full evaluation pipeline covering performance, explainability, fairness, and robustness, ending in a self-contained HTML report.

## The Dataset

**Adult Census Income** (also known as the "Census Income" or "UCI Adult" dataset) is a classic benchmark dataset originally extracted from the 1994 US Census Bureau database.

- **Task:** binary classification — predict whether a person's annual income is above or below $50K
- **Size:** ~48,800 rows, 14 original features (after dropping `fnlwgt`, an internal census sampling weight with no predictive meaning)
- **Target column:** `income` (`<=50K` or `>50K`), encoded as `0`/`1` for training

**Features:**

| Column           | Description                                            |
| ---------------- | ------------------------------------------------------ |
| `age`            | Age in years                                           |
| `workclass`      | Employment type (Private, Self-emp, Government, etc.)  |
| `education`      | Highest education level completed                      |
| `education.num`  | Education level as a number (higher = more schooling)  |
| `marital.status` | Marital status                                         |
| `occupation`     | Job category                                           |
| `relationship`   | Relationship role within household                     |
| `race`           | Race                                                   |
| `sex`            | Sex                                                    |
| `capital.gain`   | Income from investment sources, other than wage/salary |
| `capital.loss`   | Losses from investment sources                         |
| `hours.per.week` | Hours worked per week                                  |
| `native.country` | Country of origin                                      |

Categorical columns are one-hot encoded before training. **`sex`** (and optionally `race`) is kept as a separate, unencoded copy before that encoding happens — the fairness section of the pipeline needs clean group labels like `Male`/`Female` rather than one-hot columns to compute group-level metrics.

**Why this dataset for an evaluation pipeline specifically:** it has real demographic attributes (age, sex, race) that make fairness analysis meaningful, a mix of numeric and categorical features that exercises SHAP well, and known, well-documented income disparities across groups in the underlying data — so fairness metrics computed on it produce genuine, explainable findings rather than noise.

## What the Pipeline Does

The model is a `RandomForestClassifier`, trained on an 70/30 train/test split. Everything below runs against the held-out test set.

### 1. Model Performance

Standard classification metrics — accuracy, precision, recall, F1, ROC-AUC — plus a confusion matrix and ROC curve. This establishes the baseline: how good is the model before we start probing it further.

### 2. SHAP Explanations

Uses [SHAP](https://shap.readthedocs.io/) (`TreeExplainer`, which is fast and exact for tree-based models) to explain _why_ the model makes its predictions:

- **Global summary plot** — which features matter most to the model overall, and whether high or low values push predictions up or down
- **Local explanations** — for a handful of individual predictions, which specific features drove that particular decision, and in which direction

This turns the model from a black box into something you can actually audit — important for a thesis or report that needs to justify the model's behavior, not just its accuracy score.

### 3. Fairness Metrics

Uses [Fairlearn](https://fairlearn.org/) to check whether the model treats different `sex` groups differently:

- **Selection rate** per group — how often each group gets predicted as `>50K`
- **Demographic parity difference** — how far apart the groups' selection rates are
- **Equalized odds difference** — how far apart the groups' true/false positive rates are (a stricter check than parity, since it accounts for whether predictions are actually correct)
- **Disparate impact ratio** — the same idea as demographic parity, expressed as a ratio; below 0.8 is flagged by the "four-fifths rule," a common legal/regulatory threshold for adverse impact

A large disparity here doesn't necessarily mean the code is broken — it usually reflects genuine patterns already present in the training data, which is itself a legitimate and reportable finding.

### 4. Robustness Under Noise

Injects Gaussian noise into the numeric input features at increasing intensities (0%, 5%, 10%, 20%, 35% of each feature's standard deviation), re-runs predictions at each level, and tracks how much accuracy and F1 degrade. This tests how much the model relies on precise input values versus general patterns — a model that collapses under small noise is more fragile than its clean-test-set accuracy alone would suggest.

### 5. HTML Report

All of the above — metrics, charts, tables — gets assembled into a single self-contained HTML file (`evaluation_report.html`), with charts embedded directly as base64 images so it opens and renders correctly anywhere, with no external files or internet connection needed.
