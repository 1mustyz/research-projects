# Trustworthy AI Evaluation for Time-Series Classification

## Overview

This notebook presents a complete workflow for evaluating **trustworthy machine learning** using the GunPoint time-series classification dataset.

Unlike traditional machine learning experiments that focus only on predictive accuracy, this notebook evaluates models from four complementary trustworthiness dimensions:

1. Performance
2. Explainability
3. Fairness
4. Robustness

The notebook concludes by combining these dimensions into a composite **Trustworthiness Score**, allowing different models to be compared under multiple decision priorities.

---

# Objectives

The notebook aims to answer the following questions:

- Which model achieves the highest predictive performance?
- Can the model's predictions be explained?
- Does the model behave fairly across different groups?
- Is the model robust to noisy or incomplete input?
- Which model should be selected when all trustworthiness dimensions are considered?

---

# Dataset

The notebook uses the **GunPoint** dataset provided by the `sktime` library.

The task is binary classification:

- Class 0 — Point
- Class 1 — Gun-Draw

Each sample is a univariate time series consisting of approximately 150 timesteps.

---

# Notebook Structure

## 1. Load Data & Prepare

This section prepares the dataset for machine learning.

Steps include:

- Loading the GunPoint dataset
- Converting nested pandas structures into NumPy arrays
- Encoding labels into binary values
- Creating a simulated protected attribute
- Splitting data into training, validation and testing sets
- Standardizing the input signals
- Reshaping the data for the LSTM

A simulated protected attribute is introduced solely for fairness evaluation because the GunPoint dataset contains no demographic information.

---

## 2. Performance Evaluation

Two models are trained:

- LSTM
- Logistic Regression

The following evaluation metrics are computed:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Result

Logistic Regression achieved higher predictive performance across all metrics.

---

## 3. Explainability

This section investigates **why** the models make their predictions.

Two explainability methods are used:

### SHAP

Provides global explanations by identifying the most influential timesteps across the entire dataset.

### LIME

Provides local explanations by explaining individual predictions.

The notebook also compares SHAP and LIME explanations using the Spearman rank correlation.

A correlation above 0.9 indicates strong consistency between both explanation methods.

Logistic Regression also benefits from being inherently interpretable.

---

## 4. Fairness Evaluation

The notebook evaluates whether the models behave similarly across two simulated protected groups.

Metrics include:

- Selection Rate
- True Positive Rate
- False Positive Rate
- Statistical Parity Difference
- Disparate Impact Ratio
- Equal Opportunity Difference
- Average Odds Difference

Smaller differences between groups indicate better fairness.

Logistic Regression produced substantially smaller fairness disparities than the LSTM.

---

## 5. Robustness Evaluation

Robustness measures how sensitive models are to input perturbations.

Two robustness tests are performed:

### Gaussian Noise

Random noise is added to every time series.

### Timestep Masking

Parts of the time series are hidden from the model.

Performance degradation is measured using Accuracy and F1-score.

The LSTM demonstrated greater robustness to Gaussian noise, although Logistic Regression remained more accurate overall.

---

## 6. Trustworthiness Scoring

The notebook combines four normalized trustworthiness dimensions:

- Performance
- Explainability
- Fairness
- Robustness

Three weighting strategies are considered:

- Performance-focused
- Fairness-focused
- Balanced

Each strategy computes a weighted average to produce a single trustworthiness score.

---

# Key Findings

| Dimension      | Better Model        |
| -------------- | ------------------- |
| Performance    | Logistic Regression |
| Explainability | Logistic Regression |
| Fairness       | Logistic Regression |
| Robustness     | LSTM                |

Although the LSTM demonstrated slightly stronger robustness, Logistic Regression consistently achieved:

- higher predictive accuracy,
- higher fairness,
- greater interpretability,
- and the highest composite trustworthiness score.

---

# Final Conclusion

This notebook demonstrates that trustworthy AI should be evaluated using multiple complementary dimensions rather than predictive performance alone.

For the GunPoint dataset, Logistic Regression proved to be the most trustworthy model under all evaluated weighting strategies.

The study also illustrates that simpler, inherently interpretable models can outperform more complex deep learning models on relatively small datasets while providing stronger fairness and transparency.
