# Trustworthiness Evaluation Under Distribution Shift

## Overview

This project evaluates the **trustworthiness of machine-learning models under temporal and population distribution shift** using the public **ProPublica COMPAS recidivism dataset**.

The work extends a previous trustworthiness methodology demonstration in which fairness was evaluated using a simulated protected attribute. In this project, fairness is evaluated using **real protected attributes contained in the dataset**, specifically:

- **Race:** African-American and Caucasian
- **Sex:** Male and Female

The project goes beyond ordinary predictive-performance comparison. Two machine-learning models are trained on historical data and evaluated on future data across five trustworthiness dimensions:

1. **Performance**
2. **Fairness**
3. **Calibration**
4. **Explainability**
5. **Robustness under distribution shift**

The notebook also produces normalized trustworthiness scores, compares three stakeholder weighting scenarios, and gives a deployment recommendation.

---

## Task

The task is to create a **time-based machine-learning benchmark using a public dataset with real protected attributes**.

The benchmark must:

- use a real public dataset;
- avoid simulated protected groups;
- use a temporal train/test split;
- train at least two models;
- evaluate predictive performance;
- evaluate fairness;
- evaluate probability calibration;
- provide SHAP explanations;
- test robustness under two forms of distribution shift;
- report original/raw metrics and normalized scores;
- compare three trustworthiness weighting scenarios;
- provide a short deployment recommendation;
- generate a Markdown/HTML report.

This implementation uses the **COMPAS dataset** rather than ACS/Folktables. COMPAS contains real demographic attributes and timestamp information that allow both temporal evaluation and a real protected-attribute fairness audit.

---

# 1. Dataset

## ProPublica COMPAS Dataset

The notebook downloads:

```text
compas-scores-two-years.csv
```

from ProPublica's public COMPAS analysis repository.

The original data contains:

```text
7,214 rows
53 columns
```

After applying the selected data-quality filters and restricting the race comparison to African-American and Caucasian defendants, the working dataset contains:

```text
5,278 rows
```

The available screening dates initially span:

```text
2013-01-01 to 2014-12-31
```

The prediction target is:

```python
two_year_recid
```

where:

```text
1 = defendant recidivated within two years
0 = defendant did not recidivate within two years
```

---

# 2. Data Cleaning

The notebook applies commonly used COMPAS data-quality filters:

```python
(raw['days_b_screening_arrest'] <= 30)
(raw['days_b_screening_arrest'] >= -30)
(raw['is_recid'] != -1)
(raw['c_charge_degree'] != 'O')
(raw['score_text'] != 'N/A')
```

These filters remove records with problematic screening/arrest timing, unusable recidivism information, unsupported charge categories, or missing COMPAS score information.

The race analysis is then restricted to:

```python
African-American
Caucasian
```

This creates a clearly defined two-group race comparison while still using **real demographic labels from the dataset**.

---

# 3. Data-Quality Check and Label Truncation

Before choosing the temporal train/test split, the notebook examines the observed recidivism rate by calendar quarter.

The results are:

| Quarter | Recidivism Rate | Records |
|---|---:|---:|
| 2013 Q1 | 0.402 | 1,287 |
| 2013 Q2 | 0.373 | 818 |
| 2013 Q3 | 0.372 | 696 |
| 2013 Q4 | 0.368 | 896 |
| 2014 Q1 | 0.373 | 808 |
| 2014 Q2 | 0.992 | 240 |
| 2014 Q3 | 1.000 | 221 |
| 2014 Q4 | 1.000 | 312 |

From 2013 Q1 through 2014 Q1, the observed rate remains approximately **0.37–0.40**.

Beginning in 2014 Q2, however, the observed rate jumps to approximately **0.99–1.00**.

This is treated as a **label-truncation/censoring artifact rather than a genuine behavioral distribution shift**. A two-year recidivism label requires enough future observation time. Later records do not provide the same reliable follow-up window.

For that reason, the experiment uses only:

```text
2013 Q1 through 2014 Q1
```

This step is important for trustworthy ML because the reliability of the labels must be checked before the reliability of the model can be assessed.

---

# 4. Temporal Train/Test Split

Unlike a conventional random train/test split, this project uses chronological data.

## Training Set

```text
2013 Q1 + 2013 Q2
```

Number of records:

```text
2,105
```

## Near-Term Test Set

```text
2013 Q3 + 2013 Q4
```

Number of records:

```text
1,592
```

This is the primary evaluation dataset.

## Far-Term Test Set

```text
2014 Q1
```

Number of records:

```text
808
```

This is used to test temporal robustness.

The split therefore follows a realistic deployment pattern:

```text
Past data                         Future data
──────────────────────────────────────────────────────>

2013 Q1      2013 Q2      2013 Q3      2013 Q4      2014 Q1
└──────── TRAIN ────────┘ └──── TEST NEAR ─────┘   └ TEST FAR ┘
```

The populations are similar but not identical:

| Split | N | Recidivism Rate | % African-American | % Male | Mean Priors | Mean Age |
|---|---:|---:|---:|---:|---:|---:|
| Train — 2013 H1 | 2,105 | 0.390 | 0.614 | 0.832 | 3.512 | 35.128 |
| Test Near — 2013 H2 | 1,592 | 0.370 | 0.572 | 0.781 | 2.921 | 34.769 |
| Test Far — 2014 Q1 | 808 | 0.373 | 0.588 | 0.769 | 2.891 | 34.191 |

These differences represent small but real temporal population drift.

---

# 5. Features

The models use five numerical features:

```python
age
priors_count
juv_fel_count
juv_misd_count
juv_other_count
```

and three categorical features:

```python
sex
c_charge_degree
age_cat
```

The target is:

```python
two_year_recid
```

## Race Is Not Used as a Prediction Feature

`race` is deliberately excluded from the predictive feature set.

Instead, it is retained for the fairness audit.

This allows the experiment to ask:

> Do model outcomes differ significantly by race even though race itself was not directly provided to the classifier?

Removing race from the input does **not** guarantee fairness because other variables may still correlate with race. This is why fairness is evaluated explicitly after predictions are produced.

---

# 6. Preprocessing

The preprocessing pipeline uses:

```python
StandardScaler()
```

for numerical variables and:

```python
OneHotEncoder(handle_unknown='ignore')
```

for categorical variables.

The complete preprocessing step is implemented with a Scikit-learn `ColumnTransformer`.

### Numerical scaling

Standardization transforms numerical variables approximately according to:

```text
z = (x - mean) / standard deviation
```

This places numerical variables on comparable scales and is especially useful for Logistic Regression.

### One-hot encoding

Categorical variables are converted into machine-readable binary columns.

For example:

```text
sex = Male
```

may be represented through encoded columns such as:

```text
sex_Female
sex_Male
```

`handle_unknown='ignore'` also prevents the pipeline from failing if a previously unseen category appears in future data.

---

# 7. Models

Two models are trained on exactly the same historical training set.

## Logistic Regression

Logistic Regression serves as the simpler and inherently interpretable baseline.

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

It estimates the probability of recidivism from a weighted combination of input features.

Advantages include:

- simple model structure;
- directly inspectable coefficients;
- easier human interpretation;
- useful baseline for trustworthiness evaluation.

---

## Random Forest

The second model is:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)
```

The Random Forest combines predictions from 300 decision trees.

It provides a nonlinear comparison against Logistic Regression and can capture more complex interactions between features.

The maximum depth of 6 limits tree complexity and helps reduce overfitting.

---

# 8. Primary Predictive Performance

The primary performance evaluation uses the near-term test set, **2013 H2**.

The metrics are:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **0.7079** | **0.6751** | 0.4058 | 0.5069 | **0.7267** |
| Random Forest | 0.6997 | 0.6357 | **0.4414** | **0.5210** | 0.7251 |

### Interpretation

The models have very similar overall predictive ability.

Logistic Regression has:

- slightly higher accuracy;
- higher precision;
- slightly higher ROC-AUC.

Random Forest has:

- higher recall;
- higher F1-score.

The result demonstrates why model selection should not be based on one predictive metric alone.

---

# 9. Calibration

A trustworthy probability model should produce probabilities that correspond reasonably closely to observed outcomes.

For example, among defendants assigned approximately 30% predicted risk, roughly 30% should experience the target outcome in a well-calibrated system.

Two calibration measurements are used.

## Brier Score

The Brier score measures mean squared probability error:

```text
Brier = mean((predicted_probability - true_label)^2)
```

Lower values are better.

## Expected Calibration Error — ECE

Predictions are divided into probability bins.

For each bin, the difference between:

```text
average predicted probability
```

and:

```text
actual observed outcome frequency
```

is measured.

The weighted average of these differences gives ECE.

Lower values are better.

## Calibration Results

| Model | Brier Score | ECE |
|---|---:|---:|
| Logistic Regression | **0.1982** | 0.0307 |
| Random Forest | 0.1994 | **0.0273** |

Both models show relatively small calibration errors.

Random Forest has slightly lower ECE, while Logistic Regression has a slightly better Brier score.

The notebook also generates a **reliability diagram** comparing observed and predicted probabilities.

---

# 10. Fairness Evaluation

Fairness is evaluated using **real protected attributes** from the COMPAS dataset.

Two audits are performed:

1. Race
2. Sex

The race comparison uses:

```text
Reference group: Caucasian
Focal group: African-American
```

The sex comparison uses:

```text
Reference group: Female
Focal group: Male
```

The following metrics are calculated:

- Selection Rate
- True Positive Rate — TPR
- False Positive Rate — FPR
- Statistical Parity Difference — SPD
- Disparate Impact Ratio — DIR
- Equal Opportunity Difference — EOD
- Average Odds Difference — AOD

---

## 10.1 Statistical Parity Difference

```text
SPD = selection_rate_focal - selection_rate_reference
```

A value close to zero indicates similar positive prediction rates.

---

## 10.2 Disparate Impact Ratio

The implementation calculates:

```text
smaller selection rate / larger selection rate
```

A value close to:

```text
1.0
```

indicates more similar selection rates.

The commonly referenced four-fifths rule uses approximately:

```text
0.80
```

as a screening threshold, although this should be treated as a heuristic rather than a complete legal fairness determination.

---

## 10.3 Equal Opportunity Difference

```text
EOD = TPR_focal - TPR_reference
```

A value close to zero indicates similar true-positive rates between groups.

---

## 10.4 Average Odds Difference

AOD combines differences in both true-positive and false-positive rates.

A value close to zero indicates greater similarity in classification errors across groups.

---

# 11. Race Fairness Results

| Model | Selection Caucasian | Selection African-American | TPR Caucasian | TPR African-American | FPR Caucasian | FPR African-American | SPD | DIR | EOD | AOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.1085 | 0.3077 | 0.1804 | 0.5165 | 0.0799 | 0.1476 | 0.1992 | 0.3526 | 0.3360 | 0.2018 |
| Random Forest | 0.1305 | 0.3516 | 0.2216 | 0.5494 | 0.0943 | 0.2000 | 0.2211 | 0.3711 | 0.3277 | 0.2167 |

Both models exhibit substantial racial disparities.

For example, the DIR values:

```text
Logistic Regression = 0.3526
Random Forest       = 0.3711
```

are far from parity at 1.0.

The fairness audit therefore identifies a significant trustworthiness concern even though race is not included directly in the model features.

---

# 12. Sex Fairness Results

| Model | Selection Female | Selection Male | TPR Female | TPR Male | FPR Female | FPR Male | SPD | DIR | EOD | AOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0575 | 0.2685 | 0.1429 | 0.4495 | 0.0303 | 0.1448 | 0.2110 | 0.2141 | 0.3066 | 0.2106 |
| Random Forest | 0.1006 | 0.3006 | 0.2143 | 0.4792 | 0.0644 | 0.1786 | 0.2001 | 0.3345 | 0.2649 | 0.1896 |

Large differences are also present in the sex-based fairness analysis.

The fairness results reinforce the main principle of this task:

> Similar predictive performance does not imply similar or acceptable fairness.

---

# 13. Explainability With SHAP

SHAP is used to explain how individual features influence model predictions.

Two model-specific explainers are used:

```python
shap.LinearExplainer
```

for Logistic Regression and:

```python
shap.TreeExplainer
```

for Random Forest.

The notebook calculates global SHAP importance using:

```python
mean(abs(SHAP value))
```

This measures the average magnitude of each feature's influence on predictions.

The most influential features are primarily:

```text
priors_count
age
age-category variables
```

The global feature rankings from the two models have a Spearman correlation of:

```text
0.6503
```

This indicates moderate agreement in what the two models consider important.

---

# 14. Explanation Concentration

The notebook calculates the proportion of total SHAP importance contained in the three most influential features.

Results:

| Model | Top-3 Explanation Concentration |
|---|---:|
| Logistic Regression | **0.7388** |
| Random Forest | 0.7058 |

This means approximately 74% of Logistic Regression's total global explanation importance is concentrated in its three most influential features, compared with about 71% for Random Forest.

A more concentrated explanation can be easier for a human reviewer to summarize and audit, although this is a benchmark design choice rather than a universal explainability definition.

---

# 15. Explanation Stability

The SHAP sample is split into two halves.

Feature importance rankings are computed independently on both halves and compared using Spearman correlation.

Results:

| Model | Split-Half Explanation Stability |
|---|---:|
| Logistic Regression | 0.965 |
| Random Forest | **0.979** |

Both models produce highly stable global explanations.

Random Forest's SHAP feature ranking is slightly more stable across the two subsets.

---

# 16. Inherent Interpretability

The scoring framework also explicitly distinguishes between inherent and post-hoc interpretability.

The benchmark assigns:

```text
Logistic Regression = 1
Random Forest       = 0
```

because Logistic Regression has directly inspectable coefficients, whereas interpretation of the Random Forest depends more heavily on post-hoc explanation tools such as SHAP.

---

# 17. Distribution Shift Evaluation

The notebook performs **two separate distribution-shift tests**.

Both tests use genuine COMPAS records rather than simulated protected groups.

---

# 18. Shift Test 1 — Temporal Shift

The first robustness test compares:

```text
Near-term test: 2013 H2
```

against:

```text
Far-term test: 2014 Q1
```

This measures how well a model trained on 2013 H1 continues to perform as time progresses.

## Results

| Model | Accuracy Near | Accuracy Far | AUC Near | AUC Far | Accuracy Drop | AUC Drop |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7079 | 0.7054 | 0.7267 | 0.7164 | 0.0025 | 0.0103 |
| Random Forest | 0.6997 | 0.6980 | 0.7251 | 0.7145 | **0.0017** | 0.0106 |

Both models remain highly stable under this naturally occurring temporal shift.

The accuracy loss is less than one percentage point for both models.

---

# 19. Race Fairness Under Temporal Shift

Race fairness is also measured on the far-term 2014 Q1 test population.

| Model | SPD | DIR | EOD | AOD |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.1019 | 0.6097 | 0.1504 | 0.0959 |
| Random Forest | 0.1040 | 0.6340 | 0.1241 | 0.0920 |

The disparities are smaller on this later population than on the near-term test set, demonstrating that fairness metrics themselves can change over time.

This is a key reason why deployed models require **repeated fairness audits**, not only repeated accuracy checks.

---

# 20. Shift Test 2 — Population Composition Shift

The second stress test intentionally changes the composition of the far-term population.

Records with larger values of:

```python
priors_count
```

receive greater sampling probability:

```python
weight = 1 + priors_count
```

The procedure resamples **existing real defendant records with replacement**.

It does not fabricate protected attributes or synthetic defendants.

This produces a higher-risk population.

## Population Change

| Statistic | Natural Far-Term | Shifted Population |
|---|---:|---:|
| Mean priors count | 2.89 | 8.38 |
| African-American proportion | 0.588 | 0.693 |
| Recidivism rate | 0.373 | 0.546 |

The stress test therefore represents a substantial change in deployment population composition.

---

# 21. Shift Test 2 Performance Results

| Model | Natural Accuracy | Shifted Accuracy | Natural AUC | Shifted AUC | Accuracy Drop | AUC Drop |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7054 | **0.6770** | 0.7164 | **0.7195** | 0.0285 | -0.0031 |
| Random Forest | 0.6980 | 0.6733 | 0.7145 | 0.6934 | **0.0248** | 0.0212 |

Under the harder population shift:

- both models lose roughly 2.5–2.9 percentage points of accuracy;
- Logistic Regression's ROC-AUC slightly **improves**;
- Random Forest's ROC-AUC decreases by approximately 0.021;
- Random Forest has a slightly smaller accuracy drop.

This demonstrates why robustness should be evaluated with multiple metrics rather than accuracy alone.

---

# 22. Fairness Under Population Shift

Race fairness after the population reweighting is:

| Model | SPD | DIR | EOD | AOD |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.2461 | 0.5849 | 0.3071 | 0.2123 |
| Random Forest | 0.2169 | **0.6384** | **0.2651** | **0.1842** |

The stress test changes not only predictive performance but also fairness.

This shows that a model can remain reasonably accurate while the demographic behavior of its predictions changes.

Therefore monitoring a deployed model should include both:

```text
performance drift
```

and:

```text
fairness drift
```

---

# 23. Trustworthiness Scoring

The notebook combines five dimensions into normalized scores between 0 and 1:

```text
Performance
Fairness
Calibration
Explainability
Robustness
```

A score closer to 1 represents a better result according to the benchmark's chosen normalization rules.

---

## Performance Score

Based on ROC-AUC:

```python
performance_score = (roc_auc - 0.5) / 0.5
```

This maps approximately:

```text
ROC-AUC = 0.50 -> score 0
ROC-AUC = 1.00 -> score 1
```

---

## Calibration Score

Based on ECE:

```python
calibration_score = 1 - ECE / 0.10
```

and clipped to the range `[0, 1]`.

---

## Fairness Score

The fairness score combines:

```text
1 - |SPD|
DIR term
1 - |EOD|
1 - |AOD|
```

Race and sex fairness scores are calculated and then averaged.

---

## Explainability Score

The explainability score is:

```text
40% explanation concentration
40% explanation stability
20% inherent interpretability
```

or:

```python
0.4 * concentration
+ 0.4 * stability
+ 0.2 * inherent_interpretability
```

---

## Robustness Score

The robustness score uses the mean positive accuracy degradation across both shift tests:

```python
1 - mean(
    temporal_accuracy_drop,
    reweighted_accuracy_drop
)
```

Negative degradation, meaning an improvement rather than a drop, is treated as zero loss.

---

# 24. Normalized Trustworthiness Scores

| Dimension | Logistic Regression | Random Forest |
|---|---:|---:|
| Performance | **0.4534** | 0.4502 |
| Fairness | 0.6377 | **0.6607** |
| Calibration | 0.6926 | **0.7272** |
| Explainability | **0.8815** | 0.6739 |
| Robustness | 0.9845 | **0.9868** |

An important result is visible here:

- Logistic Regression has the slightly higher performance score;
- Random Forest has slightly higher fairness, calibration, and robustness scores;
- Logistic Regression has a **large explainability advantage**.

The models therefore have different trustworthiness profiles even though their ordinary predictive performance is nearly identical.

---

# 25. Raw and Normalized Results

The notebook retains the original metrics alongside normalized scores so that the composite benchmark remains transparent.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| ROC-AUC | 0.7267 | 0.7251 |
| Accuracy | 0.7079 | 0.6997 |
| F1 | 0.5069 | 0.5210 |
| ECE | 0.0307 | 0.0273 |
| Brier Score | 0.1982 | 0.1994 |
| Race SPD | 0.1992 | 0.2211 |
| Race DIR | 0.3526 | 0.3711 |
| Race EOD | 0.3360 | 0.3277 |
| Race AOD | 0.2018 | 0.2167 |
| Sex SPD | 0.2110 | 0.2001 |
| Sex DIR | 0.2141 | 0.3345 |
| Sex EOD | 0.3066 | 0.2649 |
| Sex AOD | 0.2106 | 0.1896 |
| Top-3 explanation concentration | 0.7388 | 0.7058 |
| Explanation stability | 0.9650 | 0.9790 |
| Inherent interpretability | 1.0000 | 0.0000 |
| Temporal accuracy drop | 0.0025 | 0.0017 |
| Reweighted accuracy drop | 0.0285 | 0.0248 |
| Performance score | 0.4534 | 0.4502 |
| Fairness score | 0.6377 | 0.6607 |
| Calibration score | 0.6926 | 0.7272 |
| Explainability score | 0.8815 | 0.6739 |
| Robustness score | 0.9845 | 0.9868 |

Keeping this table prevents the normalized composite score from hiding the underlying measurements.

---

# 26. Three Weighting Scenarios

Trustworthiness is application-dependent.

A single weighting scheme cannot represent every deployment environment, so the notebook evaluates three different stakeholder priorities.

## Performance-Focused

```text
Performance      45%
Fairness         10%
Calibration      10%
Robustness       20%
Explainability   15%
```

## Fairness-Focused

```text
Performance      10%
Fairness         45%
Calibration      15%
Robustness       15%
Explainability   15%
```

## Balanced

```text
Performance      20%
Fairness         20%
Calibration      20%
Robustness       20%
Explainability   20%
```

The final score is a weighted sum:

```text
Trustworthiness =
    w_performance × Performance
  + w_fairness × Fairness
  + w_calibration × Calibration
  + w_explainability × Explainability
  + w_robustness × Robustness
```

---

# 27. Composite Trustworthiness Results

| Model | Performance-Focused | Fairness-Focused | Balanced |
|---|---:|---:|---:|
| Logistic Regression | **0.6662** | **0.7161** | **0.7299** |
| Random Forest | 0.6398 | 0.7005 | 0.6998 |

**Logistic Regression has the higher composite trustworthiness score under all three weighting scenarios.**

The result should be interpreted carefully.

Random Forest is slightly better on the notebook's normalized:

- fairness score;
- calibration score;
- robustness score.

However, Logistic Regression has:

- a small predictive-performance advantage;
- a very large explainability advantage because it is both highly SHAP-stable and inherently interpretable.

That explainability advantage is sufficient for Logistic Regression to finish ahead under all three weighting schemes.

---

# 28. Main Findings

The experiment produces several important conclusions.

### 1. Accuracy alone is insufficient

The models are almost tied on ROC-AUC:

```text
Logistic Regression = 0.7267
Random Forest       = 0.7251
```

Yet their broader trustworthiness profiles differ.

---

### 2. Real fairness problems are visible

Unlike the previous methodology demonstration, this project uses actual protected demographic variables.

Both models exhibit substantial disparities across race and sex.

In particular, the race DIR values on the primary test set are:

```text
0.3526 and 0.3711
```

which are far from parity.

---

### 3. Good performance can coexist with unfair outcomes

A model can achieve reasonable ROC-AUC and calibration while still producing substantially different positive prediction and error rates across demographic groups.

---

### 4. Distribution shift affects more than accuracy

The natural temporal shift causes only very small accuracy degradation.

The stronger high-priors population shift causes a larger degradation and changes fairness measurements.

This demonstrates that distribution-shift monitoring should evaluate:

```text
performance
calibration
fairness
```

rather than only overall accuracy.

---

### 5. Explainability changes model selection

Logistic Regression's strongest advantage is not a dramatic accuracy improvement.

Its main advantage is its much higher explainability score:

```text
Logistic Regression = 0.8815
Random Forest       = 0.6739
```

This helps Logistic Regression obtain the highest composite score under all three weighting scenarios.

---

# 29. Deployment Recommendation

If one of the two models must be selected for further development, the results support **Logistic Regression** because it achieves the highest composite trustworthiness score under every tested weighting scenario and provides much stronger inherent interpretability.

However, **neither model should be deployed as-is for a consequential recidivism-risk decision system**.

The fairness audit identifies substantial demographic disparities, especially on the primary temporal test population.

Before deployment, the system would require additional work such as:

- fairness-aware model development or mitigation;
- threshold and decision-policy review;
- repeated protected-group audits;
- calibration monitoring;
- distribution-drift monitoring;
- human review and governance appropriate to the application;
- re-evaluation whenever the deployment population changes.

The main deployment lesson is:

> A model should not be considered trustworthy simply because its predictive accuracy remains stable. Fairness, calibration, explainability, and robustness must also remain acceptable as the deployment population changes.

---

# 30. HTML Report

The final notebook section generates a standalone HTML report containing:

- temporal data analysis;
- split statistics;
- performance tables;
- calibration results;
- reliability plot;
- fairness tables;
- SHAP feature importance;
- robustness tests;
- normalized trustworthiness scores;
- weighting scenarios;
- composite scores;
- deployment recommendation.

Figures are embedded directly into the HTML as Base64 data, making the report self-contained.

---

# 31. Project Structure

A suggested repository layout is:

```text
trustworthiness-distribution-shift/
│
├── compas_trustworthiness_evaluation.ipynb
├── README.md
├── trustworthiness_report.html
└── requirements.txt
```

---

# 32. Requirements

The notebook uses the following Python packages:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
shap
requests
```

Install them with:

```bash
pip install numpy pandas matplotlib scipy scikit-learn shap requests
```

---

# 33. Running the Project

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY>
```

Create and activate a virtual environment if desired:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scipy scikit-learn shap requests jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
compas_trustworthiness_evaluation.ipynb
```

and run the cells from top to bottom.

The dataset is downloaded automatically by the notebook, so no manual dataset download is required.

---

# 34. Reproducibility

The project sets:

```python
RANDOM_STATE = 42
```

and initializes NumPy random state with the same seed.

This makes operations such as model fitting, SHAP sampling, and the population-shift resampling experiment reproducible.

---

# 35. Limitations

This project is a **methodology demonstration** and should not be interpreted as a production-ready criminal-justice risk model.

Important limitations include:

- the analysis uses only a subset of available COMPAS records;
- the race audit is restricted to African-American and Caucasian groups;
- fairness metrics do not by themselves establish legal or ethical acceptability;
- the second population shift is an intentionally constructed stress test based on weighted resampling of real records;
- the normalization formulas and trustworthiness weights are benchmark design choices rather than universal standards;
- the models are relatively simple and are not tuned for production deployment;
- the experiment demonstrates trustworthiness evaluation methodology rather than recommending operational use of COMPAS.

---

# 36. Final Conclusion

This project demonstrates how model evaluation changes when the question moves from:

> **Which model is the most accurate?**

to:

> **Which model is the most trustworthy when deployed on future and changing populations?**

The notebook addresses the major weakness of the previous task by using **real protected attributes rather than simulated groups**.

It also uses a genuine chronological split and evaluates two different forms of distribution shift.

The main result is that Logistic Regression and Random Forest have nearly identical predictive discrimination, but they differ across fairness, calibration, explainability, and robustness.

Logistic Regression achieves the highest overall composite score in:

```text
Performance-focused scenario: 0.6662
Fairness-focused scenario:    0.7161
Balanced scenario:            0.7299
```

compared with Random Forest:

```text
Performance-focused scenario: 0.6398
Fairness-focused scenario:    0.7005
Balanced scenario:            0.6998
```

Therefore, **Logistic Regression is the preferred model within this benchmark**, primarily because its comparable predictive performance is combined with substantially stronger explainability.

Nevertheless, the fairness results show that **neither model is appropriate for direct consequential deployment without further fairness mitigation, governance, and repeated auditing under changing population conditions**.

---

## GitHub Repository

Add the final repository link here:

```text
<YOUR-GITHUB-REPOSITORY-URL>
```
