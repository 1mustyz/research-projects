# Multi-Model Evaluation Pipeline

A comprehensive machine learning evaluation pipeline for comparing multiple classification models across **predictive performance, explainability, fairness, robustness, and overall trustworthiness**.

The goal of this project is to move beyond the traditional approach of selecting a machine learning model based only on accuracy. Instead, the pipeline evaluates each candidate model from multiple perspectives and combines these dimensions into a final **trust/safety score** to support more informed model selection.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Why Multi-Model Evaluation?](#2-why-multi-model-evaluation)
3. [Models Evaluated](#3-models-evaluated)
4. [Evaluation Framework](#4-evaluation-framework)
5. [1. Model Performance](#5-1-model-performance)
6. [2. SHAP Explainability](#6-2-shap-explainability)
7. [3. Fairness Evaluation](#7-3-fairness-evaluation)
8. [4. Robustness Under Perturbations](#8-4-robustness-under-perturbations)
9. [5. Protected Attribute Ablation](#9-5-protected-attribute-ablation)
10. [6. Final Trust Report](#10-6-final-trust-report)
11. [Overall Results](#11-overall-results)
12. [Key Findings](#12-key-findings)
13. [Important Interpretation of the Results](#13-important-interpretation-of-the-results)
14. [Limitations and Considerations](#14-limitations-and-considerations)
15. [Conclusion](#15-conclusion)

---

# 1. Overview

The **Multi-Model Evaluation Pipeline** is designed to provide a multidimensional assessment of machine learning classification models.

Traditional model evaluation often focuses primarily on predictive performance:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

While these metrics are important, they do not provide a complete picture of whether a model is suitable for deployment.

A model may achieve excellent predictive performance but:

- Perform differently across demographic groups.
- Depend heavily on a small number of features.
- Be difficult to interpret.
- Experience significant performance degradation when input data is noisy or incomplete.

Therefore, this pipeline evaluates models across four major dimensions:

```text
                    Multi-Model Evaluation
                            |
        +-------------------+-------------------+
        |                   |                   |
   Performance        Explainability        Fairness
        |                   |                   |
   ROC-AUC, etc.          SHAP             Group disparities
        |                   |                   |
        +-------------------+-------------------+
                            |
                       Robustness
                            |
              Gaussian Noise + Feature Dropout
                            |
                            v
                     Final Trust Report
                            |
                            v
                    Composite Safety Score
```

The pipeline therefore answers a broader question:

> **Which model provides the best overall balance between predictive performance, fairness, robustness, explainability, and trustworthiness?**

---

# 2. Why Multi-Model Evaluation?

The pipeline is based on the idea that there is no universally "best" machine learning model.

Different models can perform well in different dimensions.

For example:

- One model may have the highest ROC-AUC.
- Another may be easier to explain.
- Another may be more robust to noisy inputs.
- Another may have lower demographic disparity.

This creates a multi-objective model-selection problem.

The pipeline therefore evaluates models using four main dimensions:

### Performance

How accurately does the model make predictions?

### Explainability

Can we understand which features influence the model's predictions and how?

### Fairness

Does the model behave differently across protected demographic groups?

### Robustness

Does the model maintain performance when its input data is perturbed?

The final trust report combines these dimensions using predefined weights.

---

# 3. Models Evaluated

The pipeline compares four classification algorithms:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **HistGradientBoosting**

Each model provides a different trade-off.

### Logistic Regression

A relatively simple and interpretable linear classification model.

It provides a strong baseline and is generally easier to understand than complex ensemble models.

In this evaluation, Logistic Regression demonstrated:

- Strong robustness
- High explainability
- Lower predictive performance than HistGradientBoosting
- Moderate fairness performance

---

### Decision Tree

A rule-based model that makes predictions through a sequence of feature-based decisions.

Decision Trees are naturally interpretable because their decisions can be represented as human-readable rules.

In this evaluation, Decision Tree demonstrated:

- High explainability
- The highest fairness risk according to DPD
- Lower predictive performance
- Moderate robustness

---

### Random Forest

An ensemble model that combines multiple decision trees.

Random Forest generally provides stronger predictive performance than a single Decision Tree while often being more robust to individual feature variations.

In this evaluation, Random Forest demonstrated:

- Strong robustness
- Moderate predictive performance
- Moderate fairness
- Low explainability score according to the pipeline's normalized SHAP-based criterion

---

### HistGradientBoosting

A gradient boosting ensemble model that builds a sequence of decision trees to improve predictive performance.

In this evaluation, HistGradientBoosting demonstrated:

- The highest predictive performance
- Strong fairness score
- The weakest robustness under the tested perturbations
- Low explainability according to the pipeline's normalized criterion

---

# 4. Evaluation Framework

The pipeline evaluates each model using the following framework:

```text
                 Train Candidate Models
                         |
                         v
                Performance Evaluation
                         |
                         v
                  SHAP Analysis
                         |
                         v
                 Fairness Analysis
                         |
                         v
              Robustness Evaluation
                         |
                         v
                Trust Score Calculation
                         |
                         v
                Final Model Comparison
```

The final evaluation does not rely on a single metric.

Instead, it combines multiple dimensions into a composite score.

---

# 5. 1. Model Performance

The performance section evaluates the predictive capability of each model.

Common classification metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Among these, ROC-AUC is used in the final trust report to identify the best-performing model.

The output of the pipeline identifies:

```text
Best performing model (highest ROC-AUC):
HistGradientBoosting
```

Therefore, HistGradientBoosting achieved the highest ROC-AUC among the evaluated models.

This means that, according to the performance criterion used by the pipeline, HistGradientBoosting was the strongest predictive model.

However, the pipeline does not automatically conclude that it is the best model overall.

This is because predictive performance is only one component of trustworthiness.

---

# 6. 2. SHAP Explainability

The pipeline uses **SHAP (SHapley Additive exPlanations)** to investigate how features influence model predictions.

SHAP provides two complementary levels of analysis:

### Global Explainability

Global SHAP analysis identifies which features have the strongest overall influence on the model.

The importance of a feature is commonly summarized using its mean absolute SHAP value:

```text
Mean |SHAP value|
```

This measures the average magnitude of a feature's contribution across observations.

Importantly:

> **Mean absolute SHAP value measures the strength of influence, not the direction of influence.**

For example, if a feature has a high mean absolute SHAP value, it means the feature strongly affects predictions overall.

However, this does not mean that the feature always increases the probability of the positive class.

The feature may:

- Push some predictions toward the positive class.
- Push other predictions toward the negative class.

The sign and direction of SHAP values must therefore be examined separately.

---

## SHAP-Based Results

The pipeline produced the following summary:

| Model                | Top Feature                         | Top-3 Concentration | Inherently Interpretable |
| -------------------- | ----------------------------------- | ------------------: | ------------------------ |
| Logistic Regression  | `sex_Female`                        |            0.487438 | True                     |
| Decision Tree        | `marital.status_Married-civ-spouse` |            0.689298 | True                     |
| Random Forest        | `marital.status_Married-civ-spouse` |            0.488672 | False                    |
| HistGradientBoosting | `marital.status_Married-civ-spouse` |            0.628353 | False                    |

### Top Feature

The `top1_feature` column identifies the feature with the strongest SHAP importance according to the pipeline's analysis.

For example:

- Logistic Regression → `sex_Female`
- Decision Tree → `marital.status_Married-civ-spouse`
- Random Forest → `marital.status_Married-civ-spouse`
- HistGradientBoosting → `marital.status_Married-civ-spouse`

This tells us which feature is most influential in the model's predictions, but it does **not**, by itself, tell us whether that feature increases or decreases predictions.

The direction requires inspection of the actual SHAP values or SHAP summary plot.

---

## Top-3 Concentration

The `top3_concentration` value describes how concentrated the model's SHAP importance is among its three most influential features.

Higher values indicate that a larger proportion of the model's overall measured feature importance is concentrated in the top three features.

The results were:

```text
Decision Tree             0.689298
HistGradientBoosting      0.628353
Logistic Regression       0.487438
Random Forest             0.488672
```

The Decision Tree has the highest top-three concentration.

This suggests that its predictions are more concentrated around a small number of influential features according to this measure.

HistGradientBoosting also has relatively high concentration.

Logistic Regression and Random Forest have lower top-three concentration, suggesting a more distributed contribution of features.

---

## Inherently Interpretable

The pipeline also identifies whether the model is considered inherently interpretable.

The results are:

```text
Logistic Regression       True
Decision Tree             True
Random Forest             False
HistGradientBoosting      False
```

Logistic Regression and Decision Tree are treated as inherently interpretable.

Random Forest and HistGradientBoosting are treated as less inherently interpretable because their ensemble structures are more complex.

This does not mean that Random Forest and HistGradientBoosting cannot be explained.

Instead, it means that they generally require additional explainability techniques such as SHAP to understand their predictions.

---

# 7. 3. Fairness Evaluation

The fairness section investigates whether model predictions and errors differ across demographic groups.

The pipeline uses a protected attribute, such as:

```text
sex
```

The model's performance is examined separately for different groups.

The group-level analysis includes:

- Accuracy
- Selection rate
- False Positive Rate (FPR)
- False Negative Rate (FNR)

The analysis also calculates:

- Demographic Parity Difference (DPD)
- Equalized Odds Difference (EOD)
- Disparate Impact Ratio (DIR)

---

## Group-Level Metrics

### Accuracy

Accuracy measures the proportion of correct predictions within each demographic group.

A difference in accuracy between groups indicates that the model's overall predictive performance varies across groups.

However, accuracy alone is not sufficient to determine fairness.

---

### Selection Rate

Selection rate measures how frequently the model predicts the positive class for a group.

For example:

```text
Selection Rate = P(prediction = positive | group)
```

Large differences in selection rates contribute to demographic disparity.

---

### False Positive Rate

The False Positive Rate measures how often the model incorrectly predicts the positive class for individuals who actually belong to the negative class.

[
FPR = \frac{FP}{FP+TN}
]

---

### False Negative Rate

The False Negative Rate measures how often the model incorrectly predicts the negative class for individuals who actually belong to the positive class.

[
FNR = \frac{FN}{FN+TP}
]

Differences in FPR and FNR across demographic groups indicate that the model's errors are not distributed equally.

---

## Demographic Parity Difference

Demographic Parity Difference evaluates differences in positive prediction rates between groups.

A value closer to zero generally indicates lower disparity.

Conceptually:

[
DPD =
|SelectionRate_{Group1} -
SelectionRate_{Group2}|
]

A higher DPD indicates a larger difference in positive prediction rates.

The pipeline uses DPD as an important fairness criterion.

---

## Equalized Odds Difference

Equalized Odds focuses on whether the model's error behavior is similar across demographic groups.

It considers differences in:

- True Positive Rate
- False Positive Rate

A value closer to zero generally indicates less disparity.

The purpose is to determine whether groups experience similar prediction outcomes conditional on their actual class.

---

## Disparate Impact Ratio

The Disparate Impact Ratio compares selection rates between groups.

A value closer to 1 indicates more similar selection rates.

The pipeline uses the commonly referenced 0.8 threshold as a rule-of-thumb screening criterion, although this should not be treated as a universal fairness boundary.

A low ratio indicates greater disparity in selection rates.

---

## Fairness Risk

The pipeline identifies:

```text
Model with highest fairness risk (max DPD):
Decision Tree
```

Therefore, Decision Tree has the highest measured demographic parity disparity among the evaluated models.

This is also reflected in the normalized fairness scores:

```text
HistGradientBoosting      0.9090
Random Forest             0.6090
Logistic Regression       0.5106
Decision Tree             0.0000
```

Higher fairness scores represent better relative fairness according to the normalization used by the pipeline.

---

# 8. 4. Robustness Under Perturbations

The robustness section evaluates how well each model maintains its predictive performance when the input data is deliberately degraded.

Two perturbation methods are used:

1. Gaussian Noise
2. Feature Dropout

The central question is:

> **How much does model performance degrade when the input data becomes noisy or incomplete?**

---

## Gaussian Noise

Gaussian noise introduces random perturbations to numerical input features.

Conceptually:

[
X_{noisy}=X+\epsilon
]

where:

[
\epsilon \sim N(0,\sigma^2)
]

Different noise levels are tested:

```text
0.00
0.05
0.10
0.20
0.35
```

A noise level of `0.00` represents the baseline with no perturbation.

As noise increases, model accuracy is monitored.

---

## Gaussian Noise Results

| Noise Level | Decision Tree | HistGradientBoosting | Logistic Regression | Random Forest |
| ----------- | ------------: | -------------------: | ------------------: | ------------: |
| 0.00        |        0.8457 |               0.8705 |              0.8504 |        0.8577 |
| 0.05        |        0.8427 |               0.8413 |              0.8509 |        0.8560 |
| 0.10        |        0.8391 |               0.8330 |              0.8497 |        0.8547 |
| 0.20        |        0.8286 |               0.8301 |              0.8458 |        0.8532 |
| 0.35        |        0.8226 |               0.8226 |              0.8311 |        0.8434 |

Random Forest shows the smallest average accuracy degradation under Gaussian noise.

The mean accuracy drops due to noise are:

```text
Random Forest             0.0059
Logistic Regression       0.0061
Decision Tree             0.0125
HistGradientBoosting      0.0387
```

Therefore:

> Random Forest and Logistic Regression are approximately comparable in their robustness to Gaussian noise, while HistGradientBoosting is substantially more sensitive to this perturbation.

---

## Feature Dropout

Feature dropout simulates situations in which some feature information is missing or corrupted.

The tested dropout rates are:

```text
0.00
0.05
0.10
0.20
0.35
```

Higher dropout means more input information is affected.

The purpose is to determine whether the model can maintain performance when some features are unavailable.

---

## Feature Dropout Results

| Dropout Rate | Decision Tree | HistGradientBoosting | Logistic Regression | Random Forest |
| ------------ | ------------: | -------------------: | ------------------: | ------------: |
| 0.00         |        0.8457 |               0.8705 |              0.8504 |        0.8577 |
| 0.05         |        0.8389 |               0.8640 |              0.8451 |        0.8549 |
| 0.10         |        0.8298 |               0.8563 |              0.8417 |        0.8473 |
| 0.20         |        0.8232 |               0.8467 |              0.8335 |        0.8374 |
| 0.35         |        0.8009 |               0.8283 |              0.8149 |        0.8207 |

The mean accuracy drops due to feature dropout are:

```text
Logistic Regression       0.0166
Random Forest             0.0176
HistGradientBoosting      0.0217
Decision Tree             0.0225
```

Therefore:

> Logistic Regression demonstrates the smallest average performance degradation under feature dropout.

---

## Overall Robustness

The pipeline combines the robustness results into an overall mean accuracy drop:

| Model                | Mean Accuracy Drop — Noise | Mean Accuracy Drop — Dropout | Overall Mean Accuracy Drop |
| -------------------- | -------------------------: | ---------------------------: | -------------------------: |
| Logistic Regression  |                     0.0061 |                       0.0166 |                 **0.0113** |
| Random Forest        |                     0.0059 |                       0.0176 |                 **0.0118** |
| Decision Tree        |                     0.0125 |                       0.0225 |                 **0.0175** |
| HistGradientBoosting |                     0.0387 |                       0.0217 |                 **0.0302** |

The smallest overall accuracy drop belongs to Logistic Regression.

Therefore:

```text
Most robust model:
Logistic Regression
```

An important finding is that **the model with the best clean-data performance is not necessarily the most robust**.

HistGradientBoosting has the highest baseline accuracy, but it experiences the largest average degradation under the tested perturbations.

---

# 9. 5. Protected Attribute Ablation

The pipeline also evaluates model fairness with and without the protected attribute.

The purpose is to investigate:

> **Does explicitly providing the protected attribute to the model significantly contribute to observed fairness disparities?**

Two scenarios are compared:

### With Protected Attribute

The model is trained using `sex` as an input feature.

### Without Protected Attribute

The `sex` feature is removed before model training.

The results compare:

- Demographic Parity Difference
- Disparate Impact Ratio

The observed results were:

| Model                | DPD With Sex | DI With Sex | DPD Without Sex | DI Without Sex | DPD Change |
| -------------------- | -----------: | ----------: | --------------: | -------------: | ---------: |
| Logistic Regression  |       0.1752 |      0.3166 |          0.1709 |         0.3278 |    -0.0043 |
| Decision Tree        |       0.2115 |      0.2960 |          0.2112 |         0.2992 |    -0.0003 |
| Random Forest        |       0.1597 |      0.3084 |          0.1583 |         0.3046 |    -0.0013 |
| HistGradientBoosting |       0.1738 |      0.3392 |          0.1706 |         0.3535 |    -0.0033 |

The DPD changes are all small:

```text
Logistic Regression       -0.0043
Decision Tree             -0.0003
Random Forest             -0.0013
HistGradientBoosting      -0.0033
```

This indicates that removing `sex` produces only a marginal reduction in demographic parity disparity.

The key interpretation is:

> **Simply removing the protected attribute does not eliminate the observed fairness disparity.**

Other features may contain information correlated with the protected attribute.

These features can act as proxy variables.

For example:

```text
Sex
 |
 +----> Education
 |
 +----> Occupation
 |
 +----> Marital Status
 |
 +----> Other socioeconomic characteristics
```

Even without explicitly providing `sex`, the model may learn patterns associated with sex through other features.

Therefore:

> **The results suggest that the observed demographic disparity is not primarily caused by the direct inclusion of `sex` alone.**

The protected attribute ablation experiment demonstrates why removing sensitive features is not necessarily sufficient to achieve fairness.

---

# 10. 6. Final Trust Report

The Final Trust Report combines the four major evaluation dimensions into a composite score.

The weights used are:

```text
Performance       0.35
Fairness          0.30
Robustness        0.25
Explainability    0.10
```

The total weight is:

[
0.35+0.30+0.25+0.10=1.00
]

The composite score is conceptually:

[
SafetyScore =
0.35(PerformanceScore)

- 0.30(FairnessScore)
- 0.25(RobustnessScore)
- 0.10(ExplainabilityScore)
  ]

The weights indicate that:

1. Predictive performance is the most important dimension.
2. Fairness is the second most important.
3. Robustness is the third.
4. Explainability has the smallest weight.

---

## Final Normalized Scores

| Model                | Performance | Fairness | Robustness | Explainability | Safety Score |
| -------------------- | ----------: | -------: | ---------: | -------------: | -----------: |
| HistGradientBoosting |      1.0000 |   0.9090 |     0.0000 |         0.0137 |   **0.6241** |
| Logistic Regression  |      0.1763 |   0.5106 |     1.0000 |         0.9802 |   **0.5629** |
| Random Forest        |      0.3765 |   0.6090 |     0.9783 |         0.0000 |   **0.5591** |
| Decision Tree        |      0.0874 |   0.0000 |     0.6732 |         1.0000 |   **0.2989** |

The normalized scores generally follow the principle:

```text
1.0 = Best relative score
0.0 = Worst relative score
```

These scores are relative to the models included in this evaluation.

A score of `1.0` does not mean that a model is perfect.

It means that the model achieved the best result among the evaluated candidates for that particular dimension.

---

## HistGradientBoosting

```text
Performance       1.0000
Fairness          0.9090
Robustness        0.0000
Explainability    0.0137
Safety Score      0.6241
```

HistGradientBoosting is the strongest predictive model and has a high fairness score.

However, it is the least robust model according to the perturbation experiments and has low inherent explainability.

Nevertheless, because performance has the largest weight and fairness has the second-largest weight, HistGradientBoosting achieves the highest composite score.

---

## Logistic Regression

```text
Performance       0.1763
Fairness          0.5106
Robustness        1.0000
Explainability    0.9802
Safety Score      0.5629
```

Logistic Regression provides a strong alternative.

It is:

- The most robust model.
- Highly explainable.
- Less predictive than HistGradientBoosting according to the normalized performance score.

Its final score is close to HistGradientBoosting.

This means that if robustness and interpretability were assigned greater importance, Logistic Regression could potentially become the preferred model.

---

## Random Forest

```text
Performance       0.3765
Fairness          0.6090
Robustness        0.9783
Explainability    0.0000
Safety Score      0.5591
```

Random Forest provides a strong balance between robustness and fairness.

Its robustness score is very close to Logistic Regression.

The difference between its final score and Logistic Regression is also very small:

```text
0.5629 - 0.5591 = 0.0038
```

Therefore, the two models are very close in terms of their final composite trust score.

---

## Decision Tree

```text
Performance       0.0874
Fairness          0.0000
Robustness        0.6732
Explainability    1.0000
Safety Score      0.2989
```

Decision Tree is the most inherently interpretable model according to the pipeline.

However, it has:

- The lowest performance score.
- The highest fairness risk.
- A moderate robustness score.

Because explainability has only a 10% weight, its perfect explainability score does not compensate for its weaker performance and fairness results.

It therefore receives the lowest overall composite score.

---

# 11. Overall Results

The pipeline identifies the following model for each dimension:

| Evaluation Dimension           | Best Model           |
| ------------------------------ | -------------------- |
| Highest ROC-AUC / Performance  | HistGradientBoosting |
| Most Explainable               | Decision Tree        |
| Highest Fairness Risk          | Decision Tree        |
| Most Robust                    | Logistic Regression  |
| Highest Composite Safety Score | HistGradientBoosting |

This demonstrates that the "best" model depends on the evaluation criterion.

```text
                    Model Strengths

HistGradientBoosting
        |
        +--> Best predictive performance
        +--> High fairness score
        +--> Weak robustness
        +--> Low explainability

Logistic Regression
        |
        +--> Best robustness
        +--> High explainability
        +--> Moderate fairness
        +--> Lower performance

Random Forest
        |
        +--> Very strong robustness
        +--> Moderate fairness
        +--> Moderate performance
        +--> Low explainability score

Decision Tree
        |
        +--> Best inherent explainability
        +--> Highest fairness risk
        +--> Lower performance
        +--> Moderate robustness
```

---

# 12. Key Findings

## Finding 1: Highest performance does not mean highest robustness

HistGradientBoosting achieves the highest clean-data predictive performance.

However, it also experiences the largest average accuracy degradation under the tested perturbations.

Therefore:

> A model that performs best under ideal conditions may not be the model that performs best under imperfect conditions.

---

## Finding 2: Logistic Regression is highly robust

Logistic Regression achieves the smallest overall mean accuracy drop:

```text
0.0113
```

It is therefore the most robust model according to the pipeline's robustness criterion.

It also achieves a very high explainability score.

This makes it a strong candidate when reliability under perturbed inputs and interpretability are important.

---

## Finding 3: Random Forest is close to Logistic Regression in robustness

Random Forest achieves:

```text
Overall Mean Accuracy Drop = 0.0118
```

compared with:

```text
Logistic Regression = 0.0113
```

The difference is very small.

Therefore, Random Forest should also be considered highly robust under the specific perturbations tested.

---

## Finding 4: Removing the protected attribute does not eliminate disparity

The fairness ablation results show only small changes after removing `sex`.

This suggests that fairness disparities may be influenced by other correlated features.

Therefore:

> Protected attribute removal alone is not sufficient to guarantee fairness.

---

## Finding 5: Decision Tree is interpretable but has fairness concerns

Decision Tree is the most explainable model according to the pipeline.

However, it also has the highest DPD and therefore the highest measured fairness risk.

This illustrates that interpretability and fairness are separate properties.

A model can be easy to understand and still produce unequal outcomes across demographic groups.

---

## Finding 6: The final model depends on the weighting scheme

The pipeline identifies HistGradientBoosting as the safest model according to the current weights:

```text
Performance       35%
Fairness          30%
Robustness        25%
Explainability    10%
```

However, this result is **conditional on those weights**.

If robustness or explainability were considered more important, the final ranking could change.

Therefore:

> The composite safety score should be interpreted as a decision-support mechanism rather than an absolute measure of model safety.

---

# 13. Important Interpretation of the Results

The Final Trust Report should not be interpreted as proving that HistGradientBoosting is universally the "safest" model.

Instead, it means:

> **Under the evaluation criteria, normalization methods, perturbation tests, fairness definitions, explainability analysis, and weighting scheme used in this pipeline, HistGradientBoosting achieved the highest composite score.**

This distinction is important.

The composite score is dependent on:

- The models being compared.
- The dataset.
- The protected attribute.
- The selected fairness metrics.
- The robustness perturbations.
- The explainability methodology.
- The normalization approach.
- The chosen weights.

Changing any of these may change the final ranking.

---

# 14. Limitations and Considerations

## 14.1 Fairness is multidimensional

No single fairness metric fully captures fairness.

DPD, Equalized Odds, and Disparate Impact measure different aspects of group disparity.

Therefore, fairness should be interpreted using multiple metrics rather than a single score.

---

## 14.2 Removing protected attributes is not enough

A protected attribute can be indirectly encoded through other features.

Therefore, fairness analysis should also investigate potential proxy variables and the underlying data distribution.

---

## 14.3 Robustness results are perturbation-specific

The robustness experiments use Gaussian noise and feature dropout.

Therefore, the results demonstrate robustness against these specific perturbations.

They do not automatically prove robustness against:

- Adversarial attacks.
- Distribution shifts.
- Data poisoning.
- Real-world missing-data mechanisms.
- Other forms of input corruption.

---

## 14.4 Explainability scores depend on the chosen methodology

The explainability ranking combines the pipeline's assumptions about inherent interpretability and SHAP-derived information.

A low explainability score does not mean that a model cannot be explained.

It means that the model is less inherently interpretable according to the evaluation methodology.

SHAP can still provide useful post-hoc explanations for complex models.

---

## 14.5 Composite scores are subjective

The weights:

```text
Performance       0.35
Fairness          0.30
Robustness        0.25
Explainability    0.10
```

represent a particular prioritization.

Different application domains may require different weights.

For a high-stakes decision system, for example, fairness or robustness might reasonably receive a larger weight.

Therefore, the weights should be justified according to the intended application.

---

# 15. Conclusion

The Multi-Model Evaluation Pipeline provides a comprehensive framework for evaluating machine learning models beyond conventional predictive performance.

Rather than selecting a model solely because it achieves the highest accuracy or ROC-AUC, the pipeline evaluates four complementary dimensions:

1. **Performance** — How well does the model predict?
2. **Explainability** — Can its decisions be understood?
3. **Fairness** — Does it behave equitably across demographic groups?
4. **Robustness** — Does it remain reliable when input data is perturbed?

The final trust report combines these dimensions into a weighted composite score.

The evaluation shows that:

- **HistGradientBoosting** provides the strongest predictive performance and achieves the highest overall composite score under the selected weighting scheme.
- **Logistic Regression** provides the strongest robustness and high explainability.
- **Random Forest** provides a strong robustness profile and a competitive overall score.
- **Decision Tree** provides the strongest inherent interpretability but has the highest measured fairness risk and lowest composite safety score.

The most important conclusion is therefore:

> **Model selection should not be based solely on predictive performance. A reliable machine learning system should also be evaluated for fairness, explainability, and robustness. The Multi-Model Evaluation Pipeline provides a structured approach for making this broader assessment and for understanding the trade-offs between competing models.**

The final recommendation should therefore be interpreted in the context of the intended application and the priorities assigned to performance, fairness, robustness, and explainability.
