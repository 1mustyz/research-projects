# Complete List of All Features from Your Code

After analyzing the entire pipeline, here are ALL features at each stage:

## Stage 1: Raw Data Columns (Initial Load)

| Column   | Meaning                  |
| -------- | ------------------------ |
| Date     | Match date               |
| HomeTeam | Home team name           |
| AwayTeam | Away team name           |
| FTHG     | Full Time Home Goals     |
| FTAG     | Full Time Away Goals     |
| FTR      | Full Time Result (H/D/A) |

## Stage 2: After Feature Engineering (playing_stat)

After all transformations (`get_gss()`, `get_agg_points()`, `add_form_df()`, `get_mw()`):

### Complete 23-Column Dataset

| #   | Column   | Meaning                                      |
| --- | -------- | -------------------------------------------- |
| 1   | Date     | Match date                                   |
| 2   | HomeTeam | Home team name                               |
| 3   | AwayTeam | Away team name                               |
| 4   | FTHG     | Full Time Home Goals (actual result)         |
| 5   | FTAG     | Full Time Away Goals (actual result)         |
| 6   | FTR      | Full Time Result (H/NH after transformation) |
| 7   | HTGS     | Home Team Goals Scored (cumulative)          |
| 8   | ATGS     | Away Team Goals Scored (cumulative)          |
| 9   | HTGC     | Home Team Goals Conceded (cumulative)        |
| 10  | ATGC     | Away Team Goals Conceded (cumulative)        |
| 11  | HTP      | Home Team Points (cumulative)                |
| 12  | ATP      | Away Team Points (cumulative)                |
| 13  | HM1      | Home team result 1 match ago (W/D/L/M)       |
| 14  | HM2      | Home team result 2 matches ago               |
| 15  | HM3      | Home team result 3 matches ago               |
| 16  | HM4      | Home team result 4 matches ago               |
| 17  | HM5      | Home team result 5 matches ago               |
| 18  | AM1      | Away team result 1 match ago (W/D/L/M)       |
| 19  | AM2      | Away team result 2 matches ago               |
| 20  | AM3      | Away team result 3 matches ago               |
| 21  | AM4      | Away team result 4 matches ago               |
| 22  | AM5      | Away team result 5 matches ago               |
| 23  | MW       | Match Week (1-38)                            |

## Stage 3: After Adding Calculated Features

After streak detection and goal differences:

### Additional 15 Columns

| #   | Column        | Meaning                                  |
| --- | ------------- | ---------------------------------------- |
| 24  | HTFormPtsStr  | Home team form string (e.g., "WWLDW")    |
| 25  | ATFormPtsStr  | Away team form string                    |
| 26  | HTFormPts     | Home team form points (0-15)             |
| 27  | ATFormPts     | Away team form points (0-15)             |
| 28  | HTWinStreak3  | Home team won last 3 (1/0)               |
| 29  | HTWinStreak5  | Home team won last 5 (1/0)               |
| 30  | HTLossStreak3 | Home team lost last 3 (1/0)              |
| 31  | HTLossStreak5 | Home team lost last 5 (1/0)              |
| 32  | ATWinStreak3  | Away team won last 3 (1/0)               |
| 33  | ATWinStreak5  | Away team won last 5 (1/0)               |
| 34  | ATLossStreak3 | Away team lost last 3 (1/0)              |
| 35  | ATLossStreak5 | Away team lost last 5 (1/0)              |
| 36  | HTGD          | Home Team Goal Difference (scaled by MW) |
| 37  | ATGD          | Away Team Goal Difference (scaled by MW) |
| 38  | DiffPts       | Points difference (scaled by MW)         |
| 39  | DiffFormPts   | Form points difference (scaled by MW)    |

**Total at this stage:** 39 columns

## Stage 4: After Feature Selection (dataset2)

After dropping columns:

### Remaining 10 Features + 1 Target

| #          | Feature     | Type        | Meaning                            |
| ---------- | ----------- | ----------- | ---------------------------------- |
| 1          | HTGD        | Numeric     | Home Team Goal Difference (scaled) |
| 2          | ATGD        | Numeric     | Away Team Goal Difference (scaled) |
| 3          | HTP         | Numeric     | Home Team Points (scaled)          |
| 4          | ATP         | Numeric     | Away Team Points (scaled)          |
| 5          | DiffFormPts | Numeric     | Form points difference (scaled)    |
| 6          | HM1         | Categorical | Home result 1 match ago (W/D/L/M)  |
| 7          | HM2         | Categorical | Home result 2 matches ago          |
| 8          | HM3         | Categorical | Home result 3 matches ago          |
| 9          | AM1         | Categorical | Away result 1 match ago            |
| 10         | AM2         | Categorical | Away result 2 matches ago          |
| 11         | AM3         | Categorical | Away result 3 matches ago          |
| **Target** | **FTR**     | **Binary**  | **Full Time Result (H/NH)**        |

## Stage 5: After One-Hot Encoding (X_all)

After `preprocess_features()`:

### Final 29 Model Features

#### Numeric Features (5)

- HTGD - Standardized
- ATGD - Standardized
- HTP - Standardized
- ATP - Standardized
- DiffFormPts - NOT standardized

#### One-Hot Encoded Features (24)

**Home Team Form (12 features):**

- HM1_D, HM1_L, HM1_M, HM1_W
- HM2_D, HM2_L, HM2_M, HM2_W
- HM3_D, HM3_L, HM3_M, HM3_W

**Away Team Form (12 features):**

- AM1_D, AM1_L, AM1_M, AM1_W
- AM2_D, AM2_L, AM2_M, AM2_W
- AM3_D, AM3_L, AM3_M, AM3_W

## Summary Table

| Stage             | Total Columns | Features (X) | Target (y) | Notes             |
| ----------------- | ------------- | ------------ | ---------- | ----------------- |
| Raw Data          | 6             | 0            | 1          | Initial load      |
| After Engineering | 39            | 38           | 1          | Full feature set  |
| After Selection   | 11            | 10           | 1          | Reduced dataset   |
| After Encoding    | 30            | 29           | 1          | Final model input |

## Features Dropped (Why?)

| Dropped Feature            | Reason                                   |
| -------------------------- | ---------------------------------------- |
| Date, HomeTeam, AwayTeam   | Not predictive                           |
| FTHG, FTAG                 | Target leakage (actual scores)           |
| HTGS, ATGS, HTGC, ATGC     | Raw cumulative stats (replaced by GD)    |
| HM4, HM5, AM4, AM5         | Too old to be relevant                   |
| MW                         | Already used for scaling                 |
| HTFormPtsStr, ATFormPtsStr | Converted to numeric HTFormPts/ATFormPts |
| HTFormPts, ATFormPts       | Replaced by DiffFormPts                  |
| Win/Loss Streaks           | Redundant with form features             |
| DiffPts                    | Highly correlated with HTP/ATP           |

**The final 29 features balance predictive power with simplicity and avoid multicollinearity.**
