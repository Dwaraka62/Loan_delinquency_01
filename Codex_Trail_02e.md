# Loan Delinquency Prediction Workflow — Cleaned Trail

This file is a deduplicated record of the unique prompts, decisions, actions, explanations, and results from `Codex_Trail_02.md`. The original file is retained unchanged.

## Stage 1 — Dataset overview

### Prompt

Briefly show `loans.csv`, its columns, and their relevance.

### Result

- Dataset: 10,000 loan applications and 40 columns.
- Target: `target_delinquent`.
- Class distribution: 9,822 non-delinquent and 178 delinquent loans (1.78% positive).
- Data includes numeric and categorical variables with missing values.

Important column groups:

- Applicant profile: `emp_length`, `state`, `homeownership`, `annual_income`, `debt_to_income`.
- Joint-application fields: `annual_income_joint`, `debt_to_income_joint`, `application_type`.
- Credit history: `earliest_credit_line`, `total_credit_lines`, `open_credit_lines`, `total_credit_limit`.
- Credit usage: `total_credit_utilized`, `num_total_cc_accounts`, `num_open_cc_accounts`, `num_cc_carrying_balance`.
- Delinquency and collections: `delinq_2y`, `months_since_last_delinq`, `months_since_90d_late`, `num_collections_last_12m`, `num_historical_failed_to_pay`, `current_accounts_delinq`.
- Account activity: `accounts_opened_24m`, `inquiries_last_12m`, `months_since_last_credit_inquiry`.
- Public records: `tax_liens`, `public_record_bankrupt`.
- Loan details: `loan_purpose`, `loan_amount`, `term`.
- Target: `target_delinquent` indicates whether the loan became delinquent.

## Stage 2 — Setting the stage and schema validation

### Prompt

Build the loan delinquency prediction pipeline in definite steps, beginning with schema validation.

### Validation actions

Checked column structure, data types, missingness, categorical values, numeric ranges, target labels, row widths, duplicates, and cross-field consistency.

### Result

- Structural schema passed: 10,000 rows, 40 columns, no duplicate or blank column names.
- Every CSV row contains 40 fields.
- `target_delinquent` exists, has no missing values, and contains only `0` and `1`.
- No negative numeric values were detected.
- `term` contains 36- and 60-month values.
- `account_never_delinq_percent` is within 0-100.
- Open-account counts do not exceed total-account counts.
- Joint income fields are present for joint applications and absent for individual applications.

Missing values were found in joint fields, event-history fields, `emp_length`, `debt_to_income`, and `num_accounts_120d_past_due`.

Warning: 154 records have `total_credit_utilized` greater than `total_credit_limit`. These were preserved and flagged for later review.

## Data findings and assumptions

### Key findings

- Delinquency is highly imbalanced: only 178 of 10,000 applications are positive.
- Missing joint fields are expected for individual applications.
- Credit history and payment behavior provide useful risk signals.
- Categorical values are consistent.
- The credit-utilization inconsistency requires a quality flag or further investigation.
- Accuracy alone is not appropriate; PR-AUC, recall, precision, and F1 are more informative.

### Assumptions

- Each row represents one loan application.
- `target_delinquent` records an outcome occurring after application.
- Historical features were available at application time and do not contain post-outcome leakage.
- Missing event-time values generally mean that the event was not observed; this assumption must be confirmed.
- Numeric units and category definitions are consistent.
- Rows are treated as independent because no borrower ID is available.
- The dataset is representative of the population where the model will be used.
- No date field is available for time-based validation.

## Stage 3 — Data cleaning

### Prompt

Clean the dataset.

### Actions

Created:

- [clean_loans.py](./clean_loans.py)
- [loans_structurally_cleaned.csv](./loans_structurally_cleaned.csv)

Cleaning performed:

- Removed exact duplicates; none were found.
- Validated and standardized the binary target.
- Removed `num_accounts_120d_past_due` because it was effectively constant: zero when present and missing otherwise.
- Created `combined_annual_income` and `combined_dti`.
- Added indicators showing when joint values were used.
- Added `*_never` flags for missing event-history fields and filled those fields with sentinel `999`.
- Added `emp_length_missing`.
- Added `credit_utilization_inconsistent` for the 154 questionable credit-limit records.
- Preserved the original `loans.csv`.

The cleaned file has 10,000 rows and 46 columns. Remaining missing values are limited to `emp_length` and `debt_to_income`; imputation was deliberately deferred until after splitting to prevent leakage.

## Stage 4 — Feature engineering

### Prompt

Perform feature engineering.

### Actions

Created:

- [engineer_loan_features.py](./engineer_loan_features.py)
- [loans_engineered.csv](./loans_engineered.csv)

Added features:

- `credit_history_years`
- `loan_to_income`
- `installment_to_income`
- `credit_utilization_ratio`
- `open_credit_ratio`
- `cc_balance_ratio`
- `open_cc_ratio`
- `recent_account_ratio`
- `accounts_per_credit_year`
- `inquiries_per_credit_year`
- `total_negative_marks`
- `has_delinq_history`
- `log_combined_annual_income`
- `log_loan_amount`
- `region`

The engineered dataset has 10,000 rows and 60 columns. No train-fitted statistics were introduced.

### Feature-engineering bug investigation

The diagnostic output showed an invalid `debit_utilization_ratio`:

- Mean: 409
- Standard deviation: 5,153
- Maximum: 211,005

The bug divided the credit measure `total_credit_utilized` by the unrelated debit measure `total_debit_limit`. Very small debit limits produced absurd ratios. The feature was rejected. The valid feature uses matching quantities:

```text
credit_utilization_ratio = total_credit_utilized / total_credit_limit
```

## Stage 5 — Scaling/normalization correction

### Prompt

Scale or normalize all features.

### Model correction

Scaling every column immediately would be incorrect. The correct order is:

1. Split the data.
2. Fit imputers and encoders on training data only.
3. Scale continuous numeric features.
4. Leave binary and one-hot features as `0/1`.

This avoids leakage and preserves the meaning of indicators and categorical dummies.

## Stage 6 — Stratified train/validation/test split

### Prompt

Create a 70%/10%/20% stratified train/validation/test split based on `target_delinquent` before scaling.

### Actions

Created:

- [split_loan_data.py](./split_loan_data.py)
- [train split](./data_splits/train.csv)
- [validation split](./data_splits/validation.csv)
- [test split](./data_splits/test.csv)

| Split | Rows | Delinquent | Rate |
|---|---:|---:|---:|
| Train | 7,000 | 125 | 1.786% |
| Validation | 1,000 | 18 | 1.800% |
| Test | 2,000 | 35 | 1.750% |

No overlapping records were detected between the splits.

## Stage 7 — Feature normalization

### Prompt

Normalize features across train, validation, and test.

### Actions

Created:

- [normalize_loan_features.py](./normalize_loan_features.py)
- `normalized_data/`

Processing:

- Fit imputers and the scaler on training data only.
- Applied the fitted transformations to all three splits.
- Standardized 44 continuous numerical features.
- Preserved 10 binary features as `0/1`.
- Created 21 one-hot categorical features.

Each split contains 7,000/1,000/2,000 rows, 75 features, and zero missing values.

### Final normalization sanity check

- Identical feature columns and ordering across all splits.
- No missing or infinite values.
- Continuous training features have mean approximately zero and standard deviation one.
- Binary features contain only zero and one.
- Target values remain valid and stratified.
- Test data was not used to fit transformations.

## Stage 8 — Categorical encoding

### Prompt

Use a mixed strategy: one-hot encoding for four low-cardinality fields and smoothed out-of-fold target encoding for `state`.

### Target-encoding explanation

Target encoding replaces a category with its observed target rate. For example, if 22 California applications are delinquent out of 500 California applications, the raw rate is `22/500 = 0.044`. This means 4.4% of California loan applications in the fitting data were delinquent; it does not necessarily mean 22 unique customers.

OOF means **out-of-fold**. Training rows receive category rates computed from other folds, so their own targets do not determine their encodings. Validation and test rows use a mapping fitted on the full training split only.

### Encoding decision

Use one-hot encoding for:

- `homeownership`
- `loan_purpose`
- `application_type`
- `region`

Use smoothed five-fold OOF target encoding for `state`, with smoothing strength 20 toward the training global delinquency rate.

### Actions

Created:

- [encode_mixed_features.py](./encode_mixed_features.py)
- `mixed_encoded_data/`

The resulting matrices contain 76 features, zero missing values, and no validation/test target leakage during encoding.

## Stage 9 — Baseline models

### Prompt

Train a gradient-boosted baseline, such as XGBoost or LightGBM with `scale_pos_weight`, and a logistic regression baseline.

### Initial environment issue

XGBoost and LightGBM were initially unavailable. Scikit-learn tree components were also blocked by a local DLL application-control policy. Logistic regression was trained while the issue was recorded.

### XGBoost installation and rerun

XGBoost was subsequently installed and the baselines were rerun. The training positive/negative ratio produced `scale_pos_weight = 55.0`.

Validation results at threshold 0.5:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5388 | 0.0268 | 0.0248 | 0.4444 | 0.0471 |
| Original XGBoost | 0.5777 | 0.0241 | 0.0000 | 0.0000 | 0.0000 |

The test split remained untouched. XGBoost ranked better by ROC-AUC but predicted no positives at the default threshold.

## Stage 10 — XGBoost hyperparameter tuning

### Prompt

Tune XGBoost hyperparameters first.

### Actions

Created:

- [tune_xgboost.py](./tune_xgboost.py)
- `tuned_models/xgboost_tuned.joblib`
- `tuned_models/xgboost_best_params.json`
- `tuned_models/xgboost_cv_results.csv`

Used 12 randomized configurations, three-fold stratified cross-validation, and PR-AUC as the tuning metric. The best parameters were:

| Parameter | Value |
|---|---:|
| `n_estimators` | 350 |
| `max_depth` | 4 |
| `learning_rate` | 0.03 |
| `min_child_weight` | 5 |
| `subsample` | 0.7 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 1.0 |
| `reg_lambda` | 10.0 |
| `gamma` | 0.0 |

Validation result at threshold 0.5: ROC-AUC 0.5821 and PR-AUC 0.0278. F1 remained zero at the default threshold.

## Stage 11 — Importance, calibration, test evaluation, and model comparison

### Feature importance and pruning

Created [stage11_evaluate_models.py](./stage11_evaluate_models.py). Nine features had zero gain, but removing them did not improve validation PR-AUC enough, so all 76 features were retained.

Top importance signals included `homeownership_MORTGAGE`, `term`, `months_since_last_credit_inquiry`, `loan_purpose_credit_card`, `total_collection_amount_ever`, `log_loan_amount`, and `combined_dti`.

### Calibration explanation

- Isotonic calibration is flexible and monotonic but data-hungry; it can overfit when there are only 18 positive validation examples.
- Sigmoid/Platt calibration fits a two-parameter logistic curve. It is less flexible but usually lower variance and more stable with small calibration samples.
- Brier score measures probability quality; zero is perfect and lower is better.

### LightGBM and CatBoost comparison

After installation, both models were trained on the training split and evaluated on validation data. The test set was not used.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Tuned XGBoost | 0.5821 | **0.0278** | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.5388 | 0.0268 | 0.0248 | **0.4444** | **0.0471** |
| LightGBM | **0.6066** | 0.0261 | 0.0000 | 0.0000 | 0.0000 |
| Original XGBoost | 0.5777 | 0.0241 | 0.0000 | 0.0000 | 0.0000 |
| CatBoost | 0.5800 | 0.0230 | 0.0000 | 0.0000 | 0.0000 |

LightGBM had the best ROC-AUC, while tuned XGBoost had the best PR-AUC.

### Threshold tuning

Created [tune_model_thresholds.py](./tune_model_thresholds.py). Thresholds were selected on validation data by maximizing F1.

| Model | Tuned threshold | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Tuned XGBoost | 0.4069 | **0.0278** | 0.0656 | 0.2222 | **0.1013** |
| LightGBM | 0.0042 | 0.0261 | 0.0402 | 0.5000 | 0.0744 |
| Logistic Regression | 0.8982 | 0.0268 | 0.0769 | 0.0556 | 0.0645 |
| CatBoost | 0.0794 | 0.0230 | 0.0314 | **0.6667** | 0.0600 |
| Original XGBoost | 0.1529 | 0.0241 | 0.0298 | 0.2778 | 0.0538 |

Tuned XGBoost had the best validation F1. CatBoost had the highest validation recall but many false positives.

### Final raw tuned-XGBoost test evaluation

Created [final_test_evaluation.py](./final_test_evaluation.py). The validation-derived threshold `0.4069` was held fixed for the test set.

| Metric | Test result |
|---|---:|
| ROC-AUC | 0.5395 |
| PR-AUC | 0.0260 |
| Brier score | 0.0484 |
| Precision | 0.0467 |
| Recall | 0.1429 |
| F1 | 0.0704 |
| True positives | 5 |
| False positives | 102 |
| False negatives | 30 |
| True negatives | 1,863 |

The model detected 5 of 35 delinquent test loans. F1 declined from 0.1013 on validation to 0.0704 on test, indicating limited generalization.

### Production model — tuned XGBoost plus Platt scaling

Created [final_production_platt_evaluation.py](./final_production_platt_evaluation.py) and [production model bundle](./production_platt_results/production_model_bundle.joblib).

Platt scaling was fitted on validation predictions only. Its threshold was selected on validation and then held fixed for the test set.

| Split | ROC-AUC | PR-AUC | Brier score | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.5821 | 0.0278 | 0.0177 | 0.0656 | 0.2222 | 0.1013 |
| **Test** | **0.5395** | **0.0260** | **0.0172** | **0.0467** | **0.1429** | **0.0704** |

The calibrated production threshold was `0.020237`. Platt scaling substantially improved probability quality, as shown by the lower Brier score, while ROC-AUC and PR-AUC remained unchanged because the monotonic calibration does not alter ranking.

## Overall conclusion

The final production candidate is tuned XGBoost with Platt calibration. It provides better-calibrated probabilities, but predictive performance is modest: it detects 5 of 35 delinquent test loans and produces 102 false positives. The model should be treated as a risk-ranking aid rather than a high-confidence delinquency classifier. Further work should focus on more data, stronger validation design, business-cost-based threshold selection, and monitoring for drift and subgroup performance.
