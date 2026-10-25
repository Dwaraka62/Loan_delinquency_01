# Loan Delinquency ML Workflow — Corrected Trail

This is the corrected, deduplicated version of `Codex_Trail_03.md`.

Corrections made:

- Removed repeated nested transcript blocks, token reports, and tool metadata.
- Removed garbled character encoding and corrected spelling/formatting.
- Kept one canonical prompt, action, and result for each stage.
- Used the latest results after XGBoost, LightGBM, CatBoost, threshold tuning, and the final Platt rerun were completed.
- Distinguished validation results from final test results.

The original `Codex_Trail_03.md` is preserved unchanged.

## Stage 1 — Dataset overview

`loans.csv` contains 10,000 loan applications and 40 columns. The target is `target_delinquent`:

- 9,822 non-delinquent applications
- 178 delinquent applications
- Positive rate: 1.78%

The data contains applicant information, income and debt measures, credit history, collections and delinquency history, loan details, categorical variables, and the target outcome.

## Stage 2 — Schema validation

The dataset passed structural validation:

- 10,000 rows and 40 columns
- No duplicate or blank column names
- Every CSV row has 40 fields
- Target exists, is non-null, and contains only `0` and `1`
- No negative numeric values
- Valid categorical values and loan terms of 36 or 60 months
- Valid `account_never_delinq_percent` range of 0–100
- No account-count consistency violations

Data-quality warning: 154 rows have `total_credit_utilized` greater than `total_credit_limit`.

## Data findings and assumptions

Key findings:

- The target is severely imbalanced.
- Joint-applicant fields are missing for most individual applications.
- Several event-history fields are missing and may represent events that never occurred.
- Credit-history and payment-behavior fields are useful risk indicators.
- Accuracy alone is not suitable; PR-AUC, precision, recall, and F1 are more useful.

Main assumptions:

- Each row represents one loan application.
- The target outcome occurs after application and is suitable for supervised learning.
- Historical features were available at application time.
- Applications are treated as independent because no borrower ID is available.
- The data is representative of the intended use population.
- Missing event-history values may mean that the event was not observed; this requires domain confirmation.

## Stage 3 — Data cleaning

Created [clean_loans.py](./clean_loans.py) and [loans_structurally_cleaned.csv](./loans_structurally_cleaned.csv).

Cleaning actions:

- Removed exact duplicates; none were found.
- Validated and standardized `target_delinquent`.
- Removed near-constant `num_accounts_120d_past_due`.
- Created `combined_annual_income` and `combined_dti`.
- Added indicators showing when joint values were used.
- Added `*_never` event flags and used `999` as the event-time sentinel.
- Added `emp_length_missing`.
- Added `credit_utilization_inconsistent` for the 154 questionable rows.
- Preserved the original `loans.csv`.

Remaining missing values were intentionally left for train-only imputation after splitting.

## Stage 4 — Feature engineering

Created [engineer_loan_features.py](./engineer_loan_features.py) and [loans_engineered.csv](./loans_engineered.csv).

Added affordability, utilization, credit-history, account-activity, delinquency-burden, log-transformed, and regional features, including:

`credit_history_years`, `loan_to_income`, `installment_to_income`, `credit_utilization_ratio`, `open_credit_ratio`, `cc_balance_ratio`, `open_cc_ratio`, `recent_account_ratio`, `accounts_per_credit_year`, `inquiries_per_credit_year`, `total_negative_marks`, `has_delinq_history`, `log_combined_annual_income`, `log_loan_amount`, and `region`.

### Feature-engineering bug resolved

A diagnostic `debit_utilization_ratio` had an implausible maximum of 211,005. It incorrectly divided the credit measure `total_credit_utilized` by the unrelated `total_debit_limit`. The feature was rejected. The valid feature is:

```text
credit_utilization_ratio = total_credit_utilized / total_credit_limit
```

## Stage 5 — Scaling correction

The request to scale all features was corrected. The safe order is:

1. Split the data.
2. Fit imputers and encoders on training data only.
3. Scale continuous numeric features.
4. Leave binary and one-hot features as `0/1`.

## Stage 6 — Stratified split

Created [split_loan_data.py](./split_loan_data.py) and the files in `data_splits/`.

| Split | Rows | Delinquent | Rate |
|---|---:|---:|---:|
| Train | 7,000 | 125 | 1.786% |
| Validation | 1,000 | 18 | 1.800% |
| Test | 2,000 | 35 | 1.750% |

The split is stratified, and no overlap was detected.

## Stage 7 — Feature normalization

Created [normalize_loan_features.py](./normalize_loan_features.py).

- 44 continuous numeric features were standardized.
- 10 binary features remained `0/1`.
- 21 one-hot features were created.
- Imputation and scaling were fitted on training data only.
- All three matrices contain 75 features and zero missing values.

## Stage 8 — Mixed categorical encoding

Created [encode_mixed_features.py](./encode_mixed_features.py).

One-hot encoding was used for `homeownership`, `loan_purpose`, `application_type`, and `region`. Raw `state` was replaced with smoothed five-fold out-of-fold target encoding.

Target encoding means replacing a category with its delinquency rate. OOF encoding prevents a training row's own target from determining its encoded value. Validation and test mappings are fitted from training data only.

The resulting matrices contain 76 features and zero missing values.

## Stage 9 — Baseline models

Created [train_baseline_models.py](./train_baseline_models.py).

The training class ratio produced `scale_pos_weight = 55.0`.

Initial validation results at threshold 0.5:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5388 | 0.0268 | 0.0248 | 0.4444 | 0.0471 |
| Original XGBoost | 0.5777 | 0.0241 | 0.0000 | 0.0000 | 0.0000 |

The test set was not used.

## Stage 10 — XGBoost tuning

Created [tune_xgboost.py](./tune_xgboost.py). Tuning used 12 randomized configurations, three-fold stratified cross-validation, and PR-AUC scoring.

Best parameters included:

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

Tuned validation ROC-AUC was 0.5821 and PR-AUC was 0.0278 at threshold 0.5.

## Stage 11 — Importance, calibration, model comparison, and testing

### Feature importance and pruning

Nine features had zero gain, but removing them did not improve validation PR-AUC enough. All 76 features were retained.

### LightGBM and CatBoost comparison

Both packages were later installed and trained successfully.

| Model | Validation ROC-AUC | Validation PR-AUC |
|---|---:|---:|
| LightGBM | **0.6066** | 0.0261 |
| Tuned XGBoost | 0.5821 | **0.0278** |
| CatBoost | 0.5800 | 0.0230 |
| Logistic Regression | 0.5388 | 0.0268 |

LightGBM had the strongest validation ROC-AUC; tuned XGBoost had the strongest validation PR-AUC.

### Threshold tuning

Thresholds were selected on validation data by maximizing F1:

| Model | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Tuned XGBoost | 0.4069 | 0.0656 | 0.2222 | **0.1013** |
| LightGBM | 0.0042 | 0.0402 | 0.5000 | 0.0744 |
| Logistic Regression | 0.8982 | 0.0769 | 0.0556 | 0.0645 |
| CatBoost | 0.0794 | 0.0314 | **0.6667** | 0.0600 |

Tuned XGBoost was selected for the best validation F1.

### Production calibration

Platt scaling was fitted to tuned XGBoost validation predictions. It calibrates probability values; it does not retrain the XGBoost model or improve ranking metrics. The final calibrated threshold was `0.020237`.

### Final production test evaluation

Created [final_production_platt_evaluation.py](./final_production_platt_evaluation.py) and [production model bundle](./production_platt_results/production_model_bundle.joblib).

| Metric | Validation | Test |
|---|---:|---:|
| ROC-AUC | 0.5821 | 0.5395 |
| PR-AUC | 0.0278 | 0.0260 |
| Brier score | 0.0177 | **0.0172** |
| Precision | 0.0656 | 0.0467 |
| Recall | 0.2222 | 0.1429 |
| F1 | 0.1013 | 0.0704 |

On the test set, the production model detected 5 of 35 delinquent loans, with 102 false positives. Test labels were not used for calibration or threshold selection.

## Final interpretation

The tuned XGBoost plus Platt calibration pipeline is reproducible and produces better-calibrated probabilities, but predictive performance is modest. It should be treated as a risk-ranking aid rather than a high-confidence classifier. Future improvements should focus on more positive examples, stronger temporal or grouped validation if identifiers become available, business-cost-based threshold selection, and subgroup/drift monitoring.
