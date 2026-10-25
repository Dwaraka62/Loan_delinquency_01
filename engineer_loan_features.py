"""Stage 4: deterministic feature engineering for loan delinquency data."""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("loans_structurally_cleaned.csv")
OUTPUT_PATH = Path("loans_engineered.csv")
TARGET = "target_delinquent"
# The source data's credit-line years end in 2015; 2016 is the modeling
# reference year used by the workflow, not the current calendar year.
REFERENCE_YEAR = 2016


STATE_TO_REGION = {
    **dict.fromkeys(["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"], "Northeast"),
    **dict.fromkeys(["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"], "Midwest"),
    **dict.fromkeys(["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"], "South"),
    **dict.fromkeys(["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"], "West"),
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide safely, treating zero denominators as an unavailable ratio."""
    numerator = pd.to_numeric(numerator, errors="coerce").astype(float)
    denominator = pd.to_numeric(denominator, errors="coerce").astype(float).replace(0, np.nan)
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def engineer_features(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create domain-informed features without fitting any data-dependent model."""
    features = data.copy()
    original_columns = features.columns.tolist()

    if "earliest_credit_line" in features.columns:
        features["earliest_credit_line"] = pd.to_numeric(
            features["earliest_credit_line"], errors="coerce"
        )
        features["credit_history_years"] = (
            REFERENCE_YEAR - features["earliest_credit_line"]
        ).clip(lower=0)
        features = features.drop(columns="earliest_credit_line")

    features["loan_to_income"] = safe_divide(
        features["loan_amount"], features["combined_annual_income"]
    )
    features["installment_to_income"] = safe_divide(
        features["loan_amount"] / features["term"].replace(0, np.nan),
        features["combined_annual_income"] / 12,
    )
    features["credit_utilization_ratio"] = safe_divide(
        features["total_credit_utilized"], features["total_credit_limit"]
    )
    features["open_credit_ratio"] = safe_divide(
        features["open_credit_lines"], features["total_credit_lines"]
    )
    features["cc_balance_ratio"] = safe_divide(
        features["num_cc_carrying_balance"], features["num_open_cc_accounts"]
    )
    features["open_cc_ratio"] = safe_divide(
        features["num_open_cc_accounts"], features["num_total_cc_accounts"]
    )
    features["recent_account_ratio"] = safe_divide(
        features["accounts_opened_24m"], features["total_credit_lines"]
    )
    features["accounts_per_credit_year"] = safe_divide(
        features["total_credit_lines"], features["credit_history_years"]
    )
    features["inquiries_per_credit_year"] = safe_divide(
        features["inquiries_last_12m"], features["credit_history_years"]
    )

    negative_mark_columns = [
        "delinq_2y",
        "num_collections_last_12m",
        "num_historical_failed_to_pay",
        "tax_liens",
        "public_record_bankrupt",
        "current_accounts_delinq",
        "num_accounts_30d_past_due",
    ]
    features["total_negative_marks"] = features[negative_mark_columns].fillna(0).sum(axis=1)
    features["has_delinq_history"] = (features["total_negative_marks"] > 0).astype("int64")

    features["log_combined_annual_income"] = np.log1p(
        features["combined_annual_income"].clip(lower=0)
    )
    features["log_loan_amount"] = np.log1p(features["loan_amount"].clip(lower=0))

    if "state" in features.columns:
        features["region"] = features["state"].map(STATE_TO_REGION).fillna("Unknown")

    # Keep target last for readability and downstream splitting.
    features = features[[column for column in features.columns if column != TARGET] + [TARGET]]
    engineered_columns = [column for column in features.columns if column not in original_columns]
    report = {
        "rows": len(features),
        "columns_before": len(original_columns),
        "columns_after": features.shape[1],
        "engineered_columns": engineered_columns,
        "remaining_missing_values": int(features.isna().sum().sum()),
    }
    return features, report


if __name__ == "__main__":
    source = pd.read_csv(INPUT_PATH)
    engineered, report = engineer_features(source)
    engineered.to_csv(OUTPUT_PATH, index=False)
    print("FEATURE ENGINEERING COMPLETE")
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"output: {OUTPUT_PATH.resolve()}")
