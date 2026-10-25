"""Stage 3: structural cleaning for the loan delinquency dataset.

The original CSV is never modified.  This stage performs transformations that
do not learn population statistics from the data (for example, medians).  Any
median/mode imputation belongs after the train/validation/test split.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("loans.csv")
OUTPUT_PATH = Path("loans_structurally_cleaned.csv")
TARGET = "target_delinquent"


def clean_loans(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply leakage-safe structural cleaning and return a report."""
    cleaned = data.copy()
    cleaned.columns = cleaned.columns.str.strip()
    report: dict[str, object] = {
        "rows_before": len(cleaned),
        "columns_before": cleaned.shape[1],
        "duplicates_removed": int(cleaned.duplicated().sum()),
    }

    cleaned = cleaned.drop_duplicates().copy()

    if TARGET not in cleaned.columns:
        raise ValueError(f"Required target column is missing: {TARGET}")
    cleaned[TARGET] = pd.to_numeric(cleaned[TARGET], errors="coerce")
    invalid_target = cleaned[TARGET].isna() | ~cleaned[TARGET].isin([0, 1])
    report["invalid_target_rows_removed"] = int(invalid_target.sum())
    cleaned = cleaned.loc[~invalid_target].copy()
    cleaned[TARGET] = cleaned[TARGET].astype("int64")

    # This field is either zero or missing in this dataset and carries no
    # useful variation.  Remove it rather than treating missing as a value.
    if "num_accounts_120d_past_due" in cleaned.columns:
        non_null_unique = cleaned["num_accounts_120d_past_due"].dropna().nunique()
        if non_null_unique <= 1:
            cleaned = cleaned.drop(columns="num_accounts_120d_past_due")
            report["columns_dropped"] = ["num_accounts_120d_past_due"]
        else:
            report["columns_dropped"] = []
    else:
        report["columns_dropped"] = []

    # Joint fields already represent the joint application total/rate.  Keep
    # one application-level feature and explicit indicators of its source.
    if {"annual_income", "annual_income_joint", "application_type"}.issubset(cleaned.columns):
        is_joint = cleaned["application_type"].astype("string").str.lower().eq("joint")
        use_joint = is_joint & cleaned["annual_income_joint"].notna()
        cleaned["combined_annual_income"] = np.where(
            use_joint, cleaned["annual_income_joint"], cleaned["annual_income"]
        )
        cleaned["used_joint_income"] = use_joint.astype("int64")

    if {"debt_to_income", "debt_to_income_joint", "application_type"}.issubset(cleaned.columns):
        is_joint = cleaned["application_type"].astype("string").str.lower().eq("joint")
        use_joint = is_joint & cleaned["debt_to_income_joint"].notna()
        cleaned["combined_dti"] = np.where(
            use_joint, cleaned["debt_to_income_joint"], cleaned["debt_to_income"]
        )
        cleaned["used_joint_dti"] = use_joint.astype("int64")

    joint_columns = [
        column
        for column in ("annual_income_joint", "debt_to_income_joint")
        if column in cleaned.columns
    ]
    if joint_columns:
        cleaned = cleaned.drop(columns=joint_columns)

    # A missing "months since event" means the event has not been observed.
    # Preserve that meaning with a flag and a large sentinel; this avoids
    # confusing "never" with a typical observed number of months.
    event_columns = [
        "months_since_last_delinq",
        "months_since_90d_late",
        "months_since_last_credit_inquiry",
    ]
    for column in event_columns:
        if column in cleaned.columns:
            cleaned[f"{column}_never"] = cleaned[column].isna().astype("int64")
            cleaned[column] = cleaned[column].fillna(999)

    if "emp_length" in cleaned.columns:
        cleaned["emp_length_missing"] = cleaned["emp_length"].isna().astype("int64")

    # Preserve questionable credit-limit records for investigation, but make
    # the quality issue explicit instead of silently deleting or altering them.
    if {"total_credit_utilized", "total_credit_limit"}.issubset(cleaned.columns):
        cleaned["credit_utilization_inconsistent"] = (
            cleaned["total_credit_utilized"] > cleaned["total_credit_limit"]
        ).astype("int64")

    # Keep the target last for readability.
    cleaned = cleaned[[column for column in cleaned.columns if column != TARGET] + [TARGET]]
    report["rows_after"] = len(cleaned)
    report["columns_after"] = cleaned.shape[1]
    report["remaining_missing_values"] = int(cleaned.isna().sum().sum())
    report["credit_utilization_inconsistent_rows"] = int(
        cleaned.get("credit_utilization_inconsistent", pd.Series(dtype="int64")).sum()
    )
    return cleaned, report


if __name__ == "__main__":
    source = pd.read_csv(DATA_PATH)
    cleaned_df, cleaning_report = clean_loans(source)
    cleaned_df.to_csv(OUTPUT_PATH, index=False)

    print("STRUCTURAL CLEANING COMPLETE")
    for key, value in cleaning_report.items():
        print(f"{key}: {value}")
    print(f"output: {OUTPUT_PATH.resolve()}")
