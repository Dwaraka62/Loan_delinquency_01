"""Stage 7: normalize model features across the fixed data splits.

All learned preprocessing parameters come from the training split only.  The
same fitted objects are then applied unchanged to validation and test data.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


SPLIT_DIR = Path("data_splits")
OUTPUT_DIR = Path("normalized_data")
TARGET = "target_delinquent"


def binary_like(series: pd.Series) -> bool:
    return set(series.dropna().unique()).issubset({0, 1})


if __name__ == "__main__":
    splits = {
        name: pd.read_csv(SPLIT_DIR / f"{name}.csv")
        for name in ["train", "validation", "test"]
    }

    if any(TARGET not in frame.columns for frame in splits.values()):
        raise ValueError(f"Every split must contain {TARGET}")

    # Avoid duplicating the raw state and engineered region representation.
    drop_columns = [
        column
        for column in ["state"]
        if column in splits["train"].columns and "region" in splits["train"].columns
    ]
    model_splits = {
        name: frame.drop(columns=drop_columns).copy()
        for name, frame in splits.items()
    }
    feature_columns = [column for column in model_splits["train"].columns if column != TARGET]
    numeric_columns = model_splits["train"][feature_columns].select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]

    numeric_imputer = SimpleImputer(strategy="median")
    numeric_values = {
        "train": pd.DataFrame(
            numeric_imputer.fit_transform(model_splits["train"][numeric_columns]),
            columns=numeric_columns,
        )
    }
    for name in ["validation", "test"]:
        numeric_values[name] = pd.DataFrame(
            numeric_imputer.transform(model_splits[name][numeric_columns]),
            columns=numeric_columns,
        )

    categorical_values = {}
    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        categorical_values["train"] = pd.DataFrame(
            categorical_imputer.fit_transform(
                model_splits["train"][categorical_columns]
            ),
            columns=categorical_columns,
        )
        for name in ["validation", "test"]:
            categorical_values[name] = pd.DataFrame(
                categorical_imputer.transform(
                    model_splits[name][categorical_columns]
                ),
                columns=categorical_columns,
            )
    else:
        categorical_imputer = None
        categorical_values = {
            name: pd.DataFrame(index=numeric_values[name].index)
            for name in numeric_values
        }

    train_encoded = pd.get_dummies(categorical_values["train"], dummy_na=False)
    encoded = {"train": train_encoded}
    for name in ["validation", "test"]:
        encoded[name] = pd.get_dummies(
            categorical_values[name], dummy_na=False
        ).reindex(columns=train_encoded.columns, fill_value=0)

    binary_columns = [
        column
        for column in numeric_columns
        if binary_like(numeric_values["train"][column])
    ]
    continuous_columns = [
        column for column in numeric_columns if column not in binary_columns
    ]

    scaler = StandardScaler()
    normalized_numeric = {name: frame.copy() for name, frame in numeric_values.items()}
    if continuous_columns:
        normalized_numeric["train"][continuous_columns] = scaler.fit_transform(
            numeric_values["train"][continuous_columns]
        )
        for name in ["validation", "test"]:
            normalized_numeric[name][continuous_columns] = scaler.transform(
                numeric_values[name][continuous_columns]
            )

    normalized = {
        name: pd.concat([normalized_numeric[name], encoded[name]], axis=1)
        for name in ["train", "validation", "test"]
    }
    normalized["validation"] = normalized["validation"].reindex(
        columns=normalized["train"].columns, fill_value=0
    )
    normalized["test"] = normalized["test"].reindex(
        columns=normalized["train"].columns, fill_value=0
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, matrix in normalized.items():
        matrix.to_csv(OUTPUT_DIR / f"X_{name}.csv", index=False)
        splits[name][[TARGET]].to_csv(
            OUTPUT_DIR / f"y_{name}.csv", index=False
        )

    bundle = {
        "numeric_imputer": numeric_imputer,
        "categorical_imputer": categorical_imputer,
        "scaler": scaler,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "binary_columns": binary_columns,
        "continuous_columns": continuous_columns,
        "feature_columns": normalized["train"].columns.tolist(),
        "dropped_columns": drop_columns,
    }
    joblib.dump(bundle, OUTPUT_DIR / "normalization_bundle.joblib")

    print("FEATURE NORMALIZATION COMPLETE")
    print(f"input splits: {SPLIT_DIR.resolve()}")
    print(f"dropped from model matrix: {drop_columns or 'None'}")
    print(f"continuous features normalized: {len(continuous_columns)}")
    print(f"binary features retained as 0/1: {len(binary_columns)}")
    print(f"one-hot features: {len(train_encoded.columns)}")
    for name, matrix in normalized.items():
        print(
            f"{name}: shape={matrix.shape}, "
            f"missing_values={int(matrix.isna().sum().sum())}"
        )
    print(f"output: {OUTPUT_DIR.resolve()}")
