"""Stage 5: split, impute, encode, and scale model features without leakage."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("loans_engineered.csv")
OUTPUT_DIR = Path("model_data")
TARGET = "target_delinquent"
RANDOM_STATE = 42


def is_binary_like(values: pd.Series) -> bool:
    return set(pd.Series(values).dropna().unique()).issubset({0, 1})


def prepare_data(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    if TARGET not in data.columns:
        raise ValueError(f"Required target column is missing: {TARGET}")

    # Split before fitting any imputer, scaler, or category vocabulary.
    train, temporary = train_test_split(
        data, test_size=0.30, stratify=data[TARGET], random_state=RANDOM_STATE
    )
    validation, test = train_test_split(
        temporary, test_size=2 / 3, stratify=temporary[TARGET], random_state=RANDOM_STATE
    )
    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

    # State and region contain overlapping geographic information. Keep the
    # engineered, lower-cardinality region in the model and retain state in
    # the source engineered CSV for traceability.
    model_drop = [column for column in ["state"] if column in data.columns and "region" in data.columns]
    model_splits = {
        name: frame.drop(columns=model_drop).copy() for name, frame in splits.items()
    }
    feature_columns = [column for column in model_splits["train"].columns if column != TARGET]
    numeric_columns = model_splits["train"][feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    # Fit imputers on training data only.
    numeric_imputer = SimpleImputer(strategy="median")
    train_numeric = pd.DataFrame(
        numeric_imputer.fit_transform(model_splits["train"][numeric_columns]),
        columns=numeric_columns,
    )
    transformed_numeric = {
        "train": train_numeric,
        "validation": pd.DataFrame(
            numeric_imputer.transform(model_splits["validation"][numeric_columns]),
            columns=numeric_columns,
        ),
        "test": pd.DataFrame(
            numeric_imputer.transform(model_splits["test"][numeric_columns]),
            columns=numeric_columns,
        ),
    }

    categorical_imputer = SimpleImputer(strategy="most_frequent")
    transformed_categorical: dict[str, pd.DataFrame] = {}
    if categorical_columns:
        train_categorical = pd.DataFrame(
            categorical_imputer.fit_transform(model_splits["train"][categorical_columns]),
            columns=categorical_columns,
        )
        transformed_categorical["train"] = train_categorical
        for name in ["validation", "test"]:
            transformed_categorical[name] = pd.DataFrame(
                categorical_imputer.transform(model_splits[name][categorical_columns]),
                columns=categorical_columns,
            )
    else:
        categorical_imputer = None
        transformed_categorical = {
            name: pd.DataFrame(index=transformed_numeric[name].index)
            for name in transformed_numeric
        }

    # Learn training categories only, then align validation/test columns.
    train_encoded = pd.get_dummies(transformed_categorical["train"], dummy_na=False)
    encoded = {"train": train_encoded}
    for name in ["validation", "test"]:
        encoded[name] = pd.get_dummies(
            transformed_categorical[name], dummy_na=False
        ).reindex(columns=train_encoded.columns, fill_value=0)

    # Scale continuous numeric columns only. Binary indicators and one-hot
    # columns remain 0/1, which is appropriate for interpretation and trees.
    binary_columns = [
        column for column in numeric_columns if is_binary_like(transformed_numeric["train"][column])
    ]
    continuous_columns = [column for column in numeric_columns if column not in binary_columns]
    scaler = StandardScaler()
    scaled_numeric = {name: frame.copy() for name, frame in transformed_numeric.items()}
    if continuous_columns:
        scaled_numeric["train"][continuous_columns] = scaler.fit_transform(
            transformed_numeric["train"][continuous_columns]
        )
        for name in ["validation", "test"]:
            scaled_numeric[name][continuous_columns] = scaler.transform(
                transformed_numeric[name][continuous_columns]
            )

    matrices = {
        name: pd.concat([scaled_numeric[name], encoded[name]], axis=1)
        for name in ["train", "validation", "test"]
    }
    matrices["validation"] = matrices["validation"].reindex(columns=matrices["train"].columns, fill_value=0)
    matrices["test"] = matrices["test"].reindex(columns=matrices["train"].columns, fill_value=0)

    labels = {name: frame[TARGET].astype("int64") for name, frame in model_splits.items()}
    report = {
        "rows": {name: len(frame) for name, frame in splits.items()},
        "positive_counts": {name: int(labels[name].sum()) for name in labels},
        "model_columns_dropped": model_drop,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "binary_columns": binary_columns,
        "continuous_columns": continuous_columns,
        "encoded_columns": encoded["train"].columns.tolist(),
        "final_feature_count": matrices["train"].shape[1],
    }
    artifacts = {
        "numeric_imputer": numeric_imputer,
        "categorical_imputer": categorical_imputer,
        "scaler": scaler,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "binary_columns": binary_columns,
        "continuous_columns": continuous_columns,
        "feature_columns": matrices["train"].columns.tolist(),
        "dropped_columns": model_drop,
        "random_state": RANDOM_STATE,
    }
    return {"matrices": matrices, "labels": labels, "artifacts": artifacts, "report": report}, report


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    source = pd.read_csv(INPUT_PATH)
    prepared, report = prepare_data(source)

    for name, matrix in prepared["matrices"].items():
        matrix.to_csv(OUTPUT_DIR / f"X_{name}_scaled.csv", index=False)
        prepared["labels"][name].to_csv(OUTPUT_DIR / f"y_{name}.csv", index=False, header=[TARGET])
    joblib.dump(prepared["artifacts"], OUTPUT_DIR / "preprocessing_bundle.joblib")

    print("LEAKAGE-SAFE FEATURE PREPARATION COMPLETE")
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"output: {OUTPUT_DIR.resolve()}")
