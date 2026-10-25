"""Stage 8: mixed categorical encoding.

Low-cardinality fields use one-hot encoding.  State uses smoothed, out-of-fold
target encoding on the training split, with the full training mapping applied
unchanged to validation and test.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


SPLIT_DIR = Path("data_splits")
OUTPUT_DIR = Path("mixed_encoded_data")
TARGET = "target_delinquent"
STATE_COLUMN = "state"
OHE_COLUMNS = ["homeownership", "loan_purpose", "application_type", "region"]
ENCODED_STATE_COLUMN = "state_target_encoded"
RANDOM_STATE = 42
N_FOLDS = 5
SMOOTHING = 20.0


def smoothed_mapping(categories: pd.Series, target: pd.Series, smoothing: float):
    """Return category encodings smoothed toward the supplied global rate."""
    frame = pd.DataFrame({"category": categories, "target": target})
    global_rate = float(frame["target"].mean())
    grouped = frame.groupby("category")["target"].agg(["sum", "count"])
    mapping = (grouped["sum"] + smoothing * global_rate) / (
        grouped["count"] + smoothing
    )
    return mapping, global_rate


def oof_target_encode(
    categories: pd.Series,
    target: pd.Series,
    n_folds: int,
    smoothing: float,
    random_state: int,
):
    """Create training encodings without using a row's own target."""
    encoded = pd.Series(index=categories.index, dtype="float64")
    splitter = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )
    for fit_positions, holdout_positions in splitter.split(categories, target):
        fit_categories = categories.iloc[fit_positions]
        fit_target = target.iloc[fit_positions]
        mapping, global_rate = smoothed_mapping(
            fit_categories, fit_target, smoothing
        )
        encoded.iloc[holdout_positions] = (
            categories.iloc[holdout_positions]
            .map(mapping)
            .fillna(global_rate)
            .to_numpy()
        )
    return encoded


if __name__ == "__main__":
    splits = {
        name: pd.read_csv(SPLIT_DIR / f"{name}.csv")
        for name in ["train", "validation", "test"]
    }
    if any(column not in splits["train"].columns for column in [TARGET, STATE_COLUMN]):
        raise ValueError(f"Expected columns are missing: {TARGET}, {STATE_COLUMN}")

    train = splits["train"].copy()
    y_train = train[TARGET].astype("int64")
    state_train = train[STATE_COLUMN].astype("string").fillna("__MISSING__")

    # OOF training values prevent each training row from encoding itself.
    train_state_encoding = oof_target_encode(
        state_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        n_folds=N_FOLDS,
        smoothing=SMOOTHING,
        random_state=RANDOM_STATE,
    )

    # Validation and test use a mapping fitted on all training rows only.
    full_state_mapping, train_global_rate = smoothed_mapping(
        state_train.reset_index(drop=True), y_train.reset_index(drop=True), SMOOTHING
    )
    encoded_frames = {}
    for name, frame in splits.items():
        result = frame.drop(columns=[STATE_COLUMN]).copy()
        if name == "train":
            result[ENCODED_STATE_COLUMN] = train_state_encoding.to_numpy()
        else:
            result[ENCODED_STATE_COLUMN] = (
                frame[STATE_COLUMN]
                .astype("string")
                .fillna("__MISSING__")
                .map(full_state_mapping)
                .fillna(train_global_rate)
                .to_numpy()
            )
        encoded_frames[name] = result

    # Encode only the four selected low-cardinality columns as one-hot values.
    missing_ohe = [column for column in OHE_COLUMNS if column not in train.columns]
    if missing_ohe:
        raise ValueError(f"OHE columns are missing: {missing_ohe}")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    train_ohe_values = pd.DataFrame(
        categorical_imputer.fit_transform(encoded_frames["train"][OHE_COLUMNS]),
        columns=OHE_COLUMNS,
    )
    ohe = {"train": pd.get_dummies(train_ohe_values, dummy_na=False)}
    for name in ["validation", "test"]:
        values = pd.DataFrame(
            categorical_imputer.transform(encoded_frames[name][OHE_COLUMNS]),
            columns=OHE_COLUMNS,
        )
        ohe[name] = pd.get_dummies(values, dummy_na=False).reindex(
            columns=ohe["train"].columns, fill_value=0
        )

    # Numeric preprocessing is fitted on the training split only. The target
    # encoding is treated as a continuous numeric feature and standardized.
    model_drop = [column for column in OHE_COLUMNS if column in encoded_frames["train"].columns]
    numeric_feature_source = encoded_frames["train"].drop(
        columns=model_drop + [TARGET]
    )
    numeric_columns = numeric_feature_source.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    numeric_imputer = SimpleImputer(strategy="median")
    numeric_values = {}
    for name in ["train", "validation", "test"]:
        numeric_values[name] = pd.DataFrame(
            numeric_imputer.fit_transform(encoded_frames[name][numeric_columns])
            if name == "train"
            else numeric_imputer.transform(encoded_frames[name][numeric_columns]),
            columns=numeric_columns,
        )

    binary_columns = [
        column
        for column in numeric_columns
        if set(numeric_values["train"][column].dropna().unique()).issubset({0, 1})
    ]
    continuous_columns = [
        column for column in numeric_columns if column not in binary_columns
    ]
    scaler = StandardScaler()
    if continuous_columns:
        numeric_values["train"][continuous_columns] = scaler.fit_transform(
            numeric_values["train"][continuous_columns]
        )
        for name in ["validation", "test"]:
            numeric_values[name][continuous_columns] = scaler.transform(
                numeric_values[name][continuous_columns]
            )

    final_features = {
        name: pd.concat([numeric_values[name], ohe[name]], axis=1)
        for name in ["train", "validation", "test"]
    }
    final_features["validation"] = final_features["validation"].reindex(
        columns=final_features["train"].columns, fill_value=0
    )
    final_features["test"] = final_features["test"].reindex(
        columns=final_features["train"].columns, fill_value=0
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    for name in ["train", "validation", "test"]:
        final_features[name].to_csv(OUTPUT_DIR / f"X_{name}.csv", index=False)
        splits[name][[TARGET]].to_csv(OUTPUT_DIR / f"y_{name}.csv", index=False)

    joblib.dump(
        {
            "state_mapping": full_state_mapping.to_dict(),
            "state_global_rate": train_global_rate,
            "state_smoothing": SMOOTHING,
            "state_oof_folds": N_FOLDS,
            "ohe_columns": OHE_COLUMNS,
            "ohe_feature_columns": ohe["train"].columns.tolist(),
            "numeric_imputer": numeric_imputer,
            "categorical_imputer": categorical_imputer,
            "scaler": scaler,
            "numeric_columns": numeric_columns,
            "binary_columns": binary_columns,
            "continuous_columns": continuous_columns,
            "feature_columns": final_features["train"].columns.tolist(),
        },
        OUTPUT_DIR / "mixed_encoding_bundle.joblib",
    )

    print("MIXED CATEGORICAL ENCODING COMPLETE")
    print(f"OHE columns: {OHE_COLUMNS}")
    print(
        f"state: smoothed {N_FOLDS}-fold OOF target encoding "
        f"(smoothing={SMOOTHING:g})"
    )
    print(f"training global delinquency rate: {train_global_rate:.6f}")
    print(f"final feature count: {final_features['train'].shape[1]}")
    for name, matrix in final_features.items():
        print(
            f"{name}: shape={matrix.shape}, "
            f"missing_values={int(matrix.isna().sum().sum())}"
        )
    print(f"output: {OUTPUT_DIR.resolve()}")
