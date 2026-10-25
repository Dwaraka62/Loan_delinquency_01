"""Stage 11: feature pruning, calibration, and final test evaluation."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATA_DIR = Path("mixed_encoded_data")
TUNED_DIR = Path("tuned_models")
OUTPUT_DIR = Path("stage11_results")
TARGET = "target_delinquent"


def metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_f1_threshold(y_true, probabilities):
    candidates = np.unique(
        np.clip(np.quantile(probabilities, np.linspace(0.01, 0.99, 199)), 0.001, 0.999)
    )
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in candidates
    ]
    best_index = int(np.argmax(scores))
    return float(candidates[best_index]), float(scores[best_index])


if __name__ == "__main__":
    X = {
        name: pd.read_csv(DATA_DIR / f"X_{name}.csv")
        for name in ["train", "validation", "test"]
    }
    y = {
        name: pd.read_csv(DATA_DIR / f"y_{name}.csv")[TARGET].astype("int64")
        for name in ["train", "validation", "test"]
    }
    tuned_model = joblib.load(TUNED_DIR / "xgboost_tuned.joblib")

    # Importance is learned from the fitted training model. Zero-gain columns
    # are tested as a controlled pruning experiment on validation data.
    importance = pd.DataFrame(
        {"feature": X["train"].columns, "importance": tuned_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    zero_gain_features = importance.loc[importance["importance"] <= 0, "feature"].tolist()
    full_model = clone(tuned_model)
    full_model.fit(X["train"], y["train"])
    full_validation_probabilities = full_model.predict_proba(X["validation"])[:, 1]
    full_validation_metrics = metrics(y["validation"], full_validation_probabilities)

    selected_features = X["train"].columns.tolist()
    selected_model = full_model
    pruning_used = False
    pruned_validation_metrics = None
    if zero_gain_features and len(zero_gain_features) < len(selected_features):
        candidate_features = [
            feature for feature in selected_features if feature not in zero_gain_features
        ]
        pruned_model = clone(tuned_model)
        pruned_model.fit(X["train"][candidate_features], y["train"])
        pruned_probabilities = pruned_model.predict_proba(
            X["validation"][candidate_features]
        )[:, 1]
        pruned_validation_metrics = metrics(y["validation"], pruned_probabilities)
        if pruned_validation_metrics["pr_auc"] >= full_validation_metrics["pr_auc"] + 0.001:
            selected_features = candidate_features
            selected_model = pruned_model
            pruning_used = True

    selected_validation_probabilities = selected_model.predict_proba(
        X["validation"][selected_features]
    )[:, 1]
    selected_test_probabilities = selected_model.predict_proba(
        X["test"][selected_features]
    )[:, 1]

    # Fit calibration only on validation predictions, then apply it to test.
    # Test labels are not used to choose either calibrator or threshold.
    sigmoid = LogisticRegression(solver="lbfgs", max_iter=1000)
    sigmoid.fit(selected_validation_probabilities.reshape(-1, 1), y["validation"])
    sigmoid_validation = sigmoid.predict_proba(
        selected_validation_probabilities.reshape(-1, 1)
    )[:, 1]
    sigmoid_test = sigmoid.predict_proba(
        selected_test_probabilities.reshape(-1, 1)
    )[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(selected_validation_probabilities, y["validation"])
    isotonic_validation = isotonic.predict(selected_validation_probabilities)
    isotonic_test = isotonic.predict(selected_test_probabilities)

    calibration_candidates = {
        "uncalibrated": (selected_validation_probabilities, selected_test_probabilities),
        "sigmoid_platt": (sigmoid_validation, sigmoid_test),
        "isotonic": (isotonic_validation, isotonic_test),
    }
    calibration_validation_rows = []
    thresholds = {}
    for name, (validation_probabilities, _) in calibration_candidates.items():
        threshold, _ = best_f1_threshold(y["validation"], validation_probabilities)
        thresholds[name] = threshold
        row = metrics(y["validation"], validation_probabilities, threshold)
        row["calibration"] = name
        calibration_validation_rows.append(row)
    calibration_validation = pd.DataFrame(calibration_validation_rows).sort_values(
        ["brier", "pr_auc"], ascending=[True, False]
    )
    chosen_calibration = str(calibration_validation.iloc[0]["calibration"])
    chosen_threshold = thresholds[chosen_calibration]

    test_rows = []
    for name, (_, test_probabilities) in calibration_candidates.items():
        test_rows.append(
            {
                "calibration": name,
                **metrics(y["test"], test_probabilities, thresholds[name]),
            }
        )
    test_results = pd.DataFrame(test_rows)

    # Existing optional libraries are checked for transparent reporting.
    optional_models = {}
    for package in ["lightgbm", "catboost"]:
        try:
            __import__(package)
            optional_models[package] = "available but not trained in this run"
        except Exception as exc:
            optional_models[package] = f"unavailable: {type(exc).__name__}"

    OUTPUT_DIR.mkdir(exist_ok=True)
    importance.to_csv(OUTPUT_DIR / "xgboost_feature_importance.csv", index=False)
    calibration_validation.to_csv(OUTPUT_DIR / "calibration_validation_results.csv", index=False)
    test_results.to_csv(OUTPUT_DIR / "final_test_results.csv", index=False)
    with open(OUTPUT_DIR / "stage11_summary.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "zero_gain_feature_count": len(zero_gain_features),
                "zero_gain_features": zero_gain_features,
                "pruning_used": pruning_used,
                "selected_feature_count": len(selected_features),
                "chosen_calibration": chosen_calibration,
                "chosen_threshold": chosen_threshold,
                "optional_models": optional_models,
                "test_used_for_selection": False,
            },
            file,
            indent=2,
        )
    joblib.dump(
        {
            "model": selected_model,
            "selected_features": selected_features,
            "calibration_name": chosen_calibration,
            "calibrator": {
                "sigmoid_platt": sigmoid,
                "isotonic": isotonic,
            }.get(chosen_calibration),
            "threshold": chosen_threshold,
        },
        OUTPUT_DIR / "final_model_bundle.joblib",
    )

    print("STAGE 11 COMPLETE")
    print(f"zero-gain features: {len(zero_gain_features)}")
    print(f"pruning used: {pruning_used}")
    print(f"selected features: {len(selected_features)}")
    print("\nValidation calibration/threshold results:")
    print(calibration_validation.round(4).to_string(index=False))
    print("\nFinal test results:")
    print(test_results.round(4).to_string(index=False))
    print(f"optional model status: {optional_models}")
    print(f"output: {OUTPUT_DIR.resolve()}")
