"""Tune classification thresholds on validation data and compare models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATA_DIR = Path("mixed_encoded_data")
OUTPUT_DIR = Path("threshold_results")
TARGET = "target_delinquent"


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_threshold(y_true, probabilities):
    # Every distinct probability is a possible classification boundary.
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    scored = [
        evaluate(y_true, probabilities, threshold)
        for threshold in candidates
    ]
    # Maximize F1; break ties in favor of recall, then precision.
    return max(scored, key=lambda row: (row["f1"], row["recall"], row["precision"]))


if __name__ == "__main__":
    X_validation = pd.read_csv(DATA_DIR / "X_validation.csv")
    y_validation = pd.read_csv(DATA_DIR / "y_validation.csv")[TARGET].astype("int64")

    baseline_models = joblib.load(Path("baseline_models/baseline_models.joblib"))
    comparison_models = joblib.load(
        Path("comparison_models/lightgbm_catboost_models.joblib")
    )
    tuned_xgboost = joblib.load(Path("tuned_models/xgboost_tuned.joblib"))
    models = {
        "logistic_regression": baseline_models["logistic_regression"],
        "original_xgboost": baseline_models["xgboost"],
        "tuned_xgboost": tuned_xgboost,
        "lightgbm": comparison_models["lightgbm"],
        "catboost": comparison_models["catboost"],
    }

    rows = []
    thresholds = {}
    for name, model in models.items():
        probabilities = model.predict_proba(X_validation)[:, 1]
        default = evaluate(y_validation, probabilities, 0.5)
        tuned = best_threshold(y_validation, probabilities)
        rows.append(
            {
                "model": name,
                "default_threshold": 0.5,
                "default_f1": default["f1"],
                "default_precision": default["precision"],
                "default_recall": default["recall"],
                "tuned_threshold": tuned["threshold"],
                "tuned_f1": tuned["f1"],
                "tuned_precision": tuned["precision"],
                "tuned_recall": tuned["recall"],
                "tuned_pr_auc": tuned["pr_auc"],
                "tuned_roc_auc": tuned["roc_auc"],
                "tn": tuned["tn"],
                "fp": tuned["fp"],
                "fn": tuned["fn"],
                "tp": tuned["tp"],
            }
        )
        thresholds[name] = tuned["threshold"]

    results = pd.DataFrame(rows).sort_values("tuned_f1", ascending=False)
    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_DIR / "validation_threshold_comparison.csv", index=False)
    with open(OUTPUT_DIR / "selected_thresholds.json", "w", encoding="utf-8") as file:
        json.dump(thresholds, file, indent=2)

    print("VALIDATION THRESHOLD TUNING COMPLETE")
    print("optimization objective: maximum F1; tie-breakers: recall, precision")
    print("test split: not used")
    print(results.round(4).to_string(index=False))
    print(f"output: {OUTPUT_DIR.resolve()}")
