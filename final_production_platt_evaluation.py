"""Final production evaluation: tuned XGBoost plus Platt scaling."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
MODEL_PATH = Path("tuned_models/xgboost_tuned.joblib")
OUTPUT_DIR = Path("production_platt_results")
TARGET = "target_delinquent"


def metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(y_true, probabilities):
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    rows = [metrics(y_true, probabilities, threshold) for threshold in candidates]
    return max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))


if __name__ == "__main__":
    X_validation = pd.read_csv(DATA_DIR / "X_validation.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_validation = pd.read_csv(DATA_DIR / "y_validation.csv")[TARGET].astype("int64")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")[TARGET].astype("int64")
    tuned_xgboost = joblib.load(MODEL_PATH)

    validation_raw = tuned_xgboost.predict_proba(X_validation)[:, 1]
    test_raw = tuned_xgboost.predict_proba(X_test)[:, 1]

    # Platt scaling is fitted using validation predictions and validation
    # outcomes. Test outcomes are not used for fitting or threshold selection.
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(validation_raw.reshape(-1, 1), y_validation)
    validation_calibrated = platt.predict_proba(validation_raw.reshape(-1, 1))[:, 1]
    test_calibrated = platt.predict_proba(test_raw.reshape(-1, 1))[:, 1]

    validation_result = choose_threshold(y_validation, validation_calibrated)
    threshold = validation_result["threshold"]
    test_result = metrics(y_test, test_calibrated, threshold)
    test_result.update(
        {
            "model": "tuned_xgboost_plus_platt",
            "rows": int(len(y_test)),
            "positive_count": int(y_test.sum()),
        }
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": tuned_xgboost,
            "calibrator": platt,
            "threshold": threshold,
        },
        OUTPUT_DIR / "production_model_bundle.joblib",
    )
    pd.DataFrame([test_result]).to_csv(
        OUTPUT_DIR / "production_test_metrics.csv", index=False
    )
    (OUTPUT_DIR / "production_test_metrics.json").write_text(
        json.dumps(
            {
                "validation_calibrated_metrics": validation_result,
                "test_calibrated_metrics": test_result,
                "calibration": "Platt sigmoid",
                "test_used_for_fitting_or_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("FINAL PRODUCTION PLATT EVALUATION COMPLETE")
    print("calibrator: Platt sigmoid")
    print("threshold selected on validation: {:.6f}".format(threshold))
    print("\nValidation calibrated metrics:")
    print(pd.Series(validation_result).to_string())
    print("\nTest calibrated metrics:")
    print(pd.Series(test_result).to_string())
    print("test labels were not used for fitting or selection")
    print(f"output: {OUTPUT_DIR.resolve()}")
