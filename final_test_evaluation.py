"""Final one-time test evaluation for the validation-selected tuned XGBoost."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
THRESHOLD_PATH = Path("threshold_results/selected_thresholds.json")
OUTPUT_DIR = Path("final_test_results")
TARGET = "target_delinquent"
MODEL_NAME = "tuned_xgboost"


if __name__ == "__main__":
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")[TARGET].astype("int64")
    model = joblib.load(MODEL_PATH)
    thresholds = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    threshold = float(thresholds[MODEL_NAME])

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    results = {
        "model": MODEL_NAME,
        "threshold": threshold,
        "rows": len(y_test),
        "positive_count": int(y_test.sum()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier": float(brier_score_loss(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame([results]).to_csv(OUTPUT_DIR / "tuned_xgboost_test_metrics.csv", index=False)
    (OUTPUT_DIR / "tuned_xgboost_test_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("FINAL TEST EVALUATION COMPLETE")
    print("The threshold was selected on validation data and then held fixed.")
    print(pd.Series(results).to_string())
    print(f"output: {OUTPUT_DIR.resolve()}")
