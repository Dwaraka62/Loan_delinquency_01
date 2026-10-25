"""Compare LightGBM and CatBoost with the existing validation baselines."""

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
OUTPUT_DIR = Path("comparison_models")
TARGET = "target_delinquent"
RANDOM_STATE = 42


def evaluate(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
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


if __name__ == "__main__":
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_validation = pd.read_csv(DATA_DIR / "X_validation.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")[TARGET].astype("int64")
    y_validation = pd.read_csv(DATA_DIR / "y_validation.csv")[TARGET].astype("int64")

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = negative_count / positive_count

    models = {
        "lightgbm": LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=-1,
        ),
        "catboost": CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="PRAUC",
            iterations=300,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            class_weights=[1.0, scale_pos_weight],
            verbose=False,
            thread_count=1,
        ),
    }

    results = []
    saved_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_validation)[:, 1]
        results.append({"model": name, **evaluate(y_validation, probabilities)})
        saved_models[name] = model

    # Add prior baseline results for a single comparison table. These values
    # were obtained from the same validation split and threshold.
    prior_results_path = Path("baseline_models/validation_results.csv")
    if prior_results_path.exists():
        prior = pd.read_csv(prior_results_path)
        results.extend(prior.to_dict("records"))
    tuned_summary_path = Path("tuned_models/xgboost_tuning_summary.json")
    if tuned_summary_path.exists():
        tuned = json.loads(tuned_summary_path.read_text(encoding="utf-8"))
        tuned_metrics = tuned["validation_metrics_threshold_0.5"]
        results.append({"model": "xgboost_tuned", **tuned_metrics})

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "validation_comparison.csv", index=False)
    joblib.dump(saved_models, OUTPUT_DIR / "lightgbm_catboost_models.joblib")

    print("LIGHTGBM/CATBOOST COMPARISON COMPLETE")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    print("threshold: 0.5")
    print("test split: not used")
    print(results_df.round(4).to_string(index=False))
    print(f"output: {OUTPUT_DIR.resolve()}")
