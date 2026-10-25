"""Stage 10: leakage-safe XGBoost hyperparameter tuning."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


DATA_DIR = Path("mixed_encoded_data")
OUTPUT_DIR = Path("tuned_models")
TARGET = "target_delinquent"
RANDOM_STATE = 42


if __name__ == "__main__":
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "XGBoost is required for this stage. Install it before running tuning."
        ) from exc

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_validation = pd.read_csv(DATA_DIR / "X_validation.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")[TARGET].astype("int64")
    y_validation = pd.read_csv(DATA_DIR / "y_validation.csv")[TARGET].astype("int64")

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = negative_count / positive_count

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    parameter_distributions = {
        "n_estimators": [150, 250, 350, 500],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0.0, 0.25, 0.5, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=parameter_distributions,
        n_iter=12,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        return_train_score=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    validation_probabilities = best_model.predict_proba(X_validation)[:, 1]
    validation_predictions = (validation_probabilities >= 0.5).astype("int64")
    validation_metrics = {
        "roc_auc": float(roc_auc_score(y_validation, validation_probabilities)),
        "pr_auc": float(average_precision_score(y_validation, validation_probabilities)),
        "precision": float(
            precision_score(y_validation, validation_predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_validation, validation_predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_validation, validation_predictions, zero_division=0)),
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, OUTPUT_DIR / "xgboost_tuned.joblib")
    with open(OUTPUT_DIR / "xgboost_best_params.json", "w", encoding="utf-8") as file:
        json.dump(search.best_params_, file, indent=2)
    with open(OUTPUT_DIR / "xgboost_tuning_summary.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "cv_best_pr_auc": float(search.best_score_),
                "validation_metrics_threshold_0.5": validation_metrics,
                "scale_pos_weight": scale_pos_weight,
                "n_iter": 12,
                "cv_folds": 3,
                "test_used": False,
            },
            file,
            indent=2,
        )
    cv_results = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score"
    )
    cv_results.to_csv(OUTPUT_DIR / "xgboost_cv_results.csv", index=False)

    print("XGBOOST HYPERPARAMETER TUNING COMPLETE")
    print(f"training rows: {len(X_train)}")
    print(f"validation rows: {len(X_validation)}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    print(f"CV folds: 3; randomized candidates: 12")
    print(f"best CV PR-AUC: {search.best_score_:.4f}")
    print("best parameters:")
    print(json.dumps(search.best_params_, indent=2))
    print("validation metrics at threshold=0.5:")
    print(pd.Series(validation_metrics).round(4).to_string())
    print("test split: not used")
    print(f"output: {OUTPUT_DIR.resolve()}")
