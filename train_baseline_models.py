"""Stage 9: train and compare baseline delinquency models.

The test split is intentionally not used here.  Model comparison is based on
the validation split; final test evaluation belongs to a later stage.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATA_DIR = Path("mixed_encoded_data")
OUTPUT_DIR = Path("baseline_models")
TARGET = "target_delinquent"
RANDOM_STATE = 42


def evaluate(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float | int]:
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
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_validation = pd.read_csv(DATA_DIR / "X_validation.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")[TARGET].astype("int64")
    y_validation = pd.read_csv(DATA_DIR / "y_validation.csv")[TARGET].astype("int64")

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = negative_count / positive_count
    sample_weights = np.where(y_train.to_numpy() == 1, scale_pos_weight, 1.0)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    }
    gradient_model_error = None
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    except Exception as exc:
        gradient_model_error = f"XGBoost unavailable: {type(exc).__name__}: {exc}"

    # Use sklearn boosting only when XGBoost is unavailable. In this
    # environment its tree DLL may be policy-blocked, so retain the error.
    if "xgboost" not in models:
        hgb_error = None
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier

            models["gradient_boosting_fallback"] = HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            )
        except Exception as exc:
            hgb_error = f"HistGradientBoosting unavailable: {type(exc).__name__}: {exc}"
        if hgb_error:
            gradient_model_error = f"{gradient_model_error}; {hgb_error}"

    results = []
    models_to_save = {}
    for name, model in models.items():
        if name == "gradient_boosting_fallback":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        elif name == "xgboost":
            # XGBoost receives the class weighting through scale_pos_weight;
            # do not apply the same weight a second time as sample_weight.
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_validation)[:, 1]
        metrics = evaluate(y_validation, probabilities)
        results.append({"model": name, **metrics})
        models_to_save[name] = model

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "validation_results.csv", index=False)
    joblib.dump(models_to_save, OUTPUT_DIR / "baseline_models.joblib")

    print("BASELINE MODEL TRAINING COMPLETE")
    print(f"training rows: {len(X_train)}")
    print(f"validation rows: {len(X_validation)}")
    print(f"training positives: {positive_count}")
    print(f"training negatives: {negative_count}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    print("test split: not used")
    if gradient_model_error:
        print(f"gradient boosting unavailable: {gradient_model_error}")
    print("\nValidation results at threshold=0.5:")
    print(results_df.round(4).to_string(index=False))
    print(f"output: {OUTPUT_DIR.resolve()}")
