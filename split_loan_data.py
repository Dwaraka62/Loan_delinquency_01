"""Stage 6: create leakage-safe stratified train/validation/test splits."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("loans_engineered.csv")
OUTPUT_DIR = Path("data_splits")
TARGET = "target_delinquent"
RANDOM_STATE = 42


if __name__ == "__main__":
    data = pd.read_csv(INPUT_PATH)
    if TARGET not in data.columns:
        raise ValueError(f"Required target column is missing: {TARGET}")

    train, temporary = train_test_split(
        data,
        test_size=0.30,
        stratify=data[TARGET],
        random_state=RANDOM_STATE,
    )
    validation, test = train_test_split(
        temporary,
        test_size=2 / 3,
        stratify=temporary[TARGET],
        random_state=RANDOM_STATE,
    )

    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    print("STRATIFIED DATA SPLIT COMPLETE")
    print(f"target: {TARGET}")
    print(f"random_state: {RANDOM_STATE}")
    for name, frame in splits.items():
        print(
            f"{name}: rows={len(frame)}, "
            f"delinquent={int(frame[TARGET].sum())}, "
            f"rate={frame[TARGET].mean() * 100:.3f}%"
        )
    print(f"output: {OUTPUT_DIR.resolve()}")
