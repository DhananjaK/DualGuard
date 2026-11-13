"""Step 4: Stage 2 – LightGBM multiclass classifier on attack families.

Trained on ALL training flows (benign + labelled attacks), so it outputs a
probability for every family it knows, including BENIGN.

It can only ever answer with a family it was taught: an unseen attack
is forced into the "closest" known family. That blind spot is what the
Autoencoder (Stage 1) is for.
"""
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np


def sqrt_balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-row weight = sqrt(count of largest class / count of this row's class)."""
    classes, counts = np.unique(y, return_counts=True)
    w = np.sqrt(counts.max() / counts)
    return w[np.searchsorted(classes, y)]


def train_classifier(X_tr, y_tr, X_va, y_va, p: dict, seed: int, log=print) -> lgb.LGBMClassifier:
    weights = sqrt_balanced_weights(y_tr) if p.get("class_weight") == "sqrt_balanced" else None
    # LightGBM needs "binary" for exactly 2 classes; "multiclass" only works for 3+.
    # (Real scenarios have 4 and 9 classes; the 2-class case appears in small tests.)
    binary = len(np.unique(y_tr)) == 2
    model = lgb.LGBMClassifier(
        objective="binary" if binary else "multiclass",
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        num_leaves=p["num_leaves"],
        min_child_samples=p["min_child_samples"],
        colsample_bytree=p["feature_fraction"],
        subsample=p["bagging_fraction"],
        subsample_freq=p["bagging_freq"],
        n_jobs=p.get("threads", 8),
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    # Newer LightGBM prefers eval_X/eval_y, but that path does not encode string
    # labels ('BENIGN', 'DoS'...) and crashes. eval_set still works: keep it and
    # hide only that one deprecation warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
        model.fit(
            X_tr, y_tr, sample_weight=weights,
            eval_set=[(X_va, y_va)], eval_metric="binary_logloss" if binary else "multi_logloss",
            callbacks=[lgb.early_stopping(p["early_stopping_rounds"], verbose=False),
                       lgb.log_evaluation(period=50)],
        )
    log(f"best iteration: {model.best_iteration_} trees")
    return model


def save_classifier(model, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_classifier(path: Path):
    return joblib.load(path)
