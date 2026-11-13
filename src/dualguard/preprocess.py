"""Step 2: turn cleaned flows into model-ready feature matrices.

Fitted on TRAINING rows only, then applied unchanged to val/test/live data.

    raw features ─► drop chosen + constant columns ─► signed log1p ─► standardise ─► clip ±clip_z
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dualguard.data import feature_columns


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """sign(x) * log(1 + |x|): squashes 1e9-sized values, keeps -1 sentinels and order."""
    return np.sign(x) * np.log1p(np.abs(x))


class Preprocessor:
    def __init__(self, drop_features: list[str] | None = None, clip_z: float | None = None):
        self.drop_features = list(drop_features or [])
        self.clip_z = clip_z

    def fit(self, df_train: pd.DataFrame) -> "Preprocessor":
        candidates = [c for c in feature_columns(df_train) if c not in self.drop_features]
        X = df_train[candidates].to_numpy(dtype=np.float64)
        # A column with one value in training carries no information.
        is_const = X.max(axis=0) == X.min(axis=0)
        self.constant_features_ = [c for c, k in zip(candidates, is_const) if k]
        self.features_ = [c for c, k in zip(candidates, is_const) if not k]
        self.scaler_ = StandardScaler().fit(signed_log1p(X[:, ~is_const]))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        missing = set(self.features_) - set(df.columns)
        if missing:
            raise KeyError(f"Input is missing features: {sorted(missing)}")
        X = df[self.features_].to_numpy(dtype=np.float64)
        Z = self.scaler_.transform(signed_log1p(X))
        if self.clip_z is not None:
            # Rare flag values give z of 60-180; capping keeps them "very unusual"
            # without letting one feature dominate the Autoencoder's squared error.
            Z = np.clip(Z, -self.clip_z, self.clip_z)
        return Z.astype(np.float32)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "Preprocessor":
        return joblib.load(path)
