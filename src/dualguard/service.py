"""Step 7: one object that turns raw flow records into DualGuard verdicts.

Used by the API (and could be used by any other front end). It loads the
four artefacts fixed during training and validation and never changes them.
"""
import json
from pathlib import Path

import lightgbm  # noqa: F401  (macOS: import LightGBM before PyTorch)
import numpy as np
import pandas as pd
import torch

from dualguard.autoencoder import load_autoencoder, reconstruction_error
from dualguard.classifier import load_classifier
from dualguard.data import normalise_labels
from dualguard.hybrid import BENIGN, decide, lgbm_top
from dualguard.preprocess import Preprocessor

torch.set_num_threads(1)   # one OpenMP pool at a time (see Step 5)

SCENARIO_INFO = {
    "A": {"name": "DualGuard Known", "trained_on": "random 70% of all five days",
          "use": "detects and names the 8 attack families seen in training"},
    "B": {"name": "DualGuard Zero-Day", "trained_on": "Monday to Wednesday only",
          "use": "simulates new attack types: Thu–Fri families were never seen"},
}
MODES = ("parallel", "cascade")


class Detector:
    def __init__(self, model_dir: Path, scenario: str):
        self.scenario = scenario
        self.pre = Preprocessor.load(model_dir / "preprocessor.joblib")
        self.ae = load_autoencoder(model_dir / "autoencoder.pt")
        self.clf = load_classifier(model_dir / "lightgbm.joblib")
        self.th = json.loads((model_dir / "thresholds.json").read_text())

    @property
    def features(self) -> list[str]:
        return self.pre.features_

    def info(self) -> dict:
        return {"scenario": self.scenario, **SCENARIO_INFO.get(self.scenario, {}),
                "classes": list(self.clf.classes_), "n_features": len(self.features),
                "features": self.features, "default_mode": self.th["default_mode"],
                "modes": list(MODES),
                "thresholds": {k: self.th[k] for k in ("budget", "tau", "ae", "ae_parallel")}}

    def prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Clean raw input the same way as training data. Returns (frame, valid_mask)."""
        df = df.copy()
        df.columns = df.columns.astype(str).str.strip()
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise KeyError(f"missing {len(missing)} required feature(s): {missing[:10]}")
        feats = df[self.features].apply(pd.to_numeric, errors="coerce")
        feats = feats.replace([np.inf, -np.inf], np.nan)
        valid = feats.notna().all(axis=1).to_numpy()
        df[self.features] = feats
        return df, valid

    def predict(self, df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
        """df must contain the model's features (already prepared, valid rows only)."""
        mode = mode or self.th["default_mode"]
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        X = self.pre.transform(df)
        proba, classes = self.clf.predict_proba(X), self.clf.classes_
        err = reconstruction_error(self.ae, X)
        verdict, family = decide(err, proba, classes, self.th, mode)
        top_fam, top_conf = lgbm_top(proba, classes)
        ae_t = self.th["ae"] if mode == "cascade" else self.th["ae_parallel"]
        return pd.DataFrame({
            "verdict": verdict.astype(str),
            "family": family.astype(str),
            "ae_error": err,
            "ae_threshold": ae_t,
            "ae_ratio": err / ae_t,                      # >1 means the AE finds it abnormal
            "lgbm_family": top_fam.astype(str),
            "lgbm_confidence": top_conf,
            "p_benign": proba[:, list(classes).index(BENIGN)],
        }, index=df.index)


def true_labels(df: pd.DataFrame) -> pd.Series | None:
    """If the upload carries CIC 'Label' values, return them cleaned (for comparison only)."""
    cols = {c.strip(): c for c in df.columns.astype(str)}
    return normalise_labels(df[cols["Label"]]) if "Label" in cols else None
