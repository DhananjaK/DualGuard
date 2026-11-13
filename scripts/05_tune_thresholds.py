"""Step 5 entry point: set thresholds on VALIDATION and compare decision logics.

Run:  python scripts/05_tune_thresholds.py
Writes models/<A|B>/thresholds.json and reports/tables/val_detection_<A|B>.csv
The test split is not touched in this step.
"""
import json
import time

import lightgbm  # noqa: F401  (macOS: import LightGBM before PyTorch)
import numpy as np
import pandas as pd
import torch

from dualguard.autoencoder import load_autoencoder, reconstruction_error
from dualguard.classifier import load_classifier
from dualguard.config import load_config
from dualguard.data import load_clean
from dualguard.hybrid import KNOWN, UNKNOWN, baseline_alarm, decide, fit_thresholds
from dualguard.metrics import detection_table
from dualguard.preprocess import Preprocessor
from dualguard.splits import load_splits


def step(msg: str, t0: list) -> None:
    now = time.time()
    print(f"  [{now - t0[0]:6.1f}s] {msg}", flush=True)


def main() -> None:
    torch.set_num_threads(1)   # avoid two OpenMP thread pools (PyTorch + LightGBM) fighting on macOS
    t0 = [time.time()]
    cfg = load_config()
    P, T = cfg["paths"], cfg["thresholds"]
    df = load_clean(P["processed_dir"])
    step("loaded flows.parquet", t0)

    for scenario in ("A", "B"):
        print(f"\n=== Scenario {scenario} (validation) ===")
        M = P["models_dir"] / scenario
        s = load_splits(P["processed_dir"], scenario)
        pre = Preprocessor.load(M / "preprocessor.joblib")
        ae, clf = load_autoencoder(M / "autoencoder.pt"), load_classifier(M / "lightgbm.joblib")

        val = df.iloc[s["val"]]
        X, y = pre.transform(val), val["Family"].to_numpy()
        step(f"preprocessed {len(X):,} validation flows", t0)
        proba, classes = clf.predict_proba(X), clf.classes_      # LightGBM first (OpenMP order)
        step("LightGBM probabilities", t0)
        err = reconstruction_error(ae, X)
        step("AE reconstruction errors", t0)

        th = fit_thresholds(err, proba, classes, y, T["benign_false_alarm_budget"], T["tau"])
        step("thresholds fitted", t0)
        verdicts = {m: decide(err, proba, classes, th, m)[0] for m in ("cascade", "parallel")}
        alarms = {
            "ae_only": baseline_alarm(err, proba, classes, th, "ae_only"),
            "lgbm_only": baseline_alarm(err, proba, classes, th, "lgbm_only"),
            "cascade": verdicts["cascade"] != "BENIGN",
            "parallel": verdicts["parallel"] != "BENIGN",
        }
        tab = detection_table(y, alarms)
        tab.to_csv(P["tables_dir"] / f"val_detection_{scenario}.csv")

        macro = tab.loc["MACRO (attacks)"]
        th["default_mode"] = "cascade" if macro["cascade"] > macro["parallel"] else "parallel"
        (M / "thresholds.json").write_text(json.dumps(th, indent=2))

        print(json.dumps(th, indent=2))
        print("\ndetection rate per family (BENIGN row = false-alarm rate):")
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
        for m, v in verdicts.items():
            counts = pd.Series(v).value_counts()
            print(f"{m:8s} verdicts: KNOWN {counts.get(KNOWN, 0):,}  UNKNOWN {counts.get(UNKNOWN, 0):,}  "
                  f"BENIGN {counts.get('BENIGN', 0):,}")
        print(f"default mode (higher validation macro detection): {th['default_mode']}")


if __name__ == "__main__":
    main()
