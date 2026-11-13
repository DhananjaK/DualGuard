"""Step 6: evaluate on the TEST split – the only step that opens it.

Run:  python scripts/06_evaluate_test.py --open-test
Nothing is tuned here: models and thresholds were fixed on train/validation.
Every run is logged to reports/tables/test_opened_log.csv with model fingerprints.

Writes per scenario:
  reports/tables/test_detection_<A|B>.csv   detection rate per family x method
  reports/tables/test_summary_<A|B>.csv     false alarms, macro detection, alert precision per method
  reports/tables/test_naming_<A|B>.csv      what the hybrid called each true family
  reports/figures/test_ae_error_<A|B>.png   AE score per family vs threshold
  data/processed/test_scored_<A|B>.parquet  per-flow scores (used by the dashboard)
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone

import lightgbm  # noqa: F401  (macOS: import LightGBM before PyTorch)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dualguard.autoencoder import load_autoencoder, reconstruction_error
from dualguard.classifier import load_classifier
from dualguard.config import load_config
from dualguard.data import load_clean
from dualguard.hybrid import BENIGN, baseline_alarm, decide, lgbm_top
from dualguard.metrics import detection_table
from dualguard.preprocess import Preprocessor
from dualguard.splits import load_splits

MODEL_FILES = ["preprocessor.joblib", "autoencoder.pt", "lightgbm.joblib", "thresholds.json"]


def fingerprint(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def log_opening(log_path, scenario: str, model_dir) -> None:
    new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc_time", "scenario", *MODEL_FILES])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), scenario,
                    *[fingerprint(model_dir / m) for m in MODEL_FILES]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-test", action="store_true", help="confirm you intend to open the test split")
    if not ap.parse_args().open_test:
        raise SystemExit("Refusing to touch the test split. Re-run with --open-test once everything is frozen.")

    torch.set_num_threads(1)
    cfg = load_config()
    P = cfg["paths"]
    df = load_clean(P["processed_dir"])

    for scenario in ("A", "B"):
        print(f"\n=== Scenario {scenario} (TEST) ===")
        M = P["models_dir"] / scenario
        log_opening(P["tables_dir"] / "test_opened_log.csv", scenario, M)
        s = load_splits(P["processed_dir"], scenario)
        pre = Preprocessor.load(M / "preprocessor.joblib")
        ae, clf = load_autoencoder(M / "autoencoder.pt"), load_classifier(M / "lightgbm.joblib")
        th = json.loads((M / "thresholds.json").read_text())

        test = df.iloc[s["test"]]
        X, y = pre.transform(test), test["Family"].to_numpy()
        proba, classes = clf.predict_proba(X), clf.classes_
        err = reconstruction_error(ae, X)

        v_cas, f_cas = decide(err, proba, classes, th, "cascade")
        v_par, f_par = decide(err, proba, classes, th, "parallel")
        alarms = {
            "ae_only": baseline_alarm(err, proba, classes, th, "ae_only"),
            "lgbm_only": baseline_alarm(err, proba, classes, th, "lgbm_only"),
            "cascade": v_cas != BENIGN,
            "parallel": v_par != BENIGN,
        }

        # 1. Detection per family
        tab = detection_table(y, alarms)
        tab.to_csv(P["tables_dir"] / f"test_detection_{scenario}.csv")

        # 2. Operational summary: how trustworthy is an alert?
        is_attack = y != BENIGN
        summary = pd.DataFrame({
            m: {"benign_false_alarm_rate": a[~is_attack].mean(),
                "macro_detection": tab.loc["MACRO (attacks)", m],
                "overall_detection": a[is_attack].mean(),
                "alerts": int(a.sum()),
                "alert_precision": a[is_attack].sum() / max(a.sum(), 1)}
            for m, a in alarms.items()}).T
        summary.to_csv(P["tables_dir"] / f"test_summary_{scenario}.csv")

        # 3. What did the hybrid call each true family?
        naming = pd.concat({
            "parallel": pd.crosstab(pd.Series(y, name="true"), pd.Series(f_par, name="output")),
            "cascade": pd.crosstab(pd.Series(y, name="true"), pd.Series(f_cas, name="output")),
        }, axis=1).fillna(0).astype(int)
        naming.to_csv(P["tables_dir"] / f"test_naming_{scenario}.csv")

        # 4. Per-flow scores for the dashboard
        top_fam, top_conf = lgbm_top(proba, classes)
        pd.DataFrame({"row": s["test"], "Day": test["Day"].to_numpy(), "Label": test["Label"].to_numpy(),
                      "Family": y, "ae_error": err, "lgbm_family": top_fam, "lgbm_conf": top_conf,
                      "verdict_parallel": v_par.astype(str), "verdict_cascade": v_cas.astype(str)}) \
          .to_parquet(P["processed_dir"] / f"test_scored_{scenario}.parquet", index=False)

        # 5. Figure: AE error per family against the threshold
        fams = [BENIGN] + sorted(f for f in np.unique(y) if f != BENIGN)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot([np.log10(err[y == f] + 1e-6) for f in fams], tick_labels=fams, showfliers=False)
        ax.axhline(np.log10(th["ae"]), color="red", ls="--", label="AE threshold (2% benign val)")
        ax.set(ylabel="log10 reconstruction error", title=f"AE score by family – Scenario {scenario} test")
        ax.tick_params(axis="x", rotation=30); ax.legend(); fig.tight_layout()
        fig.savefig(P["figures_dir"] / f"test_ae_error_{scenario}.png", dpi=150); plt.close(fig)

        print("detection rate per family (BENIGN row = false-alarm rate):")
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
        print("\nsummary:")
        print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
        print("\nwhat the PARALLEL hybrid called each family (rows = truth):")
        print(naming["parallel"].to_string())


if __name__ == "__main__":
    main()
