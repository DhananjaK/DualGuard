"""Step 4 entry point: train LightGBM per scenario on ALL training flows.

Run:  python scripts/04_train_classifier.py
Writes models/<A|B>/lightgbm.joblib and, from VALIDATION data only:
  reports/tables/lgbm_val_report_<A|B>.csv     precision/recall per family
  reports/tables/lgbm_val_confusion_<A|B>.csv  true family x predicted family
  reports/tables/lgbm_importance_<A|B>.csv     feature importance (gain)
  reports/figures/lgbm_importance_<A|B>.png
The test split is not touched in this step.
"""
import time

import lightgbm  # noqa: F401  (macOS: import LightGBM before PyTorch to avoid an OpenMP crash)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from dualguard.classifier import save_classifier, train_classifier
from dualguard.config import load_config
from dualguard.data import load_clean
from dualguard.preprocess import Preprocessor
from dualguard.seed import set_seed
from dualguard.splits import load_splits


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    P = cfg["paths"]
    df = load_clean(P["processed_dir"])

    for scenario in ("A", "B"):
        print(f"\n=== Scenario {scenario} ===")
        s = load_splits(P["processed_dir"], scenario)
        pre = Preprocessor.load(P["models_dir"] / scenario / "preprocessor.joblib")
        train, val = df.iloc[s["train"]], df.iloc[s["val"]]
        X_tr, y_tr = pre.transform(train), train["Family"].to_numpy()
        X_va, y_va = pre.transform(val), val["Family"].to_numpy()
        print(f"classes: {sorted(set(y_tr))}")

        t0 = time.time()
        model = train_classifier(X_tr, y_tr, X_va, y_va, cfg["lightgbm"], cfg["seed"])
        print(f"training time: {time.time() - t0:.0f}s")
        save_classifier(model, P["models_dir"] / scenario / "lightgbm.joblib")

        pred = model.predict(X_va)
        labels = list(model.classes_)
        rep = classification_report(y_va, pred, labels=labels, output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(rep).T
        rep_df.to_csv(P["tables_dir"] / f"lgbm_val_report_{scenario}.csv")
        cm = pd.DataFrame(confusion_matrix(y_va, pred, labels=labels), index=labels, columns=labels)
        cm.index.name, cm.columns.name = "true", "predicted"
        cm.to_csv(P["tables_dir"] / f"lgbm_val_confusion_{scenario}.csv")
        print("\nvalidation per-family results:")
        print(rep_df.loc[labels, ["precision", "recall", "f1-score", "support"]]
              .to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"macro recall: {rep['macro avg']['recall']:.4f}")
        print("\nvalidation confusion matrix (rows = true, columns = predicted):")
        print(cm.to_string())

        imp = pd.DataFrame({"feature": pre.features_,
                            "gain": model.booster_.feature_importance("gain")}) \
                .sort_values("gain", ascending=False)
        imp["share"] = imp["gain"] / imp["gain"].sum()
        imp.to_csv(P["tables_dir"] / f"lgbm_importance_{scenario}.csv", index=False)
        top = imp.head(15)[::-1]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(top["feature"], top["share"])
        ax.set(xlabel="share of total gain", title=f"LightGBM top 15 features – Scenario {scenario}")
        fig.tight_layout(); fig.savefig(P["figures_dir"] / f"lgbm_importance_{scenario}.png", dpi=150)
        plt.close(fig)
        print("\ntop 5 features:", ", ".join(imp["feature"].head(5)))


if __name__ == "__main__":
    main()
