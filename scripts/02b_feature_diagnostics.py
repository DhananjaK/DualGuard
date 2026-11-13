"""Step 2b: which features produce extreme z-scores after preprocessing?

Run:  python scripts/02b_feature_diagnostics.py
Writes reports/tables/feature_diagnostics_<A|B>.csv
"""
import numpy as np
import pandas as pd

from dualguard.config import load_config
from dualguard.data import load_clean
from dualguard.preprocess import Preprocessor
from dualguard.splits import load_splits


def main() -> None:
    cfg = load_config()
    P = cfg["paths"]
    df = load_clean(P["processed_dir"])

    for scenario in ("A", "B"):
        s = load_splits(P["processed_dir"], scenario)
        pre = Preprocessor.load(P["models_dir"] / scenario / "preprocessor.joblib")
        pre.clip_z = None   # diagnose the raw z-scores, before any clipping
        train = df.iloc[s["train"]]
        X = pre.transform(train)
        raw = train[pre.features_]
        benign = (train["Family"] == "BENIGN").to_numpy()

        tab = pd.DataFrame({
            "feature": pre.features_,
            "share_zero_raw": (raw == 0).mean().to_numpy(),
            "n_unique_raw": raw.nunique().to_numpy(),
            "max_abs_z": np.abs(X).max(axis=0),
            "share_abs_z_gt_10": (np.abs(X) > 10).mean(axis=0),
            "benign_rows_abs_z_gt_10": (np.abs(X[benign]) > 10).sum(axis=0),
        }).sort_values("max_abs_z", ascending=False)
        tab.to_csv(P["tables_dir"] / f"feature_diagnostics_{scenario}.csv", index=False)

        print(f"\n=== Scenario {scenario}: top 12 features by max |z| ===")
        print(tab.head(12).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        any_extreme = (np.abs(X) > 10).any(axis=1)
        print(f"train rows with any |z|>10: {any_extreme.mean():.2%} "
              f"(benign: {any_extreme[benign].mean():.2%})")


if __name__ == "__main__":
    main()
