"""Step 2 entry point: fit one Preprocessor per scenario on its TRAINING rows.

Run:  python scripts/02_preprocess.py
Writes models/<A|B>/preprocessor.joblib and reports/tables/features_<A|B>.csv
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
        pre = Preprocessor(cfg["preprocess"]["drop_features"],
                           cfg["preprocess"].get("clip_z")).fit(df.iloc[s["train"]])
        pre.save(P["models_dir"] / scenario / "preprocessor.joblib")

        rows = ([(f, "used") for f in pre.features_]
                + [(f, "dropped: constant in train") for f in pre.constant_features_]
                + [(f, "dropped: config") for f in pre.drop_features])
        pd.DataFrame(rows, columns=["feature", "status"]) \
          .to_csv(P["tables_dir"] / f"features_{scenario}.csv", index=False)

        print(f"\n=== Scenario {scenario} ===")
        print(f"features used: {len(pre.features_)}")
        print(f"dropped (config):   {pre.drop_features}")
        print(f"dropped (constant): {pre.constant_features_}")
        for split in ("train", "val", "test"):
            X = pre.transform(df.iloc[s[split]])
            print(f"{split:5s} shape={X.shape}  mean={X.mean():+.3f}  std={X.std():.3f}  "
                  f"min={X.min():+.1f}  max={X.max():+.1f}")


if __name__ == "__main__":
    main()
