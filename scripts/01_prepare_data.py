"""Step 1 entry point: raw CSVs -> cleaned parquet + split indices + summary tables.

Run:  python scripts/01_prepare_data.py
"""
import json

import numpy as np
import pandas as pd

from dualguard.config import load_config
from dualguard.data import clean, load_raw
from dualguard.seed import set_seed
from dualguard.splits import check_disjoint, family_counts, split_scenario_a, split_scenario_b


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    P = cfg["paths"]
    for d in (P["processed_dir"], P["tables_dir"]):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading raw CSVs ...")
    raw = load_raw(P["raw_dir"])
    raw_counts = raw.assign(Label=raw["Label"].astype(str).str.strip()) \
                    .groupby(["Day", "Label"]).size().rename("rows")

    print("Cleaning ...")
    df, report = clean(raw)
    del raw
    print(json.dumps(report, indent=2))

    df.to_parquet(P["processed_dir"] / "flows.parquet", index=False)
    (P["tables_dir"] / "data_cleaning_report.json").write_text(json.dumps(report, indent=2))
    raw_counts.to_csv(P["tables_dir"] / "raw_label_counts.csv")
    df.groupby(["Day", "Label"]).size().rename("rows") \
      .to_csv(P["tables_dir"] / "clean_label_counts.csv")

    for name, fn in (("A", split_scenario_a), ("B", split_scenario_b)):
        splits = fn(df, cfg, cfg["seed"])
        check_disjoint(splits)
        np.savez(P["processed_dir"] / f"splits_{name}.npz", **splits)
        counts = family_counts(df, splits)
        counts.to_csv(P["tables_dir"] / f"family_counts_{name}.csv")
        print(f"\nScenario {name} - rows per family:\n{counts}")


if __name__ == "__main__":
    main()
