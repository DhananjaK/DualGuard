"""Step 6b (POST-HOC analysis, no tuning): where do Scenario B's false alarms come from?

Reads the saved test scores only; no model or threshold is changed.
Run:  python scripts/06b_posthoc_shift.py
Writes reports/tables/posthoc_B_by_file.csv
"""
import json

import pandas as pd

from dualguard.config import load_config
from dualguard.data import load_clean


def main() -> None:
    cfg = load_config()
    P = cfg["paths"]
    th = json.loads((P["models_dir"] / "B" / "thresholds.json").read_text())
    scored = pd.read_parquet(P["processed_dir"] / "test_scored_B.parquet")
    files = load_clean(P["processed_dir"])["SourceFile"]
    scored["SourceFile"] = files.iloc[scored["row"]].to_numpy()
    scored["ae_alarm"] = scored["ae_error"] > th["ae"]

    benign = scored[scored["Family"] == "BENIGN"]
    attack = scored[scored["Family"] != "BENIGN"]
    tab = pd.DataFrame({
        "benign_flows": benign.groupby("SourceFile").size(),
        "benign_false_alarm_rate": benign.groupby("SourceFile")["ae_alarm"].mean(),
        "attack_flows": attack.groupby("SourceFile").size(),
        "attack_detection_rate": attack.groupby("SourceFile")["ae_alarm"].mean(),
    }).fillna(0)
    tab.to_csv(P["tables_dir"] / "posthoc_B_by_file.csv")
    print("Scenario B test, AE alarm (threshold from validation) per capture file:")
    print(tab.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
