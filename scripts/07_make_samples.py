"""Step 7b: small demo CSVs in raw CIC-IDS2017 format (with Label) for the dashboard.

Run:  python scripts/07_make_samples.py
Rows come from the Scenario B TEST split (already evaluated in Step 6; used for demo only)
plus a few Mon–Wed families from the Scenario A test split.
"""
import pandas as pd

from dualguard.config import load_config
from dualguard.data import load_clean
from dualguard.splits import load_splits

PER_FAMILY = 300


def main() -> None:
    cfg = load_config()
    P = cfg["paths"]
    df = load_clean(P["processed_dir"])
    out_dir = P["raw_dir"].parents[1] / "samples"
    out_dir.mkdir(exist_ok=True)
    drop = ["Family", "Day", "SourceFile"]

    test_a = df.iloc[load_splits(P["processed_dir"], "A")["test"]]
    test_b = df.iloc[load_splits(P["processed_dir"], "B")["test"]]
    picks = {
        "benign": test_b[test_b["Family"] == "BENIGN"],
        "portscan_unseen": test_b[test_b["Family"] == "PortScan"],
        "ddos_unseen": test_b[test_b["Family"] == "DDoS"],
        "webattack_unseen": test_b[test_b["Family"] == "WebAttack"],
        "bot_unseen": test_b[test_b["Family"] == "Bot"],
        "dos_known": test_a[test_a["Family"] == "DoS"],
        "bruteforce_known": test_a[test_a["Family"] == "BruteForce"],
    }
    parts = []
    for name, rows in picks.items():
        sample = rows.sample(min(PER_FAMILY, len(rows)), random_state=cfg["seed"])
        sample.drop(columns=drop).to_csv(out_dir / f"{name}.csv", index=False)
        parts.append(sample)
        print(f"{name:18s} {len(sample):4d} rows")
    mixed = pd.concat(parts).sample(frac=1, random_state=cfg["seed"])
    mixed.drop(columns=drop).to_csv(out_dir / "mixed_demo.csv", index=False)
    print(f"{'mixed_demo':18s} {len(mixed):4d} rows  -> {out_dir}")


if __name__ == "__main__":
    main()
