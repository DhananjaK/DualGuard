"""Step 9a: compact replay files for the Docker image.

The API's live replay normally joins test scores with flows.parquet (243 MB) to get each
flow's capture file. This writes replay_<A|B>.parquet with that column included, so the
image needs only ~20 MB instead. No model is run; it only reshapes Step 6 outputs.

Run:  python scripts/08_export_replay.py
"""
import pandas as pd

from dualguard.config import load_config


def main() -> None:
    P = load_config()["paths"]
    files = pd.read_parquet(P["processed_dir"] / "flows.parquet", columns=["SourceFile"])["SourceFile"]
    for s in ("A", "B"):
        sc = pd.read_parquet(P["processed_dir"] / f"test_scored_{s}.parquet")
        sc["SourceFile"] = files.iloc[sc["row"]].to_numpy()
        for c in ("Day", "Label", "Family", "lgbm_family", "verdict_parallel", "verdict_cascade", "SourceFile"):
            sc[c] = sc[c].astype("category")              # repeated strings -> small
        out = P["processed_dir"] / f"replay_{s}.parquet"
        sc.sort_values("row").to_parquet(out, index=False)
        print(f"{out.name}: {len(sc):,} rows, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
