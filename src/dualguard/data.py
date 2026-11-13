"""Step 1: load and clean the CIC-IDS2017 MachineLearningCVE CSVs.

Pipeline: load_raw() -> clean() -> DataFrame with features + Label, Family, Day.
"""
from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = ["Label", "Family", "Day", "SourceFile"]

# Fine-grained label -> attack family (the classes LightGBM will learn).
FAMILY_MAP = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "Heartbleed",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Infiltration": "Infiltration",
    "Bot": "Bot",
    "PortScan": "PortScan",
    "DDoS": "DDoS",
}


def _day_from_filename(path: Path) -> str:
    # "Wednesday-workingHours.pcap_ISCX.csv" -> "Wednesday"
    return path.name.split("-")[0].capitalize()


def load_raw(raw_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read all 8 CSVs into one DataFrame and tag each row with its capture day."""
    paths = sorted(Path(raw_dir).glob("*.pcap_ISCX.csv"))
    if not paths:
        raise FileNotFoundError(f"No *.pcap_ISCX.csv files in {raw_dir}")
    frames = []
    for path in paths:
        # Web-attack labels contain a broken dash byte; 'replace' stops a crash.
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace",
                         nrows=nrows, low_memory=False)
        df.columns = df.columns.str.strip()          # " Label" -> "Label"
        df["Day"] = _day_from_filename(path)
        df["SourceFile"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def normalise_labels(labels: pd.Series) -> pd.Series:
    """'Web Attack � XSS' -> 'Web Attack - XSS'; strip whitespace."""
    return (labels.astype(str).str.strip()
            .str.replace(r"Web Attack\W+", "Web Attack - ", regex=True))


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned frame and a report of how many rows each rule removed."""
    report = {"rows_loaded": len(df)}

    # 1. The CSVs contain 'Fwd Header Length' twice (pandas renames the copy '.1').
    dup_col = "Fwd Header Length.1"
    if dup_col in df.columns:
        if df[dup_col].equals(df["Fwd Header Length"]):
            df = df.drop(columns=dup_col)
            report["dropped_duplicate_column"] = dup_col
        else:
            report["dropped_duplicate_column"] = "none (columns differ, kept both)"

    # 2. Labels -> clean strings -> families. Fail loudly on anything unknown.
    df["Label"] = normalise_labels(df["Label"])
    unknown = set(df["Label"].unique()) - set(FAMILY_MAP)
    if unknown:
        raise ValueError(f"Unmapped labels: {unknown}")
    df["Family"] = df["Label"].map(FAMILY_MAP)

    # 3. Features must be numeric; +/-inf (from division by zero duration) -> NaN -> drop.
    feats = feature_columns(df)
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=feats)
    report["rows_dropped_nan_or_inf"] = before - len(df)

    # 4. float32 halves memory; the models train in float32 anyway.
    df[feats] = df[feats].astype(np.float32)

    # 5. Rows whose identical feature vector carries different labels are ambiguous.
    row_hash = pd.util.hash_pandas_object(df[feats], index=False)
    n_labels = df.groupby(row_hash.values)["Label"].transform("nunique")
    before = len(df)
    df = df[n_labels.values == 1]
    report["rows_dropped_conflicting_labels"] = before - len(df)

    # 6. Exact duplicates leak between train and test in a random split -> keep first.
    before = len(df)
    df = df.drop_duplicates(subset=feats + ["Label"], keep="first")
    report["rows_dropped_duplicates"] = before - len(df)

    df = df.reset_index(drop=True)
    report["rows_final"] = len(df)
    report["n_features"] = len(feats)
    return df, report


def load_clean(processed_dir: Path) -> pd.DataFrame:
    """Load the cleaned flows written by scripts/01_prepare_data.py."""
    return pd.read_parquet(Path(processed_dir) / "flows.parquet")
