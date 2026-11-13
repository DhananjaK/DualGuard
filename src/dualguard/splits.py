"""Step 1b: build row-index splits for the two evaluation scenarios.

Splits are stored as index arrays (not copies) so every later step
uses exactly the same rows.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_scenario_a(df: pd.DataFrame, cfg: dict, seed: int) -> dict[str, np.ndarray]:
    """Random 70/15/15 split, stratified by Family: every family is 'known'."""
    p = cfg["data"]["scenario_a"]
    idx = np.arange(len(df))
    y = df["Family"].values
    train, temp = train_test_split(idx, train_size=p["train"], stratify=y, random_state=seed)
    val_share = p["val"] / (p["val"] + p["test"])
    val, test = train_test_split(temp, train_size=val_share, stratify=y[temp], random_state=seed)
    return {"train": train, "val": val, "test": test}


def split_scenario_b(df: pd.DataFrame, cfg: dict, seed: int) -> dict[str, np.ndarray]:
    """Temporal split: learn on Mon-Wed, test on Thu-Fri (attack families never seen)."""
    p = cfg["data"]["scenario_b"]
    pool = np.flatnonzero(df["Day"].isin(p["train_days"]).values)
    test = np.flatnonzero(df["Day"].isin(p["test_days"]).values)
    train, val = train_test_split(pool, test_size=p["val_fraction_of_train"],
                                  stratify=df["Family"].values[pool], random_state=seed)
    return {"train": train, "val": val, "test": test}


def check_disjoint(splits: dict[str, np.ndarray]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = np.intersect1d(splits[a], splits[b]).size
            assert overlap == 0, f"{a} and {b} share {overlap} rows"


def family_counts(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> pd.DataFrame:
    """Table: rows per Family in each split."""
    cols = {name: df["Family"].iloc[ix].value_counts() for name, ix in splits.items()}
    return pd.DataFrame(cols).fillna(0).astype(int)


def load_splits(processed_dir, scenario: str) -> dict[str, np.ndarray]:
    """Load train/val/test row indices for scenario 'A' or 'B'."""
    with np.load(processed_dir / f"splits_{scenario}.npz") as z:
        return {k: z[k] for k in z.files}
