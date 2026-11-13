"""Step 3 entry point: train the Autoencoder for each scenario on BENIGN train rows.

Run:  python scripts/03_train_autoencoder.py
Writes models/<A|B>/autoencoder.pt, reports/tables/ae_history_<A|B>.csv,
       reports/tables/ae_val_errors_<A|B>.csv, reports/figures/ae_loss_<A|B>.png
Uses VALIDATION data only. The test split is not touched in this step.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dualguard.autoencoder import reconstruction_error, save_autoencoder, train_autoencoder
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

        X_tr = pre.transform(train[train["Family"] == "BENIGN"])   # benign ONLY
        X_va_benign = pre.transform(val[val["Family"] == "BENIGN"])
        print(f"benign train rows: {len(X_tr):,}   benign val rows: {len(X_va_benign):,}")

        t0 = time.time()
        model, hist = train_autoencoder(X_tr, X_va_benign, cfg["autoencoder"], cfg["seed"])
        print(f"training time: {time.time() - t0:.0f}s")
        save_autoencoder(model, P["models_dir"] / scenario / "autoencoder.pt")

        h = pd.DataFrame(hist)
        h.to_csv(P["tables_dir"] / f"ae_history_{scenario}.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(h["epoch"], h["train_loss"], label="train (benign)")
        ax.plot(h["epoch"], h["val_loss"], label="validation (benign)")
        ax.set(xlabel="epoch", ylabel="mean squared error", yscale="log",
               title=f"Autoencoder loss – Scenario {scenario}")
        ax.legend(); fig.tight_layout()
        fig.savefig(P["figures_dir"] / f"ae_loss_{scenario}.png", dpi=150); plt.close(fig)

        # Sanity check on VALIDATION: do attacks reconstruct worse than benign?
        err = reconstruction_error(model, pre.transform(val))
        fam = val["Family"].to_numpy()
        cut = np.quantile(err[fam == "BENIGN"], 0.98)     # preview only; Step 5 sets the real threshold
        rows = []
        for f in pd.unique(fam):
            e = err[fam == f]
            rows.append({"family": f, "n": len(e), "median_error": np.median(e),
                         "p90_error": np.quantile(e, 0.90), "share_above_benign_p98": (e > cut).mean()})
        tab = pd.DataFrame(rows).sort_values("median_error")
        tab.to_csv(P["tables_dir"] / f"ae_val_errors_{scenario}.csv", index=False)
        print(f"\nvalidation reconstruction error by family (benign 98th percentile = {cut:.4f}):")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
