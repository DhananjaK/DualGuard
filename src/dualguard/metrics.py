"""Detection metrics per attack family, shared by validation (Step 5) and test (Step 6)."""
import numpy as np
import pandas as pd

BENIGN = "BENIGN"


def detection_table(y: np.ndarray, alarms: dict[str, np.ndarray]) -> pd.DataFrame:
    """Rows = family, columns = method; value = share of that family's flows that raised an alarm.

    For BENIGN the value is the false-alarm rate (lower is better);
    for attacks it is the detection rate (higher is better).
    """
    fams = [BENIGN] + sorted(f for f in np.unique(y) if f != BENIGN)
    out = pd.DataFrame(index=fams, columns=list(alarms), dtype=float)
    for m, a in alarms.items():
        for f in fams:
            out.loc[f, m] = a[y == f].mean()
    out.insert(0, "n", [int((y == f).sum()) for f in fams])
    attacks = [f for f in fams if f != BENIGN]
    out.loc["MACRO (attacks)"] = [np.nan] + [out.loc[attacks, m].mean() for m in alarms]
    return out
