"""Step 5: combine Stage 1 (AE score) and Stage 2 (LightGBM probabilities) into one verdict.

Verdicts:
  BENIGN           nothing suspicious
  KNOWN_ATTACK     LightGBM confidently names an attack family it was trained on
  UNKNOWN_ANOMALY  looks abnormal to the AE, but LightGBM cannot confidently name it

Two ways to combine them:

  CASCADE   AE first, as a gate.          AE normal? -> BENIGN (LightGBM never asked)
                                          AE abnormal -> LightGBM names it -> KNOWN, else UNKNOWN
  PARALLEL  both look at every flow.      LightGBM confident attack -> KNOWN
                                          else AE abnormal -> UNKNOWN
                                          else BENIGN

Every threshold is chosen on VALIDATION data so that at most `budget` (2%) of
benign flows raise an alarm. All methods are compared at that same budget.
"""
import numpy as np

BENIGN, KNOWN, UNKNOWN = "BENIGN", "KNOWN_ATTACK", "UNKNOWN_ANOMALY"


def lgbm_top(proba: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Most likely family and its probability, for every flow."""
    idx = proba.argmax(axis=1)
    return np.asarray(classes)[idx], proba[np.arange(len(proba)), idx]


def lgbm_confident_attack(proba, classes, tau: float) -> np.ndarray:
    fam, conf = lgbm_top(proba, classes)
    return (fam != BENIGN) & (conf >= tau)


def threshold_at_budget(benign_scores: np.ndarray, budget: float,
                        already_flagged: np.ndarray | None = None) -> float:
    """Return t so that (already flagged) + (score > t) is at most `budget` of benign flows.

    `already_flagged` = benign flows another component has alarmed on; they use
    up part of the budget first, and the rest goes to this score.
    """
    n = len(benign_scores)
    pre = np.zeros(n, bool) if already_flagged is None else already_flagged
    allowed = int(np.floor(budget * n)) - int(pre.sum())
    if allowed <= 0:
        return float("inf")                       # budget already spent: never fire
    rest = np.sort(benign_scores[~pre])[::-1]     # highest scores first
    if allowed >= len(rest):
        return float("-inf")
    return float(rest[allowed])                   # exactly `allowed` scores lie above it


def fit_thresholds(ae_err, proba, classes, y, budget: float, tau: float) -> dict:
    """Choose every threshold on validation data. y = true Family (only BENIGN rows are used)."""
    b = y == BENIGN
    lgbm_flag = lgbm_confident_attack(proba[b], classes, tau)
    p_benign = proba[:, list(classes).index(BENIGN)]
    th = {
        "budget": budget,
        "tau": tau,
        "n_benign_val": int(b.sum()),
        # AE alone, and the cascade gate: AE gets the whole budget.
        "ae": threshold_at_budget(ae_err[b], budget),
        # Parallel: LightGBM's confident alarms on benign flows are paid first, AE gets the rest.
        "ae_parallel": threshold_at_budget(ae_err[b], budget, already_flagged=lgbm_flag),
        "lgbm_benign_false_alarms": int(lgbm_flag.sum()),
        # LightGBM alone, given the same budget: alarm when P(attack) = 1 - P(BENIGN) is high.
        "lgbm_attack_score": threshold_at_budget(1 - p_benign[b], budget),
    }
    return th


def decide(ae_err, proba, classes, th: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (verdict, family) per flow. family = LightGBM's attack name for KNOWN_ATTACK."""
    fam, _ = lgbm_top(proba, classes)
    confident = lgbm_confident_attack(proba, classes, th["tau"])
    verdict = np.full(len(ae_err), BENIGN, dtype=object)

    if mode == "cascade":
        gate = ae_err > th["ae"]
        verdict[gate & confident] = KNOWN
        verdict[gate & ~confident] = UNKNOWN
    elif mode == "parallel":
        verdict[confident] = KNOWN
        verdict[~confident & (ae_err > th["ae_parallel"])] = UNKNOWN
    else:
        raise ValueError(f"unknown mode {mode!r}")

    family = np.where(verdict == KNOWN, fam, np.where(verdict == UNKNOWN, "?", BENIGN))
    return verdict, family


def baseline_alarm(ae_err, proba, classes, th: dict, method: str) -> np.ndarray:
    """Single-component detectors at the same budget (True = alarm)."""
    if method == "ae_only":
        return ae_err > th["ae"]
    if method == "lgbm_only":
        p_benign = proba[:, list(classes).index(BENIGN)]
        return (1 - p_benign) > th["lgbm_attack_score"]
    raise ValueError(method)
