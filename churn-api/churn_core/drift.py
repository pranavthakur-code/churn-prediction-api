"""Population Stability Index (PSI) drift check against a training reference.

PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i) over bins.
Common reading: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from churn_core.features import CATEGORICAL, NUMERIC

EPS = 1e-4          # floor for bin proportions so ln() is defined
UNSEEN = "__unseen__"


def build_reference(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """Summarise the training inputs: decile edges for numbers, category shares."""
    ref = {"n_rows": int(len(df)), "numeric": {}, "categorical": {}}
    for c in NUMERIC:
        x = df[c].dropna().to_numpy(dtype=float)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
        ref["numeric"][c] = {"edges": edges.tolist(),
                             "proportions": _bin_props(x, edges).tolist()}
    for c in CATEGORICAL:
        vc = df[c].dropna().astype(str).value_counts(normalize=True)
        ref["categorical"][c] = {k: float(v) for k, v in vc.items()}
    return ref


def _bin_props(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, x, side="left")      # bins are (e_{i-1}, e_i]
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    return "stable" if value < 0.1 else "moderate" if value <= 0.2 else "significant"


def compute_drift(ref: dict, recent: pd.DataFrame, threshold: float = 0.2) -> dict:
    """PSI per feature for `recent` against the stored reference."""
    features = {}
    for c, r in ref["numeric"].items():
        x = pd.to_numeric(recent[c], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) == 0:
            continue
        val = psi(np.array(r["proportions"]), _bin_props(x, np.array(r["edges"])))
        features[c] = val
    for c, shares in ref["categorical"].items():
        vals = recent[c].dropna().astype(str)
        if len(vals) == 0:
            continue
        cats = list(shares) + [UNSEEN]
        exp = np.array([shares[k] for k in shares] + [0.0])
        counts = vals.where(vals.isin(shares), UNSEEN).value_counts()
        act = np.array([counts.get(k, 0) for k in cats], dtype=float) / len(vals)
        features[c] = psi(exp, act)
    detail = {k: {"psi": round(v, 4), "status": status(v), "flagged": v > threshold}
              for k, v in features.items()}
    flagged = sorted(k for k, v in detail.items() if v["flagged"])
    return {"threshold": threshold, "n_rows": int(len(recent)),
            "drift_detected": bool(flagged), "flagged_features": flagged,
            "features": detail}
