"""Figures for the model card, from reports/oof_and_test.npz (written by training)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from churn_core.costs import CostModel  # noqa: E402

R = Path(__file__).resolve().parents[1] / "reports"
d = np.load(R / "oof_and_test.npz")
rep = json.loads((R / "training_report.json").read_text())
cm, thr = CostModel(), rep["threshold"]
grid = np.round(np.arange(0.02, 0.98, 0.01), 2)
BLUE, ORANGE, GREY = "#2F5597", "#ED7D31", "#7F7F7F"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
n_tr, n_te = len(d["y_tr"]), len(d["y_te"])
ax[0].plot(grid, [cm.total_cost(d["y_tr"], d["oof"], t) / n_tr for t in grid], color=BLUE,
           label="out-of-fold (training split)")
ax[0].plot(grid, [cm.total_cost(d["y_te"], d["p_te"], t) / n_te for t in grid], color=ORANGE,
           label="held-out test")
for name, v in cm.baselines(d["y_te"]).items():
    ax[0].axhline(v / n_te, color=GREY, ls=":", lw=1)
    ax[0].text(0.97, v / n_te + 0.8, name.replace("_", " "), ha="right", color=GREY, fontsize=8)
ax[0].axvline(thr, color="black", lw=1)
ax[0].text(thr - 0.01, 66.5, f"chosen {thr:.2f}", fontsize=9, ha="right")
ax[0].axvline(cm.analytic_threshold(), color=BLUE, ls="--", lw=1)
ax[0].set_xlabel("decision threshold"); ax[0].set_ylabel("expected cost per customer (USD)")
ax[0].set_title("Expected cost vs threshold"); ax[0].legend(frameon=False, loc="upper center")

fr, mp = calibration_curve(d["y_te"], d["p_te"], n_bins=10, strategy="quantile")
ax[1].plot([0, 1], [0, 1], color=GREY, ls=":"); ax[1].plot(mp, fr, "o-", color=BLUE)
ax[1].set_xlabel("mean predicted probability"); ax[1].set_ylabel("observed churn rate")
ax[1].set_title("Calibration on the held-out test set")
fig.tight_layout(); fig.savefig(R / "threshold_and_calibration.png", dpi=150)
print("saved")
