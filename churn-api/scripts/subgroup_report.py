"""Held-out performance by customer group (gender, senior citizen, contract, tenure band).

Re-creates the exact train/test split used in training (same seed) and scores the
shipped artefact. Output: reports/subgroups.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.train import load_data  # noqa: E402

d = ROOT / "artifacts" / (ROOT / "artifacts" / "CURRENT").read_text().strip()
meta = json.loads((d / "metadata.json").read_text())
X, y = load_data(ROOT / "data" / "Telco-Customer-Churn.csv")
_, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=meta["seed"])
p = joblib.load(d / "model.joblib").predict_proba(X_te)[:, 1]
thr = meta["threshold"]
tenure = pd.to_numeric(X_te["tenure"])
groups = {"gender": X_te["gender"], "SeniorCitizen": X_te["SeniorCitizen"],
          "Contract": X_te["Contract"],
          "tenure_band": pd.cut(tenure, [-1, 12, 24, 48, 100], labels=["0-12", "13-24", "25-48", "49+"])}
rows = []
for g, s in groups.items():
    for v in sorted(s.astype(str).unique()):
        m = (s.astype(str) == v).to_numpy()
        yy, pp = y_te.to_numpy()[m], p[m]
        rows.append({"group": g, "value": v, "n": int(m.sum()),
                     "churn_rate": round(float(yy.mean()), 3),
                     "auc": round(float(roc_auc_score(yy, pp)), 3) if len(set(yy)) > 1 else None,
                     "recall_at_threshold": round(float(recall_score(yy, pp >= thr)), 3),
                     "flagged_share": round(float((pp >= thr).mean()), 3)})
(ROOT / "reports" / "subgroups.json").write_text(json.dumps(rows, indent=1))
print(pd.DataFrame(rows).to_string(index=False))
