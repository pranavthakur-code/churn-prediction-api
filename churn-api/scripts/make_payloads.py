"""Build example request bodies from the training CSV (for README / manual testing).

    python scripts/make_payloads.py drift 200 [--shift]  > drift.json
    python scripts/make_payloads.py batch 5               > batch.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"


def main(kind, n, shift):
    df = pd.read_csv(CSV, dtype=str, keep_default_na=False).sample(n, random_state=7)
    recs = df.drop(columns=["Churn"]).to_dict("records")
    for r in recs:
        r["SeniorCitizen"], r["tenure"] = int(r["SeniorCitizen"]), int(r["tenure"])
        r["MonthlyCharges"] = float(r["MonthlyCharges"])
        if shift:                                  # simulate a pricing/product change
            r["MonthlyCharges"] = round(r["MonthlyCharges"] + 35, 2)
            r["Contract"] = "Month-to-month"
    json.dump({"customers": recs}, sys.stdout)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], int(a[1]), "--shift" in sys.argv)
