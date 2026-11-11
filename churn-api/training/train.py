"""Train, tune, choose a threshold and write a versioned model artefact.

    python -m training.train --data data/Telco-Customer-Churn.csv

Writes artifacts/<version>/{model.joblib, metadata.json} and
reports/training_report.json. Everything is seeded, so re-running on the same
data and library versions reproduces the same artefact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)

from churn_core.costs import CostModel
from churn_core.drift import build_reference
from churn_core.features import CATEGORICAL, FEATURES, ID_COLUMN, TARGET, Cleaner
from churn_core.pipeline import SEED, build_pipeline

GRIDS = {
    "logistic": {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
    "hgb": {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3, 4],
            "clf__min_samples_leaf": [20, 60], "clf__l2_regularization": [0.0, 1.0]},
}
FAST_GRIDS = {"logistic": {"clf__C": [0.1, 1.0]},
              "hgb": {"clf__max_depth": [2, 3]}}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("duplicate customerID values in training data")
    y = df[TARGET].str.strip().map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Churn must be 'Yes' or 'No' for every row")
    return df[FEATURES + [ID_COLUMN]], y.astype(int)


def metrics_at(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"threshold": round(float(thr), 4),
            "precision": round(float(precision_score(y, pred)), 4),
            "recall": round(float(recall_score(y, pred)), 4),
            "f1": round(float(f1_score(y, pred)), 4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap_ci(y, p, fn, n=1000, seed=SEED) -> list:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def run(data_path: Path, out_root: Path, fast: bool = False,
        costs: CostModel | None = None, report_dir: Path | None = None) -> Path:
    costs = costs or CostModel()
    data_path = Path(data_path)
    X, y = load_data(data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    grids = FAST_GRIDS if fast else GRIDS

    # 1. cross-validated search, one grid per model family
    searches = {}
    for fam, grid in grids.items():
        gs = GridSearchCV(build_pipeline(fam), grid, scoring="roc_auc", cv=cv,
                          n_jobs=2, refit=True)
        gs.fit(X_tr, y_tr)
        k = gs.best_index_
        searches[fam] = {"search": gs, "cv_auc": float(gs.best_score_),
                         "cv_std": float(gs.cv_results_["std_test_score"][k]),
                         "params": {a: (float(b) if isinstance(b, (np.floating, float)) else b)
                                    for a, b in gs.best_params_.items()}}
        print(f"[{fam}] best CV AUC {gs.best_score_:.4f} "
              f"+/- {searches[fam]['cv_std']:.4f}  {gs.best_params_}")

    # 2. one-standard-error rule: prefer the simpler logistic model unless
    #    boosting is better by more than one standard error of the CV score
    lr, hgb = searches["logistic"], searches.get("hgb")
    se = lr["cv_std"] / np.sqrt(cv.get_n_splits())
    family = "hgb" if hgb and hgb["cv_auc"] > lr["cv_auc"] + se else "logistic"
    best = searches[family]
    model = best["search"].best_estimator_
    print(f"selected family: {family} (1-SE margin {se:.4f})")

    # 3. threshold from OUT-OF-FOLD predictions on the training split only
    oof = cross_val_predict(build_pipeline(family).set_params(
        **best["search"].best_params_), X_tr, y_tr, cv=cv,
        method="predict_proba")[:, 1]
    thr, oof_cost = costs.best_threshold(y_tr, oof)
    print(f"threshold {thr:.2f} (analytic {costs.analytic_threshold():.2f})")

    # 4. one look at the sealed test set
    p_te = model.predict_proba(X_te)[:, 1]
    n_te = len(y_te)
    test_cost = costs.total_cost(y_te, p_te, thr)
    base = costs.baselines(y_te)
    frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=8, strategy="quantile")
    sens = []
    for L, O, s in [(300, 30, .5), (300, 15, .5), (300, 60, .5), (600, 30, .5),
                    (150, 30, .5), (300, 30, .3), (300, 30, .8)]:
        cm = CostModel(L, O, s)
        t_s, _ = cm.best_threshold(y_tr, oof)
        sens.append({"customer_value": L, "offer_cost": O, "save_rate": s,
                     "analytic_threshold": round(cm.analytic_threshold(), 3),
                     "empirical_threshold": t_s,
                     "test_cost_per_customer": round(cm.total_cost(y_te, p_te, t_s) / n_te, 2),
                     "no_action_per_customer": round(cm.baselines(y_te)["no_action"] / n_te, 2)})
    test = {
        "n": int(n_te), "churn_rate": round(float(y_te.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
        "roc_auc_ci95": bootstrap_ci(y_te, p_te, roc_auc_score),
        "brier": round(float(brier_score_loss(y_te, p_te)), 4),
        "at_threshold": metrics_at(y_te, p_te, thr),
        "at_0.5": metrics_at(y_te, p_te, 0.5),
        "cost_per_customer": round(test_cost / n_te, 2),
        "cost_no_action_per_customer": round(base["no_action"] / n_te, 2),
        "cost_offer_everyone_per_customer": round(base["offer_to_everyone"] / n_te, 2),
        "calibration": {"mean_predicted": [round(float(v), 3) for v in mean_pred],
                        "fraction_positive": [round(float(v), 3) for v in frac_pos]},
    }
    cv_summary = {f: {"best_cv_auc": round(s["cv_auc"], 4), "cv_std": round(s["cv_std"], 4),
                      "best_params": s["params"]} for f, s in searches.items()}
    print(json.dumps({k: test[k] for k in ("roc_auc", "roc_auc_ci95", "brier",
                                            "at_threshold", "cost_per_customer",
                                            "cost_no_action_per_customer")}, indent=1))

    # 5. refit on train+test? No: ship the model that was evaluated.
    #    Reference distribution for drift comes from the training split.
    cleaned_train = Cleaner().transform(X_tr)
    reference = build_reference(cleaned_train)
    categories = {c: sorted(cleaned_train[c].dropna().unique().tolist())
                  for c in CATEGORICAL}

    # 6. write the versioned artefact
    data_sha = sha256_file(data_path)
    version = f"churn-{family}-{data_sha[:8]}"
    out_dir = Path(out_root) / version
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path, compress=3)
    meta = {
        "model_version": version, "model_family": family,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_sha256": sha256_file(model_path),
        "training_data": {"file": data_path.name, "sha256": data_sha,
                          "n_rows_total": int(len(X)), "n_rows_train": int(len(X_tr)),
                          "n_rows_test": int(len(X_te)),
                          "churn_rate_train": round(float(y_tr.mean()), 4)},
        "seed": SEED, "features": FEATURES, "categories": categories,
        "threshold": thr, "cost_model": costs.as_dict(),
        "hyperparameters": best["params"],
        "metrics": {"cv": cv_summary, "selected_cv_auc": round(best["cv_auc"], 4),
                    "oof_cost_per_customer": round(oof_cost / len(y_tr), 2),
                    "test": test},
        "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "numpy": np.__version__, "pandas": pd.__version__,
                      "scipy": scipy.__version__, "joblib": joblib.__version__},
        "drift_reference": reference,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    Path(out_root, "CURRENT").write_text(version + "\n")
    if report_dir:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        (Path(report_dir) / "training_report.json").write_text(json.dumps(
            {"model_version": version, "cv": cv_summary, "selected": family,
             "one_se_margin": round(float(se), 5), "threshold": thr,
             "analytic_threshold": costs.analytic_threshold(), "test": test,
             "cost_sensitivity": sens}, indent=2))
        np.savez(Path(report_dir) / "oof_and_test.npz", y_tr=y_tr.to_numpy(), oof=oof,
                 y_te=y_te.to_numpy(), p_te=p_te)
    print(f"wrote {out_dir}")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Telco-Customer-Churn.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--fast", action="store_true", help="tiny grids (for tests)")
    a = ap.parse_args(argv)
    run(Path(a.data), Path(a.out), fast=a.fast, report_dir=Path(a.reports))


if __name__ == "__main__":
    sys.exit(main())
