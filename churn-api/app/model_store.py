"""Loads a versioned artefact, verifies it, and turns records into predictions."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from churn_core.drift import compute_drift
from churn_core.features import CATEGORICAL, FEATURES

log = logging.getLogger("churn_api")


class ModelLoadError(RuntimeError):
    pass


def resolve_model_dir(model_dir: Path | None, artifacts_root: Path) -> Path:
    if model_dir:
        return Path(model_dir)
    pointer = Path(artifacts_root) / "CURRENT"
    if not pointer.exists():
        raise ModelLoadError(f"no CHURN_MODEL_DIR set and {pointer} does not exist")
    return Path(artifacts_root) / pointer.read_text().strip()


@dataclass
class LoadedModel:
    pipeline: object
    meta: dict
    canonical: dict          # category -> {casefolded label -> canonical label}

    @property
    def version(self) -> str:
        return self.meta["model_version"]

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, model_dir: Path) -> "LoadedModel":
        mp, meta_p = Path(model_dir) / "model.joblib", Path(model_dir) / "metadata.json"
        if not mp.exists() or not meta_p.exists():
            raise ModelLoadError(f"{model_dir} must contain model.joblib and metadata.json")
        meta = json.loads(meta_p.read_text())
        digest = hashlib.sha256(mp.read_bytes()).hexdigest()
        if digest != meta["model_sha256"]:
            # joblib files are pickles: never load one that does not match its record
            raise ModelLoadError("model.joblib does not match the sha256 in metadata.json")
        trained_with = meta["libraries"]["scikit-learn"]
        if trained_with != sklearn.__version__:
            log.warning("scikit-learn version differs from training",
                        extra={"trained_with": trained_with, "running": sklearn.__version__})
        pipeline = joblib.load(mp)
        canonical = {c: {v.casefold(): v for v in vals}
                     for c, vals in meta["categories"].items()}
        return cls(pipeline, meta, canonical)

    # ----------------------------------------------------------------- predict
    def _frame(self, records: list[dict]) -> tuple[pd.DataFrame, list[list[str]]]:
        """Build the model input and collect a warning list per record."""
        rows, warnings = [], []
        for rec in records:
            row, warn = {}, []
            for f in FEATURES:
                v = rec.get(f)
                if f in CATEGORICAL:
                    v = str(v).strip() if v is not None else None
                    known = self.canonical[f]
                    if v is not None and v not in known.values():
                        fixed = known.get(v.casefold())       # 'yes' -> 'Yes'
                        if fixed is not None:
                            v = fixed
                        else:
                            warn.append(f"unseen category {v!r} for {f}; treated as "
                                        f"unknown (the model never saw it in training)")
                row[f] = v
            rows.append(row)
            warnings.append(warn)
        return pd.DataFrame(rows, columns=FEATURES), warnings

    def predict(self, records: list[dict]) -> list[dict]:
        frame, warnings = self._frame(records)
        proba = self.pipeline.predict_proba(frame)[:, 1]
        return [{"customer_id": r.get("customerID"),
                 "churn_probability": round(float(p), 6),
                 "churn_predicted": bool(p >= self.threshold),
                 "threshold": self.threshold, "warnings": w}
                for r, p, w in zip(records, proba, warnings)]

    # ------------------------------------------------------------------- drift
    def drift(self, records: list[dict], threshold: float) -> dict:
        frame, _ = self._frame(records)
        # categories are compared as sent (after case repair); unseen ones show up in PSI
        return compute_drift(self.meta["drift_reference"], frame, threshold)

    def info(self) -> dict:
        m = self.meta
        return {"model_version": m["model_version"], "model_family": m["model_family"],
                "threshold": m["threshold"], "trained_at": m["trained_at"],
                "training_data_sha256": m["training_data"]["sha256"],
                "model_sha256": m["model_sha256"], "cost_model": m["cost_model"],
                "test_metrics": m["metrics"]["test"], "libraries": m["libraries"]}
