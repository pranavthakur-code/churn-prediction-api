import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.model_store import LoadedModel, ModelLoadError
from app.settings import Settings
from tests.conftest import API_KEY


def test_metadata_has_everything_needed_for_traceability(meta, small_data):
    assert meta["training_data"]["sha256"] == hashlib.sha256(small_data.read_bytes()).hexdigest()
    assert {"scikit-learn", "numpy", "pandas", "python"} <= set(meta["libraries"])
    assert meta["metrics"]["test"]["roc_auc"] > 0.7
    assert 0 < meta["threshold"] < 1
    assert meta["model_version"].endswith(meta["training_data"]["sha256"][:8])
    assert meta["cost_model"]["analytic_threshold"] == pytest.approx(0.2)


def test_artifact_files_match_recorded_hash(artifact_dir, meta):
    digest = hashlib.sha256((artifact_dir / "model.joblib").read_bytes()).hexdigest()
    assert digest == meta["model_sha256"]


def test_tampered_model_is_refused(artifact_dir, tmp_path):
    for f in ("model.joblib", "metadata.json"):
        (tmp_path / f).write_bytes((artifact_dir / f).read_bytes())
    with open(tmp_path / "model.joblib", "ab") as fh:
        fh.write(b"x")
    with pytest.raises(ModelLoadError, match="sha256"):
        LoadedModel.load(tmp_path)


def test_missing_model_files_give_clear_error(tmp_path):
    with pytest.raises(ModelLoadError, match="model.joblib"):
        LoadedModel.load(tmp_path)


def test_app_refuses_to_start_without_api_keys(artifact_dir):
    with pytest.raises(RuntimeError, match="CHURN_API_KEYS"):
        create_app(Settings(model_dir=artifact_dir, api_keys=()))


def test_app_fails_at_startup_if_model_is_bad(tmp_path):
    app = create_app(Settings(model_dir=tmp_path, api_keys=(API_KEY,)))
    with pytest.raises(ModelLoadError):
        with TestClient(app):
            pass


def test_training_is_deterministic(small_data, tmp_path, meta):
    from training.train import run
    d = run(small_data, tmp_path, fast=True)
    m2 = json.loads((d / "metadata.json").read_text())
    assert m2["metrics"]["test"]["roc_auc"] == meta["metrics"]["test"]["roc_auc"]
    assert m2["threshold"] == meta["threshold"]


def test_bad_training_data_is_rejected(tmp_path, raw_df):
    from training.train import load_data
    bad = raw_df.head(20).copy()
    bad.loc[0, "Churn"] = "Maybe"
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Churn"):
        load_data(p)
