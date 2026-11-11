import json
import logging
import statistics
import time

import pytest

from tests.conftest import API_KEY, HEADERS


# ------------------------------------------------------------------ health
def test_health_needs_no_key_and_reports_model(client, meta):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True,
                        "model_version": meta["model_version"]}


# -------------------------------------------------------------------- auth
@pytest.mark.parametrize("path,method", [("/predict", "post"), ("/predict/batch", "post"),
                                         ("/drift", "post"), ("/model-info", "get")])
def test_endpoints_require_key(client, customer, path, method):
    r = getattr(client, method)(path, **({"json": customer} if method == "post" else {}))
    assert r.status_code == 401 and r.json()["error"] == "unauthorized"
    assert r.headers["www-authenticate"] == "ApiKey"


def test_wrong_key_rejected(client, customer):
    r = client.post("/predict", json=customer, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_unauthenticated_request_learns_nothing_about_schema(client):
    r = client.post("/predict", json={"bad": 1})
    assert r.status_code == 401 and r.json()["errors"] == []


def test_hashed_key_entry_is_supported(artifact_dir, customer):
    import hashlib
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.settings import Settings
    entry = "sha256:" + hashlib.sha256(b"secret-abc").hexdigest()
    with TestClient(create_app(Settings(model_dir=artifact_dir, api_keys=(entry,)))) as c:
        assert c.post("/predict", json=customer, headers={"X-API-Key": "secret-abc"}).status_code == 200
        assert c.post("/predict", json=customer, headers={"X-API-Key": entry}).status_code == 401


# ----------------------------------------------------------------- predict
def test_predict_returns_probability_and_decision(client, customer, meta):
    r = client.post("/predict", json=customer, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    p = body["prediction"]
    assert 0 <= p["churn_probability"] <= 1
    assert p["churn_predicted"] == (p["churn_probability"] >= meta["threshold"])
    assert p["customer_id"] == "7590-VHVEG" and p["warnings"] == []
    assert body["model_version"] == meta["model_version"]


def test_risky_profile_scores_higher_than_loyal_profile(client, customer):
    risky = dict(customer, Contract="Month-to-month", tenure=2, InternetService="Fiber optic",
                 PaymentMethod="Electronic check", MonthlyCharges=95.0, TotalCharges=190.0)
    loyal = dict(customer, Contract="Two year", tenure=70, InternetService="DSL",
                 PaymentMethod="Bank transfer (automatic)", MonthlyCharges=45.0,
                 TotalCharges=3150.0)
    pr = client.post("/predict", json=risky, headers=HEADERS).json()["prediction"]
    pl = client.post("/predict", json=loyal, headers=HEADERS).json()["prediction"]
    assert pr["churn_probability"] > pl["churn_probability"] + 0.2


@pytest.mark.parametrize("total", [None, "", "  ", "29.85"])
def test_total_charges_may_be_null_blank_or_text(client, customer, total):
    customer["TotalCharges"] = total
    assert client.post("/predict", json=customer, headers=HEADERS).status_code == 200


def test_total_charges_may_be_omitted(client, customer):
    del customer["TotalCharges"], customer["customerID"]
    r = client.post("/predict", json=customer, headers=HEADERS)
    assert r.status_code == 200 and r.json()["prediction"]["customer_id"] is None


# --------------------------------------------------------------- validation
@pytest.mark.parametrize("field,value", [
    ("tenure", -1), ("tenure", "abc"), ("tenure", 3.5), ("tenure", 500),
    ("MonthlyCharges", -5), ("MonthlyCharges", "cheap"), ("MonthlyCharges", 1e9),
    ("SeniorCitizen", 2), ("SeniorCitizen", "yes"), ("gender", ""), ("gender", None),
    ("TotalCharges", -1), ("TotalCharges", "abc")])
def test_invalid_values_get_422_naming_the_field(client, customer, field, value):
    customer[field] = value
    r = client.post("/predict", json=customer, headers=HEADERS)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error"
    assert [e["field"] for e in body["errors"]] == [field]
    assert body["errors"][0]["message"]


def test_missing_and_unexpected_fields_are_reported_together(client, customer):
    del customer["Contract"]
    customer["tenuree"] = 3
    errs = {e["field"]: e["type"] for e in client.post(
        "/predict", json=customer, headers=HEADERS).json()["errors"]}
    assert errs == {"Contract": "missing", "tenuree": "extra_forbidden"}


def test_malformed_json_is_a_422_not_a_500(client):
    r = client.post("/predict", content=b"{not json", headers={**HEADERS,
                    "Content-Type": "application/json"})
    assert r.status_code == 422


def test_nan_is_rejected(client, customer):
    r = client.post("/predict", content=json.dumps(customer).replace("29.85", "NaN", 1),
                    headers={**HEADERS, "Content-Type": "application/json"})
    assert r.status_code == 422


# -------------------------------------------------------- unseen categories
def test_unseen_category_is_scored_with_a_warning(client, customer):
    customer["Contract"] = "Lifetime plan"
    r = client.post("/predict", json=customer, headers=HEADERS)
    assert r.status_code == 200
    w = r.json()["prediction"]["warnings"]
    assert len(w) == 1 and "Lifetime plan" in w[0] and "Contract" in w[0]


def test_case_differences_are_repaired_not_flagged(client, customer):
    lower = dict(customer, Partner="yes", Contract="month-to-month", gender=" female ")
    a = client.post("/predict", json=lower, headers=HEADERS).json()["prediction"]
    b = client.post("/predict", json=customer, headers=HEADERS).json()["prediction"]
    assert a["warnings"] == [] and a["churn_probability"] == b["churn_probability"]


# -------------------------------------------------------------------- batch
def test_batch_preserves_order_and_matches_single(client, customer):
    a = dict(customer, customerID="A", tenure=1, Contract="Month-to-month")
    b = dict(customer, customerID="B", tenure=70, Contract="Two year")
    r = client.post("/predict/batch", json={"customers": [a, b, a]}, headers=HEADERS)
    assert r.status_code == 200
    out = r.json()
    assert out["count"] == 3 and [p["customer_id"] for p in out["predictions"]] == ["A", "B", "A"]
    single = client.post("/predict", json=a, headers=HEADERS).json()["prediction"]
    assert out["predictions"][0]["churn_probability"] == single["churn_probability"]
    assert out["predictions"][0]["churn_probability"] == out["predictions"][2]["churn_probability"]


def test_batch_error_points_to_the_bad_row(client, customer):
    bad = dict(customer, tenure=-3)
    r = client.post("/predict/batch", json={"customers": [customer, customer, bad]},
                    headers=HEADERS)
    assert r.status_code == 422
    assert r.json()["errors"][0]["field"] == "customers.2.tenure"


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}, headers=HEADERS).status_code == 422


def test_oversized_batch_rejected(client, customer):
    r = client.post("/predict/batch", json={"customers": [customer] * 51}, headers=HEADERS)
    assert r.status_code == 422 and "maximum is 50" in r.json()["message"]


# ------------------------------------------------------------------- drift
def _records(raw_df, n, **over):
    rows = raw_df.sample(n, random_state=3).drop(columns=["Churn"]).to_dict("records")
    out = []
    for r in rows:
        r["SeniorCitizen"] = int(r["SeniorCitizen"]); r["tenure"] = int(r["tenure"])
        r["MonthlyCharges"] = float(r["MonthlyCharges"]); r.update(over)
        out.append(r)
    return out


def test_drift_quiet_on_training_like_data(client, raw_df):
    r = client.post("/drift", json={"customers": _records(raw_df, 400)}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["drift_detected"] is False and body["threshold"] == 0.2
    assert body["features"]["tenure"]["status"] in {"stable", "moderate"}


def test_drift_flags_shifted_data(client, raw_df):
    recs = _records(raw_df, 400, Contract="Month-to-month", MonthlyCharges=110.0)
    body = client.post("/drift", json={"customers": recs}, headers=HEADERS).json()
    assert {"Contract", "MonthlyCharges"} <= set(body["flagged_features"])


def test_drift_needs_enough_rows(client, raw_df):
    r = client.post("/drift", json={"customers": _records(raw_df, 10)}, headers=HEADERS)
    assert r.status_code == 422 and "at least 50" in r.json()["message"]


def test_drift_threshold_can_be_overridden(client, raw_df):
    recs = _records(raw_df, 400, MonthlyCharges=110.0)
    body = client.post("/drift?psi_threshold=100", json={"customers": recs},
                       headers=HEADERS).json()
    assert body["drift_detected"] is False and body["threshold"] == 100


# ----------------------------------------------------- ops: logging, info
def test_model_info_is_traceable(client, meta):
    body = client.get("/model-info", headers=HEADERS).json()
    assert body["training_data_sha256"] == meta["training_data"]["sha256"]
    assert body["model_sha256"] == meta["model_sha256"]
    assert body["libraries"]["scikit-learn"]


def test_request_id_and_latency_headers(client, customer):
    r = client.post("/predict", json=customer, headers={**HEADERS, "X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123" and float(r.headers["x-process-time-ms"]) >= 0
    assert client.get("/health").headers["x-request-id"]


def test_structured_log_has_latency_and_never_the_key_or_payload(client, customer, caplog):
    with caplog.at_level(logging.INFO, logger="churn_api"):
        client.post("/predict", json=customer, headers=HEADERS)
    rec = [r for r in caplog.records if r.getMessage() == "request"][-1]
    assert rec.path == "/predict" and rec.status == 200 and rec.latency_ms > 0
    assert rec.n_records == 1 and rec.key_id and rec.request_id
    text = " ".join(str(v) for v in rec.__dict__.values())
    assert API_KEY not in text and "7590-VHVEG" not in text and "Electronic check" not in text


def test_json_log_format(client, caplog):
    from app.logging_config import JsonFormatter
    with caplog.at_level(logging.INFO, logger="churn_api"):
        client.get("/health")
    rec = [r for r in caplog.records if r.getMessage() == "request"][-1]
    parsed = json.loads(JsonFormatter().format(rec))
    assert {"ts", "level", "request_id", "latency_ms", "status", "path"} <= set(parsed)
    assert parsed["event"] == "request"


def test_unknown_route_is_a_json_404(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_unhandled_error_hides_details(client, customer, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internals")
    monkeypatch.setattr(client.app.state.model, "predict", boom)
    c = type(client)(client.app, raise_server_exceptions=False)
    r = c.post("/predict", json=customer, headers=HEADERS)
    assert r.status_code == 500 and "secret" not in r.text and r.json()["error"] == "internal_error"


# ----------------------------------------------------------------- latency
def test_median_single_prediction_latency_under_100ms(client, customer):
    for _ in range(20):                                   # warm-up
        client.post("/predict", json=customer, headers=HEADERS)
    times = []
    for _ in range(200):
        t = time.perf_counter()
        assert client.post("/predict", json=customer, headers=HEADERS).status_code == 200
        times.append((time.perf_counter() - t) * 1000)
    assert statistics.median(times) < 100, statistics.median(times)
