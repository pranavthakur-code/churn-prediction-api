[README (4).md](https://github.com/user-attachments/files/34675285/README.4.md)
# Churn Prediction API

A scikit-learn churn model packaged as a secure FastAPI service: versioned and traceable model artefact, validated inputs, API-key auth, structured logs, a drift check, a multi-stage Dockerfile and 82 tests.

* Data: IBM's public Telco Customer Churn sample (7,043 rows, in `data/`).
* Result on the held-out test set: ROC-AUC 0.841, recall 0.90 / precision 0.45 at a cost-derived threshold of 0.17. See **[MODEL_CARD.md](MODEL_CARD.md)** for the data caveats, the threshold justification and the subgroup results.
* Retraining and redeploying: **[docs/RETRAINING.md](docs/RETRAINING.md)**. API contract: **[openapi.json](openapi.json)** / `openapi.yaml`.

## Quick start

```bash
pip install -r requirements-dev.txt          # Python 3.13; versions are pinned
python -m training.train                     # ~30 s: tune, pick threshold, write artifacts/<version>/
python -m pytest -q                          # 82 tests

export CHURN_API_KEYS=my-secret-key          # required: the service refuses to start without it
uvicorn --factory app.main:get_app --port 8000
# interactive docs: http://localhost:8000/docs
```

`artifacts/` already contains a trained model (`churn-logistic-16320c9c`), so you can skip the training step.

### Docker
```bash
docker build -t churn-api .
docker run --rm -p 8000:8000 -e CHURN_API_KEYS=my-secret-key churn-api
# roll back to another packaged version without rebuilding:
docker run --rm -p 8000:8000 -e CHURN_API_KEYS=my-secret-key \
  -e CHURN_MODEL_DIR=/app/artifacts/<older-version> churn-api
```
The image is two-stage (`python:3.13-slim`; dependencies are installed in a builder stage, the runtime stage contains only the virtualenv, `app/`, `churn_core/` and `artifacts/`), runs as a non-root user and has a `HEALTHCHECK`.

## Endpoints

| Method & path | Auth | Purpose |
|---|---|---|
| `GET /health` | none | liveness/readiness; shows the loaded `model_version` |
| `POST /predict` | key | score one customer |
| `POST /predict/batch` | key | score 1 – 1,000 customers (`{"customers": [...]}`), order preserved |
| `POST /drift` | key | PSI of 50 – 20,000 recent records vs. the training inputs |
| `GET /model-info` | key | model version, data hash, model hash, metrics, library versions |

Send the key in the `X-API-Key` header.

### Examples
```bash
KEY=my-secret-key

# single prediction (docs/example_request.json holds a full record)
curl -s -X POST localhost:8000/predict -H "X-API-Key: $KEY" \
     -H 'Content-Type: application/json' -d @docs/example_request.json
```
```json
{"prediction": {"customer_id": "7590-VHVEG", "churn_probability": 0.616458,
                "churn_predicted": true, "threshold": 0.17, "warnings": []},
 "model_version": "churn-logistic-16320c9c"}
```

**Unseen category**: scored, with a warning (the label is ignored by the encoder rather than crashing):
```json
{"prediction": {"churn_probability": 0.479786, "churn_predicted": true, "threshold": 0.17,
  "warnings": ["unseen category 'Lifetime plan' for Contract; treated as unknown (the model never saw it in training)"], "...": "..."}}
```
Case and whitespace differences (`"month-to-month"`, `" Female "`) are repaired silently.

**Invalid input** → `422` naming every bad field (types, ranges, missing and unexpected fields; payloads are never echoed back):
```bash
curl -s -X POST localhost:8000/predict -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"tenure": -1, "MonthlyCharges": "cheap"}'
```
```json
{"error": "validation_error", "message": "18 problem(s) found in the request; see 'errors'.",
 "request_id": "e2e4066c922642b9",
 "errors": [{"field": "gender", "message": "Field required", "type": "missing"}, "..."]}
```
In a batch the field path includes the row: `customers.2.tenure`.

**No / wrong key** → `401 {"error": "unauthorized", ...}`. A caller without a key learns nothing about the schema.

**Batch and drift**
```bash
python scripts/make_payloads.py batch 5   > batch.json
python scripts/make_payloads.py drift 200 > drift_ok.json
python scripts/make_payloads.py drift 200 --shift > drift_shifted.json   # +$35 charges, all month-to-month
curl -s -X POST localhost:8000/predict/batch -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -d @batch.json
curl -s -X POST localhost:8000/drift         -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -d @drift_shifted.json
```
The shifted payload returns `"drift_detected": true, "flagged_features": ["Contract", "MonthlyCharges"]` (PSI 3.7 and 3.1); the unshifted one returns `false` (largest PSI 0.026). Use `?psi_threshold=0.1` to change the flag level (default 0.2).

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `CHURN_API_KEYS` | **required** | comma-separated keys; an entry may be `sha256:<hex digest of the key>` |
| `CHURN_MODEL_DIR` | – | model directory to load; otherwise `artifacts/CURRENT` names the default |
| `CHURN_ARTIFACTS_ROOT` | `artifacts` | where `CURRENT` and the version folders live |
| `LOG_LEVEL` | `INFO` | |
| `CHURN_MAX_BATCH` | `1000` | batch size limit |
| `CHURN_DRIFT_MIN_ROWS` / `_MAX_ROWS` | `50` / `20000` | allowed record count for `/drift` |
| `CHURN_DRIFT_PSI_THRESHOLD` | `0.2` | PSI above which a feature is flagged |

## How the requirements are met

| Requirement | Where |
|---|---|
| Pipeline with cleaning, encoding, scaling, classifier | `churn_core/features.py` (`Cleaner`), `churn_core/pipeline.py`; one pickled `Pipeline`, so serving cannot differ from training |
| Cross-validated tuning | `training/train.py`: 5-fold grid search over two model families, one-standard-error selection rule |
| Cost-based threshold | `churn_core/costs.py`; justified in the model card |
| Versioned artefact with data hash, metrics, library versions | `artifacts/<version>/metadata.json`; the loader refuses a model whose SHA-256 does not match |
| Pydantic validation, clear errors | `app/schemas.py`, error handler in `app/main.py` |
| Single, batch, health | `app/main.py` |
| API key auth, structured logging, unseen categories | `app/security.py`, `app/logging_config.py` + request middleware, `app/model_store.py` + `OneHotEncoder(handle_unknown="ignore")` |
| Drift (PSI > 0.2) | `churn_core/drift.py`, `POST /drift` |
| Multi-stage Docker image | `Dockerfile` |
| ≥ 15 tests | `tests/` (82: unit, API integration, artefact, OpenAPI, latency) |
| Median single-prediction latency < 100 ms | measured below |
| OpenAPI spec | `openapi.json`, `openapi.yaml` (a test fails if they go stale) |

## Measured results

* **Tests**: 82 pass, both in the development environment and in a clean virtualenv built only from the pinned `requirements.txt`.
* **Latency**: 1,000 sequential `POST /predict` over real HTTP (keep-alive) against a running uvicorn process, non-root user, runtime files only: **median 15.1 ms, p95 21.7 ms, p99 25.8 ms** (`reports/benchmark_http.txt`; 2-vCPU sandbox, loopback network). A test also asserts median < 100 ms in-process. Reproduce with `python scripts/benchmark.py --key $KEY`.
* **Image size**: the pinned virtualenv is 273 MB on disk (`du`, including pip, which the Dockerfile removes) and the slim Python base is roughly 120 – 130 MB, so the image should land around 400 MB. **This estimate was not verified by a build; see the note below.**

> **Not verified here: the Docker build.** The environment this was developed in could not pull `python:3.13-slim` (Docker Hub was blocked), so `docker build` was never run. The Dockerfile is straightforward, and its inputs were checked: the exact runtime layout (only `app/`, `churn_core/`, `artifacts/` plus the locked virtualenv) was run as a non-root user and passed the benchmark above. Please run `docker build -t churn-api .` and `docker image ls churn-api` once; `.github/workflows/ci.yml` (also never run) contains the size and smoke checks.

## Design notes
* **Fail closed**: no API keys → the app does not start; a model that is missing or fails its hash check → the app does not start.
* **No PII in logs**: one JSON line per request with request id, method, path, status, latency, key fingerprint and record count. Keys and request bodies are never logged. `X-Request-ID` is accepted or generated, and returned.
* **`SeniorCitizen`/`gender` are model inputs**; see the model card before using this with real customers.
* The service is stateless and CPU-bound (`def` endpoints run in FastAPI's thread pool); scale by adding replicas. Rate limiting and TLS belong in the gateway in front of it.

## Layout
```
app/            FastAPI service (main, schemas, security, settings, logging, model loading)
churn_core/     code shared by training and serving (features, pipeline, costs, drift)
training/       train.py: tune → threshold → versioned artefact
artifacts/      <version>/{model.joblib, metadata.json} and CURRENT
scripts/        export_openapi, benchmark, make_payloads, make_figures, subgroup_report
tests/          82 tests            reports/   training report, figures, benchmark, subgroups
docs/           RETRAINING.md, example_request.json
```
