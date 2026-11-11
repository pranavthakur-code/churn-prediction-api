"""FastAPI application: churn prediction service."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import logging_config
from app.model_store import LoadedModel, resolve_model_dir
from app.schemas import (BatchRequest, BatchResponse, Customer, DriftResponse,
                         ErrorResponse, Health, ModelInfo, PredictionResponse)
from app.security import require_api_key
from app.settings import Settings

TITLE = "Churn Prediction API"
DESCRIPTION = """
Scores telecom customers for churn risk so the CRM can act in real time.

* Every request except `/health` needs an `X-API-Key` header.
* `churn_predicted` compares the probability with a **cost-based threshold**
  (see the model card), not a default 0.5.
* Unknown category labels are accepted and flagged in `warnings`; malformed
  values (wrong type, out of range, missing or unexpected fields) get a `422`
  that names each offending field.
"""
ERRORS = {401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
          422: {"model": ErrorResponse, "description": "Invalid request body"}}


def _error(status: int, error: str, message: str, request: Request, errors=None):
    body = {"error": error, "message": message,
            "request_id": getattr(request.state, "request_id", None),
            "errors": errors or []}
    return JSONResponse(body, status_code=status)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    log = logging_config.configure(settings.log_level)
    if not settings.api_keys:
        # fail closed: an unauthenticated prediction API is never the intent
        raise RuntimeError("CHURN_API_KEYS is empty; refusing to start without API keys")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = resolve_model_dir(settings.model_dir, settings.artifacts_root)
        app.state.model = LoadedModel.load(path)
        log.info("model loaded", extra={"model_version": app.state.model.version,
                                        "model_dir": str(path)})
        yield

    app = FastAPI(title=TITLE, description=DESCRIPTION, version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.model = None

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.key_id = None
        request.state.n_records = None
        t0 = time.perf_counter()
        status, response = 500, None
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            ms = (time.perf_counter() - t0) * 1000
            extra = {"request_id": request.state.request_id, "method": request.method,
                     "path": request.url.path, "status": status,
                     "latency_ms": round(ms, 2), "key_id": request.state.key_id}
            if request.state.n_records is not None:
                extra["n_records"] = request.state.n_records
            m = app.state.model
            if m is not None:
                extra["model_version"] = m.version
            log.info("request", extra=extra)
            if response is not None:
                response.headers["X-Request-ID"] = request.state.request_id
                response.headers["X-Process-Time-ms"] = f"{ms:.2f}"

    # ------------------------------------------------------- error handling
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        items = []
        for e in exc.errors():
            loc = [str(p) for p in e["loc"] if p != "body"]
            items.append({"field": ".".join(loc) or "body", "message": e["msg"],
                          "type": e["type"]})
        return _error(422, "validation_error",
                      f"{len(items)} problem(s) found in the request; see 'errors'.",
                      request, items)

    @app.exception_handler(StarletteHTTPException)   # also covers FastAPI's HTTPException and unknown routes
    async def http_handler(request: Request, exc: StarletteHTTPException):
        code = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}.get(
            exc.status_code, "error")
        r = _error(exc.status_code, code, str(exc.detail), request)
        if exc.status_code == 401:
            r.headers["WWW-Authenticate"] = "ApiKey"
        return r

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error", extra={"request_id": request.state.request_id})
        return _error(500, "internal_error", "Unexpected server error.", request)

    def model() -> LoadedModel:
        m = app.state.model
        if m is None:
            raise HTTPException(503, "Model is not loaded.")
        return m

    # --------------------------------------------------------------- routes
    @app.get("/health", response_model=Health, tags=["ops"], summary="Liveness and readiness")
    def health():
        m = app.state.model
        return {"status": "ok" if m else "unavailable", "model_loaded": m is not None,
                "model_version": m.version if m else None}

    @app.get("/model-info", response_model=ModelInfo, responses=ERRORS, tags=["ops"],
             summary="Which model is serving, and how it was trained",
             dependencies=[Depends(require_api_key)])
    def model_info(m: LoadedModel = Depends(model)):
        return m.info()

    @app.post("/predict", response_model=PredictionResponse, responses=ERRORS,
              tags=["prediction"], summary="Score one customer",
              dependencies=[Depends(require_api_key)])
    def predict(customer: Customer, request: Request, m: LoadedModel = Depends(model)):
        request.state.n_records = 1
        out = m.predict([customer.model_dump()])[0]
        return {"prediction": out, "model_version": m.version}

    @app.post("/predict/batch", response_model=BatchResponse, responses=ERRORS,
              tags=["prediction"], summary="Score up to CHURN_MAX_BATCH customers",
              dependencies=[Depends(require_api_key)])
    def predict_batch(body: BatchRequest, request: Request, m: LoadedModel = Depends(model)):
        n = len(body.customers)
        if n > settings.max_batch_size:
            raise HTTPException(
                422, f"Batch has {n} customers; the maximum is {settings.max_batch_size}.")
        request.state.n_records = n
        preds = m.predict([c.model_dump() for c in body.customers])
        return {"predictions": preds, "count": n, "model_version": m.version}

    @app.post("/drift", response_model=DriftResponse, responses=ERRORS, tags=["monitoring"],
              summary="Population Stability Index of recent inputs vs training data",
              dependencies=[Depends(require_api_key)])
    def drift(body: BatchRequest, request: Request, m: LoadedModel = Depends(model),
              psi_threshold: float | None = None):
        n = len(body.customers)
        if n < settings.drift_min_rows:
            raise HTTPException(422, f"Drift needs at least {settings.drift_min_rows} records "
                                     f"(got {n}); PSI on fewer rows is mostly noise.")
        if n > settings.drift_max_rows:
            raise HTTPException(422, f"At most {settings.drift_max_rows} records per drift call.")
        request.state.n_records = n
        thr = settings.drift_psi_threshold if psi_threshold is None else psi_threshold
        res = m.drift([c.model_dump() for c in body.customers], thr)
        return {"model_version": m.version, **res}

    return app


def get_app() -> FastAPI:   # uvicorn --factory app.main:get_app
    return create_app()
