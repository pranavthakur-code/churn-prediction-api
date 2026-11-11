# ---------- stage 1: build the virtualenv (compilers/caches never reach the final image)
FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install -r requirements.txt \
 && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
 && rm -rf /opt/venv/lib/python3.13/site-packages/pip* /opt/venv/bin/pip*

# ---------- stage 2: runtime
FROM python:3.13-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    CHURN_ARTIFACTS_ROOT=/app/artifacts
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin churn
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY churn_core ./churn_core
COPY app ./app
# the model: every released version stays in the image; artifacts/CURRENT picks the default,
# CHURN_MODEL_DIR overrides it (this is how you roll back without rebuilding)
COPY artifacts ./artifacts
USER churn
EXPOSE 8000
# CHURN_API_KEYS must be supplied at run time (the app refuses to start without it)
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)"
CMD ["uvicorn", "--factory", "app.main:get_app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
