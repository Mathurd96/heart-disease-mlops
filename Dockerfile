# ---------- Heart Disease Risk API ----------
# Slim serving image: only the inference dependencies (no MLflow / plotting).
# The trained model (models/model.joblib + model_metadata.json) is baked in,
# so the container is fully self-contained and immutable.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/model.joblib \
    METADATA_PATH=/app/models/model_metadata.json \
    PORT=8000

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/ ./src/
COPY models/ ./models/

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen(f'http://localhost:{os.getenv(\"PORT\",\"8000\")}/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn heart.api.app:app --host 0.0.0.0 --port ${PORT}"]
