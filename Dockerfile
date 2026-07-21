FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY vendor/mllm-scripts vendor/mllm-scripts
COPY backend backend
COPY demo_output demo_output

WORKDIR /app/backend

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/backend
ENV DOC_EXTRACT_JOB_ROOT=/tmp/doc-extract-web
ENV DOC_EXTRACT_CRED_MODE=personal_only
ENV DOC_EXTRACT_DATA_DIR=/var/data
ENV DOC_EXTRACT_USERS_DB=/var/data/users.sqlite

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
