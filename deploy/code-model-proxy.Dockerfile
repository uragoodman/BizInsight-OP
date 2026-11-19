FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY deploy/code-model-proxy-requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && (id -u proxy >/dev/null 2>&1 || useradd --create-home --uid 10001 proxy)
COPY app ./app

USER 10001:10001
CMD ["uvicorn", "app.code_analysis.model_proxy:app", "--host", "0.0.0.0", "--port", "8091"]
