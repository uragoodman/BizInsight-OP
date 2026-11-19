FROM node:22-alpine AS web-build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.13-slim

ARG PYTHON_PACKAGE_INDEX=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url ${PYTHON_PACKAGE_INDEX} -r requirements.txt
COPY scripts/download_torch_cpu_wheel.py /tmp/download_torch_cpu_wheel.py
RUN python /tmp/download_torch_cpu_wheel.py --output /tmp/torch-2.12.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl \
    && pip install --no-cache-dir --index-url ${PYTHON_PACKAGE_INDEX} /tmp/torch-2.12.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl \
    && rm -f /tmp/torch-2.12.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl /tmp/download_torch_cpu_wheel.py
COPY requirements-rag-qdrant-runtime.txt ./
RUN pip install --no-cache-dir --index-url ${PYTHON_PACKAGE_INDEX} -r requirements-rag-qdrant-runtime.txt
COPY requirements-mem0.txt ./
RUN pip install --no-cache-dir --index-url ${PYTHON_PACKAGE_INDEX} -r requirements-mem0.txt
COPY requirements-rag-progressive.txt ./
COPY requirements-rag-progressive-constraints.txt ./
RUN pip install --no-cache-dir --index-url ${PYTHON_PACKAGE_INDEX} \
    -r requirements-rag-progressive.txt -c requirements-rag-progressive-constraints.txt
COPY . ./
RUN python scripts/download_reranker_model.py
COPY --from=web-build /web/dist ./web/dist
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
