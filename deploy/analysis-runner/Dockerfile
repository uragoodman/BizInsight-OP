FROM python:3.13-slim
RUN pip install --no-cache-dir pandas==2.3.3 polars==1.33.1 numpy==2.3.3 scipy==1.16.2 scikit-learn==1.7.2 statsmodels==0.14.5 matplotlib==3.10.6 plotly==6.3.0 openpyxl==3.1.5 pyarrow==21.0.0
RUN useradd --create-home --uid 10001 runner
RUN mkdir -p /work/input /work/candidate /work/output && chown -R runner:runner /work
COPY runner.py /runner.py
USER runner
ENTRYPOINT ["python", "/runner.py"]
