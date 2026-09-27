# NetForecast — offline network attack forecasting.
# Everything (model, dashboard assets, sample capture) is baked into the image;
# the running container makes no outbound network calls.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch: the model is ~88K parameters and inference is ~1 ms on CPU,
# so a CUDA wheel would add gigabytes for no benefit.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch>=2.2 \
 && pip install -r requirements.txt

COPY netforecast/ ./netforecast/
COPY server/ ./server/
COPY tools/ ./tools/
COPY models/ ./models/
COPY models_real/ ./models_real/
COPY data/sample_flows.csv ./data/sample_flows.csv
COPY pyproject.toml README.md ./

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health').read()"

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
