FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface/transformers

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 user

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.0.0" \
    && python -m pip install -r requirements.txt

COPY code_signals/ code_signals/
COPY query_signals/ query_signals/
COPY routing/ routing/
COPY domain_examples.json signal_ui.py ./
COPY router_training_data/trusted_v1/lightgbm_router/lightgbm_router.joblib \
    router_training_data/trusted_v1/lightgbm_router/lightgbm_router.joblib

RUN mkdir -p /tmp/huggingface \
    && chown -R user:user /app /tmp/huggingface

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=3)"

CMD ["python", "signal_ui.py", "--host", "0.0.0.0", "--port", "7860", "--no-browser", "--skip-warmup"]
