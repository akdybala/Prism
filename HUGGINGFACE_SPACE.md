# Deploy Prism To Hugging Face Spaces

Prism is prepared for a Docker-based Hugging Face Space.

## Create The Space

1. Sign in to Hugging Face.
2. Create a new Space.
3. Choose **Docker** as the Space SDK.
4. Choose the free CPU hardware for the first demo.
5. Upload or push this repository to the Space.

The root `README.md` contains the Space metadata and declares port `7860`.
Hugging Face builds the root `Dockerfile` automatically.

## Runtime Behavior

The container starts the UI immediately without eagerly loading embedding
models:

```text
python signal_ui.py --host 0.0.0.0 --port 7860 --no-browser --skip-warmup
```

The trained LightGBM router model is included in the image. Hugging Face
embedding models download lazily when code or query signal extraction first
needs them. The first such request can therefore take several minutes on free
CPU hardware.

Health check:

```text
GET /api/health
```

## Optional Persistent Cache

Free Spaces do not guarantee that downloaded model caches survive a container
rebuild or restart. If persistent storage is enabled, set:

```text
HF_HOME=/data/huggingface
TRANSFORMERS_CACHE=/data/huggingface/transformers
```

as Space variables.

## Local Docker Test

```bash
docker build -t prism-router .
docker run --rm -p 7860:7860 prism-router
```

Then open:

```text
http://localhost:7860
```
