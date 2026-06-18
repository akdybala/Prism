---
title: Prism Router
emoji: 🔀
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Prism

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/akdybala/Prism?quickstart=1)

Prism is a proof-of-concept signal extraction system for Python code and
natural-language developer requests.

This first batch contains the foundation:

- static Python code signals built on Tree-sitter parsing
- structural, data-flow, semantic, and code-domain features
- query operation, query-domain, and multilabel concern classifiers
- curated examples and holdout data for signal evaluation

Later commits add the audited `(code, query)` datasets, routing feature layer,
LightGBM router, and local inspection UI.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python -m unittest tests.test_signals tests.test_domain tests.test_data_flow tests.test_domain_sketch tests.test_query_signals -v
```

## Try It In GitHub Codespaces

Click the **Open in GitHub Codespaces** badge above, create the codespace, and
wait for dependency installation to finish. The dev container starts the Prism
Signal Inspector on port `8000`; Codespaces opens the forwarded web interface
automatically.

The first code or query classification request downloads the configured
Hugging Face embedding models. This can take a few minutes. Later requests reuse
the caches stored in the codespace.

If the browser does not open automatically:

1. Open the **Ports** panel in Codespaces.
2. Find **Prism Signal Inspector** on port `8000`.
3. Select **Open in Browser**.

To restart the UI manually:

```bash
bash .devcontainer/start-ui.sh
```

## Host The Public Demo

Prism includes a Docker configuration for deployment as a Hugging Face Space.
The hosted container starts the Signal Inspector on port `7860`, includes the
trained LightGBM router, and downloads embedding models lazily.

See [HUGGINGFACE_SPACE.md](HUGGINGFACE_SPACE.md) for deployment instructions.

## Core Packages

- `code_signals/`: code parsing, static analysis, and code-domain extraction.
- `query_signals/`: query operation/domain/concern classifiers and examples.
