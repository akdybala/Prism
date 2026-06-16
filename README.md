# Prism

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

## Core Packages

- `code_signals/`: code parsing, static analysis, and code-domain extraction.
- `query_signals/`: query operation/domain/concern classifiers and examples.

## Status

This is an in-progress student ML systems project. The goal is to build a
signal-aware model router for code-assistance tasks.
