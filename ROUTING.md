# Routing Layer

The routing package scores each `(request, candidate)` pair and then applies a
separate cost-aware policy.

## Flow

```text
query + optional code
        |
        v
existing signal extractors
        |
        v
complete sparse numeric routing feature vector
        |
        v
P(success | request features, candidate features)
        |
        v
cheapest candidate above the quality threshold
```

The current `HeuristicQualityScorer` is a transparent cold-start baseline.
It exists so routing can be exercised before model-outcome data is available.
It should later be replaced by a learned scorer, preferably LightGBM.

## Minimum-Capable Tier PoC

The next experimental layer trains a separate LightGBM classifier that predicts
the minimum model tier required for a `(query, code)` request:

- `light`
- `medium`
- `heavy`

This classifier is trained on the trusted-v1 LLM-grounded code-query corpus and
the approved 109-feature signal vector. It is exposed in `signal_ui.py` for
local inspection.

```powershell
python build_router_training_data.py
python train_lightgbm_router.py
python signal_ui.py
```

The current test split contains 264 examples and reports 82.20% accuracy and
0.6840 macro F1. Heavy-tier recall remains weak at 30%, so this is a research
artifact rather than a production routing policy.

The tier classifier and `HeuristicQualityScorer` solve different problems. The
tier classifier predicts a coarse minimum capability class from rubric labels.
The candidate scorer estimates whether each configured model can succeed and
then applies cost-aware selection. They should remain separate until
candidate-outcome labels are available.

## Approved Signal Vector

Feature schema `3-approved-signal-panels` contains only:

- every structural signal
- every data-flow signal
- every semantic signal
- the complete code rule-domain vector
- the complete code embedding-domain vector
- the query-operation score vector and displayed confidence/ambiguity values
- the query-concern score vector
- the query-domain score vector and displayed confidence/ambiguity values

It explicitly excludes sketches, source text, text hashes, regex evidence,
concern thresholds, representation metadata, parse metadata, request-size
features, and duplicate derived aliases.

Input-token estimates remain routing metadata for context-window and cost
checks, but are not part of the model feature vector.

## CLI

```powershell
python route_request.py "Why does this recursive function return the wrong result?"
python route_request.py "Review this implementation" --code-file solution.py
python route_request.py "Apply this repository specification" `
  --context-tokens 50000 --quality-threshold 0.85
```

## Candidate Registry

Edit `routing/model_candidates.json` to represent the actual models exposed by
the client. Prices are illustrative placeholders until provider adapters supply
current cost and latency information.

Each candidate declares:

- quality, code, and reasoning capability priors
- context window
- code and tool support
- input/output cost
- estimated latency

## Learned Scorer Data

One request should produce one training row per attempted candidate:

```json
{
  "request_id": "task-123",
  "candidate_id": "balanced_code_model",
  "features": {},
  "success": 1,
  "quality_score": 0.88,
  "latency_ms": 1320,
  "input_tokens": 4200,
  "output_tokens": 980,
  "cost": 0.0064
}
```

The first learned target should be `success`. Keep observed quality, latency,
cost, retries, and user acceptance as separate labels for later policies.

Use `routing.append_outcome()` with a `RoutingOutcome` to append these rows to
a JSONL dataset without coupling telemetry storage to the router.

Do not directly train a fixed model-name label. Score candidates independently
so new candidates can be evaluated and policy thresholds can change without
retraining the whole routing system.
