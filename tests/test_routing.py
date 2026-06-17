import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from routing.candidates import ModelCandidate, load_candidate_registry
from routing.features import build_routing_features
from routing.outcomes import RoutingOutcome, append_outcome
from routing.router import route_from_signals, route_request
from routing.scorers import QualityScorer


def query_signals(operation="debug", ambiguous=False):
    return {
        "query_operation": {
            "predicted": operation,
            "confidence": 0.82,
            "margin": 0.55,
            "ambiguous": ambiguous,
            "scores": {operation: 0.82, "explain": 0.18},
        },
        "query_domain": {
            "predicted": "algorithms",
            "confidence": 0.70,
            "margin": 0.42,
            "ambiguous": ambiguous,
            "domain_signal_present": True,
            "scores": {"algorithms": 0.70, "general": 0.28},
        },
        "query_concerns": {
            "scores": {
                "security": 0.05,
                "concurrency": 0.10,
                "correctness": 0.75,
                "performance": 0.20,
                "reliability": 0.10,
                "maintainability": 0.05,
            },
            "thresholds": {
                "security": 0.6,
                "concurrency": 0.5,
                "correctness": 0.4,
                "performance": 0.6,
                "reliability": 0.6,
                "maintainability": 0.45,
            },
        },
    }


def code_signals():
    return {
        "structural": {
            "cognitive_complexity": 18,
            "cyclomatic_complexity": 9,
            "max_nesting_depth": 4,
            "sloc": 80,
            "num_function_defs": 5,
            "num_classes": 2,
            "has_recursion": True,
            "has_concurrency_primitives": False,
            "has_networking": False,
            "has_io": False,
            "has_subprocesses": False,
            "has_broad_exception": False,
            "max_local_call_chain_depth": 3,
        },
        "data_flow": {
            "max_dataflow_chain_depth": 5,
            "unresolved_flow_ratio": 0.2,
        },
        "semantic": {"call_diversity": 9},
        "domain": {
            "predicted_domain": "algorithms",
            "confidence": 0.78,
            "rule_scores": {
                "algorithms": 0.60,
                "systems_programming": 0.40,
            },
            "embedding_scores": {
                "algorithms": 0.78,
                "systems_programming": 0.22,
            },
        },
        "has_errors": False,
        "error_count": 0,
    }


def candidate(candidate_id, cost, context_window=100_000):
    return ModelCandidate(
        candidate_id=candidate_id,
        display_name=candidate_id,
        quality_tier=0.7,
        code_capability=0.8,
        reasoning_capability=0.8,
        context_window=context_window,
        supports_tools=True,
        supports_code=True,
        input_cost_per_million=cost,
        output_cost_per_million=cost,
        estimated_latency_ms=int(cost * 1000),
    )


class FixedScorer(QualityScorer):
    def __init__(self, values):
        self.values = values

    def predict_success(self, features, candidate):
        return self.values[candidate.candidate_id], {"fixed": 1.0}


class RoutingTests(unittest.TestCase):
    def test_registry_loads_unique_enabled_candidates(self):
        candidates = load_candidate_registry()
        self.assertGreaterEqual(len(candidates), 3)
        self.assertEqual(
            len({candidate.candidate_id for candidate in candidates}),
            len(candidates),
        )

    def test_feature_assembly_preserves_vectors_and_complexity(self):
        features = build_routing_features(
            query_signals(),
            code_signals(),
            query="Fix the incorrect recursive result.",
            code="def solve():\n    return recurse()",
            context_tokens=500,
        )
        self.assertEqual(features.query_operation, "debug")
        self.assertEqual(features.query_domain, "algorithms")
        self.assertEqual(features.code_domain, "algorithms")
        self.assertTrue(features.requires_code)
        self.assertTrue(features.requires_tools)
        self.assertGreater(
            features.values["code.structural.cognitive_complexity"],
            0,
        )
        self.assertEqual(
            features.values["query.concerns.scores.correctness"],
            0.75,
        )
        self.assertGreaterEqual(features.estimated_input_tokens, 500)

    def test_full_signal_vector_includes_all_signal_sections(self):
        features = build_routing_features(
            query_signals(),
            code_signals(),
            query="Fix the result.",
            code="def solve(): return 1",
        )
        expected = {
            "query.operation.scores.debug",
            "query.operation.confidence",
            "query.operation.margin",
            "query.domain.scores.algorithms",
            "query.domain.domain_signal_present",
            "query.concerns.scores.security",
            "code.structural.cognitive_complexity",
            "code.structural.has_recursion",
            "code.structural.num_classes",
            "code.data_flow.max_dataflow_chain_depth",
            "code.semantic.call_diversity",
            "code.domain.rule_scores.algorithms",
            "code.domain.embedding_scores.algorithms",
        }
        self.assertTrue(expected.issubset(features.values))

    def test_vector_contains_only_approved_signal_panels(self):
        query = query_signals()
        query["query_concerns"].update({
            "evidence": {"security": ["unsafe input"]},
            "method": "embedding",
            "detected": ["correctness"],
        })
        code = code_signals()
        code["domain"]["embedding_sketch"] = "def hidden(): pass"
        code["domain"]["embedding_representation"] = "tree_sitter_v7"
        features = build_routing_features(
            query,
            code,
            query="raw query text",
            code="raw code text",
        )
        allowed_prefixes = (
            "query.operation.",
            "query.concerns.scores.",
            "query.domain.",
            "code.structural.",
            "code.data_flow.",
            "code.semantic.",
            "code.domain.rule_scores.",
            "code.domain.embedding_scores.",
        )
        self.assertTrue(
            all(
                name.startswith(allowed_prefixes)
                for name in features.values
            )
        )
        forbidden = (
            "sketch",
            "hash",
            "evidence",
            "threshold",
            "method",
            "representation",
            "has_errors",
            "error_count",
            "request.",
            "derived.",
        )
        self.assertFalse(
            any(
                token in name
                for name in features.values
                for token in forbidden
            )
        )

    def test_selects_cheapest_candidate_above_threshold(self):
        candidates = [candidate("cheap", 0.2), candidate("expensive", 2.0)]
        decision = route_from_signals(
            query_signals(),
            candidates=candidates,
            scorer=FixedScorer({"cheap": 0.81, "expensive": 0.95}),
            quality_threshold=0.8,
        )
        self.assertEqual(decision.selected_candidate, "cheap")
        self.assertFalse(decision.fallback_used)
        self.assertEqual(
            decision.selection_reason,
            "cheapest_candidate_above_quality_threshold",
        )

    def test_falls_back_to_highest_quality_when_threshold_is_unmet(self):
        candidates = [candidate("cheap", 0.2), candidate("strong", 2.0)]
        decision = route_from_signals(
            query_signals(),
            candidates=candidates,
            scorer=FixedScorer({"cheap": 0.50, "strong": 0.72}),
            quality_threshold=0.8,
        )
        self.assertEqual(decision.selected_candidate, "strong")
        self.assertTrue(decision.fallback_used)

    def test_context_overflow_disqualifies_otherwise_eligible_candidate(self):
        candidates = [
            candidate("short", 0.1, context_window=100),
            candidate("long", 1.0, context_window=10_000),
        ]
        decision = route_from_signals(
            query_signals(),
            context_tokens=1000,
            candidates=candidates,
            scorer=FixedScorer({"short": 0.99, "long": 0.85}),
            quality_threshold=0.8,
        )
        self.assertEqual(decision.selected_candidate, "long")
        short = next(
            item for item in decision.candidates
            if item.candidate_id == "short"
        )
        self.assertFalse(short.context_fits)
        self.assertFalse(short.eligible)

    def test_route_request_composes_existing_extractors(self):
        with patch(
            "routing.router.extract_query_signals",
            return_value=query_signals(),
        ) as query_extract:
            with patch(
                "routing.router.extract_all",
                return_value=code_signals(),
            ) as code_extract:
                decision = route_request(
                    "Why is this wrong?",
                    "def solve(): pass",
                    candidates=[candidate("only", 1.0)],
                    scorer=FixedScorer({"only": 0.9}),
                )
        query_extract.assert_called_once_with("Why is this wrong?")
        code_extract.assert_called_once_with("def solve(): pass")
        self.assertEqual(decision.selected_candidate, "only")

    def test_outcome_logger_writes_training_ready_jsonl(self):
        outcome = RoutingOutcome(
            request_id="request-1",
            candidate_id="balanced",
            features={"query.correctness": 0.8},
            candidate_features={"context_window": 128000},
            success=True,
            quality_score=0.9,
            latency_ms=1200,
            input_tokens=500,
            output_tokens=200,
            cost=0.004,
            accepted_by_user=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcomes.jsonl"
            append_outcome(outcome, path)
            record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["request_id"], "request-1")
        self.assertTrue(record["success"])


if __name__ == "__main__":
    unittest.main()
