"""Quality scorers for request/candidate pairs."""

from abc import ABC, abstractmethod
import math

from .candidates import ModelCandidate
from .features import RoutingFeatures


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 30.0), -30.0)))


class QualityScorer(ABC):
    @abstractmethod
    def predict_success(
        self,
        features: RoutingFeatures,
        candidate: ModelCandidate,
    ) -> tuple[float, dict[str, float]]:
        """Return success probability and an explanation contribution map."""


class HeuristicQualityScorer(QualityScorer):
    """Transparent cold-start scorer until outcome-labelled data exists.

    This is deliberately a replaceable baseline, not a claim that hand-tuned
    routing is the final model. Its feature/candidate interface is the same one
    a learned LightGBM or logistic scorer can implement later.
    """

    def _task_demand(self, features: RoutingFeatures) -> tuple[float, dict]:
        values = features.values
        complexity = (
            0.30 * min(values.get("code.structural.cognitive_complexity", 0) / 40, 1)
            + 0.20 * min(values.get("code.structural.cyclomatic_complexity", 0) / 25, 1)
            + 0.15 * min(values.get("code.structural.max_nesting_depth", 0) / 8, 1)
            + 0.15 * min(values.get("code.data_flow.max_dataflow_chain_depth", 0) / 12, 1)
            + 0.10 * min(values.get("code.structural.max_local_call_chain_depth", 0) / 8, 1)
            + 0.10 * min(values.get("code.structural.sloc", 0) / 250, 1)
        )
        uncertainty = (
            0.35 * values.get("query.operation.ambiguous", 0)
            + 0.35 * values.get("query.domain.ambiguous", 0)
            + 0.15 * (1.0 - values.get("query.operation.margin", 0))
            + 0.15 * (1.0 - values.get("query.domain.margin", 0))
        )
        risk = max(
            values.get("query.concerns.scores.security", 0.0),
            values.get("query.concerns.scores.correctness", 0.0),
            values.get("query.concerns.scores.reliability", 0.0),
            values.get("query.concerns.scores.concurrency", 0.0),
        )
        malformed = max(
            min(values.get("code.data_flow.unresolved_call_count", 0) / 20, 1),
            values.get("code.data_flow.unresolved_flow_ratio", 0.0) * 0.5,
        )
        operation_weight = {
            "explain": 0.20,
            "generate": 0.45,
            "refactor": 0.55,
            "optimize": 0.60,
            "review": 0.65,
            "debug": 0.75,
        }.get(features.query_operation, 0.45)
        demand = min(
            1.0,
            0.28 * complexity
            + 0.20 * uncertainty
            + 0.20 * risk
            + 0.12 * malformed
            + 0.20 * operation_weight,
        )
        return demand, {
            "task_complexity": complexity,
            "signal_uncertainty": uncertainty,
            "risk": risk,
            "malformed_input": malformed,
            "operation_demand": operation_weight,
        }

    def predict_success(
        self,
        features: RoutingFeatures,
        candidate: ModelCandidate,
    ) -> tuple[float, dict[str, float]]:
        demand, evidence = self._task_demand(features)
        capability = (
            0.42 * candidate.quality_tier
            + 0.33 * candidate.reasoning_capability
            + 0.25 * (
                candidate.code_capability if features.requires_code else 0.75
            )
        )
        context_fit = min(
            candidate.context_window
            / max(features.estimated_input_tokens, 1),
            1.0,
        )
        compatibility_penalty = 0.0
        if features.requires_code and not candidate.supports_code:
            compatibility_penalty += 2.5
        if features.requires_tools and not candidate.supports_tools:
            compatibility_penalty += 0.35
        if context_fit < 1.0:
            compatibility_penalty += 3.0 * (1.0 - context_fit)

        logit = (
            -0.10
            + 4.2 * capability
            - 2.8 * demand
            + 0.8 * context_fit
            - compatibility_penalty
        )
        probability = _sigmoid(logit)
        explanation = {
            **evidence,
            "candidate_capability": round(capability, 4),
            "context_fit": round(context_fit, 4),
            "compatibility_penalty": round(compatibility_penalty, 4),
            "task_demand": round(demand, 4),
        }
        return round(probability, 4), explanation
