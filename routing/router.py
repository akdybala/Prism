"""Cost-aware model selection over predicted candidate success."""

from dataclasses import asdict, dataclass

from code_signals import extract_all
from query_signals import extract_query_signals

from .candidates import ModelCandidate, load_candidate_registry
from .features import RoutingFeatures, build_routing_features
from .scorers import HeuristicQualityScorer, QualityScorer


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    display_name: str
    success_probability: float
    estimated_cost: float
    estimated_latency_ms: int
    eligible: bool
    context_fits: bool
    evidence: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RoutingDecision:
    selected_candidate: str
    selected_display_name: str
    quality_threshold: float
    selection_reason: str
    fallback_used: bool
    features: RoutingFeatures
    candidates: tuple[CandidateScore, ...]

    def to_dict(self) -> dict:
        return {
            "selected_candidate": self.selected_candidate,
            "selected_display_name": self.selected_display_name,
            "quality_threshold": self.quality_threshold,
            "selection_reason": self.selection_reason,
            "fallback_used": self.fallback_used,
            "features": self.features.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _estimated_cost(
    candidate: ModelCandidate,
    input_tokens: int,
    expected_output_tokens: int,
) -> float:
    return (
        input_tokens * candidate.input_cost_per_million
        + expected_output_tokens * candidate.output_cost_per_million
    ) / 1_000_000.0


def route_from_signals(
    query_signals: dict,
    code_signals: dict | None = None,
    *,
    query: str | None = None,
    code: str | None = None,
    context_tokens: int = 0,
    expected_output_tokens: int = 1200,
    quality_threshold: float = 0.75,
    candidates: list[ModelCandidate] | None = None,
    scorer: QualityScorer | None = None,
) -> RoutingDecision:
    if not 0.0 < quality_threshold <= 1.0:
        raise ValueError("quality_threshold must be in (0, 1]")
    candidates = candidates or load_candidate_registry()
    if not candidates:
        raise ValueError("At least one enabled candidate is required")
    scorer = scorer or HeuristicQualityScorer()
    features = build_routing_features(
        query_signals,
        code_signals,
        query=query,
        code=code,
        context_tokens=context_tokens,
    )

    scored = []
    for candidate in candidates:
        probability, evidence = scorer.predict_success(features, candidate)
        context_fits = (
            features.estimated_input_tokens <= candidate.context_window
        )
        eligible = probability >= quality_threshold and context_fits
        scored.append(
            CandidateScore(
                candidate_id=candidate.candidate_id,
                display_name=candidate.display_name,
                success_probability=probability,
                estimated_cost=round(
                    _estimated_cost(
                        candidate,
                        features.estimated_input_tokens,
                        expected_output_tokens,
                    ),
                    6,
                ),
                estimated_latency_ms=candidate.estimated_latency_ms,
                eligible=eligible,
                context_fits=context_fits,
                evidence=evidence,
            )
        )

    eligible = [score for score in scored if score.eligible]
    fallback_used = not eligible
    if eligible:
        selected = min(
            eligible,
            key=lambda score: (
                score.estimated_cost,
                score.estimated_latency_ms,
                -score.success_probability,
            ),
        )
        reason = "cheapest_candidate_above_quality_threshold"
    else:
        context_compatible = [score for score in scored if score.context_fits]
        pool = context_compatible or scored
        selected = max(
            pool,
            key=lambda score: (
                score.success_probability,
                -score.estimated_cost,
            ),
        )
        reason = "highest_predicted_quality_fallback"

    return RoutingDecision(
        selected_candidate=selected.candidate_id,
        selected_display_name=selected.display_name,
        quality_threshold=quality_threshold,
        selection_reason=reason,
        fallback_used=fallback_used,
        features=features,
        candidates=tuple(
            sorted(
                scored,
                key=lambda score: score.success_probability,
                reverse=True,
            )
        ),
    )


def route_request(
    query: str,
    code: str | None = None,
    **kwargs,
) -> RoutingDecision:
    query_result = extract_query_signals(query)
    code_result = extract_all(code) if code and code.strip() else None
    return route_from_signals(
        query_result,
        code_result,
        query=query,
        code=code,
        **kwargs,
    )
