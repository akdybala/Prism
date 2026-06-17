"""Model-agnostic routing primitives built on extracted query and code signals."""

from .candidates import ModelCandidate, load_candidate_registry
from .features import (
    ROUTING_FEATURE_SCHEMA_VERSION,
    RoutingFeatures,
    build_routing_features,
)
from .outcomes import RoutingOutcome, append_outcome
from .router import RoutingDecision, route_request
from .scorers import HeuristicQualityScorer, QualityScorer

__all__ = [
    "HeuristicQualityScorer",
    "ModelCandidate",
    "QualityScorer",
    "RoutingDecision",
    "ROUTING_FEATURE_SCHEMA_VERSION",
    "RoutingFeatures",
    "RoutingOutcome",
    "append_outcome",
    "build_routing_features",
    "load_candidate_registry",
    "route_request",
]
