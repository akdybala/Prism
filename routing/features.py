"""Build the strict routing vector from user-approved signal panels only."""

from dataclasses import dataclass
import math


ROUTING_FEATURE_SCHEMA_VERSION = "3-approved-signal-panels"


def _number(value) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else 0.0
    return 0.0


def _estimate_tokens(text: str | None) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _copy_numeric_mapping(
    output: dict[str, float],
    prefix: str,
    values: dict | None,
) -> None:
    for name, value in sorted((values or {}).items()):
        if isinstance(value, (bool, int, float)):
            output[f"{prefix}.{name}"] = _number(value)


def _copy_prediction_panel(
    output: dict[str, float],
    prefix: str,
    panel: dict,
    *,
    include_domain_evidence: bool = False,
) -> None:
    _copy_numeric_mapping(output, f"{prefix}.scores", panel.get("scores"))
    for name in (
        "confidence",
        "secondary_confidence",
        "margin",
        "ambiguous",
    ):
        if name in panel:
            output[f"{prefix}.{name}"] = _number(panel[name])
    if include_domain_evidence and "domain_signal_present" in panel:
        output[f"{prefix}.domain_signal_present"] = _number(
            panel["domain_signal_present"]
        )
    for reason in panel.get("ambiguity_reasons", []):
        output[f"{prefix}.ambiguity_reason.{reason}"] = 1.0


@dataclass(frozen=True)
class RoutingFeatures:
    values: dict[str, float]
    query_operation: str
    query_domain: str
    code_domain: str
    estimated_input_tokens: int
    requires_code: bool
    requires_tools: bool

    def to_dict(self) -> dict:
        return {
            "values": dict(self.values),
            "feature_count": len(self.values),
            "schema_version": ROUTING_FEATURE_SCHEMA_VERSION,
            "query_operation": self.query_operation,
            "query_domain": self.query_domain,
            "code_domain": self.code_domain,
            "estimated_input_tokens": self.estimated_input_tokens,
            "requires_code": self.requires_code,
            "requires_tools": self.requires_tools,
        }


def build_routing_features(
    query_signals: dict,
    code_signals: dict | None = None,
    *,
    query: str | None = None,
    code: str | None = None,
    context_tokens: int = 0,
) -> RoutingFeatures:
    values: dict[str, float] = {}

    operation = query_signals.get("query_operation", {})
    concerns = query_signals.get("query_concerns", {})
    query_domain = query_signals.get("query_domain", {})

    _copy_prediction_panel(values, "query.operation", operation)
    _copy_numeric_mapping(
        values,
        "query.concerns.scores",
        concerns.get("scores"),
    )
    _copy_prediction_panel(
        values,
        "query.domain",
        query_domain,
        include_domain_evidence=True,
    )

    code_domain = {}
    if code_signals is not None:
        _copy_numeric_mapping(
            values,
            "code.structural",
            code_signals.get("structural"),
        )
        _copy_numeric_mapping(
            values,
            "code.data_flow",
            code_signals.get("data_flow"),
        )
        _copy_numeric_mapping(
            values,
            "code.semantic",
            code_signals.get("semantic"),
        )
        code_domain = code_signals.get("domain", {})
        _copy_numeric_mapping(
            values,
            "code.domain.rule_scores",
            code_domain.get("rule_scores"),
        )
        _copy_numeric_mapping(
            values,
            "code.domain.embedding_scores",
            code_domain.get("embedding_scores"),
        )

    predicted_operation = operation.get("predicted", "unknown")
    requires_tools = predicted_operation in {
        "debug",
        "review",
        "refactor",
        "generate",
        "optimize",
    } and code_signals is not None
    return RoutingFeatures(
        values=values,
        query_operation=predicted_operation,
        query_domain=query_domain.get("predicted", "general"),
        code_domain=code_domain.get("predicted_domain", "unknown"),
        estimated_input_tokens=(
            _estimate_tokens(query)
            + _estimate_tokens(code)
            + max(int(context_tokens), 0)
        ),
        requires_code=code_signals is not None,
        requires_tools=requires_tools,
    )
