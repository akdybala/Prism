"""Outcome records used to train a learned quality scorer later."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RoutingOutcome:
    request_id: str
    candidate_id: str
    features: dict[str, float]
    candidate_features: dict[str, Any]
    success: bool
    quality_score: float | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    accepted_by_user: bool | None = None
    failure_reason: str | None = None

    def __post_init__(self):
        if not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if self.quality_score is not None and not 0 <= self.quality_score <= 1:
            raise ValueError("quality_score must be between 0 and 1")
        for name in ("latency_ms", "input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.cost is not None and self.cost < 0:
            raise ValueError("cost cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)


def append_outcome(
    outcome: RoutingOutcome,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(outcome.to_dict(), ensure_ascii=False) + "\n")
