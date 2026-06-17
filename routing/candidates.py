"""Candidate model metadata used by the routing policy."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "model_candidates.json"


@dataclass(frozen=True)
class ModelCandidate:
    candidate_id: str
    display_name: str
    quality_tier: float
    code_capability: float
    reasoning_capability: float
    context_window: int
    supports_tools: bool
    supports_code: bool
    input_cost_per_million: float
    output_cost_per_million: float
    estimated_latency_ms: int
    enabled: bool = True

    @classmethod
    def from_dict(cls, value: dict) -> "ModelCandidate":
        candidate = cls(**value)
        for field in (
            "quality_tier",
            "code_capability",
            "reasoning_capability",
        ):
            score = getattr(candidate, field)
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1")
        if candidate.context_window <= 0:
            raise ValueError("context_window must be positive")
        return candidate

    def to_dict(self) -> dict:
        return asdict(self)


def load_candidate_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[ModelCandidate]:
    registry_path = Path(path)
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Candidate registry must be a non-empty list")
    candidates = [ModelCandidate.from_dict(item) for item in data]
    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Candidate ids must be unique")
    return [candidate for candidate in candidates if candidate.enabled]
