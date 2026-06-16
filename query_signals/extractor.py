import re
from pathlib import Path

from .classifier import (
    EmbeddingKNNClassifier,
    EmbeddingLogisticClassifier,
    EmbeddingMultiLabelLogisticClassifier,
)


QUERY_OPERATIONS = [
    "explain",
    "debug",
    "optimize",
    "review",
    "generate",
    "refactor",
]
# Compatibility alias for code that has not yet adopted the clearer name.
QUERY_TYPES = QUERY_OPERATIONS
QUERY_CONCERNS = [
    "security",
    "concurrency",
    "correctness",
    "performance",
    "reliability",
    "maintainability",
]
QUERY_DOMAINS = [
    "algorithms",
    "machine_learning",
    "backend_api",
    "database",
    "concurrency",
    "frontend",
    "systems_programming",
    "security_crypto",
    "testing",
    "devops_cli",
    "data_processing",
    "general",
]
NO_DOMAIN_EVIDENCE_CONFIDENCE_THRESHOLD = 0.65

_DOMAIN_SIGNAL_TERMS = (
    "algorithm",
    "base case",
    "binary search",
    "bfs",
    "dfs",
    "dijkstra",
    "dynamic programming",
    "data structure",
    "graph",
    "greedy",
    "heap",
    "interval",
    "linked list",
    "memoization",
    "merge sort",
    "optimal substructure",
    "recurrence",
    "recursion",
    "segment tree",
    "sorting",
    "shortest path",
    "sliding window",
    "stack",
    "subset",
    "topological sort",
    "time complexity",
    "model",
    "adam",
    "sgd",
    "training",
    "inference",
    "loss",
    "gradient",
    "neural",
    "tensor",
    "embedding",
    "classifier",
    "regression",
    "overfitting",
    "epoch",
    "batch size",
    "learning rate",
    "dropout",
    "early stopping",
    "gan",
    "transformer",
    "beam search",
    "cnn",
    "rnn",
    "regularization",
    "cross-validation",
    "data augmentation",
    "transfer learning",
    "confusion matrix",
    "api",
    "endpoint",
    "http",
    "rest",
    "graphql",
    "webhook",
    "middleware",
    "query parameter",
    "path parameter",
    "payload",
    "get",
    "post",
    "put",
    "patch",
    "response",
    "request logging",
    "serialize",
    "json",
    "status code",
    "cors",
    "database",
    "sql",
    "transaction",
    "table",
    "row-level",
    "index",
    "schema",
    "postgres",
    "postgresql",
    "mysql",
    "mongodb",
    "orm",
    "join",
    "foreign key",
    "bulk upsert",
    "cascade delete",
    "denormalize",
    "full-text search",
    "materialized view",
    "isolation level",
    "connection pool",
    "thread",
    "threading",
    "asyncio",
    "async",
    "await",
    "coroutine",
    "race condition",
    "deadlock",
    "lock",
    "mutex",
    "semaphore",
    "worker",
    "process",
    "gil",
    "event loop",
    "concurrent",
    "parallel",
    "atomic",
    "shared state",
    "synchronization",
    "producer-consumer",
    "backpressure",
    "condition variable",
    "react",
    "component",
    "browser",
    "div",
    "click handler",
    "form",
    "responsive",
    "mobile",
    "infinite scroll",
    "single-page",
    "debounce",
    "dark mode",
    "local state",
    "global store",
    "toast",
    "internationalization",
    "css",
    "html",
    "javascript",
    "typescript",
    "dom",
    "user interface",
    "bundle",
    "pointer",
    "socket",
    "memory allocation",
    "memory leak",
    "buffer size",
    "file i/o",
    "daemon",
    "segfault",
    "byte order",
    "zero-copy",
    "endianness",
    "struct",
    "memory pool",
    "ipc",
    "system call",
    "file descriptor",
    "epoll",
    "fork",
    "mmap",
    "kernel",
    "native code",
    "encryption",
    "decrypt",
    "hash",
    "password",
    "secret",
    "nonce",
    "cryptographic",
    "hmac",
    "bcrypt",
    "digital signature",
    "rsa",
    "ecdsa",
    "tls",
    "ssl",
    "totp",
    "two-factor",
    "key derivation",
    "key generation",
    "entropy",
    "padding scheme",
    "csr",
    "certificate",
    "xss",
    "csrf",
    "injection",
    "vulnerability",
    "exploit",
    "authorization",
    "authentication",
    "unit test",
    "integration test",
    "test",
    "mock",
    "fixture",
    "assertion",
    "test coverage",
    "test case",
    "pytest",
    "docker",
    "kubernetes",
    "container",
    "deployment",
    "deploy",
    "auto-scaling",
    "environment variable",
    "config file",
    "cli",
    "shell script",
    "stdin",
    "stdout",
    "subcommand",
    "ci/cd",
    "command-line",
    "argparse",
    "exit code",
    "health check",
    "load balancer",
    "pandas",
    "dataframe",
    "csv",
    "etl",
    "dataset",
    "pivot",
    "resample",
    "group by",
    "aggregate",
    "data type",
    "column",
    "record",
    "vectorized",
    "rolling window",
    "expanding window",
    "cumulative sum",
    "transformation",
    "data quality",
    "deduplicate",
    "groupby",
    "aggregation",
    "categorical",
    "missing values",
)
_DOMAIN_SIGNAL_PATTERN = re.compile(
    "|".join(
        rf"\b{re.escape(term)}(?:s|es)?\b"
        for term in sorted(_DOMAIN_SIGNAL_TERMS, key=len, reverse=True)
    ),
    re.IGNORECASE,
)

_CONCERN_PATTERNS = {
    "security": (
        r"\bsecur(?:e|ity)\b",
        r"\bvulnerab(?:le|ility|ilities)\b",
        r"\b(?:exploit|attack|injection|xss|csrf)\b",
        r"\b(?:authentication|authorization|password|secret|token)\b",
        r"\b(?:encrypt|decrypt|hash|cryptograph|timing attack)\w*\b",
        r"\b(?:path traversal|buffer overflow|sensitive data)\b",
    ),
    "concurrency": (
        r"\b(?:thread safe|thread-safe|race condition|deadlock)\b",
        r"\b(?:thread|coroutine|async|await|concurrent|parallel)\w*\b",
        r"\b(?:lock|mutex|semaphore|atomic|shared state)\b",
        r"\b(?:event loop|task cancellation|critical section)\b",
    ),
    "correctness": (
        r"\b(?:correct|incorrect|wrong result|wrong output)\b",
        r"\b(?:bug|defect|off-by-one|edge case|regression)\b",
        r"\b(?:expected|unexpected|fails?|failure|exception|crash)\w*\b",
        r"\b(?:returns? none|missing|duplicate)\w*\b",
    ),
    "performance": (
        r"\b(?:performance|optimi[sz]e|faster|slow|latency)\w*\b",
        r"\b(?:time complexity|space complexity|memory usage|scalab)\w*\b",
        r"\b(?:tle|timeout|bottleneck|throughput|allocation)\w*\b",
    ),
    "reliability": (
        r"\b(?:reliable|reliability|robust|resilien)\w*\b",
        r"\b(?:retry|recovery|fallback|timeout|partial failure)\w*\b",
        r"\b(?:data loss|corruption|idempotent|atomic write)\w*\b",
        r"\b(?:intermittent|flaky|race condition|deadlock)\b",
    ),
    "maintainability": (
        r"\b(?:maintainable|maintainability|readable|readability)\b",
        r"\b(?:refactor|clean up|simplify|code quality)\w*\b",
        r"\b(?:technical debt|modular|coupling|cohesion)\w*\b",
        r"\b(?:engineering practices|best practices|public interface)\b",
    ),
}
_COMPILED_CONCERN_PATTERNS = {
    concern: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for concern, patterns in _CONCERN_PATTERNS.items()
}

_BASE_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _BASE_DIR / "cache"

_type_classifier = EmbeddingLogisticClassifier(
    examples_path=_BASE_DIR / "query_type_examples.json",
    cache_path=_CACHE_DIR / "query_type_embeddings.npz",
    model_cache_path=_CACHE_DIR / "query_type_logistic.npz",
    c_value=10.0,
    ambiguity_confidence_threshold=0.35,
    ambiguity_margin_threshold=0.1,
)
_domain_classifier = EmbeddingLogisticClassifier(
    examples_path=_BASE_DIR / "query_domain_examples.json",
    cache_path=_CACHE_DIR / "query_domain_embeddings.npz",
    model_cache_path=_CACHE_DIR / "query_domain_logistic.npz",
    c_value=3.0,
    ambiguity_confidence_threshold=0.5,
    ambiguity_margin_threshold=0.15,
)
_concern_classifier = EmbeddingMultiLabelLogisticClassifier(
    examples_path=_BASE_DIR / "query_concern_examples.json",
    cache_path=_CACHE_DIR / "query_concern_embeddings.npz",
    model_cache_path=_CACHE_DIR / "query_concern_logistic.npz",
    labels=QUERY_CONCERNS,
    c_value=10.0,
    thresholds={
        "security": 0.60,
        "concurrency": 0.50,
        "correctness": 0.40,
        "performance": 0.60,
        "reliability": 0.60,
        "maintainability": 0.45,
    },
)

if set(_type_classifier.all_classes) != set(QUERY_OPERATIONS):
    raise ValueError(
        "query_type_examples.json classes do not match QUERY_OPERATIONS"
    )
if set(_domain_classifier.all_classes) != set(QUERY_DOMAINS):
    raise ValueError(
        "query_domain_examples.json classes do not match QUERY_DOMAINS"
    )


def _uniform_scores(classes: list[str]) -> dict[str, float]:
    uniform = 1.0 / len(classes)
    return EmbeddingKNNClassifier._rounded_distribution(
        {label: uniform for label in sorted(classes)}
    )


def _empty_result() -> dict:
    operation_scores = _uniform_scores(QUERY_OPERATIONS)
    domain_scores = _uniform_scores(QUERY_DOMAINS)
    operation_result = _type_classifier._result_for_scores(operation_scores)
    domain_result = _domain_classifier._result_for_scores(domain_scores)
    operation_result["predicted"] = "unknown"
    domain_result["predicted"] = "general"
    domain_result["domain_signal_present"] = False
    return {
        "query_operation": operation_result,
        "query_concerns": {
            "scores": {concern: 0.0 for concern in QUERY_CONCERNS},
            "detected": [],
            "primary": "none",
            "confidence": 0.0,
            "thresholds": dict(_concern_classifier.thresholds),
            "multi_label": True,
            "method": "embedding_logistic_one_vs_rest",
            "evidence": {},
        },
        "query_domain": domain_result,
    }


def _has_domain_signal(query: str) -> bool:
    return bool(_DOMAIN_SIGNAL_PATTERN.search(query))


def _apply_domain_evidence_guard(
    result: dict,
    domain_signal_present: bool,
) -> None:
    result["domain_signal_present"] = domain_signal_present
    if (
        not domain_signal_present
        and result["predicted"] != "general"
        and result["confidence"] < NO_DOMAIN_EVIDENCE_CONFIDENCE_THRESHOLD
    ):
        result["ambiguous"] = True
        if "missing_domain_signal" not in result["ambiguity_reasons"]:
            result["ambiguity_reasons"].append("missing_domain_signal")


def _extract_concern_evidence(query: str) -> dict:
    evidence = {}
    for concern in QUERY_CONCERNS:
        matches = []
        for pattern in _COMPILED_CONCERN_PATTERNS[concern]:
            matches.extend(match.group(0) for match in pattern.finditer(query))
        unique_matches = list(dict.fromkeys(match.lower() for match in matches))
        if unique_matches:
            evidence[concern] = unique_matches
    return evidence


def extract_query_signals(query: str | None) -> dict:
    if not query or not query.strip():
        return _empty_result()

    model = EmbeddingKNNClassifier._get_model()
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    operation_result = _type_classifier.predict_from_embedding(query_embedding)
    concern_result = _concern_classifier.predict_from_embedding(query_embedding)
    concern_result["evidence"] = _extract_concern_evidence(query)
    domain_result = _domain_classifier.predict_from_embedding(query_embedding)
    domain_signal_present = _has_domain_signal(query)
    _apply_domain_evidence_guard(domain_result, domain_signal_present)
    return {
        "query_operation": operation_result,
        "query_concerns": concern_result,
        "query_domain": domain_result,
    }
