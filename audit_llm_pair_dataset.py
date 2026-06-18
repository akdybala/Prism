"""Audit LLM-generated code-query pairs against signal classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


CODE_DOMAIN_TO_QUERY_DOMAIN = {
    "cryptography": "security_crypto",
    "systems_programming": "systems_programming",
    "machine_learning": "machine_learning",
    "concurrency": "concurrency",
    "algorithms": "algorithms",
    "database": "database",
    "backend_api": "backend_api",
    "frontend": "frontend",
    "css_styling": "frontend",
    "config_boilerplate": "devops_cli",
    "general_python": "general",
    "unknown": "general",
}


QUERY_DOMAIN_TO_CODE_DOMAIN = {
    value: key for key, value in CODE_DOMAIN_TO_QUERY_DOMAIN.items()
}
QUERY_DOMAIN_TO_CODE_DOMAIN.update({
    "security_crypto": "cryptography",
    "testing": "general_python",
    "devops_cli": "config_boilerplate",
    "data_processing": "general_python",
    "general": "general_python",
})


@dataclass(frozen=True)
class PredictionRow:
    pair_id: str
    family_id: str
    split: str
    source: str
    repository: str
    file_path: str
    function_name: str
    docstring: str
    code: str
    query: str
    grounding: str
    query_style: str
    ambiguity: str
    generation_role: str
    label_operation: str
    pred_operation: str
    operation_confidence: float
    operation_margin: float
    label_query_domain: str
    pred_query_domain: str
    query_domain_confidence: float
    query_domain_margin: float
    query_domain_ambiguous: bool
    label_concerns: str
    pred_concerns: str
    concern_exact_match: bool
    label_code_domain: str
    pred_code_domain: str
    code_domain_confidence: float
    code_parse_errors: int


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = defaultdict(Counter)
    for label, pred in pairs:
        table[label][pred] += 1
    return {
        label: dict(sorted(counter.items()))
        for label, counter in sorted(table.items())
    }


def accuracy(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return round(sum(label == pred for label, pred in pairs) / len(pairs), 4)


def macro_f1(labels: list[str], pairs: list[tuple[str, str]]) -> float:
    scores = []
    for label in sorted(labels):
        tp = sum(gold == label and pred == label for gold, pred in pairs)
        fp = sum(gold != label and pred == label for gold, pred in pairs)
        fn = sum(gold == label and pred != label for gold, pred in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall:
            scores.append(2 * precision * recall / (precision + recall))
        else:
            scores.append(0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def concern_metrics(rows: list[tuple[set[str], set[str]]]) -> dict:
    labels = sorted({label for gold, pred in rows for label in gold | pred})
    per_label = {}
    for label in labels:
        tp = sum(label in gold and label in pred for gold, pred in rows)
        fp = sum(label not in gold and label in pred for gold, pred in rows)
        fn = sum(label in gold and label not in pred for gold, pred in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(label in gold for gold, _ in rows),
        }
    exact = sum(gold == pred for gold, pred in rows) / len(rows) if rows else 0
    return {
        "exact_match": round(exact, 4),
        "macro_f1": round(
            sum(item["f1"] for item in per_label.values()) / len(per_label),
            4,
        ) if per_label else 0.0,
        "per_label": per_label,
    }


def family_audit(rows: list[dict]) -> dict:
    families = defaultdict(list)
    for row in rows:
        families[row["family_id"]].append(row)
    missing_operations = {}
    split_leaks = {}
    wrong_sizes = {}
    duplicate_code = {}
    for family_id, records in families.items():
        operations = {row["query_operation"] for row in records}
        missing = sorted({
            "explain", "debug", "optimize", "review", "generate", "refactor",
        } - operations)
        if missing:
            missing_operations[family_id] = missing
        splits = {row["split"] for row in records}
        if len(splits) != 1:
            split_leaks[family_id] = sorted(splits)
        if len(records) % 8 != 0:
            wrong_sizes[family_id] = len(records)
        if len({row["code"] for row in records}) != 1:
            duplicate_code[family_id] = len({row["code"] for row in records})
    return {
        "families": len(families),
        "missing_operations": missing_operations,
        "split_leaks": split_leaks,
        "unexpected_family_sizes": wrong_sizes,
        "families_with_multiple_code_values": duplicate_code,
    }


def duplicate_audit(rows: list[dict]) -> dict:
    normalized_queries = Counter(normalize(row["query"]) for row in rows)
    code_query = Counter(
        (row["code"], normalize(row["query"])) for row in rows
    )
    pair_ids = Counter(row["pair_id"] for row in rows)
    return {
        "duplicate_pair_ids": sum(count - 1 for count in pair_ids.values()),
        "duplicate_normalized_queries": sum(
            count - 1 for count in normalized_queries.values()
        ),
        "duplicate_code_query_pairs": sum(
            count - 1 for count in code_query.values()
        ),
    }


def similarity_audit(rows: list[dict]) -> dict:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return {"skipped": "scikit-learn is not installed"}
    texts = [row["query"] for row in rows]
    matrix = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
    ).fit_transform(texts)
    distances, indices = NearestNeighbors(
        n_neighbors=2,
        metric="cosine",
    ).fit(matrix).kneighbors(matrix)
    sims = 1 - distances[:, 1]
    return {
        "near_duplicates_ge_090": int(sum(float(value) >= 0.90 for value in sims)),
        "near_duplicates_ge_080": int(sum(float(value) >= 0.80 for value in sims)),
        "max_similarity": round(float(max(sims)), 4),
    }


def classifier_audit(rows: list[dict], output_dir: Path) -> dict:
    from code_signals import extract_all
    from query_signals import extract_query_signals

    predictions = []
    query_cache = {}
    code_cache = {}
    for index, row in enumerate(rows, 1):
        query = row["query"]
        code = row["code"]
        query_result = query_cache.get(query)
        if query_result is None:
            query_result = extract_query_signals(query)
            query_cache[query] = query_result
        code_result = code_cache.get(row["family_id"])
        if code_result is None:
            code_result = extract_all(code)
            code_cache[row["family_id"]] = code_result

        operation = query_result["query_operation"]
        domain = query_result["query_domain"]
        concerns = query_result["query_concerns"]
        code_domain = code_result["domain"]
        label_concerns = set(json.loads(row["query_concerns"]))
        pred_concerns = set(concerns["detected"])
        label_code_domain = QUERY_DOMAIN_TO_CODE_DOMAIN.get(
            row["query_domain"],
            "general_python",
        )
        predictions.append(PredictionRow(
            pair_id=row["pair_id"],
            family_id=row["family_id"],
            split=row["split"],
            source=row["source"],
            repository=row["repository"],
            file_path=row["file_path"],
            function_name=row["function_name"],
            docstring=row["docstring"],
            code=code,
            query=query,
            grounding=row["grounding"],
            query_style=row["query_style"],
            ambiguity=row["ambiguity"],
            generation_role=row["generation_role"],
            label_operation=row["query_operation"],
            pred_operation=operation["predicted"],
            operation_confidence=float(operation["confidence"]),
            operation_margin=float(operation["margin"]),
            label_query_domain=row["query_domain"],
            pred_query_domain=domain["predicted"],
            query_domain_confidence=float(domain["confidence"]),
            query_domain_margin=float(domain["margin"]),
            query_domain_ambiguous=bool(domain["ambiguous"]),
            label_concerns=json.dumps(sorted(label_concerns)),
            pred_concerns=json.dumps(sorted(pred_concerns)),
            concern_exact_match=label_concerns == pred_concerns,
            label_code_domain=label_code_domain,
            pred_code_domain=code_domain["predicted_domain"],
            code_domain_confidence=float(code_domain["confidence"]),
            code_parse_errors=int(code_result.get("error_count", 0)),
        ))
        if index % 250 == 0:
            print(f"classified {index}/{len(rows)}", flush=True)

    prediction_dicts = [asdict(row) for row in predictions]
    write_csv(output_dir / "classifier_predictions.csv", prediction_dicts)
    mismatches = [
        row for row in prediction_dicts
        if row["label_operation"] != row["pred_operation"]
        or row["label_query_domain"] != row["pred_query_domain"]
        or not row["concern_exact_match"]
        or row["label_code_domain"] != row["pred_code_domain"]
    ]
    write_csv(output_dir / "classifier_mismatches.csv", mismatches)

    operation_pairs = [
        (row.label_operation, row.pred_operation) for row in predictions
    ]
    domain_pairs = [
        (row.label_query_domain, row.pred_query_domain) for row in predictions
    ]
    code_pairs = [
        (row.label_code_domain, row.pred_code_domain) for row in predictions
    ]
    concern_pairs = [
        (set(json.loads(row.label_concerns)), set(json.loads(row.pred_concerns)))
        for row in predictions
    ]
    report = {
        "operation": {
            "accuracy": accuracy(operation_pairs),
            "macro_f1": macro_f1(
                {row.label_operation for row in predictions},
                operation_pairs,
            ),
            "confusion": confusion(operation_pairs),
        },
        "query_domain": {
            "accuracy": accuracy(domain_pairs),
            "macro_f1": macro_f1(
                {row.label_query_domain for row in predictions},
                domain_pairs,
            ),
            "ambiguous_rate": round(
                sum(row.query_domain_ambiguous for row in predictions)
                / len(predictions),
                4,
            ),
            "confusion": confusion(domain_pairs),
        },
        "query_concerns": concern_metrics(concern_pairs),
        "code_domain_proxy": {
            "note": (
                "Code-domain labels are proxied from query_domain, so cross-domain "
                "queries may count as expected mismatches."
            ),
            "accuracy": accuracy(code_pairs),
            "macro_f1": macro_f1(
                {row.label_code_domain for row in predictions},
                code_pairs,
            ),
            "parse_error_records": sum(row.code_parse_errors > 0 for row in predictions),
            "confusion": confusion(code_pairs),
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("router_pair_data/llm_combined/llm_grounded_pairs_combined.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("router_pair_data/audits/llm_combined"),
    )
    parser.add_argument("--skip-classifiers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "input": str(args.input),
        "rows": len(rows),
        "schema": {
            "columns": list(rows[0]) if rows else [],
            "empty_required_fields": {
                field: sum(not row.get(field, "").strip() for row in rows)
                for field in (
                    "pair_id", "family_id", "split", "code", "query",
                    "query_domain", "query_operation", "query_concerns",
                    "grounding",
                )
            },
        },
        "counts": {
            "source": Counter(row["source"] for row in rows),
            "split": Counter(row["split"] for row in rows),
            "query_operation": Counter(row["query_operation"] for row in rows),
            "query_domain": Counter(row["query_domain"] for row in rows),
            "query_concern": Counter(
                concern
                for row in rows
                for concern in json.loads(row["query_concerns"])
            ),
            "generation_role": Counter(row["generation_role"] for row in rows),
        },
        "family": family_audit(rows),
        "deduplication": duplicate_audit(rows),
        "similarity": similarity_audit(rows),
    }
    if not args.skip_classifiers:
        report["classifiers"] = classifier_audit(rows, args.output_dir)
    (args.output_dir / "dataset_audit_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
