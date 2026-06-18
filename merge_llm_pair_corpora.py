"""Merge validated LLM-grounded pair corpora and audit label coverage."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    seen_pairs = set()
    seen_queries = set()
    duplicate_pairs = 0
    duplicate_queries = 0
    for path in args.inputs:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["pair_id"] in seen_pairs:
                    duplicate_pairs += 1
                    continue
                normalized = normalize(row["query"])
                if normalized in seen_queries:
                    duplicate_queries += 1
                    continue
                seen_pairs.add(row["pair_id"])
                seen_queries.add(normalized)
                rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "llm_grounded_pairs_combined.csv"
    jsonl_path = args.output_dir / "llm_grounded_pairs_combined.jsonl"
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    family_splits = defaultdict(set)
    for row in rows:
        family_splits[row["family_id"]].add(row["split"])
    report = {
        "rows": len(rows),
        "function_families": len(family_splits),
        "seeds": sum(row["generation_role"] == "seed" for row in rows),
        "variants": sum(row["generation_role"] == "variant" for row in rows),
        "duplicates_removed": {
            "pair_id": duplicate_pairs,
            "normalized_query": duplicate_queries,
        },
        "families_crossing_splits": sum(
            len(splits) != 1 for splits in family_splits.values()
        ),
        "counts": {
            "source": Counter(row["source"] for row in rows),
            "split": Counter(row["split"] for row in rows),
            "operation": Counter(row["query_operation"] for row in rows),
            "domain": Counter(row["query_domain"] for row in rows),
            "concern": Counter(
                concern
                for row in rows
                for concern in json.loads(row["query_concerns"])
            ),
            "ambiguity": Counter(row["ambiguity"] for row in rows),
        },
    }
    (args.output_dir / "combined_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
