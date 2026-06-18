"""Create a trusted router-pair dataset from adjudicated classifier mismatches."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CONCERNS = {
    "correctness",
    "maintainability",
    "performance",
    "reliability",
    "security",
    "concurrency",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_concerns(value: str) -> set[str]:
    parsed = json.loads(value)
    return set(parsed)


def dump_concerns(values: set[str]) -> str:
    ordered = [label for label in sorted(CONCERNS) if label in values]
    return json.dumps(ordered)


def apply_change(row: dict, item: dict) -> dict | None:
    before = {}
    after = {}
    signal = item["signal"]
    corrected = item["corrected_label"]
    if signal == "operation":
        before["query_operation"] = row["query_operation"]
        row["query_operation"] = corrected
        after["query_operation"] = row["query_operation"]
    elif signal == "query_domain":
        before["query_domain"] = row["query_domain"]
        row["query_domain"] = corrected
        after["query_domain"] = row["query_domain"]
    elif signal in {"concern_missing", "concern_extra"}:
        concerns = load_concerns(row["query_concerns"])
        before["query_concerns"] = sorted(concerns)
        if corrected == "absent":
            concerns.discard(item["label"])
            concerns.discard(item["prediction"])
        elif corrected in CONCERNS:
            concerns.add(corrected)
        else:
            return None
        row["query_concerns"] = dump_concerns(concerns)
        after["query_concerns"] = sorted(concerns)
    else:
        return None
    return {
        "pair_id": row["pair_id"],
        "item_id": item["item_id"],
        "signal": signal,
        "verdict": item["verdict"],
        "dataset_action": item["dataset_action"],
        "before": before,
        "after": after,
        "reason": item["reason"],
        "classifier_failure_mode": item["classifier_failure_mode"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("router_pair_data/llm_combined/llm_grounded_pairs_combined.csv"),
    )
    parser.add_argument(
        "--adjudications",
        type=Path,
        default=Path(
            "router_pair_data/audits/llm_combined/adjudication/"
            "adjudicated_mismatches.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "router_pair_data/trusted_v1/"
            "llm_grounded_pairs_trusted_v1.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    adjudications = read_csv(args.adjudications)
    rows_by_id = {row["pair_id"]: row for row in rows}
    by_pair = defaultdict(list)
    for item in adjudications:
        by_pair[item["pair_id"]].append(item)

    changes = []
    metadata = defaultdict(lambda: {
        "change_items": [],
        "both_defensible_items": [],
        "manual_review_items": [],
    })
    skipped = []
    for pair_id, items in by_pair.items():
        row = rows_by_id.get(pair_id)
        if row is None:
            skipped.append({"pair_id": pair_id, "reason": "pair not found"})
            continue
        for item in items:
            action = item["dataset_action"]
            verdict = item["verdict"]
            if action == "change_label":
                change = apply_change(row, item)
                if change is None:
                    skipped.append({
                        "pair_id": pair_id,
                        "item_id": item["item_id"],
                        "reason": "unsupported correction",
                    })
                    continue
                changes.append(change)
                metadata[pair_id]["change_items"].append(item["item_id"])
            elif verdict == "both_defensible":
                metadata[pair_id]["both_defensible_items"].append({
                    "item_id": item["item_id"],
                    "signal": item["signal"],
                    "label": item["label"],
                    "prediction": item["prediction"],
                    "reason": item["reason"],
                })
            elif action == "manual_review" or verdict in {"neither_right", "unclear"}:
                metadata[pair_id]["manual_review_items"].append({
                    "item_id": item["item_id"],
                    "signal": item["signal"],
                    "label": item["label"],
                    "prediction": item["prediction"],
                    "corrected_label": item["corrected_label"],
                    "reason": item["reason"],
                })

    for row in rows:
        info = metadata.get(row["pair_id"])
        row["trusted_v1_applied_adjudications"] = json.dumps(
            info["change_items"] if info else []
        )
        row["trusted_v1_ambiguous_adjudications"] = json.dumps(
            info["both_defensible_items"] if info else [],
            ensure_ascii=False,
        )
        row["trusted_v1_manual_review_adjudications"] = json.dumps(
            info["manual_review_items"] if info else [],
            ensure_ascii=False,
        )

    write_csv(args.output, rows)
    write_jsonl(args.output.with_suffix(".changes.jsonl"), changes)
    report = {
        "input": str(args.input),
        "adjudications": str(args.adjudications),
        "output": str(args.output),
        "rows": len(rows),
        "adjudicated_pairs": len(by_pair),
        "changed_items": len(changes),
        "changed_pairs": len({change["pair_id"] for change in changes}),
        "skipped": skipped,
        "changes_by_signal": Counter(change["signal"] for change in changes),
        "changes_by_verdict": Counter(change["verdict"] for change in changes),
        "metadata_pairs": {
            "both_defensible": sum(
                bool(value["both_defensible_items"]) for value in metadata.values()
            ),
            "manual_review": sum(
                bool(value["manual_review_items"]) for value in metadata.values()
            ),
        },
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
