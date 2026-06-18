"""Attach code-domain classifier predictions to a trusted router dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("router_pair_data/trusted_v1/llm_grounded_pairs_trusted_v1.csv"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("router_pair_data/audits/trusted_v1/classifier_predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "router_pair_data/trusted_v1/"
            "llm_grounded_pairs_trusted_v1_with_code_domains.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    predictions = {
        row["pair_id"]: row for row in read_csv(args.predictions)
    }
    missing = []
    for row in rows:
        pred = predictions.get(row["pair_id"])
        if pred is None:
            missing.append(row["pair_id"])
            row["trusted_v1_code_domain"] = ""
            row["trusted_v1_code_domain_confidence"] = ""
            row["trusted_v1_code_parse_errors"] = ""
            continue
        row["trusted_v1_code_domain"] = pred["pred_code_domain"]
        row["trusted_v1_code_domain_confidence"] = pred["code_domain_confidence"]
        row["trusted_v1_code_parse_errors"] = pred["code_parse_errors"]
    write_csv(args.output, rows)
    report = {
        "input": str(args.input),
        "predictions": str(args.predictions),
        "output": str(args.output),
        "rows": len(rows),
        "missing_predictions": missing,
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
