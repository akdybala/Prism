"""Prepare and optionally LLM-adjudicate query signal mismatches."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


VERDICTS = (
    "label_right",
    "classifier_right",
    "both_defensible",
    "neither_right",
    "unclear",
)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def parse_json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def make_items(rows: list[dict]) -> list[dict]:
    items = []
    seen = set()
    for row in rows:
        base = {
            "pair_id": row["pair_id"],
            "family_id": row["family_id"],
            "split": row["split"],
            "source": row["source"],
            "repository": row["repository"],
            "file_path": row["file_path"],
            "function_name": row["function_name"],
            "docstring": row["docstring"],
            "query": row["query"],
            "code": row["code"],
            "grounding": row["grounding"],
        }
        if row["label_operation"] != row["pred_operation"]:
            items.append({
                **base,
                "item_id": f"{row['pair_id']}:operation",
                "signal": "operation",
                "label": row["label_operation"],
                "prediction": row["pred_operation"],
                "dispute": (
                    f"operation label={row['label_operation']} "
                    f"prediction={row['pred_operation']}"
                ),
            })
        if row["label_query_domain"] != row["pred_query_domain"]:
            items.append({
                **base,
                "item_id": f"{row['pair_id']}:query_domain",
                "signal": "query_domain",
                "label": row["label_query_domain"],
                "prediction": row["pred_query_domain"],
                "dispute": (
                    f"query_domain label={row['label_query_domain']} "
                    f"prediction={row['pred_query_domain']}"
                ),
            })
        label_concerns = set(parse_json_list(row["label_concerns"]))
        pred_concerns = set(parse_json_list(row["pred_concerns"]))
        for concern in sorted(label_concerns - pred_concerns):
            key = f"{row['pair_id']}:concern_missing:{concern}"
            if key not in seen:
                seen.add(key)
                items.append({
                    **base,
                    "item_id": key,
                    "signal": "concern_missing",
                    "label": concern,
                    "prediction": "absent",
                    "dispute": f"concern label includes {concern}; classifier omitted it",
                })
        for concern in sorted(pred_concerns - label_concerns):
            key = f"{row['pair_id']}:concern_extra:{concern}"
            if key not in seen:
                seen.add(key)
                items.append({
                    **base,
                    "item_id": key,
                    "signal": "concern_extra",
                    "label": "absent",
                    "prediction": concern,
                    "dispute": f"classifier predicted concern {concern}; label omitted it",
                })
    return items


def schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["adjudications"],
        "properties": {
            "adjudications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "item_id",
                        "verdict",
                        "corrected_label",
                        "reason",
                        "classifier_failure_mode",
                        "dataset_action",
                    ],
                    "properties": {
                        "item_id": {"type": "string"},
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                        "corrected_label": {"type": "string"},
                        "reason": {"type": "string"},
                        "classifier_failure_mode": {"type": "string"},
                        "dataset_action": {
                            "type": "string",
                            "enum": [
                                "keep_label_add_boundary_example",
                                "change_label",
                                "manual_review",
                                "drop_or_rewrite_query",
                                "no_action",
                            ],
                        },
                    },
                },
            },
        },
    }


def compact_item(item: dict, max_code_chars: int) -> dict:
    code = item["code"]
    if len(code) > max_code_chars:
        code = code[:max_code_chars] + "\n# [code truncated]"
    return {
        key: item[key]
        for key in (
            "item_id",
            "signal",
            "label",
            "prediction",
            "dispute",
            "function_name",
            "docstring",
            "query",
            "grounding",
        )
    } | {"code": code}


def prompt_for(items: list[dict], max_code_chars: int) -> str:
    compact = [compact_item(item, max_code_chars) for item in items]
    return f"""You are adjudicating classifier mismatch records for a code/query
router training dataset.

For each item, decide whether the original dataset label is right, the current
classifier prediction is right, both are defensible, neither is right, or the
case is unclear. Use the ontology below and the query as the primary evidence.
Use code/docstring/grounding only to decide whether the query is plausible and
code-grounded.

Operation labels:
- explain: describe, trace, compare, teach, summarize, answer how/why.
- debug: diagnose/fix a concrete fault, mismatch, failing case, regression, or
  unexpected behavior.
- optimize: improve measurable speed, memory, latency, throughput, complexity,
  allocations, or resource use.
- review: inspect and report issues, risks, safety, quality, or correctness
  without implementing the main change.
- generate: create tests/code/config/docs/implementation that does not exist.
- refactor: restructure existing working code for clarity/API/design while
  preserving behavior.

Query-domain labels:
algorithms, machine_learning, backend_api, database, concurrency, frontend,
systems_programming, security_crypto, testing, devops_cli, data_processing,
general. Domain means the technical context introduced by the query, not
necessarily the code's intrinsic domain.

Concern labels:
security, concurrency, correctness, performance, reliability, maintainability.
Concerns should be explicit or strongly implied quality/risk dimensions.

Failure mode should be a short reusable explanation, e.g.
"debug phrased as explanation", "review uses bug-like vocabulary",
"domain keyword from context outweighed primary subject",
"reliability implied too weakly", "generated label over-applied concern".

Return only schema-conforming JSON.

ITEMS:
{json.dumps(compact, ensure_ascii=False, indent=2)}
"""


def run_codex(cli: Path, prompt: str, schema_path: Path, output_path: Path, timeout: int) -> None:
    subprocess.run(
        [
            str(cli.resolve()),
            "exec",
            "--ephemeral",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--output-schema",
            str(schema_path.resolve()),
            "--output-last-message",
            str(output_path.resolve()),
            "-",
        ],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=timeout,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_batches(
    items: list[dict],
    output_dir: Path,
    cli: Path,
    batch_size: int,
    max_code_chars: int,
    timeout: int,
    retries: int,
) -> None:
    raw_dir = output_dir / "raw_adjudication_batches"
    raw_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "adjudication_schema.json"
    schema_path.write_text(json.dumps(schema(), indent=2), encoding="utf-8")
    failures = []
    existing_numbers = [
        int(path.stem.split("_")[-1])
        for path in raw_dir.glob("batch_*.json")
        if path.stem.split("_")[-1].isdigit()
    ]
    next_batch_number = max(existing_numbers, default=-1) + 1
    for start in range(0, len(items), batch_size):
        batch_number = next_batch_number + (start // batch_size)
        batch = items[start:start + batch_size]
        path = raw_dir / f"batch_{batch_number:04d}.json"
        prompt = prompt_for(batch, max_code_chars)
        last_error = None
        for attempt in range(retries + 1):
            try:
                temp = path.with_suffix(".tmp")
                run_codex(cli, prompt, schema_path, temp, timeout)
                parsed = json.loads(temp.read_text(encoding="utf-8"))
                ids = {item["item_id"] for item in batch}
                returned = {item["item_id"] for item in parsed["adjudications"]}
                if ids != returned:
                    raise ValueError("returned item IDs do not match batch")
                temp.replace(path)
                print(f"adjudicated batch {batch_number} ({start + len(batch)}/{len(items)})", flush=True)
                break
            except Exception as exc:  # noqa: BLE001 - audit runner records failures
                last_error = str(exc)
                time.sleep(min(2 ** attempt, 8))
        else:
            failures.append({
                "batch": batch_number,
                "item_ids": [item["item_id"] for item in batch],
                "error": last_error,
            })
            print(f"failed batch {batch_number}: {last_error}", flush=True)
            break
    (output_dir / "adjudication_failures.json").write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )


def assemble(items: list[dict], output_dir: Path) -> dict:
    by_id = {item["item_id"]: item for item in items}
    adjudicated = []
    for path in sorted((output_dir / "raw_adjudication_batches").glob("batch_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["adjudications"]:
            source = by_id[row["item_id"]]
            adjudicated.append({
                **source,
                **row,
            })
    write_jsonl(output_dir / "adjudicated_mismatches.jsonl", adjudicated)
    write_csv(output_dir / "adjudicated_mismatches.csv", adjudicated)
    report = {
        "total_items": len(items),
        "adjudicated_items": len(adjudicated),
        "pending_items": len(items) - len(adjudicated),
        "signals": Counter(item["signal"] for item in items),
        "adjudicated_signals": Counter(item["signal"] for item in adjudicated),
        "verdicts": Counter(item["verdict"] for item in adjudicated),
        "dataset_actions": Counter(item["dataset_action"] for item in adjudicated),
        "failure_modes": Counter(
            item["classifier_failure_mode"] for item in adjudicated
        ).most_common(50),
    }
    (output_dir / "adjudication_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("router_pair_data/audits/llm_combined/classifier_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("router_pair_data/audits/llm_combined/adjudication"),
    )
    parser.add_argument("--cli", type=Path, default=Path(".codex-cli.exe"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-code-chars", type=int, default=3500)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Generate only items not already present in raw adjudication batches.",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.predictions)
    items = make_items(rows)
    all_items = items
    if args.pending_only:
        raw_dir = args.output_dir / "raw_adjudication_batches"
        completed = set()
        for path in sorted(raw_dir.glob("batch_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            completed.update(row["item_id"] for row in data.get("adjudications", []))
        items = [item for item in items if item["item_id"] not in completed]
    if args.limit:
        items = items[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "mismatch_adjudication_items.jsonl", all_items)
    report = {
        "items": len(all_items),
        "selected_items": len(items),
        "signals": Counter(item["signal"] for item in all_items),
        "selected_signals": Counter(item["signal"] for item in items),
        "top_disputes": Counter(item["dispute"] for item in all_items).most_common(40),
    }
    (args.output_dir / "mismatch_item_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    if not args.prepare_only and not args.assemble_only:
        generate_batches(
            items,
            args.output_dir,
            args.cli,
            args.batch_size,
            args.max_code_chars,
            args.timeout_seconds,
            args.retries,
        )
    if not args.prepare_only:
        report = assemble(all_items, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
