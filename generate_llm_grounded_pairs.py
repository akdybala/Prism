"""Generate independently written, code-grounded query pairs with Codex CLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from build_code_query_pairs import CONCERNS, DOMAINS, OPERATIONS, stable_id


STYLES = (
    "pr_description",
    "issue_comment",
    "code_review",
    "developer_question",
    "maintenance_request",
    "incident_note",
)
AMBIGUITIES = ("low", "medium", "high")
FOCUS_CONCERNS = ("concurrency", "security", "performance")

DOMAIN_TARGETS = {
    "machine_learning": (
        "machine_learning", "data_processing", "backend_api", "testing",
    ),
    "frontend": ("frontend", "backend_api", "testing", "security_crypto"),
    "database": ("database", "backend_api", "data_processing", "concurrency"),
    "concurrency": (
        "concurrency", "systems_programming", "backend_api", "testing",
    ),
    "security_crypto": (
        "security_crypto", "backend_api", "systems_programming", "testing",
    ),
    "systems_programming": (
        "systems_programming", "concurrency", "security_crypto", "devops_cli",
    ),
    "data_processing": (
        "data_processing", "database", "machine_learning", "backend_api",
    ),
    "backend_api": (
        "backend_api", "security_crypto", "database", "concurrency",
    ),
    "testing": ("testing", "devops_cli", "backend_api", "concurrency"),
    "devops_cli": (
        "devops_cli", "systems_programming", "security_crypto", "testing",
    ),
    "algorithms": (
        "algorithms", "machine_learning", "data_processing", "testing",
    ),
    "general": ("general", "backend_api", "data_processing", "testing"),
}


def load_unique_functions(path: Path) -> list[dict]:
    by_family = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_family.setdefault(row["family_id"], {
                key: row[key]
                for key in (
                    "family_id",
                    "split",
                    "source",
                    "repository",
                    "source_url",
                    "revision",
                    "license",
                    "file_path",
                    "function_name",
                    "docstring",
                    "code",
                    "query_domain",
                )
            })
    return list(by_family.values())


def balanced_sample(functions: list[dict], count: int) -> list[dict]:
    groups = defaultdict(list)
    for function in functions:
        groups[(function["source"], function["query_domain"])].append(function)
    for key, values in groups.items():
        values.sort(key=lambda item: hashlib.sha256(
            f"{key}:{item['family_id']}".encode()
        ).hexdigest())

    selected = []
    active = sorted(groups)
    while len(selected) < count and active:
        next_active = []
        for key in active:
            values = groups[key]
            if values and len(selected) < count:
                selected.append(values.pop())
            if values:
                next_active.append(key)
        active = next_active
    return selected


def load_family_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["family_id"] for row in csv.DictReader(handle)}


def gap_targeted_sample(
    functions: list[dict],
    count: int,
    excluded_ids: set[str],
) -> list[dict]:
    by_domain = defaultdict(list)
    fallback_by_domain = defaultdict(list)
    for function in functions:
        fallback_by_domain[function["query_domain"]].append(function)
        if function["family_id"] not in excluded_ids:
            by_domain[function["query_domain"]].append(function)
    for groups in (by_domain, fallback_by_domain):
        for domain, values in groups.items():
            values.sort(key=lambda item: hashlib.sha256(
                f"gap:{domain}:{item['family_id']}".encode()
            ).hexdigest())

    desired = {
        "machine_learning": 10,
        "systems_programming": 10,
        "data_processing": 15,
        "frontend": 20,
        "database": 20,
        "concurrency": 20,
        "security_crypto": 20,
        "backend_api": 15,
        "devops_cli": 10,
        "testing": 5,
        "algorithms": 5,
    }
    selected = []
    selected_ids = set()
    for domain, quota in desired.items():
        candidates = list(by_domain.get(domain, ()))
        if len(candidates) < quota:
            candidates.extend(
                item
                for item in fallback_by_domain.get(domain, ())
                if item["family_id"] not in excluded_ids
                and item["family_id"] not in {
                    candidate["family_id"] for candidate in candidates
                }
            )
        if len(candidates) < quota:
            candidates.extend(
                item
                for item in fallback_by_domain.get(domain, ())
                if item["family_id"] not in selected_ids
                and item["family_id"] not in {
                    candidate["family_id"] for candidate in candidates
                }
            )
        for function in candidates[:quota]:
            if function["family_id"] not in selected_ids:
                selected.append(function)
                selected_ids.add(function["family_id"])

    remaining = [
        function for function in functions
        if function["family_id"] not in excluded_ids
        and function["family_id"] not in selected_ids
    ]
    remaining.sort(key=lambda item: hashlib.sha256(
        f"gap-fill:{item['family_id']}".encode()
    ).hexdigest())
    selected.extend(remaining[: max(0, count - len(selected))])
    selected = selected[:count]
    for index, function in enumerate(selected):
        targets = list(DOMAIN_TARGETS[function["query_domain"]])
        shift = index % len(targets)
        function["target_query_domains"] = targets[shift:] + targets[:shift]
        function["required_concerns"] = list(CONCERNS)
        function["focus_concerns"] = list(FOCUS_CONCERNS)
    return selected


def schema(queries_per_function: int) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["functions"],
        "properties": {
            "functions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["family_id", "queries"],
                    "properties": {
                        "family_id": {"type": "string"},
                        "queries": {
                            "type": "array",
                            "minItems": queries_per_function,
                            "maxItems": queries_per_function,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "query",
                                    "style",
                                    "domain",
                                    "operation",
                                    "concerns",
                                    "ambiguity",
                                    "grounding",
                                ],
                                "properties": {
                                    "query": {"type": "string", "minLength": 20},
                                    "style": {"type": "string", "enum": list(STYLES)},
                                    "domain": {"type": "string", "enum": list(DOMAINS)},
                                    "operation": {
                                        "type": "string",
                                        "enum": list(OPERATIONS),
                                    },
                                    "concerns": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "enum": list(CONCERNS),
                                        },
                                    },
                                    "ambiguity": {
                                        "type": "string",
                                        "enum": list(AMBIGUITIES),
                                    },
                                    "grounding": {
                                        "type": "string",
                                        "minLength": 8,
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def compact_function(function: dict, max_code_chars: int) -> dict:
    code = function["code"]
    if len(code) > max_code_chars:
        code = code[:max_code_chars] + "\n# [function body truncated for context budget]"
    result = {
        "family_id": function["family_id"],
        "source": function["source"],
        "repository": function["repository"],
        "file_path": function["file_path"],
        "function_name": function["function_name"],
        "docstring": function["docstring"],
        "code": code,
        "existing_domain_hint": function["query_domain"],
    }
    for name in (
        "target_query_domains",
        "required_concerns",
        "focus_concerns",
    ):
        if name in function:
            result[name] = function[name]
    return result


def prompt_for(batch: list[dict], queries_per_function: int) -> str:
    return f"""You are authoring a high-quality dataset of realistic developer
requests paired with Python functions.

For EACH supplied function, write exactly {queries_per_function} queries.
The first query is the carefully authored SEED. The remaining
{queries_per_function - 1} are independently composed variants inspired by the
same code, not paraphrases of the seed and not slot-filled templates.

Requirements:
- Read the implementation and docstring. Every query must be plausible for this
  exact function. In `grounding`, briefly name the concrete code behavior that
  supports the query. Do not invent a definite bug that the code cannot plausibly
  have; uncertainty and reported symptoms are welcome.
- Sound like real PR descriptions, issue comments, code review comments,
  developer questions, maintenance requests, or incident notes.
- Vary length, register, specificity, sentence shape, and implied context.
- Include ambiguity: incomplete reports, competing hypotheses, mixed requests,
  understated concerns, and wording whose primary operation is still defensible.
- Across each function's set, cover all six operations exactly once before
  repeating any: explain, debug, optimize, review, generate, refactor.
- Across each set, use at least 3 domain labels and at least 4 concern labels.
  The query domain is the technical context introduced by the request, which may
  differ from the function's default domain. Cross-domain scenarios must remain
  plausible, such as a generic parser used in an API, data pipeline, test helper,
  concurrent worker, CLI, or security boundary.
- Use 0-2 concerns per query. Concerns are requested quality dimensions, not
  automatic properties of the operation.
- If a function includes `target_query_domains`, use every listed domain at
  least once. Keep each scenario credible for that exact implementation.
- If `required_concerns` is present, cover every listed concern across the set.
  Each `focus_concerns` label must appear in at least two different queries.
- Do not begin queries with label words repeatedly. Do not mention dataset
  labels, routing, classifiers, HumanEval, MBPP, or that text was generated.
- Do not reuse stock phrases across functions. Avoid generic requests that could
  be attached unchanged to unrelated code.
- Do not include code fences or solutions in the query.

Return only the schema-conforming JSON.

FUNCTIONS:
{json.dumps(batch, ensure_ascii=False, indent=2)}
"""


def normalize_query(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def validate_batch(
    result: dict,
    expected: list[dict],
    queries_per_function: int,
) -> list[str]:
    errors = []
    expected_ids = {item["family_id"] for item in expected}
    items = result.get("functions", [])
    actual_ids = [item.get("family_id") for item in items]
    if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
        errors.append("family IDs do not exactly match the requested batch")
    seen_queries = set()
    for item in items:
        family_id = item.get("family_id", "")
        queries = item.get("queries", [])
        if len(queries) != queries_per_function:
            errors.append(f"{family_id}: expected {queries_per_function} queries")
            continue
        operations = Counter(query["operation"] for query in queries)
        if any(operations[operation] < 1 for operation in OPERATIONS):
            errors.append(f"{family_id}: missing one or more operations")
        if len({query["domain"] for query in queries}) < 3:
            errors.append(f"{family_id}: fewer than 3 domains")
        expected_function = next(
            (
                function for function in expected
                if function["family_id"] == family_id
            ),
            {},
        )
        missing_target_domains = set(
            expected_function.get("target_query_domains", ())
        ) - {query["domain"] for query in queries}
        if missing_target_domains:
            errors.append(
                f"{family_id}: missing target domains "
                f"{sorted(missing_target_domains)}"
            )
        concern_set = {
            concern for query in queries for concern in query["concerns"]
        }
        if any(
            len(query["concerns"]) != len(set(query["concerns"]))
            for query in queries
        ):
            errors.append(f"{family_id}: duplicate concern in a query")
        if len(concern_set) < 4:
            errors.append(f"{family_id}: fewer than 4 concern labels")
        missing_concerns = set(
            expected_function.get("required_concerns", ())
        ) - concern_set
        if missing_concerns:
            errors.append(
                f"{family_id}: missing concerns {sorted(missing_concerns)}"
            )
        concern_counts = Counter(
            concern for query in queries for concern in query["concerns"]
        )
        weak_focus = [
            concern
            for concern in expected_function.get("focus_concerns", ())
            if concern_counts[concern] < 2
        ]
        if weak_focus:
            errors.append(
                f"{family_id}: focus concerns need 2+ uses {weak_focus}"
            )
        if len({query["style"] for query in queries}) < 4:
            errors.append(f"{family_id}: fewer than 4 styles")
        if len({query["ambiguity"] for query in queries}) < 2:
            errors.append(f"{family_id}: insufficient ambiguity range")
        for query in queries:
            normalized = normalize_query(query["query"])
            if normalized in seen_queries:
                errors.append(f"{family_id}: duplicate query")
            seen_queries.add(normalized)
            if len(query["query"].split()) < 5:
                errors.append(f"{family_id}: query is too short")
    return errors


def run_codex(
    cli: Path,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    timeout_seconds: int,
) -> None:
    command = [
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
    ]
    subprocess.run(
        command,
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=output_path.parent.parent,
        check=True,
        timeout=timeout_seconds,
    )


def generate(
    functions: list[dict],
    *,
    cli: Path,
    output_dir: Path,
    batch_size: int,
    queries_per_function: int,
    max_code_chars: int,
    retries: int,
    timeout_seconds: int,
    batch_shard_count: int,
    batch_shard_index: int,
) -> None:
    raw_dir = output_dir / "raw_batches"
    raw_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "generation_schema.json"
    schema_path.write_text(
        json.dumps(schema(queries_per_function), indent=2),
        encoding="utf-8",
    )
    failures = []
    for start in range(0, len(functions), batch_size):
        batch_number = start // batch_size
        if batch_number % batch_shard_count != batch_shard_index:
            continue
        batch = functions[start : start + batch_size]
        path = raw_dir / f"batch_{batch_number:04d}.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if not validate_batch(existing, batch, queries_per_function):
                    print(f"skip valid batch {batch_number}", flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        compact = [
            compact_function(function, max_code_chars) for function in batch
        ]
        base_prompt = prompt_for(compact, queries_per_function)
        last_errors = []
        for attempt in range(retries + 1):
            prompt = base_prompt
            if last_errors:
                prompt += (
                    "\nThe previous attempt failed these checks. Regenerate the "
                    "entire batch and correct them:\n- "
                    + "\n- ".join(last_errors[:20])
                )
            temporary = path.with_suffix(".tmp")
            try:
                run_codex(cli, prompt, schema_path, temporary, timeout_seconds)
                result = json.loads(temporary.read_text(encoding="utf-8"))
                last_errors = validate_batch(
                    result, batch, queries_per_function
                )
                if not last_errors:
                    temporary.replace(path)
                    print(
                        f"generated batch {batch_number} "
                        f"({start + len(batch)}/{len(functions)})",
                        flush=True,
                    )
                    break
            except (
                OSError,
                json.JSONDecodeError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as exc:
                last_errors = [f"generation error: {exc}"]
            time.sleep(min(2 ** attempt, 8))
        else:
            failures.append({
                "batch": batch_number,
                "family_ids": [item["family_id"] for item in batch],
                "errors": last_errors,
            })
    failure_path = output_dir / (
        f"generation_failures_{batch_shard_index:02d}.json"
    )
    failure_path.write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"{len(failures)} batches failed validation")


def assemble(functions: list[dict], output_dir: Path) -> dict:
    by_id = {function["family_id"]: function for function in functions}
    seed_targets = {
        family_id: OPERATIONS[index % len(OPERATIONS)]
        for index, family_id in enumerate(sorted(by_id))
    }
    rows = []
    for path in sorted((output_dir / "raw_batches").glob("batch_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        for item in result["functions"]:
            function = by_id.get(item["family_id"])
            if function is None:
                continue
            target_operation = seed_targets[function["family_id"]]
            seed_candidates = [
                index
                for index, query in enumerate(item["queries"])
                if query["operation"] == target_operation
            ]
            seed_index = seed_candidates[0]
            for query_index, query in enumerate(item["queries"]):
                query_id = stable_id(
                    function["family_id"], query["query"], "llm-grounded-v1"
                )
                rows.append({
                    "pair_id": query_id,
                    "family_id": function["family_id"],
                    "split": function["split"],
                    "source": function["source"],
                    "repository": function["repository"],
                    "source_url": function["source_url"],
                    "revision": function["revision"],
                    "license": function["license"],
                    "file_path": function["file_path"],
                    "function_name": function["function_name"],
                    "docstring": function["docstring"],
                    "code": function["code"],
                    "query": query["query"],
                    "query_style": query["style"],
                    "query_domain": query["domain"],
                    "query_operation": query["operation"],
                    "query_concerns": json.dumps(query["concerns"]),
                    "ambiguity": query["ambiguity"],
                    "grounding": query["grounding"],
                    "generation_role": (
                        "seed" if query_index == seed_index else "variant"
                    ),
                    "generator": "codex_cli",
                    "generation_version": "llm-grounded-v1",
                })

    output_csv = output_dir / "llm_grounded_pairs.csv"
    output_jsonl = output_dir / "llm_grounded_pairs.jsonl"
    fields = list(rows[0]) if rows else []
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    family_rows = defaultdict(list)
    for row in rows:
        family_rows[row["family_id"]].append(row)
    openings = Counter(
        " ".join(normalize_query(row["query"]).split()[:5]) for row in rows
    )
    operation_gaps = {
        family_id: sorted(
            set(OPERATIONS)
            - {row["query_operation"] for row in family_records}
        )
        for family_id, family_records in family_rows.items()
        if set(OPERATIONS)
        - {row["query_operation"] for row in family_records}
    }
    report = {
        "rows": len(rows),
        "seeds": sum(row["generation_role"] == "seed" for row in rows),
        "variants": sum(row["generation_role"] == "variant" for row in rows),
        "function_families": len({row["family_id"] for row in rows}),
        "unique_queries": len({normalize_query(row["query"]) for row in rows}),
        "quality_checks": {
            "duplicate_queries": (
                len(rows)
                - len({normalize_query(row["query"]) for row in rows})
            ),
            "families_crossing_splits": sum(
                len({row["split"] for row in records}) != 1
                for records in family_rows.values()
            ),
            "families_missing_operations": operation_gaps,
            "families_with_wrong_size": {
                family_id: len(records)
                for family_id, records in family_rows.items()
                if len(records) != 8
            },
            "top_five_word_openings": openings.most_common(20),
            "empty_grounding": sum(not row["grounding"].strip() for row in rows),
        },
        "counts": {
            "source": Counter(row["source"] for row in rows),
            "split": Counter(row["split"] for row in rows),
            "operation": Counter(row["query_operation"] for row in rows),
            "domain": Counter(row["query_domain"] for row in rows),
            "style": Counter(row["query_style"] for row in rows),
            "ambiguity": Counter(row["ambiguity"] for row in rows),
            "concern": Counter(
                concern
                for row in rows
                for concern in json.loads(row["query_concerns"])
            ),
        },
    }
    (output_dir / "llm_dataset_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("router_pair_data/code_query_pairs.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("router_pair_data/llm_grounded"),
    )
    parser.add_argument("--cli", type=Path, default=Path(".codex-cli.exe"))
    parser.add_argument("--functions", type=int, default=300)
    parser.add_argument("--queries-per-function", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-code-chars", type=int, default=7000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--batch-shard-count", type=int, default=1)
    parser.add_argument("--batch-shard-index", type=int, default=0)
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--exclude-input", type=Path)
    parser.add_argument("--focus-gaps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_functions = load_unique_functions(args.input)
    if args.focus_gaps:
        functions = gap_targeted_sample(
            all_functions,
            args.functions,
            load_family_ids(args.exclude_input),
        )
    else:
        functions = balanced_sample(all_functions, args.functions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selected_functions.json"
    selection_path.write_text(
        json.dumps(functions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.assemble_only:
        if not args.cli.exists():
            raise FileNotFoundError(f"Codex CLI not found: {args.cli}")
        generate(
            functions,
            cli=args.cli,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            queries_per_function=args.queries_per_function,
            max_code_chars=args.max_code_chars,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            batch_shard_count=args.batch_shard_count,
            batch_shard_index=args.batch_shard_index,
        )
    if not args.generate_only:
        report = assemble(functions, args.output_dir)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
