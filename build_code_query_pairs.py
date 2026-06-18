"""Build labeled (code, query) pairs from benchmarks and Python repositories."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


OPERATIONS = ("explain", "debug", "optimize", "review", "generate", "refactor")
DOMAINS = (
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
)
CONCERNS = (
    "security",
    "concurrency",
    "correctness",
    "performance",
    "reliability",
    "maintainability",
)

DEFAULT_REPOS = (
    "Textualize/rich",
    "pallets/flask",
    "pallets/click",
    "encode/httpx",
    "encode/starlette",
    "pytest-dev/pytest",
    "python-attrs/attrs",
    "marshmallow-code/marshmallow",
    "jd/tenacity",
    "hynek/structlog",
    "agronholm/anyio",
    "python-trio/trio",
    "aio-libs/aiohttp",
    "pydantic/pydantic",
    "psf/black",
    "sqlalchemy/alembic",
    "tox-dev/tox",
    "python-poetry/poetry-core",
    "dateutil/dateutil",
    "Delgan/loguru",
    "scrapy/parsel",
    "pallets/jinja",
    "sphinx-doc/sphinx",
    "pyca/cryptography",
)

DOMAIN_TERMS = {
    "machine_learning": ("model", "tensor", "train", "predict", "feature", "loss"),
    "backend_api": ("request", "response", "route", "http", "api", "webhook"),
    "database": ("sql", "query", "database", "session", "transaction", "schema"),
    "concurrency": ("async", "await", "task", "lock", "thread", "worker", "future"),
    "frontend": ("render", "widget", "view", "template", "style", "component"),
    "systems_programming": ("buffer", "socket", "process", "signal", "memory", "byte"),
    "security_crypto": (
        "encrypt", "decrypt", "hash", "secret", "token", "password", "certificate",
    ),
    "testing": ("test", "fixture", "mock", "assert", "pytest"),
    "devops_cli": ("cli", "command", "config", "environment", "deploy", "build"),
    "data_processing": ("csv", "json", "parse", "data", "row", "column", "serialize"),
    "algorithms": (
        "sort", "search", "tree", "graph", "path", "heap", "sequence", "recursive",
    ),
}

REPO_DOMAIN_HINTS = {
    "flask": "backend_api",
    "httpx": "backend_api",
    "starlette": "backend_api",
    "aiohttp": "backend_api",
    "alembic": "database",
    "anyio": "concurrency",
    "trio": "concurrency",
    "cryptography": "security_crypto",
    "tox": "devops_cli",
    "poetry-core": "devops_cli",
    "click": "devops_cli",
    "rich": "frontend",
    "jinja": "frontend",
    "pytest": "testing",
    "parsel": "data_processing",
}

STYLE_NAMES = (
    "pr_description",
    "issue_comment",
    "code_review",
    "maintenance_request",
    "investigation_note",
    "mixed_request",
)


@dataclass(frozen=True)
class FunctionRecord:
    function_id: str
    source: str
    repository: str
    source_url: str
    revision: str
    license: str
    file_path: str
    qualified_name: str
    docstring: str
    code: str
    domain: str


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    family_id: str
    split: str
    source: str
    repository: str
    source_url: str
    revision: str
    license: str
    file_path: str
    function_name: str
    docstring: str
    code: str
    query: str
    query_style: str
    query_domain: str
    query_operation: str
    query_concerns: str
    ambiguity: str


def stable_id(*parts: str, length: int = 24) -> str:
    text = "\0".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def clean_text(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit].rstrip()


def words(name: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    return value.replace("_", " ").strip().lower()


def infer_domain(name: str, docstring: str, code: str, repo: str = "") -> str:
    repo_name = repo.rsplit("/", 1)[-1].lower()
    if repo_name in REPO_DOMAIN_HINTS:
        return REPO_DOMAIN_HINTS[repo_name]
    haystack = f"{name} {docstring} {code[:1200]}".lower()
    scores = {
        domain: sum(term in haystack for term in terms)
        for domain, terms in DOMAIN_TERMS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] else "general"


def function_source(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    start = min(
        [node.lineno]
        + [decorator.lineno for decorator in getattr(node, "decorator_list", [])]
    )
    return "\n".join(lines[start - 1 : node.end_lineno])


def extract_python_functions(
    path: Path,
    *,
    source_name: str,
    repository: str,
    source_url: str,
    revision: str,
    license_name: str,
    root: Path,
) -> list[FunctionRecord]:
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    records = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        docstring = ast.get_docstring(node, clean=True)
        if not docstring or node.name.startswith("_") or node.end_lineno - node.lineno < 2:
            continue
        names = [node.name]
        parent = parents.get(node)
        while isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(parent.name)
            parent = parents.get(parent)
        qualified = ".".join(reversed(names))
        code = function_source(text, node)
        relative = path.relative_to(root).as_posix()
        function_id = stable_id(source_name, repository, revision, relative, qualified)
        records.append(FunctionRecord(
            function_id=function_id,
            source=source_name,
            repository=repository,
            source_url=source_url,
            revision=revision,
            license=license_name,
            file_path=relative,
            qualified_name=qualified,
            docstring=clean_text(docstring, 500),
            code=code,
            domain=infer_domain(qualified, docstring, code, repository),
        ))
    return records


def split_for_family(family_id: str) -> str:
    bucket = int(hashlib.sha256(family_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def concern_for(operation: str, index: int) -> tuple[str, ...]:
    primary = {
        "explain": "correctness",
        "debug": "correctness",
        "optimize": "performance",
        "review": "maintainability",
        "generate": "reliability",
        "refactor": "maintainability",
    }[operation]
    if index % 5 == 0:
        return (primary, CONCERNS[(index + 2) % len(CONCERNS)])
    if index % 7 == 0:
        return ()
    return (primary,)


def query_variants(record: FunctionRecord, count: int) -> list[dict]:
    name = words(record.qualified_name.split(".")[-1])
    purpose = clean_text(record.docstring, 130)
    subject = f"`{record.qualified_name}`"
    domain = record.domain
    templates = [
        (
            "pr_description", "debug", "high",
            "Fix the intermittent failure in {subject}. It usually works, but "
            "with empty or slightly malformed input it returns something plausible "
            "instead of raising. Keep the public behavior unchanged for normal calls.",
        ),
        (
            "issue_comment", "optimize", "medium",
            "Why is {subject} so slow after a few thousand calls? The result looks "
            "right, so I am not sure whether the problem is the loop, retained state, "
            "or the way {name} builds its output.",
        ),
        (
            "code_review", "refactor", "low",
            "This works, but the control flow in {subject} feels harder to follow "
            "than it needs to be. Could this be made more Pythonic without changing "
            "the edge-case behavior?",
        ),
        (
            "maintenance_request", "review", "high",
            "Can someone sanity-check {subject}? I am mostly worried about hidden "
            "assumptions in \"{purpose}\" and whether callers can trigger surprising "
            "behavior. Suggestions are enough; no rewrite yet.",
        ),
        (
            "investigation_note", "explain", "medium",
            "I can follow the happy path in {subject}, but not why it handles the "
            "boundary this way. Is that required by \"{purpose}\", or is it just an "
            "implementation detail?",
        ),
        (
            "mixed_request", "generate", "high",
            "We need coverage around {subject} before touching it. Add focused tests "
            "for the documented behavior and one awkward input; if the current result "
            "is intentional, make that clear in the test names.",
        ),
        (
            "issue_comment", "debug", "medium",
            "{subject} occasionally gives a different answer for equivalent inputs. "
            "Could there be mutation or ordering involved, or am I calling it wrong?",
        ),
        (
            "pr_description", "optimize", "high",
            "Reduce the memory growth in {subject} when processing a long stream. "
            "Please avoid a broad rewrite because downstream code relies on the exact "
            "return shape.",
        ),
        (
            "code_review", "review", "low",
            "The name {name} sounds straightforward, but this function is doing more "
            "than I expected. Is the extra branching justified, and are we missing a "
            "case that the docstring quietly implies?",
        ),
        (
            "maintenance_request", "refactor", "medium",
            "Can we separate validation from the main logic in {subject}? It is "
            "currently working, though the error path and normal path are tangled "
            "enough that I am nervous about changing either.",
        ),
    ]
    by_operation = defaultdict(list)
    for template_record in templates:
        by_operation[template_record[1]].append(template_record)
    seed = int(record.function_id[:8], 16)
    chosen = []
    for operation_index, operation in enumerate(OPERATIONS):
        choices = by_operation[operation]
        chosen.append(choices[(seed + operation_index) % len(choices)])
    unused = [template for template in templates if template not in chosen]
    random.Random(seed).shuffle(unused)
    chosen.extend(unused)

    selected = []
    for index, (style, operation, ambiguity, template) in enumerate(chosen[:count]):
        concerns = concern_for(operation, index + seed)
        query = template.format(
            subject=subject,
            name=name,
            purpose=purpose or f"the documented {name} behavior",
        )
        if concerns == ("security",) or "security" in concerns:
            query += " Please also consider whether untrusted values change the risk."
        elif "concurrency" in concerns:
            query += " This may also run from overlapping tasks."
        elif "reliability" in concerns and operation != "generate":
            query += " It has to behave predictably during retries."
        selected.append({
            "query": query,
            "style": style,
            "operation": operation,
            "domain": domain,
            "concerns": concerns,
            "ambiguity": ambiguity,
        })
    return selected


def benchmark_records(limit_humaneval: int, limit_mbpp: int) -> list[FunctionRecord]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements-data.txt to load benchmarks") from exc

    records = []
    specs = [
        (
            "humaneval",
            "openai/openai_humaneval",
            "test",
            limit_humaneval,
            "prompt",
            "canonical_solution",
            "task_id",
            "https://github.com/openai/human-eval",
            "MIT",
        ),
        (
            "mbpp",
            "google-research-datasets/mbpp",
            "train",
            limit_mbpp,
            "code",
            None,
            "task_id",
            "https://github.com/google-research/google-research/tree/master/mbpp",
            "CC BY 4.0",
        ),
    ]
    for source_name, dataset_name, split, limit, code_key, solution_key, id_key, url, license_name in specs:
        dataset = load_dataset(dataset_name, split=split)
        indices = list(range(len(dataset)))
        random.Random(19).shuffle(indices)
        for row_index in indices[:limit]:
            row = dataset[row_index]
            code = row.get(code_key, "")
            if solution_key:
                code += row.get(solution_key, "")
            task_id = str(row.get(id_key, row_index))
            doc = row.get("text") or row.get("prompt") or ""
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            node = next(
                (
                    item for item in tree.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ),
                None,
            )
            if node is None:
                continue
            docstring = ast.get_docstring(node, clean=True) or clean_text(doc, 500)
            qualified = node.name
            function_id = stable_id(source_name, task_id)
            records.append(FunctionRecord(
                function_id=function_id,
                source=source_name,
                repository=dataset_name,
                source_url=url,
                revision=task_id,
                license=license_name,
                file_path=task_id,
                qualified_name=qualified,
                docstring=clean_text(docstring, 500),
                code=code.strip(),
                domain=infer_domain(qualified, docstring, code),
            ))
    return records


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def clone_or_update(repo: str, cache_dir: Path) -> Path:
    target = cache_dir / repo.replace("/", "__")
    url = f"https://github.com/{repo}.git"
    if not target.exists():
        git("clone", "--depth", "1", "--filter=blob:none", url, str(target))
    return target


def detect_license(root: Path) -> str:
    candidates = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.name.lower().startswith(("license", "copying"))
    )
    if not candidates:
        return "See repository license"
    text = candidates[0].read_text(encoding="utf-8", errors="ignore")[:3000].lower()
    for marker, name in (
        ("apache license", "Apache-2.0"),
        ("mit license", "MIT"),
        ("bsd 3-clause", "BSD-3-Clause"),
        ("redistribution and use in source and binary forms", "BSD"),
        ("mozilla public license", "MPL-2.0"),
        ("python software foundation license", "PSF-2.0"),
    ):
        if marker in text:
            return name
    return candidates[0].name


def repository_records(
    repos: Iterable[str],
    cache_dir: Path,
    min_files: int,
    max_files: int,
    max_repos: int,
    functions_per_repo: int,
) -> tuple[list[FunctionRecord], list[dict]]:
    records = []
    audit = []
    accepted = 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    for repo in repos:
        if accepted >= max_repos:
            break
        try:
            root = clone_or_update(repo, cache_dir)
            revision = git("rev-parse", "HEAD", cwd=root)
        except (subprocess.CalledProcessError, OSError) as exc:
            audit.append({"repository": repo, "status": "clone_failed", "error": str(exc)})
            continue
        files = [
            path for path in root.rglob("*.py")
            if ".git" not in path.parts
            and not any(part in {"venv", ".venv", "build", "dist"} for part in path.parts)
        ]
        if not min_files <= len(files) <= max_files:
            audit.append({
                "repository": repo,
                "status": "outside_file_range",
                "python_files": len(files),
                "revision": revision,
            })
            continue
        license_name = detect_license(root)
        extracted = []
        for path in files:
            extracted.extend(extract_python_functions(
                path,
                source_name="github_repository",
                repository=repo,
                source_url=f"https://github.com/{repo}",
                revision=revision,
                license_name=license_name,
                root=root,
            ))
        random.Random(stable_id(repo)).shuffle(extracted)
        selected = extracted[:functions_per_repo]
        if not selected:
            audit.append({
                "repository": repo,
                "status": "no_docstring_functions",
                "python_files": len(files),
            })
            continue
        records.extend(selected)
        accepted += 1
        audit.append({
            "repository": repo,
            "status": "accepted",
            "python_files": len(files),
            "docstring_functions": len(extracted),
            "selected_functions": len(selected),
            "revision": revision,
            "license": license_name,
        })
    return records, audit


def build_pairs(functions: Iterable[FunctionRecord], queries_per_function: int) -> list[PairRecord]:
    pairs = []
    seen_queries = set()
    for function in functions:
        for variant in query_variants(function, queries_per_function):
            normalized = re.sub(r"\W+", " ", variant["query"].lower()).strip()
            key = (function.function_id, normalized)
            if key in seen_queries:
                continue
            seen_queries.add(key)
            pair_id = stable_id(function.function_id, variant["query"])
            pairs.append(PairRecord(
                pair_id=pair_id,
                family_id=function.function_id,
                split=split_for_family(function.function_id),
                source=function.source,
                repository=function.repository,
                source_url=function.source_url,
                revision=function.revision,
                license=function.license,
                file_path=function.file_path,
                function_name=function.qualified_name,
                docstring=function.docstring,
                code=function.code,
                query=variant["query"],
                query_style=variant["style"],
                query_domain=variant["domain"],
                query_operation=variant["operation"],
                query_concerns=json.dumps(variant["concerns"]),
                ambiguity=variant["ambiguity"],
            ))
    return pairs


def write_outputs(output_dir: Path, pairs: list[PairRecord], audit: list[dict]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(PairRecord.__dataclass_fields__)
    csv_path = output_dir / "code_query_pairs.csv"
    jsonl_path = output_dir / "code_query_pairs.jsonl"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(pair) for pair in pairs)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")
    with (output_dir / "repository_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    report = {
        "rows": len(pairs),
        "function_families": len({pair.family_id for pair in pairs}),
        "repositories": len({
            pair.repository for pair in pairs if pair.source == "github_repository"
        }),
        "counts": {
            "source": Counter(pair.source for pair in pairs),
            "split": Counter(pair.split for pair in pairs),
            "operation": Counter(pair.query_operation for pair in pairs),
            "domain": Counter(pair.query_domain for pair in pairs),
            "style": Counter(pair.query_style for pair in pairs),
            "ambiguity": Counter(pair.ambiguity for pair in pairs),
        },
        "files": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
        },
        "training_note": (
            "These rows train request/code signal models. For a router quality model, "
            "join each pair to per-candidate observed success/quality/latency outcomes."
        ),
    }
    with (output_dir / "dataset_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("router_pair_data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".code_query_repo_cache"))
    parser.add_argument("--queries-per-function", type=int, default=8, choices=range(5, 11))
    parser.add_argument("--humaneval-functions", type=int, default=164)
    parser.add_argument("--mbpp-functions", type=int, default=500)
    parser.add_argument("--max-repos", type=int, default=15)
    parser.add_argument("--functions-per-repo", type=int, default=60)
    parser.add_argument("--min-python-files", type=int, default=100)
    parser.add_argument("--max-python-files", type=int, default=500)
    parser.add_argument("--repos", nargs="*", default=list(DEFAULT_REPOS))
    parser.add_argument("--benchmarks-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    functions = benchmark_records(args.humaneval_functions, args.mbpp_functions)
    audit = []
    if not args.benchmarks_only:
        repo_functions, audit = repository_records(
            args.repos,
            args.cache_dir,
            args.min_python_files,
            args.max_python_files,
            args.max_repos,
            args.functions_per_repo,
        )
        functions.extend(repo_functions)
    pairs = build_pairs(functions, args.queries_per_function)
    report = write_outputs(args.output_dir, pairs, audit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
