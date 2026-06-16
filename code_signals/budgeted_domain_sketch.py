"""Token-budgeted AST-unit sketching for code-domain embeddings."""

import ast
import copy
import math
import re
from collections import Counter
from dataclasses import dataclass

from .domain import API_FINGERPRINTS, IMPORT_FINGERPRINTS
from .parser import get_node_text, is_async_node, walk


COMPONENT_SHARES = {
    "imports": 0.10,
    "signatures": 0.15,
    "literals_exceptions": 0.10,
    "assignments_calls": 0.20,
    "control": 0.15,
    "statements": 0.30,
}
CONTROL_TYPES = {
    "if_statement",
    "elif_clause",
    "else_clause",
    "for_statement",
    "while_statement",
    "with_statement",
    "try_statement",
    "except_clause",
    "finally_clause",
    "match_statement",
    "case_clause",
}
MUTATION_METHODS = {
    "add", "append", "clear", "discard", "extend", "insert", "pop",
    "remove", "reverse", "setdefault", "sort", "update",
}
DOMAIN_LITERAL_PATTERN = re.compile(
    r"https?://|wss?://|/api(?:/|$)|"
    r"\b(?:select|insert|update|delete|join|where|create table)\b|"
    r"<[A-Za-z][^>]*>|\\[AbBdDsSwWZ]|\(\?[=!<:]",
    re.IGNORECASE,
)
IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_]\w*\b")
GENERIC_IDENTIFIERS = {
    "data", "item", "items", "result", "results", "value", "values",
    "self", "cls", "args", "kwargs", "i", "j", "x", "y",
}


@dataclass
class Candidate:
    category: str
    text: str
    start_byte: int
    domain_score: float = 0.0
    data_flow_score: float = 0.0
    structural_score: float = 0.0
    uniqueness_score: float = 0.0
    function_score: float = 0.0
    token_cost: int = 0

    def utility(
        self,
        *,
        include_data_flow: bool,
        include_function_importance: bool,
    ) -> float:
        value = self.domain_score + self.structural_score + self.uniqueness_score
        if include_data_flow:
            value += self.data_flow_score
        if include_function_importance:
            value += self.function_score
        return max(value, 0.1)


@dataclass
class CoherentCandidate:
    category: str
    node: ast.AST
    text: str
    source_position: tuple[int, int]
    domain_score: float = 0.0
    data_flow_score: float = 0.0
    structural_score: float = 0.0
    uniqueness_score: float = 0.0
    function_score: float = 0.0
    token_cost: int = 0

    def utility(
        self,
        *,
        include_data_flow: bool,
        include_function_importance: bool,
    ) -> float:
        value = self.domain_score + self.structural_score + self.uniqueness_score
        if include_data_flow:
            value += self.data_flow_score
        if include_function_importance:
            value += self.function_score
        return max(value, 0.1)


def _signature(node) -> str:
    name = node.child_by_field_name("name")
    parameters = node.child_by_field_name("parameters")
    if name is None or parameters is None:
        return ""
    prefix = "async def" if is_async_node(node) else "def"
    result = f"{prefix} {get_node_text(name)}{get_node_text(parameters)}"
    return_type = node.child_by_field_name("return_type")
    if return_type is not None:
        result += f" -> {get_node_text(return_type)}"
    return result + ":"


def _class_header(node) -> str:
    name = node.child_by_field_name("name")
    if name is None:
        return ""
    superclasses = node.child_by_field_name("superclasses")
    suffix = get_node_text(superclasses) if superclasses is not None else ""
    return f"class {get_node_text(name)}{suffix}:"


def _header(node) -> str:
    body = node.child_by_field_name("body")
    if body is None:
        return get_node_text(node).splitlines()[0]
    text = node.text[: body.start_byte - node.start_byte].decode(
        "utf-8", errors="replace"
    )
    return re.sub(r"\s+", " ", text).strip()


def _call_name(node) -> str:
    function = node.child_by_field_name("function")
    if function is None and node.named_children:
        function = node.named_children[0]
    return get_node_text(function) if function is not None else ""


def _enclosing_function(node):
    current = node.parent
    while current is not None:
        if current.type == "function_definition":
            return current
        current = current.parent
    return None


def _candidate_key(candidate: Candidate):
    return candidate.category, re.sub(r"\s+", " ", candidate.text).strip()


def extract_candidates(code: str, root_node) -> list[Candidate]:
    candidates = []
    covered = set()

    def add(category, text, node):
        text = text.strip()
        if not text:
            return
        candidate = Candidate(category, text, node.start_byte)
        key = _candidate_key(candidate)
        if key not in covered:
            candidates.append(candidate)
            covered.add(key)

    for node in walk(root_node):
        if node.type in {"import_statement", "import_from_statement"}:
            add("imports", get_node_text(node), node)
        elif node.type == "class_definition":
            add("signatures", _class_header(node), node)
        elif node.type == "decorator":
            add("signatures", get_node_text(node), node)
        elif node.type == "function_definition":
            add("signatures", _signature(node), node)
        elif node.type == "string" and DOMAIN_LITERAL_PATTERN.search(
            get_node_text(node)
        ):
            add("literals_exceptions", get_node_text(node), node)
        elif node.type in {"raise_statement", "except_clause"}:
            add("literals_exceptions", _header(node), node)
        elif node.type in {"assignment", "augmented_assignment"}:
            right = node.child_by_field_name("right")
            left = node.child_by_field_name("left")
            has_call = right is not None and any(
                child.type == "call" for child in walk(right)
            )
            mutation = (
                node.type == "augmented_assignment"
                or left is not None and left.type in {"attribute", "subscript"}
            )
            category = "assignments_calls" if has_call or mutation else "statements"
            add(category, get_node_text(node), node)
        elif node.type == "call":
            parent = node.parent
            if parent is not None and parent.type in {
                "assignment", "augmented_assignment", "call",
            }:
                continue
            add("assignments_calls", get_node_text(node), node)
        elif node.type in CONTROL_TYPES:
            add("control", _header(node), node)
        elif node.type in {
            "return_statement", "yield", "delete_statement",
        }:
            add("statements", get_node_text(node), node)

    identifier_frequency = Counter(
        IDENTIFIER_PATTERN.findall(code)
    )
    text_frequency = Counter(
        token
        for candidate in candidates
        for token in set(IDENTIFIER_PATTERN.findall(candidate.text))
    )
    function_stats = Counter()
    for candidate in candidates:
        function = next(
            (
                node
                for node in walk(root_node)
                if node.start_byte <= candidate.start_byte < node.end_byte
                and node.type == "function_definition"
            ),
            None,
        )
        if function is not None:
            function_stats[function.start_byte] += (
                candidate.category in {"assignments_calls", "control"}
            )

    all_api_patterns = [
        pattern
        for patterns in API_FINGERPRINTS.values()
        for pattern in patterns
    ]
    all_imports = [
        name
        for names in IMPORT_FINGERPRINTS.values()
        for name in names
    ]
    for candidate in candidates:
        lower = candidate.text.lower()
        candidate.domain_score = (
            1.5 * sum(pattern.lower() in lower for pattern in all_api_patterns)
            + 1.0 * sum(
                re.search(rf"\b{re.escape(name.lower())}\b", lower) is not None
                for name in all_imports
            )
            + (1.5 if DOMAIN_LITERAL_PATTERN.search(candidate.text) else 0.0)
        )
        identifiers = IDENTIFIER_PATTERN.findall(candidate.text)
        informative = [
            name for name in identifiers if name not in GENERIC_IDENTIFIERS
        ]
        candidate.data_flow_score = min(
            4.0,
            sum(math.log1p(identifier_frequency[name]) for name in informative)
            / max(len(informative), 1),
        )
        candidate.structural_score = {
            "imports": 0.5,
            "signatures": 1.5,
            "literals_exceptions": 2.0,
            "assignments_calls": 2.5,
            "control": 2.0,
            "statements": 1.0,
        }[candidate.category]
        if any(
            f".{method}(" in candidate.text
            for method in MUTATION_METHODS
        ):
            candidate.structural_score += 1.0
        candidate.uniqueness_score = sum(
            1.0 / text_frequency[name]
            for name in set(informative)
            if text_frequency[name]
        ) / max(len(set(informative)), 1)
        function = _enclosing_function(
            next(
                (
                    node for node in walk(root_node)
                    if node.start_byte == candidate.start_byte
                ),
                root_node,
            )
        )
        if function is not None:
            candidate.function_score = min(
                function_stats[function.start_byte] * 0.25,
                2.0,
            )
    return candidates


def _token_cost(tokenizer, text: str) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    )


def _select(
    candidates,
    tokenizer,
    budget,
    *,
    include_data_flow,
    include_function_importance,
):
    for candidate in candidates:
        candidate.token_cost = max(_token_cost(tokenizer, candidate.text), 1)

    selected = []
    selected_ids = set()
    remaining = budget
    by_category = {
        category: [candidate for candidate in candidates if candidate.category == category]
        for category in COMPONENT_SHARES
    }

    def ranked(values):
        return sorted(
            values,
            key=lambda candidate: (
                -candidate.utility(
                    include_data_flow=include_data_flow,
                    include_function_importance=include_function_importance,
                ) / candidate.token_cost,
                candidate.start_byte,
            ),
        )

    for category, share in COMPONENT_SHARES.items():
        allowance = min(int(budget * share), remaining)
        used = 0
        for candidate in ranked(by_category[category]):
            identity = id(candidate)
            if identity in selected_ids:
                continue
            if used + candidate.token_cost > allowance:
                continue
            selected.append(candidate)
            selected_ids.add(identity)
            used += candidate.token_cost
            remaining -= candidate.token_cost

    omitted = [candidate for candidate in candidates if id(candidate) not in selected_ids]
    for candidate in ranked(omitted):
        if candidate.token_cost > remaining:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        remaining -= candidate.token_cost

    omitted = [candidate for candidate in candidates if id(candidate) not in selected_ids]
    selected_utility = sum(
        candidate.utility(
            include_data_flow=include_data_flow,
            include_function_importance=include_function_importance,
        )
        for candidate in selected
    )
    omitted_utility = sum(
        candidate.utility(
            include_data_flow=include_data_flow,
            include_function_importance=include_function_importance,
        )
        for candidate in omitted
    )
    return selected, omitted_utility / max(selected_utility, 1e-9)


def build_budgeted_domain_sketch(
    code: str,
    root_node,
    tokenizer,
    *,
    base_budget: int = 512,
    expanded_budget: int = 1024,
    expansion_threshold: float = 0.30,
    include_data_flow: bool = True,
    include_function_importance: bool = True,
    dynamic_expansion: bool = True,
) -> tuple[str, dict]:
    """Build a budgeted sketch and return it with selection diagnostics."""
    candidates = extract_candidates(code, root_node)
    header_template = (
        "# domain-sketch-v7-budgeted "
        f"budget={expanded_budget}"
    )
    content_base_budget = max(
        base_budget - _token_cost(tokenizer, header_template),
        1,
    )
    selected, omitted_ratio = _select(
        candidates,
        tokenizer,
        content_base_budget,
        include_data_flow=include_data_flow,
        include_function_importance=include_function_importance,
    )
    budget = base_budget
    expanded = False
    if dynamic_expansion and omitted_ratio > expansion_threshold:
        content_expanded_budget = max(
            expanded_budget - _token_cost(tokenizer, header_template),
            1,
        )
        selected, omitted_ratio = _select(
            candidates,
            tokenizer,
            content_expanded_budget,
            include_data_flow=include_data_flow,
            include_function_importance=include_function_importance,
        )
        budget = expanded_budget
        expanded = True

    selected.sort(key=lambda candidate: candidate.start_byte)
    header = (
        "# domain-sketch-v7-budgeted "
        f"budget={budget}"
    )
    sketch = "\n".join([header] + [candidate.text for candidate in selected])
    return sketch, {
        "budget": budget,
        "expanded": expanded,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "omitted_utility_ratio": round(omitted_ratio, 4),
        "token_count": _token_cost(tokenizer, sketch),
    }


def build_coherent_budgeted_domain_sketch(
    code: str,
    root_node,
    tokenizer,
    *,
    base_budget: int = 512,
    expanded_budget: int = 1024,
    expansion_threshold: float = 0.30,
) -> tuple[str, dict]:
    """Preserve natural source whenever the complete snippet fits the budget.

    This is the coherent V7 short-code path. It intentionally avoids candidate
    extraction and reconstruction when compression is unnecessary, eliminating
    parent/child duplication and preserving syntax, indentation, and context.
    Long-code coherent pruning remains a separate experiment; until then, the
    existing candidate selector is used only after the full source exceeds the
    base budget.
    """
    source = code.strip()
    source_tokens = _token_cost(tokenizer, source)
    if source_tokens <= base_budget:
        return source, {
            "budget": base_budget,
            "expanded": False,
            "candidate_count": None,
            "selected_count": None,
            "omitted_utility_ratio": 0.0,
            "token_count": source_tokens,
            "mode": "full_source",
        }

    sketch, details = build_budgeted_domain_sketch(
        code,
        root_node,
        tokenizer,
        base_budget=base_budget,
        expanded_budget=expanded_budget,
        expansion_threshold=expansion_threshold,
        include_data_flow=True,
        include_function_importance=True,
        dynamic_expansion=True,
    )
    details["mode"] = "legacy_candidate_fallback"
    return sketch, details


def _coherent_category(node: ast.AST) -> str | None:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "imports"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "signatures"
    if isinstance(node, (ast.Raise, ast.ExceptHandler)):
        return "literals_exceptions"
    if isinstance(
        node,
        (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.With,
            ast.AsyncWith,
            ast.Try,
            ast.Match,
        ),
    ):
        return "control"
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        value = getattr(node, "value", None)
        target = getattr(node, "target", None)
        targets = getattr(node, "targets", [])
        mutation = isinstance(target, (ast.Attribute, ast.Subscript)) or any(
            isinstance(item, (ast.Attribute, ast.Subscript))
            for item in targets
        )
        has_call = value is not None and any(
            isinstance(item, ast.Call) for item in ast.walk(value)
        )
        if mutation or has_call:
            return "assignments_calls"
        return "statements"
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return "assignments_calls"
    if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom, ast.Delete)):
        return "statements"
    return None


def _coherent_candidate_text(node: ast.AST) -> str:
    clone = copy.deepcopy(node)
    if isinstance(clone, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        clone.body = [ast.Pass()]
    elif isinstance(
        clone,
        (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.With,
            ast.AsyncWith,
        ),
    ):
        clone.body = [ast.Pass()]
        if hasattr(clone, "orelse"):
            clone.orelse = []
    elif isinstance(clone, ast.Try):
        clone.body = [ast.Pass()]
        clone.handlers = []
        clone.orelse = []
        clone.finalbody = []
    elif isinstance(clone, ast.Match):
        clone.cases = []
    elif isinstance(clone, ast.ExceptHandler):
        clone.body = [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(clone)).strip()


def _coherent_candidates(code: str) -> tuple[ast.Module, list[CoherentCandidate]]:
    tree = ast.parse(code)
    parents = {}
    enclosing_function = {}

    def index(node, parent=None, function=None):
        if parent is not None:
            parents[id(node)] = parent
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node
        if function is not None:
            enclosing_function[id(node)] = function
        for child in ast.iter_child_nodes(node):
            index(child, node, function)

    index(tree)
    candidates = []
    seen_nodes = set()
    for node in ast.walk(tree):
        category = _coherent_category(node)
        if category is None or id(node) in seen_nodes:
            continue
        text = _coherent_candidate_text(node)
        if not text:
            continue
        candidates.append(
            CoherentCandidate(
                category=category,
                node=node,
                text=text,
                source_position=(
                    getattr(node, "lineno", 0),
                    getattr(node, "col_offset", 0),
                ),
            )
        )
        seen_nodes.add(id(node))

    identifier_frequency = Counter(IDENTIFIER_PATTERN.findall(code))
    candidate_frequency = Counter(
        name
        for candidate in candidates
        for name in set(IDENTIFIER_PATTERN.findall(candidate.text))
    )
    function_density = Counter(
        id(enclosing_function[id(candidate.node)])
        for candidate in candidates
        if id(candidate.node) in enclosing_function
        and candidate.category in {"assignments_calls", "control"}
    )
    all_api_patterns = [
        pattern.lower()
        for patterns in API_FINGERPRINTS.values()
        for pattern in patterns
    ]
    all_imports = [
        name.lower()
        for names in IMPORT_FINGERPRINTS.values()
        for name in names
    ]
    for candidate in candidates:
        lower = candidate.text.lower()
        candidate.domain_score = (
            1.5 * sum(pattern in lower for pattern in all_api_patterns)
            + sum(
                re.search(rf"\b{re.escape(name)}\b", lower) is not None
                for name in all_imports
            )
            + (1.5 if DOMAIN_LITERAL_PATTERN.search(candidate.text) else 0.0)
        )
        identifiers = [
            name
            for name in IDENTIFIER_PATTERN.findall(candidate.text)
            if name not in GENERIC_IDENTIFIERS
        ]
        candidate.data_flow_score = min(
            4.0,
            sum(math.log1p(identifier_frequency[name]) for name in identifiers)
            / max(len(identifiers), 1),
        )
        candidate.structural_score = {
            "imports": 0.5,
            "signatures": 1.5,
            "literals_exceptions": 2.0,
            "assignments_calls": 2.5,
            "control": 2.0,
            "statements": 1.0,
        }[candidate.category]
        candidate.uniqueness_score = sum(
            1.0 / candidate_frequency[name]
            for name in set(identifiers)
            if candidate_frequency[name]
        ) / max(len(set(identifiers)), 1)
        function = enclosing_function.get(id(candidate.node))
        if function is not None:
            candidate.function_score = min(
                function_density[id(function)] * 0.25,
                2.0,
            )
    return tree, candidates


def _prune_coherent_node(node: ast.AST, selected: set[int]):
    if isinstance(node, ast.Module):
        node.body = [
            kept
            for child in node.body
            if (kept := _prune_coherent_node(child, selected)) is not None
        ]
        return node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = [
            kept
            for child in node.body
            if (kept := _prune_coherent_node(child, selected)) is not None
        ]
        if id(node) not in selected and not body:
            return None
        node.body = body or [ast.Pass()]
        return node
    if isinstance(
        node,
        (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.With,
            ast.AsyncWith,
        ),
    ):
        body = [
            kept
            for child in node.body
            if (kept := _prune_coherent_node(child, selected)) is not None
        ]
        orelse = [
            kept
            for child in getattr(node, "orelse", [])
            if (kept := _prune_coherent_node(child, selected)) is not None
        ]
        if id(node) not in selected and not body and not orelse:
            return None
        node.body = body or [ast.Pass()]
        if hasattr(node, "orelse"):
            node.orelse = orelse
        return node
    if isinstance(node, ast.Try):
        body = [
            kept
            for child in node.body
            if (kept := _prune_coherent_node(child, selected)) is not None
        ]
        handlers = [
            kept
            for child in node.handlers
            if (kept := _prune_coherent_node(child, selected)) is not None
        ]
        orelse = [
            kept
            for child in node.orelse
            if (kept := _prune_coherent_node(child, selected)) is not None
        ]
        finalbody = [
            kept
            for child in node.finalbody
            if (kept := _prune_coherent_node(child, selected)) is not None
        ]
        if id(node) not in selected and not any(
            (body, handlers, orelse, finalbody)
        ):
            return None
        node.body = body or [ast.Pass()]
        node.handlers = handlers
        node.orelse = orelse
        node.finalbody = finalbody
        if not node.handlers and not node.finalbody:
            node.finalbody = [ast.Pass()]
        return node
    if isinstance(node, ast.ExceptHandler):
        body = [
            kept
            for child in node.body
            if (kept := _prune_coherent_node(child, selected)) is not None
        ]
        if id(node) not in selected and not body:
            return None
        node.body = body or [ast.Pass()]
        return node
    if isinstance(node, ast.Match):
        cases = [
            kept
            for child in node.cases
            if (kept := _prune_coherent_node(child, selected)) is not None
        ]
        if id(node) not in selected and not cases:
            return None
        node.cases = cases or [
            ast.match_case(pattern=ast.MatchAs(), body=[ast.Pass()])
        ]
        return node
    if isinstance(node, ast.match_case):
        body = [
            kept
            for child in node.body
            if (kept := _prune_coherent_node(child, selected)) is not None
        ]
        if not body:
            return None
        node.body = body
        return node
    return node if id(node) in selected else None


def _render_coherent(tree: ast.Module, selected: list[CoherentCandidate]) -> str:
    selected_ids = {id(candidate.node) for candidate in selected}
    pruned = _prune_coherent_node(copy.deepcopy(tree), selected_ids)
    # deepcopy changes node identities, so map selections by source position.
    selected_positions = {
        candidate.source_position for candidate in selected
    }

    def mark(node):
        if (
            getattr(node, "lineno", None),
            getattr(node, "col_offset", None),
        ) in selected_positions:
            selected_ids.add(id(node))
        for child in ast.iter_child_nodes(node):
            mark(child)

    fresh = copy.deepcopy(tree)
    selected_ids.clear()
    mark(fresh)
    pruned = _prune_coherent_node(fresh, selected_ids)
    return ast.unparse(ast.fix_missing_locations(pruned)).strip()


def _select_coherent(
    tree,
    candidates,
    tokenizer,
    budget,
    *,
    include_data_flow,
    include_function_importance,
):
    for candidate in candidates:
        candidate.token_cost = max(_token_cost(tokenizer, candidate.text), 1)

    def utility(candidate):
        return candidate.utility(
            include_data_flow=include_data_flow,
            include_function_importance=include_function_importance,
        )

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -(utility(candidate) / candidate.token_cost),
            candidate.source_position,
        ),
    )
    selected = []

    def try_add(candidate):
        if candidate in selected:
            return False
        trial = selected + [candidate]
        rendered = _render_coherent(tree, trial)
        if _token_cost(tokenizer, rendered) <= budget:
            selected.append(candidate)
            return True
        return False

    # First honor soft component reservations, then return all unused capacity
    # to a global utility-per-token pass.
    for category, share in COMPONENT_SHARES.items():
        allowance = int(budget * share)
        used = 0
        category_candidates = [
            candidate
            for candidate in ranked
            if candidate.category == category
        ]
        for candidate in category_candidates:
            if used + candidate.token_cost > allowance:
                continue
            if try_add(candidate):
                used += candidate.token_cost

    for candidate in ranked:
        try_add(candidate)

    rendered = _render_coherent(tree, selected)
    selected_ids = {id(candidate) for candidate in selected}
    omitted = [
        candidate for candidate in candidates if id(candidate) not in selected_ids
    ]
    selected_utility = sum(utility(candidate) for candidate in selected)
    omitted_utility = sum(utility(candidate) for candidate in omitted)
    return rendered, selected, omitted_utility / max(selected_utility, 1e-9)


def build_coherent_v7_sketch(
    code: str,
    tokenizer,
    *,
    base_budget: int = 512,
    expanded_budget: int = 1024,
    expansion_threshold: float = 0.30,
    include_data_flow: bool = True,
    include_function_importance: bool = True,
    dynamic_expansion: bool = True,
) -> tuple[str, dict]:
    """Always run coherent candidate selection and reconstruct valid Python."""
    tree, candidates = _coherent_candidates(code)
    sketch, selected, omitted_ratio = _select_coherent(
        tree,
        candidates,
        tokenizer,
        base_budget,
        include_data_flow=include_data_flow,
        include_function_importance=include_function_importance,
    )
    budget = base_budget
    expanded = False
    if dynamic_expansion and omitted_ratio > expansion_threshold:
        sketch, selected, omitted_ratio = _select_coherent(
            tree,
            candidates,
            tokenizer,
            expanded_budget,
            include_data_flow=include_data_flow,
            include_function_importance=include_function_importance,
        )
        budget = expanded_budget
        expanded = True
    return sketch, {
        "budget": budget,
        "expanded": expanded,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "omitted_utility_ratio": round(omitted_ratio, 4),
        "token_count": _token_cost(tokenizer, sketch),
        "mode": "coherent_candidates",
    }
