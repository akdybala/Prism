"""Tree-sitter-native coherent V7 domain sketching.

Candidates are scored at statement ownership boundaries, selected under a
token budget, and reconstructed from original source lines. Required compound
ancestors are retained and empty retained suites receive ``pass``.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from .budgeted_domain_sketch import (
    COMPONENT_SHARES,
    DOMAIN_LITERAL_PATTERN,
    GENERIC_IDENTIFIERS,
    IDENTIFIER_PATTERN,
    MUTATION_METHODS,
    _token_cost,
)
from .domain import API_FINGERPRINTS, IMPORT_FINGERPRINTS
from .parser import count_errors, get_node_text, parse, walk


COMPOUND_TYPES = {
    "class_definition",
    "function_definition",
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
CONTROL_TYPES = COMPOUND_TYPES - {
    "class_definition",
    "function_definition",
}
ATOMIC_STATEMENT_TYPES = {
    "assert_statement",
    "assignment",
    "augmented_assignment",
    "break_statement",
    "continue_statement",
    "delete_statement",
    "expression_statement",
    "import_from_statement",
    "import_statement",
    "raise_statement",
    "return_statement",
    "yield",
}
ASSIGNMENT_TYPES = {
    "assignment",
    "augmented_assignment",
}
DEFAULT_FUNCTION_COVERAGE_SHARE = 0.15
DEFAULT_DOMAIN_SIGNAL_SHARE = 0.20
DEFAULT_DEPENDENCY_SIGNAL_SHARE = 0.20


@dataclass
class TreeSitterCandidate:
    category: str
    node: object
    text: str
    source_position: tuple[int, int]
    domain_score: float = 0.0
    data_flow_score: float = 0.0
    structural_score: float = 0.0
    uniqueness_score: float = 0.0
    function_score: float = 0.0
    dependency_chain_score: float = 0.0
    call_graph_score: float = 0.0
    token_cost: int = 0

    @property
    def key(self) -> tuple[str, int, int]:
        return self.node.type, self.node.start_byte, self.node.end_byte

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
        value += self.dependency_chain_score + self.call_graph_score
        return max(value, 0.1)


def _node_key(node) -> tuple[str, int, int]:
    return node.type, node.start_byte, node.end_byte


def _body_node(node):
    for field in ("body", "consequence"):
        body = node.child_by_field_name(field)
        if body is not None and body.type == "block":
            return body
    return next(
        (child for child in node.named_children if child.type == "block"),
        None,
    )


def _header_text(node) -> str:
    body = _body_node(node)
    if body is None:
        return get_node_text(node).splitlines()[0].strip()
    raw = node.text[: body.start_byte - node.start_byte].decode(
        "utf-8",
        errors="replace",
    )
    return re.sub(r"\s+", " ", raw).strip()


def _contains(node, node_types: set[str]) -> bool:
    return any(descendant.type in node_types for descendant in walk(node))


def _nearest_statement_owner(node):
    current = node
    while current is not None:
        if (
            current.type in ASSIGNMENT_TYPES
            and current.parent is not None
            and current.parent.type == "expression_statement"
        ):
            current = current.parent
            continue
        if current.type in ATOMIC_STATEMENT_TYPES | COMPOUND_TYPES | {"decorator"}:
            return current
        current = current.parent
    return None


def _enclosing_function(node):
    current = node.parent
    while current is not None:
        if current.type == "function_definition":
            return current
        current = current.parent
    return None


def _candidate_function(node):
    if node.type == "function_definition":
        return node
    return _enclosing_function(node)


def _canonical_identifier(name: str) -> str:
    """Collapse generated numeric suffixes when measuring uniqueness."""
    return re.sub(r"_?\d+$", "", name)


def _simple_target_names(node) -> set[str]:
    if node is None:
        return set()
    if node.type == "identifier":
        return {get_node_text(node)}
    if node.type in {
        "list_pattern",
        "pattern_list",
        "tuple_pattern",
        "list_splat_pattern",
        "dictionary_splat_pattern",
    }:
        return {
            get_node_text(item)
            for item in walk(node)
            if item.type == "identifier"
        }
    return set()


def _assignment_node(node):
    if node.type in ASSIGNMENT_TYPES:
        return node
    if node.type == "expression_statement":
        return next(
            (
                child
                for child in node.named_children
                if child.type in ASSIGNMENT_TYPES
            ),
            None,
        )
    return None


def _read_identifiers(node, excluded_nodes=()) -> set[str]:
    excluded = {_node_key(item) for item in excluded_nodes if item is not None}
    reads = set()

    def visit(current):
        if _node_key(current) in excluded:
            return
        if current.type == "function_definition" and current is not node:
            return
        if current.type == "identifier":
            parent = current.parent
            if (
                parent is not None
                and parent.type == "attribute"
                and parent.child_by_field_name("attribute") is current
            ):
                return
            if (
                parent is not None
                and parent.type == "keyword_argument"
                and parent.child_by_field_name("name") is current
            ):
                return
            reads.add(get_node_text(current))
            return
        for child in current.named_children:
            visit(child)

    visit(node)
    return reads


def _candidate_definitions_and_reads(candidate):
    node = candidate.node
    assignment = _assignment_node(node)
    if assignment is not None:
        left = assignment.child_by_field_name("left")
        right = assignment.child_by_field_name("right")
        definitions = _simple_target_names(left)
        reads = _read_identifiers(right) if right is not None else set()
        if assignment.type == "augmented_assignment":
            reads.update(_read_identifiers(left))
        return definitions, reads
    if node.type == "for_statement":
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        return _simple_target_names(left), _read_identifiers(right)
    return set(), _read_identifiers(node)


def _calls_owned_by(node):
    calls = []

    def visit(current):
        if current is not node and current.type == "function_definition":
            return
        if current.type == "call":
            calls.append(current)
        for child in current.named_children:
            visit(child)

    visit(node)
    return calls


def _enclosing_class_name(function):
    current = function.parent
    while current is not None:
        if current.type == "class_definition":
            name = current.child_by_field_name("name")
            return get_node_text(name) if name is not None else None
        if current.type == "function_definition":
            return None
        current = current.parent
    return None


def _callee_parts(call):
    callee = call.child_by_field_name("function")
    if callee is None:
        return None, None
    if callee.type == "identifier":
        return get_node_text(callee), None
    if callee.type == "attribute":
        obj = callee.child_by_field_name("object")
        attribute = callee.child_by_field_name("attribute")
        return (
            get_node_text(attribute) if attribute is not None else None,
            get_node_text(obj) if obj is not None else None,
        )
    return None, None


def _apply_dependency_scores(candidates, root_node):
    by_key = {candidate.key: candidate for candidate in candidates}
    by_function = {}
    for candidate in candidates:
        function = _candidate_function(candidate.node)
        if function is not None:
            by_function.setdefault(_node_key(function), []).append(candidate)

    dependency_edges = set()
    dependency_depth = {}
    for function_candidates in by_function.values():
        definitions = {}
        for candidate in sorted(
            function_candidates,
            key=lambda item: item.source_position,
        ):
            defined_names, read_names = _candidate_definitions_and_reads(candidate)
            predecessor_depths = []
            for name in read_names:
                source = definitions.get(name)
                if source is None or source.key == candidate.key:
                    continue
                dependency_edges.add((source.key, candidate.key))
                predecessor_depths.append(dependency_depth.get(source.key, 1))
            dependency_depth[candidate.key] = 1 + max(
                predecessor_depths,
                default=0,
            )
            for name in defined_names:
                definitions[name] = candidate

    incident = Counter()
    for source_key, target_key in dependency_edges:
        incident[source_key] += 1
        incident[target_key] += 1
    for key, candidate in by_key.items():
        depth = dependency_depth.get(key, 1)
        candidate.dependency_chain_score = min(
            3.0,
            incident[key] * 0.45 + max(depth - 1, 0) * 0.40,
        )

    functions = [
        node for node in walk(root_node) if node.type == "function_definition"
    ]
    by_simple_name = {}
    methods_by_class = {}
    for function in functions:
        name_node = function.child_by_field_name("name")
        if name_node is None:
            continue
        name = get_node_text(name_node)
        by_simple_name.setdefault(name, []).append(function)
        class_name = _enclosing_class_name(function)
        if class_name is not None:
            methods_by_class.setdefault((class_name, name), []).append(function)

    signature_candidates = {
        _node_key(candidate.node): candidate
        for candidate in candidates
        if candidate.node.type == "function_definition"
    }
    call_edges = set()
    caller_targets = {}
    target_callers = {}
    for candidate in candidates:
        caller = _candidate_function(candidate.node)
        if caller is None:
            continue
        class_name = _enclosing_class_name(caller)
        resolved_targets = set()
        for call in _calls_owned_by(candidate.node):
            callee, obj = _callee_parts(call)
            target = None
            if callee is not None and obj is None:
                options = by_simple_name.get(callee, [])
                if len(options) == 1:
                    target = options[0]
            elif callee is not None and obj in {"self", "cls"} and class_name:
                options = methods_by_class.get((class_name, callee), [])
                if len(options) == 1:
                    target = options[0]
            if target is not None:
                resolved_targets.add(_node_key(target))
                call_edges.add((_node_key(caller), _node_key(target)))
        if resolved_targets:
            candidate.call_graph_score = min(
                3.0,
                1.25 + 0.50 * len(resolved_targets),
            )
            caller_targets.setdefault(_node_key(caller), set()).update(
                resolved_targets
            )
            for target_key in resolved_targets:
                target_callers.setdefault(target_key, set()).add(
                    _node_key(caller)
                )

    for function_key, targets in caller_targets.items():
        signature = signature_candidates.get(function_key)
        if signature is not None:
            signature.call_graph_score += min(1.5, 0.4 * len(targets))
    for function_key, callers in target_callers.items():
        signature = signature_candidates.get(function_key)
        if signature is not None:
            signature.call_graph_score += min(1.5, 0.5 * len(callers))

    return len(dependency_edges), len(call_edges)


def _category(node) -> str | None:
    if node.type in {"import_statement", "import_from_statement"}:
        return "imports"
    if node.type in {"class_definition", "function_definition", "decorator"}:
        return "signatures"
    if node.type in {"raise_statement", "except_clause"}:
        return "literals_exceptions"
    if node.type in CONTROL_TYPES:
        return "control"
    if node.type in ASSIGNMENT_TYPES:
        if node.parent is not None and node.parent.type == "expression_statement":
            return None
        right = node.child_by_field_name("right")
        left = node.child_by_field_name("left")
        has_call = right is not None and _contains(right, {"call"})
        mutation = (
            node.type == "augmented_assignment"
            or left is not None and left.type in {"attribute", "subscript"}
        )
        return "assignments_calls" if has_call or mutation else "statements"
    if node.type == "expression_statement":
        assignment = next(
            (
                child
                for child in node.named_children
                if child.type in ASSIGNMENT_TYPES
            ),
            None,
        )
        if assignment is not None:
            right = assignment.child_by_field_name("right")
            left = assignment.child_by_field_name("left")
            has_call = right is not None and _contains(right, {"call"})
            mutation = (
                assignment.type == "augmented_assignment"
                or left is not None and left.type in {"attribute", "subscript"}
            )
            return "assignments_calls" if has_call or mutation else "statements"
        return "assignments_calls" if _contains(node, {"call"}) else None
    if node.type in {
        "assert_statement",
        "break_statement",
        "continue_statement",
        "delete_statement",
        "return_statement",
        "yield",
    }:
        return "statements"
    return None


def _candidate_text(node) -> str:
    if node.type in COMPOUND_TYPES:
        return _header_text(node)
    return get_node_text(node).strip()


def extract_tree_sitter_candidates(
    code: str,
    root_node,
    *,
    include_graph_stats: bool = False,
) -> list[TreeSitterCandidate] | tuple[list[TreeSitterCandidate], int, int]:
    """Extract deduplicated candidates owned by reconstructable CST units."""
    candidates = []
    by_key = {}

    def add(node, category=None):
        category = category or _category(node)
        if category is None:
            return
        key = _node_key(node)
        if key in by_key:
            if category == "literals_exceptions":
                by_key[key].category = category
            return
        text = _candidate_text(node)
        if not text:
            return
        candidate = TreeSitterCandidate(
            category=category,
            node=node,
            text=text,
            source_position=(node.start_point.row, node.start_point.column),
        )
        candidates.append(candidate)
        by_key[key] = candidate

    for node in walk(root_node):
        add(node)
        if (
            node.type == "string"
            and DOMAIN_LITERAL_PATTERN.search(get_node_text(node))
        ):
            owner = _nearest_statement_owner(node)
            if owner is not None:
                add(owner, "literals_exceptions")

    identifier_frequency = Counter(IDENTIFIER_PATTERN.findall(code))
    candidate_frequency = Counter(
        _canonical_identifier(name)
        for candidate in candidates
        for name in set(IDENTIFIER_PATTERN.findall(candidate.text))
    )
    function_density = Counter(
        _node_key(function)
        for candidate in candidates
        if (function := _candidate_function(candidate.node)) is not None
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
        if any(f".{method}(" in candidate.text for method in MUTATION_METHODS):
            candidate.structural_score += 1.0
        candidate.uniqueness_score = sum(
            1.0 / candidate_frequency[_canonical_identifier(name)]
            for name in set(identifiers)
            if candidate_frequency[_canonical_identifier(name)]
        ) / max(len(set(identifiers)), 1)
        function = _candidate_function(candidate.node)
        if function is not None:
            candidate.function_score = min(
                function_density[_node_key(function)] * 0.25,
                2.0,
            )
    dependency_edge_count, call_edge_count = _apply_dependency_scores(
        candidates,
        root_node,
    )
    if include_graph_stats:
        return candidates, dependency_edge_count, call_edge_count
    return candidates


def _required_compounds(selected_nodes):
    required = {}
    for node in selected_nodes:
        current = node if node.type in COMPOUND_TYPES else node.parent
        while current is not None:
            if current.type in COMPOUND_TYPES:
                required[_node_key(current)] = current
            current = current.parent

        if node.type == "decorator":
            parent = node.parent
            definition = (
                parent.child_by_field_name("definition")
                if parent is not None and parent.type == "decorated_definition"
                else None
            )
            if definition is not None:
                required[_node_key(definition)] = definition
    return required


def _ensure_grammar_dependencies(required):
    changed = True
    while changed:
        changed = False
        for node in list(required.values()):
            if node.type == "try_statement":
                clauses = [
                    child
                    for child in node.named_children
                    if child.type in {"except_clause", "finally_clause"}
                ]
                if clauses and not any(_node_key(clause) in required for clause in clauses):
                    required[_node_key(clauses[0])] = clauses[0]
                    changed = True
            elif node.type == "match_statement":
                body = _body_node(node)
                cases = (
                    [
                        child
                        for child in body.named_children
                        if child.type == "case_clause"
                    ]
                    if body is not None
                    else []
                )
                if cases and not any(_node_key(case) in required for case in cases):
                    required[_node_key(cases[0])] = cases[0]
                    changed = True
    return required


def _mark_rows(marked_rows, start_row: int, end_row: int):
    marked_rows.update(range(start_row, end_row + 1))


def _render_tree_sitter(
    code: str,
    selected: list[TreeSitterCandidate],
) -> str:
    lines = code.splitlines()
    selected_nodes = [candidate.node for candidate in selected]
    required = _ensure_grammar_dependencies(
        _required_compounds(selected_nodes)
    )
    marked_rows = set()

    for node in selected_nodes:
        if node.type not in COMPOUND_TYPES:
            _mark_rows(
                marked_rows,
                node.start_point.row,
                node.end_point.row,
            )

    for node in required.values():
        body = _body_node(node)
        if body is None or body.start_point.row == node.start_point.row:
            _mark_rows(
                marked_rows,
                node.start_point.row,
                node.end_point.row if body is None else node.start_point.row,
            )
            continue
        _mark_rows(
            marked_rows,
            node.start_point.row,
            body.start_point.row - 1,
        )

    insertions = {}
    for node in required.values():
        body = _body_node(node)
        if body is None or body.start_point.row == node.start_point.row:
            continue
        has_content = any(
            body.start_point.row <= row <= body.end_point.row
            and row in marked_rows
            for row in marked_rows
        )
        if not has_content:
            insertions.setdefault(
                body.start_point.row,
                " " * body.start_point.column + "pass",
            )

    rendered = []
    for row, line in enumerate(lines):
        if row in insertions:
            rendered.append(insertions[row])
        if row in marked_rows:
            rendered.append(line.rstrip())
    for row in sorted(row for row in insertions if row >= len(lines)):
        rendered.append(insertions[row])
    return "\n".join(rendered).strip()


def _select_tree_sitter(
    code,
    candidates,
    tokenizer,
    budget,
    *,
    include_data_flow,
    include_function_importance,
    function_coverage_share,
    domain_signal_share,
    dependency_signal_share,
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
    selected_keys = set()
    covered_function_keys = set()

    def try_add(candidate):
        if candidate.key in selected_keys:
            return False
        trial = selected + [candidate]
        rendered = _render_tree_sitter(code, trial)
        if _token_cost(tokenizer, rendered) > budget:
            return False
        selected.append(candidate)
        selected_keys.add(candidate.key)
        function = _candidate_function(candidate.node)
        if function is not None:
            covered_function_keys.add(_node_key(function))
        return True

    # Explicit domain evidence gets a capped first pass. Longer SQL, crypto,
    # ML, HTTP, or concurrency statements should not lose solely because
    # generic assignments are shorter.
    domain_allowance = max(int(budget * domain_signal_share), 0)
    domain_used = 0
    domain_candidates = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.domain_score > 0
        ),
        key=lambda candidate: (
            -candidate.domain_score,
            -(utility(candidate) / candidate.token_cost),
            candidate.source_position,
        ),
    )
    for candidate in domain_candidates:
        if domain_used + candidate.token_cost > domain_allowance:
            continue
        if try_add(candidate):
            domain_used += candidate.token_cost

    dependency_allowance = max(int(budget * dependency_signal_share), 0)
    dependency_used = 0
    dependency_candidates = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.dependency_chain_score > 0
            or candidate.call_graph_score > 0
        ),
        key=lambda candidate: (
            -(candidate.dependency_chain_score + candidate.call_graph_score),
            -(utility(candidate) / candidate.token_cost),
            candidate.source_position,
        ),
    )
    for candidate in dependency_candidates:
        if dependency_used + candidate.token_cost > dependency_allowance:
            continue
        if try_add(candidate):
            dependency_used += candidate.token_cost

    # Give functions a small, capped opportunity to retain one representative
    # unit before category reservations. This prevents quiet helper starvation
    # without turning the budget into equal per-function slices.
    function_candidates = {}
    for candidate in ranked:
        function = _candidate_function(candidate.node)
        if function is None:
            continue
        function_candidates.setdefault(_node_key(function), []).append(candidate)
    representatives = []
    for function_key, options in function_candidates.items():
        representative = min(
            options,
            key=lambda candidate: (
                -(utility(candidate) / candidate.token_cost),
                candidate.source_position,
            ),
        )
        representatives.append((function_key, representative))
    representatives.sort(
        key=lambda item: (
            -(utility(item[1]) / item[1].token_cost),
            item[1].source_position,
        )
    )
    function_allowance = max(int(budget * function_coverage_share), 0)
    function_used = 0
    for _, candidate in representatives:
        if function_used + candidate.token_cost > function_allowance:
            continue
        if try_add(candidate):
            function_used += candidate.token_cost

    for category, share in COMPONENT_SHARES.items():
        allowance = int(budget * share)
        used = 0
        for candidate in ranked:
            if candidate.category != category:
                continue
            if used + candidate.token_cost > allowance:
                continue
            if try_add(candidate):
                used += candidate.token_cost

    for candidate in ranked:
        try_add(candidate)

    rendered = _render_tree_sitter(code, selected)
    omitted = [
        candidate for candidate in candidates if candidate.key not in selected_keys
    ]
    selected_utility = sum(utility(candidate) for candidate in selected)
    omitted_utility = sum(utility(candidate) for candidate in omitted)
    return (
        rendered,
        selected,
        omitted_utility / max(selected_utility, 1e-9),
        len(function_candidates),
        len(covered_function_keys),
    )


def build_tree_sitter_coherent_v7_sketch(
    code: str,
    tokenizer,
    *,
    root_node=None,
    base_budget: int = 512,
    expanded_budget: int = 1024,
    expansion_threshold: float = 0.30,
    include_data_flow: bool = True,
    include_function_importance: bool = True,
    dynamic_expansion: bool = True,
    function_coverage_share: float = DEFAULT_FUNCTION_COVERAGE_SHARE,
    domain_signal_share: float = DEFAULT_DOMAIN_SIGNAL_SHARE,
    dependency_signal_share: float = DEFAULT_DEPENDENCY_SIGNAL_SHARE,
) -> tuple[str, dict]:
    """Build a coherent sketch using Tree-sitter for the entire pipeline."""
    if not 0.0 <= function_coverage_share <= 1.0:
        raise ValueError("function_coverage_share must be between 0 and 1")
    if not 0.0 <= domain_signal_share <= 1.0:
        raise ValueError("domain_signal_share must be between 0 and 1")
    if not 0.0 <= dependency_signal_share <= 1.0:
        raise ValueError("dependency_signal_share must be between 0 and 1")
    root_node = root_node or parse(code).root_node
    input_errors = count_errors(root_node)
    candidates, dependency_edge_count, call_edge_count = (
        extract_tree_sitter_candidates(
            code,
            root_node,
            include_graph_stats=True,
        )
    )
    (
        sketch,
        selected,
        omitted_ratio,
        function_count,
        covered_function_count,
    ) = _select_tree_sitter(
        code,
        candidates,
        tokenizer,
        base_budget,
        include_data_flow=include_data_flow,
        include_function_importance=include_function_importance,
        function_coverage_share=function_coverage_share,
        domain_signal_share=domain_signal_share,
        dependency_signal_share=dependency_signal_share,
    )
    budget = base_budget
    expanded = False
    if dynamic_expansion and omitted_ratio > expansion_threshold:
        (
            sketch,
            selected,
            omitted_ratio,
            function_count,
            covered_function_count,
        ) = _select_tree_sitter(
            code,
            candidates,
            tokenizer,
            expanded_budget,
            include_data_flow=include_data_flow,
            include_function_importance=include_function_importance,
            function_coverage_share=function_coverage_share,
            domain_signal_share=domain_signal_share,
            dependency_signal_share=dependency_signal_share,
        )
        budget = expanded_budget
        expanded = True

    output_errors = count_errors(parse(sketch).root_node) if sketch else 0
    return sketch, {
        "budget": budget,
        "expanded": expanded,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "omitted_utility_ratio": round(omitted_ratio, 4),
        "token_count": _token_cost(tokenizer, sketch),
        "mode": "tree_sitter_coherent_candidates",
        "input_parse_errors": input_errors,
        "output_parse_errors": output_errors,
        "function_count": function_count,
        "covered_function_count": covered_function_count,
        "function_coverage_rate": round(
            covered_function_count / max(function_count, 1),
            4,
        ),
        "function_coverage_share": function_coverage_share,
        "domain_signal_share": domain_signal_share,
        "dependency_signal_share": dependency_signal_share,
        "candidate_dependency_edge_count": dependency_edge_count,
        "candidate_call_edge_count": call_edge_count,
    }
