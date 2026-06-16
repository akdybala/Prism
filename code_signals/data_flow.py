"""Conservative lexical data-flow features for Python snippets.

The analyzer resolves simple names within lexical scopes. Attribute storage,
subscript keys, aliases beyond direct ``a = b`` assignments, imported behavior,
dynamic dispatch, and runtime-dependent control flow remain unresolved.
"""

import re
from collections import Counter

from .call_graph import analyze_local_call_graph
from .node_types import (
    ASSIGNMENT,
    ATTRIBUTE,
    AUGMENTED_ASSIGNMENT,
    CLASS_DEFINITION,
    FUNCTION_DEFINITION,
    IDENTIFIER,
    IMPORT_FROM_STATEMENT,
    IMPORT_STATEMENT,
)
from .parser import get_node_text, walk


SCOPE_TYPES = {FUNCTION_DEFINITION, CLASS_DEFINITION, "lambda"}
TARGET_CONTAINERS = {
    "list_pattern",
    "pattern_list",
    "tuple_pattern",
}


def _simple_target_names(node) -> list[str]:
    if node is None:
        return []
    if node.type == IDENTIFIER:
        return [get_node_text(node)]
    if node.type in TARGET_CONTAINERS:
        return [
            get_node_text(item)
            for item in walk(node)
            if item.type == IDENTIFIER
        ]
    if node.type in {"list_splat_pattern", "dictionary_splat_pattern"}:
        return [
            get_node_text(item)
            for item in walk(node)
            if item.type == IDENTIFIER
        ]
    return []


def _parameter_names(function) -> list[str]:
    parameters = function.child_by_field_name("parameters")
    if parameters is None:
        return []
    names = []
    for child in parameters.named_children:
        if child.type == IDENTIFIER:
            names.append(get_node_text(child))
            continue
        name = child.child_by_field_name("name")
        if name is not None and name.type == IDENTIFIER:
            names.append(get_node_text(name))
            continue
        if child.type in {"list_splat_pattern", "dictionary_splat_pattern"}:
            names.extend(_simple_target_names(child))
    return names


def _imported_names(node) -> list[str]:
    text = get_node_text(node)
    if node.type == IMPORT_STATEMENT:
        body = re.sub(r"^\s*import\s+", "", text)
        names = []
        for item in body.split(","):
            parts = item.strip().split()
            if not parts:
                continue
            names.append(parts[-1] if len(parts) >= 3 and parts[-2] == "as" else parts[0].split(".")[0])
        return names
    match = re.match(r"\s*from\s+[.\w]+\s+import\s+(.+)", text)
    if not match:
        return []
    names = []
    for item in match.group(1).strip("()").split(","):
        parts = item.strip().split()
        if not parts or parts[0] == "*":
            continue
        names.append(parts[-1] if len(parts) >= 3 and parts[-2] == "as" else parts[0])
    return names


def _scope_local_names(scope) -> set[str]:
    names = set(_parameter_names(scope)) if scope.type == FUNCTION_DEFINITION else set()

    def visit(node):
        if node is not scope and node.type in SCOPE_TYPES:
            name = node.child_by_field_name("name")
            if name is not None:
                names.add(get_node_text(name))
            return
        if node.type in {ASSIGNMENT, AUGMENTED_ASSIGNMENT}:
            names.update(_simple_target_names(node.child_by_field_name("left")))
        elif node.type in {"for_statement", "named_expression"}:
            names.update(_simple_target_names(node.child_by_field_name("left")))
        elif node.type == "with_item":
            alias = node.child_by_field_name("alias")
            if alias is not None:
                names.update(_simple_target_names(alias))
        elif node.type == "except_clause":
            value = node.child_by_field_name("value")
            if value is not None and value.type == "as_pattern":
                names.update(
                    get_node_text(item)
                    for item in walk(value)
                    if item.type == IDENTIFIER
                )
        elif node.type in {IMPORT_STATEMENT, IMPORT_FROM_STATEMENT}:
            names.update(_imported_names(node))
        for child in node.named_children:
            visit(child)

    visit(scope)
    return names


def _assignment_names(node) -> set[str]:
    names = set()

    def visit(current):
        if current is not node and current.type in SCOPE_TYPES:
            return
        if current.type in {ASSIGNMENT, AUGMENTED_ASSIGNMENT}:
            names.update(
                _simple_target_names(current.child_by_field_name("left"))
            )
        elif current.type == "for_statement":
            names.update(
                _simple_target_names(current.child_by_field_name("left"))
            )
        for child in current.named_children:
            visit(child)

    visit(node)
    return names


def _branch_merge_count(root_node) -> int:
    merges = 0
    for node in walk(root_node):
        arms = []
        if node.type == "if_statement":
            consequence = node.child_by_field_name("consequence")
            if consequence is not None:
                arms.append(_assignment_names(consequence))
            for child in node.named_children:
                if child.type not in {"elif_clause", "else_clause"}:
                    continue
                consequence = child.child_by_field_name("consequence")
                body = consequence or next(
                    (
                        item
                        for item in child.named_children
                        if item.type == "block"
                    ),
                    None,
                )
                if body is not None:
                    arms.append(_assignment_names(body))
        elif node.type == "match_statement":
            arms = [
                _assignment_names(case)
                for case in walk(node)
                if case.type == "case_clause"
            ]
        if len(arms) >= 2:
            counts = Counter(name for arm in arms for name in arm)
            merges += sum(count >= 2 for count in counts.values())
    return merges


class _ScopeAnalyzer:
    def __init__(self, scope, ancestor_names: set[str]):
        self.scope = scope
        self.ancestor_names = ancestor_names
        self.local_names = _scope_local_names(scope)
        self.definitions = {}
        self.next_definition = 0
        self.depths = {}
        self.local_edges = 0
        self.cross_scope = 0
        self.unresolved_reads = 0
        self.alias_assignments = 0
        self.max_depth = 0
        self.nested_scopes = []

    def define(self, name: str, dependency_depths=()):
        definition = self.next_definition
        self.next_definition += 1
        depth = 1 + max(dependency_depths, default=0)
        self.definitions[name] = definition
        self.depths[definition] = depth
        self.max_depth = max(self.max_depth, depth)

    def read(self, name: str) -> int | None:
        definition = self.definitions.get(name)
        if definition is not None:
            self.local_edges += 1
            return self.depths[definition]
        if name not in self.local_names and name in self.ancestor_names:
            self.cross_scope += 1
            return None
        self.unresolved_reads += 1
        return None

    def consume_expression(self, node) -> list[int]:
        if node is None:
            return []
        if node.type in SCOPE_TYPES:
            return []
        if node.type == IDENTIFIER:
            depth = self.read(get_node_text(node))
            return [depth] if depth is not None else []
        if node.type == ATTRIBUTE:
            return self.consume_expression(node.child_by_field_name("object"))
        if node.type == "call":
            depths = []
            function = node.child_by_field_name("function")
            if function is not None and function.type == ATTRIBUTE:
                depths.extend(
                    self.consume_expression(
                        function.child_by_field_name("object")
                    )
                )
            arguments = node.child_by_field_name("arguments")
            if arguments is not None:
                for child in arguments.named_children:
                    if child.type == "keyword_argument":
                        value = child.child_by_field_name("value")
                        depths.extend(self.consume_expression(value))
                    else:
                        depths.extend(self.consume_expression(child))
            return depths
        if node.type == "keyword_argument":
            return self.consume_expression(node.child_by_field_name("value"))
        depths = []
        for child in node.named_children:
            depths.extend(self.consume_expression(child))
        return depths

    def visit(self, node):
        if node is not self.scope and node.type in SCOPE_TYPES:
            name = node.child_by_field_name("name")
            if name is not None:
                self.define(get_node_text(name))
            self.nested_scopes.append(node)
            return
        if node.type in {IMPORT_STATEMENT, IMPORT_FROM_STATEMENT}:
            for name in _imported_names(node):
                self.define(name)
            return
        if node.type == ASSIGNMENT:
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            dependencies = self.consume_expression(right)
            targets = _simple_target_names(left)
            if (
                len(targets) == 1
                and right is not None
                and right.type == IDENTIFIER
            ):
                self.alias_assignments += 1
            for name in targets:
                self.define(name, dependencies)
            if not targets:
                self.consume_expression(left)
            return
        if node.type == AUGMENTED_ASSIGNMENT:
            left = node.child_by_field_name("left")
            dependencies = self.consume_expression(left)
            dependencies.extend(
                self.consume_expression(node.child_by_field_name("right"))
            )
            for name in _simple_target_names(left):
                self.define(name, dependencies)
            return
        if node.type == "for_statement":
            dependencies = self.consume_expression(
                node.child_by_field_name("right")
            )
            for name in _simple_target_names(
                node.child_by_field_name("left")
            ):
                self.define(name, dependencies)
            body = node.child_by_field_name("body")
            if body is not None:
                self.visit(body)
            alternative = node.child_by_field_name("alternative")
            if alternative is not None:
                self.visit(alternative)
            return
        if node.type == "with_item":
            dependencies = self.consume_expression(
                node.child_by_field_name("value")
            )
            alias = node.child_by_field_name("alias")
            for name in _simple_target_names(alias):
                self.define(name, dependencies)
            return
        if node.type == "call":
            self.consume_expression(node)
            return
        if node.type == IDENTIFIER:
            self.consume_expression(node)
            return
        for child in node.named_children:
            self.visit(child)

    def analyze(self):
        for name in _parameter_names(self.scope):
            self.define(name)
        body = (
            self.scope.child_by_field_name("body")
            if self.scope.type in SCOPE_TYPES
            else self.scope
        )
        if body is not None:
            for child in body.named_children:
                self.visit(child)
        return self


def _mutation_counts(root_node) -> tuple[int, int]:
    attributes = 0
    subscripts = 0
    for node in walk(root_node):
        if node.type not in {
            ASSIGNMENT,
            AUGMENTED_ASSIGNMENT,
            "delete_statement",
        }:
            continue
        target = node.child_by_field_name("left")
        if target is None and node.type == "delete_statement":
            targets = node.named_children
        else:
            targets = [target] if target is not None else []
        for item in targets:
            attributes += sum(
                descendant.type == ATTRIBUTE for descendant in walk(item)
            )
            subscripts += sum(
                descendant.type == "subscript" for descendant in walk(item)
            )
    return attributes, subscripts


def analyze_data_flow(root_node, unresolved_call_count: int = 0) -> dict:
    """Extract lexical dependency and uncertainty metrics.

    Def-use edges only connect reads to earlier simple-name definitions in the
    same scope. Cross-scope reads are counted but not connected into dependency
    chains because runtime closure/global state is not modeled.
    """
    pending = [(root_node, set())]
    local_edges = 0
    cross_scope = 0
    unresolved_reads = 0
    aliases = 0
    max_depth = 0

    while pending:
        scope, ancestor_names = pending.pop()
        analyzer = _ScopeAnalyzer(scope, ancestor_names).analyze()
        local_edges += analyzer.local_edges
        cross_scope += analyzer.cross_scope
        unresolved_reads += analyzer.unresolved_reads
        aliases += analyzer.alias_assignments
        max_depth = max(max_depth, analyzer.max_depth)
        nested_ancestors = ancestor_names | analyzer.local_names
        pending.extend(
            (nested, nested_ancestors) for nested in analyzer.nested_scopes
        )

    attribute_mutations, subscript_mutations = _mutation_counts(root_node)
    total_observations = (
        local_edges
        + cross_scope
        + unresolved_reads
        + unresolved_call_count
    )
    unresolved = unresolved_reads + unresolved_call_count
    return {
        "local_def_use_edge_count": local_edges,
        "max_dataflow_chain_depth": max_depth,
        "cross_scope_flow_count": cross_scope,
        "branch_merge_count": _branch_merge_count(root_node),
        "alias_assignment_count": aliases,
        "attribute_mutation_count": attribute_mutations,
        "subscript_mutation_count": subscript_mutations,
        "unresolved_call_count": unresolved_call_count,
        "unresolved_flow_ratio": round(
            unresolved / max(total_observations, 1),
            3,
        ),
    }


def extract_data_flow(root_node) -> dict:
    """Return the standalone data-flow signal group for a parsed snippet."""
    call_graph = analyze_local_call_graph(root_node)
    return analyze_data_flow(
        root_node,
        unresolved_call_count=call_graph["unresolved_local_call_count"],
    )
