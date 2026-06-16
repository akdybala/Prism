from .cognitive_complexity import (
    compute_cognitive_complexity,
)
from .call_graph import analyze_local_call_graph
from .node_types import (
    BOOLEAN_OPERATOR,
    BRANCH_TYPES,
    CASE_CLAUSE,
    CLASS_DEFINITION,
    COMPREHENSION_TYPES,
    CONDITIONAL_EXPRESSION,
    EXCEPT_CLAUSE,
    FOR_STATEMENT,
    FUNCTION_DEFINITION,
    IF_STATEMENT,
    LOOP_TYPES,
    NESTING_TYPES,
    NOT_OPERATOR,
    RETURN_STATEMENT,
    WHILE_STATEMENT,
    WITH_STATEMENT,
)
from .parser import is_async_node, walk
from .operational_features import extract_operational_features


def _decision_cost(node) -> int:
    if node.type in {
        IF_STATEMENT,
        "elif_clause",
        FOR_STATEMENT,
        WHILE_STATEMENT,
        EXCEPT_CLAUSE,
        CONDITIONAL_EXPRESSION,
        CASE_CLAUSE,
    }:
        return 1
    if node.type == BOOLEAN_OPERATOR:
        return sum(child.type in {"and", "or"} for child in node.children)
    if node.type == "if_clause" and node.parent and (
        node.parent.type in COMPREHENSION_TYPES
        or node.parent.type == "for_in_clause"
    ):
        return 1
    return 0


def _cyclomatic_for_scope(scope) -> int:
    complexity = 1

    def visit(node):
        nonlocal complexity
        if node.type == "ERROR":
            return
        if node is not scope and node.type == FUNCTION_DEFINITION:
            return
        complexity += _decision_cost(node)
        for child in node.children:
            visit(child)

    visit(scope)
    return complexity


def _max_cyclomatic(root_node) -> int:
    functions = [node for node in walk(root_node) if node.type == FUNCTION_DEFINITION]
    if not functions:
        return _cyclomatic_for_scope(root_node)
    return max(_cyclomatic_for_scope(function) for function in functions)


def _max_nesting(root_node) -> int:
    maximum = 0

    def visit(node, depth):
        nonlocal maximum
        if node.type == "ERROR":
            return
        new_depth = depth
        if node.type == "block" and node.parent and (
            node.parent.type in NESTING_TYPES
            or node.parent.type in {FUNCTION_DEFINITION, CLASS_DEFINITION}
        ):
            new_depth += 1
            maximum = max(maximum, new_depth)
        for child in node.children:
            visit(child, new_depth)

    visit(root_node, 0)
    return maximum


def _docstring_lines(root_node) -> set[int]:
    lines = set()
    scopes = [root_node] + [
        node
        for node in walk(root_node)
        if node.type in {FUNCTION_DEFINITION, CLASS_DEFINITION}
    ]
    for scope in scopes:
        body = scope if scope.type == "module" else scope.child_by_field_name("body")
        if body is None:
            continue
        statements = body.named_children
        if not statements:
            continue
        first = statements[0]
        if first.type != "expression_statement" or not first.named_children:
            continue
        if first.named_children[0].type != "string":
            continue
        lines.update(range(first.start_point.row, first.end_point.row + 1))
    return lines


def _sloc(root_node, code: str) -> int:
    docstrings = _docstring_lines(root_node)
    return sum(
        1
        for index, line in enumerate(code.splitlines())
        if index not in docstrings
        and line.strip()
        and not line.strip().startswith("#")
    )


def _has_nested_loops(root_node) -> bool:
    for node in walk(root_node):
        if node.type not in LOOP_TYPES:
            continue
        for descendant in walk(node):
            if descendant is not node and (
                descendant.type in LOOP_TYPES
                or descendant.type in COMPREHENSION_TYPES
            ):
                return True
    return False


def _bool_complexity(root_node) -> int:
    def count(node):
        value = 0
        if node.type == BOOLEAN_OPERATOR:
            value += sum(child.type in {"and", "or"} for child in node.children)
        elif node.type == NOT_OPERATOR:
            value += 1
        return value + sum(count(child) for child in node.named_children)

    roots = [
        node
        for node in walk(root_node)
        if node.type in {BOOLEAN_OPERATOR, NOT_OPERATOR}
        and (
            node.parent is None
            or node.parent.type not in {BOOLEAN_OPERATOR, NOT_OPERATOR}
        )
    ]
    return max((count(node) for node in roots), default=0)


def _max_function_length(root_node, code: str) -> int:
    lines = code.splitlines()
    lengths = []
    for function in walk(root_node):
        if function.type != FUNCTION_DEFINITION:
            continue
        extent = function
        if function.parent and function.parent.type == "decorated_definition":
            extent = function.parent
        start, end = extent.start_point.row, extent.end_point.row
        length = sum(
            1
            for line in lines[start : end + 1]
            if line.strip() and not line.strip().startswith("#")
        )
        lengths.append(length)
    return max(lengths, default=0)


def extract_structural(root_node, code: str) -> dict:
    nodes = list(walk(root_node))
    call_graph = analyze_local_call_graph(root_node)
    operational = extract_operational_features(root_node)
    cognitive = compute_cognitive_complexity(root_node)
    branches = sum(node.type in BRANCH_TYPES for node in nodes)
    return {
        "cognitive_complexity": cognitive,
        "cyclomatic_complexity": _max_cyclomatic(root_node),
        "max_nesting_depth": _max_nesting(root_node),
        "sloc": _sloc(root_node, code),
        "num_branches": branches,
        "num_loops": sum(node.type in LOOP_TYPES for node in nodes),
        "has_nested_loops": _has_nested_loops(root_node),
        "has_recursion": call_graph["recursive_function_count"] > 0,
        "max_bool_expr_complexity": _bool_complexity(root_node),
        "num_function_defs": sum(
            node.type == FUNCTION_DEFINITION for node in nodes
        ),
        "num_classes": sum(
            node.type == CLASS_DEFINITION for node in nodes
        ),
        "max_function_length": _max_function_length(root_node, code),
        "num_returns": sum(node.type == RETURN_STATEMENT for node in nodes),
        "exception_handler_count": sum(
            node.type == EXCEPT_CLAUSE for node in nodes
        ),
        "has_async": any(
            node.type == "await"
            or (
                node.type in {FUNCTION_DEFINITION, FOR_STATEMENT, WITH_STATEMENT}
                and is_async_node(node)
            )
            for node in nodes
        ),
        "num_comprehensions": sum(
            node.type in COMPREHENSION_TYPES for node in nodes
        ),
        "complexity_per_branch": round(cognitive / max(branches, 1), 2),
        **operational,
        **{
            key: value
            for key, value in call_graph.items()
            if not key.startswith("_")
            and key != "unresolved_local_call_count"
        },
    }
