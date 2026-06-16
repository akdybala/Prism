from .call_graph import analyze_local_call_graph
from .node_types import (
    BOOLEAN_OPERATOR,
    BREAK_STATEMENT,
    CLASS_DEFINITION,
    CONDITIONAL_EXPRESSION,
    CONTINUE_STATEMENT,
    ELIF_CLAUSE,
    ELSE_CLAUSE,
    EXCEPT_CLAUSE,
    FOR_STATEMENT,
    FUNCTION_DEFINITION,
    IF_STATEMENT,
    LAMBDA,
    TRY_STATEMENT,
    WHILE_STATEMENT,
    WITH_STATEMENT,
)


PENALIZED_TYPES = {
    IF_STATEMENT,
    FOR_STATEMENT,
    WHILE_STATEMENT,
    EXCEPT_CLAUSE,
}
PLAIN_INCREMENT_TYPES = {
    ELIF_CLAUSE,
    ELSE_CLAUSE,
    WITH_STATEMENT,
    BREAK_STATEMENT,
    CONTINUE_STATEMENT,
    CONDITIONAL_EXPRESSION,
}
NESTING_PARENTS = {
    IF_STATEMENT,
    ELIF_CLAUSE,
    ELSE_CLAUSE,
    FOR_STATEMENT,
    WHILE_STATEMENT,
    TRY_STATEMENT,
    EXCEPT_CLAUSE,
    "finally_clause",
    WITH_STATEMENT,
    LAMBDA,
}


def _boolean_sequence_cost(node) -> int:
    operators = []

    def collect(current):
        for child in current.children:
            if child.type in {"and", "or"}:
                operators.append(child.type)
            elif child.type == BOOLEAN_OPERATOR:
                collect(child)

    collect(node)
    if not operators:
        return 0
    return 1 + sum(a != b for a, b in zip(operators, operators[1:]))


def count_recursive_functions(root_node) -> int:
    """Count functions in direct or statically resolved indirect cycles."""
    return analyze_local_call_graph(root_node)["recursive_function_count"]


def compute_cognitive_complexity(root_node) -> int:
    # Sonar-style cognitive complexity adds one increment for each function
    # that participates in direct recursion, regardless of call count.
    total = count_recursive_functions(root_node)

    def visit(node, nesting):
        nonlocal total
        if node.type == "ERROR":
            return

        if node.type in PENALIZED_TYPES:
            total += 1 + nesting
        elif node.type in PLAIN_INCREMENT_TYPES:
            total += 1
        elif node.type == BOOLEAN_OPERATOR and (
            node.parent is None or node.parent.type != BOOLEAN_OPERATOR
        ):
            total += _boolean_sequence_cost(node)

        child_nesting = 0 if node.type in {
            FUNCTION_DEFINITION,
            CLASS_DEFINITION,
        } else nesting
        for child in node.children:
            next_nesting = child_nesting
            if child.type == "block" and node.type in NESTING_PARENTS:
                next_nesting += 1
            visit(child, next_nesting)

    visit(root_node, 0)
    return total
