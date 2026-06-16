import re
from collections import Counter

from .node_types import (
    ASSIGNMENT,
    ATTRIBUTE,
    AUGMENTED_ASSIGNMENT,
    CALL,
    CLASS_DEFINITION,
    DECORATED_DEFINITION,
    FUNCTION_DEFINITION,
    GLOBAL_STATEMENT,
    IDENTIFIER,
    IMPORT_FROM_STATEMENT,
    IMPORT_STATEMENT,
    NONLOCAL_STATEMENT,
)
from .parser import get_node_text, walk


SCOPE_TYPES = {FUNCTION_DEFINITION, CLASS_DEFINITION}


def _simple_target(node):
    left = node.child_by_field_name("left")
    if left is None and node.named_children:
        left = node.named_children[0]
    return get_node_text(left) if left is not None and left.type == IDENTIFIER else None


def _reassignment_count(root_node) -> int:
    counts = {}

    def visit(node, scope):
        if node.type == "ERROR":
            return
        if node is not root_node and node.type in SCOPE_TYPES:
            scope = (node.start_byte, node.end_byte)
        if node.type in {ASSIGNMENT, AUGMENTED_ASSIGNMENT}:
            name = _simple_target(node)
            if name:
                counts.setdefault(scope, Counter())[name] += 1
        for child in node.children:
            visit(child, scope)

    visit(root_node, ("module",))
    return sum(
        count - 1
        for scope_counts in counts.values()
        for count in scope_counts.values()
        if count > 1
    )


def _call_parts(node):
    callee = node.child_by_field_name("function")
    arguments = node.child_by_field_name("arguments")
    if callee is None and node.named_children:
        callee = node.named_children[0]
    return callee, arguments


def _has_dynamic_features(root_node, code: str) -> bool:
    dynamic_calls = {"eval", "exec", "getattr", "setattr", "delattr", "__import__"}
    for node in walk(root_node):
        if node.type == CALL:
            callee, arguments = _call_parts(node)
            if callee is not None and callee.type == IDENTIFIER:
                name = get_node_text(callee)
                if name in dynamic_calls:
                    return True
                if name == "type" and arguments is not None:
                    if len(arguments.named_children) == 3:
                        return True
        elif node.type == CLASS_DEFINITION:
            bases = node.child_by_field_name("superclasses")
            if bases and any(
                child.type == "keyword_argument"
                and get_node_text(child).lstrip().startswith("metaclass")
                for child in bases.named_children
            ):
                return True
        elif node.type == FUNCTION_DEFINITION:
            name = node.child_by_field_name("name")
            if name is not None and get_node_text(name) == "__init_subclass__":
                return True
        elif node.type in {IMPORT_STATEMENT, IMPORT_FROM_STATEMENT}:
            if re.search(r"\bimportlib\b", get_node_text(node)):
                return True
    return bool(
        re.search(
            r"\b(eval|exec|getattr|setattr|delattr|__import__|metaclass)\s*[\(=]",
            code,
        )
    )


def _target_identifiers(root_node):
    names = []
    for node in walk(root_node):
        target = None
        if node.type in {ASSIGNMENT, AUGMENTED_ASSIGNMENT}:
            target = node.child_by_field_name("left")
        elif node.type == "for_statement":
            target = node.child_by_field_name("left")
        elif node.type == "except_clause":
            value = node.child_by_field_name("value")
            if value is not None and value.type == "as_pattern":
                target = value.child_by_field_name("alias")
        elif node.type == "with_item":
            value = node.child_by_field_name("value")
            if value is not None and value.type == "as_pattern":
                target = value.child_by_field_name("alias")
        if target is None:
            continue
        candidates = [target] if target.type == IDENTIFIER else [
            child for child in walk(target) if child.type == IDENTIFIER
        ]
        names.extend(get_node_text(item) for item in candidates)
    return [name for name in names if name != "_"]


def _call_diversity(root_node) -> int:
    calls = set()
    for node in walk(root_node):
        if node.type != CALL:
            continue
        callee, _ = _call_parts(node)
        if callee is None:
            continue
        if callee.type == IDENTIFIER:
            calls.add(get_node_text(callee))
        elif callee.type == ATTRIBUTE:
            attr = callee.child_by_field_name("attribute")
            if attr is not None:
                calls.add(get_node_text(attr))
    return len(calls)


def extract_semantic(root_node, code: str) -> dict:
    nodes = list(walk(root_node))
    target_names = _target_identifiers(root_node)
    identifiers = {
        get_node_text(node) for node in nodes if node.type == IDENTIFIER
    }
    return {
        "variable_reassignment_count": _reassignment_count(root_node),
        "has_global_nonlocal": any(
            node.type in {GLOBAL_STATEMENT, NONLOCAL_STATEMENT} for node in nodes
        ),
        "has_dynamic_features": _has_dynamic_features(root_node, code),
        "has_decorators": any(
            node.type == DECORATED_DEFINITION for node in nodes
        ),
        "single_char_var_ratio": round(
            sum(len(name) == 1 for name in target_names)
            / max(len(target_names), 1),
            3,
        ),
        "unique_identifier_count": len(identifiers),
        "call_diversity": _call_diversity(root_node),
    }
