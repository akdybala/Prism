"""Conservative AST and lexical operational features."""

import re

from .node_types import (
    ASSIGNMENT,
    ATTRIBUTE,
    AUGMENTED_ASSIGNMENT,
    CALL,
    IMPORT_FROM_STATEMENT,
    IMPORT_STATEMENT,
)
from .parser import get_node_text, is_async_node, walk


BITWISE_OPERATORS = {"&", "|", "^", "~", "<<", ">>", "&=", "|=", "^=", "<<=", ">>="}
MUTATING_METHODS = {
    "add",
    "append",
    "clear",
    "discard",
    "extend",
    "insert",
    "pop",
    "remove",
    "reverse",
    "setdefault",
    "sort",
    "update",
}
CONCURRENCY_MODULES = {
    "asyncio",
    "concurrent",
    "multiprocessing",
    "threading",
    "trio",
}
SYNCHRONIZATION_NAMES = {
    "Barrier",
    "Condition",
    "Event",
    "Lock",
    "Queue",
    "RLock",
    "Semaphore",
    "acquire",
    "release",
}
IO_MODULES = {"io", "pathlib"}
IO_NAMES = {"open"}
NETWORK_MODULES = {
    "aiohttp",
    "http",
    "requests",
    "socket",
    "urllib",
    "websockets",
}
SUBPROCESS_MODULES = {"subprocess"}
SUBPROCESS_NAMES = {
    "Popen",
    "call",
    "check_call",
    "check_output",
    "run",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "system",
}


def _import_roots(root_node):
    modules = set()
    for node in walk(root_node):
        if node.type not in {IMPORT_STATEMENT, IMPORT_FROM_STATEMENT}:
            continue
        text = get_node_text(node)
        match = re.match(r"\s*(?:from|import)\s+([.\w]+)", text)
        if match:
            root = match.group(1).lstrip(".").split(".")[0]
            if root:
                modules.add(root)
    return modules


def _call_name(call):
    callee = call.child_by_field_name("function")
    if callee is None and call.named_children:
        callee = call.named_children[0]
    if callee is None:
        return None
    if callee.type == "identifier":
        return get_node_text(callee)
    if callee.type == ATTRIBUTE:
        attr = callee.child_by_field_name("attribute")
        return get_node_text(attr) if attr is not None else None
    return None


def _has_operator(nodes, operators):
    return any(
        child.type in operators
        for node in nodes
        if node.type in {"binary_operator", "unary_operator", AUGMENTED_ASSIGNMENT}
        for child in node.children
    )


def _assignment_target(node):
    target = node.child_by_field_name("left")
    if target is None and node.named_children:
        target = node.named_children[0]
    return target


def _has_mutation(nodes):
    for node in nodes:
        if node.type == AUGMENTED_ASSIGNMENT:
            return True
        if node.type == ASSIGNMENT:
            target = _assignment_target(node)
            if target is not None and target.type in {ATTRIBUTE, "subscript"}:
                return True
        if node.type == "delete_statement":
            return True
        if node.type == CALL and _call_name(node) in MUTATING_METHODS:
            return True
    return False


def _has_type_annotations(nodes):
    return any(
        node.type in {
            "type",
            "typed_parameter",
            "typed_default_parameter",
            "type_alias_statement",
        }
        for node in nodes
    )


def _has_broad_exception(nodes):
    for node in nodes:
        if node.type != "except_clause":
            continue
        value = node.child_by_field_name("value")
        if value is None:
            named = [
                child
                for child in node.named_children
                if child.type != "block"
            ]
            if not named:
                return True
            value = named[0]
        names = {
            get_node_text(item)
            for item in walk(value)
            if item.type == "identifier"
        }
        if names & {"Exception", "BaseException"}:
            return True
    return False


def _numeric_value(node):
    text = get_node_text(node).replace("_", "")
    try:
        return float(text) if node.type == "float" else int(text, 0)
    except ValueError:
        return None


def _has_magic_numbers(nodes):
    for node in nodes:
        if node.type not in {"integer", "float"}:
            continue
        value = _numeric_value(node)
        if value is not None and value not in {-1, 0, 1}:
            return True
    return False


def extract_operational_features(root_node) -> dict:
    nodes = list(walk(root_node))
    imports = _import_roots(root_node)
    call_names = {
        name
        for node in nodes
        if node.type == CALL
        for name in [_call_name(node)]
        if name is not None
    }
    has_async = any(
        node.type == "await"
        or (
            node.type in {"function_definition", "for_statement", "with_statement"}
            and is_async_node(node)
        )
        for node in nodes
    )
    source = get_node_text(root_node)
    has_synchronization = bool(
        imports & CONCURRENCY_MODULES
        and call_names & SYNCHRONIZATION_NAMES
        or re.search(
            r"\b(?:asyncio|threading|multiprocessing)\."
            r"(?:Lock|RLock|Semaphore|Event|Condition|Barrier|Queue)\b",
            source,
        )
    )
    return {
        "has_bitwise_operations": _has_operator(nodes, BITWISE_OPERATORS),
        "has_concurrency_primitives": bool(
            has_async
            or imports & CONCURRENCY_MODULES
            or has_synchronization
        ),
        "has_synchronization": has_synchronization,
        "has_io": bool(
            imports & IO_MODULES
            or call_names & IO_NAMES
            or re.search(
                r"\b(?:os\.(?:read|write|open)|"
                r"Path\([^)]*\)\.(?:read|write)_(?:text|bytes))\s*\(",
                source,
            )
        ),
        "has_networking": bool(
            imports & NETWORK_MODULES
            or re.search(
                r"\b(?:socket|requests|urllib|aiohttp|websockets)\.",
                source,
            )
        ),
        "has_subprocesses": bool(
            imports & SUBPROCESS_MODULES
            or call_names & SUBPROCESS_NAMES
            and re.search(
                r"\b(?:subprocess|os)\.",
                source,
            )
        ),
        "has_generator": any(
            node.type in {"yield", "generator_expression"} for node in nodes
        ),
        "has_type_annotations": _has_type_annotations(nodes),
        "has_mutation": _has_mutation(nodes),
        "has_pattern_matching": any(
            node.type == "match_statement" for node in nodes
        ),
        "has_broad_exception": _has_broad_exception(nodes),
        "has_magic_numbers": _has_magic_numbers(nodes),
    }
