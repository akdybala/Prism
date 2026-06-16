"""Conservative local call-graph extraction for Python snippets."""

from collections import defaultdict

from .node_types import (
    ATTRIBUTE,
    CALL,
    CLASS_DEFINITION,
    FUNCTION_DEFINITION,
    IDENTIFIER,
)
from .parser import get_node_text, walk


def _ancestor_names(node):
    names = []
    current = node.parent
    while current is not None:
        if current.type in {FUNCTION_DEFINITION, CLASS_DEFINITION}:
            name = current.child_by_field_name("name")
            if name is not None:
                names.append(get_node_text(name))
        current = current.parent
    return list(reversed(names))


def _function_id(function):
    name = function.child_by_field_name("name")
    parts = _ancestor_names(function)
    parts.append(get_node_text(name) if name is not None else "<anonymous>")
    return ".".join(parts)


def _enclosing_class_name(function):
    current = function.parent
    while current is not None:
        if current.type == CLASS_DEFINITION:
            name = current.child_by_field_name("name")
            return get_node_text(name) if name is not None else None
        if current.type == FUNCTION_DEFINITION:
            return None
        current = current.parent
    return None


def _callee_parts(call):
    callee = call.child_by_field_name("function")
    if callee is None and call.named_children:
        callee = call.named_children[0]
    if callee is None:
        return None, None
    if callee.type == IDENTIFIER:
        return get_node_text(callee), None
    if callee.type == ATTRIBUTE:
        obj = callee.child_by_field_name("object")
        attr = callee.child_by_field_name("attribute")
        return (
            get_node_text(attr) if attr is not None else None,
            get_node_text(obj) if obj is not None else None,
        )
    return None, None


def _calls_in_function(function):
    calls = []

    def visit(node):
        if node.type == "ERROR":
            return
        if node is not function and node.type == FUNCTION_DEFINITION:
            return
        if node.type == CALL:
            calls.append(node)
        for child in node.children:
            visit(child)

    visit(function)
    return calls


def _strongly_connected_components(graph):
    index = 0
    indices = {}
    lowlinks = {}
    stack = []
    on_stack = set()
    components = []

    def connect(node):
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in graph[node]:
            if target not in indices:
                connect(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])

        if lowlinks[node] == indices[node]:
            component = set()
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == node:
                    break
            components.append(component)

    for node in graph:
        if node not in indices:
            connect(node)
    return components


def _max_simple_chain_depth(graph):
    def depth(node, path):
        if node in path:
            return 0
        next_path = path | {node}
        return 1 + max(
            (depth(target, next_path) for target in graph[node]),
            default=0,
        )

    return max((depth(node, set()) for node in graph), default=0)


def analyze_local_call_graph(root_node) -> dict:
    """Resolve direct local calls and unambiguous self/cls method calls.

    Dynamic dispatch, callbacks, imported functions, constructor-returned
    objects, and ambiguous duplicate names are intentionally unresolved.
    """
    functions = [
        node for node in walk(root_node) if node.type == FUNCTION_DEFINITION
    ]
    base_ids = [_function_id(function) for function in functions]
    base_counts = defaultdict(int)
    for base_id in base_ids:
        base_counts[base_id] += 1
    function_ids = {}
    for base_id, function in zip(base_ids, functions):
        function_id = (
            base_id
            if base_counts[base_id] == 1
            else f"{base_id}@{function.start_byte}"
        )
        function_ids[function_id] = function
    id_by_node = {id(function): name for name, function in function_ids.items()}
    by_simple_name = defaultdict(list)
    methods_by_class = defaultdict(list)
    class_by_function = {}

    for function_id, function in function_ids.items():
        name = function.child_by_field_name("name")
        if name is None:
            continue
        simple_name = get_node_text(name)
        by_simple_name[simple_name].append(function_id)
        class_name = _enclosing_class_name(function)
        class_by_function[function_id] = class_name
        if class_name is not None:
            methods_by_class[(class_name, simple_name)].append(function_id)

    graph = {function_id: set() for function_id in function_ids}
    unresolved_call_count = 0
    for function in functions:
        source = id_by_node[id(function)]
        class_name = class_by_function.get(source)
        for call in _calls_in_function(function):
            callee, obj = _callee_parts(call)
            target = None
            if callee is not None and obj is None:
                candidates = by_simple_name.get(callee, [])
                if len(candidates) == 1:
                    target = candidates[0]
            elif (
                callee is not None
                and obj in {"self", "cls"}
                and class_name is not None
            ):
                candidates = methods_by_class.get((class_name, callee), [])
                if len(candidates) == 1:
                    target = candidates[0]

            if target is None:
                unresolved_call_count += 1
            else:
                graph[source].add(target)

    components = _strongly_connected_components(graph)
    direct_recursive = {
        node for node, targets in graph.items() if node in targets
    }
    indirect_components = [
        component for component in components if len(component) > 1
    ]
    indirect_recursive = set().union(*indirect_components) if indirect_components else set()
    recursive = direct_recursive | indirect_recursive
    largest_cycle = max(
        [1 for _ in direct_recursive]
        + [len(component) for component in indirect_components],
        default=0,
    )

    return {
        "local_function_count": len(functions),
        "local_call_edge_count": sum(len(targets) for targets in graph.values()),
        "max_local_call_chain_depth": _max_simple_chain_depth(graph),
        "has_direct_recursion": bool(direct_recursive),
        "has_indirect_recursion": bool(indirect_recursive),
        "recursive_function_count": len(recursive),
        "largest_recursive_cycle": largest_cycle,
        "unresolved_local_call_count": unresolved_call_count,
        "_graph": graph,
    }
