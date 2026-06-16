import tree_sitter_python as tspython
from tree_sitter import Language, Parser


PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


def parse(code: str):
    return _parser.parse(code.encode("utf-8"))


def count_errors(root_node) -> int:
    count = 0

    def visit(node):
        nonlocal count
        if node.type == "ERROR" or node.is_missing:
            count += 1
        for child in node.children:
            visit(child)

    visit(root_node)
    return count


def get_node_text(node) -> str:
    return node.text.decode("utf-8")


def is_async_node(node) -> bool:
    return any(child.type == "async" for child in node.children)


def walk(node, skip_errors: bool = True):
    yield node
    if skip_errors and node.type == "ERROR":
        return
    for child in node.children:
        yield from walk(child, skip_errors)


def find_children_by_type(node, type_name: str):
    return [child for child in node.children if child.type == type_name]


def find_descendants_by_type(node, type_name: str, skip_errors: bool = True):
    return [desc for desc in walk(node, skip_errors) if desc.type == type_name]
