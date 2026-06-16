import re

from .domain import API_FINGERPRINTS
from .parser import get_node_text, is_async_node, walk


SKETCH_VERSION = "domain-sketch-v6"
SQL_PATTERN = re.compile(
    r"\b(select|insert\s+into|update|delete\s+from|create\s+table|"
    r"alter\s+table|drop\s+table|join|where|group\s+by|order\s+by)\b",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"(https?://|wss?://|/api(?:/|$))", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[A-Za-z][^>]*>")
REGEX_PATTERN = re.compile(
    r"(\\[AbBdDsSwWZ]|\(\?[=!<:]|\[[^\]]+\]|\{\d+(?:,\d*)?\}|"
    r"\.\*|\.\+|\^\S|\S\$)"
)


def _unique(values):
    return list(dict.fromkeys(value for value in values if value))


def _function_signature(node) -> str:
    name = node.child_by_field_name("name")
    parameters = node.child_by_field_name("parameters")
    return_type = node.child_by_field_name("return_type")
    if name is None or parameters is None:
        return ""
    prefix = "async def" if is_async_node(node) else "def"
    signature = f"{prefix} {get_node_text(name)}{get_node_text(parameters)}"
    if return_type is not None:
        signature += f" -> {get_node_text(return_type)}"
    return signature + ":"


def _class_header(node) -> str:
    name = node.child_by_field_name("name")
    if name is None:
        return ""
    superclasses = node.child_by_field_name("superclasses")
    suffix = get_node_text(superclasses) if superclasses is not None else ""
    return f"class {get_node_text(name)}{suffix}:"


def _call_name(node) -> str:
    function = node.child_by_field_name("function")
    if function is None and node.named_children:
        function = node.named_children[0]
    if function is None or function.type not in {"identifier", "attribute"}:
        return ""
    return get_node_text(function)


def _exception_types(root_node) -> list[str]:
    types = []
    for node in walk(root_node):
        if node.type == "except_clause":
            value = node.child_by_field_name("value")
            if value is None:
                types.append("Exception")
                continue
            if value.type == "as_pattern" and value.named_children:
                value = value.named_children[0]
            if value.type in {"tuple", "list"}:
                types.extend(get_node_text(child) for child in value.named_children)
            else:
                types.append(get_node_text(value))
        elif node.type == "raise_statement" and node.named_children:
            raised = node.named_children[0]
            if raised.type == "call":
                name = _call_name(raised)
                if name:
                    types.append(name)
            elif raised.type in {"identifier", "attribute"}:
                types.append(get_node_text(raised))
    return _unique(types)


def _is_regex_argument(node) -> bool:
    current = node.parent
    while current is not None and current.type in {
        "concatenated_string",
        "argument_list",
        "keyword_argument",
    }:
        current = current.parent
    if current is None or current.type != "call":
        return False
    name = _call_name(current)
    return bool(
        re.search(
            r"(^|\.)(compile|match|fullmatch|search|findall|finditer|sub|split)$",
            name,
        )
    )


def _domain_literals(root_node) -> list[str]:
    literals = []
    for node in walk(root_node):
        if node.type != "string":
            continue
        text = get_node_text(node)
        if (
            SQL_PATTERN.search(text)
            or URL_PATTERN.search(text)
            or HTML_PATTERN.search(text)
            or REGEX_PATTERN.search(text)
            or _is_regex_argument(node)
        ):
            literals.append(text)
    return _unique(literals)


def _docstring_rows(root_node) -> set[int]:
    rows = set()
    scopes = [root_node] + [
        node
        for node in walk(root_node)
        if node.type in {"function_definition", "class_definition"}
    ]
    for scope in scopes:
        body = scope if scope.type == "module" else scope.child_by_field_name("body")
        if body is None or not body.named_children:
            continue
        first = body.named_children[0]
        if (
            first.type == "expression_statement"
            and first.named_children
            and first.named_children[0].type == "string"
        ):
            rows.update(range(first.start_point.row, first.end_point.row + 1))
    return rows


def _excluded_logic_rows(root_node) -> set[int]:
    rows = _docstring_rows(root_node)
    for node in walk(root_node):
        if node.type in {
            "import_statement",
            "import_from_statement",
            "decorator",
        }:
            rows.update(range(node.start_point.row, node.end_point.row + 1))
        elif node.type in {"function_definition", "class_definition"}:
            body = node.child_by_field_name("body")
            header_end = body.start_point.row if body is not None else node.start_point.row
            rows.update(range(node.start_point.row, header_end))
    return rows


def _comment_columns(root_node) -> dict[int, list[tuple[int, int]]]:
    columns = {}
    for node in walk(root_node):
        if node.type != "comment":
            continue
        row = node.start_point.row
        columns.setdefault(row, []).append(
            (node.start_point.column, node.end_point.column)
        )
    return columns


def _all_logic_lines(code: str, root_node) -> list[str]:
    excluded = _excluded_logic_rows(root_node)
    comments = _comment_columns(root_node)
    selected = []
    for row, source_line in enumerate(code.splitlines()):
        if row in excluded:
            continue
        line = source_line
        for start, _ in sorted(comments.get(row, []), reverse=True):
            line = line[:start]
        line = line.strip()
        if not line:
            continue
        selected.append(line)
    return selected


def score_line_domain_signal(line: str) -> float:
    """Score how much domain information a single source line carries."""
    score = 0.0
    stripped = line.strip()

    if "(" in stripped and "." in stripped:
        score += 3.0

    if "=" in stripped:
        right = stripped.split("=", 1)[1]
        if "(" in right:
            score += 2.0

    for patterns in API_FINGERPRINTS.values():
        score += sum(pattern in stripped for pattern in patterns) * 2.0

    if stripped.startswith(
        ("for ", "while ", "async for", "async with", "with ")
    ):
        score += 1.5

    keyword_arguments = stripped.count("=") - (1 if "==" in stripped else 0)
    score += max(keyword_arguments, 0) * 0.5

    if stripped.startswith(("print(", "return", "#", "pass", "else:")):
        score -= 1.0

    return score


def _ranked_logic_lines(lines: list[str], limit: int) -> list[str]:
    scored = [
        (score_line_domain_signal(line), index, line)
        for index, line in enumerate(lines)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        line
        for _, _, line in sorted(scored[:limit], key=lambda item: item[1])
    ]


def _collapse_source(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _significant_assignments(root_node) -> list[str]:
    assignments = []
    for node in walk(root_node):
        if node.type != "assignment":
            continue
        right = node.child_by_field_name("right")
        if right is None:
            continue
        if any(descendant.type == "call" for descendant in walk(right)):
            assignments.append(_collapse_source(get_node_text(node)))
    return _unique(assignments)


def _context_headers(code: str, root_node) -> list[str]:
    encoded = code.encode("utf-8")
    headers = []
    for node in walk(root_node):
        if node.type not in {
            "for_statement",
            "while_statement",
            "with_statement",
        }:
            continue
        body = node.child_by_field_name("body")
        if body is None:
            continue
        header = encoded[node.start_byte : body.start_byte].decode(
            "utf-8",
            errors="replace",
        )
        headers.append(_collapse_source(header))
    return _unique(headers)


def _deduplicate_against(lines: list[str], existing: list[str]) -> list[str]:
    seen = {_collapse_source(line) for line in existing}
    result = []
    for line in lines:
        normalized = _collapse_source(line)
        if normalized and normalized not in seen:
            result.append(line)
            seen.add(normalized)
    return result


def _fit_budget(lines: list[str], max_chars: int) -> str:
    selected = []
    size = 0
    for line in lines:
        required = len(line) + (1 if selected else 0)
        if size + required > max_chars:
            remaining = max_chars - size - (1 if selected else 0)
            if remaining > 0:
                selected.append(line[:remaining])
            break
        selected.append(line)
        size += required
    return "\n".join(selected)


def build_domain_sketch(
    code: str,
    root_node,
    *,
    logic_line_limit: int = 20,
    max_chars: int = 950,
    call_format: str = "comment",
    logic_selection: str = "ranked",
) -> str:
    """Build a compact, code-like representation optimized for domain signals."""
    if call_format not in {"comment", "pseudo"}:
        raise ValueError("call_format must be 'comment' or 'pseudo'")
    if logic_selection not in {"first", "ranked"}:
        raise ValueError("logic_selection must be 'first' or 'ranked'")
    imports = []
    classes = []
    decorators = []
    signatures = []
    calls = []

    for node in walk(root_node):
        if node.type in {"import_statement", "import_from_statement"}:
            imports.append(get_node_text(node))
        elif node.type == "class_definition":
            classes.append(_class_header(node))
        elif node.type == "decorator":
            decorators.append(get_node_text(node))
        elif node.type == "function_definition":
            signatures.append(_function_signature(node))
        elif node.type == "call":
            calls.append(_call_name(node))

    lines = [f"# {SKETCH_VERSION}-{logic_selection}-{call_format}"]
    lines.extend(_unique(imports))
    lines.extend(_unique(classes))
    lines.extend(_unique(decorators))
    lines.extend(_unique(signatures))

    exceptions = _exception_types(root_node)
    if exceptions:
        lines.append(f"# exceptions: {', '.join(exceptions)}")

    literals = _domain_literals(root_node)
    if literals:
        lines.append(f"DOMAIN_LITERALS = ({', '.join(literals)})")

    all_logic = _all_logic_lines(code, root_node)
    if logic_selection == "ranked":
        significant_assignments = _significant_assignments(root_node)
        context_headers = _context_headers(code, root_node)
        lines.extend(significant_assignments)
        lines.extend(
            _deduplicate_against(context_headers, significant_assignments)
        )
        ranked = _ranked_logic_lines(all_logic, min(logic_line_limit, 15))
        lines.extend(
            _deduplicate_against(
                ranked,
                significant_assignments + context_headers,
            )
        )
    else:
        distinct_calls = _unique(calls)
        if distinct_calls:
            if call_format == "pseudo":
                lines.extend(f"{call}()" for call in distinct_calls)
            else:
                lines.append(f"# calls: {', '.join(distinct_calls)}")
        lines.extend(all_logic[:logic_line_limit])
    return _fit_budget(lines, max_chars)
