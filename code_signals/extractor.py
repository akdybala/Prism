from pathlib import Path

from .data_flow import extract_data_flow
from .domain import DomainClassifier
from .parser import count_errors, parse
from .semantic import extract_semantic
from .structural import extract_structural


_default_examples = Path(__file__).resolve().parent.parent / "domain_examples.json"
_domain_classifier = DomainClassifier(_default_examples)


def extract_all(code: str) -> dict:
    tree = parse(code)
    root = tree.root_node
    error_count = count_errors(root)
    return {
        "structural": extract_structural(root, code),
        "data_flow": extract_data_flow(root),
        "semantic": extract_semantic(root, code),
        "domain": _domain_classifier.predict(code, root_node=root),
        "has_errors": error_count > 0,
        "error_count": error_count,
    }
