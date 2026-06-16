import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from code_signals.domain import (
    ALL_DOMAINS,
    DEFAULT_EMBEDDING_MODEL,
    DomainClassifier,
    compute_rule_scores,
    extract_imports,
)
from code_signals.extractor import extract_all
from code_signals.parser import parse
from code_signals.semantic import extract_semantic
from code_signals.structural import extract_structural


def structural(code):
    return extract_structural(parse(code).root_node, code)


def semantic(code):
    return extract_semantic(parse(code).root_node, code)


class HardStructuralTests(unittest.TestCase):
    def test_operational_features_use_conservative_static_evidence(self):
        code = """\
import asyncio
import socket
import subprocess
from pathlib import Path

async def process(flags: int):
    lock = asyncio.Lock()
    with open("input.txt") as handle:
        value = handle.read()
    socket.socket().connect(("localhost", 8080))
    subprocess.run(["echo", "ready"])
    flags ^= 0x08
    values.append(value)
    yield flags
    match flags:
        case 2:
            pass
    try:
        run()
    except Exception:
        pass
"""
        result = structural(code)
        expected_true = {
            "has_bitwise_operations",
            "has_concurrency_primitives",
            "has_synchronization",
            "has_io",
            "has_networking",
            "has_subprocesses",
            "has_generator",
            "has_type_annotations",
            "has_mutation",
            "has_pattern_matching",
            "has_broad_exception",
            "has_magic_numbers",
        }
        for signal in expected_true:
            with self.subTest(signal=signal):
                self.assertTrue(result[signal])

    def test_generic_method_names_do_not_imply_operational_domains(self):
        code = """\
class Service:
    def read(self):
        return self.connect()

    def connect(self):
        return Lock()
"""
        result = structural(code)
        self.assertFalse(result["has_io"])
        self.assertFalse(result["has_networking"])
        self.assertFalse(result["has_synchronization"])
        self.assertFalse(result["has_concurrency_primitives"])

    def test_local_call_chain_metrics(self):
        code = """\
def first():
    return second()

def second():
    return third()

def third():
    return 1
"""
        result = structural(code)
        self.assertEqual(result["local_function_count"], 3)
        self.assertEqual(result["local_call_edge_count"], 2)
        self.assertEqual(result["max_local_call_chain_depth"], 3)
        self.assertFalse(result["has_direct_recursion"])
        self.assertFalse(result["has_indirect_recursion"])
        self.assertEqual(result["recursive_function_count"], 0)
        self.assertEqual(result["largest_recursive_cycle"], 0)

    def test_indirect_recursion_and_method_cycle(self):
        code = """\
def first():
    return second()

def second():
    return third()

def third():
    return first()

class Pair:
    def left(self):
        return self.right()

    def right(self):
        return self.left()
"""
        result = structural(code)
        self.assertFalse(result["has_direct_recursion"])
        self.assertTrue(result["has_indirect_recursion"])
        self.assertEqual(result["recursive_function_count"], 5)
        self.assertEqual(result["largest_recursive_cycle"], 3)
        self.assertEqual(result["cognitive_complexity"], 5)

    def test_ambiguous_and_dynamic_calls_are_not_guessed(self):
        code = """\
def work():
    return 1

class Worker:
    def work(self):
        return 2

def caller(callback):
    callback()
    return work()
"""
        result = structural(code)
        self.assertEqual(result["local_function_count"], 3)
        self.assertEqual(result["local_call_edge_count"], 0)
        self.assertEqual(result["max_local_call_chain_depth"], 1)

    def test_duplicate_local_definitions_remain_distinct_and_unresolved(self):
        code = """\
if condition:
    def work():
        return 1
else:
    def work():
        return 2

def caller():
    return work()
"""
        result = structural(code)
        self.assertEqual(result["local_function_count"], 3)
        self.assertEqual(result["local_call_edge_count"], 0)

    def test_cyclomatic_uses_max_function_and_excludes_nested_function_body(self):
        code = """\
def outer(x):
    if x:
        return x

    def inner(a, b, c):
        if a and b or c:
            return 1
        return 0

    return inner(True, True, False)
"""
        result = structural(code)
        self.assertEqual(result["num_function_defs"], 2)
        self.assertEqual(result["cyclomatic_complexity"], 4)

    def test_comprehension_inside_loop_counts_as_nested_loop(self):
        code = """\
for row in rows:
    values = [x for x in row if x > 0 if x % 2]
"""
        result = structural(code)
        self.assertEqual(result["num_loops"], 1)
        self.assertEqual(result["num_comprehensions"], 1)
        self.assertTrue(result["has_nested_loops"])
        self.assertEqual(result["cyclomatic_complexity"], 4)

    def test_method_recursion_and_decorated_function_length(self):
        code = """\
class Walker:
    @trace
    def walk(self, node):
        # ignored
        if node:
            return self.walk(node.parent)
        return None
"""
        result = structural(code)
        self.assertTrue(result["has_recursion"])
        self.assertEqual(result["cognitive_complexity"], 2)
        self.assertEqual(result["max_function_length"], 5)

    def test_boolean_not_chain_and_operator_switches(self):
        code = """\
if not (a and b or not c and d):
    act()
"""
        result = structural(code)
        self.assertEqual(result["max_bool_expr_complexity"], 5)
        self.assertEqual(result["cognitive_complexity"], 4)
        self.assertEqual(result["cyclomatic_complexity"], 5)

    def test_match_except_async_and_generator_expression(self):
        code = """\
async def consume(source):
    try:
        async for item in source:
            match item:
                case {"ok": value} if value:
                    await save(value)
                case _:
                    raise ValueError
    except (TypeError, ValueError):
        return (x async for x in source)
"""
        result = structural(code)
        self.assertTrue(result["has_async"])
        self.assertEqual(result["exception_handler_count"], 1)
        self.assertEqual(result["num_loops"], 1)
        self.assertEqual(result["num_comprehensions"], 1)
        self.assertGreaterEqual(result["cyclomatic_complexity"], 4)

    def test_empty_input_contract(self):
        with patch(
            "code_signals.extractor._domain_classifier.compute_embedding_scores",
            return_value=None,
        ):
            result = extract_all("")
        self.assertFalse(result["has_errors"])
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["structural"]["sloc"], 0)
        self.assertEqual(result["structural"]["cyclomatic_complexity"], 1)
        self.assertEqual(
            result["domain"]["predicted_domain"], "unknown"
        )


class HardSemanticTests(unittest.TestCase):
    def test_reassignment_isolated_across_module_function_and_class_scopes(self):
        code = """\
x = 1

def first():
    x = 1
    x = 2

def second():
    x = 1

class Box:
    x = 1
    x = 2

    def method(self):
        x = 1
"""
        result = semantic(code)
        self.assertEqual(result["variable_reassignment_count"], 2)

    def test_augmented_assignment_counts_as_another_assignment(self):
        result = semantic("total = 0\ntotal += 1\ntotal += 2\n")
        self.assertEqual(result["variable_reassignment_count"], 2)

    def test_dynamic_feature_variants(self):
        cases = [
            "Generated = type('Generated', (Base,), {'x': 1})",
            "class Plugin(Base, metaclass=Registry):\n    pass",
            "def __init_subclass__(cls):\n    pass",
            "from importlib import import_module",
        ]
        for code in cases:
            with self.subTest(code=code):
                self.assertTrue(semantic(code)["has_dynamic_features"])

    def test_single_character_targets_cover_unpacking_except_and_with(self):
        code = """\
a, long_name = pair
for i, item in rows:
    pass
try:
    run()
except RuntimeError as e:
    pass
with open(path) as f:
    pass
"""
        result = semantic(code)
        self.assertEqual(result["single_char_var_ratio"], round(4 / 6, 3))

    def test_call_diversity_uses_method_name_and_deduplicates(self):
        code = """\
print(value)
obj.save()
other.save()
factory()()
items[0]()
"""
        result = semantic(code)
        self.assertEqual(result["call_diversity"], 3)


class HardDomainTests(unittest.TestCase):
    def test_uses_public_nomic_code_model_id(self):
        classifier = DomainClassifier(Path("missing.json"))
        self.assertEqual(
            classifier.model_name, "nomic-ai/CodeRankEmbed"
        )
        self.assertEqual(classifier.model_name, DEFAULT_EMBEDDING_MODEL)

    def test_query_embedding_uses_tree_sitter_v7_sketch(self):
        classifier = DomainClassifier(Path("missing.json"))
        classifier.examples = [
            {"code": "import threading", "domain": "concurrency"},
        ]
        classifier.example_counts = classifier._count_examples()
        classifier._model = _FakeModel([1.0, 0.0])
        classifier._embeddings = np.array([[1.0, 0.0]])
        original = classifier._embedding_text
        transformed = []

        def capture_embedding_text(code, root_node=None):
            text = original(code, root_node)
            transformed.append(text)
            return text

        with patch.object(
            classifier,
            "_embedding_text",
            side_effect=capture_embedding_text,
        ) as embedding_text:
            classifier.compute_embedding_scores(
                "import threading\nlock = threading.Lock()"
            )

        embedding_text.assert_called_once()
        self.assertIn("import threading", transformed[0])
        self.assertIn("lock = threading.Lock()", transformed[0])
        self.assertNotIn("# domain-sketch-v6-ranked-comment", transformed[0])

    def test_import_extraction_handles_aliases_multiple_and_relative_imports(self):
        code = """\
import hashlib as hs, asyncio
from sqlalchemy.orm import Session
from .local import helper
from ..package.module import value
"""
        imports = extract_imports(parse(code).root_node)
        self.assertEqual(
            imports, {"hashlib", "asyncio", "sqlalchemy", "local", "package"}
        )

    def test_every_rule_score_is_normalized_and_finite(self):
        code = """\
import hashlib
import asyncio
from fastapi import FastAPI
from sqlalchemy import select

async def endpoint(request, session):
    ciphertext = hashlib.sha256(request.body).digest()
    await asyncio.gather(session.execute(select(User)))
    return ciphertext
"""
        scores = compute_rule_scores(code)
        self.assertEqual(set(scores), set(ALL_DOMAINS))
        self.assertTrue(all(math.isfinite(value) for value in scores.values()))
        self.assertAlmostEqual(sum(scores.values()), 1.0)
        self.assertGreater(scores["cryptography"], 0)
        self.assertGreater(scores["concurrency"], 0)
        self.assertGreater(scores["backend_api"], 0)
        self.assertGreater(scores["database"], 0)

    def test_knn_weighted_vote_with_fake_normalized_embeddings(self):
        classifier = DomainClassifier(Path("missing.json"))
        classifier.examples = [
            {"code": "a", "domain": "cryptography"},
            {"code": "b", "domain": "database"},
            {"code": "c", "domain": "database"},
        ]
        classifier.example_counts = {
            domain: 0 for domain in ALL_DOMAINS
        }
        classifier.example_counts["cryptography"] = 1
        classifier.example_counts["database"] = 2
        classifier._model = _FakeModel([1.0, 0.0])
        classifier._embeddings = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.6, 0.8],
            ]
        )

        scores = classifier.compute_embedding_scores("query", k=3)

        database_vote = (0.8**3 + 0.6**3) / 2
        total_vote = 1.0 + database_vote
        self.assertAlmostEqual(scores["cryptography"], 1.0 / total_vote)
        self.assertAlmostEqual(scores["database"], database_vote / total_vote)
        self.assertAlmostEqual(sum(scores.values()), 1.0)

    def test_balanced_vote_removes_class_size_advantage(self):
        classifier = DomainClassifier(Path("missing.json"))
        classifier.examples = [
            {"code": "crypto", "domain": "cryptography"},
            *[
                {"code": f"db-{index}", "domain": "database"}
                for index in range(4)
            ],
        ]
        classifier.example_counts = {
            domain: 0 for domain in ALL_DOMAINS
        }
        classifier.example_counts["cryptography"] = 1
        classifier.example_counts["database"] = 4
        classifier._model = _FakeModel([1.0, 0.0])
        classifier._embeddings = np.array(
            [
                [0.9, math.sqrt(1 - 0.9**2)],
                *[
                    [0.8, math.sqrt(1 - 0.8**2)]
                    for _ in range(4)
                ],
            ]
        )

        scores = classifier.compute_embedding_scores("query", k=5)

        self.assertGreater(scores["cryptography"], scores["database"])

    def test_prediction_and_confidence_use_embeddings_when_available(self):
        classifier = DomainClassifier(Path("missing.json"))
        rule = {domain: 0.0 for domain in ALL_DOMAINS}
        embedding = {domain: 0.0 for domain in ALL_DOMAINS}
        rule["cryptography"] = 0.9
        rule["database"] = 0.1
        embedding["cryptography"] = 0.1
        embedding["database"] = 0.9

        with patch.object(classifier, "compute_rule_scores", return_value=rule):
            with patch.object(
                classifier,
                "compute_embedding_scores",
                return_value=embedding,
            ):
                result = classifier.predict("code")

        self.assertEqual(result["predicted_domain"], "database")
        self.assertAlmostEqual(result["confidence"], 0.9, places=3)
        self.assertEqual(result["rule_scores"], rule)
        self.assertEqual(result["embedding_scores"], embedding)

    def test_prediction_and_confidence_fall_back_to_rules(self):
        classifier = DomainClassifier(Path("missing.json"))
        rule = {domain: 0.0 for domain in ALL_DOMAINS}
        rule["cryptography"] = 0.73
        rule["database"] = 0.27

        with patch.object(classifier, "compute_rule_scores", return_value=rule):
            with patch.object(
                classifier,
                "compute_embedding_scores",
                return_value=None,
            ):
                result = classifier.predict("code")

        self.assertEqual(result["predicted_domain"], "cryptography")
        self.assertAlmostEqual(result["confidence"], 0.73, places=3)

    def test_malformed_and_unknown_example_records_are_ignored(self):
        content = """{
          "cryptography": [
            {"name": "ok", "description": "valid", "code": "import hashlib"},
            {"name": "missing code"},
            {"code": "   "},
            42
          ],
          "unknown_domain": [{"code": "ignored"}]
        }"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.json"
            path.write_text(content, encoding="utf-8")
            classifier = DomainClassifier(path)
        self.assertEqual(len(classifier.examples), 1)
        self.assertEqual(classifier.examples[0]["domain"], "cryptography")

    def test_embedding_cache_round_trip_and_content_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "cache"
            classifier = DomainClassifier(
                Path("missing.json"),
                embedding_cache_dir=cache_dir,
            )
            classifier.examples = [
                {"code": "import hashlib", "domain": "cryptography"},
                {"code": "import sqlite3", "domain": "database"},
            ]
            classifier.example_counts = classifier._count_examples()
            classifier._embeddings = np.array(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            )
            classifier._save_embedding_cache()

            loaded = DomainClassifier(
                Path("missing.json"),
                embedding_cache_dir=cache_dir,
            )
            loaded.examples = list(classifier.examples)
            loaded.example_counts = loaded._count_examples()
            cached = loaded._load_embedding_cache()

            np.testing.assert_array_equal(cached, classifier._embeddings)
            self.assertTrue(loaded.embedding_cache_hit)

            loaded.examples[0] = {
                "code": "import secrets",
                "domain": "cryptography",
            }
            self.assertIsNone(loaded._load_embedding_cache())


class _FakeModel:
    def __init__(self, vector):
        self.vector = np.asarray(vector)

    def encode(self, values, **kwargs):
        return np.asarray([self.vector for _ in values])


if __name__ == "__main__":
    unittest.main()
