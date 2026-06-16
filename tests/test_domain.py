import ast
import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from code_signals.domain import (
    ALL_DOMAINS,
    DEFAULT_EMBEDDING_REPRESENTATION,
    DomainClassifier,
    compute_rule_scores,
    is_unknown_code,
)


class DomainClassifierTests(unittest.TestCase):
    def test_production_embedding_representation_is_tree_sitter_v7(self):
        classifier = DomainClassifier("does-not-exist.json")
        self.assertEqual(DEFAULT_EMBEDDING_REPRESENTATION, "tree_sitter_v7")
        self.assertEqual(classifier.embedding_representation, "tree_sitter_v7")

        code = """\
import threading

lock = threading.Lock()

def run(payload):
    with lock:
        return process(payload)
"""
        text = classifier._embedding_text(code)
        self.assertIn("import threading", text)
        self.assertIn("lock = threading.Lock()", text)
        self.assertNotIn("# domain-sketch-v6-ranked-comment", text)

        result = classifier.predict(code)
        self.assertEqual(result["embedding_sketch"], text)
        self.assertEqual(
            result["embedding_representation"],
            "tree_sitter_v7",
        )

    def test_full_code_representation_is_explicit_opt_in(self):
        code = "import hashlib\nvalue = hashlib.md5(data).hexdigest()\n"
        classifier = DomainClassifier(
            "does-not-exist.json",
            embedding_representation="full_code",
        )
        self.assertEqual(classifier._embedding_text(code), code)

    def test_embedding_cache_is_separated_by_representation(self):
        sketch = DomainClassifier(
            "does-not-exist.json",
            embedding_representation="ranked_sketch",
        )
        full = DomainClassifier(
            "does-not-exist.json",
            embedding_representation="full_code",
        )
        sketch.examples = [
            {"code": "import hashlib", "domain": "cryptography"},
        ]
        full.examples = list(sketch.examples)
        v7 = DomainClassifier(
            "does-not-exist.json",
            embedding_representation="tree_sitter_v7",
        )
        v7.examples = list(sketch.examples)
        self.assertNotEqual(
            sketch.embedding_cache_path,
            full.embedding_cache_path,
        )
        self.assertNotEqual(
            sketch.embedding_cache_path,
            v7.embedding_cache_path,
        )

    def test_production_dataset_is_balanced_and_contrastive(self):
        path = Path(__file__).parents[1] / "domain_examples.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(data), set(ALL_DOMAINS))
        self.assertEqual(len(data["unknown"]), 22)
        self.assertEqual(
            {
                len(records)
                for domain, records in data.items()
                if domain != "unknown"
            },
            {38},
        )

        names = {
            domain: {record["name"] for record in records}
            for domain, records in data.items()
        }
        self.assertIn("threaded_json_record_store", names["concurrency"])
        self.assertIn("local_payload_hmac", names["cryptography"])
        self.assertIn("local_event_submission", names["general_python"])
        self.assertIn("http_payload_record_route", names["backend_api"])
        self.assertIn("signed_payload_webhook_route", names["backend_api"])

        removed_backend_names = {
            "oauth2_pkce_flow",
            "circuit_breaker_pattern",
            "distributed_lock_redis",
            "event_driven_saga",
            "idempotency_key_store",
            "health_check_probe",
        }
        self.assertTrue(removed_backend_names.isdisjoint(names["backend_api"]))

    def test_unknown_examples_are_not_loaded_into_embedding_index(self):
        path = Path(__file__).parents[1] / "domain_examples.json"
        classifier = DomainClassifier(path)
        self.assertEqual(len(classifier.examples), 38 * 11)
        self.assertEqual(classifier.example_counts["unknown"], 0)
        self.assertFalse(
            any(
                example["domain"] == "unknown"
                for example in classifier.examples
            )
        )

    def test_training_data_contains_import_evidence_hard_negatives(self):
        path = (
            Path(__file__).parents[1]
            / "domain_examples_import_balanced.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        for domain, records in data.items():
            if domain == "unknown":
                continue
            import_free = 0
            misleading = 0
            for record in records:
                tree = ast.parse(record["code"])
                if not any(
                    isinstance(node, (ast.Import, ast.ImportFrom))
                    for node in ast.walk(tree)
                ):
                    import_free += 1
                misleading += record["name"].startswith(
                    "misleading_import_"
                )
            with self.subTest(domain=domain):
                self.assertGreaterEqual(import_free, 12)
                self.assertGreaterEqual(misleading, 5)

    def test_training_code_has_only_intentional_parse_errors_and_fits_budget(self):
        from code_signals.domain_sketch import build_domain_sketch
        from code_signals.parser import count_errors, parse

        path = Path(__file__).parents[1] / "domain_examples.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for domain, records in data.items():
            for record in records:
                root = parse(record["code"]).root_node
                intentional_error = "broken_" in record["name"]
                if intentional_error:
                    self.assertGreater(
                        count_errors(root),
                        0,
                        msg=f"{domain}/{record['name']}",
                    )
                else:
                    self.assertEqual(
                        count_errors(root),
                        0,
                        msg=f"{domain}/{record['name']}",
                    )
                if domain != "unknown":
                    sketch = build_domain_sketch(record["code"], root)
                    self.assertLessEqual(
                        len(sketch),
                        950,
                        msg=f"{domain}/{record['name']}",
                    )

    def test_production_data_has_naked_and_long_v7_coverage(self):
        from code_signals.tree_sitter_coherent_sketch import (
            build_tree_sitter_coherent_v7_sketch,
        )

        class SimpleTokenizer:
            def encode(
                self,
                text,
                *,
                add_special_tokens=False,
                truncation=False,
            ):
                return re.findall(r"\w+|[^\w\s]", text)

        path = Path(__file__).parents[1] / "domain_examples.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        all_code = []
        for domain, records in data.items():
            all_code.extend(record["code"].strip() for record in records)
            if domain == "unknown":
                continue
            import_free = 0
            for record in records:
                try:
                    tree = ast.parse(record["code"])
                except SyntaxError:
                    has_import = bool(
                        re.search(
                            r"^\s*(?:from|import)\s+",
                            record["code"],
                            re.MULTILINE,
                        )
                    )
                else:
                    has_import = any(
                        isinstance(node, (ast.Import, ast.ImportFrom))
                        for node in ast.walk(tree)
                    )
                import_free += not has_import
            self.assertGreaterEqual(import_free, 5, msg=domain)
            expected_contrast = domain in {
                "machine_learning",
                "database",
                "backend_api",
                "general_python",
            }
            self.assertEqual(
                sum(
                    record["name"]
                    == f"v7_contrast_{domain}_misleading_import"
                    for record in records
                ),
                int(expected_contrast),
                msg=domain,
            )

            long_records = [
                record
                for record in records
                if record["name"] == f"v7_long_{domain}_module"
            ]
            self.assertEqual(len(long_records), 1, msg=domain)
            _, details = build_tree_sitter_coherent_v7_sketch(
                long_records[0]["code"],
                SimpleTokenizer(),
                base_budget=512,
                dynamic_expansion=False,
            )
            self.assertLess(
                details["selected_count"],
                details["candidate_count"],
                msg=domain,
            )
            self.assertEqual(details["output_parse_errors"], 0, msg=domain)
            self.assertLessEqual(details["token_count"], 512, msg=domain)

        self.assertEqual(len(all_code), len(set(all_code)))

    def test_every_indexed_domain_has_naked_and_malformed_examples(self):
        from code_signals.parser import count_errors, parse

        data = json.loads(
            (Path(__file__).parents[1] / "domain_examples.json").read_text(
                encoding="utf-8"
            )
        )
        for domain, records in data.items():
            if domain == "unknown":
                continue
            malformed = [
                record
                for record in records
                if count_errors(parse(record["code"]).root_node) > 0
            ]
            import_free = [
                record
                for record in records
                if not re.search(
                    r"^\s*(?:from|import)\s+",
                    record["code"],
                    re.MULTILINE,
                )
            ]
            with self.subTest(domain=domain):
                self.assertGreaterEqual(len(malformed), 1)
                self.assertGreaterEqual(len(import_free), 1)

    def test_mixed_domain_examples_are_separate_and_well_formed(self):
        path = Path(__file__).parents[1] / "mixed_domain_examples.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(records), 46)
        self.assertEqual(
            len({record["code"] for record in records}),
            len(records),
        )
        for record in records:
            self.assertIn(record["primary_domain"], ALL_DOMAINS)
            self.assertTrue(record["code"].strip())
            if "broken_" in record["name"]:
                with self.assertRaises(SyntaxError):
                    ast.parse(record["code"])
            else:
                ast.parse(record["code"])
            self.assertNotIn(
                record["primary_domain"],
                record["secondary_domains"],
            )
            self.assertTrue(set(record["secondary_domains"]) <= set(ALL_DOMAINS))
        self.assertGreaterEqual(
            sum("broken_" in record["name"] for record in records),
            3,
        )

    def test_single_domain_holdout_is_separate_and_balanced(self):
        root = Path(__file__).parents[1]
        training = json.loads(
            (root / "domain_examples.json").read_text(encoding="utf-8")
        )
        holdout = json.loads(
            (root / "code_domain_holdout.json").read_text(encoding="utf-8")
        )
        training_code = {
            record["code"].strip()
            for domain, records in training.items()
            if domain != "unknown"
            for record in records
        }
        counts = Counter(record["domain"] for record in holdout)
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(set(counts), set(ALL_DOMAINS) - {"unknown"})
        self.assertTrue(
            all(record["code"].strip() not in training_code for record in holdout)
        )

    def test_grouped_examples_load_all_valid_records(self):
        expected_counts = {
            domain: index + 3 for index, domain in enumerate(ALL_DOMAINS)
        }
        data = {
            domain: [
                {
                    "name": f"{domain}_{index}",
                    "description": "example",
                    "code": f"value_{index} = {index}",
                }
                for index in range(count)
            ]
            for domain, count in expected_counts.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            classifier = DomainClassifier(path)

        counts = Counter(example["domain"] for example in classifier.examples)
        expected_indexed = {
            domain: count
            for domain, count in expected_counts.items()
            if domain != "unknown"
        }
        self.assertEqual(dict(counts), expected_indexed)
        self.assertEqual(
            len(classifier.examples),
            sum(expected_indexed.values()),
        )

    def test_rule_scores_use_revised_domains(self):
        scores = compute_rule_scores(
            "import hashlib\nresult = hashlib.sha256(payload).digest()"
        )
        self.assertEqual(set(scores), set(ALL_DOMAINS))
        self.assertEqual(max(scores, key=scores.get), "cryptography")

    def test_algorithm_naked_snippet(self):
        code = """\
visited = set()
stack = [start]
while stack:
    node = stack.pop()
    for neighbor in graph[node]:
        if neighbor not in visited:
            stack.append(neighbor)
"""
        scores = compute_rule_scores(code)
        self.assertEqual(max(scores, key=scores.get), "algorithms")

    def test_no_signal_uses_valid_fallback(self):
        scores = compute_rule_scores("value = 42")
        self.assertEqual(max(scores, key=scores.get), "general_python")

    def test_empty_code_is_unknown(self):
        scores = compute_rule_scores("")
        self.assertEqual(max(scores, key=scores.get), "unknown")

    def test_placeholder_code_is_unknown_but_basic_function_is_general(self):
        self.assertTrue(is_unknown_code("def todo():\n    pass"))
        self.assertTrue(is_unknown_code("class Placeholder:\n    ..."))
        self.assertFalse(is_unknown_code("def add(a, b):\n    return a + b"))
        placeholder = compute_rule_scores("def todo():\n    pass")
        basic = compute_rule_scores("def add(a, b):\n    return a + b")
        self.assertEqual(max(placeholder, key=placeholder.get), "unknown")
        self.assertEqual(max(basic, key=basic.get), "general_python")

    def test_missing_examples_falls_back_to_rules(self):
        classifier = DomainClassifier("does-not-exist.json")
        result = classifier.predict("from fastapi import FastAPI")
        self.assertIsNone(result["embedding_scores"])
        self.assertEqual(result["predicted_domain"], "backend_api")


if __name__ == "__main__":
    unittest.main()
