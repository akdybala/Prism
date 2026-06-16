import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from query_signals.classifier import (
    EmbeddingKNNClassifier,
    EmbeddingLogisticClassifier,
)
from query_signals.extractor import (
    NO_DOMAIN_EVIDENCE_CONFIDENCE_THRESHOLD,
    QUERY_CONCERNS,
    QUERY_DOMAINS,
    QUERY_OPERATIONS,
    QUERY_TYPES,
    _concern_classifier,
    _domain_classifier,
    _apply_domain_evidence_guard,
    _extract_concern_evidence,
    _has_domain_signal,
    _type_classifier,
    extract_query_signals,
)


class _FakeModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        vectors = []
        for text in texts:
            value = sum(ord(char) for char in text)
            vectors.append(
                [1.0 + value % 7, 1.0 + value % 11, 1.0 + value % 13]
            )
        return np.asarray(vectors, dtype=np.float32)


class QueryDatasetTests(unittest.TestCase):
    def test_populated_dataset_classes_and_counts(self):
        self.assertEqual(QUERY_TYPES, QUERY_OPERATIONS)
        self.assertEqual(
            set(_type_classifier.all_classes),
            set(QUERY_OPERATIONS),
        )
        self.assertEqual(set(_domain_classifier.all_classes), set(QUERY_DOMAINS))
        self.assertGreaterEqual(len(_type_classifier.labels), 929)
        self.assertGreaterEqual(len(_domain_classifier.labels), 612)
        self.assertLessEqual(
            max(map(len, _type_classifier.examples_by_class.values()))
            - min(map(len, _type_classifier.examples_by_class.values())),
            5,
        )
        self.assertGreaterEqual(len(_concern_classifier.records), 791)
        self.assertTrue(
            all(
                count >= 95
                for count in _concern_classifier.targets.sum(axis=0)
            )
        )
        self.assertGreaterEqual(
            min(map(len, _domain_classifier.examples_by_class.values())),
            51,
        )
        self.assertLessEqual(
            max(map(len, _domain_classifier.examples_by_class.values()))
            - min(map(len, _domain_classifier.examples_by_class.values())),
            5,
        )

    def test_operation_ontology_does_not_contain_concerns(self):
        self.assertNotIn("security", QUERY_OPERATIONS)
        self.assertNotIn("concurrency_check", QUERY_OPERATIONS)
        self.assertIn("security", QUERY_CONCERNS)
        self.assertIn("concurrency", QUERY_CONCERNS)

    def test_operation_dataset_contains_balanced_domain_substitutions(self):
        examples = _type_classifier.examples_by_class
        contexts = (
            "binary-search implementation",
            "ONNX inference pipeline",
            "HTTP endpoint",
            "SQL transaction",
            "threaded queue consumer",
            "React component",
            "socket reader",
            "password-hashing function",
            "pytest fixture",
            "Docker deployment",
            "dataframe transformation",
            "Python helper function",
        )
        for operation in QUERY_OPERATIONS:
            joined = "\n".join(examples[operation])
            for context in contexts:
                self.assertIn(context, joined)

    def test_targeted_concern_data_is_hand_written_and_slice_balanced(self):
        base = Path(__file__).resolve().parents[1] / "query_signals"
        training = json.loads(
            (base / "query_concern_targeted_examples.json").read_text(
                encoding="utf-8"
            )
        )
        holdout = json.loads(
            (base / "query_concern_targeted_holdout.json").read_text(
                encoding="utf-8"
            )
        )
        weak_concerns = set(QUERY_CONCERNS)
        slices = {
            "implicit_positive",
            "domain_only_negative",
            "minimal_pair",
            "mixed_positive",
        }
        self.assertEqual(len(training), 216)
        self.assertEqual(len(holdout), 48)
        self.assertEqual({item["concern"] for item in training}, weak_concerns)
        self.assertEqual({item["slice"] for item in training}, slices)
        self.assertFalse(
            {
                item["text"].casefold()
                for item in training
            }
            & {
                item["text"].casefold()
                for item in holdout
            }
        )
        for concern in weak_concerns:
            concern_training = [
                item for item in training if item["concern"] == concern
            ]
            self.assertEqual(len(concern_training), 36)
            self.assertEqual(
                sum(item["slice"] == "implicit_positive" for item in concern_training),
                10,
            )
            self.assertEqual(
                sum(item["slice"] == "domain_only_negative" for item in concern_training),
                10,
            )
            pair_ids = {
                item["pair_id"]
                for item in concern_training
                if item.get("pair_id")
            }
            self.assertEqual(len(pair_ids), 5)
            for pair_id in pair_ids:
                pair = [
                    item for item in concern_training
                    if item.get("pair_id") == pair_id
                ]
                self.assertEqual(len(pair), 2)
                self.assertEqual(
                    sorted(concern in item["labels"] for item in pair),
                    [False, True],
                )

    def test_query_domain_contains_deadlock_boundary_examples(self):
        examples = _domain_classifier.examples_by_class
        self.assertIn(
            "Is the lock acquisition order causing this tree-processing deadlock?",
            examples["concurrency"],
        )
        self.assertIn(
            "Why do two transactions deadlock while updating cached rows?",
            examples["database"],
        )
        self.assertIn(
            "Why does my transformer encoder produce identical node embeddings?",
            examples["machine_learning"],
        )
        self.assertIn(
            "How does the kernel detect a deadlock between blocked processes?",
            examples["systems_programming"],
        )

    def test_query_domain_contains_generic_safety_boundary_examples(self):
        examples = _domain_classifier.examples_by_class
        self.assertIn(
            "Can you check whether this code is correct and safe, and improve it without changing how it works?",
            examples["general"],
        )
        self.assertIn(
            "Could this implementation leak passwords, tokens, or secret keys?",
            examples["security_crypto"],
        )
        self.assertIn(
            "Which unit tests prove this function preserves existing behavior?",
            examples["testing"],
        )

    def test_query_domain_holdout_is_balanced_and_disjoint(self):
        base = Path(__file__).resolve().parents[1] / "query_signals"
        training = json.loads(
            (base / "query_domain_examples.json").read_text(encoding="utf-8")
        )
        holdout = json.loads(
            (base / "query_domain_holdout.json").read_text(encoding="utf-8")
        )
        training_text = {
            text.strip().casefold()
            for records in training.values()
            for text in records
        }
        counts = Counter(record["domain"] for record in holdout)
        self.assertEqual(set(counts), set(QUERY_DOMAINS))
        self.assertEqual(set(counts.values()), {2})
        self.assertTrue(
            all(
                record["text"].strip().casefold() not in training_text
                for record in holdout
            )
        )

    def test_production_uses_independently_tuned_logistic_heads(self):
        self.assertIsInstance(_type_classifier, EmbeddingLogisticClassifier)
        self.assertIsInstance(_domain_classifier, EmbeddingLogisticClassifier)
        self.assertEqual(_type_classifier.c_value, 10.0)
        self.assertEqual(_concern_classifier.c_value, 10.0)
        self.assertEqual(
            _concern_classifier.thresholds,
            {
                "security": 0.6,
                "concurrency": 0.5,
                "correctness": 0.4,
                "performance": 0.6,
                "reliability": 0.6,
                "maintainability": 0.45,
            },
        )
        self.assertEqual(_domain_classifier.c_value, 3.0)
        self.assertEqual(_type_classifier.ambiguity_confidence_threshold, 0.35)
        self.assertEqual(_type_classifier.ambiguity_margin_threshold, 0.1)
        self.assertEqual(
            _domain_classifier.ambiguity_confidence_threshold,
            0.5,
        )
        self.assertEqual(_domain_classifier.ambiguity_margin_threshold, 0.15)

    def test_empty_query_is_uniform_without_loading_model(self):
        with patch.object(
            EmbeddingKNNClassifier,
            "_get_model",
            side_effect=AssertionError("model should not load"),
        ):
            result = extract_query_signals("   ")
        self.assertEqual(result["query_operation"]["predicted"], "unknown")
        self.assertEqual(result["query_domain"]["predicted"], "general")
        self.assertEqual(result["query_concerns"]["detected"], [])
        self.assertAlmostEqual(
            sum(result["query_operation"]["scores"].values()),
            1.0,
        )
        self.assertAlmostEqual(
            sum(result["query_domain"]["scores"].values()), 1.0
        )
        self.assertTrue(result["query_operation"]["ambiguous"])
        self.assertTrue(result["query_domain"]["ambiguous"])
        self.assertIn("ambiguity_reasons", result["query_operation"])


class GenericClassifierTests(unittest.TestCase):
    def make_classifier(self, directory):
        examples_path = Path(directory) / "examples.json"
        cache_path = Path(directory) / "cache" / "embeddings.npz"
        examples_path.write_text(
            json.dumps(
                {
                    "alpha": ["alpha one", "alpha two"],
                    "beta": ["beta one", "beta two"],
                }
            ),
            encoding="utf-8",
        )
        return EmbeddingKNNClassifier(examples_path, cache_path, k=3)

    def test_prediction_scores_sum_to_one(self):
        with tempfile.TemporaryDirectory() as directory:
            classifier = self.make_classifier(directory)
            classifier.embeddings = np.asarray(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.1, 0.9],
                ]
            )
            result = classifier.predict_from_embedding(
                np.asarray([1.0, 0.0])
            )
        self.assertEqual(result["predicted"], "alpha")
        self.assertAlmostEqual(sum(result["scores"].values()), 1.0)
        self.assertEqual(result["secondary"], "beta")
        self.assertIn("secondary_confidence", result)
        self.assertIn("margin", result)
        self.assertIn("ambiguous", result)

    def test_ambiguity_uses_confidence_or_top_two_margin(self):
        low_confidence = EmbeddingKNNClassifier._prediction_result(
            {"alpha": 0.49, "beta": 0.31, "gamma": 0.2}
        )
        narrow_margin = EmbeddingKNNClassifier._prediction_result(
            {"alpha": 0.55, "beta": 0.45}
        )
        confident = EmbeddingKNNClassifier._prediction_result(
            {"alpha": 0.7, "beta": 0.3}
        )

        self.assertTrue(low_confidence["ambiguous"])
        self.assertTrue(narrow_margin["ambiguous"])
        self.assertFalse(confident["ambiguous"])
        self.assertEqual(
            low_confidence["ambiguity_reasons"],
            ["low_confidence"],
        )

    def test_domain_signal_guard_distinguishes_specific_and_generic_queries(self):
        self.assertTrue(_has_domain_signal("Can two threads race here?"))
        self.assertTrue(_has_domain_signal("Should I add a database index?"))
        self.assertTrue(_has_domain_signal("Use a monotonic stack?"))
        self.assertFalse(_has_domain_signal("What does this parser do?"))

    def test_missing_domain_signal_only_flags_moderate_confidence(self):
        moderate = {
            "predicted": "systems_programming",
            "confidence": 0.6,
            "ambiguous": False,
            "ambiguity_reasons": [],
        }
        high = {
            "predicted": "systems_programming",
            "confidence": 0.9,
            "ambiguous": False,
            "ambiguity_reasons": [],
        }

        _apply_domain_evidence_guard(moderate, False)
        _apply_domain_evidence_guard(high, False)

        self.assertEqual(NO_DOMAIN_EVIDENCE_CONFIDENCE_THRESHOLD, 0.65)
        self.assertTrue(moderate["ambiguous"])
        self.assertIn(
            "missing_domain_signal",
            moderate["ambiguity_reasons"],
        )
        self.assertFalse(high["ambiguous"])
        self.assertEqual(high["ambiguity_reasons"], [])
        self.assertFalse(high["domain_signal_present"])

    def test_concerns_are_independent_and_multi_label(self):
        evidence = _extract_concern_evidence(
            "Review this async token cache for races and leaked secrets."
        )
        self.assertIn("security", evidence)
        self.assertIn("concurrency", evidence)

    def test_leave_one_out_excludes_self(self):
        with tempfile.TemporaryDirectory() as directory:
            classifier = self.make_classifier(directory)
            classifier.k = 1
            classifier.embeddings = np.asarray(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.1, 0.9],
                ]
            )
            result = classifier.predict_loo(0)
        self.assertEqual(result["predicted"], "alpha")

    def test_cache_round_trip_and_content_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            classifier = self.make_classifier(directory)
            classifier.embeddings = np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
                dtype=np.float32,
            )
            classifier._save_cache()

            loaded = EmbeddingKNNClassifier(
                classifier.examples_path,
                classifier.cache_path,
            )
            cached = loaded._load_cache()
            np.testing.assert_array_equal(cached, classifier.embeddings)

            loaded.texts[0] = "changed"
            self.assertIsNone(loaded._load_cache())

    def test_malformed_examples_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Malformed"):
                EmbeddingKNNClassifier(path, Path(directory) / "cache.npz")

    def test_logistic_head_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.make_classifier(directory)
            classifier = EmbeddingLogisticClassifier(
                base.examples_path,
                base.cache_path,
                Path(directory) / "cache" / "model.npz",
                c_value=3.0,
            )
            classifier.embeddings = np.asarray(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.1, 0.9],
                ],
                dtype=np.float32,
            )
            result = classifier.predict_from_embedding(
                np.asarray([1.0, 0.0])
            )
            self.assertEqual(result["predicted"], "alpha")

            loaded = EmbeddingLogisticClassifier(
                base.examples_path,
                base.cache_path,
                classifier.model_cache_path,
                c_value=3.0,
            )
            loaded.embeddings = classifier.embeddings
            loaded._ensure_prediction_model()
            self.assertTrue(loaded.model_cache_hit)
            cached_result = loaded.predict_from_embedding(
                np.asarray([1.0, 0.0])
            )
            self.assertEqual(result, cached_result)


class QueryExtractorOptimizationTests(unittest.TestCase):
    def test_query_is_encoded_once_for_both_indices(self):
        fake = _FakeModel()
        type_embeddings = _type_classifier.embeddings
        concern_embeddings = _concern_classifier.embeddings
        domain_embeddings = _domain_classifier.embeddings
        type_head = (
            _type_classifier.coef_,
            _type_classifier.intercept_,
            _type_classifier.model_classes_,
        )
        domain_head = (
            _domain_classifier.coef_,
            _domain_classifier.intercept_,
            _domain_classifier.model_classes_,
        )
        concern_head = (
            _concern_classifier.coef_,
            _concern_classifier.intercept_,
        )
        shared_model = EmbeddingKNNClassifier._model
        try:
            EmbeddingKNNClassifier._model = fake
            _type_classifier.embeddings = fake.encode(_type_classifier.texts)
            _concern_classifier.embeddings = fake.encode(
                _concern_classifier.texts
            )
            _domain_classifier.embeddings = fake.encode(
                _domain_classifier.texts
            )
            _type_classifier.coef_ = None
            _type_classifier.intercept_ = None
            _type_classifier.model_classes_ = None
            _domain_classifier.coef_ = None
            _domain_classifier.intercept_ = None
            _domain_classifier.model_classes_ = None
            _concern_classifier.coef_ = None
            _concern_classifier.intercept_ = None
            fake.calls = 0

            result = extract_query_signals("Why is this query slow?")

            self.assertEqual(fake.calls, 1)
            self.assertEqual(
                set(result),
                {
                    "query_operation",
                    "query_concerns",
                    "query_domain",
                },
            )
            self.assertAlmostEqual(
                sum(result["query_operation"]["scores"].values()),
                1.0,
            )
            self.assertAlmostEqual(
                sum(result["query_domain"]["scores"].values()), 1.0
            )
        finally:
            EmbeddingKNNClassifier._model = shared_model
            _type_classifier.embeddings = type_embeddings
            _concern_classifier.embeddings = concern_embeddings
            _domain_classifier.embeddings = domain_embeddings
            (
                _type_classifier.coef_,
                _type_classifier.intercept_,
                _type_classifier.model_classes_,
            ) = type_head
            (
                _domain_classifier.coef_,
                _domain_classifier.intercept_,
                _domain_classifier.model_classes_,
            ) = domain_head
            (
                _concern_classifier.coef_,
                _concern_classifier.intercept_,
            ) = concern_head


if __name__ == "__main__":
    unittest.main()
