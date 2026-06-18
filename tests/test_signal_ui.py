import unittest
from unittest.mock import patch
from urllib.parse import urlparse

import signal_ui


class SignalUITests(unittest.TestCase):
    def test_page_contains_separate_inputs_and_results(self):
        self.assertIn('id="code-input"', signal_ui.PAGE)
        self.assertIn('id="query-input"', signal_ui.PAGE)
        self.assertIn('id="code-results"', signal_ui.PAGE)
        self.assertIn('id="query-results"', signal_ui.PAGE)
        self.assertIn("Rule-domain vector", signal_ui.PAGE)
        self.assertIn("Embedding-domain vector", signal_ui.PAGE)
        self.assertIn("Data-flow signals", signal_ui.PAGE)
        self.assertIn("embedding confidence", signal_ui.PAGE)
        self.assertIn("Tree-sitter V7", signal_ui.PAGE)
        self.assertIn("V7 embedding sketch", signal_ui.PAGE)
        self.assertIn("domain.embedding_sketch", signal_ui.PAGE)
        self.assertIn("Query-operation vector", signal_ui.PAGE)
        self.assertIn("Query concerns (multi-label)", signal_ui.PAGE)
        self.assertIn('id="router-button"', signal_ui.PAGE)
        self.assertIn('id="router-results"', signal_ui.PAGE)
        self.assertIn("Minimum Capable Router", signal_ui.PAGE)
        self.assertNotIn("combined confidence", signal_ui.PAGE)

    @patch("code_signals.extract_all")
    def test_code_payload_delegates_to_extractor(self, extract_all):
        extract_all.return_value = {"structural": {}, "domain": {}}
        result = signal_ui.extract_code_payload("value = 1")
        self.assertEqual(result, extract_all.return_value)
        extract_all.assert_called_once_with("value = 1")

    @patch("query_signals.extract_query_signals")
    def test_query_payload_delegates_to_extractor(self, extract_query_signals):
        extract_query_signals.return_value = {"query_operation": {}}
        result = signal_ui.extract_query_payload("Why is this slow?")
        self.assertEqual(result, extract_query_signals.return_value)
        extract_query_signals.assert_called_once_with("Why is this slow?")

    @patch("routing.route_request")
    def test_route_payload_delegates_to_router(self, route_request):
        decision = route_request.return_value
        decision.to_dict.return_value = {"selected_candidate": "balanced"}
        result = signal_ui.extract_route_payload(
            "Review this code.",
            "value = 1",
            context_tokens=100,
            expected_output_tokens=500,
            quality_threshold=0.8,
        )
        self.assertEqual(result, {"selected_candidate": "balanced"})
        route_request.assert_called_once_with(
            "Review this code.",
            "value = 1",
            context_tokens=100,
            expected_output_tokens=500,
            quality_threshold=0.8,
        )

    @patch("code_signals.extractor._domain_classifier._ensure_embeddings")
    def test_code_model_warmup(self, ensure_embeddings):
        ensure_embeddings.return_value = True
        self.assertTrue(signal_ui.warm_code_model())
        ensure_embeddings.assert_called_once_with()

    def test_health_endpoint_path_is_stable_for_container_probes(self):
        self.assertEqual(urlparse("/api/health").path, "/api/health")
        self.assertIn(
            "/api/health",
            signal_ui.SignalUIHandler.do_GET.__code__.co_consts,
        )


if __name__ == "__main__":
    unittest.main()
