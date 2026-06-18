import unittest

from generate_llm_grounded_pairs import (
    balanced_sample,
    gap_targeted_sample,
    validate_batch,
)


class LLMGroundedPairTests(unittest.TestCase):
    def test_balanced_sample_returns_unique_families(self):
        functions = [
            {
                "family_id": str(index),
                "source": "a" if index % 2 else "b",
                "query_domain": "general" if index % 3 else "testing",
            }
            for index in range(20)
        ]
        selected = balanced_sample(functions, 10)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({item["family_id"] for item in selected}), 10)

    def test_validation_accepts_independent_coverage(self):
        operations = (
            "explain", "debug", "optimize", "review",
            "generate", "refactor", "debug", "review",
        )
        queries = []
        for index, operation in enumerate(operations):
            queries.append({
                "query": f"Realistic grounded developer request number {index}",
                "style": (
                    "pr_description", "issue_comment", "code_review",
                    "developer_question", "maintenance_request", "incident_note",
                    "issue_comment", "code_review",
                )[index],
                "domain": ("general", "testing", "backend_api")[index % 3],
                "operation": operation,
                "concerns": [
                    ("correctness", "performance", "maintainability", "reliability")[
                        index % 4
                    ]
                ],
                "ambiguity": ("low", "medium", "high")[index % 3],
                "grounding": "Uses the concrete return branch.",
            })
        result = {"functions": [{"family_id": "f1", "queries": queries}]}
        self.assertEqual(
            validate_batch(result, [{"family_id": "f1"}], 8),
            [],
        )

    def test_gap_sample_adds_domain_and_concern_targets(self):
        functions = [
            {
                "family_id": str(index),
                "source": "repo",
                "query_domain": domain,
            }
            for index, domain in enumerate(
                ["machine_learning", "database", "frontend", "general"] * 10
            )
        ]
        selected = gap_targeted_sample(functions, 12, set())
        self.assertEqual(len(selected), 12)
        self.assertTrue(all(item["target_query_domains"] for item in selected))
        self.assertTrue(all(
            set(item["required_concerns"])
            == {
                "security", "concurrency", "correctness",
                "performance", "reliability", "maintainability",
            }
            for item in selected
        ))


if __name__ == "__main__":
    unittest.main()
