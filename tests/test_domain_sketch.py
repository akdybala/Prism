import unittest

from code_signals.domain_sketch import (
    build_domain_sketch,
    score_line_domain_signal,
)
from code_signals.parser import parse


class DomainSketchTests(unittest.TestCase):
    def sketch(self, code, **kwargs):
        return build_domain_sketch(code, parse(code).root_node, **kwargs)

    def test_contains_all_signal_components_without_bodies_or_docstrings(self):
        code = '''\
import re
from fastapi import APIRouter

class UserAPI(Base, metaclass=Registry):
    """class docs should disappear"""

    @router.get("/api/users")
    async def users(self, user_id: int) -> list[User]:
        """method docs should disappear"""
        query = "SELECT * FROM users WHERE id = ?"
        pattern = re.compile(r"^user-\\d+$")
        try:
            return await db.fetch(query, user_id)
        except db.Error as exc:
            raise HTTPException("https://example.test/error") from exc
'''
        sketch = self.sketch(code, logic_selection="first")

        self.assertIn("import re", sketch)
        self.assertIn("from fastapi import APIRouter", sketch)
        self.assertIn("class UserAPI(Base, metaclass=Registry):", sketch)
        self.assertIn('@router.get("/api/users")', sketch)
        self.assertIn(
            "async def users(self, user_id: int) -> list[User]:",
            sketch,
        )
        self.assertIn("# exceptions: db.Error, HTTPException", sketch)
        self.assertIn("SELECT * FROM users", sketch)
        self.assertIn(r"^user-\d+$", sketch)
        self.assertIn("https://example.test/error", sketch)
        self.assertIn(
            "# calls: router.get, re.compile, db.fetch, HTTPException",
            sketch,
        )
        self.assertNotIn("docs should disappear", sketch)
        self.assertNotIn('"""', sketch)

    def test_calls_are_distinct_and_logic_is_limited(self):
        code = "\n".join(
            [
                "def run(client):",
                "    client.get('/api/one')",
                "    client.get('/api/two')",
                "    client.post('/api/three')",
                *[f"    value_{index} = {index}" for index in range(30)],
            ]
        )
        sketch = self.sketch(code, logic_selection="first")

        calls_line = next(
            line for line in sketch.splitlines() if line.startswith("# calls:")
        )
        self.assertEqual(calls_line.count("client.get"), 1)
        self.assertEqual(calls_line.count("client.post"), 1)
        self.assertIn("value_16 = 16", sketch)
        self.assertNotIn("value_17 = 17", sketch)

    def test_pseudo_call_format_emits_code_like_calls(self):
        code = """\
def train(loss, optimizer):
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        validate()
"""
        sketch = self.sketch(
            code,
            call_format="pseudo",
            logic_selection="first",
        )

        self.assertIn("loss.backward()", sketch)
        self.assertIn("optimizer.step()", sketch)
        self.assertIn("optimizer.zero_grad()", sketch)
        self.assertIn("torch.no_grad()", sketch)
        self.assertNotIn("# calls:", sketch)

    def test_invalid_call_format_is_rejected(self):
        with self.assertRaises(ValueError):
            self.sketch("run()", call_format="unknown")

    def test_ranked_logic_prefers_domain_calls_over_early_generic_lines(self):
        code = "\n".join(
            [
                "def train(model, loader, optimizer, criterion):",
                *[f"    value_{index} = {index}" for index in range(20)],
                "    output = model(batch)",
                "    loss = criterion(output, target)",
                "    loss.backward()",
                "    optimizer.step()",
            ]
        )
        sketch = self.sketch(code, logic_selection="ranked")

        self.assertIn("loss.backward()", sketch)
        self.assertIn("optimizer.step()", sketch)
        self.assertNotIn("value_19 = 19", sketch)

    def test_ranked_sketch_extracts_significant_assignments_and_headers(self):
        code = """\
def train(dataset, model):
    model = Sequential([Dense(128), Dropout(0.5), Dense(10)])
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(num_epochs):
        for batch in train_loader:
            with torch.no_grad():
                validate(batch)
"""
        sketch = self.sketch(code, logic_selection="ranked")

        self.assertIn("model = Sequential(", sketch)
        self.assertIn("optimizer = Adam(", sketch)
        self.assertIn("criterion = CrossEntropyLoss()", sketch)
        self.assertIn("train_loader = DataLoader(", sketch)
        self.assertIn("for epoch in range(num_epochs):", sketch)
        self.assertIn("for batch in train_loader:", sketch)
        self.assertIn("with torch.no_grad():", sketch)
        self.assertEqual(sketch.count("optimizer = Adam("), 1)
        self.assertNotIn("# calls:", sketch)

    def test_line_scoring_rewards_domain_dense_statements(self):
        self.assertGreater(
            score_line_domain_signal("optimizer.step()"),
            score_line_domain_signal("x = 5"),
        )
        self.assertGreater(
            score_line_domain_signal(
                "app = FastAPI(debug=True)"
            ),
            score_line_domain_signal("return result"),
        )

    def test_invalid_logic_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            self.sketch("run()", logic_selection="unknown")

    def test_comments_and_non_revealing_strings_are_removed(self):
        code = """\
# natural language comment
def greet(name):
    message = "hello world"  # inline explanation
    html = "<div class='user'>"
    return render(html, name)
"""
        sketch = self.sketch(code)

        self.assertNotIn("natural language comment", sketch)
        self.assertNotIn("inline explanation", sketch)
        self.assertNotIn('"hello world"', sketch.split("DOMAIN_LITERALS")[0])
        self.assertIn("<div class='user'>", sketch)

    def test_budget_is_enforced(self):
        code = "\n".join(
            f"import package_{index}" for index in range(200)
        )
        sketch = self.sketch(code, max_chars=512)
        self.assertLessEqual(len(sketch), 512)

    def test_default_budget_targets_short_embedding_input(self):
        code = "\n".join(
            [f"import package_{index}" for index in range(100)]
            + [f"value_{index} = call_{index}()" for index in range(100)]
        )
        sketch = self.sketch(code)
        self.assertLessEqual(len(sketch), 950)


if __name__ == "__main__":
    unittest.main()
