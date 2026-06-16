import unittest

from code_signals.parser import count_errors, parse
from code_signals.semantic import extract_semantic
from code_signals.structural import extract_structural


class StructuralSignalTests(unittest.TestCase):
    def extract(self, code):
        root = parse(code).root_node
        return extract_structural(root, code)

    def test_flat_dispatch(self):
        code = """\
def handle(cmd):
    if cmd == "a":
        return 1
    elif cmd == "b":
        return 2
    else:
        return 3
"""
        result = self.extract(code)
        self.assertEqual(result["num_branches"], 3)
        self.assertEqual(result["cognitive_complexity"], 3)
        self.assertEqual(result["max_nesting_depth"], 2)
        self.assertFalse(result["has_nested_loops"])

    def test_recursion_nested_loops_and_boolean_complexity(self):
        code = """\
def visit(graph, node):
    while node:
        for child in graph[node]:
            if child and not seen:
                visit(graph, child)
"""
        result = self.extract(code)
        self.assertTrue(result["has_recursion"])
        self.assertTrue(result["has_nested_loops"])
        self.assertEqual(result["num_loops"], 2)
        self.assertEqual(result["max_bool_expr_complexity"], 2)
        self.assertEqual(result["cognitive_complexity"], 8)

    def test_recursion_adds_one_regardless_of_recursive_call_count(self):
        recursive = self.extract(
            """\
def countdown(n):
    if n <= 0:
        return 0
    countdown(n - 1)
    return countdown(n - 2)
"""
        )
        non_recursive = self.extract(
            """\
def countdown(n):
    if n <= 0:
        return 0
    helper(n - 1)
    return helper(n - 2)
"""
        )
        self.assertEqual(
            recursive["cognitive_complexity"],
            non_recursive["cognitive_complexity"] + 1,
        )

    def test_boolean_operator_switches_affect_cognitive_complexity(self):
        code = """\
if a and b or c and d:
    run()
"""
        result = self.extract(code)
        self.assertEqual(result["cognitive_complexity"], 4)
        self.assertEqual(result["max_bool_expr_complexity"], 3)

    def test_docstrings_are_excluded_from_sloc(self):
        code = '''"""module docs
continued
"""

def answer():
    """function docs"""
    # comment
    return 42
'''
        result = self.extract(code)
        self.assertEqual(result["sloc"], 2)
        self.assertEqual(result["max_function_length"], 3)

    def test_counts_top_level_and_nested_classes(self):
        code = """\
class Outer:
    class Inner:
        pass

def build():
    class Local:
        pass
    return Local()
"""
        result = self.extract(code)
        self.assertEqual(result["num_classes"], 3)

    def test_broken_code_still_extracts(self):
        code = """\
def process(items):
    for item in items:
        if item:
            print(item)

values = [1, 2,
"""
        root = parse(code).root_node
        result = extract_structural(root, code)
        self.assertGreaterEqual(count_errors(root), 1)
        self.assertEqual(result["num_function_defs"], 1)
        self.assertGreaterEqual(result["num_loops"], 1)


class SemanticSignalTests(unittest.TestCase):
    def test_semantic_signals(self):
        code = """\
@decorate
def f(items):
    global state
    x = 1
    x = 2
    for i in items:
        print(i)
    return getattr(items, "value")
"""
        root = parse(code).root_node
        result = extract_semantic(root, code)
        self.assertEqual(result["variable_reassignment_count"], 1)
        self.assertTrue(result["has_global_nonlocal"])
        self.assertTrue(result["has_dynamic_features"])
        self.assertTrue(result["has_decorators"])
        self.assertEqual(result["single_char_var_ratio"], 1.0)
        self.assertEqual(result["call_diversity"], 2)


if __name__ == "__main__":
    unittest.main()
