import unittest

from code_signals.data_flow import extract_data_flow
from code_signals.parser import parse


def data_flow(code: str) -> dict:
    return extract_data_flow(parse(code).root_node)


class DataFlowSignalTests(unittest.TestCase):
    def test_local_def_use_edges_and_dependency_depth(self):
        result = data_flow(
            """\
def transform(source):
    first = source
    second = first
    combined = second + first
    return combined
"""
        )
        self.assertEqual(result["local_def_use_edge_count"], 5)
        self.assertEqual(result["max_dataflow_chain_depth"], 4)
        self.assertEqual(result["alias_assignment_count"], 2)
        self.assertEqual(result["unresolved_flow_ratio"], 0.0)

    def test_branch_merge_and_cross_scope_reads(self):
        result = data_flow(
            """\
seed = 3

def outer(flag, source):
    if flag:
        value = source
    else:
        value = seed

    def inner(item):
        return item + value + seed

    return inner
"""
        )
        self.assertEqual(result["branch_merge_count"], 1)
        self.assertEqual(result["cross_scope_flow_count"], 3)

    def test_mutation_targets_are_not_treated_as_simple_definitions(self):
        result = data_flow(
            """\
def update(obj, rows, key, source):
    alias = source
    obj.value = alias
    rows[key] += alias
    return callback(alias)
"""
        )
        self.assertEqual(result["alias_assignment_count"], 1)
        self.assertEqual(result["attribute_mutation_count"], 1)
        self.assertEqual(result["subscript_mutation_count"], 1)
        self.assertEqual(result["unresolved_call_count"], 1)
        self.assertGreater(result["unresolved_flow_ratio"], 0.0)

    def test_local_calls_resolve_but_callbacks_remain_unresolved(self):
        result = data_flow(
            """\
def normalize(value):
    return value

def run(value, callback):
    cleaned = normalize(value)
    return callback(cleaned)
"""
        )
        self.assertEqual(result["unresolved_call_count"], 1)
        self.assertEqual(result["local_def_use_edge_count"], 3)


if __name__ == "__main__":
    unittest.main()
