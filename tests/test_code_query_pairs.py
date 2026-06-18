import tempfile
import unittest
from pathlib import Path

from build_code_query_pairs import (
    FunctionRecord,
    build_pairs,
    extract_python_functions,
)


class CodeQueryPairTests(unittest.TestCase):
    def test_extracts_only_public_docstring_functions(self):
        source = '''
def documented(value):
    """Return a normalized value."""
    return value.strip()

def undocumented(value):
    return value

def _private(value):
    """Internal helper."""
    return value
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample.py"
            path.write_text(source, encoding="utf-8")
            records = extract_python_functions(
                path,
                source_name="test",
                repository="owner/repo",
                source_url="https://example.test",
                revision="abc",
                license_name="MIT",
                root=root,
            )
        self.assertEqual([record.qualified_name for record in records], ["documented"])

    def test_pair_generation_keeps_function_family_in_one_split(self):
        function = FunctionRecord(
            function_id="abc123",
            source="test",
            repository="owner/repo",
            source_url="https://example.test",
            revision="abc",
            license="MIT",
            file_path="sample.py",
            qualified_name="normalize_rows",
            docstring="Normalize rows before export.",
            code="def normalize_rows(rows):\n    return rows\n",
            domain="data_processing",
        )
        pairs = build_pairs([function], 8)
        self.assertEqual(len(pairs), 8)
        self.assertEqual(len({pair.split for pair in pairs}), 1)
        self.assertEqual({pair.query_domain for pair in pairs}, {"data_processing"})
        self.assertGreaterEqual(len({pair.query_operation for pair in pairs}), 5)
        self.assertGreaterEqual(len({pair.query_style for pair in pairs}), 5)


if __name__ == "__main__":
    unittest.main()
