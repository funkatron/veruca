"""
This file demonstrates the same tests written in both pytest and unittest styles.
The tests are identical in functionality but show the different syntax and approaches.
"""

# ============== Test Data ==============
SAMPLE_MARKDOWN = """---
title: Test Note
tags: [test, example]
date: 2024-03-20
---

# Test Note
Here are some #tags: #project #ideas #2024
"""

# ============== Pytest Style ==============
import pytest
from pathlib import Path

# Pytest fixtures are very clean and reusable
@pytest.fixture
def sample_content():
    return SAMPLE_MARKDOWN

@pytest.fixture
def test_vault(tmp_path):
    vault_dir = tmp_path / "test_vault"
    vault_dir.mkdir()
    test_file = vault_dir / "test.md"
    test_file.write_text(SAMPLE_MARKDOWN)
    return vault_dir

# Pytest tests are simple functions with clear assertions
def test_parse_frontmatter_pytest(sample_content):
    from obsidian import parse_frontmatter
    frontmatter, content = parse_frontmatter(sample_content)

    assert frontmatter["title"] == "Test Note"
    assert frontmatter["tags"] == ["test", "example"]
    assert frontmatter["date"] == "2024-03-20"
    assert "# Test Note" in content

def test_extract_tags_pytest(sample_content):
    from obsidian import extract_tags
    tags = extract_tags(sample_content)

    assert "project" in tags
    assert "ideas" in tags
    assert "2024" in tags
    assert len(tags) == 3

# Pytest makes it easy to test exceptions
def test_error_handling_pytest():
    from obsidian import load_markdown_files
    with pytest.raises(SystemExit):
        load_markdown_files("/nonexistent/path")

# Pytest makes it easy to skip tests
@pytest.mark.skipif(True, reason="Example of skipping")
def test_skipped_pytest():
    assert False  # This won't run

# ============== Unittest Style ==============
import unittest
from pathlib import Path
import tempfile
import shutil

class TestObsidian(unittest.TestCase):
    # Unittest uses setUp for fixtures
    def setUp(self):
        self.sample_content = SAMPLE_MARKDOWN
        self.temp_dir = tempfile.mkdtemp()
        self.vault_dir = Path(self.temp_dir) / "test_vault"
        self.vault_dir.mkdir()
        test_file = self.vault_dir / "test.md"
        test_file.write_text(self.sample_content)

    def tearDown(self):
        # Clean up after tests
        shutil.rmtree(self.temp_dir)

    def test_parse_frontmatter_unittest(self):
        from obsidian import parse_frontmatter
        frontmatter, content = parse_frontmatter(self.sample_content)

        self.assertEqual(frontmatter["title"], "Test Note")
        self.assertEqual(frontmatter["tags"], ["test", "example"])
        self.assertEqual(frontmatter["date"], "2024-03-20")
        self.assertIn("# Test Note", content)

    def test_extract_tags_unittest(self):
        from obsidian import extract_tags
        tags = extract_tags(self.sample_content)

        self.assertIn("project", tags)
        self.assertIn("ideas", tags)
        self.assertIn("2024", tags)
        self.assertEqual(len(tags), 3)

    def test_error_handling_unittest(self):
        from obsidian import load_markdown_files
        with self.assertRaises(SystemExit):
            load_markdown_files("/nonexistent/path")

    @unittest.skip("Example of skipping")
    def test_skipped_unittest(self):
        self.fail("This won't run")

# ============== Key Differences ==============
"""
1. Test Organization:
   - Pytest: Simple functions with clear names
   - Unittest: Classes with test methods

2. Fixtures:
   - Pytest: @pytest.fixture decorator, very flexible
   - Unittest: setUp/tearDown methods, more rigid

3. Assertions:
   - Pytest: assert statements (more readable)
   - Unittest: self.assert* methods (more verbose)

4. Test Discovery:
   - Pytest: Any function starting with 'test_'
   - Unittest: Methods in Test* classes

5. Temporary Files:
   - Pytest: Built-in tmp_path fixture
   - Unittest: Manual tempfile handling

6. Skipping Tests:
   - Pytest: @pytest.mark.skipif (more flexible)
   - Unittest: @unittest.skip (less flexible)

7. Error Messages:
   - Pytest: More detailed, shows exact values
   - Unittest: Basic error messages

8. Parameterized Tests:
   - Pytest: @pytest.mark.parametrize (very clean)
   - Unittest: subTest context (more verbose)
"""

# Example of parameterized tests in both styles
@pytest.mark.parametrize("input_text,expected_tags", [
    ("#tag1 #tag2", ["tag1", "tag2"]),
    ("No tags here", []),
    ("#tag1 text #tag2", ["tag1", "tag2"])
])
def test_parametrized_pytest(input_text, expected_tags):
    from obsidian import extract_tags
    assert set(extract_tags(input_text)) == set(expected_tags)

class TestParametrized(unittest.TestCase):
    def test_parametrized_unittest(self):
        test_cases = [
            ("#tag1 #tag2", ["tag1", "tag2"]),
            ("No tags here", []),
            ("#tag1 text #tag2", ["tag1", "tag2"])
        ]

        for input_text, expected_tags in test_cases:
            with self.subTest(input_text=input_text):
                from obsidian import extract_tags
                self.assertEqual(set(extract_tags(input_text)), set(expected_tags))