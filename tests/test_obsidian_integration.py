"""
Tests for Obsidian integration features including frontmatter parsing,
tag extraction, and file loading.
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

# ============== Pytest Tests ==============
import pytest
from pathlib import Path
from veruca.obsidian import parse_frontmatter, extract_tags, load_markdown_files

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

def test_parse_frontmatter(sample_content):
    """
    Test parsing Obsidian frontmatter.
    In unittest, this would be a method in a TestObsidian class with self.assert* methods.
    """
    frontmatter, content = parse_frontmatter(sample_content)

    assert frontmatter["title"] == "Test Note"
    assert frontmatter["tags"] == ["test", "example"]
    assert frontmatter["date"] == "2024-03-20"
    assert "# Test Note" in content

def test_extract_tags():
    """
    Test tag extraction from Obsidian content.
    Pytest's assert statements are more readable than unittest's self.assert* methods.
    """
    content = """
    Here are some #tags: #project #ideas #2024
    """
    tags = extract_tags(content)
    assert "project" in tags
    assert "ideas" in tags
    assert "2024" in tags
    assert "tags" in tags  # "tags" is a valid tag
    assert len(tags) == 4  # Updated to expect 4 tags

def test_error_handling():
    """
    Test error handling when loading non-existent paths.
    Pytest's pytest.raises is more concise than unittest's self.assertRaises.
    """
    with pytest.raises(SystemExit):
        load_markdown_files("/nonexistent/path")

@pytest.mark.parametrize("input_text,expected_tags", [
    ("#tag1 #tag2", ["tag1", "tag2"]),
    ("No tags here", []),
    ("#tag1 text #tag2", ["tag1", "tag2"])
])
def test_parametrized(input_text, expected_tags):
    """
    Example of parameterized tests.
    Pytest's @pytest.mark.parametrize is much cleaner than unittest's subTest context.
    """
    assert set(extract_tags(input_text)) == set(expected_tags)

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