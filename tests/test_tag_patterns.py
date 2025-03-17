"""
Tests for Obsidian tag pattern matching.

This test suite verifies that our tag regex pattern correctly matches Obsidian tags
according to their rules and common usage patterns.
"""

import re
import pytest
from obsidian import TAG_PATTERN

def find_tags(text: str) -> list[str]:
    """Helper function to find all tags in a line of text."""
    return [m.group(1) for m in re.finditer(TAG_PATTERN, text)]

@pytest.mark.parametrize("test_input,expected", [
    # Basic tags
    ("#tag", ["tag"]),
    ("Hello #tag", ["tag"]),
    ("#tag world", ["tag"]),
    ("Hello #tag world", ["tag"]),

    # Multiple tags
    ("#tag1 #tag2", ["tag1", "tag2"]),
    ("Hello #tag1 #tag2 world", ["tag1", "tag2"]),

    # Tags with numbers
    ("#tag123", ["tag123"]),
    ("#123tag", ["123tag"]),

    # Tags with underscores and hyphens
    ("#tag_one", ["tag_one"]),
    ("#tag-one", ["tag-one"]),
    ("#tag_one-two", ["tag_one-two"]),

    # Nested tags
    ("#project/active", ["project/active"]),
    ("#programming/python/django", ["programming/python/django"]),

    # Tags with punctuation (should not include punctuation)
    ("#tag.", ["tag"]),
    ("#tag,", ["tag"]),
    ("#tag:", ["tag"]),
    ("#tag!", ["tag"]),
    ("#tag?", ["tag"]),
    ("#tag;", ["tag"]),

    # Tags in sentences
    ("This is a #tag in a sentence.", ["tag"]),
    ("Multiple #tag1 and #tag2 in sentence.", ["tag1", "tag2"]),
    ("Sentence with #tag/subtag!", ["tag/subtag"]),

    # Edge cases
    ("", []),
    ("#", []),
    ("##", []),
    ("###", []),
    ("# ", []),
    ("#/", []),
    ("#tag/", ["tag"]),
    ("/tag", []),
    ("tag#", []),

    # Invalid tags (should not match)
    ("# tag", []),  # Space after #
    ("#tag tag", ["tag"]),  # Space in tag
    ("#tag#tag", ["tag"]),  # Multiple # in tag
    ("##tag", []),  # Multiple # at start
    ("#tag##", ["tag"]),  # Multiple # at end
    ("#/tag", []),  # Starting with /
    ("#tag//subtag", ["tag"]),  # Multiple /

    # Tags in code-like contexts (should still match)
    ("var#tag", []),  # # not at start or after space
    ("print('#tag')", ["tag"]),  # In quotes
    ("function#tag()", []),  # Part of identifier

    # Tags with special characters (should not match)
    ("#tag@123", ["tag"]),  # @ is not word character
    ("#tag$123", ["tag"]),  # $ is not word character
    ("#tag*123", ["tag"]),  # * is not word character

    # Real-world examples
    ("Working on my #project/tasks for #work/2024 today!", ["project/tasks", "work/2024"]),
    ("Added #feature-request to my #coding/python project.", ["feature-request", "coding/python"]),
    ("Important #meeting_notes from #team/dev_ops.", ["meeting_notes", "team/dev_ops"]),
])
def test_tag_pattern(test_input, expected):
    """Test that the tag pattern correctly matches various tag formats."""
    assert find_tags(test_input) == expected

def test_tag_pattern_in_context():
    """Test tag pattern in more complex document contexts.
    Note: This test only verifies the pattern itself, not code block handling.
    Code block and heading handling is done by the extract_tags function."""
    text = '''# Heading (not a #tag)
## Subheading with #not-a-tag

This is a paragraph with #real-tag and #another/nested/tag.

Back to normal text with #valid-tag.

> A quote with #quoted-tag
'''
    expected = ["real-tag", "another/nested/tag", "valid-tag", "quoted-tag"]
    # Note: This test only checks the pattern, not the full extract_tags function
    # which handles code blocks and headings separately
    tags = []
    for line in text.split('\n'):
        if not line.strip().startswith('#'):  # Simple heading check
            tags.extend(find_tags(line))
    assert sorted(tags) == sorted(expected)

def test_extract_tags_with_code_blocks():
    """Test the full extract_tags function with code blocks."""
    text = '''# Heading (not a #tag)
## Subheading with #not-a-tag

This is a paragraph with #real-tag and #another/nested/tag.
Here's a code block:

```python
# This #not-a-tag is in a code block
print("#also-not-a-tag")
```

Back to normal text with #valid-tag.

> A quote with #quoted-tag
'''
    expected = ["real-tag", "another/nested/tag", "valid-tag", "quoted-tag"]
    from obsidian import extract_tags
    tags = extract_tags(text)
    assert sorted(tags) == sorted(expected)