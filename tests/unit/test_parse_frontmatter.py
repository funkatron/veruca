"""Unit tests for frontmatter parsing functionality."""
import pytest
from veruca.sources.obsidian.parser import parse_frontmatter

def test_basic_frontmatter():
    """Test parsing basic YAML frontmatter with simple key-value pairs."""
    content = """---
tags: python, programming
status: active
---
# Content
"""
    expected_metadata = {
        "tags": "python, programming",
        "status": "active"
    }
    metadata, content_without_frontmatter = parse_frontmatter(content)
    assert metadata == expected_metadata
    assert content_without_frontmatter.strip() == "# Content"

def test_empty_frontmatter():
    """Test handling of empty frontmatter section."""
    content = """---
---
# Content
"""
    metadata, content_without_frontmatter = parse_frontmatter(content)
    assert metadata == {}
    assert content_without_frontmatter.strip() == "# Content"

def test_no_frontmatter():
    """Test handling of content without frontmatter."""
    content = "# Content without frontmatter"
    metadata, content_without_frontmatter = parse_frontmatter(content)
    assert metadata == {}
    assert content_without_frontmatter == content

def test_complex_frontmatter():
    """Test parsing frontmatter with nested structures."""
    content = """---
tags:
  - python
  - programming
metadata:
  status: active
  priority: high
---
# Content
"""
    expected_metadata = {
        "tags": ["python", "programming"],
        "metadata": {
            "status": "active",
            "priority": "high"
        }
    }
    metadata, content_without_frontmatter = parse_frontmatter(content)
    assert metadata == expected_metadata
    assert content_without_frontmatter.strip() == "# Content"

def test_invalid_yaml():
    """Test handling of invalid YAML in frontmatter."""
    content = """---
invalid: : yaml : :
---
# Content
"""
    with pytest.raises(ValueError, match="Invalid YAML frontmatter"):
        parse_frontmatter(content)

def test_missing_closing_delimiter():
    """Test handling of frontmatter with missing closing delimiter."""
    content = """---
tags: python
# Content
"""
    with pytest.raises(ValueError, match="Missing closing frontmatter delimiter"):
        parse_frontmatter(content)