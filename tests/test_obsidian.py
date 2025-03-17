"""
Tests for Obsidian vault processing and querying functionality.

This test suite verifies the core functionality of the Obsidian vault processor,
including:
- YAML frontmatter parsing
- Tag extraction
- Link processing
- Callout processing
- File loading and indexing
- Querying with tag filtering
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from obsidian import (
    parse_frontmatter,
    extract_tags,
    process_obsidian_links,
    process_callouts,
    load_markdown_files,
    index_vault,
    query_vault
)

# Define test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_VAULT = TEST_DATA_DIR / "test_vault"

@pytest.fixture(autouse=True)
def temp_chroma_dir():
    """Create a temporary directory for the Chroma database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["CHROMA_DB_PATH"] = temp_dir
        yield temp_dir

@pytest.fixture
def sample_markdown_content():
    return """---
title: Test Note
tags: [test, example]
date: 2024-03-20
---

# Test Note

This is a test note with various Obsidian features.

## Tags
Here are some #tags: #project #ideas #2024

## Links
Here are some [[internal links]] and [[Project Ideas|My Ideas]]

## Callouts
> [!NOTE] This is a note
> [!WARNING] This is a warning
"""

@pytest.fixture
def test_vault(tmp_path):
    """Create a temporary test vault with sample files."""
    vault_dir = tmp_path / "test_vault"
    vault_dir.mkdir()

    # Create a test file
    test_file = vault_dir / "test.md"
    test_file.write_text("""---
title: Test Note
tags: [test, example]
date: 2024-03-20
---

# Test Note

This is a test note with various Obsidian features.

## Tags
Here are some #tags: #project #ideas #2024

## Links
Here are some [[internal links]] and [[Project Ideas|My Ideas]]

## Callouts
> [!NOTE] This is a note
> [!WARNING] This is a warning
""")

    return vault_dir

def test_parse_frontmatter(sample_markdown_content):
    """Test frontmatter parsing."""
    frontmatter, content = parse_frontmatter(sample_markdown_content)

    assert frontmatter["title"] == "Test Note"
    assert frontmatter["tags"] == ["test", "example"]
    assert frontmatter["date"] == "2024-03-20"
    assert "# Test Note" in content

def test_extract_tags(sample_markdown_content):
    """Test tag extraction."""
    tags = extract_tags(sample_markdown_content)

    assert "project" in tags
    assert "ideas" in tags
    assert "2024" in tags
    assert "tags" in tags  # "tags" is a valid tag
    assert len(tags) == 4  # Updated to expect 4 tags

def test_process_obsidian_links():
    """Test internal link processing."""
    content = "[[Project Ideas]] and [[Project Ideas|My Ideas]]"
    processed = process_obsidian_links(content, "")

    assert "Project Ideas" in processed
    assert "My Ideas" in processed

def test_process_callouts():
    """Test callout processing."""
    content = "> [!NOTE] This is a note\n> [!WARNING] This is a warning"
    processed = process_callouts(content)

    assert "[NOTE] This is a note" in processed
    assert "[WARNING] This is a warning" in processed

def test_load_markdown_files(test_vault):
    """Test loading markdown files from vault."""
    files = load_markdown_files(str(test_vault))

    assert len(files) == 1
    filename, metadata, content = files[0]

    assert filename == "test.md"
    assert metadata["title"] == "Test Note"
    assert "project" in metadata["tags"]
    assert "# Test Note" in content

def test_index_and_query(test_vault):
    """Test indexing and querying the vault."""
    # Create a mock Chroma instance that just stores and returns documents
    mock_chroma = MagicMock()
    mock_chroma.from_documents.return_value = mock_chroma
    mock_chroma.as_retriever.return_value = MagicMock()

    # Mock the RetrievalQA chain
    mock_qa = MagicMock()
    mock_qa.invoke.return_value = {"result": "Mock response for query containing test"}

    # Patch both Chroma and RetrievalQA
    with patch('obsidian.Chroma', mock_chroma), \
         patch('obsidian.RetrievalQA.from_chain_type', return_value=mock_qa):
        # Test indexing
        index_vault(str(test_vault))
        mock_chroma.from_documents.assert_called_once()
        mock_chroma.persist.assert_called_once()

        # Test querying with filters
        query = "What are the tags in the test note?"
        filters = {"tags": "test"}
        result = query_vault(query, filters)
        assert isinstance(result, str)
        assert "Mock response" in result
        assert "test" in result

def test_error_handling(capsys):
    """Test error handling for invalid inputs."""
    # Test non-existent vault
    with pytest.raises(SystemExit) as excinfo:
        load_markdown_files("/nonexistent/path")
    assert excinfo.value.code == 1
    assert "Error: Vault path '/nonexistent/path' does not exist." in capsys.readouterr().out

    # Test invalid frontmatter with unmatched brackets
    with pytest.raises(SystemExit) as excinfo:
        parse_frontmatter("""---
title: Test Note
tags: [test, example
date: 2024-03-20
---""")
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    print("Captured output:", captured.out)  # Debug print
    assert "Error: Invalid YAML frontmatter:" in captured.out