import pytest
import tempfile
import shutil
from pathlib import Path
from src.veruca.sources.obsidian import ObsidianVault

@pytest.fixture
def test_vault():
    """Create a temporary test vault with a single document containing tags."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    vault_path = Path(temp_dir)

    # Create a test document with known tags
    doc_path = vault_path / "test.md"
    doc_path.write_text("""---
tags: test-tag,python,documentation
status: active
---

# Test Document

This is a test document with known tags in the frontmatter.
""")

    yield vault_path

    # Cleanup
    shutil.rmtree(temp_dir)

def test_tag_retrieval(test_vault):
    """Test that we can correctly store and retrieve tags from a document."""
    vault = ObsidianVault(str(test_vault))

    # First verify the document is loaded correctly
    docs = vault.load_documents()
    assert len(docs) == 1, "Expected exactly one document"
    filename, metadata, content = docs[0]
    print("\nDEBUG - Document metadata:", metadata)
    assert "tags" in metadata, "Tags not found in metadata"

    # Verify all expected tags are present (order doesn't matter)
    stored_tags = [t.strip() for t in metadata["tags"].split(",")]
    expected_tags = ["test-tag", "python", "documentation"]
    for tag in expected_tags:
        assert tag in stored_tags, f"Expected tag '{tag}' not found in stored tags"

    # Query specifically for tags
    result = vault.query("Look at ONLY the metadata section. What tags are present in the tags field?")
    print("\nDEBUG - Query result:", result)

    # Verify the response contains all and only the expected tags
    result_text = result['result'] if isinstance(result, dict) else result
    result_tags = [t.strip() for t in result_text.split(",")]
    for tag in expected_tags:
        assert tag in result_tags, f"Expected tag '{tag}' not found in response"

    # Verify no extra/invented tags
    unexpected_tags = ["personal", "notes", "work", "project1", "important", "research"]
    for tag in unexpected_tags:
        assert tag not in result_tags, f"Found unexpected tag '{tag}' in response"

    # Test exact tag filtering
    filtered_result = vault.query(
        "Look at ONLY the metadata section. What tags are present in the tags field?",
        filters={"tags": "python"}
    )
    print("\nDEBUG - Filtered result:", filtered_result)
    filtered_text = filtered_result['result'] if isinstance(filtered_result, dict) else filtered_result
    filtered_tags = [t.strip() for t in filtered_text.split(",")]
    assert "python" in filtered_tags, "Tag filtering failed to return document with matching tag"