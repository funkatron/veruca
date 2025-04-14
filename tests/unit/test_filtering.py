"""Unit tests for document filtering functionality."""
import pytest
from langchain_core.documents import Document
from veruca.sources.obsidian.query import apply_filters

def test_basic_filtering():
    """Test basic filter application to documents.

    TODO:
    - Test filtering by single field
    - Test filtering by multiple fields
    - Test filtering with no matches
    - Test filtering with all matches
    """
    docs = [
        Document(page_content="Doc1", metadata={"status": "active", "priority": "high"}),
        Document(page_content="Doc2", metadata={"status": "draft", "priority": "low"}),
        Document(page_content="Doc3", metadata={"status": "active", "priority": "medium"})
    ]

    # Test filtering by single field
    filtered = apply_filters(docs, {"status": "active"})
    assert len(filtered) == 2
    assert all(doc.metadata["status"] == "active" for doc in filtered)

    # Test filtering by multiple fields
    filtered = apply_filters(docs, {"status": "active", "priority": "high"})
    assert len(filtered) == 1
    assert filtered[0].metadata["status"] == "active"
    assert filtered[0].metadata["priority"] == "high"

    # Test filtering with no matches
    filtered = apply_filters(docs, {"status": "archived"})
    assert len(filtered) == 0

    # Test filtering with all matches
    filtered = apply_filters(docs, {"status": ["active", "draft"]})
    assert len(filtered) == 3

def test_tag_filtering():
    """Test tag-specific filtering logic."""
    docs = [
        Document(page_content="Doc1", metadata={"tags": ["python", "web"]}),
        Document(page_content="Doc2", metadata={"tags": ["java", "web"]}),
        Document(page_content="Doc3", metadata={"tags": "python,data"}),  # Test comma-separated string
        Document(page_content="Doc4", metadata={"tags": None})  # Test missing tags
    ]

    # Test filtering by single tag (should match both list and comma-separated string)
    filtered = apply_filters(docs, {"tags": "python"})
    assert len(filtered) == 2  # Should match Doc1 and Doc3
    assert any(doc.page_content == "Doc1" for doc in filtered)
    assert any(doc.page_content == "Doc3" for doc in filtered)

    # Test filtering by multiple tags (OR logic)
    filtered = apply_filters(docs, {"tags": ["python", "java"]})
    assert len(filtered) == 3  # Should match Doc1, Doc2, and Doc3

    # Test filtering with exact comma-separated string match
    filtered = apply_filters(docs, {"tags": "python,data"})
    assert len(filtered) == 1  # Should only match Doc3
    assert filtered[0].page_content == "Doc3"

    # Test filtering with non-existent tag
    filtered = apply_filters(docs, {"tags": "nonexistent"})
    assert len(filtered) == 0

def test_filter_validation():
    """Test filter validation and error handling.

    TODO:
    - Test invalid filter field names
    - Test invalid filter value types
    - Test empty filters
    - Test None filters
    """
    docs = [Document(page_content="Doc1", metadata={"status": "active"})]

    # Test invalid filter field names
    with pytest.raises(ValueError, match="Invalid filter field"):
        apply_filters(docs, {"nonexistent_field": "value"})

    # Test invalid filter value types
    with pytest.raises(ValueError, match="Invalid filter value type"):
        apply_filters(docs, {"status": {"complex": "value"}})

    # Test empty filters
    filtered = apply_filters(docs, {})
    assert len(filtered) == 1  # Should return all documents

    # Test None filters
    filtered = apply_filters(docs, None)
    assert len(filtered) == 1  # Should return all documents