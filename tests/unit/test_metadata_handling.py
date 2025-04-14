"""Unit tests for metadata handling in document chains."""
import pytest
from langchain_core.documents import Document
from veruca.sources.obsidian.parser import parse_frontmatter, format_document

def test_document_formatting():
    """Test document formatting for LLM chain.

    TODO:
    - Test basic metadata formatting
    - Test handling of missing metadata fields
    - Test handling of complex metadata structures
    - Test metadata serialization for LLM context
    """
    # Test basic metadata formatting
    doc = Document(
        page_content="Test content",
        metadata={"status": "active", "tags": "python,programming"}
    )
    formatted = format_document(doc)
    assert "status: active" in formatted["metadata"]
    assert "tags: python,programming" in formatted["metadata"]
    assert formatted["page_content"] == "Test content"

    # Test handling of missing metadata fields
    doc = Document(page_content="Test content", metadata={})
    formatted = format_document(doc)
    assert formatted["metadata"] == ""
    assert formatted["page_content"] == "Test content"

    # Test handling of complex metadata structures
    doc = Document(
        page_content="Test content",
        metadata={
            "status": "active",
            "tags": ["python", "programming"],
            "metadata": {"priority": "high"}
        }
    )
    formatted = format_document(doc)
    assert "status: active" in formatted["metadata"]
    assert "tags: ['python', 'programming']" in formatted["metadata"]
    assert "metadata: {'priority': 'high'}" in formatted["metadata"]

def test_metadata_preservation():
    """Test that metadata is preserved through the document chain.

    TODO:
    - Test metadata preservation in document loading
    - Test metadata preservation in vector store
    - Test metadata preservation in retrieval
    - Test metadata preservation in final LLM context
    """
    # Test metadata preservation in document loading
    content = """---
status: active
tags: python,programming
---
# Test content
"""
    metadata, content_without_frontmatter = parse_frontmatter(content)
    assert metadata["status"] == "active"
    assert metadata["tags"] == "python,programming"
    assert content_without_frontmatter.strip() == "# Test content"

    # Test metadata preservation in document creation
    doc = Document(
        page_content=content_without_frontmatter,
        metadata=metadata
    )
    assert doc.metadata["status"] == "active"
    assert doc.metadata["tags"] == "python,programming"
    assert doc.page_content.strip() == "# Test content"

def test_metadata_serialization():
    """Test metadata serialization for different types.

    TODO:
    - Test serialization of strings
    - Test serialization of lists
    - Test serialization of nested dictionaries
    - Test serialization of mixed types
    """
    # Test serialization of strings
    doc = Document(
        page_content="Test content",
        metadata={"status": "active"}
    )
    formatted = format_document(doc)
    assert "status: active" in formatted["metadata"]

    # Test serialization of lists
    doc = Document(
        page_content="Test content",
        metadata={"tags": ["python", "programming"]}
    )
    formatted = format_document(doc)
    assert "tags: ['python', 'programming']" in formatted["metadata"]

    # Test serialization of nested dictionaries
    doc = Document(
        page_content="Test content",
        metadata={"metadata": {"status": "active", "priority": "high"}}
    )
    formatted = format_document(doc)
    assert "metadata: {'status': 'active', 'priority': 'high'}" in formatted["metadata"]

    # Test serialization of mixed types
    doc = Document(
        page_content="Test content",
        metadata={
            "status": "active",
            "tags": ["python", "programming"],
            "metadata": {"priority": "high"}
        }
    )
    formatted = format_document(doc)
    assert "status: active" in formatted["metadata"]
    assert "tags: ['python', 'programming']" in formatted["metadata"]
    assert "metadata: {'priority': 'high'}" in formatted["metadata"]