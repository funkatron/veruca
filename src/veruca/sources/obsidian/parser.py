"""Parsing functionality for Obsidian files."""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import yaml
from langchain_core.documents import Document
from markdown import Markdown

# Tag pattern for Obsidian-style tags
TAG_PATTERN = r'(?:^|\s)#([a-zA-Z0-9/_-]+)(?=[^\w/]|$)'


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: The markdown content to parse

    Returns:
        Tuple of (metadata dict, content without frontmatter)

    Raises:
        ValueError: If frontmatter is invalid or missing closing delimiter
    """
    # Check for frontmatter
    if not content.startswith("---"):
        return {}, content

    # Split content into frontmatter and body
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Missing closing frontmatter delimiter (---)")

    frontmatter_str = parts[1].strip()
    content_without_frontmatter = parts[2].strip()

    # Parse YAML
    try:
        frontmatter = yaml.safe_load(frontmatter_str)
        if frontmatter is None:
            frontmatter = {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML frontmatter: {str(e)}")

    return frontmatter, content_without_frontmatter


def extract_tags(content: str) -> Set[str]:
    """Extract Obsidian-style tags from content.

    Args:
        content: The markdown content

    Returns:
        A set of tags
    """
    tags = set()

    # Extract tags from content
    for line in content.split('\n'):
        # Skip code blocks and headings
        if line.strip().startswith('```') or line.strip().startswith('#'):
            continue

        # Find all tags in the line
        matches = re.finditer(TAG_PATTERN, line)
        tags.update(match.group(1) for match in matches)

    return tags


def process_obsidian_links(content: str, base_path: str) -> str:
    """Process Obsidian-style internal links.

    Args:
        content: The markdown content
        base_path: The base path for resolving links

    Returns:
        The processed content
    """
    # Convert [[Link]] to Link
    content = re.sub(r'\[\[([^\]|]+)\]\]', r'\1', content)

    # Convert [[Link|Alias]] to Alias
    content = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', content)

    return content


def process_callouts(content: str) -> str:
    """Process Obsidian-style callouts.

    Args:
        content: The markdown content

    Returns:
        The processed content
    """
    # Convert > [!NOTE] to [NOTE]
    content = re.sub(r'>\s*\[!([^\]]+)\]', r'[\1]', content)
    return content


def format_document(doc: Document) -> Dict[str, str]:
    """Format a document for the LLM chain.

    Args:
        doc: The document to format

    Returns:
        Dict with formatted page_content and metadata
    """
    # Format metadata as key: value pairs
    formatted_metadata = []
    for key, value in sorted(doc.metadata.items()):
        if isinstance(value, (list, dict)):
            formatted_metadata.append(f"{key}: {value}")
        else:
            formatted_metadata.append(f"{key}: {value}")

    return {
        "page_content": doc.page_content,
        "metadata": "\n".join(formatted_metadata) if formatted_metadata else ""
    }