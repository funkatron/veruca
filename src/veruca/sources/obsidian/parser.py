"""Parsing functionality for Obsidian files."""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import yaml
from markdown import Markdown

# Tag pattern for Obsidian-style tags
TAG_PATTERN = r'(?:^|\s)#([a-zA-Z0-9/_-]+)(?=[^\w/]|$)'


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from a markdown file.

    Args:
        content: The markdown content

    Returns:
        A tuple of (frontmatter dict, remaining content)
    """
    if not content.startswith('---\n'):
        return {}, content

    try:
        # Find the end of the frontmatter
        _, rest = content.split('---\n', 1)
        if '\n---\n' not in rest:
            return {}, content
        frontmatter_str, content = rest.split('\n---\n', 1)

        # Parse the frontmatter
        frontmatter = yaml.safe_load(frontmatter_str)
        if not isinstance(frontmatter, dict):
            frontmatter = {}

        return frontmatter, content.strip()

    except (yaml.YAMLError, ValueError) as e:
        print(f"Error: Invalid YAML frontmatter: {str(e)}", file=sys.stderr)
        sys.exit(1)


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