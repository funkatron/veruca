"""Obsidian vault data source for Veruca."""

from .parser import (
    parse_frontmatter,
    extract_tags,
    process_obsidian_links,
    process_callouts,
    TAG_PATTERN,
)
from .query import ObsidianVault

__all__ = [
    "ObsidianVault",
    "parse_frontmatter",
    "extract_tags",
    "process_obsidian_links",
    "process_callouts",
    "TAG_PATTERN",
]