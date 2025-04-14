"""Veruca - A collection of tools for working with local LLMs."""

__version__ = "0.2.0"

from .core.base import DataSource
from .sources.obsidian.query import ObsidianVault

__all__ = ["DataSource", "ObsidianVault"]