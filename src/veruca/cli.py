"""Command line interface for Veruca."""

import argparse
from pathlib import Path

from .sources.obsidian.query import ObsidianVault


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query your Obsidian vault using local LLMs."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The query to search for in your vault",
    )
    parser.add_argument(
        "--vault-path",
        type=Path,
        default=Path.home() / "Obsidian",
        help="Path to your Obsidian vault (default: ~/Obsidian)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Filter results by metadata (e.g., 'tags=python,status=active')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama2",
        help="Ollama model to use (default: llama2)",
    )
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Parse filter if provided
    filters = {}
    if args.filter:
        for filter_part in args.filter.split(","):
            key, value = filter_part.split("=")
            filters[key.strip()] = value.strip()

    # Create and query the vault
    vault = ObsidianVault(args.vault_path, model=args.model)
    result = vault.query(args.query, filters=filters)
    print(result)


if __name__ == "__main__":
    main()