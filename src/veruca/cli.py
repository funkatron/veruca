"""Command line interface for Veruca."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict

from .sources.obsidian.query import ObsidianVault


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query your Obsidian vault using local LLMs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query your Obsidian vault")
    query_parser.add_argument(
        "query",
        type=str,
        help="The query to search for in your vault",
    )
    query_parser.add_argument(
        "--vault-path",
        type=Path,
        default=Path.home() / "Obsidian",
        help="Path to your Obsidian vault (default: ~/Obsidian)",
    )
    query_parser.add_argument(
        "--filter",
        type=str,
        action="append",
        help="Filter results by metadata (e.g., 'tags=python,status=active' or 'priority:gt=3')",
    )

    # Index command
    index_parser = subparsers.add_parser("index", help="Index your Obsidian vault")
    index_parser.add_argument(
        "--vault-path",
        type=Path,
        default=Path.home() / "Obsidian",
        help="Path to your Obsidian vault (default: ~/Obsidian)",
    )
    index_parser.add_argument(
        "--model",
        type=str,
        default="nomic-embed-text",
        help="Ollama model to use for embeddings (default: nomic-embed-text)",
    )

    # Ollama command
    ollama_parser = subparsers.add_parser("ollama", help="Manage Ollama server")
    ollama_parser.add_argument("action", choices=["status"], help="Action to perform")

    return parser.parse_args()


def check_ollama_status():
    """Check if Ollama server is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def parse_filter(filter_str: str) -> Dict[str, str]:
    """Parse a filter string into a key-value pair.

    Examples:
        status=active -> {"status": "active"}
        tags:in=python,programming -> {"tags:in": "python,programming"}
        priority:gt=3 -> {"priority:gt": "3"}
    """
    try:
        key, value = filter_str.split("=", 1)  # Split on first = only
        return {key.strip(): value.strip()}
    except ValueError:
        raise ValueError(f"Invalid filter format: {filter_str}. Expected format: field=value or field:operator=value")


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    if args.command == "ollama":
        if args.action == "status":
            if check_ollama_status():
                print("Ollama server is running")
            else:
                print("Ollama server is not running")
                sys.exit(1)
        else:
            print("Invalid Ollama command")
            sys.exit(1)

    # For query and index commands, ensure Ollama is running
    if not check_ollama_status():
        print("Error: Ollama server is not running")
        print("Please start it with: veruca ollama status")
        sys.exit(1)

    if args.command == "query":
        filters = {}
        if args.filter:
            for filter_part in args.filter:
                filters.update(parse_filter(filter_part))

        vault = ObsidianVault(args.vault_path)
        results = vault.query(args.query, filters=filters)
        print(results)

    elif args.command == "index":
        vault = ObsidianVault(args.vault_path, model=args.model)
        vault.index_documents()
        print("Vault indexed successfully")


if __name__ == "__main__":
    main()