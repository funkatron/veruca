"""Query functionality for Obsidian vaults."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from langchain_core.documents import Document

from ...core.base import DataSource
from ...core.embeddings import create_embeddings, create_vector_store, create_qa_chain
from .parser import (
    parse_frontmatter,
    extract_tags,
    process_obsidian_links,
    process_callouts,
)


class ObsidianVault(DataSource):
    """A data source for Obsidian vaults."""

    def __init__(self, vault_path: str, model: str = "nomic-embed-text"):
        """Initialize the Obsidian vault data source.

        Args:
            vault_path: Path to the Obsidian vault
            model: The Ollama model to use for embeddings (default: nomic-embed-text)
        """
        self.vault_path = Path(vault_path)
        self.model = model
        self.vector_store = None

        if not self.vault_path.exists():
            print(f"Error: Vault path '{vault_path}' does not exist.", file=sys.stderr)
            sys.exit(1)

    def load_documents(self) -> List[Tuple[str, Dict[str, Any], str]]:
        """Load markdown files from the vault.

        Returns:
            List of tuples containing (filename, metadata, content)
        """
        documents = []

        # Find all markdown files
        for file_path in self.vault_path.rglob("*.md"):
            try:
                content = file_path.read_text()

                # Parse frontmatter and process content
                frontmatter, content = parse_frontmatter(content)

                # Extract tags from both frontmatter and content
                tags = set(frontmatter.get("tags", []))
                tags.update(extract_tags(content))
                # Convert tags to comma-separated string for ChromaDB
                frontmatter["tags"] = ",".join(tags)

                # Process Obsidian-specific syntax
                content = process_obsidian_links(content, str(self.vault_path))
                content = process_callouts(content)

                documents.append((
                    file_path.name,
                    frontmatter,
                    content
                ))

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}", file=sys.stderr)
                continue

        return documents

    def index_documents(self) -> None:
        """Index the vault documents for querying."""
        documents = self.load_documents()

        # Convert to LangChain documents
        langchain_docs = []
        for filename, metadata, content in documents:
            doc = Document(
                page_content=content,
                metadata={
                    "source": filename,
                    **metadata
                }
            )
            langchain_docs.append(doc)

        # Create embeddings and vector store
        embeddings = create_embeddings(self.model)
        self.vector_store = create_vector_store(
            documents=langchain_docs,
            embeddings=embeddings,
            persist_dir=os.environ.get("CHROMA_DB_PATH")
        )

    def query(self, query: str, filters: Optional[Dict[str, str]] = None) -> str:
        """Query the indexed vault.

        Args:
            query: The query string
            filters: Optional filters to apply

        Returns:
            The query result as a string
        """
        if not self.vector_store:
            self.index_documents()

        # Create and run the QA chain
        chain = create_qa_chain(
            vector_store=self.vector_store,
            model="mistral",  # Use mistral for query responses
            filters=filters
        )

        return chain.invoke(query)