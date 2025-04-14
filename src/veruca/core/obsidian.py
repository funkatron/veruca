"""Obsidian vault integration for Veruca."""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

from .base import DataSource
from .embeddings import create_embeddings, create_vector_store, create_qa_chain


# Suppress ResourceWarnings about unclosed socket connections
# These warnings occur because the Ollama clients use async HTTP connections
# that are managed by the client libraries. While Python's garbage collector
# is cautious about reporting these connections, they are properly managed
# and closed by the client libraries.
warnings.filterwarnings("ignore", category=ResourceWarning)


class ObsidianDataSource(DataSource):
    """DataSource implementation for Obsidian vaults."""

    def __init__(
        self,
        vault_path: Path,
        model: str = "llama2",
        persist_dir: Optional[Path] = None,
    ):
        """Initialize the Obsidian data source.

        :param vault_path: Path to the Obsidian vault
        :param model: The Ollama model to use
        :param persist_dir: Optional directory to persist the vector store
        :raises ValueError: If the vault path is invalid
        """
        if not vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")

        self.vault_path = vault_path
        self.model = model
        self.persist_dir = persist_dir
        self.embeddings = create_embeddings(model)
        self.vector_store = None
        self.qa_chain = None

    def load_documents(self) -> List[Tuple[str, Dict[str, Any], str]]:
        """Load documents from the Obsidian vault.

        :return: List of (filename, metadata, content) tuples
        """
        documents = []
        for file_path in self.vault_path.glob("**/*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                metadata = {
                    "source": str(file_path.relative_to(self.vault_path)),
                    **self._extract_metadata(content)
                }
                documents.append((str(file_path.name), metadata, content))
        return documents

    def index_documents(self) -> None:
        """Index the documents in the Obsidian vault.

        :raises OSError: If there are issues during indexing
        """
        try:
            documents = self.load_documents()
            langchain_docs = [
                Document(page_content=content, metadata=metadata)
                for _, metadata, content in documents
            ]
            self.vector_store = create_vector_store(
                langchain_docs,
                self.embeddings,
                self.persist_dir,
            )
            self.qa_chain = create_qa_chain(self.vector_store, self.model)
        except Exception as e:
            raise OSError(f"Failed to index documents: {str(e)}")

    def query(self, query: str, filters: Optional[Dict[str, str]] = None) -> str:
        """Query the Obsidian vault.

        :param query: The query string
        :param filters: Optional filters to apply
        :return: The query response
        :raises ValueError: If the vector store is not initialized
        """
        if not self.vector_store:
            self.index_documents()

        if not self.qa_chain:
            self.qa_chain = create_qa_chain(self.vector_store, self.model, filters)

        return self.qa_chain.invoke(query)

    def _extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from document content.

        :param content: The document content to extract metadata from
        :return: Dictionary of metadata
        """
        metadata = {}

        # Extract frontmatter
        frontmatter_match = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            for line in frontmatter.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

        # Extract tags
        tags = re.findall(r"#([a-zA-Z0-9_-]+)", content)
        if tags:
            metadata["tags"] = ", ".join(tags)

        # Extract links
        links = re.findall(r"\[\[(.*?)\]\]", content)
        if links:
            metadata["links"] = ", ".join(links)

        return metadata