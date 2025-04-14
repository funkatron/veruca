"""Query functionality for Obsidian vaults."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from ...core.base import DataSource
from ...core.embeddings import (
    create_embeddings,
    create_vector_store,
    create_qa_chain,
)
from .parser import (
    parse_frontmatter,
    extract_tags,
    process_obsidian_links,
    process_callouts,
    TAG_PATTERN,
)

# Custom prompt template for better context
CUSTOM_PROMPT = """You are a helpful assistant that answers questions based on the provided context from an Obsidian vault.
Each document has two sections:
1. Content: The actual content of the document
2. Metadata: Information about the document like status, tags, etc.

When asked about metadata fields (like status, tags, etc.), ONLY look at the metadata section of the documents.
Do not try to infer metadata values from the content.
Always include the exact metadata values in your response.
When asked about metadata, start your response with "Looking at the metadata section:" followed by the relevant values.

Context:
{context}

Question: {question}

Answer:"""

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
        """Query the vector store with a question."""
        try:
            if not self.vector_store:
                self.index_documents()

            # Create QA chain with custom prompt
            chain = create_qa_chain(
                vector_store=self.vector_store,
                filters=filters,
                prompt_template=CUSTOM_PROMPT
            )

            # Run the query
            return chain.invoke(query)

        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)

def apply_filters(docs: List[Document], filters: Optional[Dict[str, Union[str, List[str]]]] = None) -> List[Document]:
    """Apply filters to a list of documents.

    Args:
        docs: List of documents to filter
        filters: Dictionary of field:value pairs to filter by. For tags, can be string or list of strings.

    Returns:
        Filtered list of documents

    Raises:
        ValueError: If filter field is invalid or filter value type is unsupported
    """
    if not filters:
        return docs

    filtered_docs = []
    valid_fields = {"status", "tags", "priority", "metadata"}  # Add other valid fields as needed

    for doc in docs:
        doc_matches = True
        for field, value in filters.items():
            # Validate filter field
            if field not in valid_fields:
                raise ValueError(f"Invalid filter field: {field}")

            # Validate filter value type
            if not isinstance(value, (str, list)):
                raise ValueError(f"Invalid filter value type for {field}: {type(value)}")

            # Handle tag filtering specially
            if field == "tags":
                doc_tags = doc.metadata.get("tags", "")
                if not doc_tags:
                    doc_matches = False
                    break

                # Convert filter value to list of individual tags
                if isinstance(value, str):
                    # If exact comma-separated string match is requested
                    if "," in value:
                        if doc_tags != value:
                            doc_matches = False
                            break
                        continue
                    filter_tags = [value]
                else:
                    filter_tags = value

                # Convert doc tags to list if it's a comma-separated string
                if isinstance(doc_tags, str):
                    if "," in doc_tags:
                        doc_tag_list = [t.strip() for t in doc_tags.split(",")]
                    else:
                        doc_tag_list = [doc_tags]
                else:
                    doc_tag_list = doc_tags

                # Check if any filter tag matches
                if not any(tag in doc_tag_list for tag in filter_tags):
                    doc_matches = False
                    break

            # Handle other fields
            else:
                doc_value = doc.metadata.get(field)
                if doc_value is None:
                    doc_matches = False
                    break
                filter_values = value if isinstance(value, list) else [value]
                if doc_value not in filter_values:
                    doc_matches = False
                    break

        if doc_matches:
            filtered_docs.append(doc)

    return filtered_docs