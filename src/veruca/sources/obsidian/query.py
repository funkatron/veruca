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
CUSTOM_PROMPT = """You are a helpful assistant that answers questions about Obsidian vault content.
When answering questions about metadata or tags:
1. ONLY look at the content between the "--- Metadata Section ---" markers
2. NEVER invent or make up metadata values
3. For tags, ONLY list the exact tags found in the "tags:" line, separated by commas
4. Do not infer tags from the content section
5. If asked about tags and no "tags:" line is found in metadata, say "No tags found in metadata"
6. Format your response as a clear, concise list of the exact tags found, exactly as they appear

When answering other questions:
1. Focus ONLY on the content provided
2. Do not make assumptions or inferences
3. Format your response as a clear, concise answer

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
                tags = set()
                # Handle tags from frontmatter
                frontmatter_tags = frontmatter.get("tags", "")
                if isinstance(frontmatter_tags, str):
                    # Split comma-separated tags and strip whitespace
                    tags.update(t.strip() for t in frontmatter_tags.split(",") if t.strip())
                elif isinstance(frontmatter_tags, list):
                    tags.update(t.strip() for t in frontmatter_tags if t.strip())

                # Add tags from content
                tags.update(extract_tags(content))

                # Convert tags to comma-separated string for ChromaDB
                frontmatter["tags"] = ",".join(sorted(tags))  # Sort for consistency

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

        # Use a temporary directory for tests
        persist_dir = None
        if "CHROMA_DB_PATH" in os.environ:
            persist_dir = Path(os.environ["CHROMA_DB_PATH"])
            # Clear existing data if it exists
            if persist_dir.exists():
                import shutil
                shutil.rmtree(persist_dir)

        self.vector_store = create_vector_store(
            documents=langchain_docs,
            embeddings=embeddings,
            persist_dir=persist_dir
        )

    def query(self, query: str, filters: Optional[Dict[str, str]] = None) -> str:
        """Query the vector store with a question."""
        try:
            # Check if we have any documents before proceeding
            docs = self.load_documents()
            if not docs:
                return "No documents found in the vault."

            if not self.vector_store:
                # Create a new vector store for each query if one doesn't exist
                self.index_documents()

            # Create QA chain with custom prompt
            chain = create_qa_chain(
                vector_store=self.vector_store,
                filters=filters,
                prompt_template=CUSTOM_PROMPT
            )

            # Run the query
            result = chain.invoke(query)
            return result

        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)

def apply_filters(docs: List[Document], filters: Optional[Dict[str, Union[str, List[str]]]] = None) -> List[Document]:
    """Apply filters to a list of documents.

    Args:
        docs: List of documents to filter
        filters: Dictionary of filters to apply. Each filter can be:
            - Simple equality: {"status": "active"}
            - Tag filtering: {"tags:in": "python,programming"}
            - Numeric comparison: {"priority:gt": "3"}

    Returns:
        List of documents that match all filters
    """
    if not filters:
        return docs

    filtered_docs = []
    for doc in docs:
        doc_matches = True
        for field, value in filters.items():
            # Handle special operators
            if ":" in field:
                field_name, operator = field.split(":", 1)
                if field_name == "tags":
                    doc_tags = doc.metadata.get("tags", "")
                    if not doc_tags:
                        doc_matches = False
                        break

                    # Convert doc tags to list, trimming whitespace
                    if isinstance(doc_tags, str):
                        doc_tag_list = [t.strip() for t in doc_tags.split(",") if t.strip()]
                    else:
                        doc_tag_list = [t.strip() for t in doc_tags if t.strip()]

                    # Convert filter value to list, trimming whitespace
                    filter_tags = [v.strip() for v in value.split(",") if v.strip()]

                    if operator == "in":
                        # Match if any filter tag is in doc tags
                        if not any(tag in doc_tag_list for tag in filter_tags):
                            doc_matches = False
                            break
                    elif operator == "nin":
                        # Match if no filter tag is in doc tags
                        if any(tag in doc_tag_list for tag in filter_tags):
                            doc_matches = False
                            break
                    else:
                        raise ValueError(f"Invalid operator for tags: {operator}")
                else:
                    # Handle numeric comparisons
                    doc_value = doc.metadata.get(field_name)
                    if doc_value is None:
                        doc_matches = False
                        break

                    try:
                        doc_value = float(doc_value)
                        filter_value = float(value)
                    except (ValueError, TypeError):
                        doc_matches = False
                        break

                    if operator == "gt":
                        if not doc_value > filter_value:
                            doc_matches = False
                            break
                    elif operator == "gte":
                        if not doc_value >= filter_value:
                            doc_matches = False
                            break
                    elif operator == "lt":
                        if not doc_value < filter_value:
                            doc_matches = False
                            break
                    elif operator == "lte":
                        if not doc_value <= filter_value:
                            doc_matches = False
                            break
                    elif operator == "ne":
                        if doc_value == filter_value:
                            doc_matches = False
                            break
                    else:
                        raise ValueError(f"Invalid operator: {operator}")
            else:
                # Simple equality check
                doc_value = doc.metadata.get(field)
                if doc_value is None:
                    doc_matches = False
                    break
                if field == "tags":
                    # For tags, check if the exact tag is in the comma-separated list
                    if isinstance(doc_value, str):
                        doc_tags = [t.strip() for t in doc_value.split(",") if t.strip()]
                        if value not in doc_tags:
                            doc_matches = False
                            break
                    else:
                        doc_matches = False
                        break
                elif doc_value != value:
                    doc_matches = False
                    break

        if doc_matches:
            filtered_docs.append(doc)

    return filtered_docs