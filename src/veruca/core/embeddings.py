"""Common embedding functionality for Veruca."""

import os
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import VectorStore
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field


# Default configuration
DEFAULT_MODEL = "llama2"
DEFAULT_TEMPLATE = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""


class FormattingRetriever(BaseRetriever, BaseModel):
    """A retriever that formats documents before returning them."""
    base_retriever: BaseRetriever = Field(description="The base retriever to wrap")

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents and format them."""
        docs = self.base_retriever.invoke(query)
        formatted_docs = []
        for doc in docs:
            # Format metadata as key: value pairs
            formatted_metadata = []
            for key, value in sorted(doc.metadata.items()):
                if key == "tags":
                    # Handle tags specially
                    if isinstance(value, str):
                        tags = [t.strip() for t in value.split(",") if t.strip()]
                    else:
                        tags = [t.strip() for t in value if t.strip()]
                    formatted_metadata.append(f"tags: {', '.join(sorted(tags))}")  # Sort for consistent display
                elif isinstance(value, (list, dict)):
                    formatted_metadata.append(f"{key}: {value}")
                else:
                    formatted_metadata.append(f"{key}: {value}")

            # Create new document with metadata included in page content
            formatted_docs.append(
                Document(
                    page_content=f"""Metadata:
{'\n'.join(formatted_metadata)}

Content:
{doc.page_content}""",
                    metadata=doc.metadata  # Keep original metadata for filtering
                )
            )
        return formatted_docs


def create_embeddings(model: str = "nomic-embed-text") -> OllamaEmbeddings:
    """Create an embedding model.

    :param model: The Ollama model to use for embeddings
    :return: An OllamaEmbeddings instance
    :raises ValueError: If the model name is invalid
    """
    try:
        return OllamaEmbeddings(model=model)
    except Exception as e:
        raise ValueError(f"Failed to create embeddings model: {str(e)}")


def create_vector_store(
    documents: List[Document],
    embeddings: OllamaEmbeddings,
    persist_dir: Optional[Path] = None,
) -> VectorStore:
    """Create a vector store from documents.

    :param documents: List of documents to index
    :param embeddings: Embedding model to use
    :param persist_dir: Optional directory to persist the vector store
    :return: A Chroma vector store
    :raises ValueError: If the documents list is empty
    :raises OSError: If the persist directory cannot be created
    """
    if not documents:
        raise ValueError("Cannot create vector store with empty documents list")

    if persist_dir:
        try:
            persist_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create persist directory: {str(e)}")

    # Create a new collection each time
    collection_name = "veruca_" + os.urandom(4).hex()
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir) if persist_dir else None,
        collection_name=collection_name
    )

    if persist_dir:
        db.persist()

    return db


def format_document(doc: Document) -> Dict[str, str]:
    """Format a document for the LLM chain."""
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


def create_qa_chain(
    vector_store: VectorStore,
    model: str = "mistral",
    filters: Optional[Dict[str, str]] = None,
    prompt_template: Optional[str] = None
) -> RetrievalQA:
    """Create a QA chain for querying the vector store.

    Args:
        vector_store: The vector store to query
        model: The Ollama model to use for generating answers
        filters: Optional dictionary of filters to apply to the search. Supports:
            - Simple equality: `{"status": "active"}` - matches documents where status equals "active"
            - Tag filtering: `{"tags:in": "python,programming"}` - matches documents with either "python" or "programming" tags
            - Numeric comparisons:
                - `{"priority:gt": "3"}` - matches documents with priority > 3
                - `{"priority:lte": "3"}` - matches documents with priority ≤ 3
                - `{"priority:gte": "3"}` - matches documents with priority ≥ 3
                - `{"priority:lt": "3"}` - matches documents with priority < 3
            - Other operators:
                - `{"status:ne": "draft"}` - matches documents where status is not "draft"
                - `{"tags:nin": "archived,old"}` - matches documents without "archived" or "old" tags
        prompt_template: Optional custom prompt template for the QA chain

    Returns:
        A RetrievalQA chain configured with the given parameters

    Examples:
        ```python
        # Simple status filter
        filters = {"status": "active"}

        # Filter by multiple tags
        filters = {"tags:in": "python,programming"}

        # Filter by numeric value
        filters = {"priority:gt": "3"}

        # Exclude certain tags
        filters = {"tags:nin": "archived,old"}

        # Combine multiple filters
        filters = {
            "status:ne": "draft",
            "tags:in": "python,programming",
            "priority:gt": "3"
        }
        ```
    """
    # Create document prompt that includes metadata
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )

    # Create QA chain with custom prompt
    prompt = PromptTemplate(
        template=prompt_template or "Question: {question}\n\nContext: {context}\n\nAnswer:",
        input_variables=["context", "question"]
    )

    # Create retriever with filters
    search_kwargs = {}
    if filters:
        filter_conditions = []
        for field, value in filters.items():
            # Handle special operators
            if ":" in field:
                field_name, operator = field.split(":", 1)
                if operator == "in":
                    # Handle $in operator (comma-separated values)
                    values = [v.strip() for v in value.split(",")]
                    filter_conditions.append({field_name: {"$in": values}})
                elif operator == "nin":
                    # Handle $nin operator (comma-separated values)
                    values = [v.strip() for v in value.split(",")]
                    filter_conditions.append({field_name: {"$nin": values}})
                elif operator == "ne":
                    # Handle $ne operator
                    filter_conditions.append({field_name: {"$ne": value}})
                elif operator == "gt":
                    # Handle $gt operator
                    filter_conditions.append({field_name: {"$gt": value}})
                elif operator == "gte":
                    # Handle $gte operator
                    filter_conditions.append({field_name: {"$gte": value}})
                elif operator == "lt":
                    # Handle $lt operator
                    filter_conditions.append({field_name: {"$lt": value}})
                elif operator == "lte":
                    # Handle $lte operator
                    filter_conditions.append({field_name: {"$lte": value}})
                else:
                    raise ValueError(f"Unknown operator: {operator}")
            else:
                # Default to $eq operator
                filter_conditions.append({field: {"$eq": value}})

        # Only use $and if we have multiple conditions
        if len(filter_conditions) > 1:
            search_kwargs["filter"] = {"$and": filter_conditions}
        else:
            search_kwargs["filter"] = filter_conditions[0]

    # Create base retriever and wrap it with formatting
    base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    retriever = FormattingRetriever(base_retriever=base_retriever)

    # Create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOllama(model=model),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt,
            "document_prompt": document_prompt
        }
    )

    return qa_chain