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


# Default configuration
DEFAULT_MODEL = "mistral"
DEFAULT_TEMPLATE = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""


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
) -> Chroma:
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

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir) if persist_dir else None,
    )

    if persist_dir:
        db.persist()

    return db


def create_qa_chain(
    vector_store: Chroma,
    model: str = DEFAULT_MODEL,
    filters: Optional[Dict[str, str]] = None,
) -> RunnablePassthrough:
    """Create a QA chain for querying documents.

    :param vector_store: The vector store to query
    :param model: The Ollama model to use
    :param filters: Optional filters to apply to the retriever
    :return: A RunnablePassthrough chain
    :raises ValueError: If the vector store is invalid
    """
    if not vector_store:
        raise ValueError("Cannot create QA chain with invalid vector store")

    retriever = vector_store.as_retriever(
        search_kwargs={"filter": filters} if filters else {}
    )

    # Create the LLM
    llm = ChatOllama(model=model)

    # Create the prompt
    prompt = PromptTemplate.from_template(DEFAULT_TEMPLATE)

    # Create the chain
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain