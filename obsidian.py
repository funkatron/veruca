#!/usr/bin/env python

import os
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings

# Define constants
CHROMA_DB_PATH = "./chroma_db"


# Function to load Markdown files
def load_markdown_files(vault_path):
    md_files = []
    for root, _, files in os.walk(vault_path):
        for file in files:
            if file.endswith(".md"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    md_files.append((file, f.read()))
    return md_files


# Function to process and index Markdown files
def index_vault(vault_path):
    print("Loading Markdown files...")
    markdown_data = load_markdown_files(vault_path)
    print(f"Loaded {len(markdown_data)} Markdown files.")

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = [Document(page_content=chunk, metadata={"source": filename})
                 for filename, text in markdown_data for chunk in text_splitter.split_text(text)]

    print(f"Split into {len(documents)} chunks.")

    # Store embeddings in ChromaDB using local Ollama embeddings
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_documents(documents, embedding=embedding_model, persist_directory=CHROMA_DB_PATH)
    vector_store.persist()

    print("Vault indexed successfully.")


# Function to query the indexed data
def query_vault(question):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
    retriever = vector_store.as_retriever()

    llm = Ollama(model="mistral")  # Change to another local model if needed

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = qa_chain.run(question)

    print("\n🔍 Answer:")
    print(response)


# CLI handling
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query your Obsidian vault using Ollama.")
    parser.add_argument("--vault", help="Path to your Obsidian vault", required=False)
    parser.add_argument("--query", help="Question to ask the vault", required=False)

    args = parser.parse_args()

    if args.vault:
        index_vault(args.vault)
    elif args.query:
        query_vault(args.query)
    else:
        print(f"Usage: {parser.prog} --vault /path/to/obsidian OR --query 'your question'")
