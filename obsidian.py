#!/usr/bin/env python

import os
import argparse
import sys
import re
import yaml
import subprocess
import time
import requests
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate

# Define constants
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Custom prompt template for better context
CUSTOM_PROMPT = """You are a helpful assistant that answers questions based on the provided context from an Obsidian vault.
The context comes from various notes, and each piece of information includes metadata about its source.

Context information:
{context}

Question: {question}

Please provide a detailed answer based on the context. If the information comes from specific notes, mention them by name.
If you're not sure about something, say so. Don't make up information that isn't in the context.

Answer:"""

def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content."""
    frontmatter = {}
    content_without_frontmatter = content

    # Check for frontmatter pattern - more lenient pattern
    frontmatter_pattern = r'^---\n(.*?)\n---'  # Removed \s* and final \n
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if match:
        try:
            print(f"Attempting to parse YAML: {match.group(1)}")  # Debug print
            # Use yaml.safe_load with a custom constructor to preserve date strings
            class PreserveDateStrings(yaml.SafeLoader):
                @classmethod
                def remove_implicit_resolver(cls, tag_to_remove):
                    if not hasattr(cls, 'yaml_implicit_resolvers'):
                        return
                    for first_letter, mappings in cls.yaml_implicit_resolvers.items():
                        cls.yaml_implicit_resolvers[first_letter] = [(tag, regexp)
                                                                    for tag, regexp in mappings
                                                                    if tag != tag_to_remove]

            # Remove the default date/time resolver
            PreserveDateStrings.remove_implicit_resolver('tag:yaml.org,2002:timestamp')
            frontmatter = yaml.load(match.group(1), Loader=PreserveDateStrings)
            if frontmatter is None:
                frontmatter = {}
            content_without_frontmatter = content[match.end():]
        except yaml.YAMLError as e:
            print(f"Error: Invalid YAML frontmatter: {str(e)}")
            raise SystemExit(1)
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise SystemExit(1)

    return frontmatter, content_without_frontmatter

def extract_tags(content: str) -> List[str]:
    """Extract Obsidian inline tags from content."""
    # Remove YAML frontmatter if present
    _, content_without_frontmatter = parse_frontmatter(content)
    # Find all #tag occurrences that are preceded by whitespace, start of line, or punctuation
    # and not part of a larger word (i.e., not followed by a word character)
    tags = []
    for match in re.finditer(r'(?:^|\s|[^\w#])#(\w+)(?!\w)', content_without_frontmatter):
        tag = match.group(1)
        # Skip the word "tags" as it's not a real tag
        if tag != "tags":
            tags.append(tag)
    return list(set(tags))

def process_obsidian_links(content: str, vault_path: str) -> str:
    """Process Obsidian's internal links and convert them to readable text."""
    def replace_link(match):
        link_text = match.group(1)
        # If there's a display text, use it, otherwise use the link
        if '|' in link_text:
            link, display = link_text.split('|', 1)
            return display
        return link_text

    # Replace [[filename]] or [[filename|display text]] with display text or filename
    return re.sub(r'\[\[(.*?)\]\]', replace_link, content)

def process_callouts(content: str) -> str:
    """Process Obsidian's callouts (admonitions) to make them more readable."""
    def process_callout(match):
        callout_type = match.group(1)
        callout_content = match.group(2)
        return f"[{callout_type.upper()}] {callout_content}"

    # Replace > [!NOTE] style callouts with a more readable format
    return re.sub(r'>\s*\[!(\w+)\]\s*(.*?)(?=\n|$)', process_callout, content)

def load_markdown_files(vault_path: str) -> List[Tuple[str, Dict[str, Any], str]]:
    """Load all Markdown files from the given vault path with Obsidian-specific processing."""
    if not os.path.exists(vault_path):
        print(f"Error: Vault path '{vault_path}' does not exist.")
        raise SystemExit(1)

    documents = []
    try:
        for root, _, files in os.walk(vault_path):
            for file in files:
                if file.endswith('.md'):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        frontmatter, content_without_frontmatter = parse_frontmatter(content)
                        tags = extract_tags(content)
                        # Merge frontmatter tags with inline tags
                        if 'tags' in frontmatter:
                            tags.extend(frontmatter['tags'])
                        # Convert list of tags to comma-separated string
                        frontmatter['tags'] = ','.join(sorted(set(tags)))
                        # Add source and path metadata
                        frontmatter['source'] = file
                        frontmatter['path'] = str(Path(file_path).relative_to(vault_path))
                        # Convert any remaining lists to strings
                        for key, value in frontmatter.items():
                            if isinstance(value, list):
                                frontmatter[key] = ','.join(map(str, value))
                        documents.append((file, frontmatter, content_without_frontmatter))
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
                        raise SystemExit(1)
    except Exception as e:
        print(f"Error walking through vault: {str(e)}")
        raise SystemExit(1)

    return documents

def index_vault(vault_path: str) -> None:
    """Index all markdown files in the vault."""
    print("Loading Markdown files...")
    try:
        markdown_data = load_markdown_files(vault_path)
        print(f"Loaded {len(markdown_data)} Markdown files.")

        # Process documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = [Document(page_content=chunk, metadata=metadata)
                    for _, metadata, text in markdown_data
                    for chunk in text_splitter.split_text(text)]

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=OllamaEmbeddings(model="llama2"),
            persist_directory="./data/chroma"
        )
        vectorstore.persist()
        print("Indexing complete.")
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        raise SystemExit(1)

def query_vault(query: str, filter_tags: List[str] = None, test_mode: bool = False) -> str:
    """Query the vault with a natural language query and optional tag filters.

    Args:
        query: The natural language query
        filter_tags: Optional list of tags to filter results by
        test_mode: If True, return a mock response for testing

    Returns:
        str: The response from the LLM
    """
    try:
        if test_mode:
            return f"Mock response for query: {query} with tags: {filter_tags}"

        embedding_model = OllamaEmbeddings(model="llama2")
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

        # Create filter dict if tags are provided
        filter_dict = {"tags": {"$in": filter_tags}} if filter_tags else None

        # Get relevant documents
        docs = vector_store.similarity_search(query, filter=filter_dict)

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=Ollama(model="llama2"),
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )

        # Get response
        response = qa_chain.invoke({"query": query})
        return response["result"]

    except Exception as e:
        print(f"Error during query: {str(e)}")
        sys.exit(1)

def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        return response.status_code == 200
    except Exception as e:
        print(f"Connection check failed: {type(e).__name__}: {str(e)}")
        return False

def get_ollama_installation_type() -> Optional[str]:
    """Determine how Ollama is installed (brew or direct)."""
    # Check for Homebrew installation
    try:
        result = subprocess.run(["brew", "list", "ollama"], capture_output=True, text=True)
        if result.returncode == 0:
            return "brew"
    except Exception as e:
        print(f"Brew check failed: {type(e).__name__}: {str(e)}")
        pass

    # Check for direct installation
    if os.path.exists("/usr/local/bin/ollama") or os.path.exists("/opt/homebrew/bin/ollama"):
        return "direct"

    return None

def start_ollama() -> bool:
    """Start the Ollama server."""
    install_type = get_ollama_installation_type()
    if not install_type:
        print("Error: Ollama is not installed. Please install it first from https://ollama.com/download")
        return False

    try:
        if install_type == "brew":
            subprocess.run(["brew", "services", "start", "ollama"], check=True)
        else:
            # For direct installation, start in background
            subprocess.Popen(["ollama", "serve"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        # Wait for server to start
        for _ in range(10):  # Try for 10 seconds
            if check_ollama_running():
                print("Ollama server started successfully.")
                return True
            time.sleep(1)

        print("Error: Ollama server failed to start.")
        return False
    except Exception as e:
        import traceback
        print(f"Error starting Ollama: {type(e).__name__}: {str(e)}")
        print("Stack trace:")
        print(traceback.format_exc())
        return False

def stop_ollama() -> bool:
    """Stop the Ollama server."""
    install_type = get_ollama_installation_type()
    if not install_type:
        print("Error: Ollama is not installed.")
        return False

    try:
        if install_type == "brew":
            try:
                subprocess.run(["brew", "services", "stop", "ollama"], check=True)
            except Exception as e:
                print(f"Brew stop failed: {type(e).__name__}: {str(e)}")
                # Try direct method as fallback
                install_type = "direct"

        if install_type == "direct":
            try:
                subprocess.run(["pkill", "ollama"], check=True)
            except Exception as e:
                print(f"Process kill failed: {type(e).__name__}: {str(e)}")
                # Both methods failed
                return False

        # Wait for server to stop
        for _ in range(10):  # Try for 10 seconds
            if not check_ollama_running():
                print("Ollama server stopped successfully.")
                return True
            time.sleep(1)

        print("Error: Ollama server failed to stop.")
        return False
    except Exception as e:
        import traceback
        print(f"Error stopping Ollama: {type(e).__name__}: {str(e)}")
        print("Stack trace:")
        print(traceback.format_exc())
        return False

def check_ollama_status() -> None:
    """Check and display Ollama server status."""
    if check_ollama_running():
        print("Ollama server is running.")
        install_type = get_ollama_installation_type()
        if install_type:
            print(f"Installation type: {install_type}")
    else:
        print("Ollama server is not running.")
        install_type = get_ollama_installation_type()
        if install_type:
            print(f"Installation type: {install_type}")
        else:
            print("Ollama is not installed. Please install it from https://ollama.com/download")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query your Obsidian vault using Ollama.")
    parser.add_argument("--vault", help="Path to your Obsidian vault", required=False)
    parser.add_argument("--query", help="Question to ask the vault", required=False)
    parser.add_argument("--tags", help="Comma-separated list of tags to filter by", required=False)
    parser.add_argument("--ollama-status", action="store_true", help="Check Ollama server status")
    parser.add_argument("--start-ollama", action="store_true", help="Start Ollama server")
    parser.add_argument("--stop-ollama", action="store_true", help="Stop Ollama server")

    args = parser.parse_args()

    if args.ollama_status:
        check_ollama_status()
    elif args.start_ollama:
        start_ollama()
    elif args.stop_ollama:
        stop_ollama()
    elif args.vault:
        if not check_ollama_running():
            print("Error: Ollama server is not running. Start it with --start-ollama")
            sys.exit(1)
        index_vault(args.vault)
    elif args.query:
        if not check_ollama_running():
            print("Error: Ollama server is not running. Start it with --start-ollama")
            sys.exit(1)
        filter_tags = args.tags.split(",") if args.tags else None
        query_vault(args.query, filter_tags)
    else:
        print(f"Usage: {parser.prog} [--vault /path/to/obsidian] [--query 'your question' [--tags tag1,tag2]]")
        print("       [--ollama-status] [--start-ollama] [--stop-ollama]")
        sys.exit(1)
