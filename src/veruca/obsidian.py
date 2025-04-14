#!/usr/bin/env python3
"""Query your Obsidian vault using Ollama."""

import os
import argparse
import sys
import re
import yaml
import subprocess
import time
import requests
import warnings
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Suppress ResourceWarnings from unclosed socket connections
# These warnings occur because the Ollama clients use async HTTP connections
# that are managed by the client libraries but Python's garbage collector
# is extra careful about reporting them. The connections are properly
# managed and closed by the client libraries.
warnings.filterwarnings("ignore", category=ResourceWarning)

__version__ = "0.1.0"

# Get the user's data directory
USER_DATA_DIR = os.path.expanduser("~/.local/share/veruca")
CHROMA_DB_PATH = os.path.join(USER_DATA_DIR, "chroma")

# Ensure the data directory exists
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Define constants
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Tag regex pattern explanation:
# (?:^|\s|[^\w#])     # Tag must start at beginning, after space, or non-word-non-# char
# #                    # Literal #
# (                    # Start capturing
#   [\w-]+            # First part: word chars or hyphens
#   (?:               # Start non-capturing group for nested tags
#     /               # Forward slash
#     [\w-]+         # More word chars or hyphens
#   )*                # Allow multiple nested levels
# )                   # End capturing
# (?=                 # Positive lookahead for what comes after
#   (?:              # Non-capturing group for valid endings
#     \s             # Whitespace
#     |              # OR
#     [^\w#/]        # Non-word char except # and /
#     |              # OR
#     /(?!\w)        # Forward slash not followed by word char
#     |              # OR
#     #              # Hash
#     |              # OR
#     $              # End of string
#     |              # OR
#     :              # Colon (for Obsidian's tag syntax)
#   )
# )
TAG_PATTERN = r'(?:^|\s|[^\w#])#([\w-]+(?:/[\w-]+)*)(?=(?:\s|[^\w#/]|/(?!\w)|#|$|:))'

# Custom prompt template for better context
CUSTOM_PROMPT = """You are a string matching assistant. Your ONLY job is to find exact matches in the provided text.

When asked about tags:
1. Look for the line starting with "tags:" in the METADATA section
2. Copy and paste ONLY the exact text that appears after "tags:"
3. Do not add ANY other text or explanation
4. Do not try to be helpful or format the response
5. Do not look at the content section at all

Context:
{context}

Question: {question}

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
            yaml_content = match.group(1)
            print(f"Attempting to parse YAML: {yaml_content}")  # Debug print

            # Skip if content looks like a template
            if '{{' in yaml_content or '}}' in yaml_content:
                print("Skipping template-like YAML content")
                return {}, content

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
            try:
                parsed = yaml.load(yaml_content, Loader=PreserveDateStrings)
                if parsed is not None:
                    frontmatter = parsed
                content_without_frontmatter = content[match.end():]
            except yaml.YAMLError as e:
                print(f"Error: Invalid YAML frontmatter: {str(e)}")
                raise SystemExit(1)
        except Exception as e:
            print(f"Error: Error parsing frontmatter: {str(e)}")
            raise SystemExit(1)

    return frontmatter, content_without_frontmatter

def extract_tags(content: str) -> List[str]:
    """Extract Obsidian inline tags from content.

    Tags in Obsidian follow these rules:
    1. Start with #
    2. Can contain letters, numbers, underscores, and hyphens
    3. Can contain forward slashes for nested tags (e.g., #project/active)
    4. Cannot contain spaces
    5. Can be at the start, middle, or end of a line
    6. Are not recognized inside code blocks
    7. Are not recognized as part of a heading (e.g., ## Tags)
    8. Can be followed by punctuation which is not part of the tag
    9. Can be nested (e.g., #programming/python/django)
    """
    tags = []
    in_code_block = False

    for line in content.split('\n'):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Handle code blocks
        if line.startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # Skip headings (lines starting with one or more #)
        if re.match(r'^#+\s', line):
            continue

        # Skip YAML frontmatter markers
        if line == '---':
            continue

        # Find all tags in the line
        for match in re.finditer(TAG_PATTERN, line):
            tag = match.group(1)
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
    """Load and process all markdown files in the vault."""
    vault_dir = Path(vault_path)
    if not vault_dir.exists():
        print(f"Error: Vault path '{vault_path}' does not exist.")
        sys.exit(1)
    markdown_files = []

    # Get all markdown files recursively
    for md_file in vault_dir.rglob("*.md"):
        try:
            print(f"\nDEBUG - Processing file: {md_file}")
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse frontmatter
            frontmatter, content_without_frontmatter = parse_frontmatter(content)
            print(f"DEBUG - Parsed frontmatter: {frontmatter}")

            # Extract inline tags
            inline_tags = extract_tags(content_without_frontmatter)
            print(f"DEBUG - Extracted inline tags: {inline_tags}")

            # Merge frontmatter tags with inline tags
            all_tags = set()
            if "tags" in frontmatter:
                if isinstance(frontmatter["tags"], list):
                    all_tags.update(frontmatter["tags"])
                else:
                    all_tags.add(frontmatter["tags"])
            all_tags.update(inline_tags)

            # Add source and path metadata
            metadata = {
                "source": md_file.name,
                "path": str(md_file.relative_to(vault_dir)),
                **{k: v for k, v in frontmatter.items() if k != "tags"},  # Add all frontmatter fields except tags
                "tags": list(all_tags)  # Store merged tags as a list
            }
            print(f"DEBUG - Final metadata: {metadata}")

            # Process Obsidian-specific features
            processed_content = process_obsidian_links(content_without_frontmatter, vault_dir)
            processed_content = process_callouts(processed_content)

            markdown_files.append((md_file.name, metadata, processed_content))

        except Exception as e:
            print(f"Error processing {md_file}: {str(e)}")
            continue

    return markdown_files

def index_vault(vault_path: str) -> None:
    """Index all markdown files in the vault."""
    print("Loading Markdown files...")
    try:
        markdown_data = load_markdown_files(vault_path)
        print(f"Loaded {len(markdown_data)} Markdown files.")

        # Process documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = []
        for _, metadata, text in markdown_data:
            # Convert tags list to CSV string for Chroma storage
            if "tags" in metadata:
                if isinstance(metadata["tags"], list):
                    metadata["tags"] = ','.join(metadata["tags"])
                elif not isinstance(metadata["tags"], str):
                    metadata["tags"] = str(metadata["tags"])

            # Create documents for each chunk
            for chunk in text_splitter.split_text(text):
                doc = Document(page_content=chunk, metadata=metadata.copy())
                documents.append(doc)

        print("Document metadata examples:")
        for doc in documents[:2]:  # Print first two documents' metadata
            print(f"Metadata: {doc.metadata}")

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory=CHROMA_DB_PATH
        )
        vectorstore.persist()  # Persist the vector store
        print("Indexing complete.")
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        raise SystemExit(1)

def query_vault(query: str, filters: Dict[str, str] = None) -> str:
    """Query the vector store with a question."""
    try:
        # Silently convert deprecated tags field to filter
        if filters and "tags" in filters:
            tags_value = filters.pop("tags")
            filters["tags:in"] = tags_value

        print(f"Querying with: {query}, filters: {filters}")

        # Create embedding model once and reuse
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        print("Created embedding model")

        # Create vector store with the embedding model
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_model
        )
        print("Created vector store")

        # Get relevant documents
        docs = vector_store.similarity_search(query)
        print(f"Found {len(docs)} relevant documents")
        print("\nDEBUG - Retrieved documents:")
        for i, doc in enumerate(docs):
            print(f"\nDocument {i+1}:")
            print("Content:", doc.page_content[:100] + "...")
            print("Metadata:", doc.metadata)

        # Filter documents by metadata if specified
        if filters:
            filtered_docs = []
            for doc in docs:
                doc_matches = True
                for field, value in filters.items():
                    # Handle special operators
                    if ":" in field:
                        field_name, operator = field.split(":", 1)
                        if field_name == "tags":
                            # Convert comma-separated values to list
                            values = [v.strip() for v in value.split(",") if v.strip()]
                            if operator == "in":
                                # Match if any filter tag is in doc tags
                                doc_tags = [t.strip() for t in doc.metadata.get("tags", "").split(",") if t.strip()]
                                if any(tag in doc_tags for tag in values):
                                    doc_matches = True
                                else:
                                    doc_matches = False
                            elif operator == "nin":
                                # Match if no filter tag is in doc tags
                                doc_tags = [t.strip() for t in doc.metadata.get("tags", "").split(",") if t.strip()]
                                if not any(tag in doc_tags for tag in values):
                                    doc_matches = True
                                else:
                                    doc_matches = False
                            else:
                                raise ValueError(f"Invalid operator for tags: {operator}")
                        else:
                            # Handle numeric comparisons
                            try:
                                filter_value = float(value)
                                if operator == "gt":
                                    if doc.metadata.get(field) > filter_value:
                                        doc_matches = True
                                    else:
                                        doc_matches = False
                                elif operator == "gte":
                                    if doc.metadata.get(field) >= filter_value:
                                        doc_matches = True
                                    else:
                                        doc_matches = False
                                elif operator == "lt":
                                    if doc.metadata.get(field) < filter_value:
                                        doc_matches = True
                                    else:
                                        doc_matches = False
                                elif operator == "lte":
                                    if doc.metadata.get(field) <= filter_value:
                                        doc_matches = True
                                    else:
                                        doc_matches = False
                                elif operator == "ne":
                                    if doc.metadata.get(field) != filter_value:
                                        doc_matches = True
                                    else:
                                        doc_matches = False
                                else:
                                    raise ValueError(f"Invalid operator: {operator}")
                            except ValueError:
                                # If not numeric, treat as string comparison
                                if operator == "ne":
                                    if doc.metadata.get(field) != value:
                                        doc_matches = True
                                    else:
                                        doc_matches = False
                                else:
                                    raise ValueError(f"Invalid operator for non-numeric field: {operator}")
                    else:
                        # Simple equality check
                        if doc.metadata.get(field) == value:
                            doc_matches = True
                        else:
                            doc_matches = False
                if doc_matches:
                    filtered_docs.append(doc)
            docs = filtered_docs
            print(f"\nAfter filtering: {len(docs)} documents")
            print("\nDEBUG - Filtered documents:")
            for i, doc in enumerate(docs):
                print(f"\nFiltered Document {i+1}:")
                print("Content:", doc.page_content[:100] + "...")
                print("Metadata:", doc.metadata)

        for doc in docs:
            print(f"Document metadata: {doc.metadata}")

        # Debug: Print document formatting
        def format_doc_with_debug(doc):
            # Safely get metadata values
            tags = doc.metadata.get("tags", "")
            title = doc.metadata.get("title", "")

            # Format metadata as a simple string
            metadata_str = f"tags: {tags}\ntitle: {title}"

            formatted = {
                "page_content": doc.page_content,
                "metadata": metadata_str
            }

            print("\nDEBUG - Document being formatted:")
            print("Raw metadata:", doc.metadata)
            print("Formatted document:")
            print("Content:", formatted["page_content"])
            print("Metadata:", formatted["metadata"])
            print("---")
            return formatted

        # Create document prompt that includes metadata
        document_prompt = PromptTemplate(
            input_variables=["page_content", "metadata"],
            template="""METADATA:
{metadata}

CONTENT:
{page_content}"""
        )

        # Create QA chain with custom prompt
        prompt = PromptTemplate(template=CUSTOM_PROMPT, input_variables=["context", "question"])

        # Create retriever with filters
        search_kwargs = {}
        if filters:
            search_kwargs["filter"] = {}
            for field, value in filters.items():
                # Handle special operators
                if ":" in field:
                    field_name, operator = field.split(":", 1)
                    if field_name == "tags":
                        # Convert comma-separated values to list
                        values = [v.strip() for v in value.split(",") if v.strip()]
                        if operator == "in":
                            # Match if any filter tag is in doc tags
                            search_kwargs["filter"][field_name] = {"$in": values}
                        elif operator == "nin":
                            # Match if no filter tag is in doc tags
                            search_kwargs["filter"][field_name] = {"$nin": values}
                        else:
                            raise ValueError(f"Invalid operator for tags: {operator}")
                    else:
                        # Handle numeric comparisons
                        try:
                            filter_value = float(value)
                            if operator == "gt":
                                search_kwargs["filter"][field_name] = {"$gt": filter_value}
                            elif operator == "gte":
                                search_kwargs["filter"][field_name] = {"$gte": filter_value}
                            elif operator == "lt":
                                search_kwargs["filter"][field_name] = {"$lt": filter_value}
                            elif operator == "lte":
                                search_kwargs["filter"][field_name] = {"$lte": filter_value}
                            elif operator == "ne":
                                search_kwargs["filter"][field_name] = {"$ne": filter_value}
                            else:
                                raise ValueError(f"Invalid operator: {operator}")
                        except ValueError:
                            # If not numeric, treat as string comparison
                            if operator == "ne":
                                search_kwargs["filter"][field_name] = {"$ne": value}
                            else:
                                raise ValueError(f"Invalid operator for non-numeric field: {operator}")
                else:
                    # Simple equality check
                    search_kwargs["filter"][field] = value

        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOllama(model="llama2"),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "document_prompt": document_prompt,
                "document_variable_name": "context",
                "document_separator": "\n\n",
                "format_document": format_doc_with_debug
            }
        )
        print("Created QA chain")

        # Get response
        response = qa_chain.invoke({"query": query})
        print("Got response")
        print(response["result"])
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
    parser.add_argument("--filter", help="Filter by metadata field (format: field=value)", required=False, action="append")
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
        filters = {}
        if args.filter:
            for f in args.filter:
                field, value = f.split("=", 1)
                filters[field] = value
        query_vault(args.query, filters)
    else:
        print(f"Usage: {parser.prog} [--vault /path/to/obsidian] [--query 'your question' [--filter field=value]]")
        print("       [--ollama-status] [--start-ollama] [--stop-ollama]")
        sys.exit(1)
