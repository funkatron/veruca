# Veruca - Obsidian Vault Query Tool

Named after Veruca Salt from "Charlie and the Chocolate Factory", Veruca is a demanding tool that gets exactly what you want from your Obsidian vault. Just like Veruca who always got her way, this tool helps you find exactly what you're looking for in your notes.

This tool allows you to index and query your Obsidian vault using local embeddings and LLMs. It provides a simple way to search through your notes using natural language queries, while preserving Obsidian's unique features and metadata. **All data is processed locally, without sending any data to external servers.**

## Documentation

- [User Guide](README.md) - This file, containing usage instructions and examples
- [Technical Documentation](TECHNICAL.md) - Detailed technical explanation of how the tool works

## Features

- Load and process Obsidian vault files recursively
- Preserve and process Obsidian-specific features:
  - Internal links (`[[filename]]` and `[[filename|display text]]`)
  - Frontmatter (YAML metadata)
  - Tags (`#tag`, `#nested/tag`, `#project/active/2024`)
  - Callouts (admonitions)
- Index the content using local embeddings (nomic-embed-text model)
- Query the indexed data using natural language
- Filter queries by tags
- View source documents and their metadata
- **Local processing - no data is sent to external servers**
- Persistent storage of embeddings in `~/.local/share/veruca`

## Requirements

- Python 3.12+ (it may work with older versions, but it has not been tested)
- Ollama installed and running locally (for embeddings and LLM)
- Sufficient disk space for storing embeddings
- PyYAML (for parsing frontmatter)

## Installation

1. Clone this repository:
    ```sh
    git clone <repository-url>
    cd <repository-name>
    ```

2. Create a virtual environment in the root of the project:
    ```sh
    python -m venv venv
    ```
    Then, activate the virtual environment:
    ```sh
    source venv/bin/activate  # On Unix/macOS
    # or
    .\venv\Scripts\activate  # On Windows
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Install and start Ollama:
    - Visit [Ollama's website](https://ollama.ai) for installation instructions
    - Pull the required models:
      ```sh
      ollama pull nomic-embed-text
      ollama pull mistral
      ```

## Usage

### Indexing the Vault

To index your Obsidian vault, run:
```sh
./obsidian.py --vault /path/to/your/obsidian/vault
```

This will:
1. Recursively scan all Markdown files in the vault
2. Process Obsidian-specific features:
   - Parse frontmatter and tags
   - Convert internal links to readable text
   - Process callouts
3. Split the content into chunks
4. Generate embeddings using the nomic-embed-text model
5. Store the embeddings and metadata in a local ChromaDB database

### Querying the Vault

You can query your vault in several ways:

1. **Basic Query**
```sh
./obsidian.py --query "What do I know about project X?"
```

2. **Query with Tag Filtering**
```sh
./obsidian.py --query "What are my active projects?" --tags project/active,work/2024
```

The tool will:
1. Load the stored embeddings and metadata
2. Find the most relevant chunks for your query (filtered by tags if specified)
3. Use the Mistral model to generate a response based on the relevant content
4. Show the source documents and their metadata

### Example Queries

Here are some example queries to demonstrate how to use the tool effectively:

1. **Finding Specific Information**
```sh
./obsidian.py --query "What are my notes about Python programming?"
```

2. **Using Tags to Filter**
```sh
./obsidian.py --query "What are my project ideas?" --tags project,ideas
```

3. **Combining Concepts**
```sh
./obsidian.py --query "What do I know about machine learning in Python?" --tags python,ml
```

4. **Finding Related Notes**
```sh
./obsidian.py --query "What notes are related to my research on AI?"
```

The tool will provide:
- A detailed answer based on the relevant content
- Source documents with their paths and tags
- Confidence scores for the retrieved information

## Obsidian Features Support

The tool processes the following Obsidian-specific features:

1. **Internal Links**
   - Converts `[[filename]]` to readable text
   - Preserves display text in `[[filename|display text]]`

2. **Frontmatter**
   - Parses YAML frontmatter at the start of files
   - Includes frontmatter data in document metadata

3. **Tags**
   - Extracts all `#tag` instances
   - Supports nested tags (`#project/active`, `#area/work/tasks`)
   - Handles tags with numbers, hyphens, and underscores
   - Correctly identifies tags in various contexts (inline, lists, etc.)
   - Ignores tags in code blocks and headings
   - Includes tags in document metadata
   - Allows filtering queries by tags

4. **Callouts**
   - Processes Obsidian's callout syntax (`> [!NOTE]`)
   - Converts to a readable format

## Troubleshooting

1. **No Markdown files found**
   - Ensure the vault path is correct
   - Check if the files have the `.md` extension

2. **Ollama connection issues**
   - Verify Ollama is running (`ollama serve`)
   - Check if the required models are installed

3. **Memory issues**
   - If processing large vaults, you might need to adjust the chunk size in the code
   - Default is 500 tokens with 50 token overlap

4. **Frontmatter parsing issues**
   - Ensure your frontmatter is valid YAML
   - Check for proper `---` delimiters

5. **Query not returning results**
   - Try rephrasing your question
   - Check if the relevant tags are correct
   - Verify that the information exists in your vault

6. **Storage location issues**
   - The tool stores data in `~/.local/share/veruca`
   - Ensure you have write permissions to this directory
   - Check available disk space in this location

## Contributing

Feel free to submit issues and enhancement requests! Before contributing, please read our [Technical Documentation](TECHNICAL.md) to understand the architecture and design decisions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.