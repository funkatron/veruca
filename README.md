# Veruca - Obsidian Vault Query Tool

Veruca is a project to build a personal, local LLM-powered knowledge base. This tool, the Obsidian importer, helps you build that knowledge base by indexing and querying your Obsidian vault using local embeddings and LLMs. It processes your notes while preserving Obsidian's features and metadata. **All data is processed locally, without sending any data to external servers.**

## Documentation

- [User Guide](README.md) - This file, containing usage instructions and examples
- [Technical Documentation](TECHNICAL.md) - Detailed technical explanation of how the tool works

## Features

- Load and process Obsidian vault files recursively
- Preserve and process Obsidian-specific features:
  - Internal links (`[[filename]]` and `[[filename|display text]]`)
  - Frontmatter (YAML metadata)
  - Tags (`#tag`)
  - Callouts (admonitions)
- Index the content using local embeddings (nomic-embed-text model)
- Query the indexed data using natural language
- Filter queries by tags
- View source documents and their metadata
- **Local processing - no data is sent to external servers**
- Persistent storage of embeddings for faster subsequent queries

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

2. **Query with Filters**
```sh
# Filter by tags
./obsidian.py --query "What are my goals?" --filter tags=goals,2024

# Filter by status
./obsidian.py --query "What are my project tasks?" --filter status=in-progress

# Filter by type
./obsidian.py --query "What are my research findings?" --filter type=research

# Multiple filters
./obsidian.py --query "What are my active research projects?" --filter status=active --filter type=research

# Combine tag and metadata filters
./obsidian.py --query "What are my Python projects?" --filter tags=python --filter status=active
```

The tool will:
1. Load the stored embeddings and metadata
2. Find the most relevant chunks for your query (filtered by specified criteria)
3. Use the Mistral model to generate a response based on the relevant content
4. Show the source documents and their metadata

### Example Queries

Here are some example queries to demonstrate how to use the tool effectively:

1. **Finding Specific Information**
```sh
./obsidian.py --query "What are my notes about Python programming?"
```

2. **Using Tag Filters**
```sh
./obsidian.py --query "What are my project ideas?" --filter tags=project,ideas
```

3. **Combining Concepts**
```sh
./obsidian.py --query "What do I know about machine learning in Python?" --filter tags=python,ml
```

4. **Finding Related Notes**
```sh
./obsidian.py --query "What notes are related to my research on AI?"
```

5. **Filtering by Metadata Fields**
```sh
# Filter by status
./obsidian.py --query "What are my project tasks?" --filter status=in-progress

# Filter by type
./obsidian.py --query "What are my research findings?" --filter type=research

# Filter by date
./obsidian.py --query "What did I write about recently?" --filter date=2024-03

# Multiple filters
./obsidian.py --query "What are my active research projects?" --filter status=active --filter type=research

# Combine with tags
./obsidian.py --query "What are my Python projects?" --filter tags=python --filter status=active
```

6. **Viewing Source Documents and Metadata**
```sh
# Example response showing source documents and metadata:
Answer: Based on your research notes, you've been exploring machine learning applications in healthcare.

Sources:
1. healthcare-ml-notes.md (Score: 0.89)
   Tags: #research, #healthcare, #ml
   Path: projects/healthcare/healthcare-ml-notes.md
   Created: 2024-02-15
   Status: in-progress

2. ml-papers-summary.md (Score: 0.85)
   Tags: #research, #ml, #papers
   Path: research/papers/ml-papers-summary.md
   Last Modified: 2024-03-10
   Type: literature-review
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
   - Includes tags in document metadata
   - Allows filtering queries by tags using `--filter tags=tag1,tag2`

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
   - Check if the relevant filters are correct
   - Verify that the information exists in your vault

## Contributing

Feel free to submit issues and enhancement requests! Before contributing, please read our [Technical Documentation](TECHNICAL.md) to understand the architecture and design decisions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.