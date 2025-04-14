# Veruca

Veruca is a command-line tool that enables semantic search over your Obsidian vault using local language models. It indexes your notes and allows you to query them using natural language, leveraging the power of Ollama for embeddings and text generation.

## Features

- Local-first: All processing happens on your machine using Ollama
- Semantic search: Find relevant notes based on meaning, not just keywords
- Filter support: Narrow down results using frontmatter metadata
- Fast indexing: Efficiently process and update your vault
- Privacy-focused: Your notes never leave your computer

## Installation

1. Install [Ollama](https://ollama.ai)
2. Install Veruca:
   ```bash
   pip install veruca
   ```

## Usage

Veruca uses a command-based interface with the following structure:

```bash
veruca <command> [options]
```

Available commands:

### Query

Search your vault using natural language:

```bash
veruca query "What are my project deadlines?" --vault-path ~/vault
```

Options:
- `--vault-path`: Path to your Obsidian vault (default: ~/Obsidian)
- `--filter`: Filter results by frontmatter fields (e.g., "tags=project,status=active")

### Index

Index or reindex your vault:

```bash
veruca index --vault-path ~/vault
```

Options:
- `--vault-path`: Path to your Obsidian vault (default: ~/Obsidian)
- `--model`: Specify the embedding model to use (default: nomic-embed-text)

### Ollama Management

Manage the Ollama server:

```bash
veruca ollama status  # Check if Ollama is running
veruca ollama start   # Start the Ollama server
veruca ollama stop    # Stop the Ollama server
```

## How It Works

1. When you index your vault, Veruca:
   - Scans your vault for markdown files
   - Extracts content and frontmatter metadata
   - Generates embeddings using Ollama
   - Stores the index locally

2. When you query:
   - Your question is converted to an embedding
   - Relevant notes are retrieved using semantic similarity
   - A language model summarizes the results
   - Filters are applied based on frontmatter

## Requirements

- Python 3.8 or higher
- Ollama installed and running
- Required models:
  - nomic-embed-text (for embeddings)

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.