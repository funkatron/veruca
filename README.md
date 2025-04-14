# Veruca

A collection of tools for working with local LLMs, starting with Obsidian vault querying.

## Features

- Query your Obsidian vault using local LLMs
- Support for Obsidian-specific features:
  - Internal links (`[[filename]]` and `[[filename|display text]]`)
  - Frontmatter (YAML metadata)
  - Tags (`#tag` and nested tags `#tag/subtag`)
  - Callouts (admonitions)
- Local embedding generation using Ollama
- Natural language querying with metadata filtering
- Persistent storage of embeddings using ChromaDB
- Extensible architecture for adding new data sources

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/funkatron/veruca.git
   cd veruca
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. Install Ollama and pull the required model:
   ```bash
   ollama pull llama2
   ```

## Usage

### Querying Your Obsidian Vault

```bash
python -m veruca.cli --query "What are my active projects?" --filter status=active
```

### Command Line Options

- `--query`: The query to search for in your vault (required)
- `--vault-path`: Path to your Obsidian vault (default: ~/Obsidian)
- `--filter`: Filter results by metadata (e.g., 'tags=python,status=active')
- `--model`: Ollama model to use (default: llama2)

## Development

### Project Structure

```
src/veruca/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base.py        # Base classes for data sources
│   ├── embeddings.py  # Common embedding functionality
│   └── utils.py       # Shared utilities
├── sources/
│   ├── __init__.py
│   └── obsidian/
│       ├── __init__.py
│       ├── parser.py  # Frontmatter, tags, links parsing
│       └── query.py   # Obsidian-specific query handling
└── cli.py
```

### Adding a New Data Source

1. Create a new module in `src/veruca/sources/`
2. Implement the `DataSource` interface from `core/base.py`
3. Add your data source to the CLI or create a new interface

### Running Tests

```bash
python -m pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.