# Veruca

A tool for searching and querying Obsidian notes using local language models.

## What is Veruca?

Veruca is a command-line tool that enables you to:
- Search your Obsidian notes using natural language queries
- Filter results based on metadata and tags
- Process and index your notes locally
- Maintain privacy by running entirely on your machine

## Getting Started

### 1. Install Ollama

First, you need to install Ollama, which runs the language models locally:
- Visit [ollama.com/download](https://ollama.com/download)
- Download and install Ollama for your system
- After installation, run:
  ```bash
  ollama pull nomic-embed-text  # For embeddings
  ollama pull llama2           # For query responses
  ```

### 2. Install Veruca

```bash
# Clone the repository
git clone https://github.com/funkatron/veruca.git
cd veruca

# Create a virtual environment (like a clean workspace)
python -m venv venv

# Activate the virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install Veruca
pip install -e .
```

## Using Veruca

### Basic Usage

To search your Obsidian vault:
```bash
python -m veruca.cli --query "What are my active projects?" --filter status=active
```

### Command Options

- `--query`: Your question (required)
- `--vault-path`: Where your Obsidian vault is (default: ~/Obsidian)
- `--filter`: Filter by tags or other metadata (e.g., 'tags=python,status=active')
- `--model`: Which language model to use for responses (default: llama2)
- `--ollama-status`: Check if Ollama server is running
- `--start-ollama`: Start the Ollama server
- `--stop-ollama`: Stop the Ollama server

## How It Works

Veruca processes your notes in several steps:

1. **Document Processing**
   - Reads your Obsidian markdown files
   - Extracts metadata, tags, and links
   - Processes Obsidian-specific features

2. **Indexing**
   - Splits documents into manageable chunks
   - Generates embeddings using nomic-embed-text
   - Stores vectors in a local database

3. **Querying**
   - Converts your question into embeddings
   - Finds similar content in the vector store
   - Filters results based on metadata
   - Generates responses using llama2

## Features

- Works with Obsidian features:
  - Internal links (`[[filename]]` and `[[filename|display text]]`)
  - Frontmatter (YAML metadata)
  - Tags (`#tag` and nested tags `#tag/subtag`)
  - Callouts (admonitions)
- Everything runs locally on your computer
- No data is sent to the cloud
- Command-line interface

## Need Help?

If you run into any issues:
1. Check if Ollama is running (`ollama serve` or `--ollama-status`)
2. Make sure your Obsidian vault path is correct
3. Try a simple query first to test
4. Ensure you have the required models pulled (`nomic-embed-text` and `llama2`)

## Contributing

Want to help improve Veruca? Great! Here's how:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.