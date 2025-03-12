# Obsidian Vault Query Tool

This simple shows how you can use local embeddings to index and query your Obsidian vault, or other text-based data.

## Features

- Load and process Markdown files from a folder (the "vault" in Obsidian) recursively.
- Index the content using local embeddings
- Query the indexed data to retrieve relevant information

## Requirements

- Python 3.12+ (it may work with older versions, but it has not been tested)

## Installation

1. create a virtual environment in the root of the project.
    ```sh
    python -m venv venv
    ```
    Then, activate the virtual environment:
    ```sh
    source venv/bin/activate
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Indexing the Vault

To index your Obsidian vault, run:
```sh
./obsidian.py --vault /path/to/your/obsidian/vault
```

### Querying the Vault

To query your Obsidian vault, run:
```sh
./obsidian.py --query "your query here"
```
