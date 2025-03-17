# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-03-17

### Added
- Alpha release of Veruca - Obsidian Vault Query Tool
- Support for processing Obsidian vault files recursively
- Obsidian-specific feature support:
  - Internal links (`[[filename]]` and `[[filename|display text]]`)
  - Frontmatter (YAML metadata)
  - Tags (`#tag` and nested tags `#tag/subtag`)
  - Callouts (admonitions)
- Local embedding generation using nomic-embed-text model
- Natural language querying with tag filtering
- Persistent storage of embeddings using ChromaDB
- Comprehensive test suite with 66+ test cases
- Ollama server management commands
- User-friendly error messages and documentation

### Notes
- This is an alpha release. The API and features are not yet stable and may change in future versions.
- Feedback and bug reports are welcome.