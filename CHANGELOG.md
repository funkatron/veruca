# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2024-04-14

### Changed
- Improved test organization and documentation
- Renamed test files to better reflect their purpose
- Cleaned up test code and removed duplicate implementations

## [0.3.1] - 2024-04-14

### Changed
- Restructured project into proper Python package with sources module
- Improved code organization and maintainability
- Enhanced test coverage and organization

## [0.3.0] - 2024-04-14

### Added
- New `DataSource` abstract base class for implementing different data sources
- Improved type hints throughout the codebase
- Better error handling with specific exception types
- Constants for default configuration values

### Changed
- Restructured code into core and sources packages
- Improved path handling using `pathlib.Path` consistently
- Enhanced docstrings with `:param`, `:return`, and `:raises` format
- Simplified and cleaned up code organization
- Improved error messages with more context

### Fixed
- Type inconsistencies in document loading
- Path handling inconsistencies
- Duplicate code in query and indexing
- Unused methods and redundant code

## [0.2.0] - 2024-03-17

### Added
- Support for filtering query results by metadata
- Improved error handling and user feedback
- Better documentation and examples

### Changed
- Updated dependencies to latest versions
- Improved code organization and structure
- Enhanced type hints and documentation

### Fixed
- Resource warnings from Ollama clients
- Various minor bugs and issues

## [0.1.0] - 2024-03-11

### Added
- Initial release
- Basic Obsidian vault querying functionality
- Support for local LLMs via Ollama
- Document indexing and vector search
- Command-line interface

### Notes
- This is an alpha release. The API and features are not yet stable and may change in future versions.
- Feedback and bug reports are welcome.