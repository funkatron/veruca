# CLI End-to-End Tests

This directory contains end-to-end test scripts for Veruca's command-line interface. These tests verify the actual CLI functionality by running real commands against your system.

## Test Scripts

- `basic_test.sh`: Tests core functionality including:
  - Ollama server management
  - Basic queries
  - Filtered queries
  - Vault indexing

- `advanced_test.sh`: Tests edge cases and error handling including:
  - Invalid paths
  - Multiple filters
  - Different models
  - Complex queries

## Running the Tests

1. Make sure you have a test Obsidian vault available
2. Ensure Ollama is installed and the required models are pulled
3. Run the tests:

```bash
# Make scripts executable
chmod +x tests/cli/*.sh

# Run basic tests
./tests/cli/basic_test.sh

# Run advanced tests
./tests/cli/advanced_test.sh
```

## Notes

- These tests require a real Obsidian vault and Ollama installation
- They are separate from the Python unit tests in the main `tests` directory
- These tests verify the actual CLI behavior rather than individual components
- Some tests may take longer to run as they involve actual model inference

## Customizing Tests

You can modify the test scripts to:
- Change the vault path (default: ~/Obsidian)
- Use different models
- Add or remove test cases
- Adjust filter parameters

## Troubleshooting

If tests fail:
1. Check if Ollama is running (`veruca ollama status`)
2. Verify your Obsidian vault path is correct
3. Ensure you have the required models pulled
4. Check the test output for specific error messages