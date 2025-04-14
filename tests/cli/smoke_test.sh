#!/bin/bash

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Error: Ollama server is not running"
    exit 1
fi

# Check if required models are available
if ! curl -s http://localhost:11434/api/tags | grep -q "llama2"; then
    echo "Error: llama2 model not found"
    exit 1
fi

# Index the test vault
echo "Indexing test vault..."
veruca index --vault-path tests/cli/test_vault

# Test 1: List all documents
echo -e "\nTest 1: List all documents"
veruca query "List all document titles" --vault-path tests/cli/test_vault

# Test 2: Find documents with specific tag
echo -e "\nTest 2: Find documents with python tag"
veruca query "List documents with python tag" --vault-path tests/cli/test_vault --filter "tags=python"

# Test 3: Find documents with specific title
echo -e "\nTest 3: Find meeting notes"
veruca query "Find document titled 'Team Meeting Notes'" --vault-path tests/cli/test_vault

echo -e "\nSmoke test completed"