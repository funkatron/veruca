#!/bin/bash

# Basic CLI Test Script
# This script tests the basic functionality of Veruca's CLI

set -e  # Exit on error

# Create a temporary test vault
TEST_VAULT="/tmp/veruca_test_vault"
rm -rf "$TEST_VAULT"
mkdir -p "$TEST_VAULT"

# Create test notes
cat > "$TEST_VAULT/test1.md" << EOL
---
tags: python, programming
status: active
---

# Python Programming Notes

Python is a high-level programming language known for its simplicity and readability.
Key features include:
- Dynamic typing
- Extensive standard library
- Rich ecosystem of packages
EOL

cat > "$TEST_VAULT/test2.md" << EOL
---
tags: meeting, notes
status: draft
---

# Team Meeting Notes

Discussion topics:
1. Project timeline
2. Resource allocation
3. Next steps
EOL

echo "=== Starting Veruca CLI Basic Tests ==="

# Test 1: Check Ollama Server Status
echo "Test 1: Check Ollama Server Status"
veruca ollama status || (echo "Error: Ollama server not running" && exit 1)

# Test 2: Check and Pull Required Models
echo "Test 2: Check and Pull Required Models"
# Check if mistral:latest is available, pull if not
if ! ollama list | grep -q "mistral:latest"; then
    echo "Pulling mistral:latest model..."
    ollama pull mistral:latest
fi

# Check if nomic-embed-text is available, pull if not
if ! ollama list | grep -q "nomic-embed-text"; then
    echo "Pulling nomic-embed-text model..."
    ollama pull nomic-embed-text
fi

echo "✅ Required models are available"

# Test 3: Index Vault
echo "Test 3: Index Vault"
veruca index --vault-path "$TEST_VAULT" || (echo "Error: Failed to index vault" && exit 1)
echo "✅ Vault indexing test passed"

# Test 4: Query Vault
echo "Test 4: Query Vault"
veruca query "What programming language is discussed?" --vault-path "$TEST_VAULT" || (echo "Error: Query failed" && exit 1)
echo "✅ Basic query test passed"

# Test 5: Query with Filters
echo "Test 5: Query with Filters"
RESULT=$(veruca query "Look at ONLY the metadata section of the documents. What is the value of the status field?" --vault-path "$TEST_VAULT" --filter "status=active")
echo "DEBUG - Query Response:"
echo "$RESULT"
if [[ ! "$RESULT" =~ "metadata" ]] || [[ ! "$RESULT" =~ "active" ]]; then
    echo "Error: Expected metadata with status 'active' in response"
    exit 1
fi
echo "✅ Filtered query test passed"

# Clean up
rm -rf "$TEST_VAULT"

echo "=== All Basic Tests Passed! ==="