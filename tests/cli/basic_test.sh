#!/bin/bash

# Basic CLI Test Script
# This script tests the basic functionality of Veruca's CLI

set -e  # Exit on error

# Create a temporary test vault
TEST_VAULT="/tmp/veruca_test_vault"
rm -rf "$TEST_VAULT"
mkdir -p "$TEST_VAULT"

echo "=== Starting Veruca CLI Basic Tests ==="

# Test 0: Check Empty Vault
echo "Test 0: Check Empty Vault"
RESULT=$(veruca query "What documents are available?" --vault-path "$TEST_VAULT" 2>&1) || true
if [[ ! "$RESULT" =~ "No documents found" ]]; then
    echo "Error: Expected 'No documents found' for empty vault"
    exit 1
fi
echo "✅ Empty vault test passed"

# Create test notes
cat > "$TEST_VAULT/test1.md" << EOL
---
tags: python,programming
status: active
priority: 3
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
tags: meeting,notes
status: draft
priority: 1
---

# Team Meeting Notes

Discussion topics:
1. Project timeline
2. Resource allocation
3. Next steps
EOL

cat > "$TEST_VAULT/test3.md" << EOL
---
tags: archived,old
status: archived
priority: 5
---

# Archived Notes

These are old notes that have been archived.
EOL

# Test 1: Check Ollama Server Status
echo "Test 1: Check Ollama Server Status"
veruca ollama status || (echo "Error: Ollama server not running" && exit 1)

# Test 2: Check and Pull Required Models
echo "Test 2: Check and Pull Required Models"
# Check if llama2:latest is available, pull if not
if ! ollama list | grep -q "llama2:latest"; then
    echo "Pulling llama2:latest model..."
    ollama pull llama2:latest
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

# Test 5: Simple Equality Filter
echo "Test 5: Simple Equality Filter"
RESULT=$(veruca query "What is the status?" --vault-path "$TEST_VAULT" --filter "status=active")
if [[ ! "$RESULT" =~ "active" ]]; then
    echo "Error: Expected status 'active' in response"
    exit 1
fi
echo "✅ Simple equality filter test passed"

# Test 6: $in Operator
echo "Test 6: $in Operator"
RESULT=$(veruca query "Look at ONLY the metadata section. What tags are present?" --vault-path "$TEST_VAULT" --filter "tags:in=python,meeting")
echo "DEBUG - Test 6 Response:"
echo "$RESULT"
if [[ ! "$RESULT" =~ "python" ]] && [[ ! "$RESULT" =~ "meeting" ]]; then
    echo "Error: Expected either 'python' or 'meeting' tags in response"
    exit 1
fi
echo "✅ $in operator test passed"

# Test 7: $nin Operator
echo "Test 7: $nin Operator"
RESULT=$(veruca query "Look at ONLY the metadata section. What tags are present?" --vault-path "$TEST_VAULT" --filter "tags:nin=archived,old")
if [[ "$RESULT" =~ "archived" ]] || [[ "$RESULT" =~ "old" ]]; then
    echo "Error: Expected no 'archived' or 'old' tags in response"
    exit 1
fi
echo "✅ $nin operator test passed"

# Test 8: $ne Operator
echo "Test 8: $ne Operator"
RESULT=$(veruca query "What is the status?" --vault-path "$TEST_VAULT" --filter "status:ne=draft")
if [[ "$RESULT" =~ "draft" ]]; then
    echo "Error: Expected status not equal to 'draft'"
    exit 1
fi
echo "✅ $ne operator test passed"

# Test 9: Numeric Comparison Operators
echo "Test 9: Numeric Comparison Operators"
# Test $gt
RESULT=$(veruca query "What is the priority?" --vault-path "$TEST_VAULT" --filter "priority:gt=2")
if [[ ! "$RESULT" =~ "3" ]] && [[ ! "$RESULT" =~ "5" ]]; then
    echo "Error: Expected priority greater than 2"
    exit 1
fi

# Test $gte
RESULT=$(veruca query "What is the priority?" --vault-path "$TEST_VAULT" --filter "priority:gte=3")
if [[ ! "$RESULT" =~ "3" ]] && [[ ! "$RESULT" =~ "5" ]]; then
    echo "Error: Expected priority greater than or equal to 3"
    exit 1
fi

# Test $lt
RESULT=$(veruca query "What is the priority?" --vault-path "$TEST_VAULT" --filter "priority:lt=3")
if [[ ! "$RESULT" =~ "1" ]]; then
    echo "Error: Expected priority less than 3"
    exit 1
fi

# Test $lte
RESULT=$(veruca query "What is the priority?" --vault-path "$TEST_VAULT" --filter "priority:lte=3")
if [[ ! "$RESULT" =~ "1" ]] && [[ ! "$RESULT" =~ "3" ]]; then
    echo "Error: Expected priority less than or equal to 3"
    exit 1
fi
echo "✅ Numeric comparison operators test passed"

# Test 10: Combined Filters
echo "Test 10: Combined Filters"
RESULT=$(veruca query "What is the content?" --vault-path "$TEST_VAULT" --filter "status:ne=draft" --filter "priority:gt=2" --filter "tags:nin=archived,old")
if [[ "$RESULT" =~ "draft" ]] || [[ "$RESULT" =~ "archived" ]] || [[ "$RESULT" =~ "old" ]]; then
    echo "Error: Combined filters did not work as expected"
    exit 1
fi
echo "✅ Combined filters test passed"

# Clean up
rm -rf "$TEST_VAULT"

echo "=== All Basic Tests Passed! ==="