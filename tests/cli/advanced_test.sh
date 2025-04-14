#!/bin/bash

# Advanced CLI Test Script
# This script tests edge cases and error handling of Veruca's CLI

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
priority: high
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
priority: medium
---

# Team Meeting Notes

Discussion topics:
1. Project timeline: Q2 2024 launch
2. Resource allocation: 2 developers, 1 designer
3. Next steps: Begin API design
EOL

echo "=== Starting Veruca CLI Advanced Tests ==="

# Test 1: Invalid Vault Path
echo "Test 1: Testing Invalid Vault Path..."
if veruca query "test" --vault-path /nonexistent/path 2>/dev/null; then
    echo "❌ Invalid vault path test failed"
    exit 1
else
    echo "✅ Invalid vault path test passed"
fi

# Test 2: Multiple Filters
echo "Test 2: Testing Multiple Filters..."
RESULT=$(veruca query "What is the priority?" --filter "tags=python,status=active" --vault-path "$TEST_VAULT")
if [[ ! "$RESULT" =~ "high" ]]; then
    echo "Error: Expected priority 'high' in response"
    exit 1
fi
echo "✅ Multiple filters test passed"

# Test 3: Different Model
echo "Test 3: Testing Different Model..."
veruca query "test" --vault-path "$TEST_VAULT"
echo "✅ Different model test passed"

# Test 4: Complex Query
echo "Test 4: Testing Complex Query..."
RESULT=$(veruca query "What are the key points from my meeting notes about the project timeline?" --filter "tags=meeting" --vault-path "$TEST_VAULT")
if [[ ! "$RESULT" =~ "Q2 2024" ]]; then
    echo "Error: Expected project timeline in response"
    exit 1
fi
echo "✅ Complex query test passed"

# Test 5: Index with Different Model
echo "Test 5: Testing Index with Different Model..."
veruca index --model nomic-embed-text --vault-path "$TEST_VAULT"
echo "✅ Index with different model test passed"

# Clean up
rm -rf "$TEST_VAULT"

echo "=== All Advanced Tests Passed! ==="