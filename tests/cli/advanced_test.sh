#!/bin/bash

# Advanced CLI Test Script
# This script tests edge cases and error handling of Veruca's CLI

set -e  # Exit on error

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
veruca query "test" --filter "tags=python,status=active" --vault-path ~/Obsidian
echo "✅ Multiple filters test passed"

# Test 3: Different Model
echo "Test 3: Testing Different Model..."
veruca query "test" --model llama2 --vault-path ~/Obsidian
echo "✅ Different model test passed"

# Test 4: Complex Query
echo "Test 4: Testing Complex Query..."
veruca query "What are the key points from my meeting notes about the project timeline?" --filter "tags=meeting" --vault-path ~/Obsidian
echo "✅ Complex query test passed"

# Test 5: Index with Different Model
echo "Test 5: Testing Index with Different Model..."
veruca index --model nomic-embed-text --vault-path ~/Obsidian
echo "✅ Index with different model test passed"

echo "=== All Advanced Tests Passed! ==="