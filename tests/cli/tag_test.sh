#!/bin/bash

# Tag Test Script
# This script specifically tests tag insertion and querying functionality

set -e  # Exit on error

# Create a temporary test vault
TEST_VAULT="/tmp/veruca_tag_test_vault"
rm -rf "$TEST_VAULT"
mkdir -p "$TEST_VAULT"

# Create a test note with specific tags
cat > "$TEST_VAULT/tag_test.md" << EOL
---
tags: test-tag,unit-test,verification
status: active
priority: 1
---

# Tag Test Document

This document is specifically created for testing tag functionality.
It contains the following tags in its metadata:
- test-tag
- unit-test
- verification
EOL

echo "=== Starting Tag Test ==="

# Test 1: Check Ollama Server Status
echo "Test 1: Check Ollama Server Status"
veruca ollama status || (echo "Error: Ollama server not running" && exit 1)

# Test 2: Index Vault
echo "Test 2: Index Vault"
veruca index --vault-path "$TEST_VAULT" || (echo "Error: Failed to index vault" && exit 1)
echo "✅ Vault indexing test passed"

# Test 3: Query with Exact Tag Match
echo "Test 3: Query with Exact Tag Match"
RESULT=$(veruca query "What tags are present in the metadata?" --vault-path "$TEST_VAULT" --filter "tags=test-tag")
echo "DEBUG - Test 3 Response:"
echo "$RESULT"
if [[ ! "$RESULT" =~ "test-tag" ]]; then
    echo "Error: Expected 'test-tag' in response"
    exit 1
fi
echo "✅ Exact tag match test passed"

# Test 4: Query with $in Operator
echo "Test 4: Query with $in Operator"
RESULT=$(veruca query "What tags are present in the metadata?" --vault-path "$TEST_VAULT" --filter "tags:in=test-tag,unit-test")
echo "DEBUG - Test 4 Response:"
echo "$RESULT"
if [[ ! "$RESULT" =~ "test-tag" ]] || [[ ! "$RESULT" =~ "unit-test" ]]; then
    echo "Error: Expected both 'test-tag' and 'unit-test' in response"
    exit 1
fi
echo "✅ $in operator test passed"

# Test 5: Query with $nin Operator
echo "Test 5: Query with $nin Operator"
RESULT=$(veruca query "What tags are present in the metadata?" --vault-path "$TEST_VAULT" --filter "tags:nin=archived,old")
echo "DEBUG - Test 5 Response:"
echo "$RESULT"
if [[ "$RESULT" =~ "archived" ]] || [[ "$RESULT" =~ "old" ]]; then
    echo "Error: Expected no 'archived' or 'old' tags in response"
    exit 1
fi
echo "✅ $nin operator test passed"

# Test 6: Combined Tag Filters
echo "Test 6: Combined Tag Filters"
RESULT=$(veruca query "What tags are present in the metadata?" --vault-path "$TEST_VAULT" --filter "tags:in=test-tag,unit-test" --filter "tags:nin=archived,old")
echo "DEBUG - Test 6 Response:"
echo "$RESULT"
if [[ ! "$RESULT" =~ "test-tag" ]] || [[ ! "$RESULT" =~ "unit-test" ]] || [[ "$RESULT" =~ "archived" ]] || [[ "$RESULT" =~ "old" ]]; then
    echo "Error: Combined tag filters did not work as expected"
    exit 1
fi
echo "✅ Combined tag filters test passed"

# Clean up
rm -rf "$TEST_VAULT"

echo "=== All Tag Tests Passed! ==="