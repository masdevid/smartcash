#!/bin/bash
# Script to remove all test files except test_imports.py

# Find all test files and remove them, except test_imports.py
find /Users/masdevid/Projects/smartcash -name "test_*.py" | grep -v "test_imports.py" | xargs rm -f

echo "All test files except test_imports.py have been removed."
