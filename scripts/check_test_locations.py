#!/usr/bin/env python3
"""
Check that test files are in the correct locations.

This script ensures that all test files follow the project's test directory structure.
"""
import os
import sys
from pathlib import Path

# Allowed test directories
ALLOWED_TEST_DIRS = {
    "tests/unit",
    "tests/integration",
    # Add any other allowed test directories here
}

def is_test_file(path: str) -> bool:
    """Check if a file is a test file."""
    filename = os.path.basename(path)
    return (filename.endswith("_test.py") or 
            (filename.startswith("test_") and filename.endswith(".py")))

def should_skip_directory(path: str) -> bool:
    """Check if a directory should be skipped during traversal."""
    # Skip hidden directories
    if any(part.startswith('.') for part in Path(path).parts if part not in ('.', '..')):
        return True
        
    # Skip common non-source directories
    skip_dirs = {
        '__pycache__', '.pytest_cache', '.git', 'venv', 'env', '.venv',
        'node_modules', 'build', 'dist', 'logs', '.mypy_cache', '.tox'
    }
    
    return any(part in skip_dirs for part in Path(path).parts)

def check_test_locations() -> int:
    """Check that all test files are in allowed locations."""
    project_root = Path(__file__).parent.parent
    test_root = project_root / "tests"
    
    errors = []
    
    for root, dirs, files in os.walk(project_root, topdown=True):
        # Skip unwanted directories
        if should_skip_directory(root):
            dirs[:] = []  # Don't traverse into subdirectories
            continue
            
        relative_path = os.path.relpath(root, project_root)
        
        # Skip if we're in an allowed test directory
        if any(relative_path.startswith(allowed) for allowed in ALLOWED_TEST_DIRS):
            continue
            
        # Check files in this directory
        for file in files:
            if is_test_file(file):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, project_root)
                
                # Skip files in virtual environments
                if 'site-packages' in full_path or 'dist-packages' in full_path:
                    continue
                    
                # Skip log files
                if file.endswith('.log'):
                    continue
                    
                errors.append(f"Test file in wrong location: {rel_path}")
    
    if errors:
        print("Error: Test files found in incorrect locations:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print("\nAll test files must be in one of these directories:", file=sys.stderr)
        for allowed in sorted(ALLOWED_TEST_DIRS):
            print(f"  - {allowed}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(check_test_locations())
