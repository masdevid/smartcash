#!/bin/bash

# Script to delete Python test files
# This script removes common Python test file patterns

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to confirm deletion
confirm_deletion() {
    local files_count=$1
    echo -e "${YELLOW}Found $files_count Python test files.${NC}"
    read -p "Do you want to delete them? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Operation cancelled."
        exit 0
    fi
}

# Function to delete files matching a pattern
delete_pattern() {
    local pattern=$1
    local description=$2
    local count=0
    
    print_info "Searching for $description..."
    
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]]; then
            echo "  Deleting: $file"
            rm "$file"
            ((count++))
        fi
    done < <(find . -name "$pattern" -type f -print0 2>/dev/null)
    
    if [[ $count -gt 0 ]]; then
        print_info "Deleted $count $description"
    fi
    
    return $count
}

# Main execution
main() {
    print_info "Python Test File Cleanup Script"
    print_info "==============================="
    
    # Check if we're in a directory with Python files
    if ! find . -name "*.py" -type f -quit >/dev/null 2>&1; then
        print_warning "No Python files found in current directory and subdirectories."
        print_warning "Are you sure you're in the right location?"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
    
    # Count total test files first
    total_files=0
    
    # Count different test file patterns
    total_files=$((total_files + $(find . -name "test_*.py" -type f 2>/dev/null | wc -l)))
    total_files=$((total_files + $(find . -name "*_test.py" -type f 2>/dev/null | wc -l)))
    total_files=$((total_files + $(find . -name "test*.py" -type f 2>/dev/null | wc -l)))
    total_files=$((total_files + $(find . -path "*/test/*" -name "*.py" -type f 2>/dev/null | wc -l)))
    total_files=$((total_files + $(find . -path "*/tests/*" -name "*.py" -type f 2>/dev/null | wc -l)))
    
    if [[ $total_files -eq 0 ]]; then
        print_info "No Python test files found."
        exit 0
    fi
    
    # Show what will be deleted and confirm
    print_info "The following patterns will be searched and deleted:"
    echo "  - test_*.py (files starting with 'test_')"
    echo "  - *_test.py (files ending with '_test')"
    echo "  - test*.py (files starting with 'test')"
    echo "  - Files in 'test/' directories"
    echo "  - Files in 'tests/' directories"
    echo
    
    confirm_deletion $total_files
    
    # Delete files by pattern
    deleted_count=0
    
    delete_pattern "test_*.py" "files starting with 'test_'"
    deleted_count=$((deleted_count + $?))
    
    delete_pattern "*_test.py" "files ending with '_test'"
    deleted_count=$((deleted_count + $?))
    
    delete_pattern "test*.py" "files starting with 'test'"
    deleted_count=$((deleted_count + $?))
    
    # Delete files in test directories
    print_info "Searching for files in test directories..."
    test_dir_count=0
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]]; then
            echo "  Deleting: $file"
            rm "$file"
            ((test_dir_count++))
        fi
    done < <(find . -path "*/test/*" -name "*.py" -type f -print0 2>/dev/null)
    
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]]; then
            echo "  Deleting: $file"
            rm "$file"
            ((test_dir_count++))
        fi
    done < <(find . -path "*/tests/*" -name "*.py" -type f -print0 2>/dev/null)
    
    if [[ $test_dir_count -gt 0 ]]; then
        print_info "Deleted $test_dir_count files from test directories"
    fi
    
    deleted_count=$((deleted_count + test_dir_count))
    
    # Summary
    echo
    print_info "==============================="
    if [[ $deleted_count -gt 0 ]]; then
        print_info "Successfully deleted $deleted_count Python test files!"
    else
        print_info "No files were deleted."
    fi
}

# Run main function
main "$@"