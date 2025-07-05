#!/usr/bin/env python3
"""
Script to automatically update imports from deprecated modules to their new locations.

This script updates Python files to use the new module structure:
- smartcash.ui.handlers.* -> smartcash.ui.core.*
- smartcash.ui.initializers.* -> smartcash.ui.core.initializers.*
- smartcash.ui.components.error.error_component -> smartcash.ui.core.errors.error_component
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Mapping of old import paths to new import paths
IMPORT_MAPPINGS = {
    # Error handler imports
    'smartcash\.ui\.handlers\.error_handler': 'smartcash.ui.core.errors.handlers',
    'smartcash\.ui\.handlers\.base_handler': 'smartcash.ui.core.handlers',
    'smartcash\.ui\.handlers\.config_handlers': 'smartcash.ui.core.config',
    
    # Initializer imports
    'smartcash\.ui\.initializers': 'smartcash.ui.core.initializers',
    
    # Error component imports
    'smartcash\.ui\.components\.error\.error_component': 'smartcash.ui.core.errors.error_component',
}

# Specific import renamings for when we need to be more precise
SPECIFIC_IMPORTS = {
    # Error handler specific imports
    'from\s+smartcash\.ui\.handlers\.error_handler\s+import\s+': 
        'from smartcash.ui.core.errors.handlers import ',
    'import\s+smartcash\.ui\.handlers\.error_handler\s+as': 
        'import smartcash.ui.core.errors.handlers as',
    
    # Base handler specific imports
    'from\s+smartcash\.ui\.handlers\.base_handler\s+import\s+': 
        'from smartcash.ui.core.handlers import ',
    'import\s+smartcash\.ui\.handlers\.base_handler\s+as': 
        'import smartcash.ui.core.handlers as',
    
    # Config handler specific imports
    'from\s+smartcash\.ui\.handlers\.config_handlers\s+import\s+': 
        'from smartcash.ui.core.config import ',
    'import\s+smartcash\.ui\.handlers\.config_handlers\s+as': 
        'import smartcash.ui.core.config as',
    
    # Initializer specific imports
    'from\s+smartcash\.ui\.initializers\s+import\s+': 
        'from smartcash.ui.core.initializers import ',
    'import\s+smartcash\.ui\.initializers\s+as': 
        'import smartcash.ui.core.initializers as',
    
    # Error component specific imports
    'from\s+smartcash\.ui\.components\.error\.error_component\s+import\s+': 
        'from smartcash.ui.core.errors.error_component import ',
    'import\s+smartcash\.ui\.components\.error\.error_component\s+as': 
        'import smartcash.ui.core.errors.error_component as',
}

def update_file_imports(file_path: Path) -> Tuple[bool, List[str]]:
    """Update imports in a single file.
    
    Args:
        file_path: Path to the file to update
        
    Returns:
        Tuple of (file_was_modified, list_of_changes_made)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Skipping non-text file: {file_path}")
        return False, []
    
    original_content = content
    changes = []
    
    # First handle specific import patterns
    for old_pattern, new_pattern in SPECIFIC_IMPORTS.items():
        if re.search(old_pattern, content):
            new_content, count = re.subn(old_pattern, new_pattern, content)
            if count > 0:
                changes.append(f"Updated '{old_pattern}' to '{new_pattern}'")
                content = new_content
    
    # Then handle general module path updates
    for old_path, new_path in IMPORT_MAPPINGS.items():
        # Handle 'from x import y' style imports
        pattern = f'from\s+{old_path}(\s+import\s+)'
        if re.search(pattern, content):
            new_content, count = re.subn(
                pattern, 
                f'from {new_path}\1', 
                content
            )
            if count > 0:
                changes.append(f"Updated 'from {old_path} import' to 'from {new_path} import'")
                content = new_content
        
        # Handle 'import x.y as z' style imports
        pattern = f'import\s+{old_path}(\s+as\s+\w+|\s*\n|\s*$|\s*#|\s*;)'
        if re.search(pattern, content):
            new_content, count = re.subn(
                pattern, 
                f'import {new_path}\1', 
                content
            )
            if count > 0:
                changes.append(f"Updated 'import {old_path}' to 'import {new_path}'")
                content = new_content
    
    # Only write the file if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes
    
    return False, changes

def find_python_files(directory: str) -> List[Path]:
    """Find all Python files in the given directory, excluding virtual environments."""
    python_files = []
    for root, _, files in os.walk(directory):
        # Skip virtual environments and other directories
        if any(part.startswith(('.', '_', 'venv', 'env', 'build', 'dist')) for part in root.split(os.sep)):
            continue
            
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    return python_files

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    print(f"Updating imports in project: {project_root}")
    
    # Find all Python files in the project
    python_files = find_python_files(project_root)
    print(f"Found {len(python_files)} Python files to check")
    
    # Track statistics
    files_updated = 0
    total_changes = 0
    
    # Update imports in each file
    for file_path in python_files:
        relative_path = file_path.relative_to(project_root)
        try:
            updated, changes = update_file_imports(file_path)
            if updated:
                files_updated += 1
                total_changes += len(changes)
                print(f"\nUpdated {relative_path}:")
                for change in changes:
                    print(f"  - {change}")
        except Exception as e:
            print(f"\nError processing {relative_path}: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print(f"Update complete!")
    print(f"Files updated: {files_updated}/{len(python_files)}")
    print(f"Total changes made: {total_changes}")
    print("\nPlease review the changes and run tests to ensure everything works as expected.")
    print("See MIGRATION_GUIDE.md for more information on the changes.")

if __name__ == "__main__":
    main()
