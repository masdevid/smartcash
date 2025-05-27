#!/usr/bin/env python3
"""
SmartCash Unused Files Checker

This script analyzes the SmartCash project to identify potentially unused files
in specific directories and provides safe cleanup options.

Usage:
    python unused_files_checker.py /path/to/smartcash/project
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import defaultdict
import argparse


class UnusedFilesChecker:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.python_files = {}  # filename -> full_path
        self.imports_map = defaultdict(set)  # file -> set of imported modules
        self.references_map = defaultdict(set)  # file -> set of referenced files
        self.used_files = set()
        
        # Target directories to check for unused files
        self.target_dirs = {
            'ui_utils': 'smartcash/ui/utils',
            'ui_components': 'smartcash/ui/components',
            'ui_handlers': 'smartcash/ui/handlers',
            'dataset_utils': 'smartcash/dataset/utils',
            'dataset_in_ui': 'smartcash/dataset'  # dataset files used in ui
        }
        
    def scan_python_files(self):
        """Scan all Python files in the project."""
        print("🔍 Scanning Python files...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    full_path = Path(root) / file
                    relative_path = full_path.relative_to(self.project_root)
                    self.python_files[str(relative_path)] = full_path
        
        print(f"   Found {len(self.python_files)} Python files")
    
    def extract_imports_from_file(self, file_path: Path) -> Set[str]:
        """Extract all imports from a Python file."""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to get imports
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module)
                            # Also add submodules
                            for alias in node.names:
                                if alias.name != '*':
                                    imports.add(f"{node.module}.{alias.name}")
            except SyntaxError:
                # If AST parsing fails, fall back to regex
                pass
            
            # Also use regex for additional patterns
            import_patterns = [
                r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
                r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
                r'from\s+\.([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',  # relative imports
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                imports.update(matches)
            
            # Look for string references that might indicate file usage
            string_patterns = [
                r'["\']([a-zA-Z_][a-zA-Z0-9_.]*)["\']',  # quoted strings
            ]
            
            for pattern in string_patterns:
                matches = re.findall(pattern, content)
                imports.update(matches)
                
        except Exception as e:
            print(f"   Warning: Could not read {file_path}: {e}")
        
        return imports
    
    def analyze_imports(self):
        """Analyze imports in all Python files."""
        print("📊 Analyzing imports and references...")
        
        for rel_path, full_path in self.python_files.items():
            imports = self.extract_imports_from_file(full_path)
            self.imports_map[rel_path] = imports
        
        print(f"   Analyzed imports in {len(self.python_files)} files")
    
    def resolve_file_references(self):
        """Resolve which files are referenced by others."""
        print("🔗 Resolving file references...")
        
        for file_path, imports in self.imports_map.items():
            for import_name in imports:
                # Try to match imports to actual files
                possible_files = self.find_matching_files(import_name)
                for matched_file in possible_files:
                    self.references_map[file_path].add(matched_file)
                    self.used_files.add(matched_file)
        
        print(f"   Found references to {len(self.used_files)} files")
    
    def find_matching_files(self, import_name: str) -> List[str]:
        """Find files that match an import name."""
        matches = []
        
        # Direct module name match
        for file_path in self.python_files.keys():
            # Convert file path to module path
            module_path = file_path.replace('/', '.').replace('\\', '.').replace('.py', '')
            
            if module_path.endswith(import_name) or import_name in module_path:
                matches.append(file_path)
            
            # Check if the import matches the file name
            file_name = Path(file_path).stem
            if file_name == import_name or import_name.endswith(file_name):
                matches.append(file_path)
        
        return matches
    
    def check_unused_files_in_directory(self, dir_type: str, dir_path: str) -> List[str]:
        """Check for unused files in a specific directory."""
        full_dir_path = self.project_root / dir_path
        if not full_dir_path.exists():
            print(f"   Warning: Directory {dir_path} does not exist")
            return []
        
        unused_files = []
        
        for root, dirs, files in os.walk(full_dir_path):
            # Skip __pycache__
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    full_path = Path(root) / file
                    relative_path = str(full_path.relative_to(self.project_root))
                    
                    if relative_path not in self.used_files:
                        # Double-check by looking for any reference to the file name
                        file_stem = Path(file).stem
                        is_referenced = False
                        
                        for other_file, imports in self.imports_map.items():
                            if any(file_stem in imp for imp in imports):
                                is_referenced = True
                                break
                        
                        if not is_referenced:
                            unused_files.append(relative_path)
        
        return unused_files
    
    def check_dataset_files_in_ui(self) -> List[str]:
        """Check for dataset files that are not used in UI."""
        dataset_dir = self.project_root / "smartcash/dataset"
        ui_dir = self.project_root / "smartcash/ui"
        
        if not dataset_dir.exists() or not ui_dir.exists():
            return []
        
        # Get all dataset files
        dataset_files = []
        for root, dirs, files in os.walk(dataset_dir):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for file in files:
                if file.endswith('.py'):
                    full_path = Path(root) / file
                    relative_path = str(full_path.relative_to(self.project_root))
                    dataset_files.append(relative_path)
        
        # Check which dataset files are used in UI
        unused_in_ui = []
        
        for dataset_file in dataset_files:
            is_used_in_ui = False
            dataset_module_name = Path(dataset_file).stem
            
            # Check all UI files for references
            for ui_file, imports in self.imports_map.items():
                if ui_file.startswith('smartcash/ui/'):
                    if any(dataset_module_name in imp or 'dataset' in imp for imp in imports):
                        is_used_in_ui = True
                        break
            
            if not is_used_in_ui and dataset_file != 'smartcash/dataset/__init__.py':
                unused_in_ui.append(dataset_file)
        
        return unused_in_ui
    
    def run_analysis(self):
        """Run the complete analysis."""
        print("🚀 Starting SmartCash unused files analysis...\n")
        
        self.scan_python_files()
        self.analyze_imports()
        self.resolve_file_references()
        
        print("\n" + "="*60)
        print("📋 UNUSED FILES ANALYSIS RESULTS")
        print("="*60)
        
        all_results = {}
        
        # Check each target directory
        for dir_type, dir_path in self.target_dirs.items():
            if dir_type == 'dataset_in_ui':
                unused_files = self.check_dataset_files_in_ui()
                title = f"Dataset files unused in UI ({dir_path})"
            else:
                unused_files = self.check_unused_files_in_directory(dir_type, dir_path)
                title = f"Unused files in {dir_path}"
            
            all_results[dir_type] = unused_files
            
            print(f"\n🔍 {title}")
            print("-" * len(title))
            
            if unused_files:
                for i, file in enumerate(unused_files, 1):
                    print(f"   {i:2d}. {file}")
                print(f"\n   Total: {len(unused_files)} unused files")
            else:
                print("   ✅ No unused files found!")
        
        return all_results
    
    def confirm_and_delete(self, results: Dict[str, List[str]]):
        """Confirm and delete unused files."""
        print("\n" + "="*60)
        print("🗑️  FILE DELETION CONFIRMATION")
        print("="*60)
        
        for dir_type, unused_files in results.items():
            if not unused_files:
                continue
            
            dir_path = self.target_dirs[dir_type]
            title = f"Delete unused files from {dir_path}?"
            
            print(f"\n{title}")
            print("-" * len(title))
            
            for i, file in enumerate(unused_files, 1):
                print(f"   {i:2d}. {file}")
            
            while True:
                response = input(f"\nDelete these {len(unused_files)} files? [y/n/s(skip)]: ").lower().strip()
                
                if response == 'y':
                    deleted_count = 0
                    for file in unused_files:
                        file_path = self.project_root / file
                        try:
                            if file_path.exists():
                                file_path.unlink()
                                print(f"   ✅ Deleted: {file}")
                                deleted_count += 1
                            else:
                                print(f"   ⚠️  File not found: {file}")
                        except Exception as e:
                            print(f"   ❌ Failed to delete {file}: {e}")
                    
                    print(f"\n   🎉 Successfully deleted {deleted_count} files!")
                    break
                
                elif response == 'n':
                    print("   ⏭️  Skipping deletion for this directory.")
                    break
                
                elif response == 's':
                    print("   ⏭️  Skipping this directory.")
                    break
                
                else:
                    print("   Please enter 'y' for yes, 'n' for no, or 's' to skip.")


def main():
    parser = argparse.ArgumentParser(description='Check for unused files in SmartCash project')
    parser.add_argument('project_path', help='Path to the SmartCash project root')
    parser.add_argument('--no-delete', action='store_true', help='Only analyze, do not offer deletion')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.project_path):
        print(f"❌ Error: Project path '{args.project_path}' does not exist!")
        sys.exit(1)
    
    checker = UnusedFilesChecker(args.project_path)
    results = checker.run_analysis()
    
    # Summary
    total_unused = sum(len(files) for files in results.values())
    print(f"\n🎯 SUMMARY: Found {total_unused} potentially unused files across all directories.")
    
    if not args.no_delete and total_unused > 0:
        print("\n⚠️  WARNING: Please review the results carefully before deletion!")
        print("   Make sure to backup your project or use version control.")
        
        proceed = input("\nProceed with deletion confirmation? [y/n]: ").lower().strip()
        if proceed == 'y':
            checker.confirm_and_delete(results)
        else:
            print("   Operation cancelled. Files were not deleted.")
    
    print("\n✨ Analysis complete!")


if __name__ == "__main__":
    main()
