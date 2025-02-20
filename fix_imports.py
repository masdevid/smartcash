#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def fix_imports(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix imports
    content = content.replace('from smartcash.utils.', 'from smartcash.utils.')
    content = content.replace('from smartcash.handlers.', 'from smartcash.handlers.')
    content = content.replace('from smartcash.models.', 'from smartcash.models.')
    
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    project_root = Path(__file__).parent
    
    # Find all Python files
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Fixing imports in {file_path}")
                fix_imports(file_path)

if __name__ == '__main__':
    main()
    print("Done fixing imports!")
