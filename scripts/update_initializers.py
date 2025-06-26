"""
Script to update all initializers to use the new widget_utils.

This script will:
1. Find all initializer files in the project
2. Update them to use the new widget_utils.safe_display function
3. Ensure consistent widget display behavior across the codebase
"""

import os
import re
from pathlib import Path

def update_initializer_file(file_path: str) -> bool:
    """Update a single initializer file to use widget_utils."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the file needs updating
    if 'from IPython.display import display' not in content:
        return False
        
    if 'if display_ui' in content and 'display(' in content:
        # This is a candidate for updating
        print(f"Updating {file_path}...")
        
        # Add import for safe_display
        if 'from smartcash.ui.utils import safe_display' not in content:
            content = content.replace(
                'from IPython.display import display',
                'from smartcash.ui.utils import safe_display\nfrom IPython.display import display'
            )
        
        # Replace display patterns
        patterns = [
            (
                r'if\s+display_ui\s+and\s+result\s*:(?:\s*\n\s*)?(?:from\s+IPython\.display\s+import\s+display\s*\n\s*)?(?:if\s+isinstance\(result\s*,\s*dict\)\s+and\s+["\']ui["\']\s+in\s+result\s*:.*?display\(result\[["\']ui["\']\]\)\s*\n\s*)?(?:else:.*?display\(result\)\s*\n\s*)?',
                'safe_display(result, condition=display_ui)\n'
            ),
            (
                r'if\s+result\s+and\s+isinstance\(result\s*,\s*dict\)\s+and\s+["\']ui["\']\s+in\s+result\s*:.*?display\(result\[["\']ui["\']\]\)',
                'safe_safe_safe_safe_display(result)'
            ),
            (
                r'display\(result\)',
                'safe_safe_safe_safe_display(result)'
            )
        ]
        
        for pattern, replacement in patterns:
            content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
            if count > 0:
                break
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return True
    
    return False

def main():
    """Main function to update all initializers."""
    project_root = Path(__file__).parent.parent
    initializer_files = list(project_root.glob('**/*initializer*.py'))
    
    updated_files = 0
    
    for file_path in initializer_files:
        if str(file_path).endswith('__init__.py') or 'test' in str(file_path).lower():
            continue
            
        try:
            if update_initializer_file(str(file_path)):
                updated_files += 1
                print(f"✅ Updated {file_path.relative_to(project_root)}")
        except Exception as e:
            print(f"❌ Error updating {file_path}: {str(e)}")
    
    print(f"\n✅ Updated {updated_files} initializer files")

if __name__ == '__main__':
    main()
