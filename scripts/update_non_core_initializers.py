"""
Script to update all initializers outside of smartcash/ui/initializers
"""

import re
from pathlib import Path

def update_file(file_path: Path) -> bool:
    """Update a single file to use widget_utils.safe_display."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if already updated or doesn't need updating
        if 'from smartcash.ui.utils import safe_display' in content:
            return False
            
        if 'from IPython.display import display' not in content:
            return False
            
        # Add import for safe_display
        content = content.replace(
            'from IPython.display import display',
            'from smartcash.ui.utils import safe_display\nfrom IPython.display import display'
        )
        
        # Replace display patterns
        patterns = [
            # Pattern for if display_ui and result: display(result['ui'] if dict else result)
            (
                r'if\s+display_ui\s+and\s+result\s*:(?:\s*\n\s*)?(?:from\s+IPython\.display\s+import\s+display\s*\n\s*)?(?:if\s+isinstance\(result\s*,\s*dict\)\s+and\s+["\']ui["\']\s+in\s+result\s*:.*?display\(result\[["\']ui["\']\]\)\s*\n\s*)?(?:else:.*?display\(result\)\s*\n\s*)?',
                'safe_display(result, condition=display_ui)\n'
            ),
            # Pattern for if result and 'ui' in result: display(result['ui'])
            (
                r'if\s+result\s+and\s+isinstance\(result\s*,\s*dict\)\s+and\s+["\']ui["\']\s+in\s+result\s*:.*?display\(result\[["\']ui["\']\]\)',
                'safe_display(result)'
            ),
            # Simple display(result) pattern
            (
                r'display\(result\)',
                'safe_display(result)'
            )
        ]
        
        updated = False
        for pattern, replacement in patterns:
            new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
            if count > 0:
                content = new_content
                updated = True
                break
        
        if updated:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
            
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
    
    return False

def main():
    """Update all initializer files outside of smartcash/ui/initializers."""
    project_root = Path(__file__).parent.parent
    initializer_files = [
        project_root / 'smartcash' / 'ui' / 'dataset' / 'augmentation' / 'augmentation_initializer.py',
        project_root / 'smartcash' / 'ui' / 'dataset' / 'downloader' / 'downloader_initializer.py',
        project_root / 'smartcash' / 'ui' / 'dataset' / 'preprocessing' / 'preprocessing_initializer.py',
        project_root / 'smartcash' / 'ui' / 'evaluation' / 'evaluation_initializer.py',
        project_root / 'smartcash' / 'ui' / 'setup' / 'dependency' / 'dependency_initializer.py',
        project_root / 'smartcash' / 'ui' / 'setup' / 'env_config' / 'env_config_initializer.py',
    ]
    
    updated_count = 0
    for file_path in initializer_files:
        if file_path.exists():
            if update_file(file_path):
                print(f"✅ Updated {file_path.relative_to(project_root)}")
                updated_count += 1
            else:
                print(f"ℹ️  No changes needed for {file_path.relative_to(project_root)}")
        else:
            print(f"⚠️  File not found: {file_path.relative_to(project_root)}")
    
    print(f"\n✅ Updated {updated_count} files")

if __name__ == '__main__':
    main()
