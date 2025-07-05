"""Script to help reorganize test files."""
import os
import shutil
from pathlib import Path

def update_file_imports(file_path, old_import_path, new_import_path):
    """Update import paths in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    updated_content = content.replace(old_import_path, new_import_path)
    
    if content != updated_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Updated imports in {file_path}")

def move_test_file(source, destination):
    """Move a test file and update its imports if needed."""
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Move the file
    shutil.move(source, destination)
    print(f"Moved {source} to {destination}")
    
    # Update imports in the moved file
    return destination

def main():
    # Define the test reorganization mapping
    test_reorg = [
        # Logger tests
        {
            'source': 'tests/unit/ui/components/test_log_accordion.py',
            'destination': 'tests/unit/ui/components/log_accordion/test_log_accordion.py',
            'import_updates': []
        },
        {
            'source': 'tests/unit/ui/components/test_log_accordion_uniqueness.py',
            'destination': 'tests/unit/ui/components/log_accordion/test_uniqueness.py',
            'import_updates': [
                ('test_log_accordion', 'tests.unit.ui.components.log_accordion.test_log_accordion')
            ]
        },
        {
            'source': 'tests/unit/ui/components/test_footer_container_logs.py',
            'destination': 'tests/unit/ui/components/test_footer_container.py',
            'import_updates': []
        },
        {
            'source': 'tests/unit/ui/components/test_logging_redirection.py',
            'destination': 'tests/unit/core/logger/test_redirection.py',
            'import_updates': []
        },
        {
            'source': 'tests/ui/setup/colab/test_logging_and_progress.py',
            'destination': 'tests/integration/ui/setup/test_logging_and_progress.py',
            'import_updates': []
        },
        {
            'source': 'tests/ui/setup/colab/test_operation_logs.py',
            'destination': 'tests/integration/ui/setup/test_operation_logs.py',
            'import_updates': []
        }
    ]
    
    # Process each file
    for item in test_reorg:
        source = Path(item['source'])
        destination = Path(item['destination'])
        
        if not source.exists():
            print(f"Warning: Source file not found: {source}")
            continue
            
        # Move the file
        new_path = move_test_file(str(source), str(destination))
        
        # Update imports in the moved file
        for old_import, new_import in item.get('import_updates', []):
            update_file_imports(new_path, old_import, new_import)
    
    print("Test reorganization complete!")

if __name__ == "__main__":
    main()
