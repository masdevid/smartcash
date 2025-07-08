#!/usr/bin/env python3
"""
Simple test runner for colab module without coverage
"""
import subprocess
import sys
import os

# Change to project directory
os.chdir('/Users/masdevid/Projects/smartcash')

# Run specific test files
test_files = [
    'tests/ui/setup/colab/operations/test_operation_manager.py',
    'tests/ui/setup/colab/operations/test_init_operation.py',
    'tests/ui/setup/colab/operations/test_drive_mount_operation.py',
    'tests/ui/setup/colab/operations/test_symlink_operation.py'
]

for test_file in test_files:
    print(f"\n{'='*60}")
    print(f"Running tests in: {test_file}")
    print('='*60)
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, 
            '-v', '--tb=short', '--disable-warnings', '--no-header'
        ], capture_output=True, text=True, env={**os.environ, 'PYTEST_CURRENT_TEST': ''})
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            print(f"❌ Tests failed in {test_file}")
        else:
            print(f"✅ Tests passed in {test_file}")
            
    except Exception as e:
        print(f"Error running tests: {e}")

print(f"\n{'='*60}")
print("Test run complete")
print('='*60)