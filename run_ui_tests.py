#!/usr/bin/env python3
"""
Run UI component tests.
"""
import unittest
import sys
from pathlib import Path

def run_tests():
    """Run all UI component tests."""
    # Add the project root to the Python path
    project_root = str(Path(__file__).parent.absolute())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        'tests/ui/setup/dependency',
        pattern='test_*.py',
        top_level_dir='tests'
    )
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return non-zero exit code if any tests failed
    return not result.wasSuccessful()

if __name__ == '__main__':
    sys.exit(run_tests())
