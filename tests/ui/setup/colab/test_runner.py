"""
Test runner for smartcash.ui.setup.colab module tests
"""

import pytest
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def run_colab_tests():
    """Run all tests for the colab module."""
    test_dir = Path(__file__).parent
    
    # Run tests with pytest
    pytest_args = [
        str(test_dir),
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--strict-markers',  # Strict marker checking
        '-x',  # Stop on first failure
        '--disable-warnings'  # Disable warnings for cleaner output
    ]
    
    print(f"Running tests in: {test_dir}")
    print("=" * 60)
    
    result = pytest.main(pytest_args)
    
    if result == 0:
        print("\n" + "=" * 60)
        print("✅ All colab module tests passed!")
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed. Check the output above.")
    
    return result

def run_specific_test_file(test_file: str):
    """Run tests from a specific test file."""
    test_path = Path(__file__).parent / test_file
    
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return 1
    
    pytest_args = [
        str(test_path),
        '-v',
        '--tb=short',
        '--disable-warnings'
    ]
    
    print(f"Running tests in: {test_path}")
    print("=" * 60)
    
    return pytest.main(pytest_args)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        exit_code = run_specific_test_file(test_file)
    else:
        # Run all tests
        exit_code = run_colab_tests()
    
    sys.exit(exit_code)