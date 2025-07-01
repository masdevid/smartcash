"""Script to run the test and capture detailed error output using pytest."""
import os
import sys
import pytest
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

def run_test():
    """Run the test using pytest and capture output."""
    # Configure pytest arguments
    test_path = 'tests/ui/setup/env_config/handlers/test_folder_handler.py::TestFolderHandler::test_initialization'
    
    # Create a string buffer to capture the output
    stdout_buf = StringIO()
    stderr_buf = StringIO()
    
    # Redirect stdout and stderr
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        # Run pytest with the specific test
        exit_code = pytest.main([
            '-v',
            '-s',  # Show output
            '--tb=native',  # Show full traceback
            test_path
        ])
    
    # Get the captured output
    stdout_output = stdout_buf.getvalue()
    stderr_output = stderr_buf.getvalue()
    
    # Print the captured output
    print("=== STDOUT ===")
    print(stdout_output)
    print("=== STDERR ===")
    print(stderr_output)
    
    return exit_code

if __name__ == "__main__":
    sys.exit(run_test())
