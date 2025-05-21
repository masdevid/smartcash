"""
File: smartcash/common/tests/test_logger.py
Deskripsi: Script untuk menguji implementasi logger yang telah direfactor
"""

import logging
import sys
import os
from pathlib import Path
import shutil

# Setup basic logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_logs_folder():
    """Delete temporary test log files"""
    temp_logs = Path(__file__).parent / "temp_logs"
    if temp_logs.exists():
        shutil.rmtree(temp_logs, ignore_errors=True)
    temp_logs.mkdir(exist_ok=True)
    return temp_logs

def test_smartcash_logger():
    """Test the SmartCashLogger implementation"""
    from smartcash.common.logger import get_logger, LogLevel
    
    print("=== Testing SmartCashLogger ===")
    
    # Create logger
    sc_logger = get_logger("test_smartcash")
    
    # Test log levels
    sc_logger.debug("This is a debug message")
    sc_logger.info("This is an info message")
    sc_logger.success("This is a success message")
    sc_logger.warning("This is a warning message")
    sc_logger.error("This is an error message")
    sc_logger.critical("This is a critical message")
    
    # Test callback functionality
    def log_callback(level, message):
        print(f"Callback received: [{level.name}] {message}")
    
    sc_logger.add_callback(log_callback)
    sc_logger.info("This info message should trigger the callback")
    
    # Test log level setting
    sc_logger.set_level(LogLevel.DEBUG)
    sc_logger.debug("This debug message should appear after setting level")
    
    # Clean up
    sc_logger.remove_callback(log_callback)
    
    return sc_logger

def test_ui_logger():
    """Test the UILogger implementation with mock UI components"""
    from smartcash.ui.utils.ui_logger import UILogger
    import ipywidgets as widgets
    from IPython.display import display
    
    print("\n=== Testing UILogger ===")
    
    # Create temp logs folder
    temp_logs = clear_logs_folder()
    
    # Create mock UI components
    mock_ui_components = {
        'status': None,  # No real UI components, will use fallback to stdout
        'log_output': None
    }
    
    # Create UILogger with temporary log file
    ui_logger = UILogger(mock_ui_components, "test_ui_logger", True, str(temp_logs))
    
    # Test log levels
    ui_logger.debug("This is a UI debug message")
    ui_logger.info("This is a UI info message")
    ui_logger.success("This is a UI success message")
    ui_logger.warning("This is a UI warning message")
    ui_logger.error("This is a UI error message")
    ui_logger.critical("This is a UI critical message")
    
    # Check if log file was created
    log_files = list(temp_logs.glob("test_ui_logger_*.log"))
    if log_files:
        print(f"Log file created: {log_files[0]}")
        # Display content of log file
        with open(log_files[0], 'r') as f:
            print("\nLog file content (first 5 lines):")
            lines = f.readlines()
            for line in lines[:5]:
                print(f"  {line.strip()}")
    else:
        print("Warning: No log file was created")
    
    # Test setting log level
    ui_logger.set_level(logging.DEBUG)
    ui_logger.debug("This debug message should be logged to file after changing level")
    
    return ui_logger

def test_integration():
    """Test the integration between SmartCashLogger and UILogger"""
    from smartcash.common.logger import get_logger, LogLevel
    from smartcash.ui.utils.ui_logger import create_ui_logger
    
    print("\n=== Testing Logger Integration ===")
    
    # Create temp logs folder
    temp_logs = clear_logs_folder()
    
    # Create mock UI components
    mock_ui_components = {
        'status': None,  # No real UI components, will use fallback to stdout
        'log_output': None
    }
    
    # Create UI logger with file logging
    ui_logger = create_ui_logger(
        mock_ui_components, 
        name="test_integration", 
        log_to_file=True,
        log_dir=str(temp_logs)
    )
    
    # Get smartcash_logger from ui_components
    sc_logger = mock_ui_components.get('smartcash_logger')
    if not sc_logger:
        print("Warning: SmartCashLogger integration failed")
        from smartcash.common.logger import get_logger
        sc_logger = get_logger("test_integration")
    
    # Test log messages - should be captured by both loggers
    print("\nSending logs via SmartCashLogger (should be captured by UILogger):")
    sc_logger.info("Integration info test")
    sc_logger.warning("Integration warning test")
    sc_logger.error("Integration error test")
    
    # Check if log file contains entries from SmartCashLogger
    log_files = list(temp_logs.glob("test_integration_*.log"))
    if log_files:
        print(f"\nChecking integration in log file: {log_files[0]}")
        with open(log_files[0], 'r') as f:
            log_content = f.read()
            print(f"Log contains 'Integration info test': {'Integration info test' in log_content}")
            print(f"Log contains 'Integration warning test': {'Integration warning test' in log_content}")
            print(f"Log contains 'Integration error test': {'Integration error test' in log_content}")
    
    return sc_logger, ui_logger

def run_tests():
    """Run all logger tests"""
    # Create temp logs folder
    temp_logs = clear_logs_folder()
    
    try:
        test_smartcash_logger()
        test_ui_logger()
        test_integration()
        
        print("\n=== All logger tests completed ===")
    finally:
        # Clean up logs
        if os.environ.get('KEEP_TEST_LOGS') != '1':
            shutil.rmtree(temp_logs, ignore_errors=True)
            print("Test logs cleaned up")
        else:
            print(f"Test logs kept in: {temp_logs.absolute()}")

if __name__ == "__main__":
    run_tests() 