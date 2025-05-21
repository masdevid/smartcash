"""
File: smartcash/ui/setup/env_config/tests/test_refactored_handlers.py
Deskripsi: Script untuk menguji refactoring konfigurasi environment
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment_handler():
    """Test the environment handler"""
    from smartcash.ui.handlers.environment_handler import EnvironmentHandler
    
    # Create a mock UI callback
    ui_callbacks = {
        'log_message': lambda msg: logger.info(f"UI Log: {msg}"),
        'update_status': lambda msg, status: logger.info(f"UI Status: {msg} ({status})"),
        'update_progress': lambda val, msg: logger.info(f"UI Progress: {val} - {msg}")
    }
    
    # Create environment handler
    env_handler = EnvironmentHandler(ui_callbacks)
    
    # Test check_required_dirs
    logger.info("Testing check_required_dirs...")
    dirs_exist = env_handler.check_required_dirs()
    logger.info(f"Required directories exist: {dirs_exist}")
    
    # Test environment manager initialization
    logger.info("Testing environment manager...")
    logger.info(f"Is Colab: {env_handler.env_manager.is_colab}")
    logger.info(f"Base dir: {env_handler.env_manager.base_dir}")
    
    return env_handler

def test_auto_check_handler():
    """Test the auto check handler"""
    from smartcash.ui.handlers.auto_check_handler import AutoCheckHandler
    
    # Create a mock UI callback
    ui_callbacks = {
        'log_message': lambda msg: logger.info(f"UI Log: {msg}"),
        'update_status': lambda msg, status: logger.info(f"UI Status: {msg} ({status})"),
        'update_progress': lambda val, msg: logger.info(f"UI Progress: {val} - {msg}")
    }
    
    # Create auto check handler
    auto_check = AutoCheckHandler(ui_callbacks)
    
    # Test check_environment
    logger.info("Testing check_environment...")
    env_info = auto_check.check_environment()
    
    # Print environment info
    logger.info("Environment info:")
    for key, value in env_info.items():
        if key not in ('existing_dirs', 'missing_dirs'):
            logger.info(f"  {key}: {value}")
    
    return auto_check

def main():
    """Run all tests"""
    logger.info("=== Testing Environment Handler ===")
    env_handler = test_environment_handler()
    
    logger.info("\n=== Testing Auto Check Handler ===")
    auto_check = test_auto_check_handler()
    
    logger.info("\n=== All tests completed ===")

if __name__ == "__main__":
    main() 