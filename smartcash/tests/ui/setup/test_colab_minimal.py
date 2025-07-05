import pytest
from unittest.mock import MagicMock, patch
import sys

# Mocking the logger from the correct path
sys.modules['smartcash.ui.core.shared.logger'] = MagicMock()

# Impor yang diperbaiki untuk ColabEnvInitializer dari submodule yang benar
from smartcash.ui.setup.colab.colab_initializer import ColabEnvInitializer

class SimpleMockLogger:
    def info(self, *args, **kwargs):
        print(f"INFO: {args} {kwargs}")
    def warning(self, *args, **kwargs):
        print(f"WARNING: {args} {kwargs}")
    def error(self, *args, **kwargs):
        print(f"ERROR: {args} {kwargs}")
    def debug(self, *args, **kwargs):
        print(f"DEBUG: {args} {kwargs}")
    def critical(self, *args, **kwargs):
        print(f"CRITICAL: {args} {kwargs}")
    def exception(self, *args, **kwargs):
        print(f"EXCEPTION: {args} {kwargs}")
    def log(self, level, *args, **kwargs):
        print(f"LOG {level}: {args} {kwargs}")

def test_colab_env_initializer_post_checks():
    # Create mock for setup_handler
    mock_setup_handler = MagicMock()
    mock_setup_handler.perform_initial_status_check.return_value = {'status': 'ok'}
    mock_setup_handler.should_sync_config_templates.return_value = False
    mock_setup_handler.sync_config_templates.return_value = {'status': 'synced'}

    # Create instance of ColabEnvInitializer with mocked logger
    initializer = ColabEnvInitializer()
    initializer.logger = SimpleMockLogger()

    # Mock initialize method to set _initialized
    initializer.initialize = MagicMock(return_value=None)
    initializer._initialized = True
    initializer._is_initialized = True

    print("Before _post_checks call")
    print(f"_initialized: {hasattr(initializer, '_initialized') and initializer._initialized}")
    print(f"_is_initialized: {hasattr(initializer, '_is_initialized') and initializer._is_initialized}")

    # Call _post_checks
    try:
        initializer._post_checks()
        print("_post_checks executed successfully")
    except Exception as e:
        print(f"Error in _post_checks: {str(e)}")
        raise

    print("After _post_checks call")
