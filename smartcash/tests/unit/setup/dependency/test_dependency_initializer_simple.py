"""
Simplified test for DependencyInitializer with minimal dependencies.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch, ANY

# Mock the logger first to avoid import issues
sys.modules['smartcash.ui.utils.ui_logger'] = MagicMock()

# Now import the module under test
from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer

def test_initialization_handlers_failure():
    """Test that initialization handles handler setup failures gracefully."""
    # Create a mock for the ModuleInitializer class
    mock_module_initializer = MagicMock()
    mock_module_initializer.initialize_module_ui.return_value = {'ui': 'mock_ui', 'handlers': {}}
    
    # Create a mock for the DependencyUIHandler class
    mock_ui_handler = MagicMock()
    mock_ui_handler.return_value.setup.side_effect = Exception("Handler setup failed")
    
    # Patch the dependencies
    with patch('smartcash.ui.core.initializers.module_initializer.ModuleInitializer', 
              return_value=mock_module_initializer), \
         patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyUIHandler',
              mock_ui_handler), \
         patch('smartcash.ui.setup.dependency.dependency_initializer.get_default_dependency_config',
              return_value={}):
        
        # Create an instance of our initializer
        initializer = DependencyInitializer()
        
        # Add required attributes that would normally be set in __init__
        initializer._initialized = False
        initializer.logger = MagicMock()
        initializer._ui_components = {}
        initializer._operation_handlers = {}
        
        # Call initialize which should handle the exception
        result = initializer.initialize()
        
        # Verify the error was handled gracefully
        assert result['success'] is False
        assert 'error' in result
        assert 'Handler setup failed' in str(result['error'])
        
        # Verify the handler was created and setup was called
        mock_ui_handler.return_value.setup.assert_called_once()
        
        # Verify logger was called with error
        initializer.logger.error.assert_called()
