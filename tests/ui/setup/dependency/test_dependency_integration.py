"""
Integration tests for dependency management functionality.

These tests verify the end-to-end functionality of the dependency management system,
including UI initialization, handler setup, and basic operations.
"""

# Standard library imports
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, ANY, PropertyMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

# Test imports
from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.common.logger import get_logger

# Mock the logger bridge
class MockLoggerBridge:
    def __init__(self, *args, **kwargs):
        self.logs = []
    
    def __call__(self, message, level='info'):
        self.logs.append((level, message))
    
    def info(self, message):
        self.logs.append(('info', message))
    
    def error(self, message):
        self.logs.append(('error', message))
    
    def warning(self, message):
        self.logs.append(('warning', message))
    
    def success(self, message):
        self.logs.append(('success', message))
    
    def get_logs(self):
        return self.logs

# Mock for create_ui_logger_bridge
def mock_create_ui_logger_bridge(ui_components, logger_name):
    return MockLoggerBridge()

class TestDependencyIntegration(unittest.TestCase):
    """Integration tests for dependency management."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with common configurations."""
        cls.logger = get_logger('test.dependency.integration')
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock config handler
        self.mock_config_handler = MagicMock(spec=ConfigHandler)
        self.mock_config_handler.extract_config.return_value = {}
        self.mock_config_handler.get_default_config.return_value = {}
        
        # Create the initializer with our mock config handler
        self.initializer = DependencyInitializer(
            config_handler_class=lambda *args, **kwargs: self.mock_config_handler
        )
        
        # Create mock UI components with all required fields
        self.mock_ui_components = {
            # Required by DependencyInitializer._create_ui_components
            'ui': MagicMock(),
            'status_panel': MagicMock(),
            'log_output': MagicMock(),  # Required directly in required_components
            
            # Additional components that might be required
            'container': MagicMock(),
            'header': MagicMock(),
            'categories_section': MagicMock(),
            'custom_section': MagicMock(),
            'action_section': MagicMock(),
            'progress_tracker': {
                'main_progress': MagicMock(),
                'step_progress': MagicMock()
            },
            'log_components': {
                'log_output': MagicMock(),
                'log_accordion': MagicMock(),
                'entries_container': MagicMock()
            },
            'install_btn': MagicMock(),
            'check_updates_btn': MagicMock(),
            'uninstall_btn': MagicMock()
        }
        
        # Create a mock logger bridge
        self.mock_logger = MockLoggerBridge()
        
        # Patch the create_dependency_main_ui function
        self.ui_patcher = patch(
            'smartcash.ui.setup.dependency.components.ui_components.create_dependency_main_ui',
            return_value=self.mock_ui_components
        )
        self.mock_create_ui = self.ui_patcher.start()
        
        # Patch the logger bridge creation
        self.logger_patcher = patch(
            'smartcash.ui.setup.dependency.dependency_initializer.create_ui_logger_bridge',
            side_effect=mock_create_ui_logger_bridge
        )
        self.mock_logger_bridge = self.logger_patcher.start()
        
        # Patch the _initialize_logger_bridge method
        self.init_logger_patcher = patch.object(
            self.initializer,
            '_initialize_logger_bridge',
            return_value=self.mock_logger
        )
        self.mock_init_logger = self.init_logger_patcher.start()
    
    def tearDown(self):
        """Clean up after each test method."""
        self.ui_patcher.stop()
        self.logger_patcher.stop()
        self.init_logger_patcher.stop()
    
    @patch('smartcash.ui.setup.dependency.handlers.event_handlers.setup_all_handlers')
    def test_initialize_ui(self, mock_setup_handlers):
        """Test UI initialization."""
        # Setup test data
        test_handlers = {'test': MagicMock()}
        mock_setup_handlers.return_value = test_handlers
        
        # Initialize UI with proper mocks
        with patch('smartcash.ui.setup.dependency.dependency_initializer.create_ui_logger_bridge', 
                 return_value=self.mock_logger):
            result = self.initializer.initialize_ui({})
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertEqual(result, self.mock_ui_components)
        self.mock_create_ui.assert_called_once_with({})
        self.mock_init_logger.assert_called_once()
        mock_setup_handlers.assert_called_once()
        
        # Verify the logger bridge was set on the instance
        self.assertEqual(self.initializer._logger_bridge, self.mock_logger)
    
    @patch('smartcash.ui.setup.dependency.handlers.event_handlers.setup_all_handlers')
    def test_initialize_ui_with_config(self, mock_setup_handlers):
        """Test UI initialization with custom config."""
        # Setup test data
        test_config = {'test': 'config'}
        test_handlers = {'test': MagicMock()}
        mock_setup_handlers.return_value = test_handlers
        
        # Initialize UI with config and proper mocks
        with patch('smartcash.ui.setup.dependency.dependency_initializer.create_ui_logger_bridge', 
                 return_value=self.mock_logger):
            result = self.initializer.initialize_ui(test_config)
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertEqual(result, self.mock_ui_components)
        self.mock_create_ui.assert_called_once_with(test_config)
        self.mock_init_logger.assert_called_once()
        mock_setup_handlers.assert_called_once()
        
        # Verify the logger bridge was set on the instance
        self.assertEqual(self.initializer._logger_bridge, self.mock_logger)
        
        # Verify the config was passed through
        self.mock_create_ui.assert_called_with(test_config)
    
    def test_initialize_ui_error_handling(self):
        """Test error handling during UI initialization."""
        # Setup mock to raise an exception
        self.mock_create_ui.side_effect = Exception("Test error")
        
        # Verify exception is properly handled
        with self.assertRaises(Exception) as context:
            self.initializer.initialize_ui({})
        
        self.assertIn("Test error", str(context.exception))
        self.mock_create_ui.assert_called_once()
    
    @patch('smartcash.ui.setup.dependency.handlers.event_handlers.setup_all_handlers')
    def test_setup_handlers(self, mock_setup_handlers):
        """Test handler setup with logger bridge integration."""
        # Setup test data
        test_handlers = {'test': MagicMock()}
        test_config = {'test': 'config'}
        mock_setup_handlers.return_value = test_handlers
        
        # Set the logger bridge on the instance
        self.initializer._logger_bridge = self.mock_logger
        
        # Call the method
        result = self.initializer._setup_handlers(self.mock_ui_components, test_config)
        
        # Verify setup_all_handlers was called with correct arguments
        mock_setup_handlers.assert_called_once_with(
            self.mock_ui_components,
            test_config,
            self.mock_config_handler
        )
        
        # Verify the result contains the handlers
        self.assertEqual(result['handlers'], test_handlers)
        
        # Verify logger bridge was added to components
        self.assertEqual(result['logger_bridge'], self.mock_logger)
    
    @patch('smartcash.ui.setup.dependency.handlers.event_handlers.setup_all_handlers')
    def test_setup_handlers_error_handling(self, mock_setup_handlers):
        """Test error handling in handler setup."""
        # Setup test config and mock logger
        test_config = {'test': 'config'}
        self.initializer._logger_bridge = self.mock_logger
        
        # Setup mock to raise exception
        mock_setup_handlers.side_effect = Exception("Test error")
        
        # Verify the exception is properly wrapped
        with self.assertRaises(ValueError) as context:
            self.initializer._setup_handlers(self.mock_ui_components, test_config)
        
        # Verify the error message
        self.assertIn("Gagal menginisialisasi dependency handlers", str(context.exception))
        
        # Verify the setup was attempted with correct arguments
        mock_setup_handlers.assert_called_once_with(
            self.mock_ui_components,
            test_config,
            self.mock_config_handler
        )
        
        # Verify the logger bridge was set on the components
        self.assertEqual(self.mock_ui_components['logger_bridge'], self.mock_logger)
    
    def test_ui_creation(self):
        """Test UI component creation."""
        # Setup test config
        test_config = {'test': 'config'}
        
        # Call the method with proper mocks
        with patch('smartcash.ui.setup.dependency.dependency_initializer.create_ui_logger_bridge', 
                 return_value=self.mock_logger):
            components = self.initializer._create_ui_components(test_config)
        
        # Verify the results
        self.assertIsNotNone(components)
        self.mock_create_ui.assert_called_once_with(test_config)
        
        # Verify required components are present
        required_components = ['ui', 'status_panel', 'log_output']
        for comp in required_components:
            self.assertIn(comp, components)
        
        # Verify metadata was added
        self.assertEqual(components['module_name'], 'dependency')
        self.assertEqual(components['config_handler'], self.mock_config_handler)
        self.assertIn('initialization_timestamp', components)
        
        # Verify the returned components include our mock components
        self.assertEqual(components['ui'], self.mock_ui_components['ui'])
        self.assertEqual(components['status_panel'], self.mock_ui_components['status_panel'])
        self.assertEqual(components['log_output'], self.mock_ui_components['log_output'])
    
    @patch('smartcash.ui.setup.dependency.handlers.config_handler.DependencyConfigHandler')
    def test_config_handler_initialization(self, mock_config_handler_class):
        """Test config handler initialization."""
        # Create a mock handler instance
        mock_handler = MagicMock()
        mock_config_handler_class.return_value = mock_handler
        
        # Create a new initializer with the mocked config handler class
        initializer = DependencyInitializer(
            config_handler_class=mock_config_handler_class
        )
        
        # Verify config handler was initialized
        mock_config_handler_class.assert_called_once()
        self.assertEqual(initializer.config_handler, mock_handler)
        
        # Verify the module name is set correctly
        self.assertEqual(initializer.module_name, 'dependency')
    
    @patch('smartcash.ui.setup.dependency.handlers.event_handlers.setup_all_handlers')
    def test_initialize_with_components(self, mock_setup_handlers):
        """Test initialization with pre-created components."""
        # Setup test data
        test_handlers = {'test': MagicMock()}
        mock_setup_handlers.return_value = test_handlers
        
        # Call initialize with components
        with patch('smartcash.ui.setup.dependency.dependency_initializer.create_ui_logger_bridge', 
                 return_value=self.mock_logger):
            result = self.initializer.initialize_ui({})
        
        # Verify results
        self.assertEqual(result, self.mock_ui_components)
        self.mock_create_ui.assert_called_once_with({})
        self.mock_init_logger.assert_called_once()
        
        # Verify handlers were set up
        mock_setup_handlers.assert_called_once()
        
        # Verify components are set on the instance
        self.assertEqual(self.initializer.ui_components, self.mock_ui_components)
        self.assertEqual(self.initializer.handlers, test_handlers)
    
    @patch('smartcash.ui.setup.dependency.handlers.event_handlers.setup_all_handlers')
    def test_initialize_without_components(self, mock_setup_handlers):
        """Test initialization without pre-created components."""
        # Setup test data
        test_handlers = {'test': MagicMock()}
        mock_setup_handlers.return_value = test_handlers
        
        # Reset mocks to ensure we test the full initialization
        self.mock_create_ui.reset_mock()
        self.mock_init_logger.reset_mock()
        
        # Call initialize with empty config
        with patch('smartcash.ui.setup.dependency.dependency_initializer.create_ui_logger_bridge', 
                 return_value=self.mock_logger):
            result = self.initializer.initialize_ui({})
        
        # Verify results
        self.assertEqual(result, self.mock_ui_components)
        self.mock_create_ui.assert_called_once_with({})
        self.mock_init_logger.assert_called_once()
        
        # Verify handlers were set up
        mock_setup_handlers.assert_called_once()
        
        # Verify components and handlers are set on the instance
        self.assertEqual(self.initializer.ui_components, self.mock_ui_components)
        self.assertEqual(self.initializer.handlers, test_handlers)
        


if __name__ == '__main__':
    unittest.main()
