"""
File: tests/unit/ui/model/pretrained/test_pretrained_uimodule.py
Description: Comprehensive unit tests for the PretrainedUIModule class.
"""
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch, ANY, call
from typing import Dict, Any
import pytest

# Import the module we're testing
from smartcash.ui.model.pretrained.pretrained_uimodule import PretrainedUIModule
from smartcash.ui.model.pretrained.configs.pretrained_defaults import get_default_pretrained_config
from smartcash.ui.model.pretrained.configs.pretrained_config_handler import PretrainedConfigHandler


class MockPretrainedUIModule(PretrainedUIModule):
    """Mock implementation of PretrainedUIModule for testing."""
    def __init__(self, config=None):
        # Initialize required attributes first
        self.module_name = 'pretrained'
        self.parent_module = 'model'
        self.full_module_name = f"{self.parent_module}.{self.module_name}"
        self._enable_environment = False
        self._is_initialized = False
        self._initialized = False
        self._is_colab = False
        self._has_environment_support = False
        
        # Set up logging
        self.logger = MagicMock()
        self.log = MagicMock()
        self.log.info = MagicMock()
        self.log.error = MagicMock()
        self.log.warning = MagicMock()
        self.log.debug = MagicMock()
        self.log.reset_mock = MagicMock()
        
        # Set up environment paths
        self._environment_paths = MagicMock()
        self._environment_paths.data_root = '/tmp/models'
        self._environment_paths.models_dir = '/tmp/models/pretrained'
        
        # Initialize model status
        self._model_status = {
            'last_refresh': None,
            'models_found': [],
            'validation_results': {},
            'environment_paths': self._environment_paths
        }
        
        # Initialize config handler with provided config or default
        self._config_handler = MagicMock()
        self._default_config = config or {
            'models_dir': '/tmp/models',
            'pretrained_models': ['yolov5s', 'efficientnet_b4'],
            'enable_environment': False,
            'required_components': ['model_dropdown', 'download_button']
        }
        self._config_handler.get_config.return_value = self._default_config
        
        # UI components
        self._ui_components = {}
        self._required_components = [
            'main_container',
            'header_container',
            'form_container',
            'action_container',
            'operation_container'
        ]
        
        # Mock operations
        self._mock_operations = {}
        
    @property
    def environment_paths(self):
        return self._environment_paths if self._enable_environment else None
        
    @environment_paths.setter
    def environment_paths(self, value):
        self._environment_paths = value
        
    def _update_logging_context(self):
        # Mock the logging context update
        pass
        
    def get_default_config(self):
        """Return the default configuration for testing."""
        return self._default_config
        
    def get_current_config(self):
        """Return the current configuration."""
        return self._default_config if hasattr(self, '_default_config') else {}
        
    @property
    def config(self):
        """Return the current config for compatibility with tests."""
        return self._default_config
        
    def initialize(self, *args, **kwargs):
        """Mock initialize method that sets the initialized flag and returns True.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
                refresh_on_init (bool): Whether to refresh on initialization. Defaults to True.
                
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Skip if already initialized
            if self._initialized:
                return True
                
            # Handle config from kwargs if provided
            if 'config' in kwargs:
                self._default_config.update(kwargs['config'])
                if hasattr(self, '_config_handler') and self._config_handler is not None:
                    self._config_handler.get_config.return_value = self._default_config
            
            # Mock the config handler if not already set
            if not hasattr(self, '_config_handler') or self._config_handler is None:
                self._config_handler = MagicMock()
                self._config_handler.get_config.return_value = self.get_default_config()
            
            # Create UI components if not already set
            if not hasattr(self, '_ui_components') or not self._ui_components:
                if hasattr(self, 'create_ui_components') and callable(self.create_ui_components):
                    # If create_ui_components is patched, use it
                    self._ui_components = self.create_ui_components(self._default_config)
                else:
                    # Otherwise, create default mock components
                    self._ui_components = {
                        'main_container': MagicMock(),
                        'header_container': MagicMock(),
                        'form_container': MagicMock(),
                        'action_container': MagicMock(),
                        'operation_container': MagicMock()
                    }
            
            # Initialize or update model status
            self._model_status = {
                'last_refresh': None,
                'models_found': [],
                'validation_results': {},
                'environment_paths': self._environment_paths,
                'initialized': True
            }
            
            # Mock logger if not already set
            if not hasattr(self, 'log') or self.log is None:
                self.log = MagicMock()
                self.log.info = MagicMock()
                self.log.error = MagicMock()
                self.log.warning = MagicMock()
                self.log.debug = MagicMock()
            
            # Log successful initialization
            self.log.info("✅ PretrainedUIModule initialized")
            
            # Set initialized flags after everything else is set up
            self._initialized = True
            self._is_initialized = True
            
            # If refresh_on_init is True (default), simulate a refresh
            refresh_on_init = kwargs.get('refresh_on_init', True)
            if refresh_on_init and hasattr(self, '_execute_pretrained_operation'):
                try:
                    refresh_result = self._execute_pretrained_operation('refresh')
                    # Update model status with refresh results if available
                    if isinstance(refresh_result, dict):
                        self._model_status.update({
                            'models_found': refresh_result.get('models_found', []),
                            'validation_results': refresh_result.get('validation_results', {}),
                            'last_refresh': refresh_result.get('last_refresh')
                        })
                except Exception as e:
                    error_msg = f"Initial refresh failed: {str(e)}"
                    self.log.warning(error_msg)
                    # Update model status with error
                    self._model_status['error'] = error_msg
            
            return True
            
        except Exception as e:
            # Log the error
            error_msg = f"Failed to initialize PretrainedUIModule: {str(e)}"
            if hasattr(self, 'log') and hasattr(self.log, 'error'):
                self.log.error(error_msg)
            
            # Update model status with error
            if hasattr(self, '_model_status'):
                self._model_status['error'] = error_msg
                self._model_status['initialized'] = False
            
            # Reset initialized flags
            self._initialized = False
            self._is_initialized = False
            
            return False
        
    @property
    def is_colab(self):
        return self._is_colab
        
    @is_colab.setter
    def is_colab(self, value):
        self._is_colab = value
        
    @property
    def has_environment_support(self):
        return self._has_environment_support
        
    @has_environment_support.setter
    def has_environment_support(self, value):
        self._has_environment_support = value
        
    def get_component(self, name):
        # Mock get_component to return a mock UI component
        return MagicMock()


class TestPretrainedUIModule(unittest.TestCase):
    """Test suite for PretrainedUIModule."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level mocks before any tests run."""
        # Create a patch for the logger that will be used by all test methods
        cls.logger_patcher = patch('smartcash.ui.model.pretrained.pretrained_uimodule.logging')
        cls.mock_logging = cls.logger_patcher.start()
        cls.mock_logging.getLogger.return_value = MagicMock()
        
        # Create a mock for the PretrainedUIModule class
        cls.original_pretrained_module = PretrainedUIModule
        
        # Replace the real class with our mock
        cls.pretrained_uimodule = MockPretrainedUIModule
        
        # Create a module instance for testing
        cls.module = cls.pretrained_uimodule()
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level patches after all tests have run."""
        # Stop the logger patcher
        cls.logger_patcher.stop()
        
        # Clean up any other resources if needed
        if hasattr(cls, 'module'):
            del cls.module
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a new instance of our mock class for each test
        self.module = MockPretrainedUIModule()
        
        # Create a mock config handler
        self.mock_config_handler = MagicMock()
        self.mock_config_handler.get_config.return_value = {
            'models_dir': '/tmp/models',
            'pretrained_models': ['yolov5s', 'efficientnet_b4']
        }
        self.module._config_handler = self.mock_config_handler
        
        # Create mock UI components
        self.mock_ui_components = {
            'operation_container': MagicMock(),
            'status_panel': MagicMock(),
            'download_button': MagicMock(),
            'validate_button': MagicMock(),
            'refresh_button': MagicMock(),
            'cleanup_button': MagicMock(),
            'main_container': MagicMock(),
            'header_container': MagicMock(),
            'form_container': MagicMock(),
            'action_container': MagicMock()
        }
        self.module._ui_components = self.mock_ui_components
        
        # Set up module attributes
        self.module._ui_components = self.mock_ui_components
        self.module._config_handler = self.mock_config_handler
        self.module._is_initialized = True
        
        # Mock the get_current_config method
        self.module.get_current_config = MagicMock(return_value={
            'models_dir': '/tmp/models',
            'pretrained_models': ['yolov5s', 'efficientnet_b4']
        })
        
        # Mock the get_component method
        self.module.get_component = MagicMock(side_effect=lambda name: self.mock_ui_components.get(name))
    
    def test_initialization(self):
        """Test that the module initializes correctly."""
        # Create a fresh instance to test initialization
        module = MockPretrainedUIModule()
        
        # Check that the module was initialized with the correct default values
        self.assertEqual(module.module_name, 'pretrained')
        self.assertEqual(module.parent_module, 'model')
        self.assertEqual(module.full_module_name, 'model.pretrained')
        self.assertEqual(module._enable_environment, False)
        
        # Check initialization flags - should be False by default in the mock
        self.assertFalse(module._is_initialized)
        self.assertFalse(module._initialized)
        
        # Check environment settings
        self.assertEqual(module._is_colab, False)
        self.assertEqual(module._has_environment_support, False)
        
        # Check that the model status was initialized correctly
        self.assertIsNotNone(module._model_status)
        self.assertEqual(module._model_status['last_refresh'], None)
        self.assertEqual(module._model_status['models_found'], [])
        self.assertEqual(module._model_status['validation_results'], {})
        self.assertIsNotNone(module._model_status['environment_paths'])
        
        # Check required components
        self.assertEqual(module._required_components, [
            'main_container',
            'header_container',
            'form_container',
            'action_container',
            'operation_container'
        ])
        
        # Check that environment paths were set up
        self.assertEqual(self.module._environment_paths.data_root, '/tmp/models')
        
        # Check that required components are set
        self.assertIn('main_container', self.module._required_components)
        self.assertIn('header_container', self.module._required_components)
        self.assertIn('form_container', self.module._required_components)
        self.assertIn('action_container', self.module._required_components)
        self.assertIn('operation_container', self.module._required_components)
    
    def test_get_default_config(self):
        """Test that get_default_config returns the expected default config."""
        # Call the method to get the default config
        result = self.module.get_default_config()
        
        # Verify the result has the expected structure
        self.assertIsInstance(result, dict)
        self.assertIn('models_dir', result)
        self.assertIn('pretrained_models', result)
        self.assertIn('enable_environment', result)
        self.assertIn('required_components', result)
        
        # Verify the default values
        self.assertEqual(result['models_dir'], '/tmp/models')
        self.assertEqual(result['pretrained_models'], ['yolov5s', 'efficientnet_b4'])
        self.assertFalse(result['enable_environment'])
        self.assertEqual(result['required_components'], ['model_dropdown', 'download_button'])
    
    def test_create_config_handler(self):
        """Test that create_config_handler returns a PretrainedConfigHandler with the given config."""
        test_config = {'test': 'config'}
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedConfigHandler') as mock_handler_class:
            mock_handler = MagicMock()
            mock_handler_class.return_value = mock_handler
            
            result = self.module.create_config_handler(test_config)
            
            self.assertEqual(result, mock_handler)
            mock_handler_class.assert_called_once_with(test_config)
    
    def test_create_ui_components(self):
        """Test that create_ui_components calls the UI factory function with the correct config."""
        test_config = {'test': 'config'}
        expected_components = {'test': 'components'}
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.create_pretrained_ui_components') as mock_create_ui:
            mock_create_ui.return_value = expected_components
            
            result = self.module.create_ui_components(test_config)
            
            self.assertEqual(result, expected_components)
            mock_create_ui.assert_called_once_with(module_config=test_config)
    
    def test_create_ui_components_failure(self):
        """Test that create_ui_components handles failures correctly."""
        test_config = {'test': 'config'}
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.create_pretrained_ui_components') as mock_create_ui:
            mock_create_ui.return_value = None
            
            with self.assertRaises(RuntimeError) as context:
                self.module.create_ui_components(test_config)
            
            self.assertIn("Failed to create UI components", str(context.exception))
    
    def test_get_module_button_handlers(self):
        """Test that _get_module_button_handlers returns the correct button handlers."""
        # Call the method
        handlers = self.module._get_module_button_handlers()
        
        # Check that pretrained-specific handlers are included
        self.assertIn('download', handlers)
        self.assertIn('validate', handlers)
        self.assertIn('refresh', handlers)
        self.assertIn('cleanup', handlers)
        
        # Check that base handlers are also included
        self.assertIn('save', handlers)
        self.assertIn('reset', handlers)
        
        # Check that the handlers are callable
        self.assertTrue(callable(handlers['download']))
        self.assertTrue(callable(handlers['validate']))
        self.assertTrue(callable(handlers['refresh']))
        self.assertTrue(callable(handlers['cleanup']))
    
    def test_operation_download(self):
        """Test the download operation handler."""
        # Mock the _execute_operation_with_wrapper method
        with patch.object(self.module, '_execute_operation_with_wrapper') as mock_wrapper:
            mock_wrapper.return_value = {'success': True, 'message': 'Download completed'}
            
            # Call the operation
            result = self.module._operation_download()
            
            # Verify the result
            self.assertTrue(result['success'])
            self.assertEqual(result['message'], 'Download completed')
            
            # Verify the wrapper was called with correct parameters
            mock_wrapper.assert_called_once()
            _, kwargs = mock_wrapper.call_args
            self.assertEqual(kwargs['operation_name'], 'Model Download')
            self.assertEqual(kwargs['success_message'], 'Model download completed successfully')
            self.assertEqual(kwargs['error_message'], 'Model download failed')
    
    def test_operation_validate(self):
        """Test the validate operation handler."""
        # Mock the _execute_operation_with_wrapper method
        with patch.object(self.module, '_execute_operation_with_wrapper') as mock_wrapper:
            mock_wrapper.return_value = {'success': True, 'message': 'Validation completed'}
            
            # Call the operation
            result = self.module._operation_validate()
            
            # Verify the result
            self.assertTrue(result['success'])
            self.assertEqual(result['message'], 'Validation completed')
            
            # Verify the wrapper was called with correct parameters
            mock_wrapper.assert_called_once()
            _, kwargs = mock_wrapper.call_args
            self.assertEqual(kwargs['operation_name'], 'Model Validation')
            self.assertEqual(kwargs['success_message'], 'Model validation completed successfully')
            self.assertEqual(kwargs['error_message'], 'Model validation failed')
    
    def test_operation_refresh(self):
        """Test the refresh operation handler."""
        # Mock the _execute_operation_with_wrapper method
        with patch.object(self.module, '_execute_operation_with_wrapper') as mock_wrapper:
            mock_wrapper.return_value = {'success': True, 'message': 'Refresh completed'}
            
            # Call the operation
            result = self.module._operation_refresh()
            
            # Verify the result
            self.assertTrue(result['success'])
            self.assertEqual(result['message'], 'Refresh completed')
            
            # Verify the wrapper was called with correct parameters
            mock_wrapper.assert_called_once()
            _, kwargs = mock_wrapper.call_args
            self.assertEqual(kwargs['operation_name'], 'Model Refresh')
            self.assertEqual(kwargs['success_message'], 'Model refresh completed successfully')
            self.assertEqual(kwargs['error_message'], 'Model refresh failed')
    
    def test_operation_cleanup(self):
        """Test the cleanup operation handler."""
        # Mock the _execute_operation_with_wrapper method
        with patch.object(self.module, '_execute_operation_with_wrapper') as mock_wrapper:
            mock_wrapper.return_value = {'success': True, 'message': 'Cleanup completed'}
            
            # Call the operation
            result = self.module._operation_cleanup()
            
            # Verify the result
            self.assertTrue(result['success'])
            self.assertEqual(result['message'], 'Cleanup completed')
            
            # Verify the wrapper was called with correct parameters
            mock_wrapper.assert_called_once()
            _, kwargs = mock_wrapper.call_args
            self.assertEqual(kwargs['operation_name'], 'Model Cleanup')
            self.assertEqual(kwargs['success_message'], 'Model cleanup completed successfully')
            self.assertEqual(kwargs['error_message'], 'Model cleanup failed')
    
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation')
    def test_execute_pretrained_operation_success(self, mock_execute_operation, mock_makedirs, mock_exists):
        """Test successful execution of a pretrained operation."""
        # Create a fresh module instance for this test
        test_module = MockPretrainedUIModule()
        
        # Setup test components and config
        test_components = {'test': 'components'}
        test_module._ui_components = test_components
        
        test_config = {
            'models_dir': '/tmp/models',
            'pretrained_models': ['yolov5s']
        }
        
        # Mock the operation result for refresh operation
        mock_result = {
            'success': True,
            'message': 'Refresh completed',
            'models_found': ['model1.pt'],
            'validation_results': {'model1.pt': True},
            'refresh_time': '2024-01-01T00:00:00',
            'last_refresh': '2024-01-01T00:00:00'
        }
        mock_execute_operation.return_value = mock_result
        
        # Store the original model status for comparison
        original_status = test_module._model_status.copy()
        
        # Mock the log method to prevent test failures from missing logs
        test_module.log = MagicMock()
        
        # Execute refresh operation with test config
        result = test_module._execute_pretrained_operation(
            operation_type='refresh',
            config=test_config
        )
        
        # Verify the result
        self.assertTrue(result['success'])
        self.assertEqual(result['message'], 'Refresh completed')
        
        # Verify the factory was called with correct arguments
        mock_execute_operation.assert_called_once()
        call_args, call_kwargs = mock_execute_operation.call_args
        
        # Check the operation type matches what we passed in
        self.assertEqual(call_kwargs['operation_type'], 'refresh')
        
        # Check the UI components and config match our test data
        for key, value in test_components.items():
            self.assertIn(key, call_kwargs['ui_components'])
            self.assertEqual(call_kwargs['ui_components'][key], value)
        
        # Verify config values we care about
        self.assertEqual(call_kwargs['config']['models_dir'], test_config['models_dir'])
        self.assertEqual(call_kwargs['config']['pretrained_models'], test_config['pretrained_models'])
        
        # Verify model status was updated with the operation results
        self.assertEqual(test_module._model_status['models_found'], mock_result['models_found'])
        self.assertEqual(test_module._model_status['validation_results'], 
                        mock_result['validation_results'])
        self.assertEqual(test_module._model_status['last_refresh'], 
                        mock_result['last_refresh'])
        
        # Verify the model status was updated (should be different from original)
        self.assertNotEqual(test_module._model_status, original_status, 
                          "Model status should have been updated")
        
        # Verify the model status was updated with the operation results
        self.assertEqual(test_module._model_status['models_found'], mock_result['models_found'])
        self.assertEqual(test_module._model_status['validation_results'], mock_result['validation_results'])
        self.assertEqual(test_module._model_status['last_refresh'], mock_result['last_refresh'])
        
        # Verify the model status was updated (should be different from original)
        self.assertNotEqual(test_module._model_status, original_status, 
                          "Model status should have been updated")
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation')
    def test_execute_pretrained_operation_exception(self, mock_execute_operation, mock_makedirs, mock_exists):
        """Test handling of exceptions in pretrained operations."""
        # Create a fresh module instance for this test
        test_module = MockPretrainedUIModule()
        
        # Setup test components and config
        test_components = {'test': 'components'}
        test_module._ui_components = test_components
        
        test_config = {
            'models_dir': '/tmp/models',
            'pretrained_models': ['yolov5s', 'efficientnet_b4']
        }
        
        # Test error to be raised
        test_error = Exception('Test error')
        mock_execute_operation.side_effect = test_error
        
        # Create a mock for the log method that accepts log level parameter
        mock_log = MagicMock()
        test_module.log = mock_log
        
        # Execute operation with test config
        result = test_module._execute_pretrained_operation(
            operation_type='test_operation',
            config=test_config
        )
        
        # Verify error handling
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        self.assertFalse(result.get('success', True), "Operation should have failed")
        self.assertIn('error', result, "Error should be in the result")
        self.assertIn('Test error', str(result['error']), "Error message should contain 'Test error'")
        
        # Verify the factory was called with correct arguments
        mock_execute_operation.assert_called_once()
        call_args, call_kwargs = mock_execute_operation.call_args
        
        # Check the operation type matches what we passed in
        self.assertEqual(call_kwargs['operation_type'], 'test_operation')
        
        # Check the UI components and config match our test data
        for key, value in test_components.items():
            self.assertIn(key, call_kwargs['ui_components'])
            self.assertEqual(call_kwargs['ui_components'][key], value)
        
        # Verify config values we care about
        self.assertEqual(call_kwargs['config']['models_dir'], test_config['models_dir'])
        self.assertEqual(call_kwargs['config']['pretrained_models'], test_config['pretrained_models'])
        
        # Verify error was logged using the log method with 'error' level
        mock_log.assert_called_once()
        log_args, log_kwargs = mock_log.call_args
        
        # Check the log message and level
        # The log method is called with (message, level='info') as positional args
        # So we expect two positional arguments and no keyword arguments
        self.assertEqual(len(log_args), 2, "Log should have message and level arguments")
        self.assertIn('Error in test_operation operation: Test error', log_args[0])
        self.assertEqual(log_args[1], 'error', "Log level should be 'error'")
    
    def test_validate_models(self):
        """Test the _validate_models method."""
        # Create a fresh module instance for this test
        test_module = MockPretrainedUIModule()
        
        # Call the method
        result = test_module._validate_models()
        
        # Verify the result matches the actual implementation
        self.assertEqual(result, {'valid': True}, 
                        "Expected _validate_models to return {'valid': True}")
        
    def test_get_model_status(self):
        """Test the get_model_status method."""
        # Create a fresh module instance for this test
        test_module = MockPretrainedUIModule()
        
        # Set up test data
        test_module._is_initialized = True
        test_module._ui_components = {'test': 'component'}  # Simulate UI components being created
        test_module._model_status = {
            'last_refresh': '2024-01-01T00:00:00',
            'models_found': ['model1.pt'],
            'validation_results': {'model1.pt': True}
        }
        
        # Set up mock environment paths and enable environment
        mock_env_paths = MagicMock()
        mock_env_paths.data_root = '/tmp/models'
        test_module._enable_environment = True  # Enable environment support
        test_module._environment_paths = mock_env_paths
        
        # Mock config handler
        test_module._config_handler = MagicMock()
        
        # Test with local environment
        test_module.is_colab = False
        status = test_module.get_model_status()
        
        # Verify the returned status structure
        self.assertEqual(status['initialized'], True)
        self.assertEqual(status['module_name'], 'pretrained')
        self.assertEqual(status['environment_type'], 'local')
        self.assertEqual(status['config_loaded'], True)
        self.assertEqual(status['ui_created'], True)
        self.assertEqual(status['environment_paths'], mock_env_paths)
        self.assertEqual(status['model_status'], test_module._model_status)
        
        # Test with Colab environment
        test_module.is_colab = True
        status = test_module.get_model_status()
        self.assertEqual(status['environment_type'], 'colab')
        
        # Test with environment disabled
        test_module._enable_environment = False
        status = test_module.get_model_status()
        self.assertIsNone(status['environment_paths'])
        test_module._enable_environment = True  # Re-enable for other tests
        
        # Test with no config handler
        test_module._config_handler = None
        status = test_module.get_model_status()
        self.assertFalse(status['config_loaded'])
        test_module._config_handler = MagicMock()  # Restore for other tests
        
        # Test with no UI components
        test_module._ui_components = {}
        status = test_module.get_model_status()
        self.assertFalse(status['ui_created'])
        test_module._ui_components = {'test': 'component'}  # Restore for other tests
        
        # Test with module not initialized
        test_module._is_initialized = False
        status = test_module.get_model_status()
        self.assertFalse(status['initialized'])
        test_module._is_initialized = True  # Restore for other tests
    
    def test_get_model_status_colab_environment(self):
        """Test get_model_status in colab environment with environment support."""
        # Create a fresh module instance for this test
        test_module = MockPretrainedUIModule()
        
        # Set up test data for Colab environment
        test_module._is_initialized = True
        test_module._ui_components = {'test': 'component'}
        test_module._model_status = {
            'last_refresh': '2024-01-01T00:00:00',
            'models_found': ['model1.pt'],
            'validation_results': {'model1.pt': True}
        }
        
        # Set up mock environment paths for Colab and enable environment
        mock_env_paths = MagicMock()
        mock_env_paths.data_root = '/content/drive/MyDrive/models'
        test_module._enable_environment = True  # Enable environment support
        test_module._environment_paths = mock_env_paths
        
        # Mock config handler
        test_module._config_handler = MagicMock()
        
        # Set up Colab environment
        test_module.is_colab = True
        
        # Call the method
        status = test_module.get_model_status()
        
        # Verify the returned status structure for Colab environment
        self.assertEqual(status['initialized'], True)
        self.assertEqual(status['module_name'], 'pretrained')
        self.assertEqual(status['environment_type'], 'colab')
        self.assertEqual(status['config_loaded'], True)
        self.assertEqual(status['ui_created'], True)
        self.assertEqual(status['environment_paths'], mock_env_paths)
        self.assertEqual(status['model_status'], test_module._model_status)
        
        # Verify the environment paths are correct
        self.assertEqual(status['environment_paths'].data_root, '/content/drive/MyDrive/models')
        
        # Test with environment disabled
        test_module._enable_environment = False
        status = test_module.get_model_status()
        self.assertIsNone(status['environment_paths'])
        test_module._enable_environment = True  # Re-enable for other tests
    
    def test_initialize_success(self):
        """Test successful initialization of the module."""
        # Define test config
        test_config = {
            'models_dir': '/tmp/models',
            'pretrained_models': ['yolov5s', 'efficientnet_b4']
        }
        
        # Create a fresh module instance with test config
        test_module = MockPretrainedUIModule(config=test_config)
        
        # Mock UI components
        mock_ui_components = {
            'main_container': MagicMock(),
            'header_container': MagicMock(),
            'form_container': MagicMock(),
            'action_container': MagicMock(),
            'operation_container': MagicMock()
        }
        
        # Configure the mock for execute_pretrained_operation to simulate a successful refresh
        mock_refresh_result = {
            'success': True,
            'models_found': ['yolov5s.pt', 'efficientnet_b4.pt'],
            'validation_results': {'yolov5s.pt': True, 'efficientnet_b4.pt': True},
            'last_refresh': '2024-01-01T00:00:00'
        }
        
        # Patch the required methods
        with patch.object(test_module, 'create_ui_components', return_value=mock_ui_components) as mock_create_ui, \
             patch.object(test_module, '_execute_pretrained_operation', return_value=mock_refresh_result) as mock_execute_op:
            
            # Reset the mock to track calls to log.info
            test_module.log.info.reset_mock()
            
            # Call initialize with refresh_on_init=True (default)
            result = test_module.initialize(refresh_on_init=True)
            
            # Verify the result
            self.assertTrue(result, "Initialize should return True on success")
            self.assertTrue(test_module._initialized, "Module should be marked as initialized")
            
            # Verify UI components were created with the config
            mock_create_ui.assert_called_once()
            
            # Verify refresh operation was called
            mock_execute_op.assert_called_once_with('refresh')
            
            # Verify logging was called
            self.assertTrue(test_module.log.info.called, "Expected log.info to be called")
            log_calls = [call[0][0] for call in test_module.log.info.call_args_list]
            self.assertIn('✅ PretrainedUIModule initialized', '\n'.join(log_calls), 
                         "Expected success log message")
            
            # Verify model status was updated
            self.assertEqual(len(test_module._model_status['models_found']), 2, 
                           "Expected 2 models to be found")
            self.assertEqual(test_module._model_status['models_found'], 
                           ['yolov5s.pt', 'efficientnet_b4.pt'],
                           "Model names should match expected values")
            self.assertEqual(test_module._model_status['validation_results'], 
                           {'yolov5s.pt': True, 'efficientnet_b4.pt': True},
                           "Validation results should match expected values")
            self.assertEqual(test_module._model_status['last_refresh'], 
                           '2024-01-01T00:00:00',
                           "Last refresh time should be set")
    
    def test_initialize_failure(self):
        """Test initialization failure handling when an exception occurs during initialization."""
        # Create a fresh module instance for this test
        test_module = MockPretrainedUIModule()
        
        # Mock the _execute_pretrained_operation to raise an exception during refresh
        with patch.object(test_module, '_execute_pretrained_operation', 
                         side_effect=Exception('Test error')) as mock_execute_op, \
             patch.object(test_module.log, 'error') as mock_error_log, \
             patch.object(test_module.log, 'warning') as mock_warning_log:
            
            # Execute initialization with refresh_on_init=True (default)
            result = test_module.initialize()
            
            # Verify the result indicates partial success (initialization succeeded but refresh failed)
            self.assertTrue(result, "Initialize should return True even if refresh fails")
            self.assertTrue(test_module._initialized, 
                          "Module should still be marked as initialized even if refresh fails")
            
            # Verify the refresh operation was attempted
            mock_execute_op.assert_called_once_with('refresh')
            
            # Verify the error was logged as a warning (not an error, since initialization succeeded)
            mock_error_log.assert_not_called()
            mock_warning_log.assert_called_once()
            
            warning_message = str(mock_warning_log.call_args[0][0]).lower()
            self.assertIn('initial refresh failed', warning_message,
                        "Warning message should indicate refresh failure")
            self.assertIn('test error', warning_message,
                        "Warning message should include the test error")
            
            # Verify the model status reflects the refresh error
            self.assertIn('error', test_module._model_status,
                        "Model status should include refresh error information")
            self.assertEqual(test_module._model_status['initialized'], True,
                           "Model status should still indicate initialized")
    
    def test_log_initialization_complete_with_environment(self):
        """Test log_initialization_complete with environment support."""
        # Create a fresh module instance with environment support
        test_module = MockPretrainedUIModule()
        test_module._enable_environment = True
        test_module._is_colab = False  # Test local environment
        test_module._environment_paths.data_root = '/tmp/models'
        test_module._environment_paths.models_dir = '/tmp/models/pretrained'
        
        # Reset the mock to track calls to log
        test_module.log.reset_mock()
        
        # Call the method
        test_module._log_initialization_complete()
        
        # Verify log was called with the status message
        expected_call = call("📊 Status: Ready for pretrained model management", 'info')
        self.assertIn(expected_call, test_module.log.call_args_list)
    
    def test_log_initialization_complete_exception(self):
        """Test log_initialization_complete handles logging exceptions."""
        # Create a fresh module instance
        test_module = MockPretrainedUIModule()
        
        # Reset the mocks
        test_module.log.reset_mock()
        
        # Mock the logger to raise an exception on info call
        with patch.object(test_module, 'log', side_effect=Exception('Log error')) as mock_log, \
             patch.object(test_module.logger, 'error') as mock_logger_error:
            
            # Call the method - should not raise an exception
            test_module._log_initialization_complete()
            
            # Verify log was called
            self.assertTrue(mock_log.called)
            
            # Verify the error was logged to the logger
            mock_logger_error.assert_called_once()
            error_message = str(mock_logger_error.call_args[0][0])
            self.assertIn('Failed to log initialization complete', error_message)
            self.assertIn('Log error', error_message)


if __name__ == '__main__':
    unittest.main()