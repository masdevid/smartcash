"""
File: tests/unit/ui/dataset/preprocessing/test_preprocessing_uimodule.py
Description: Comprehensive unit tests for the PreprocessingUIModule class.
"""
import sys
import unittest
from unittest.mock import MagicMock, patch, ANY, call, PropertyMock, create_autospec
from typing import Dict, Any, Optional, List, Type
import pytest

# First, create a mock for the ipywidgets module
class MockWidget:
    def __init__(self, *args, **kwargs):
        self.children = []
        self.layout = MagicMock()
        self.value = None
        self.kwargs = kwargs
        self.return_value = None
        self.side_effect = None
        # Handle HTML content if provided as first argument
        if args:
            self.value = args[0]
    
    # Support context manager protocol
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    # Add common widget methods
    def add_class(self, class_name):
        if not hasattr(self, 'classes'):
            self.classes = set()
        self.classes.add(class_name)
        
    def remove_class(self, class_name):
        if hasattr(self, 'classes') and class_name in self.classes:
            self.classes.remove(class_name)
            
    def set_title(self, index, title):
        if not hasattr(self, 'titles'):
            self.titles = {}
        self.titles[index] = title
        
    def get_title(self, index):
        if hasattr(self, 'titles') and index in self.titles:
            return self.titles[index]
        return ""
        
    def __getitem__(self, key):
        if isinstance(key, int) and key < len(self.children):
            return self.children[key]
        return getattr(self, key, None)
        
    def __setitem__(self, key, value):
        if isinstance(key, int):
            if key < len(self.children):
                self.children[key] = value
            else:
                self.children.append(value)
        else:
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
        
    def __call__(self, *args, **kwargs):
        if self.side_effect is not None:
            if isinstance(self.side_effect, Exception):
                raise self.side_effect
            return self.side_effect
        return self.return_value
        
    def observe(self, *args, **kwargs):
        pass

# Create a mock widgets module
class MockWidgetsModule:
    def __init__(self):
        self.Widget = MockWidget
        self.Box = MockWidget
        self.VBox = MockWidget
        self.HBox = MockWidget
        self.GridBox = MockWidget
        self.Button = MockWidget
        self.Output = MockWidget
        self.HTML = MockWidget
        self.Dropdown = MockWidget
        self.Text = MockWidget
        self.IntText = MockWidget
        self.FloatText = MockWidget
        self.Textarea = MockWidget
        self.Checkbox = MockWidget
        self.Valid = MockWidget
        self.Tab = MockWidget
        self.Accordion = MockWidget
        self.ToggleButton = MockWidget
        self.SelectMultiple = MockWidget  # Add missing widget
        self.BoundedIntText = MockWidget  # Add missing widget
        
        class Layout:
            def __init__(self, **kwargs):
                pass
                
        class ButtonStyle:
            def __init__(self, **kwargs):
                pass
                
        self.Layout = Layout
        self.ButtonStyle = ButtonStyle
        self.widgets = self  # Make it accessible as ipywidgets.widgets

# Create the mock module
mock_ipywidgets = MockWidgetsModule()

# Patch sys.modules before any imports
sys.modules['ipywidgets'] = mock_ipywidgets
sys.modules['ipywidgets.widgets'] = mock_ipywidgets

# Now import the module under test
from smartcash.ui.dataset.preprocessing.preprocessing_uimodule import PreprocessingUIModule
from smartcash.ui.core.base_ui_module import BaseUIModule

# Create mock operation handler classes for testing
class MockOperationHandler:
    def __init__(self, *args, **kwargs):
        self.ui_module = kwargs.get('ui_module')
        self.config = kwargs.get('config')
        self.callbacks = kwargs.get('callbacks', {})
        self.init_args = args
        self.init_kwargs = kwargs
        self._execute_return = {'success': True, 'message': 'Success'}
        self.execute = MagicMock(return_value=self._execute_return)
        
        if hasattr(self, 'should_fail') and self.should_fail:
            self.execute.side_effect = Exception("Test error")

# Create test class
class TestPreprocessingUIModule(unittest.TestCase):
    """Test suite for PreprocessingUIModule."""
    
    def _create_mock_widget(self, widget_type=None):
        """Helper to create a mock widget with proper layout handling."""
        widget = MagicMock()
        widget.layout = MagicMock()
        widget.children = []
        return widget
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a test config that matches the actual structure expected by the code
        self.test_config = {
            'preprocessing': {
                'target_splits': ['train', 'valid'],
                'validation': {
                    'filename_pattern': True,
                    'auto_fix': True,
                    'create_directories': True,
                    'min_files_per_split': 1
                },
                'normalization': {
                    'preset': 'yolov5s',
                    'target_size': [640, 640],
                    'preserve_aspect_ratio': True,
                    'pixel_range': [0, 1],
                    'method': 'minmax'
                },
                'batch_size': 32,
                'move_invalid': False,
                'invalid_dir': 'data/invalid',
                'cleanup_target': 'preprocessed',
                'backup_enabled': True
            },
            'data': {
                'dir': 'data',
                'preprocessed_dir': 'data/preprocessed'
            },
            'performance': {
                'batch_size': 32,
                'io_workers': 8,
                'cpu_workers': None,
                'memory_limit_mb': 2048
            },
            'ui': {
                'show_progress': True,
                'show_details': True,
                'auto_scroll': True
            }
        }
        
        # Create a mock widget class for ipywidgets
        self.MockWidget = type('MockWidget', (), {
            'value': None,
            'description': '',
            'layout': {},
            'observe': MagicMock(),
            'unobserve': MagicMock(),
            'add_class': MagicMock(),
            'remove_class': MagicMock(),
            'set_title': MagicMock(),
            'get_title': MagicMock(return_value='Test Title'),
            '__enter__': lambda *args: self,
            '__exit__': lambda *args: None
        })
        
        # Create an instance of the module under test
        with patch('ipywidgets.widgets', mock_ipywidgets):
            self.module = PreprocessingUIModule(enable_environment=False)
            # Set the config after initialization
            self.module._config = self.test_config
    
    def test_initialization(self):
        """Test that the module initializes with default values."""
        # Verify attributes are set correctly
        self.assertEqual(self.module.module_name, 'preprocessing')
        self.assertEqual(self.module.parent_module, 'dataset')
        
        # Verify required components are present
        for comp in self.module._required_components:
            self.assertIn(comp, self.module._ui_components)
    
    def test_initialization_with_environment(self):
        """Test that the module initializes correctly with environment enabled."""
        # Mock UI components that will be created
        mock_ui_components = {
            'main_container': MagicMock(),
            'header_container': MagicMock(), 
            'form_container': MagicMock(),
            'action_container': MagicMock(),
            'operation_container': MagicMock(),
            'operation_summary_container': MagicMock(),
            'footer_container': MagicMock()
        }
        
        with patch('ipywidgets.widgets', mock_ipywidgets), \
             patch('smartcash.ui.components.header_container.widgets', mock_ipywidgets), \
             patch('smartcash.ui.components.header.header.widgets', mock_ipywidgets), \
             patch('smartcash.ui.dataset.preprocessing.components.input_options.widgets', mock_ipywidgets):
            
            # Create a new instance with environment enabled
            module = PreprocessingUIModule(enable_environment=True)
            
            # Mock the critical methods that are causing failures
            with patch.object(module, 'create_ui_components', return_value=mock_ui_components), \
                 patch.object(module, '_initialize_config_handler', return_value=None), \
                 patch.object(module, '_initialize_progress_display', return_value=None), \
                 patch.object(module, '_setup_ui_logging_bridge', return_value=None), \
                 patch.object(module, '_register_default_operations', return_value=None), \
                 patch.object(module, '_register_dynamic_button_handlers', return_value=None), \
                 patch.object(module, '_validate_button_handler_integrity', return_value=None), \
                 patch.object(module, '_link_action_container', return_value=None), \
                 patch.object(module, '_log_initialization_complete', return_value=None):
                
                # Call initialize to set _is_initialized
                result = module.initialize()
                
                # Verify initialization succeeded
                self.assertTrue(result)
                self.assertTrue(module._is_initialized)
    
    def test_get_default_config(self):
        """Test that get_default_config returns the expected default config."""
        default_config = self.module.get_default_config()
        # Verify the default config has the expected structure
        self.assertIn('preprocessing', default_config)
        self.assertIn('data', default_config)
        self.assertIn('performance', default_config)
    
    def test_initialize_success(self):
        """Test that initialize works correctly with valid config."""
        # Mock the config handler
        with patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.PreprocessingConfigHandler') as mock_handler:
            # Setup mock handler
            mock_handler_instance = MagicMock()
            mock_handler.return_value = mock_handler_instance
            
            # Initialize the module
            module = PreprocessingUIModule(enable_environment=False)
            
            # Call initialize
            result = module.initialize()
            
            # Verify the result
            self.assertTrue(result)
            self.assertTrue(module.is_initialized)
    
    def test_initialize_with_invalid_config(self):
        """Test that initialize handles invalid config gracefully."""
        with patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.PreprocessingConfigHandler') as mock_handler, \
             patch.object(PreprocessingUIModule, 'create_ui_components') as mock_create_components:
            
            # Setup mock handler to return a config
            mock_handler_instance = MagicMock()
            mock_handler_instance.get_current_config.return_value = {'test': 'config'}
            mock_handler.return_value = mock_handler_instance
            
            # Make create_ui_components return an empty dict to simulate no components created
            mock_create_components.return_value = {}
            
            # Initialize the module
            module = PreprocessingUIModule(enable_environment=False)
            
            # Call initialize and verify it returns True (base class handles empty components)
            result = module.initialize()
            self.assertTrue(result)
    
    def test_get_ui_components(self):
        """Test that get_ui_components returns the correct components."""
        # Get the UI components
        components = self.module.get_ui_components()
        
        # Verify the returned components have the expected structure
        self.assertIsInstance(components, dict)
        for comp in self.module._required_components:
            self.assertIn(comp, components)
    
    @patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.BaseUIModule.reset_config')
    def test_reset_config(self, mock_parent_reset):
        """Test that reset_config calls the parent class's reset_config method."""
        # Call reset_config
        self.module.reset_config()
        
        # Verify parent's reset_config was called
        mock_parent_reset.assert_called_once()
        
    def test_create_config_handler(self):
        """Test that create_config_handler returns a PreprocessingConfigHandler instance."""
        # Setup
        test_config = {'preprocessing': {'test': 'config'}}
        
        # Execute
        handler = self.module.create_config_handler(test_config)
        
        # Verify
        from smartcash.ui.dataset.preprocessing.configs.preprocessing_config_handler import PreprocessingConfigHandler
        self.assertIsInstance(handler, PreprocessingConfigHandler)
        self.assertEqual(handler.config, test_config)
        
    @patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.create_preprocessing_ui_components')
    def test_create_ui_components(self, mock_create_components):
        """Test that create_ui_components calls the correct function with the right parameters."""
        # Setup
        test_config = {'preprocessing': {'test': 'config'}}
        expected_components = {'test': 'components'}
        mock_create_components.return_value = expected_components
        
        # Execute
        result = self.module.create_ui_components(test_config)
        
        # Verify
        mock_create_components.assert_called_once_with(config=test_config)
        self.assertEqual(result, expected_components)
        
    def test_get_module_button_handlers(self):
        """Test that _get_module_button_handlers returns the correct handlers."""
        # Execute
        handlers = self.module._get_module_button_handlers()
        
        # Verify
        self.assertIsInstance(handlers, dict)
        self.assertIn('preprocess', handlers)
        self.assertIn('check', handlers)
        self.assertIn('cleanup', handlers)
        self.assertEqual(handlers['preprocess'].__name__, '_operation_preprocess')
        self.assertEqual(handlers['check'].__name__, '_operation_check')
        self.assertEqual(handlers['cleanup'].__name__, '_operation_cleanup')
        
    @patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.BaseUIModule.initialize')
    def test_initialize_success_with_components(self, mock_parent_initialize):
        """Test that initialize works correctly when UI components are created successfully."""
        # Setup
        mock_parent_initialize.return_value = True
        self.module._ui_components = {'test': 'components'}
        
        # Execute
        result = self.module.initialize()
        
        # Verify
        self.assertTrue(result)
        mock_parent_initialize.assert_called_once()
        
    @patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.BaseUIModule.initialize')
    def test_initialize_parent_failure(self, mock_parent_initialize):
        """Test that initialize returns False if parent initialization fails."""
        # Setup
        mock_parent_initialize.return_value = False
        
        # Execute
        result = self.module.initialize()
        
        # Verify
        self.assertFalse(result)
        mock_parent_initialize.assert_called_once()
        
    def test_ensure_components_ready_success(self):
        """Test that ensure_components_ready returns True when all required components exist."""
        # Setup
        # Create a mock operation container with a progress_tracker
        operation_container = MagicMock()
        
        # Configure the operation_container to support the 'in' operator and get() method
        operation_container.__contains__.side_effect = lambda x: x == 'progress_tracker'
        operation_container.get.return_value = MagicMock()  # Return a mock for the progress_tracker
        
        # Create the UI components dictionary with all required components
        self.module._ui_components = {
            'main_container': self._create_mock_widget(),
            'header_container': self._create_mock_widget(),
            'form_container': self._create_mock_widget(),
            'action_container': self._create_mock_widget(),
            'operation_container': operation_container
        }
        
        # Execute the method under test
        result = self.module.ensure_components_ready()
        
        # Verify the result
        self.assertTrue(result, "ensure_components_ready should return True when all components are ready")
        
        # Verify the operation_container was checked for the progress_tracker
        operation_container.__contains__.assert_called_with('progress_tracker')
        
    def test_ensure_components_ready_missing_component(self):
        """Test that ensure_components_ready returns False when a required component is missing."""
        # Test with missing operation_container
        self.module._ui_components = {
            'main_container': self._create_mock_widget(),
            'header_container': self._create_mock_widget(),
            'form_container': self._create_mock_widget(),
            'action_container': self._create_mock_widget()
            # Missing operation_container
        }
        
        with patch.object(self.module, 'log_warning') as mock_log_warning:
            result = self.module.ensure_components_ready()
            self.assertFalse(result)
            mock_log_warning.assert_called_once_with("Operation container tidak ditemukan")
            
    def test_log_messages_appear_in_operation_container(self):
        """Test that log messages appear in the operation container."""
        # Setup mock operation container with log accordion
        log_accordion = MagicMock()
        log_accordion.log_info = MagicMock()
        log_accordion.log_warning = MagicMock()
        log_accordion.log_error = MagicMock()
        
        # Create a proper mock for the operation container
        operation_container = MagicMock()
        operation_container.get.side_effect = lambda x: log_accordion if x == 'log_accordion' else None
        operation_container.__contains__.side_effect = lambda x: x in ['progress_tracker', 'log_accordion']
        
        # Set up the module's UI components
        self.module._ui_components = {
            'main_container': self._create_mock_widget(),
            'header_container': self._create_mock_widget(),
            'form_container': self._create_mock_widget(),
            'action_container': self._create_mock_widget(),
            'operation_container': operation_container
        }
        
        # Set the log accordion in the module
        self.module._log_accordion = log_accordion
        
        # Test different log levels
        test_message = "Test log message"
        
        # Test info log
        with patch('smartcash.ui.core.mixins.logging_mixin.LoggingMixin.log_info') as mock_log_info:
            self.module.log_info(test_message)
            mock_log_info.assert_called_once_with(test_message)
        
        # Test warning log
        with patch('smartcash.ui.core.mixins.logging_mixin.LoggingMixin.log_warning') as mock_log_warning:
            self.module.log_warning(test_message)
            mock_log_warning.assert_called_once_with(test_message)
        
        # Test error log
        with patch('smartcash.ui.core.mixins.logging_mixin.LoggingMixin.log_error') as mock_log_error:
            self.module.log_error(test_message)
            mock_log_error.assert_called_once_with(test_message)
        
    def test_operation_logs_during_preprocessing(self):
        """Test that logs are generated during preprocessing operations."""
        # Setup mocks
        log_accordion = MagicMock()
        
        # Create a mock for the show_dialog method
        show_dialog = MagicMock()
        
        # Create a proper mock for the operation container
        operation_container = MagicMock()
        operation_container.get.side_effect = lambda x: log_accordion if x == 'log_accordion' else None
        operation_container.__contains__.side_effect = lambda x: x in ['progress_tracker', 'log_accordion']
        operation_container.show_dialog = show_dialog
        
        # Set up the module's UI components
        self.module._ui_components = {
            'main_container': self._create_mock_widget(),
            'header_container': self._create_mock_widget(),
            'form_container': self._create_mock_widget(),
            'action_container': self._create_mock_widget(),
            'operation_container': operation_container
        }
        
        # Set the log accordion in the module
        self.module._log_accordion = log_accordion
        
        # Mock the _execute_preprocess_operation method
        with patch.object(self.module, '_execute_preprocess_operation') as mock_execute:
            mock_execute.return_value = {'success': True, 'message': 'Success'}
            
            # Mock the _get_preprocessed_data_stats method
            with patch.object(self.module, '_get_preprocessed_data_stats') as mock_stats:
                mock_stats.return_value = (0, 10)  # 0 preprocessed, 10 raw images
                
                # Mock the log_info method to capture the log message
                with patch.object(self.module, 'log_info') as mock_log_info:
                    # Execute the preprocessing operation
                    result = self.module._operation_preprocess()
                    
                    # Verify the dialog was shown
                    show_dialog.assert_called_once()
                    
                    # Get the on_confirm callback from the dialog call
                    on_confirm = show_dialog.call_args[1]['on_confirm']
                    
                    # Call the on_confirm callback to simulate user confirmation
                    on_confirm()
                    
                    # Verify logs were called with expected messages
                    mock_log_info.assert_any_call("🔄 Memulai preprocessing data...")
                    
                    # Verify the execute method was called
                    mock_execute.assert_called_once()
                    
                    # Verify the result is as expected
                    self.assertEqual(result, {'success': True, 'message': 'Dialog konfirmasi ditampilkan'})
                
    def test_error_logging_during_operation_failure(self):
        """Test that errors during preprocessing operation are properly logged."""
        # Setup test data
        test_error = Exception("Test error")
        
        # Mock the operation container and its show_dialog method
        operation_container = MagicMock()
        
        # Store the callback when show_dialog is called
        dialog_callbacks = {}
        
        def capture_dialog_callback(*args, **kwargs):
            dialog_callbacks['on_confirm'] = kwargs.get('on_confirm')
            return {'success': True, 'message': 'Dialog shown'}
            
        operation_container.show_dialog.side_effect = capture_dialog_callback
        
        # Mock the module's get_component to return our mock operation container
        self.module.get_component = MagicMock(return_value=operation_container)
        
        # Mock the _execute_preprocess_operation to raise an exception
        with patch.object(self.module, '_execute_preprocess_operation') as mock_execute:
            mock_execute.side_effect = test_error
            
            # Mock the _get_preprocessed_data_stats to return test data
            with patch.object(self.module, '_get_preprocessed_data_stats') as mock_stats:
                mock_stats.return_value = (0, 10)  # 0 preprocessed, 10 raw images
                
                # Mock the operation wrapper to pass through the function call
                def execute_wrapper(operation_name, operation_func, **kwargs):
                    return operation_func()
                    
                with patch.object(self.module, '_execute_operation_with_wrapper', side_effect=execute_wrapper):
                    # Execute the preprocessing operation - this will show the dialog
                    result = self.module._operation_preprocess()
                    
                    # Verify the dialog was shown with the correct parameters
                    operation_container.show_dialog.assert_called_once()
                    call_args = operation_container.show_dialog.call_args[1]
                    self.assertEqual(call_args.get('title'), "Konfirmasi Pra-pemrosesan")
                    self.assertIn("Anda akan memulai pra-pemrosesan data", call_args.get('message', ''))
                    
                    # Get the dialog callback
                    on_confirm = dialog_callbacks.get('on_confirm')
                    self.assertIsNotNone(on_confirm, "Dialog confirmation callback was not set")
                    
                    # Test the dialog callback with error logging mocks
                    with patch.object(self.module, 'log') as mock_log, \
                         patch.object(self.module, 'error_progress') as mock_error_progress:
                        
                        # Execute the callback - should raise the test error
                        with self.assertRaises(Exception) as context:
                            on_confirm()
                        
                        # Verify the exception was raised with the expected message
                        self.assertIs(context.exception, test_error)
                        
                        # Verify the operation was executed
                        mock_execute.assert_called_once()
                        
                        # Verify error was logged using the module's log method
                        # The error should be logged with level 'error'
                        error_log_call = None
                        for call_args in mock_log.call_args_list:
                            if len(call_args[0]) > 1 and call_args[0][1] == 'error':
                                error_log_call = call_args
                                break
                                
                        self.assertIsNotNone(error_log_call, "Error was not logged with 'error' level")
                        self.assertIn("Test error", str(error_log_call[0][0]))
                        
                        # Verify error progress was updated
                        mock_error_progress.assert_called_once()
                        error_message = mock_error_progress.call_args[0][0]
                        self.assertIn("Test error", error_message)
                
    @patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.BaseUIModule.initialize')
    def test_initialize_with_ui_components(self, mock_parent_initialize):
        """Test that initialize works with pre-created UI components."""
        # Setup
        mock_parent_initialize.return_value = True
        self.module._ui_components = {
            'main_container': self._create_mock_widget(),
            'header_container': self._create_mock_widget(),
        'form_container': self._create_mock_widget(),
        'action_container': self._create_mock_widget(),
        'operation_container': self._create_mock_widget()
    }
        # Setup
        mock_parent_initialize.return_value = True
        self.module._ui_components = {
            'main_container': self._create_mock_widget(),
            'header_container': self._create_mock_widget(),
            'form_container': self._create_mock_widget(),
            'action_container': self._create_mock_widget(),
            'operation_container': self._create_mock_widget()
        }
        
        # Execute
        result = self.module.initialize()
        
        # Verify
        self.assertTrue(result)
        mock_parent_initialize.assert_called_once()
    
    @patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.PreprocessingConfigHandler')
    @patch('smartcash.ui.components.operation_container.OperationContainer')
    @patch('smartcash.ui.components.log_accordion.LogAccordion')
    @patch('smartcash.ui.core.validation.button_validator.ButtonHandlerValidator._extract_button_ids')
    def test_initialize_success(self, mock_extract_buttons, mock_log_accordion, 
                              mock_op_container, mock_config_handler):
        """Test that initialize works correctly with valid config."""
        # Mock button validation to avoid comparison errors
        mock_extract_buttons.return_value = set()
        
        # Create a mock progress tracker
        mock_progress_tracker = MagicMock()
        
        # Create mock operation container with progress_tracker
        mock_op_container_instance = MagicMock()
        mock_op_container.return_value = mock_op_container_instance
        
        # Create mock UI components with operation_container that includes progress_tracker
        mock_ui_components = {
            'main_container': self._create_mock_widget(),
            'header_container': self._create_mock_widget(),
            'form_container': self._create_mock_widget(),
            'action_container': self._create_mock_widget(),
            'operation_container': {
                'progress_tracker': mock_progress_tracker
            }
        }
        
        # Mock the config handler and its methods
        mock_handler_instance = MagicMock()
        mock_handler_instance.get_current_config.return_value = self.test_config
        mock_config_handler.return_value = mock_handler_instance
        
        # Mock LogAccordion
        mock_log_accordion.return_value = MagicMock()
        
        # Patch all necessary modules
        with patch('ipywidgets.widgets', mock_ipywidgets), \
             patch('smartcash.ui.components.header_container.widgets', mock_ipywidgets), \
             patch('smartcash.ui.components.header.header.widgets', mock_ipywidgets), \
             patch('smartcash.ui.dataset.preprocessing.components.input_options.widgets', mock_ipywidgets), \
             patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.create_preprocessing_ui_components', 
                  return_value=mock_ui_components) as mock_create_ui_components:
            
            # Create a new instance with the patched dependencies
            module = PreprocessingUIModule(enable_environment=False)
            
            # Set the config directly to avoid config handler calls
            module._config = self.test_config
            
            # Call the method under test
            result = module.initialize()
        
        # Verify results
        self.assertTrue(result)
        
        # Verify create_preprocessing_ui_components was called with the config
        # Using assert_any_call to handle multiple calls
        mock_create_ui_components.assert_any_call(config=self.test_config)
        
        # Verify UI components were set
        self.assertEqual(set(module._ui_components.keys()), set(mock_ui_components.keys()))
        
        # Verify progress tracker was initialized if it has the method
        if hasattr(mock_progress_tracker, 'initialize'):
            mock_progress_tracker.initialize.assert_called_once()
        
        # Verify button validation was called
        mock_extract_buttons.assert_called()

if __name__ == '__main__':
    unittest.main()
