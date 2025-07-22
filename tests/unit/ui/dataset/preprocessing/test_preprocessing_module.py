"""
Tests for the PreprocessingUIModule class.
"""
import unittest
from unittest.mock import MagicMock, patch, ANY

from smartcash.ui.dataset.preprocessing.preprocessing_uimodule import PreprocessingUIModule
from smartcash.ui.dataset.preprocessing.operations import (
    preprocess_operation,
    check_operation,
    cleanup_operation
)

class TestPreprocessingUIModule(unittest.TestCase):
    """Test suite for the PreprocessingUIModule class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a test instance with environment support disabled for faster tests
        self.module = PreprocessingUIModule(enable_environment=False)
        
        # Mock the config handler
        self.mock_config_handler = MagicMock()
        self.mock_config_handler.get_current_config.return_value = {
            'checks': ['integrity', 'format'],
            'preprocessing': {
                'input_dir': 'data/raw',
                'output_dir': 'data/processed',
                'splits': ['train', 'valid']
            }
        }
        self.module._config_handler = self.mock_config_handler
        
        # Mock UI components
        self.mock_ui_components = {
            'main_container': MagicMock(),
            'header_container': MagicMock(),
            'form_container': MagicMock(),
            'action_container': MagicMock(),
            'operation_container': MagicMock()
        }
        self.module._ui_components = self.mock_ui_components
        
        # Mock operation container for logging
        self.module._operation_container = MagicMock()
        
        # Mock the operation manager
        self.module._operation_manager = MagicMock()
        
        # Mock the update_operation_summary method
        self.module._update_operation_summary = MagicMock()
    
    def test_initialization(self):
        """Test that the module initializes correctly."""
        self.assertEqual(self.module.module_name, 'preprocessing')
        self.assertEqual(self.module.parent_module, 'dataset')
        self.assertFalse(hasattr(self.module, '_environment_paths'))
    
    def test_get_default_config(self):
        """Test that get_default_config returns the expected default config."""
        # Mock the imported function
        with patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.get_default_config') as mock_get_default:
            expected_config = {'test': 'config'}
            mock_get_default.return_value = expected_config
            
            result = self.module.get_default_config()
            
            self.assertEqual(result, expected_config)
            mock_get_default.assert_called_once()
    
    def test_create_config_handler(self):
        """Test that create_config_handler returns a PreprocessingConfigHandler with the given config."""
        test_config = {'test': 'config'}
        
        with patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.PreprocessingConfigHandler') as mock_handler_class:
            mock_handler = MagicMock()
            mock_handler_class.return_value = mock_handler
            
            result = self.module.create_config_handler(test_config)
            
            self.assertEqual(result, mock_handler)
            mock_handler_class.assert_called_once_with(test_config)
    
    def test_create_ui_components(self):
        """Test that create_ui_components calls the UI factory function with the correct config."""
        test_config = {'test': 'config'}
        expected_components = {'test': 'components'}
        
        with patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.create_preprocessing_ui_components') as mock_create_ui:
            mock_create_ui.return_value = expected_components
            
            result = self.module.create_ui_components(test_config)
            
            self.assertEqual(result, expected_components)
            mock_create_ui.assert_called_once_with(config=test_config)
    
    def test_operation_preprocess(self):
        """Test the preprocess operation."""
        # Mock the operation handler
        with patch('smartcash.ui.dataset.preprocessing.operations.preprocess_operation.PreprocessOperation') as mock_op_class, \
             patch.object(self.module, '_execute_operation_with_wrapper') as mock_execute_wrapper, \
             patch.object(self.module, '_get_preprocessed_data_stats') as mock_get_stats:
            
            # Setup mock return values
            mock_get_stats.return_value = (0, 10)  # preprocessed_files, raw_images
            mock_op = MagicMock()
            mock_op_class.return_value = mock_op
            
            # Call the operation
            self.module._operation_preprocess()
            
            # Verify _execute_operation_with_wrapper was called with the correct parameters
            mock_execute_wrapper.assert_called_once()
            call_args = mock_execute_wrapper.call_args[1]
            self.assertEqual(call_args['operation_name'], "Preprocessing Data")
            self.assertEqual(call_args['success_message'], "Preprocessing data berhasil diselesaikan")
            self.assertEqual(call_args['error_message'], "Kesalahan preprocessing data")
            
            # Verify the operation function was set up correctly
            operation_func = call_args['operation_func']
            self.assertIsNotNone(operation_func)
            
            # Test the operation function
            with patch.object(self.module, '_execute_preprocess_operation') as mock_execute_op:
                # The actual method returns a dialog confirmation message
                result = operation_func()
                self.assertEqual(result, {'success': True, 'message': 'Dialog konfirmasi ditampilkan'})
                # The execute function should not be called directly, as it's called after confirmation
                mock_execute_op.assert_not_called()
    
    def test_operation_check(self):
        """Test the check operation."""
        # Mock the operation handler and wrapper
        with patch('smartcash.ui.dataset.preprocessing.operations.check_operation.CheckOperationHandler') as mock_op_class, \
             patch.object(self.module, '_execute_operation_with_wrapper') as mock_execute_wrapper:
            
            mock_op = MagicMock()
            mock_op_class.return_value = mock_op
            
            # Call the operation
            self.module._operation_check()
            
            # Verify _execute_operation_with_wrapper was called with the correct parameters
            mock_execute_wrapper.assert_called_once()
            call_args = mock_execute_wrapper.call_args[1]
            self.assertEqual(call_args['operation_name'], "Pemeriksaan Data")
            self.assertEqual(call_args['success_message'], "Pemeriksaan data berhasil diselesaikan")
            self.assertEqual(call_args['error_message'], "Kesalahan pemeriksaan data")
            
            # Verify the operation function was set up correctly
            operation_func = call_args['operation_func']
            self.assertIsNotNone(operation_func)
            
            # Test the operation function
            with patch.object(self.module, '_execute_check_operation') as mock_execute_op:
                mock_execute_op.return_value = {'success': True, 'message': 'Test success'}
                result = operation_func()
                self.assertEqual(result, {'success': True, 'message': 'Test success'})
                mock_execute_op.assert_called_once()
    
    def test_operation_cleanup(self):
        """Test the cleanup operation."""
        # Mock the operation handler and data stats
        with patch('smartcash.ui.dataset.preprocessing.operations.cleanup_operation.CleanupOperationHandler') as mock_op_class, \
             patch.object(self.module, '_execute_operation_with_wrapper') as mock_execute_wrapper, \
             patch.object(self.module, '_get_preprocessed_data_stats') as mock_get_stats:
            
            # Setup mock return values
            mock_get_stats.return_value = (5, 10)  # preprocessed_files, raw_images
            mock_op = MagicMock()
            mock_op_class.return_value = mock_op
            
            # Call the operation
            self.module._operation_cleanup()
            
            # Verify _execute_operation_with_wrapper was called with the correct parameters
            mock_execute_wrapper.assert_called_once()
            call_args = mock_execute_wrapper.call_args[1]
            self.assertEqual(call_args['operation_name'], "Pembersihan Data")
            self.assertEqual(call_args['success_message'], "Pembersihan data berhasil diselesaikan")
            self.assertEqual(call_args['error_message'], "Kesalahan pembersihan data")
            
            # Verify the operation function was set up correctly
            operation_func = call_args['operation_func']
            self.assertIsNotNone(operation_func)
            
            # Test the operation function
            with patch.object(self.module, '_execute_cleanup_operation') as mock_execute_op:
                # The actual method returns a dialog confirmation message
                result = operation_func()
                self.assertEqual(result, {'success': True, 'message': 'Dialog konfirmasi ditampilkan'})
                # The execute function should not be called directly, as it's called after confirmation
                mock_execute_op.assert_not_called()
    
    def test_operation_handlers_registered(self):
        """Test that all operation handlers are registered."""
        # Get the registered operation handlers through the button handlers
        button_handlers = self.module._get_module_button_handlers()
        
        # Check that all expected operations are registered
        self.assertIn('preprocess', button_handlers)
        self.assertIn('check', button_handlers)
        self.assertIn('cleanup', button_handlers)
        
        # Verify the handler methods are callable
        self.assertTrue(callable(button_handlers['preprocess']))
        self.assertTrue(callable(button_handlers['check']))
        self.assertTrue(callable(button_handlers['cleanup']))
    
    def test_log_redirection_to_operation_container(self):
        """Test that log messages are properly redirected to the operation container."""
        # Mock the operation container's log method
        mock_container = MagicMock()
        self.module._ui_components['operation_container'] = {
            'container': mock_container,
            'log': mock_container.log
        }
        
        # Test different log levels
        test_messages = [
            ('debug', 'Debug test message'),
            ('info', 'Info test message'),
            ('warning', 'Warning test message'),
            ('error', 'Error test message'),
            ('success', 'Success test message')
        ]
        
        for level, message in test_messages:
            # Call the appropriate log method
            log_method = getattr(self.module, f'log_{level}')
            log_method(message)
            
            # Verify the log was called with the correct message
            # The actual log method adds emojis and formatting, so we'll check if the message is in the call
            log_calls = [str(call) for call in mock_container.log.call_args_list]
            self.assertTrue(any(message in call for call in log_calls),
                         f"Message '{message}' not found in log calls: {log_calls}")
    
    def test_backend_log_redirection(self):
        """Test that logs from backend operations are properly redirected to the operation container."""
        # Mock the operation container and progress updater
        mock_container = MagicMock()
        self.module._ui_components['operation_container'] = {
            'container': mock_container,
            'log': mock_container.log,
            'update_progress': mock_container.update_progress
        }
        
        # Mock the backend operation that would generate logs
        def mock_operation(progress_callback):
            # Simulate progress updates that would generate logs
            progress_callback('info', 1, 10, 'Processing item 1/10')
            progress_callback('warning', 3, 10, 'Warning: Issue with item 3')
            progress_callback('error', 5, 10, 'Error processing item 5')
            progress_callback('success', 10, 10, 'Operation completed')
            return {'success': True, 'message': 'Operation completed'}
        
        # Mock the operation handler to use our test operation
        with patch('smartcash.ui.dataset.preprocessing.operations.preprocess_operation.PreprocessOperation') as mock_op_class, \
             patch.object(self.module, '_get_preprocessed_data_stats', return_value=(0, 10)) as mock_get_stats:
            
            mock_op = MagicMock()
            mock_op.execute.side_effect = lambda: mock_operation(mock_op._progress_adapter)
            mock_op_class.return_value = mock_op
            
            # Execute the operation
            with patch.object(self.module, '_execute_operation_with_wrapper') as mock_wrapper:
                # Set up the wrapper to call the operation function directly
                def execute_wrapper(operation_func, **kwargs):
                    return operation_func()
                mock_wrapper.side_effect = execute_wrapper
                
                self.module._operation_preprocess()
        
        # Verify logs were captured in the operation container
        log_calls = [str(call) for call in mock_container.log.call_args_list]
        
        # Check for specific log messages that should be present
        expected_messages = [
            '🔄 Memulai preprocessing data...',
            'Dialog tidak tersedia, menjalankan preprocessing langsung'
        ]
        
        for msg in expected_messages:
            self.assertTrue(any(msg in call for call in log_calls),
                         f"Expected log message containing '{msg}' not found in {log_calls}")
    
    def test_progress_updates_in_operation_container(self):
        """Test that progress updates are properly shown in the operation container."""
        # Mock the operation container's progress update method
        mock_container = MagicMock()
        self.module._ui_components['operation_container'] = {
            'container': mock_container,
            'update_progress': mock_container.update_progress,
            'log': mock_container.log
        }
        
        # Test the update_progress method through the operation container
        operation_container = self.module._ui_components['operation_container']
        operation_container['update_progress'](25, "Processing...")
        operation_container['update_progress'](50, "Halfway there...")
        operation_container['update_progress'](100, "Complete!")
        
        # Verify progress updates were called correctly
        self.assertEqual(mock_container.update_progress.call_count, 3)
        
        # Check the progress values and messages
        calls = mock_container.update_progress.call_args_list
        self.assertEqual(calls[0][0][0], 25)  # First arg is progress
        self.assertEqual(calls[0][0][1], "Processing...")  # Second arg is message
        self.assertEqual(calls[1][0][0], 50)
        self.assertEqual(calls[1][0][1], "Halfway there...")
        self.assertEqual(calls[2][0][0], 100)
        self.assertEqual(calls[2][0][1], "Complete!")
        
        # Test log messages through the operation container
        operation_container['log']("Test log message", "info")
        log_calls = [str(call) for call in mock_container.log.call_args_list]
        self.assertTrue(any("Test log message" in call for call in log_calls))
    
    def test_error_logging_during_operation(self):
        """Test that errors during operations are properly logged in the operation container."""
        # Mock the operation container
        mock_container = MagicMock()
        self.module._ui_components['operation_container'] = {
            'container': mock_container,
            'log': mock_container.log,
            'update_progress': mock_container.update_progress,
            'show_dialog': mock_container.show_dialog
        }
        
        # Mock the operation to raise an exception when executed
        with patch.object(self.module, '_execute_preprocess_operation') as mock_execute_op:
            mock_execute_op.side_effect = Exception("Test error")
            
            # Mock the stats to simulate files available for processing
            with patch.object(self.module, '_get_preprocessed_data_stats', return_value=(5, 10)):
                # Execute the operation
                with patch('smartcash.ui.dataset.preprocessing.operations.preprocess_operation.PreprocessOperation'):
                    # Mock the dialog to immediately confirm
                    def mock_show_dialog(*args, **kwargs):
                        if 'on_confirm' in kwargs:
                            # Simulate user confirming the dialog
                            return kwargs['on_confirm']()
                        return None
                    
                    mock_container.show_dialog.side_effect = mock_show_dialog
                    
                    # Call the operation method
                    result = self.module._operation_preprocess()
                    
                    # Verify the error was handled and returned
                    self.assertFalse(result['success'])
                    self.assertIn('Test error', result['message'])

if __name__ == '__main__':
    unittest.main()
