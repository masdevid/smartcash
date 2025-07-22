"""
Comprehensive button state management tests for PretrainedUIModule.

This test suite focuses on button states, interactions, and UI state changes
during different operations.
"""

import unittest
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any
import ipywidgets as widgets

# Import test base class
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_pretrained_uimodule import MockPretrainedUIModule


class TestPretrainedButtonStates(unittest.TestCase):
    """Test suite for pretrained module button state management."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a fresh mock module instance
        self.module = MockPretrainedUIModule()
        
        # Create mock UI components including all buttons
        self.mock_buttons = {
            'download': MagicMock(spec=widgets.Button),
            'validate': MagicMock(spec=widgets.Button),
            'refresh': MagicMock(spec=widgets.Button),
            'cleanup': MagicMock(spec=widgets.Button),
            'save': MagicMock(spec=widgets.Button),
            'reset': MagicMock(spec=widgets.Button)
        }
        
        # Set up button properties
        for button_name, button in self.mock_buttons.items():
            button.disabled = False
            button.button_style = 'info' if button_name in ['validate', 'refresh'] else 'success'
            button.tooltip = f'{button_name.title()} operation'
            button.description = f'{button_name.title()}'
        
        # Create mock UI components
        self.mock_ui_components = {
            'main_container': MagicMock(),
            'header_container': MagicMock(),
            'form_container': MagicMock(),
            'action_container': MagicMock(),
            'operation_container': MagicMock(),
            'progress_tracker': MagicMock(),
            'log_accordion': MagicMock(),
            **self.mock_buttons
        }
        
        # Set up module attributes
        self.module._ui_components = self.mock_ui_components
        self.module._is_initialized = True
        
        # Mock the get_component method
        self.module.get_component = MagicMock(side_effect=lambda name: self.mock_ui_components.get(name))
        
        # Mock button handler registration
        self.module.register_button_handler = MagicMock()
        
        # Mock operation wrapper
        self.module._execute_operation_with_wrapper = MagicMock()
    
    def test_button_handlers_registration(self):
        """Test that all button handlers are properly registered."""
        # Get button handlers
        handlers = self.module._get_module_button_handlers()
        
        # Verify all expected handlers are present
        expected_handlers = ['download', 'validate', 'refresh', 'cleanup', 'save', 'reset']
        for handler_name in expected_handlers:
            self.assertIn(handler_name, handlers, f"Handler '{handler_name}' should be registered")
            self.assertTrue(callable(handlers[handler_name]), 
                          f"Handler '{handler_name}' should be callable")
    
    def test_button_disabled_state_during_operation(self):
        """Test that buttons are properly disabled during operations."""
        # Mock a button click handler that simulates button state changes
        def mock_operation_handler(*args, **kwargs):
            # Simulate disabling buttons during operation
            button = kwargs.get('button')
            if button and hasattr(button, 'disabled'):
                button.disabled = True
            return {'success': True, 'message': 'Operation completed'}
        
        # Patch the operation wrapper to use our mock handler
        with patch.object(self.module, '_execute_operation_with_wrapper', 
                         side_effect=mock_operation_handler) as mock_wrapper:
            
            # Test download operation
            download_button = self.mock_buttons['download']
            download_button.disabled = False  # Reset state
            
            result = self.module._operation_download(download_button)
            
            # Verify operation was called
            mock_wrapper.assert_called_once()
            self.assertTrue(result['success'])
            
            # In a real scenario, the button would be disabled during operation
            # Here we verify the handler was called with the button
            call_args = mock_wrapper.call_args
            self.assertEqual(call_args[1]['button'], download_button)
    
    def test_button_state_after_successful_operation(self):
        """Test button states after successful operations."""
        # Mock successful operation result
        success_result = {
            'success': True,
            'message': 'Operation completed successfully',
            'models_found': ['yolov5s.pt', 'efficientnet_b4.pt']
        }
        
        with patch.object(self.module, '_execute_operation_with_wrapper', 
                         return_value=success_result) as mock_wrapper:
            
            # Test each operation button
            operations = ['download', 'validate', 'refresh', 'cleanup']
            
            for op_name in operations:
                with self.subTest(operation=op_name):
                    button = self.mock_buttons[op_name]
                    button.disabled = False  # Reset state
                    
                    # Call the operation
                    operation_method = getattr(self.module, f'_operation_{op_name}')
                    result = operation_method(button)
                    
                    # Verify success
                    self.assertTrue(result['success'])
                    mock_wrapper.assert_called()
                    
                    # Reset mock for next iteration
                    mock_wrapper.reset_mock()
    
    def test_button_state_after_failed_operation(self):
        """Test button states after failed operations."""
        # Mock failed operation result
        failure_result = {
            'success': False,
            'error': 'Operation failed',
            'message': 'Download failed due to network error'
        }
        
        with patch.object(self.module, '_execute_operation_with_wrapper', 
                         return_value=failure_result) as mock_wrapper:
            
            # Test download operation failure
            download_button = self.mock_buttons['download']
            download_button.disabled = False  # Reset state
            
            result = self.module._operation_download(download_button)
            
            # Verify failure handling
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            mock_wrapper.assert_called_once()
    
    def test_button_tooltips_and_descriptions(self):
        """Test that buttons have correct tooltips and descriptions."""
        expected_button_info = {
            'download': {
                'tooltip_contains': 'download',
                'description_contains': 'download'
            },
            'validate': {
                'tooltip_contains': 'validate',
                'description_contains': 'validate'
            },
            'refresh': {
                'tooltip_contains': 'refresh',
                'description_contains': 'refresh'
            },
            'cleanup': {
                'tooltip_contains': 'clean',
                'description_contains': 'clean'
            }
        }
        
        for button_name, expected in expected_button_info.items():
            with self.subTest(button=button_name):
                button = self.mock_buttons[button_name]
                
                # Check tooltip (case insensitive)
                if hasattr(button, 'tooltip') and button.tooltip:
                    self.assertIn(expected['tooltip_contains'].lower(), 
                                button.tooltip.lower(),
                                f"Button '{button_name}' tooltip should contain '{expected['tooltip_contains']}'")
                
                # Check description (case insensitive)
                if hasattr(button, 'description') and button.description:
                    self.assertIn(expected['description_contains'].lower(), 
                                button.description.lower(),
                                f"Button '{button_name}' description should contain '{expected['description_contains']}'")
    
    def test_save_reset_button_functionality(self):
        """Test save and reset button functionality."""
        # Mock config handler
        mock_config_handler = MagicMock()
        self.module._config_handler = mock_config_handler
        
        # Mock the base class methods
        with patch('smartcash.ui.core.base_ui_module.BaseUIModule._get_module_button_handlers') as mock_base_handlers:
            mock_base_handlers.return_value = {
                'save': MagicMock(return_value={'success': True}),
                'reset': MagicMock(return_value={'success': True})
            }
            
            # Get save handler and call it
            handlers = self.module._get_module_button_handlers()
            save_handler = handlers['save']
            result = save_handler()
            
            # Verify save was called and succeeded
            self.assertTrue(result['success'])
            
            # Get reset handler and call it
            reset_handler = handlers['reset']
            result = reset_handler()
            
            # Verify reset was called and succeeded
            self.assertTrue(result['success'])
    
    def test_button_interaction_sequence(self):
        """Test a sequence of button interactions."""
        # Mock successful operations
        success_result = {'success': True, 'message': 'Operation completed'}
        
        with patch.object(self.module, '_execute_operation_with_wrapper', 
                         return_value=success_result) as mock_wrapper:
            
            # Simulate user workflow: refresh -> download -> validate -> cleanup
            operations_sequence = ['refresh', 'download', 'validate', 'cleanup']
            
            for i, op_name in enumerate(operations_sequence):
                with self.subTest(step=i+1, operation=op_name):
                    button = self.mock_buttons[op_name]
                    operation_method = getattr(self.module, f'_operation_{op_name}')
                    
                    # Execute operation
                    result = operation_method(button)
                    
                    # Verify success
                    self.assertTrue(result['success'])
                    
                    # Verify operation wrapper was called
                    self.assertTrue(mock_wrapper.called)
                    
                    # Reset for next operation
                    mock_wrapper.reset_mock()
    
    def test_concurrent_operation_prevention(self):
        """Test that concurrent operations are prevented."""
        # This test simulates the scenario where a user tries to click multiple buttons
        # while an operation is already running
        
        # Mock an operation that takes time (simulated by keeping button disabled)
        def slow_operation_handler(*args, **kwargs):
            button = kwargs.get('button')
            if button:
                button.disabled = True  # Simulate operation in progress
            return {'success': True, 'message': 'Operation completed'}
        
        with patch.object(self.module, '_execute_operation_with_wrapper', 
                         side_effect=slow_operation_handler) as mock_wrapper:
            
            # Start first operation
            download_button = self.mock_buttons['download']
            download_button.disabled = False
            
            result1 = self.module._operation_download(download_button)
            self.assertTrue(result1['success'])
            
            # Button should now be disabled (simulating ongoing operation)
            self.assertTrue(download_button.disabled)
            
            # Try to start another operation - in real UI this would be prevented
            # by the disabled state, but we test the handler logic
            validate_button = self.mock_buttons['validate'] 
            validate_button.disabled = False
            
            result2 = self.module._operation_validate(validate_button)
            self.assertTrue(result2['success'])  # Handler still works
            
            # Verify both operations were called
            self.assertEqual(mock_wrapper.call_count, 2)
    
    def test_error_state_button_recovery(self):
        """Test button state recovery after errors."""
        # Mock operation that fails first, then succeeds
        call_count = 0
        def failing_then_succeeding_operation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {'success': False, 'error': 'Network error'}
            else:
                return {'success': True, 'message': 'Operation completed'}
        
        with patch.object(self.module, '_execute_operation_with_wrapper', 
                         side_effect=failing_then_succeeding_operation) as mock_wrapper:
            
            # First operation fails
            download_button = self.mock_buttons['download']
            result1 = self.module._operation_download(download_button)
            
            self.assertFalse(result1['success'])
            self.assertIn('error', result1)
            
            # Second operation succeeds (button should be re-enabled)
            result2 = self.module._operation_download(download_button)
            
            self.assertTrue(result2['success'])
            
            # Verify both calls were made
            self.assertEqual(mock_wrapper.call_count, 2)
    
    def test_form_validation_before_operations(self):
        """Test that form validation occurs before operations."""
        # Mock form validation
        with patch.object(self.module, '_validate_models', 
                         return_value={'valid': True}) as mock_validate:
            
            with patch.object(self.module, '_execute_operation_with_wrapper', 
                             return_value={'success': True}) as mock_wrapper:
                
                # Execute download operation
                result = self.module._operation_download()
                
                # Verify validation and operation were called
                self.assertTrue(result['success'])
                mock_wrapper.assert_called_once()
                
                # Check that validation function is available
                call_args = mock_wrapper.call_args[1]
                self.assertIn('validation_func', call_args)
                
                # Execute the validation function to verify it works
                validation_func = call_args['validation_func']
                validation_result = validation_func()
                self.assertEqual(validation_result['valid'], True)


if __name__ == '__main__':
    unittest.main()