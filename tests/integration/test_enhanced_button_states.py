"""
Integration test for enhanced button state management.

This test verifies that the enhanced button handler mixin works correctly
with the refactored backbone module and provides consistent behavior.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

# Mock the UI components to avoid import issues in testing
class MockButton:
    def __init__(self, button_id: str):
        self.button_id = button_id
        self.disabled = False
        self.description = f"Mock {button_id} button"
        self._on_click_handlers = []
    
    def on_click(self, handler):
        self._on_click_handlers.append(handler)


class MockUIComponents:
    def __init__(self):
        self.action_container = {
            'buttons': {
                'validate': MockButton('validate'),
                'build': MockButton('build'),
                'rescan_models': MockButton('rescan_models')
            }
        }
    
    def get(self, key):
        if key == 'action_container':
            return self.action_container
        return None


class TestEnhancedButtonStates(unittest.TestCase):
    """Test enhanced button state management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the backend API to avoid file system dependencies
        self.backend_api_patcher = patch('smartcash.model.api.backbone_api.check_built_models')
        self.mock_check_built_models = self.backend_api_patcher.start()
        
        self.data_api_patcher = patch('smartcash.model.api.backbone_api.check_data_prerequisites')
        self.mock_check_data_prerequisites = self.data_api_patcher.start()
        
        # Create mock UI components
        self.mock_ui_components = MockUIComponents()
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.backend_api_patcher.stop()
        self.data_api_patcher.stop()
    
    def test_button_dependency_setup(self):
        """Test that button dependencies are set up correctly."""
        # Import here to avoid import issues
        from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule
        
        # Create module instance
        module = BackboneUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set up operation service mock
        module._operation_service = Mock()
        
        # Setup dependencies
        module._setup_backbone_button_dependencies()
        
        # Verify dependencies were set
        self.assertTrue(hasattr(module, '_button_dependencies'))
        self.assertIn('validate', module._button_dependencies)
        self.assertIn('build', module._button_dependencies)
    
    def test_validate_button_dependency_with_models(self):
        """Test validate button dependency when models are available."""
        from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule
        
        module = BackboneUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Mock operation service to return models available
        module._operation_service = Mock()
        module._operation_service.rescan_built_models.return_value = {
            'success': True,
            'total_models': 5
        }
        
        # Test dependency check
        result = module._check_validate_button_dependency()
        self.assertTrue(result)
    
    def test_validate_button_dependency_without_models(self):
        """Test validate button dependency when no models are available."""
        from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule
        
        module = BackboneUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Mock operation service to return no models
        module._operation_service = Mock()
        module._operation_service.rescan_built_models.return_value = {
            'success': True,
            'total_models': 0
        }
        
        # Test dependency check
        result = module._check_validate_button_dependency()
        self.assertFalse(result)
    
    def test_build_button_dependency_with_data(self):
        """Test build button dependency when data prerequisites are met."""
        from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule
        
        module = BackboneUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Mock operation service to return data available
        module._operation_service = Mock()
        module._operation_service.validate_data_prerequisites.return_value = {
            'prerequisites_ready': True
        }
        
        # Test dependency check
        result = module._check_build_button_dependency()
        self.assertTrue(result)
    
    def test_build_button_dependency_without_data(self):
        """Test build button dependency when data prerequisites are not met."""
        from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule
        
        module = BackboneUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Mock operation service to return no data
        module._operation_service = Mock()
        module._operation_service.validate_data_prerequisites.return_value = {
            'prerequisites_ready': False
        }
        
        # Test dependency check
        result = module._check_build_button_dependency()
        self.assertFalse(result)
    
    def test_button_state_updates_from_scan_results(self):
        """Test that button states are updated correctly from scan results."""
        from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule
        
        module = BackboneUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Mock the get_current_config method
        module.get_current_config = Mock(return_value={
            'backbone': {'model_type': 'efficientnet_b4'}
        })
        
        # Mock logging methods
        module.log_debug = Mock()
        module.log_error = Mock()
        
        # Mock the button state update methods
        module.update_button_states_based_on_condition = Mock()
        module._update_model_status_display = Mock()
        
        # Test with models available
        scan_result_with_models = {
            'success': True,
            'total_models': 3,
            'by_backbone': {
                'efficientnet_b4': [
                    {'path': '/path/to/model1.pt', 'size_mb': 120},
                    {'path': '/path/to/model2.pt', 'size_mb': 118},
                    {'path': '/path/to/model3.pt', 'size_mb': 122}
                ]
            }
        }
        
        module._update_ui_from_scan_results(scan_result_with_models)
        
        # Verify button states were updated correctly
        module.update_button_states_based_on_condition.assert_called_once()
        call_args = module.update_button_states_based_on_condition.call_args
        
        # Check that validate button should be enabled
        button_conditions = call_args[0][0]
        self.assertTrue(button_conditions['validate'])
        
        # Check that no reason is provided for enabled button
        button_reasons = call_args[0][1]
        self.assertIsNone(button_reasons['validate'])
    
    def test_button_state_updates_without_models(self):
        """Test button state updates when no models are available."""
        from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule
        
        module = BackboneUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Mock required methods
        module.get_current_config = Mock(return_value={
            'backbone': {'model_type': 'efficientnet_b4'}
        })
        module.log_debug = Mock()
        module.log_error = Mock()
        module.update_button_states_based_on_condition = Mock()
        module._update_model_status_display = Mock()
        
        # Test with no models
        scan_result_no_models = {
            'success': True,
            'total_models': 0,
            'by_backbone': {}
        }
        
        module._update_ui_from_scan_results(scan_result_no_models)
        
        # Verify button states were updated correctly
        module.update_button_states_based_on_condition.assert_called_once()
        call_args = module.update_button_states_based_on_condition.call_args
        
        # Check that validate button should be disabled
        button_conditions = call_args[0][0]
        self.assertFalse(button_conditions['validate'])
        
        # Check that reason is provided for disabled button
        button_reasons = call_args[0][1]
        self.assertEqual(button_reasons['validate'], "No built models available")
    
    def test_error_handling_in_ui_updates(self):
        """Test error handling during UI updates."""
        from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule
        
        module = BackboneUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Mock methods to simulate error
        module.get_current_config = Mock(side_effect=Exception("Config error"))
        module.log_debug = Mock()
        module.log_error = Mock()
        module.update_button_states_based_on_condition = Mock()
        module._update_model_status_display = Mock()
        
        # Test error handling
        scan_result = {'success': True, 'total_models': 1}
        module._update_ui_from_scan_results(scan_result)
        
        # Verify error was logged
        module.log_error.assert_called()
        
        # Verify error state was set for buttons
        module.update_button_states_based_on_condition.assert_called()
        call_args = module.update_button_states_based_on_condition.call_args
        
        # Check that all buttons are disabled in error state
        button_conditions = call_args[0][0]
        self.assertFalse(button_conditions['validate'])
        self.assertFalse(button_conditions['build'])


class TestButtonMixinEnhancements(unittest.TestCase):
    """Test the enhanced button handler mixin functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        from smartcash.ui.core.mixins.button_handler_mixin import ButtonHandlerMixin
        
        # Create a test class that uses the mixin
        class TestModule(ButtonHandlerMixin):
            def __init__(self):
                super().__init__()
                self.logger = Mock()
                self._ui_components = {
                    'action_container': {
                        'buttons': {
                            'test_button': MockButton('test_button')
                        }
                    }
                }
        
        self.test_module = TestModule()
    
    def test_button_dependency_setting(self):
        """Test setting button dependencies."""
        dependency_check = Mock(return_value=True)
        
        self.test_module.set_button_dependency('test_button', dependency_check)
        
        self.assertTrue(hasattr(self.test_module, '_button_dependencies'))
        self.assertIn('test_button', self.test_module._button_dependencies)
        self.assertEqual(self.test_module._button_dependencies['test_button'], dependency_check)
    
    def test_button_visibility_with_reason(self):
        """Test setting button visibility with reason tracking."""
        # Test disabling with reason
        self.test_module.set_button_visibility('test_button', False, "Test reason")
        
        # Check that button is disabled
        button = self.test_module._find_button_widget('test_button')
        self.assertTrue(button.disabled)
        
        # Check that reason is tracked
        reason = self.test_module.get_button_disable_reason('test_button')
        self.assertEqual(reason, "Test reason")
    
    def test_batch_button_updates(self):
        """Test updating multiple button states at once."""
        # Add another button for testing
        self.test_module._ui_components['action_container']['buttons']['another_button'] = MockButton('another_button')
        
        conditions = {
            'test_button': True,
            'another_button': False
        }
        
        reasons = {
            'test_button': None,
            'another_button': "Not ready"
        }
        
        self.test_module.update_button_states_based_on_condition(conditions, reasons)
        
        # Check first button is enabled
        test_button = self.test_module._find_button_widget('test_button')
        self.assertFalse(test_button.disabled)
        
        # Check second button is disabled with reason
        another_button = self.test_module._find_button_widget('another_button')
        self.assertTrue(another_button.disabled)
        self.assertEqual(self.test_module.get_button_disable_reason('another_button'), "Not ready")
    
    def test_button_enabled_check(self):
        """Test checking if button is enabled."""
        # Initially enabled
        self.assertTrue(self.test_module.is_button_enabled('test_button'))
        
        # Disable button
        self.test_module.disable_button('test_button')
        self.assertFalse(self.test_module.is_button_enabled('test_button'))
        
        # Re-enable button
        self.test_module.enable_button('test_button', force=True)
        self.assertTrue(self.test_module.is_button_enabled('test_button'))


if __name__ == '__main__':
    unittest.main()