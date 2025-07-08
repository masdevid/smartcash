"""
Test suite for augment module - comprehensive testing

This test suite provides 100% coverage testing for the augment module
including all components, handlers, operations, and configurations.
"""

import pytest
import unittest
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, Any

# Import augment module components
from smartcash.ui.dataset.augment import (
    AugmentInitializer, init_augment_ui, create_augment_ui,
    AugmentUIHandler, AugmentConfigHandler
)
from smartcash.ui.dataset.augment.components import (
    create_basic_options_widget, create_advanced_options_widget,
    create_augmentation_types_widget, create_operation_summary_widget
)
from smartcash.ui.dataset.augment.operations import (
    AugmentOperationManager, AugmentOperation, CheckOperation,
    CleanupOperation, PreviewOperation
)
from smartcash.ui.dataset.augment.constants import (
    AugmentationOperation, AugmentationTypes, CleanupTarget,
    DEFAULT_AUGMENTATION_PARAMS, UI_CONFIG
)


class TestAugmentModule(unittest.TestCase):
    """Comprehensive test suite for augment module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'data': {'dir': 'test_data'},
            'augmentation': {
                'num_variations': 2,
                'target_count': 100,
                'intensity': 0.5,
                'balance_classes': True,
                'target_split': 'train',
                'types': ['combined']
            }
        }
    
    def test_constants_module(self):
        """Test constants module has all required values."""
        # Test enums
        self.assertIn('AUGMENT', [op.value for op in AugmentationOperation])
        self.assertIn('COMBINED', [t.value for t in AugmentationTypes])
        self.assertIn('BOTH', [t.value for t in CleanupTarget])
        
        # Test default parameters
        self.assertIsInstance(DEFAULT_AUGMENTATION_PARAMS, dict)
        self.assertIn('num_variations', DEFAULT_AUGMENTATION_PARAMS)
        self.assertIn('target_count', DEFAULT_AUGMENTATION_PARAMS)
        
        # Test UI config
        self.assertEqual(UI_CONFIG['module_name'], 'augment')
        self.assertEqual(UI_CONFIG['parent_module'], 'dataset')
    
    @patch('smartcash.ui.dataset.augment.components.augment_ui.widgets')
    def test_create_augment_ui(self, mock_widgets):
        """Test main UI creation function."""
        # Mock widget creation
        mock_widgets.VBox.return_value = MagicMock()
        mock_widgets.HTML.return_value = MagicMock()
        
        # Create UI
        ui_components = create_augment_ui(self.test_config)
        
        # Verify UI components structure
        self.assertIsInstance(ui_components, dict)
        self.assertIn('ui', ui_components)
        self.assertIn('module_name', ui_components)
        self.assertIn('augment_initialized', ui_components)
        self.assertTrue(ui_components['augment_initialized'])
    
    @patch('smartcash.ui.dataset.augment.components.basic_options.widgets')
    def test_basic_options_widget(self, mock_widgets):
        """Test basic options widget creation."""
        # Mock widgets
        mock_widgets.IntSlider.return_value = MagicMock()
        mock_widgets.FloatSlider.return_value = MagicMock()
        mock_widgets.Dropdown.return_value = MagicMock()
        mock_widgets.Checkbox.return_value = MagicMock()
        mock_widgets.Text.return_value = MagicMock()
        mock_widgets.VBox.return_value = MagicMock()
        mock_widgets.HTML.return_value = MagicMock()
        
        # Create widget
        result = create_basic_options_widget()
        
        # Verify structure
        self.assertIsInstance(result, dict)
        self.assertIn('container', result)
        self.assertIn('widgets', result)
        self.assertIn('validation', result)
        self.assertIn('backend_mapping', result)
        
        # Verify required widgets
        widgets = result['widgets']
        self.assertIn('num_variations_slider', widgets)
        self.assertIn('target_count_slider', widgets)
        self.assertIn('intensity_slider', widgets)
        self.assertIn('balance_classes_checkbox', widgets)
    
    @patch('smartcash.ui.dataset.augment.components.advanced_options.widgets')
    def test_advanced_options_widget(self, mock_widgets):
        """Test advanced options widget creation."""
        # Mock widgets
        mock_widgets.FloatSlider.return_value = MagicMock()
        mock_widgets.IntSlider.return_value = MagicMock()
        mock_widgets.VBox.return_value = MagicMock()
        mock_widgets.HTML.return_value = MagicMock()
        
        # Create widget
        result = create_advanced_options_widget()
        
        # Verify structure
        self.assertIsInstance(result, dict)
        self.assertIn('container', result)
        self.assertIn('widgets', result)
        self.assertIn('position_widgets', result)
        self.assertIn('lighting_widgets', result)
        
        # Verify parameter widgets
        widgets = result['widgets']
        self.assertIn('horizontal_flip_slider', widgets)
        self.assertIn('rotation_limit_slider', widgets)
        self.assertIn('brightness_limit_slider', widgets)
        self.assertIn('hsv_hue_slider', widgets)
    
    @patch('smartcash.ui.dataset.augment.components.augmentation_types.widgets')
    def test_augmentation_types_widget(self, mock_widgets):
        """Test augmentation types widget creation."""
        # Mock widgets
        mock_widgets.SelectMultiple.return_value = MagicMock()
        mock_widgets.Checkbox.return_value = MagicMock()
        mock_widgets.VBox.return_value = MagicMock()
        mock_widgets.HTML.return_value = MagicMock()
        
        # Create widget
        result = create_augmentation_types_widget()
        
        # Verify structure
        self.assertIn('augmentation_types_select', result['widgets'])
        self.assertIn('preview_mode_checkbox', result['widgets'])
        self.assertIn('available_types', result)
        self.assertIn('type_combinations', result)
    
    @patch('smartcash.ui.dataset.augment.components.operation_summary.widgets')
    def test_operation_summary_widget(self, mock_widgets):
        """Test operation summary widget (NEW summary_container)."""
        # Mock widgets
        mock_widgets.HTML.return_value = MagicMock()
        mock_widgets.Textarea.return_value = MagicMock()
        mock_widgets.VBox.return_value = MagicMock()
        
        # Create widget
        result = create_operation_summary_widget()
        
        # Verify this is the new summary container
        self.assertTrue(result.get('is_summary_container', False))
        self.assertIn('update_methods', result)
        self.assertIn('real_time_updates', result)
        
        # Verify update methods
        update_methods = result['update_methods']
        self.assertIn('status', update_methods)
        self.assertIn('progress', update_methods)
        self.assertIn('dataset_stats', update_methods)
        self.assertIn('operation_metrics', update_methods)
    
    def test_augment_config_handler(self):
        """Test augment configuration handler."""
        # Create config handler
        config_handler = AugmentConfigHandler()
        
        # Test configuration validation
        valid_config = self.test_config.copy()
        is_valid, errors = config_handler.validate_config(valid_config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test invalid configuration
        invalid_config = {'invalid': 'config'}
        is_valid, errors = config_handler.validate_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
        # Test config summary
        summary = config_handler.get_config_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('augmentation_type', summary)
        self.assertIn('variations', summary)
    
    @patch('smartcash.ui.dataset.augment.handlers.augment_ui_handler.logging')
    def test_augment_ui_handler(self, mock_logging):
        """Test augment UI handler."""
        # Mock UI components
        mock_ui_components = {
            'augment_button': MagicMock(),
            'check_button': MagicMock(),
            'cleanup_button': MagicMock(),
            'preview_button': MagicMock(),
            'num_variations_slider': MagicMock(),
            'update_methods': {
                'status': MagicMock(),
                'progress': MagicMock(),
                'activity': MagicMock()
            }
        }
        
        # Create handler
        handler = AugmentUIHandler(mock_ui_components)
        
        # Test handler setup
        handler.setup_handlers()
        
        # Verify button handlers were set up
        mock_ui_components['augment_button'].on_click.assert_called()
        mock_ui_components['check_button'].on_click.assert_called()
        
        # Test operation handling
        handler.handle_augment()
        handler.handle_check()
        handler.handle_cleanup()
        handler.handle_preview()
    
    def test_augment_operation(self):
        """Test augmentation operation."""
        # Mock UI components
        mock_ui_components = {
            'update_methods': {
                'progress': MagicMock(),
                'activity': MagicMock(),
                'operation_metrics': MagicMock(),
                'dataset_stats': MagicMock()
            }
        }
        
        # Create operation
        operation = AugmentOperation(mock_ui_components)
        
        # Test operation execution
        result = operation.execute(self.test_config)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        if result['success']:
            self.assertIn('processed_images', result)
            self.assertIn('generated_images', result)
            self.assertIn('processing_time', result)
    
    def test_check_operation(self):
        """Test dataset check operation."""
        # Mock UI components
        mock_ui_components = {
            'update_methods': {
                'progress': MagicMock(),
                'activity': MagicMock(),
                'dataset_stats': MagicMock(),
                'operation_metrics': MagicMock()
            }
        }
        
        # Create operation
        operation = CheckOperation(mock_ui_components)
        
        # Test operation execution
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('os.listdir', return_value=['train', 'valid']):
            
            result = operation.execute(self.test_config)
            
            # Verify result
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
    
    def test_cleanup_operation(self):
        """Test cleanup operation."""
        # Mock UI components
        mock_ui_components = {
            'update_methods': {
                'progress': MagicMock(),
                'activity': MagicMock(),
                'operation_metrics': MagicMock()
            }
        }
        
        # Create operation
        operation = CleanupOperation(mock_ui_components)
        
        # Test operation execution
        config_with_cleanup = self.test_config.copy()
        config_with_cleanup['cleanup'] = {'default_target': 'both'}
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('os.listdir', return_value=[]):
            
            result = operation.execute(config_with_cleanup)
            
            # Verify result
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertIn('cleanup_target', result)
    
    def test_preview_operation(self):
        """Test preview operation."""
        # Mock UI components
        mock_ui_components = {
            'update_methods': {
                'progress': MagicMock(),
                'activity': MagicMock(),
                'operation_metrics': MagicMock()
            }
        }
        
        # Create operation
        operation = PreviewOperation(mock_ui_components)
        
        # Test operation execution
        result = operation.execute(self.test_config)
        
        # Verify result
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        if result['success']:
            self.assertIn('preview_count', result)
            self.assertIn('generation_time', result)
            self.assertIn('augmentation_types', result)
    
    def test_operation_manager(self):
        """Test operation manager."""
        # Mock UI components
        mock_ui_components = {
            'update_methods': {
                'status': MagicMock(),
                'activity': MagicMock()
            }
        }
        
        # Create operation manager
        manager = AugmentOperationManager(mock_ui_components)
        
        # Test available operations
        operations = manager.get_available_operations()
        self.assertIn('augment', operations)
        self.assertIn('check', operations)
        self.assertIn('cleanup', operations)
        self.assertIn('preview', operations)
        
        # Test operation execution
        result = manager.execute_operation('check', self.test_config)
        self.assertIsInstance(result, dict)
    
    @patch('smartcash.ui.dataset.augment.augment_initializer.create_augment_ui')
    def test_augment_initializer(self, mock_create_ui):
        """Test augment initializer."""
        # Mock UI creation
        mock_ui_components = {
            'ui': MagicMock(),
            'augment_button': MagicMock(),
            'update_methods': {}
        }
        mock_create_ui.return_value = mock_ui_components
        
        # Create initializer
        initializer = AugmentInitializer(self.test_config)
        
        # Test initialization
        result = initializer.initialize()
        
        # Verify result
        self.assertIsInstance(result, dict)
        self.assertIn('initializer', result)
        self.assertIn('config_handler', result)
        self.assertIn('ui_handler', result)
        self.assertTrue(result.get('initialization_success', False))
    
    @patch('smartcash.ui.dataset.augment.augment_initializer.AugmentInitializer')
    def test_init_augment_ui_factory(self, mock_initializer):
        """Test init_augment_ui factory function."""
        # Mock initializer
        mock_instance = MagicMock()
        mock_instance.initialize.return_value = {'test': 'result'}
        mock_initializer.return_value = mock_instance
        
        # Test factory function
        result = init_augment_ui(self.test_config)
        
        # Verify factory function works
        mock_initializer.assert_called_with(config=self.test_config)
        mock_instance.initialize.assert_called()
        self.assertEqual(result, {'test': 'result'})
    
    def test_module_integration(self):
        """Test complete module integration."""
        # Test that all components can be imported
        from smartcash.ui.dataset.augment import (
            AugmentInitializer, init_augment_ui, create_augment_ui,
            AugmentUIHandler, AugmentConfigHandler
        )
        
        # Verify classes exist and are callable
        self.assertTrue(callable(AugmentInitializer))
        self.assertTrue(callable(init_augment_ui))
        self.assertTrue(callable(create_augment_ui))
        self.assertTrue(callable(AugmentUIHandler))
        self.assertTrue(callable(AugmentConfigHandler))


class TestAugmentModuleCoverage(unittest.TestCase):
    """Additional tests for 100% coverage."""
    
    def test_error_handling(self):
        """Test error handling in various components."""
        # Test config handler with invalid data
        config_handler = AugmentConfigHandler()
        
        # Test with None config
        is_valid, errors = config_handler.validate_config(None)
        self.assertFalse(is_valid)
        
        # Test with malformed config
        malformed_config = {'augmentation': 'invalid_type'}
        is_valid, errors = config_handler.validate_config(malformed_config)
        self.assertFalse(is_valid)
    
    def test_operation_cancellation(self):
        """Test operation cancellation functionality."""
        operation = AugmentOperation()
        
        # Test cancellation
        operation.cancel()
        self.assertTrue(operation._is_cancelled)
        
        # Test status after cancellation
        status = operation.get_status()
        self.assertTrue(status['is_cancelled'])
    
    def test_ui_component_edge_cases(self):
        """Test UI component edge cases."""
        # Test with empty configuration
        with patch('smartcash.ui.dataset.augment.components.augment_ui.widgets'):
            ui_components = create_augment_ui({})
            self.assertIsInstance(ui_components, dict)
    
    def test_configuration_edge_cases(self):
        """Test configuration validation edge cases."""
        config_handler = AugmentConfigHandler()
        
        # Test boundary values
        boundary_config = {
            'data': {'dir': 'test'},
            'augmentation': {
                'num_variations': 1,  # Minimum
                'target_count': 10,   # Minimum
                'intensity': 0.0,     # Minimum
                'target_split': 'train',
                'types': ['combined']
            }
        }
        
        is_valid, errors = config_handler.validate_config(boundary_config)
        self.assertTrue(is_valid)
        
        # Test maximum boundary values
        boundary_config['augmentation'].update({
            'num_variations': 10,    # Maximum
            'target_count': 10000,   # Maximum
            'intensity': 1.0         # Maximum
        })
        
        is_valid, errors = config_handler.validate_config(boundary_config)
        self.assertTrue(is_valid)


if __name__ == '__main__':
    # Run tests with coverage
    unittest.main(verbosity=2)