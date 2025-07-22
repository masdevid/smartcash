"""
File: tests/unit/ui/dataset/preprocessing/configs/test_preprocessing_config_handler.py
Description: Comprehensive unit tests for the PreprocessingConfigHandler class.
"""
import unittest
from unittest.mock import MagicMock, patch, ANY, call, PropertyMock
from typing import Dict, Any, Optional

from smartcash.ui.dataset.preprocessing.configs.preprocessing_config_handler import PreprocessingConfigHandler
from smartcash.ui.dataset.preprocessing.configs.preprocessing_defaults import get_default_config
from smartcash.ui.dataset.preprocessing.constants import YOLO_PRESETS, CleanupTarget


class TestPreprocessingConfigHandler(unittest.TestCase):
    """Test suite for PreprocessingConfigHandler."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a test instance with default config
        self.handler = PreprocessingConfigHandler()
        
        # Mock the logger to avoid actual logging
        self.handler.logger = MagicMock()
        self.handler.logger.info = MagicMock()
        self.handler.logger.debug = MagicMock()
        self.handler.logger.warning = MagicMock()
        self.handler.logger.error = MagicMock()
        
        # Sample UI components for testing
        self.sample_ui_components = {
            'resolution_dropdown': MagicMock(value='yolov5s'),
            'normalization_dropdown': MagicMock(value='imagenet'),
            'preserve_aspect_checkbox': MagicMock(value=True),
            'target_splits_select': MagicMock(value=['train', 'val']),
            'batch_size_input': MagicMock(value=64),
            'validation_checkbox': MagicMock(value=True),
            'move_invalid_checkbox': MagicMock(value=True),
            'invalid_dir_input': MagicMock(value='data/invalid'),
            'cleanup_target_dropdown': MagicMock(value='all'),
            'backup_checkbox': MagicMock(value=True)
        }
        
        # Set UI components on the handler
        self.handler.set_ui_components(self.sample_ui_components)
    
    def test_initialization(self):
        """Test that the handler initializes correctly."""
        # Create a new handler for this test
        with patch('smartcash.ui.dataset.preprocessing.configs.preprocessing_config_handler.get_module_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            handler = PreprocessingConfigHandler()
            
            mock_get_logger.assert_called_once_with('smartcash.ui.dataset.preprocessing.configs')
            self.assertEqual(handler.module_name, 'preprocessing')
            self.assertEqual(handler.parent_module, 'dataset')
            
            # Verify initialization message was logged
            mock_logger.info.assert_called_once_with("✅ Preprocessing config handler initialized")
    
    def test_get_default_config(self):
        """Test that get_default_config returns a copy of the default config."""
        default_config = get_default_config()
        result = self.handler.get_default_config()
        
        # Should return a copy, not the same object
        self.assertIsNot(result, default_config)
        self.assertEqual(result, default_config)
    
    def test_create_config_handler(self):
        """Test that create_config_handler returns self."""
        test_config = {'test': 'config'}
        result = self.handler.create_config_handler(test_config)
        
        self.assertIs(result, self.handler)
    
    def test_set_ui_components(self):
        """Test that set_ui_components sets the UI components correctly."""
        test_components = {'test': MagicMock()}
        self.handler.set_ui_components(test_components)
        
        self.assertEqual(self.handler._ui_components, test_components)
        self.handler.logger.debug.assert_called_with("UI components set: ['test']")
    
    def test_get_ui_value_component_found(self):
        """Test get_ui_value when the component exists and has a value."""
        # Test with a component that exists and has a value
        self.sample_ui_components['test_component'] = MagicMock(value='test_value')
        result = self.handler.get_ui_value('test_component', 'default')
        
        self.assertEqual(result, 'test_value')
    
    def test_get_ui_value_component_not_found(self):
        """Test get_ui_value when the component doesn't exist."""
        # Clear the UI components
        self.handler._ui_components = {}
        
        # Call the method - should return default without logging a warning
        result = self.handler.get_ui_value('nonexistent_component', 'default')
        
        self.assertEqual(result, 'default')
        
    def test_get_ui_value_with_exception(self):
        """Test get_ui_value when an exception occurs."""
        # Create a mock component that raises an exception when value is accessed
        class FailingComponent:
            @property
            def value(self):
                raise Exception("Test error")
                
        self.handler._ui_components = {'test_component': FailingComponent()}
        
        # Mock the logger to track calls
        with patch.object(self.handler.logger, 'warning') as mock_warning:
            result = self.handler.get_ui_value('test_component', 'default')
            
            self.assertEqual(result, 'default')
            # Verify logger.warning was called with the expected message
            mock_warning.assert_called_once()
            self.assertIn("Failed to get UI value for 'test_component'", mock_warning.call_args[0][0])
        
    def test_get_ui_value_component_no_ui_components(self):
        """Test get_ui_value when _ui_components is not set."""
        # Remove _ui_components attribute
        if hasattr(self.handler, '_ui_components'):
            delattr(self.handler, '_ui_components')
        
        self.handler.logger.warning = MagicMock()
            
        result = self.handler.get_ui_value('any_component', 'default')
        self.assertEqual(result, 'default')
        self.handler.logger.warning.assert_not_called()
    
    def test_extract_config_from_ui_no_components(self):
        """Test extract_config_from_ui when no UI components are set."""
        # Clear UI components
        self.handler.set_ui_components(None)
        
        result = self.handler.extract_config_from_ui()
        
        # Should return current config when no UI components
        self.assertEqual(result, self.handler.get_current_config())
        self.handler.logger.warning.assert_called_with("No UI components available")
    
    def test_extract_config_from_ui_with_components(self):
        """Test extract_config_from_ui with valid UI components."""
        result = self.handler.extract_config_from_ui()
        
        # Verify the structure of the returned config
        self.assertIn('preprocessing', result)
        prep_cfg = result['preprocessing']
        
        # Verify values from UI components
        self.assertEqual(prep_cfg['normalization']['preset'], 'yolov5s')
        self.assertEqual(prep_cfg['normalization']['method'], 'imagenet')
        self.assertTrue(prep_cfg['normalization']['preserve_aspect_ratio'])
        self.assertEqual(prep_cfg['target_splits'], ['train', 'val'])
        self.assertEqual(prep_cfg['batch_size'], 64)
        self.assertTrue(prep_cfg['validation']['enabled'])
        self.assertTrue(prep_cfg['move_invalid'])
        self.assertEqual(prep_cfg['invalid_dir'], 'data/invalid')
        self.assertEqual(prep_cfg['cleanup_target'], 'all')
        self.assertTrue(prep_cfg['backup_enabled'])
        
        # Verify YOLO preset was processed
        self.assertEqual(
            prep_cfg['normalization']['target_size'],
            YOLO_PRESETS['yolov5s']['target_size']
        )
    
    def test_update_ui_from_config(self):
        """Test that update_ui_from_config updates the UI components correctly."""
        # Create a test config
        test_config = {
            'preprocessing': {
                'normalization': {
                    'preset': 'yolov5s',
                    'method': 'imagenet',
                    'preserve_aspect_ratio': True
                },
                'target_splits': ['train', 'val'],
                'batch_size': 32,
                'validation': {
                    'enabled': True
                },
                'move_invalid': True,
                'invalid_dir': 'data/invalid',
                'cleanup_target': 'all',
                'backup_enabled': True
            }
        }
        
        # Add set_ui_value method to the handler
        self.handler.set_ui_value = MagicMock()
        
        # Call the method
        self.handler.update_ui_from_config(test_config)
        
        # Verify set_ui_value was called with the expected arguments
        expected_calls = [
            call('resolution_dropdown', 'yolov5s'),
            call('normalization_dropdown', 'imagenet'),
            call('preserve_aspect_checkbox', True),
            call('target_splits_select', ('train', 'val')),  # Note: converted to tuple in the implementation
            call('batch_size_input', 32),
            call('validation_checkbox', True),
            call('move_invalid_checkbox', True),
            call('invalid_dir_input', 'data/invalid'),
            call('cleanup_target_dropdown', 'all'),
            call('backup_checkbox', True)
        ]
        
        self.handler.set_ui_value.assert_has_calls(expected_calls, any_order=True)
    
    def test_validate_config(self):
        """Test validate_config method with valid configuration."""
        # Mock the current config to return a valid configuration
        valid_config = {
            'data': {
                'splits': ['train', 'val']
            },
            'preprocessing': {
                'normalization': {
                    'preset': 'yolov5s',
                    'method': 'imagenet',
                    'preserve_aspect_ratio': True
                },
                'target_splits': ['train', 'val'],
                'batch_size': 32,
                'validation': {'enabled': True},
                'move_invalid': True,
                'invalid_dir': 'data/invalid',
                'cleanup_target': 'all',
                'backup_enabled': True
            }
        }
        
        # Mock get_current_config to return our test config
        self.handler.get_current_config = MagicMock(return_value=valid_config)
        
        # Call validate_config (which doesn't take any parameters)
        result = self.handler.validate_config()
        
        # Should return a dict with validation results
        self.assertIsInstance(result, dict)
        self.assertIn('valid', result)
        self.assertTrue(result['valid'])
        self.assertIn('errors', result)
        self.assertEqual(len(result.get('errors', [])), 0)
    
    def test_validate_config_with_invalid_values(self):
        """Test validate_config with various invalid configurations."""
        # Test with missing data section
        empty_config = {}
        self.handler.get_current_config = MagicMock(return_value=empty_config)
        result = self.handler.validate_config()
        self.assertIn('valid', result)
        self.assertFalse(result['valid'])
        self.assertIn('errors', result)
        self.assertGreater(len(result.get('errors', [])), 0)
        
        # Test with invalid YOLO preset
        invalid_preset = {
            'data': {
                'splits': ['train']
            },
            'preprocessing': {
                'normalization': {'preset': 'invalid_preset'},
                'target_splits': ['train']
            }
        }
        self.handler.get_current_config = MagicMock(return_value=invalid_preset)
        result = self.handler.validate_config()
        self.assertIn('valid', result)
        self.assertFalse(result['valid'])
        self.assertIn('errors', result)
        self.assertGreater(len(result.get('errors', [])), 0)


    def test_extract_config_from_ui_with_nested_components(self):
        """Test config extraction with complex nested UI components."""
        # Setup nested mock components
        nested_components = {
            'input_options': {
                'resolution_dropdown': MagicMock(value='yolov5s'),
                'normalization_dropdown': MagicMock(value='imagenet'),
                'batch_size_input': MagicMock(value=64)
            },
            'validation_options': {
                'validation_checkbox': MagicMock(value=True),
                'move_invalid_checkbox': MagicMock(value=False)
            }
        }
        
        self.handler.set_ui_components(nested_components)
        config = self.handler.get_current_config()
        
        # Should extract from nested structure
        self.assertIn('preprocessing', config)
        self.assertIn('normalization', config['preprocessing'])

    def test_validate_config_with_missing_required_fields(self):
        """Test config validation with missing required fields."""
        # Test completely empty config
        self.handler.get_current_config = MagicMock(return_value={})
        result = self.handler.validate_config()
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
        
        # Test config missing preprocessing section
        minimal_config = {'data': {'splits': ['train']}}
        self.handler.get_current_config = MagicMock(return_value=minimal_config)
        result = self.handler.validate_config()
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)

    def test_validate_config_with_invalid_types(self):
        """Test config validation with invalid data types."""
        invalid_type_config = {
            'data': {
                'splits': 'should_be_list'  # Should be list
            },
            'preprocessing': {
                'normalization': {
                    'preset': 123,  # Should be string
                    'preserve_aspect_ratio': 'not_boolean'  # Should be boolean
                },
                'batch_size': 'not_number'  # Should be number
            }
        }
        
        self.handler.get_current_config = MagicMock(return_value=invalid_type_config)
        result = self.handler.validate_config()
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)

    def test_update_ui_from_config_with_missing_components(self):
        """Test updating UI when some components are missing."""
        # Setup partial UI components
        partial_components = {
            'resolution_dropdown': MagicMock(),
            # Missing other components
        }
        
        self.handler.set_ui_components(partial_components)
        
        # Should handle missing components gracefully without raising exceptions
        try:
            self.handler.update_ui_from_config({
                'preprocessing': {
                    'normalization': {'preset': 'yolov5s'},
                    'batch_size': 128
                }
            })
        except Exception as e:
            self.fail(f"Should handle missing components gracefully, but raised: {e}")

    def test_get_ui_value_with_different_component_types(self):
        """Test getting values from different types of UI components."""
        # Test different component types
        test_components = {
            'text_widget': MagicMock(value='text_value'),
            'checkbox_widget': MagicMock(value=True),
            'dropdown_widget': MagicMock(value='selected_option'),
            'number_widget': MagicMock(value=42),
        }
        
        self.handler.set_ui_components(test_components)
        
        # Test each component type
        self.assertEqual(self.handler.get_ui_value('text_widget'), 'text_value')
        self.assertEqual(self.handler.get_ui_value('checkbox_widget'), True)
        self.assertEqual(self.handler.get_ui_value('dropdown_widget'), 'selected_option')
        self.assertEqual(self.handler.get_ui_value('number_widget'), 42)

    def test_config_handler_with_preset_combinations(self):
        """Test config handler with various YOLO preset combinations."""
        # Test all valid YOLO presets with complete config structure
        for preset in YOLO_PRESETS.keys():
            with self.subTest(preset=preset):
                test_config = {
                    'data': {'splits': ['train', 'val']},
                    'preprocessing': {
                        'normalization': {
                            'preset': preset,
                            'method': 'imagenet',
                            'preserve_aspect_ratio': True
                        },
                        'target_splits': ['train', 'val'],
                        'batch_size': 32,
                        'validation': {'enabled': True},
                        'move_invalid': True,
                        'invalid_dir': 'data/invalid',
                        'cleanup_target': 'preprocessed',
                        'backup_enabled': True
                    }
                }
                
                self.handler.get_current_config = MagicMock(return_value=test_config)
                result = self.handler.validate_config()
                self.assertTrue(result['valid'], 
                               f"Preset {preset} should be valid but validation failed: {result.get('errors')}")

    def test_config_handler_edge_cases(self):
        """Test config handler edge cases and boundary conditions."""
        # Test with extreme batch sizes
        edge_cases = [
            {'batch_size': 1},  # Minimum
            {'batch_size': 1024},  # Large
            {'batch_size': 0},  # Invalid (should fail validation)
            {'batch_size': -1},  # Invalid (should fail validation)
        ]
        
        for case in edge_cases:
            with self.subTest(case=case):
                test_config = {
                    'data': {'splits': ['train']},
                    'preprocessing': {
                        'normalization': {'preset': 'yolov5s'},
                        'target_splits': ['train'],
                        **case
                    }
                }
                
                self.handler.get_current_config = MagicMock(return_value=test_config)
                result = self.handler.validate_config()
                
                if case['batch_size'] > 0:
                    self.assertTrue(result['valid'], f"Valid batch size {case['batch_size']} failed")
                else:
                    self.assertFalse(result['valid'], f"Invalid batch size {case['batch_size']} passed")

    def test_config_serialization_compatibility(self):
        """Test config can be properly serialized and deserialized."""
        import json
        
        # Get a default config
        original_config = self.handler.get_default_config()
        
        # Test JSON serialization
        try:
            json_str = json.dumps(original_config)
            restored_config = json.loads(json_str)
            
            # Should be identical after round-trip
            self.assertEqual(original_config, restored_config)
            
            # Should pass validation
            self.handler.get_current_config = MagicMock(return_value=restored_config)
            result = self.handler.validate_config()
            self.assertTrue(result['valid'], f"Serialized config validation failed: {result.get('errors')}")
            
        except (TypeError, ValueError) as e:
            self.fail(f"Config serialization failed: {e}")

    def test_config_with_cleanup_targets(self):
        """Test config handling with different cleanup targets."""
        from smartcash.ui.dataset.preprocessing.constants import CleanupTarget
        
        # Test all cleanup target options (using actual enum values)
        valid_targets = [e.value for e in CleanupTarget]
        for target in valid_targets:
            with self.subTest(target=target):
                test_config = {
                    'data': {'splits': ['train', 'val']},
                    'preprocessing': {
                        'normalization': {
                            'preset': 'yolov5s',
                            'method': 'imagenet',
                            'preserve_aspect_ratio': True
                        },
                        'target_splits': ['train', 'val'],
                        'batch_size': 32,
                        'validation': {'enabled': True},
                        'move_invalid': True,
                        'invalid_dir': 'data/invalid',
                        'cleanup_target': target,
                        'backup_enabled': True
                    }
                }
                
                self.handler.get_current_config = MagicMock(return_value=test_config)
                result = self.handler.validate_config()
                self.assertTrue(result['valid'], 
                               f"Cleanup target {target} should be valid: {result.get('errors')}")

    def test_config_error_handling_and_recovery(self):
        """Test config handler error handling and recovery mechanisms."""
        # Test with malformed config that should trigger error handling
        malformed_configs = [
            None,  # None config
            [],    # List instead of dict
            "string",  # String instead of dict
            {'data': None},  # None values in critical sections
        ]
        
        for malformed_config in malformed_configs:
            with self.subTest(config=malformed_config):
                self.handler.get_current_config = MagicMock(return_value=malformed_config)
                
                # Should handle malformed configs gracefully
                result = self.handler.validate_config()
                self.assertFalse(result['valid'])
                self.assertIsInstance(result.get('errors'), list)
                self.assertGreater(len(result.get('errors', [])), 0)


if __name__ == '__main__':
    unittest.main()
