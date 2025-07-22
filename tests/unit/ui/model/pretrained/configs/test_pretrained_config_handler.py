"""
File: tests/unit/ui/model/pretrained/configs/test_pretrained_config_handler.py
Description: Comprehensive unit tests for the PretrainedConfigHandler class.
"""
import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from smartcash.ui.model.pretrained.configs.pretrained_config_handler import PretrainedConfigHandler
from smartcash.ui.model.pretrained.configs.pretrained_defaults import get_default_pretrained_config


class TestPretrainedConfigHandler(unittest.TestCase):
    """Test suite for PretrainedConfigHandler."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_config = {
            'pretrained': {
                'models_dir': '/test/models',
                'model_urls': {
                    'yolov5s': 'http://test.com/yolo',
                    'efficientnet_b4': 'http://test.com/efficient'
                },
                'auto_download': True,
                'validate_downloads': True,
                'cleanup_failed': False
            },
            'models': {
                'yolov5s': {'enabled': True, 'priority': 1},
                'efficientnet_b4': {'enabled': False, 'priority': 2}
            },
            'operations': {
                'download': {'enabled': True, 'retry_count': 3},
                'validate': {'enabled': True, 'check_size': True}
            }
        }
        
        self.handler = PretrainedConfigHandler(self.test_config)
        
        # Mock UI components
        self.mock_ui_components = {
            'models_dir_input': MagicMock(),
            'yolo_url_input': MagicMock(),
            'efficient_url_input': MagicMock(),
            'auto_download_checkbox': MagicMock(),
            'validate_downloads_checkbox': MagicMock(),
            'cleanup_failed_checkbox': MagicMock(),
            'yolo_enabled_checkbox': MagicMock(),
            'efficient_enabled_checkbox': MagicMock()
        }
        
        # Set up mock values
        self.mock_ui_components['models_dir_input'].value = '/new/test/models'
        self.mock_ui_components['yolo_url_input'].value = 'http://new.com/yolo'
        self.mock_ui_components['efficient_url_input'].value = 'http://new.com/efficient'
        self.mock_ui_components['auto_download_checkbox'].value = False
        self.mock_ui_components['validate_downloads_checkbox'].value = False
        self.mock_ui_components['cleanup_failed_checkbox'].value = True
        self.mock_ui_components['yolo_enabled_checkbox'].value = False
        self.mock_ui_components['efficient_enabled_checkbox'].value = True
    
    def test_initialization(self):
        """Test that the config handler initializes correctly."""
        self.assertEqual(self.handler.config, self.test_config)
        self.assertIsNone(self.handler._ui_components)
    
    def test_get_current_config(self):
        """Test getting the current configuration."""
        config = self.handler.get_current_config()
        self.assertEqual(config, self.test_config)
    
    def test_update_config(self):
        """Test updating the configuration."""
        new_config = {
            'pretrained': {
                'models_dir': '/updated/path',
                'auto_download': False
            }
        }
        
        self.handler.update_config(new_config)
        
        # Verify the config was updated
        updated_config = self.handler.get_current_config()
        self.assertEqual(updated_config['pretrained']['models_dir'], '/updated/path')
        self.assertEqual(updated_config['pretrained']['auto_download'], False)
        
        # Verify other values are preserved
        self.assertEqual(updated_config['pretrained']['validate_downloads'], True)
    
    def test_set_ui_components(self):
        """Test setting UI components."""
        self.handler.set_ui_components(self.mock_ui_components)
        self.assertEqual(self.handler._ui_components, self.mock_ui_components)
    
    def test_extract_config_from_ui_no_components(self):
        """Test extracting config when no UI components are set."""
        result = self.handler.extract_config_from_ui()
        self.assertEqual(result, {})
    
    def test_extract_config_from_ui_with_components(self):
        """Test extracting config from UI components."""
        self.handler.set_ui_components(self.mock_ui_components)
        
        extracted_config = self.handler.extract_config_from_ui()
        
        # Verify pretrained config
        pretrained_config = extracted_config['pretrained']
        self.assertEqual(pretrained_config['models_dir'], '/new/test/models')
        self.assertEqual(pretrained_config['model_urls']['yolov5s'], 'http://new.com/yolo')
        self.assertEqual(pretrained_config['model_urls']['efficientnet_b4'], 'http://new.com/efficient')
        self.assertFalse(pretrained_config['auto_download'])
        self.assertFalse(pretrained_config['validate_downloads'])
        self.assertTrue(pretrained_config['cleanup_failed'])
        
        # Verify models config
        models_config = extracted_config['models']
        self.assertFalse(models_config['yolov5s']['enabled'])
        self.assertTrue(models_config['efficientnet_b4']['enabled'])
    
    def test_get_ui_value_component_found(self):
        """Test getting UI value when component exists."""
        self.handler.set_ui_components(self.mock_ui_components)
        
        value = self.handler._get_ui_value('models_dir_input', '/default/path')
        self.assertEqual(value, '/new/test/models')
    
    def test_get_ui_value_component_not_found(self):
        """Test getting UI value when component doesn't exist."""
        self.handler.set_ui_components(self.mock_ui_components)
        
        value = self.handler._get_ui_value('nonexistent_input', '/default/path')
        self.assertEqual(value, '/default/path')
    
    def test_get_ui_value_no_ui_components(self):
        """Test getting UI value when no UI components are set."""
        value = self.handler._get_ui_value('models_dir_input', '/default/path')
        self.assertEqual(value, '/default/path')
    
    def test_get_ui_value_with_exception(self):
        """Test getting UI value when an exception occurs."""
        self.handler.set_ui_components(self.mock_ui_components)
        
        # Mock component to raise exception when accessing value
        self.mock_ui_components['models_dir_input'].value = property(
            lambda self: exec('raise Exception("Test error")')
        )
        
        # Should return default value on exception
        value = self.handler._get_ui_value('models_dir_input', '/default/path')
        self.assertEqual(value, '/default/path')
    
    def test_update_ui_from_config(self):
        """Test updating UI components from configuration."""
        self.handler.set_ui_components(self.mock_ui_components)
        
        # Update UI from current config
        self.handler.update_ui_from_config()
        
        # Verify UI components were updated
        self.mock_ui_components['models_dir_input'].value = '/test/models'
        self.mock_ui_components['yolo_url_input'].value = 'http://test.com/yolo'
        self.mock_ui_components['efficient_url_input'].value = 'http://test.com/efficient'
        self.mock_ui_components['auto_download_checkbox'].value = True
        self.mock_ui_components['validate_downloads_checkbox'].value = True
        self.mock_ui_components['cleanup_failed_checkbox'].value = False
        self.mock_ui_components['yolo_enabled_checkbox'].value = True
        self.mock_ui_components['efficient_enabled_checkbox'].value = False
    
    def test_sync_to_ui(self):
        """Test syncing configuration to UI (alias for update_ui_from_config)."""
        self.handler.set_ui_components(self.mock_ui_components)
        
        # This should call update_ui_from_config
        with patch.object(self.handler, 'update_ui_from_config') as mock_update:
            self.handler.sync_to_ui()
            mock_update.assert_called_once()
    
    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        result = self.handler.validate_config()
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_validate_config_with_invalid_values(self):
        """Test validation with invalid configuration values."""
        # Create config with invalid values
        invalid_config = {
            'pretrained': {
                'models_dir': '',  # Empty path
                'model_urls': 'not_a_dict',  # Wrong type
                'auto_download': 'not_a_bool',  # Wrong type
                'download_timeout': -1  # Invalid value
            },
            'models': 'not_a_dict'  # Wrong type
        }
        
        invalid_handler = PretrainedConfigHandler(invalid_config)
        result = invalid_handler.validate_config()
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
        
        # Check for specific error types
        error_messages = ' '.join(result['errors'])
        self.assertIn('models_dir', error_messages)
        self.assertIn('model_urls', error_messages)
        self.assertIn('auto_download', error_messages)
        self.assertIn('models', error_messages)
    
    def test_get_models_dir(self):
        """Test getting models directory from config."""
        models_dir = self.handler.get_models_dir()
        self.assertEqual(models_dir, '/test/models')
    
    def test_get_models_dir_default(self):
        """Test getting models directory with default fallback."""
        # Create handler without models_dir in config
        config_without_dir = {'pretrained': {}}
        handler = PretrainedConfigHandler(config_without_dir)
        
        models_dir = handler.get_models_dir()
        self.assertEqual(models_dir, '/data/pretrained')  # Default value
    
    def test_get_model_url(self):
        """Test getting model URL from config."""
        yolo_url = self.handler.get_model_url('yolov5s')
        self.assertEqual(yolo_url, 'http://test.com/yolo')
        
        efficient_url = self.handler.get_model_url('efficientnet_b4')
        self.assertEqual(efficient_url, 'http://test.com/efficient')
    
    def test_get_model_url_not_found(self):
        """Test getting model URL for non-existent model."""
        url = self.handler.get_model_url('nonexistent_model')
        self.assertIsNone(url)
    
    def test_is_model_enabled(self):
        """Test checking if model is enabled."""
        self.assertTrue(self.handler.is_model_enabled('yolov5s'))
        self.assertFalse(self.handler.is_model_enabled('efficientnet_b4'))
    
    def test_is_model_enabled_not_found(self):
        """Test checking if non-existent model is enabled."""
        self.assertFalse(self.handler.is_model_enabled('nonexistent_model'))
    
    def test_get_operation_config(self):
        """Test getting operation configuration."""
        download_config = self.handler.get_operation_config('download')
        self.assertEqual(download_config['enabled'], True)
        self.assertEqual(download_config['retry_count'], 3)
        
        validate_config = self.handler.get_operation_config('validate')
        self.assertEqual(validate_config['enabled'], True)
        self.assertEqual(validate_config['check_size'], True)
    
    def test_get_operation_config_not_found(self):
        """Test getting configuration for non-existent operation."""
        config = self.handler.get_operation_config('nonexistent_operation')
        self.assertEqual(config, {})
    
    def test_create_config_handler_factory(self):
        """Test the factory function for creating config handler."""
        from smartcash.ui.model.pretrained.configs.pretrained_config_handler import create_config_handler
        
        handler = create_config_handler(self.test_config)
        self.assertIsInstance(handler, PretrainedConfigHandler)
        self.assertEqual(handler.config, self.test_config)
    
    def test_merge_configs(self):
        """Test merging configurations."""
        base_config = {
            'pretrained': {'models_dir': '/base', 'auto_download': True},
            'models': {'yolov5s': {'enabled': True}}
        }
        
        override_config = {
            'pretrained': {'models_dir': '/override'},
            'models': {'efficientnet_b4': {'enabled': False}}
        }
        
        merged = self.handler._merge_configs(base_config, override_config)
        
        # Verify merge
        self.assertEqual(merged['pretrained']['models_dir'], '/override')
        self.assertEqual(merged['pretrained']['auto_download'], True)  # Preserved from base
        self.assertEqual(merged['models']['yolov5s']['enabled'], True)  # Preserved from base
        self.assertEqual(merged['models']['efficientnet_b4']['enabled'], False)  # From override


if __name__ == '__main__':
    unittest.main()