"""
File: test_pretrained_initializer.py
Deskripsi: Unit tests untuk PretrainedInitializer
"""

import pytest
from unittest.mock import MagicMock, patch, ANY, call

# Test configuration
TEST_CONFIG = {
    'pretrained_models': {
        'pretrained_type': 'yolov5s',
        'models_dir': '/tmp/models',
        'drive_models_dir': '/content/drive/MyDrive/Models',
        'auto_download': True,
        'sync_drive': False
    }
}

# Mock UI components
MOCK_UI = {
    'ui': MagicMock(),
    'main_container': MagicMock(),
    'status': MagicMock(),
    'error_widget': MagicMock(),
    'download_sync_button': MagicMock(),
    'config': {},
    'ui_initialized': True
}

class MockPretrainedConfigHandler:
    def __init__(self, *args, **kwargs):
        pass
        
    def extract_config(self, ui_components):
        return {'pretrained_models': {}}
        
    def update_ui(self, ui_components, config):
        pass
        
    def validate_config(self, config):
        # Check if pretrained_models exists and has required fields
        pretrained_config = config.get('pretrained_models', {})
        
        if not pretrained_config:
            return {'valid': False, 'errors': ['Missing pretrained_models configuration']}
            
        # Check for required fields
        required_fields = ['models_dir']
        missing_fields = [field for field in required_fields if not pretrained_config.get(field)]
        
        if missing_fields:
            return {
                'valid': False, 
                'errors': [f'Missing required field: {field}' for field in missing_fields]
            }
            
        return {'valid': True, 'errors': []}

class MockCommonInitializer:
    def __init__(self, module_name=None, config_handler_class=None):
        self.module_name = module_name
        self.config_handler_class = config_handler_class
        self.config_handler = config_handler_class()
    
    def _get_critical_components(self):
        return ['ui', 'main_container', 'status', 'error_widget', 'download_sync_button']
    
    def _setup_event_handlers(self, ui_components, config):
        pass

def create_mock_ui_components(env=None, config=None):
    return MOCK_UI

def test_initializer_creation():
    """Test inisialisasi PretrainedInitializer"""
    with patch('smartcash.ui.pretrained.pretrained_init.CommonInitializer', MockCommonInitializer), \
         patch('smartcash.ui.pretrained.pretrained_init.PretrainedConfigHandler', MockPretrainedConfigHandler):
        
        # Import here to apply the mocks
        from smartcash.ui.pretrained.pretrained_init import PretrainedInitializer
        
        # Create instance
        initializer = PretrainedInitializer()
        
        # Verify instance is created
        assert initializer is not None
        
        # Verify the module_name is set
        assert hasattr(initializer, 'module_name')
        assert initializer.module_name == 'pretrained_models'
        
        # Verify config_handler_class is set correctly
        assert initializer.config_handler_class == MockPretrainedConfigHandler

def test_create_ui_components_success():
    """Test pembuatan UI components berhasil"""
    with patch('smartcash.ui.pretrained.pretrained_init.CommonInitializer', MockCommonInitializer), \
         patch('smartcash.ui.pretrained.components.ui_components.create_pretrained_ui_components', 
              side_effect=create_mock_ui_components) as mock_create_ui:
        
        # Import here to apply the mocks
        from smartcash.ui.pretrained.pretrained_init import PretrainedInitializer
        
        # Create instance
        initializer = PretrainedInitializer()
        
        # Test
        result = initializer._create_ui_components(TEST_CONFIG)
        
        # Verify UI components creation
        mock_create_ui.assert_called_once_with(
            env=None,
            config={'pretrained_models': TEST_CONFIG['pretrained_models']}
        )
        
        # Verify result
        assert result == MOCK_UI
        assert result['ui_initialized'] is True

def test_initialize_pretrained_ui():
    """Test factory function initialize_pretrained_ui"""
    with patch('smartcash.ui.pretrained.pretrained_init._pretrained_initializer') as mock_initializer:
        # Setup return value
        mock_initializer.initialize.return_value = MOCK_UI
        
        # Import here to apply the mocks
        from smartcash.ui.pretrained.pretrained_init import initialize_pretrained_ui
        
        # Test
        result = initialize_pretrained_ui(env='test', config=TEST_CONFIG)
        
        # Verify
        mock_initializer.initialize.assert_called_once_with(
            env='test',
            config=TEST_CONFIG
        )
        assert result == MOCK_UI

def test_config_handler_methods():
    """Test config handler methods"""
    with patch('smartcash.ui.pretrained.pretrained_init.CommonInitializer', MockCommonInitializer), \
         patch('smartcash.ui.pretrained.pretrained_init.PretrainedConfigHandler', MockPretrainedConfigHandler):
        
        # Import here to apply the mocks
        from smartcash.ui.pretrained.pretrained_init import PretrainedInitializer
        
        # Create instance
        initializer = PretrainedInitializer()
        
        # Test that the config handler was properly initialized
        assert hasattr(initializer, 'config_handler_class')
        assert initializer.config_handler_class == MockPretrainedConfigHandler
        
        # Test config extraction through the handler
        test_ui_components = {
            'models_dir_input': MagicMock(value='/test/models'),
            'drive_models_dir_input': MagicMock(value='/test/drive/models'),
            'pretrained_type_dropdown': MagicMock(value='yolov5s'),
            'auto_download_checkbox': MagicMock(value=True),
            'sync_drive_checkbox': MagicMock(value=False)
        }
        
        # Get the config handler instance
        config_handler = initializer.config_handler_class()
        
        # Test config extraction
        extracted_config = config_handler.extract_config(test_ui_components)
        assert 'pretrained_models' in extracted_config
        
        # Test config update
        test_config = {
            'pretrained_models': {
                'models_dir': '/new/models',
                'drive_models_dir': '/drive/new_models',
                'pretrained_type': 'yolov5m',
                'auto_download': True,
                'sync_drive': False
            }
        }
        
        # Test validation
        validation_result = config_handler.validate_config(test_config)
        assert validation_result['valid'] is True
        
        # Test with invalid config - empty pretrained_models
        invalid_config = {'pretrained_models': {}}
        validation_result = config_handler.validate_config(invalid_config)
        assert validation_result['valid'] is False
        assert 'Missing pretrained_models configuration' in validation_result['errors']


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
