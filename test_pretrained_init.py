"""
File: test_pretrained_init.py
Deskripsi: Unit tests untuk PretrainedInitializer
"""

import sys
import pytest
from unittest.mock import MagicMock, patch, call, ANY, PropertyMock

# Mock the parent class before importing PretrainedInitializer
with patch('smartcash.ui.initializers.common_initializer.CommonInitializer') as mock_common:
    # Import after setting up the mock
    from smartcash.ui.pretrained.pretrained_init import PretrainedInitializer

    # Configure the mock to work with our tests
    mock_common.return_value.module_name = 'pretrained_models'
    mock_common.return_value.config_handler = MagicMock()
    mock_common.return_value._create_ui_components.return_value = {}
    mock_common.return_value.initialize.return_value = {}

# Mock the config handler
with patch('smartcash.ui.pretrained.handlers.config_handler.PretrainedConfigHandler') as mock_config_handler:
    mock_config_handler.return_value.extract_config.return_value = {'pretrained_models': {}}
    mock_config_handler.return_value.validate_config.return_value = {'valid': True, 'errors': []}

# === Fixtures ===

@pytest.fixture
def mock_config():
    """Fixture untuk konfigurasi test"""
    return {
        'pretrained_models': {
            'pretrained_type': 'yolov5s',
            'models_dir': '/tmp/models',
            'drive_models_dir': '/content/drive/MyDrive/Models',
            'auto_download': True,
            'sync_drive': False
        },
        'ui_initialized': True
    }

@pytest.fixture
def mock_ui_components():
    """Fixture untuk UI components mock"""
    return {
        'ui': MagicMock(),
        'main_container': MagicMock(),
        'status': MagicMock(),
        'error_widget': MagicMock(),
        'download_sync_button': MagicMock(),
        'config': {}
    }

@pytest.fixture
def initializer():
    """Fixture untuk PretrainedInitializer instance"""
    with patch('smartcash.ui.initializers.common_initializer.CommonInitializer') as mock_common:
        # Create the initializer with the mocked parent
        initializer = PretrainedInitializer()
        
        # Configure the mock
        mock_common.assert_called_once()
        initializer.module_name = 'pretrained_models'
        initializer.config_handler = MagicMock()
        
        return initializer

# === Tests ===

def test_initializer_creation(initializer):
    """Test inisialisasi PretrainedInitializer"""
    assert initializer is not None
    assert hasattr(initializer, 'module_name')
    assert hasattr(initializer, 'config_handler')
    assert initializer.module_name == 'pretrained_models'

def test_create_ui_components_success(initializer, mock_config, mock_ui_components):
    """Test pembuatan UI components berhasil"""
    with patch('smartcash.ui.pretrained.components.ui_components.create_pretrained_ui_components', 
              return_value=mock_ui_components) as mock_create_ui:
        
        # Setup mock untuk config handler
        mock_handler = MagicMock()
        initializer.config_handler = mock_handler
        
        result = initializer._create_ui_components(mock_config)
        
        mock_create_ui.assert_called_once_with(
            env=None,
            config={'pretrained_models': mock_config['pretrained_models']}
        )
        assert result == mock_ui_components
        assert 'ui_initialized' in result

def test_create_ui_components_import_error(initializer, mock_config):
    """Test error handling saat import gagal"""
    with patch.dict('sys.modules', {'smartcash.ui.pretrained.components.ui_components': None}):
        with pytest.raises(ImportError):
            initializer._create_ui_components(mock_config)

def test_create_ui_components_general_error(initializer, mock_config):
    """Test error handling umum saat pembuatan UI"""
    with patch('smartcash.ui.pretrained.components.ui_components.create_pretrained_ui_components', 
              side_effect=Exception("Test error")) as mock_create_ui:
        
        result = initializer._create_ui_components(mock_config)
        
        assert result['fallback_mode'] is True
        assert 'error' in result
        assert 'Test error' in result['error']

def test_setup_event_handlers(initializer, mock_ui_components, mock_config):
    """Test setup event handlers"""
    with patch('smartcash.ui.pretrained.handlers.pretrained_handlers._setup_config_handlers') as mock_setup_config, \
         patch('smartcash.ui.pretrained.handlers.pretrained_handlers._setup_operation_handlers') as mock_setup_ops, \
         patch('smartcash.ui.pretrained.handlers.pretrained_handlers._handle_download_sync') as mock_handle_dl:
        
        initializer._setup_event_handlers(mock_ui_components, mock_config)
        
        mock_setup_config.assert_called_once_with(mock_ui_components)
        mock_setup_ops.assert_called_once_with(mock_ui_components)
        mock_ui_components['download_sync_button'].on_click.assert_called_once()

def test_load_initial_config(initializer, mock_ui_components, mock_config):
    """Test loading konfigurasi awal"""
    # Setup mock config handler
    mock_handler = MagicMock()
    mock_handler.get_default_config.return_value = mock_config
    
    # Panggil method yang di-test
    initializer._load_initial_config(mock_ui_components, mock_handler)
    
    # Verifikasi
    mock_handler.get_default_config.assert_called_once()
    
    # Verifikasi update_ui dipanggil dengan parameter yang benar
    mock_handler.update_ui.assert_called_once()
    
    # Verifikasi config disimpan dengan benar
    assert 'config' in mock_ui_components
    assert mock_ui_components['config'] == mock_config

def test_load_initial_config_with_list_pretrained_type(initializer, mock_ui_components):
    """Test konversi pretrained_type dari list ke string"""
    # Setup mock config dengan pretrained_type sebagai list
    mock_config_list = {
        'pretrained_models': {
            'pretrained_type': ['yolov5s'],
            'models_dir': '/tmp/models'
        }
    }
    
    # Mock handler dengan side effect untuk menguji konversi
    def update_ui_side_effect(ui_components, config):
        # Simpan config yang diupdate
        mock_ui_components['updated_config'] = config
    
    mock_handler = MagicMock()
    mock_handler.get_default_config.return_value = mock_config_list
    mock_handler.update_ui.side_effect = update_ui_side_effect
    
    # Panggil method yang di-test
    initializer._load_initial_config(mock_ui_components, mock_handler)
    
    # Verifikasi konversi list ke string
    updated_config = mock_ui_components.get('updated_config', {})
    pretrained_type = updated_config.get('pretrained_models', {}).get('pretrained_type')
    assert isinstance(pretrained_type, str)
    assert pretrained_type == 'yolov5s'

def test_initialize_pretrained_ui(initializer, mock_ui_components, mock_config):
    """Test factory function initialize_pretrained_ui"""
    # Mock global _pretrained_initializer
    with patch('smartcash.ui.pretrained.pretrained_init._pretrained_initializer', initializer), \
         patch.object(initializer, '_create_ui_components', return_value=mock_ui_components) as mock_create_ui, \
         patch.object(initializer, '_load_initial_config') as mock_load_config:
        
        # Panggil factory function
        from smartcash.ui.pretrained.pretrained_init import initialize_pretrained_ui
        result = initialize_pretrained_ui(env='test', config=mock_config)
        
        # Verifikasi
        mock_create_ui.assert_called_once_with(mock_config, env='test')
        mock_load_config.assert_called_once()
        assert result == mock_ui_components

if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
