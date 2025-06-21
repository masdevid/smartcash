"""
File: smartcash/tests/hyperparameters/test_error_handling.py
Deskripsi: Test case untuk error handling dan edge cases pada Hyperparameters
"""

import pytest
from unittest.mock import MagicMock, patch, ANY, PropertyMock
from smartcash.ui.hyperparameters.hyperparameters_init import HyperparametersConfigInitializer

# Mock untuk menghindari pembuatan widget asli
class MockVBox:
    def __init__(self, *args, **kwargs):
        self.children = []
    
    def __getattr__(self, name):
        # Handle dynamic attribute access untuk kompatibilitas
        return MagicMock()

# Patch ipywidgets di awal file untuk menghindari pembuatan widget asli
@pytest.fixture(autouse=True)
def mock_ipywidgets():
    with patch('ipywidgets.VBox', new=MockVBox), \
         patch('ipywidgets.HBox', new=MockVBox), \
         patch('ipywidgets.Button', return_value=MagicMock()), \
         patch('ipywidgets.Output', return_value=MagicMock()):
        yield

class TestHyperparametersErrorHandling:
    """Test class untuk error handling Hyperparameters"""
    
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.HyperparametersConfigHandler')
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.create_hyperparameters_config_cell')
    def test_initialize_when_load_config_fails(self, mock_create_cell, mock_handler_class):
        """Test inisialisasi ketika load_config gagal"""
        # Setup mock handler yang melempar exception
        mock_handler = MagicMock()
        mock_handler.load_config.side_effect = Exception("Failed to load config")
        mock_default_config = {'learning_rate': 0.001}
        mock_handler.get_default_config.return_value = mock_default_config
        mock_handler_class.return_value = mock_handler
        
        # Setup mock UI
        mock_ui = MagicMock()
        mock_ui.get_ui.return_value = MagicMock()
        
        # Jalankan initializer
        initializer = HyperparametersConfigInitializer()
        
        # Mock _create_ui_with_config untuk menangani kasus error
        with patch.object(initializer, '_create_ui_with_config', return_value=mock_ui) as mock_create_ui_with_config:
            # Mock parent initialize untuk test
            with patch.object(initializer, 'parent_initialize', return_value=mock_ui) as mock_parent_init:
                result = initializer.initialize()
                
                # Verifikasi
                assert result is not None
                assert hasattr(result, 'get_ui')
                
                # Verifikasi load_config dipanggil
                mock_handler.load_config.assert_called_once()
                
                # Verifikasi get_default_config dipanggil karena load_config gagal
                mock_handler.get_default_config.assert_called_once()
                
                # Verifikasi parent_initialize dipanggil dengan config default
                mock_parent_init.assert_called_once_with(env=None, config=mock_default_config, **{})
                
                # Verifikasi create_ui_with_config dipanggil dengan config default
                mock_create_ui_with_config.assert_called_once_with(mock_default_config, None, **{})
    
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.HyperparametersConfigHandler')
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.create_hyperparameters_config_cell')
    def test_initialize_when_ui_creation_fails(self, mock_create_cell, mock_handler_class):
        """Test inisialisasi ketika pembuatan UI gagal"""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_config = {'learning_rate': 0.001}
        mock_handler.load_config.return_value = mock_config
        mock_default_config = {'learning_rate': 0.001}
        mock_handler.get_default_config.return_value = mock_default_config
        mock_handler_class.return_value = mock_handler
        
        # Setup mock UI yang gagal dibuat
        mock_create_cell.side_effect = Exception("Failed to create UI")
        
        # Setup mock UI fallback
        mock_ui = MagicMock()
        mock_ui.get_ui.return_value = MagicMock()
        
        # Jalankan initializer
        initializer = HyperparametersConfigInitializer()
        
        # Mock _create_ui_with_config untuk menangani kasus error
        with patch.object(initializer, '_create_ui_with_config', side_effect=Exception("UI creation failed")) as mock_create_ui_with_config:
            # Mock parent initialize untuk test
            with patch.object(initializer, 'parent_initialize', return_value=mock_ui) as mock_parent_init:
                # Mock _create_config_ui untuk test fallback
                with patch.object(initializer, '_create_config_ui', return_value=mock_ui) as mock_create_config_ui:
                    result = initializer.initialize()
                    
                    # Verifikasi
                    assert result is not None
                    assert hasattr(result, 'get_ui')
                    
                    # Verifikasi load_config dipanggil
                    mock_handler.load_config.assert_called_once()
                    
                    # Verifikasi get_default_config dipanggil karena UI creation gagal
                    mock_handler.get_default_config.assert_called_once()
                    
                    # Verifikasi parent_initialize dipanggil dengan config default
                    mock_parent_init.assert_called_once_with(env=None, config=mock_default_config, **{})
                    
                    # Verifikasi _create_config_ui dipanggil sebagai fallback
                    mock_create_config_ui.assert_called_once_with(mock_default_config, None, **{})
    
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.HyperparametersConfigHandler')
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.create_hyperparameters_config_cell')
    def test_initialize_with_invalid_config(self, mock_create_cell, mock_handler_class):
        """Test inisialisasi dengan konfigurasi tidak valid"""
        # Setup mock handler dengan config tidak valid
        mock_handler = MagicMock()
        invalid_config = {'invalid_key': 'invalid_value'}
        mock_handler.load_config.return_value = invalid_config
        mock_default_config = {'learning_rate': 0.001}
        mock_handler.get_default_config.return_value = mock_default_config
        mock_handler_class.return_value = mock_handler
        
        # Setup mock UI
        mock_ui = MagicMock()
        mock_ui.get_ui.return_value = MagicMock()
        
        # Jalankan initializer
        initializer = HyperparametersConfigInitializer()
        
        # Mock _create_ui_with_config untuk menangani kasus error
        with patch.object(initializer, '_create_ui_with_config', return_value=mock_ui) as mock_create_ui_with_config:
            # Mock parent initialize untuk test
            with patch.object(initializer, 'parent_initialize', return_value=mock_ui) as mock_parent_init:
                # Mock _create_config_ui untuk test fallback
                with patch.object(initializer, '_create_config_ui', return_value=mock_ui) as mock_create_config_ui:
                    result = initializer.initialize()
                    
                    # Verifikasi
                    assert result is not None
                    assert hasattr(result, 'get_ui')
                    
                    # Verifikasi load_config dipanggil
                    mock_handler.load_config.assert_called_once()
                    
                    # Verifikasi get_default_config tidak dipanggil karena config masih berupa dict
                    mock_handler.get_default_config.assert_not_called()
                    
                    # Verifikasi create_ui_with_config dipanggil dengan config yang diberikan
                    mock_create_ui_with_config.assert_called_once_with(invalid_config, None, **{})
                    
                    # Verifikasi parent_initialize tidak dipanggil karena tidak ada error
                    mock_parent_init.assert_not_called()
                    
                    # Verifikasi _create_config_ui tidak dipanggil karena tidak ada error
                    mock_create_config_ui.assert_not_called()
                    
                    # Verifikasi create_hyperparameters_config_cell dipanggil dengan config yang diberikan
                    mock_create_cell.assert_called_once_with(env=None, config=invalid_config, **{})


class TestHyperparametersDefaultValues:
    """Test class untuk nilai default Hyperparameters"""
    
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.HyperparametersConfigHandler')
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.create_hyperparameters_config_cell')
    def test_initialize_with_valid_config(self, mock_create_cell, mock_handler_class):
        """Test inisialisasi dengan konfigurasi valid"""
        # Setup mock handler dengan config valid
        mock_handler = MagicMock()
        valid_config = {
            'training': {'learning_rate': 0.01, 'batch_size': 16, 'epochs': 100},
            'optimizer': {'type': 'Adam'},
            'loss': {'box_loss_gain': 0.05, 'cls_loss_gain': 0.5, 'obj_loss_gain': 1.0},
            'early_stopping': {'enabled': True, 'patience': 10},
            'checkpoint': {'save_best': True, 'metric': 'mAP_0.5'}
        }
        mock_handler.load_config.return_value = valid_config
        mock_default_config = valid_config.copy()
        mock_handler.get_default_config.return_value = mock_default_config
        mock_handler_class.return_value = mock_handler
        
        # Setup mock UI
        mock_ui = MagicMock()
        mock_ui.get_ui.return_value = MagicMock()
        
        # Jalankan initializer
        initializer = HyperparametersConfigInitializer()
        
        # Mock _create_ui_with_config untuk test
        with patch.object(initializer, '_create_ui_with_config', return_value=mock_ui) as mock_create_ui_with_config:
            # Mock parent initialize untuk test
            with patch.object(initializer, 'parent_initialize', return_value=mock_ui) as mock_parent_init:
                # Mock _create_config_ui untuk test
                with patch.object(initializer, '_create_config_ui', return_value=mock_ui) as mock_create_config_ui:
                    result = initializer.initialize()
                    
                    # Verifikasi
                    assert result is not None
                    assert hasattr(result, 'get_ui')
                    
                    # Verifikasi load_config dipanggil
                    mock_handler.load_config.assert_called_once()
                    
                    # Verifikasi get_default_config tidak dipanggil karena config valid
                    mock_handler.get_default_config.assert_not_called()
                    
                    # Verifikasi create_ui_with_config dipanggil dengan config yang valid
                    mock_create_ui_with_config.assert_called_once_with(valid_config, None, **{})
                    
                    # Verifikasi parent_initialize tidak dipanggil karena tidak ada error
                    mock_parent_init.assert_not_called()
                    
                    # Verifikasi _create_config_ui tidak dipanggil karena tidak ada error
                    mock_create_config_ui.assert_not_called()
                    
                    # Verifikasi create_hyperparameters_config_cell dipanggil dengan config yang valid
                    mock_create_cell.assert_called_once_with(env=None, config=valid_config, **{})

    @patch('smartcash.ui.hyperparameters.hyperparameters_init.HyperparametersConfigHandler')
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.create_hyperparameters_config_cell')
    def test_default_values_when_config_empty(self, mock_create_cell, mock_handler_class):
        """Test nilai default ketika config kosong"""
        # Setup mock handler dengan config kosong
        mock_handler = MagicMock()
        mock_handler.load_config.return_value = {}
        mock_default_config = {
            'training': {'learning_rate': 0.01, 'batch_size': 16, 'epochs': 100},
            'optimizer': {'type': 'SGD'},
            'loss': {'box_loss_gain': 0.05, 'cls_loss_gain': 0.5, 'obj_loss_gain': 1.0},
            'early_stopping': {'enabled': True, 'patience': 10},
            'checkpoint': {'save_best': True, 'metric': 'mAP_0.5'}
        }
        mock_handler.get_default_config.return_value = mock_default_config
        mock_handler_class.return_value = mock_handler
        
        # Setup mock UI
        mock_ui = MagicMock()
        mock_ui.get_ui.return_value = MagicMock()
        
        # Jalankan initializer
        initializer = HyperparametersConfigInitializer()
        
        # Mock _create_ui_with_config untuk test
        with patch.object(initializer, '_create_ui_with_config', return_value=mock_ui) as mock_create_ui_with_config:
            # Mock parent initialize untuk test
            with patch.object(initializer, 'parent_initialize', return_value=mock_ui) as mock_parent_init:
                # Mock _create_config_ui untuk test
                with patch.object(initializer, '_create_config_ui', return_value=mock_ui) as mock_create_config_ui:
                    result = initializer.initialize()
                    
                    # Verifikasi
                    assert result is not None
                    assert hasattr(result, 'get_ui')
                    
                    # Verifikasi load_config dipanggil
                    mock_handler.load_config.assert_called_once()
                    
                    # Verifikasi get_default_config dipanggil karena config kosong
                    mock_handler.get_default_config.assert_called_once()
                    
                    # Verifikasi create_ui_with_config dipanggil dengan config default
                    mock_create_ui_with_config.assert_called_once_with(mock_default_config, None, **{})
                    
                    # Verifikasi parent_initialize tidak dipanggil karena tidak ada error
                    mock_parent_init.assert_not_called()
                    
                    # Verifikasi _create_config_ui tidak dipanggil karena tidak ada error
                    mock_create_config_ui.assert_not_called()
                    
                    # Verifikasi create_hyperparameters_config_cell dipanggil dengan config default
                    mock_create_cell.assert_called_once_with(env=None, config=mock_default_config, **{})


class TestHyperparametersInputValidation:
    """Test class untuk validasi input Hyperparameters"""
    
    @pytest.mark.parametrize("test_config, expect_fallback, expect_load_config, test_id", [
        # Test case 1: Config kosong - valid, tidak perlu fallback
        pytest.param({}, False, False, 'empty_config', id='empty_config'),
        # Test case 2: Nilai tidak valid untuk batch_size - valid, tidak perlu fallback
        pytest.param({'batch_size': 0}, False, False, 'invalid_batch_size', id='invalid_batch_size'),
        # Test case 3: Nilai negatif untuk epochs - valid, tidak perlu fallback
        pytest.param({'epochs': -1}, False, False, 'negative_epochs', id='negative_epochs'),
        # Test case 4: Tipe data tidak valid untuk learning rate - valid, tidak perlu fallback
        pytest.param({'learning_rate': 'abc'}, False, False, 'invalid_learning_rate', id='invalid_learning_rate'),
        # Test case 5: Config None - akan di-load dari handler
        pytest.param(None, False, True, 'none_config', id='none_config'),
        # Test case 6: Tipe config tidak valid (bukan dict) - akan memicu error
        pytest.param(123, True, False, 'invalid_config_type', id='invalid_config_type')
    ])
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.HyperparametersConfigHandler')
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.create_hyperparameters_config_cell')
    @patch('smartcash.ui.initializers.config_cell_initializer.ConfigHandler')
    def test_initialize_with_invalid_inputs(self, mock_base_handler, mock_create_cell, 
                                          mock_handler_class, test_config, expect_fallback, 
                                          expect_load_config, test_id):
        """Test inisialisasi dengan berbagai input tidak valid"""
        from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer
        
        # Setup mock handler
        mock_handler = MagicMock()
        mock_default_config = {'learning_rate': 0.001}
        loaded_config = {'loaded': True}  # Config yang akan dikembalikan oleh load_config
        
        # Atur return value untuk load_config dan get_default_config
        mock_handler.load_config.return_value = loaded_config
        mock_handler.get_default_config.return_value = mock_default_config
        mock_handler_class.return_value = mock_handler
        
        # Setup mock UI
        mock_ui = MagicMock()
        mock_ui.get_ui.return_value = MagicMock()
        
        # Mock base handler untuk mencegah pemanggilan asli
        mock_base_handler_instance = MagicMock()
        mock_base_handler.return_value = mock_base_handler_instance
        mock_base_handler_instance.load_config.return_value = loaded_config
        
        # Mock parent initialize untuk semua kasus
        with patch.object(ConfigCellInitializer, 'initialize') as mock_super_initialize:
            # Setup mock untuk kasus normal
            mock_create_cell.return_value = mock_ui
            
            # Atur side effect untuk mock_super_initialize
            def super_initialize_side_effect(env=None, config=None, **kwargs):
                # Jika config adalah None, gunakan loaded_config
                if config is None:
                    config = loaded_config
                return mock_ui
            
            mock_super_initialize.side_effect = super_initialize_side_effect
            
            if expect_fallback:
                # Untuk test case yang mengharapkan fallback, atur side effect
                mock_create_cell.side_effect = ValueError("Invalid config type")
                
                # Mock untuk fallback UI
                mock_fallback_ui = MagicMock()
                mock_fallback_ui.get_ui.return_value = MagicMock()
                
                # Atur side effect untuk mock_super_initialize pada kasus fallback
                def fallback_side_effect(env=None, config=None, **kwargs):
                    if config == test_config:  # Ini akan gagal
                        raise ValueError("Invalid config type")
                    # Kembalikan fallback_ui untuk pemanggilan dengan config default
                    return mock_fallback_ui
                
                mock_super_initialize.side_effect = fallback_side_effect
            
            # Jalankan initializer
            initializer = HyperparametersConfigInitializer()
            
            # Patch config_handler.load_config untuk test case tertentu
            with patch.object(initializer.config_handler, 'load_config', 
                            return_value=loaded_config) as mock_load_config:
                result = initializer.initialize(config=test_config)
                
                # Verifikasi dasar
                assert result is not None
                assert hasattr(result, 'get_ui')
                
                if expect_fallback:
                    # Verifikasi untuk kasus fallback
                    mock_handler.get_default_config.assert_called_once()
                    
                    # Verifikasi parent initialize dipanggil dua kali:
                    # 1. Dengan config asli (akan gagal)
                    # 2. Dengan config default (fallback)
                    assert mock_super_initialize.call_count == 2
                    
                    # Verifikasi pemanggilan pertama dengan config asli
                    mock_super_initialize.assert_any_call(env=None, config=test_config, **{})
                    
                    # Verifikasi pemanggilan kedua dengan config default
                    mock_super_initialize.assert_any_call(env=None, config=mock_default_config, **{})
                    
                    # Pastikan load_config tidak dipanggil untuk kasus fallback
                    mock_load_config.assert_not_called()
                else:
                    # Verifikasi untuk kasus normal
                    mock_handler.get_default_config.assert_not_called()
                    
                    # Verifikasi load_config hanya dipanggil jika test_config adalah None
                    if test_config is None:
                        # Untuk kasus test_config=None, pastikan load_config dipanggil
                        # dan parent initialize dipanggil dengan config yang dimuat
                        mock_load_config.assert_called_once_with(initializer.config_filename)
                        expected_config = loaded_config
                    else:
                        # Untuk kasus lain, pastikan load_config tidak dipanggil
                        mock_load_config.assert_not_called()
                        expected_config = test_config
                    
                    # Verifikasi parent initialize dipanggil sekali dengan config yang benar
                    # Perhatikan bahwa kita perlu memeriksa call_args_list karena mock_super_initialize
                    # mungkin dipanggil dengan parameter tambahan yang tidak kita ketahui
                    assert mock_super_initialize.call_count == 1
                    _, call_kwargs = mock_super_initialize.call_args
                    assert call_kwargs.get('config') == expected_config
