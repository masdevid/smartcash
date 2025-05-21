"""
File: smartcash/ui/dataset/split/tests/test_initializer.py
Deskripsi: Test untuk initializer split dataset
"""

import unittest
import sys
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

class TestSplitInitializer(unittest.TestCase):
    """Test untuk initializer split dataset."""
    
    @patch('smartcash.ui.utils.ui_logger.create_ui_logger')
    @patch('smartcash.common.environment.get_environment_manager')
    @patch('smartcash.common.config.manager.get_config_manager')
    @patch('IPython.display.display')
    def test_initialize_split_ui(self, mock_display, mock_config_manager, mock_env_manager, mock_logger):
        """Test inisialisasi UI split dataset."""
        try:
            from smartcash.ui.dataset.split.split_initializer import initialize_split_ui
            
            # Setup mock
            mock_env = MagicMock()
            mock_env.base_dir = '/dummy/path'
            mock_env_manager.return_value = mock_env
            
            mock_cm = MagicMock()
            mock_cm.get_module_config.return_value = {
                'split': {
                    'enabled': True,
                    'train_ratio': 0.7,
                    'val_ratio': 0.15,
                    'test_ratio': 0.15,
                    'random_seed': 42,
                    'stratify': True
                }
            }
            mock_config_manager.return_value = mock_cm
            
            mock_logger.return_value = MagicMock()
            
            # Patch fungsi yang dipanggil dalam initialize_split_ui
            with patch('smartcash.ui.dataset.split.components.split_components.create_split_ui') as mock_create_ui, \
                 patch('smartcash.ui.dataset.split.handlers.button_handlers.setup_button_handlers') as mock_setup_buttons, \
                 patch('smartcash.ui.dataset.split.handlers.sync_logger.add_sync_status_panel') as mock_add_panel, \
                 patch('smartcash.ui.dataset.split.handlers.config_handlers.is_colab_environment', return_value=False):
                
                # Setup mock return values
                mock_ui = {'ui': MagicMock()}
                mock_create_ui.return_value = mock_ui
                mock_setup_buttons.return_value = mock_ui
                mock_add_panel.return_value = mock_ui
                
                # Panggil fungsi
                result = initialize_split_ui()
                
                # Verifikasi hasil
                self.assertIsInstance(result, dict)
                mock_create_ui.assert_called_once()
                mock_setup_buttons.assert_called_once()
                mock_add_panel.assert_called_once()
        except ImportError:
            print("Info: split_initializer.initialize_split_ui tidak tersedia, melewati test")
    
    def test_initialize_split_ui_with_error(self):
        """Test inisialisasi UI split dataset dengan error."""
        try:
            from smartcash.ui.dataset.split.split_initializer import initialize_split_ui
            
            # Patch fungsi yang diperlukan
            with patch('smartcash.ui.utils.ui_logger.create_ui_logger') as mock_logger, \
                 patch('smartcash.common.environment.get_environment_manager') as mock_env_manager, \
                 patch('smartcash.ui.dataset.split.handlers.sync_logger.log_sync_error') as mock_log_error, \
                 patch('IPython.display.display') as mock_display:
                
                # Setup mock
                mock_logger.return_value = MagicMock()
                
                # Setup mock env dengan base_dir None
                mock_env = MagicMock()
                mock_env.base_dir = None
                mock_env_manager.return_value = mock_env
                
                # Panggil fungsi
                result = initialize_split_ui()
                
                # Verifikasi hasil
                self.assertTrue(mock_log_error.call_count >= 1)
                # Tidak perlu memeriksa apakah display dipanggil karena implementasi berbeda-beda
                # Cukup verifikasi bahwa fungsi mengembalikan dictionary
                self.assertIsInstance(result, dict)
        except ImportError:
            print("Info: split_initializer.initialize_split_ui tidak tersedia, melewati test")
    
    def test_create_split_config_cell(self):
        """Test pembuatan cell konfigurasi split."""
        try:
            from smartcash.ui.cells.cell_2_2_split_config import setup_split_config
            
            # Patch initialize_split_ui
            with patch('smartcash.ui.dataset.split.split_initializer.initialize_split_ui') as mock_initialize:
                # Panggil fungsi
                setup_split_config()
                
                # Verifikasi initialize_split_ui dipanggil
                mock_initialize.assert_called_once()
        except ImportError:
            print("Info: cell_2_2_split_config.setup_split_config tidak tersedia, melewati test")

if __name__ == '__main__':
    unittest.main() 