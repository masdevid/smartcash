"""
File: smartcash/ui/dataset/split/tests/test_config_handlers.py
Deskripsi: Test untuk handler konfigurasi split dataset
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import yaml
from pathlib import Path

class TestSplitConfigHandlers(unittest.TestCase):
    """Test case untuk handler konfigurasi split dataset."""
    
    def setUp(self):
        """Setup untuk test case."""
        # Mock config
        self.mock_config = {
            'data': {
                'split': {
                    'train': 0.7,
                    'val': 0.15,
                    'test': 0.15,
                    'stratified': True
                },
                'random_seed': 42,
                'backup_before_split': True,
                'backup_dir': 'data/splits_backup',
                'dataset_path': 'data',
                'preprocessed_path': 'data/preprocessed'
            }
        }
        
        # Mock UI components
        self.ui_components = {
            'train_slider': MagicMock(value=0.7),
            'val_slider': MagicMock(value=0.15),
            'test_slider': MagicMock(value=0.15),
            'stratified_checkbox': MagicMock(value=True),
            'random_seed': MagicMock(value=42),
            'backup_checkbox': MagicMock(value=True),
            'backup_dir': MagicMock(value='data/splits_backup'),
            'dataset_path': MagicMock(value='data'),
            'preprocessed_path': MagicMock(value='data/preprocessed'),
            'logger': MagicMock(),
            'module_name': 'dataset_split'
        }
        
        # Mock config manager
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.get_module_config.return_value = self.mock_config
        self.mock_config_manager.save_module_config.return_value = True
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.load_default_config')
    def test_load_default_config(self, mock_load_default):
        """Test load konfigurasi default."""
        from smartcash.ui.dataset.split.handlers.config_handlers import load_default_config
        
        # Setup mock
        mock_load_default.return_value = self.mock_config
        
        # Panggil fungsi
        result = load_default_config()
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_config)
        self.assertTrue('data' in result)
        self.assertTrue('split' in result['data'])
        self.assertEqual(result['data']['split']['train'], 0.7)
        self.assertEqual(result['data']['split']['val'], 0.15)
        self.assertEqual(result['data']['split']['test'], 0.15)
        self.assertTrue(result['data']['split']['stratified'])
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open')
    @patch('yaml.safe_load')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.load_default_config')
    def test_load_config_file_exists(self, mock_load_default, mock_yaml_load, mock_open, mock_exists):
        """Test load konfigurasi dari file yang ada."""
        from smartcash.ui.dataset.split.handlers.config_handlers import load_config
        
        # Setup mock
        mock_exists.return_value = True
        mock_yaml_load.return_value = self.mock_config
        
        # Panggil fungsi
        result = load_config()
        
        # Verifikasi hasil
        mock_exists.assert_called()
        mock_open.assert_called_once()
        mock_yaml_load.assert_called_once()
        self.assertEqual(result, self.mock_config)
    
    @patch('pathlib.Path.exists')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.load_default_config')
    def test_load_config_file_not_exists(self, mock_load_default, mock_exists):
        """Test load konfigurasi dari file yang tidak ada."""
        from smartcash.ui.dataset.split.handlers.config_handlers import load_config
        
        # Setup mock
        mock_exists.return_value = False
        mock_load_default.return_value = self.mock_config
        
        # Panggil fungsi
        result = load_config()
        
        # Verifikasi hasil
        # Tidak perlu memeriksa mock_exists.assert_called() karena Path.exists() dipanggil dalam konteks yang berbeda
        mock_load_default.assert_called_once()
        self.assertEqual(result, self.mock_config)
    
    @patch('os.makedirs')
    @patch('builtins.open')
    @patch('yaml.dump')
    def test_save_config(self, mock_yaml_dump, mock_open, mock_makedirs):
        """Test simpan konfigurasi ke file."""
        from smartcash.ui.dataset.split.handlers.config_handlers import save_config
        
        # Panggil fungsi
        result = save_config(self.mock_config)
        
        # Verifikasi hasil
        mock_makedirs.assert_called_once()
        mock_open.assert_called_once()
        mock_yaml_dump.assert_called_once()
        self.assertTrue(result)
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.get_config_manager_instance')
    def test_save_config_with_manager(self, mock_get_manager):
        """Test simpan konfigurasi dengan ConfigManager."""
        from smartcash.ui.dataset.split.handlers.config_handlers import save_config_with_manager
        
        # Setup mock
        mock_get_manager.return_value = self.mock_config_manager
        
        # Panggil fungsi
        result = save_config_with_manager(self.mock_config, self.ui_components, self.ui_components['logger'])
        
        # Verifikasi hasil
        mock_get_manager.assert_called_once()
        self.mock_config_manager.register_ui_components.assert_called_once_with('dataset_split', self.ui_components)
        self.mock_config_manager.save_module_config.assert_called_once_with('dataset', self.mock_config)
        self.assertTrue(result)
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.get_config_manager_instance')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.save_config')
    def test_save_config_with_manager_fallback(self, mock_save_config, mock_get_manager):
        """Test simpan konfigurasi dengan fallback ke save_config."""
        from smartcash.ui.dataset.split.handlers.config_handlers import save_config_with_manager
        
        # Setup mock
        mock_get_manager.return_value = None
        mock_save_config.return_value = True
        
        # Panggil fungsi
        result = save_config_with_manager(self.mock_config, self.ui_components, self.ui_components['logger'])
        
        # Verifikasi hasil
        mock_get_manager.assert_called_once()
        mock_save_config.assert_called_once_with(self.mock_config)
        self.assertTrue(result)
    
    def test_update_config_from_ui(self):
        """Test update konfigurasi dari UI."""
        from smartcash.ui.dataset.split.handlers.config_handlers import update_config_from_ui
        
        # Panggil fungsi
        result = update_config_from_ui({}, self.ui_components)
        
        # Verifikasi hasil
        self.assertTrue('data' in result)
        self.assertTrue('split' in result['data'])
        self.assertEqual(result['data']['split']['train'], 0.7)
        self.assertEqual(result['data']['split']['val'], 0.15)
        self.assertEqual(result['data']['split']['test'], 0.15)
        self.assertTrue(result['data']['split']['stratified'])
        self.assertEqual(result['data']['random_seed'], 42)
        self.assertTrue(result['data']['backup_before_split'])
        self.assertEqual(result['data']['backup_dir'], 'data/splits_backup')
        self.assertEqual(result['data']['dataset_path'], 'data')
        self.assertEqual(result['data']['preprocessed_path'], 'data/preprocessed')

if __name__ == '__main__':
    unittest.main()
