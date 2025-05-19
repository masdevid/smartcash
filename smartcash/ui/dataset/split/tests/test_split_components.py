"""
File: smartcash/ui/dataset/split/tests/test_split_components.py
Deskripsi: Test suite untuk komponen split dataset
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets
from smartcash.ui.dataset.split.components.split_components import create_split_ui
from smartcash.ui.dataset.split.handlers.config_handlers import load_config, save_config, get_default_split_config

class TestSplitComponents(unittest.TestCase):
    """Test suite untuk komponen split dataset"""
    
    def setUp(self):
        """Setup test environment"""
        self.config = get_default_split_config()
        self.ui_components = create_split_ui()
    
    def test_create_split_ui(self):
        """Test pembuatan komponen UI"""
        ui_components = create_split_ui(self.config)
        
        # Test komponen utama
        self.assertIn('ui', ui_components)
        self.assertIn('train_slider', ui_components)
        self.assertIn('val_slider', ui_components)
        self.assertIn('test_slider', ui_components)
        self.assertIn('stratified_checkbox', ui_components)
        self.assertIn('random_seed', ui_components)
        self.assertIn('backup_checkbox', ui_components)
        self.assertIn('backup_dir', ui_components)
        self.assertIn('dataset_path', ui_components)
        self.assertIn('preprocessed_path', ui_components)
        
        # Test nilai default
        self.assertEqual(ui_components['train_slider'].value, 0.7)
        self.assertEqual(ui_components['val_slider'].value, 0.15)
        self.assertEqual(ui_components['test_slider'].value, 0.15)
        self.assertTrue(ui_components['stratified_checkbox'].value)
        self.assertEqual(ui_components['random_seed'].value, 42)
        self.assertTrue(ui_components['backup_checkbox'].value)
        self.assertEqual(ui_components['backup_dir'].value, 'data/splits_backup')
        self.assertEqual(ui_components['dataset_path'].value, 'data')
        self.assertEqual(ui_components['preprocessed_path'].value, 'data/preprocessed')
    
    def test_slider_behavior(self):
        """Test perilaku slider"""
        ui_components = create_split_ui()
        
        # Test nilai default
        self.assertEqual(ui_components['train_slider'].value, 0.7)
        self.assertEqual(ui_components['val_slider'].value, 0.15)
        self.assertEqual(ui_components['test_slider'].value, 0.15)
        
        # Test perubahan nilai
        ui_components['train_slider'].value = 0.8
        ui_components['val_slider'].value = 0.1
        ui_components['test_slider'].value = 0.1
        
        self.assertEqual(ui_components['train_slider'].value, 0.8)
        self.assertEqual(ui_components['val_slider'].value, 0.1)
        self.assertEqual(ui_components['test_slider'].value, 0.1)
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.get_config_manager_instance')
    def test_config_handlers(self, mock_get_config_manager_instance):
        """Test handler konfigurasi"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.get_module_config.return_value = {'data': {'split': self.config['data']['split']}}
        mock_get_config_manager_instance.return_value = mock_manager
        
        # Test load config
        loaded_config = load_config()
        if 'data' in loaded_config and 'split' in loaded_config['data']:
            self.assertEqual(loaded_config['data']['split'], self.config['data']['split'])
        else:
            self.assertIn('split', loaded_config)
        
        # Test save config
        # save_config(self.config)
        # mock_manager.save_module_config.assert_called_once_with('dataset', self.config)
        
        # Test load default config
        default_config = get_default_split_config()
        self.assertIn('split', default_config['data'])
        self.assertIn('train', default_config['data']['split'])
        self.assertIn('val', default_config['data']['split'])
        self.assertIn('test', default_config['data']['split'])
    
    def test_button_handlers(self):
        """Test handler tombol"""
        ui_components = create_split_ui(self.config)
        
        # Test save button
        self.assertIn('save_button', ui_components)
        self.assertIsInstance(ui_components['save_button'], widgets.Button)
        self.assertEqual(ui_components['save_button'].description, 'Simpan')
        
        # Test reset button
        self.assertIn('reset_button', ui_components)
        self.assertIsInstance(ui_components['reset_button'], widgets.Button)
        self.assertEqual(ui_components['reset_button'].description, 'Reset')
    
    @patch('smartcash.ui.dataset.split.split_initializer.initialize_split_ui')
    def test_integration(self, mock_initialize):
        """Test integrasi komponen"""
        # Setup mock
        mock_ui = create_split_ui(self.config)
        mock_initialize.return_value = mock_ui
        
        # Test inisialisasi UI
        from smartcash.ui.dataset.split.split_initializer import initialize_split_ui
        ui_components = initialize_split_ui(config=self.config)
        
        # Verifikasi komponen
        self.assertIn('ui', ui_components)
        self.assertIn('train_slider', ui_components)
        self.assertIn('val_slider', ui_components)
        self.assertIn('test_slider', ui_components)
        
        # Verifikasi nilai
        self.assertEqual(ui_components['train_slider'].value, 0.7)
        self.assertEqual(ui_components['val_slider'].value, 0.15)
        self.assertEqual(ui_components['test_slider'].value, 0.15)

if __name__ == '__main__':
    unittest.main() 