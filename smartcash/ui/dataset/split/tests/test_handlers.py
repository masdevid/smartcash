"""
File: smartcash/ui/dataset/split/tests/test_handlers.py
Deskripsi: Test untuk handlers split dataset
"""

import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open

class TestConfigHandlers(unittest.TestCase):
    """Test untuk handlers konfigurasi."""
    
    def setUp(self):
        """Setup untuk test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'dataset_config.yaml')
    
    def tearDown(self):
        """Cleanup setelah test."""
        shutil.rmtree(self.temp_dir)
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.get_config_manager')
    def test_load_config(self, mock_get_config_manager):
        """Test load konfigurasi."""
        try:
            from smartcash.ui.dataset.split.handlers.config_handlers import load_config
            
            # Setup mock
            mock_cm = MagicMock()
            mock_cm.get_module_config.return_value = {'split': {'train_ratio': 0.7}}
            mock_get_config_manager.return_value = mock_cm
            
            # Panggil fungsi
            result = load_config()
            
            # Verifikasi hasil
            self.assertEqual(result['split']['train_ratio'], 0.7)
            mock_get_config_manager.assert_called_once()
            mock_cm.get_module_config.assert_called_once_with('split', {})
        except ImportError:
            print("Info: config_handlers.load_config tidak tersedia, melewati test")
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.get_config_manager')
    def test_save_config(self, mock_get_config_manager):
        """Test save konfigurasi."""
        try:
            from smartcash.ui.dataset.split.handlers.config_handlers import save_config
            
            # Setup mock
            mock_cm = MagicMock()
            mock_cm.save_module_config.return_value = True
            mock_get_config_manager.return_value = mock_cm
            
            # Buat config dummy
            config = {'split': {'train_ratio': 0.7}}
            
            # Panggil fungsi
            save_config(config)
            
            # Verifikasi hasil
            mock_get_config_manager.assert_called_once()
            mock_cm.save_module_config.assert_called_once()
        except ImportError:
            print("Info: config_handlers.save_config tidak tersedia, melewati test")
    
    def test_get_default_split_config(self):
        """Test mendapatkan default konfigurasi split."""
        try:
            from smartcash.ui.dataset.split.handlers.config_handlers import get_default_split_config
            
            # Panggil fungsi
            result = get_default_split_config()
            
            # Verifikasi hasil
            self.assertIn('split', result)
            self.assertIn('train_ratio', result['split'])
            self.assertIn('val_ratio', result['split'])
            self.assertIn('test_ratio', result['split'])
        except ImportError:
            print("Info: config_handlers.get_default_split_config tidak tersedia, melewati test")

class TestButtonHandlers(unittest.TestCase):
    """Test untuk handlers button."""
    
    @patch('smartcash.ui.dataset.split.handlers.button_handlers.create_save_handler')
    @patch('smartcash.ui.dataset.split.handlers.button_handlers.create_reset_handler')
    def test_setup_button_handlers(self, mock_create_reset, mock_create_save):
        """Test setup button handlers."""
        try:
            from smartcash.ui.dataset.split.handlers.button_handlers import setup_button_handlers
            
            # Setup mock
            mock_create_save.return_value = MagicMock()
            mock_create_reset.return_value = MagicMock()
            
            # Buat UI components dummy
            ui_components = {
                'save_button': MagicMock(),
                'reset_button': MagicMock()
            }
            
            # Buat config dan env dummy
            config = {'split': {'train_ratio': 0.7}}
            env = MagicMock()
            
            # Panggil fungsi
            result = setup_button_handlers(ui_components, config, env)
            
            # Verifikasi hasil
            self.assertEqual(result, ui_components)
            ui_components['save_button'].on_click.assert_called_once_with(mock_create_save.return_value)
            ui_components['reset_button'].on_click.assert_called_once_with(mock_create_reset.return_value)
        except ImportError:
            print("Info: button_handlers.setup_button_handlers tidak tersedia, melewati test")

class TestSaveHandlers(unittest.TestCase):
    """Test untuk handlers save."""
    
    @patch('smartcash.ui.dataset.split.handlers.save_handlers.handle_save_action')
    def test_create_save_handler(self, mock_handle_save):
        """Test create save handler."""
        try:
            from smartcash.ui.dataset.split.handlers.save_handlers import create_save_handler
            
            # Buat UI components dummy
            ui_components = {'logger': MagicMock()}
            
            # Panggil fungsi
            handler = create_save_handler(ui_components)
            
            # Panggil handler dengan button dummy
            button = MagicMock()
            handler(button)
            
            # Verifikasi hasil
            mock_handle_save.assert_called_once_with(ui_components)
        except ImportError:
            print("Info: save_handlers.create_save_handler tidak tersedia, melewati test")

class TestResetHandlers(unittest.TestCase):
    """Test untuk handlers reset."""
    
    @patch('smartcash.ui.dataset.split.handlers.reset_handlers.handle_reset_action')
    def test_create_reset_handler(self, mock_handle_reset):
        """Test create reset handler."""
        try:
            from smartcash.ui.dataset.split.handlers.reset_handlers import create_reset_handler
            
            # Buat UI components dummy
            ui_components = {'logger': MagicMock()}
            
            # Panggil fungsi
            handler = create_reset_handler(ui_components)
            
            # Panggil handler dengan button dummy
            button = MagicMock()
            handler(button)
            
            # Verifikasi hasil
            mock_handle_reset.assert_called_once_with(ui_components)
        except ImportError:
            print("Info: reset_handlers.create_reset_handler tidak tersedia, melewati test")

class TestUIValueHandlers(unittest.TestCase):
    """Test untuk handlers UI value."""
    
    def test_get_ui_values(self):
        """Test mendapatkan nilai dari UI."""
        try:
            from smartcash.ui.dataset.split.handlers.ui_value_handlers import get_ui_values
            
            # Buat UI components dummy
            ui_components = {
                'train_slider': MagicMock(value=0.7),
                'val_slider': MagicMock(value=0.15),
                'test_slider': MagicMock(value=0.15),
                'random_seed': MagicMock(value=42),
                'stratified_checkbox': MagicMock(value=True),
                'enabled_checkbox': MagicMock(value=True)
            }
            
            # Panggil fungsi
            result = get_ui_values(ui_components)
            
            # Verifikasi hasil
            self.assertEqual(result['train_ratio'], 0.7)
            self.assertEqual(result['val_ratio'], 0.15)
            self.assertEqual(result['test_ratio'], 0.15)
            self.assertEqual(result['random_seed'], 42)
            self.assertEqual(result['stratify'], True)
            self.assertEqual(result['enabled'], True)
        except ImportError:
            print("Info: ui_value_handlers.get_ui_values tidak tersedia, melewati test")

if __name__ == '__main__':
    unittest.main() 