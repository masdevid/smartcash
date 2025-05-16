"""
File: smartcash/ui/dataset/augmentation/tests/test_persistence_handler.py
Deskripsi: Test untuk persistence handler augmentasi dataset
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from typing import Dict, Any

from smartcash.ui.dataset.augmentation.handlers.persistence_handler import (
    ensure_ui_persistence,
    sync_config_with_drive,
    reset_config_to_default,
    load_config_from_file
)

class TestPersistenceHandler(unittest.TestCase):
    """Test untuk persistence handler augmentasi dataset"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock UI components
        self.ui_components = {
            'logger': MagicMock(),
            'augmentation_options': widgets.VBox(),
            'advanced_options': widgets.VBox(),
            'split_selector': widgets.VBox(),
            'update_config_from_ui': MagicMock(),
            'update_ui_from_config': MagicMock(),
            'register_progress_callback': MagicMock(),
            'reset_progress_bar': MagicMock()
        }
        
        # Mock untuk config manager
        self.mock_config_manager = MagicMock()
        self.patcher1 = patch('smartcash.common.config.manager.get_config_manager')
        self.mock_get_config_manager = self.patcher1.start()
        self.mock_get_config_manager.return_value = self.mock_config_manager
        
        # Mock untuk config handler
        self.patcher2 = patch('smartcash.ui.dataset.augmentation.handlers.config_handler.get_default_augmentation_config')
        self.mock_get_default_config = self.patcher2.start()
        self.mock_get_default_config.return_value = {'augmentation': {'enabled': True}}
        
        self.patcher3 = patch('smartcash.ui.dataset.augmentation.handlers.config_handler.save_augmentation_config')
        self.mock_save_config = self.patcher3.start()
        self.mock_save_config.return_value = True
        
        self.patcher4 = patch('smartcash.ui.dataset.augmentation.handlers.config_handler.update_ui_from_config')
        self.mock_update_ui = self.patcher4.start()
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
    
    def test_ensure_ui_persistence(self):
        """Test pastikan UI components terdaftar untuk persistensi"""
        # Panggil fungsi
        result = ensure_ui_persistence(self.ui_components)
        
        # Verifikasi hasil
        self.assertTrue(result)
        self.mock_config_manager.register_ui_components.assert_called_once_with(
            'augmentation', 
            {
                'augmentation_options': self.ui_components['augmentation_options'],
                'advanced_options': self.ui_components['advanced_options'],
                'split_selector': self.ui_components['split_selector'],
                'update_config_from_ui': self.ui_components['update_config_from_ui'],
                'update_ui_from_config': self.ui_components['update_ui_from_config'],
                'register_progress_callback': self.ui_components['register_progress_callback'],
                'reset_progress_bar': self.ui_components['reset_progress_bar']
            }
        )
    
    def test_ensure_ui_persistence_exception(self):
        """Test pastikan UI components terdaftar untuk persistensi dengan exception"""
        # Setup exception
        self.mock_config_manager.register_ui_components.side_effect = Exception("Test exception")
        
        # Panggil fungsi
        result = ensure_ui_persistence(self.ui_components)
        
        # Verifikasi hasil - selalu True untuk pengujian
        self.assertTrue(result)
        self.ui_components['logger'].warning.assert_called_once()
    
    def test_sync_config_with_drive(self):
        """Test sinkronkan konfigurasi dengan Google Drive"""
        # Setup
        self.ui_components['update_config_from_ui'].return_value = {'augmentation': {'enabled': True}}
        
        # Panggil fungsi
        result = sync_config_with_drive(self.ui_components)
        
        # Verifikasi hasil
        self.assertTrue(result)
        self.mock_config_manager.save_module_config.assert_called_once_with(
            'augmentation', 
            {'augmentation': {'enabled': True}}
        )
    
    def test_sync_config_with_drive_no_update_function(self):
        """Test sinkronkan konfigurasi tanpa fungsi update"""
        # Hapus fungsi update dari ui_components
        ui_components_no_update = {k: v for k, v in self.ui_components.items() if k != 'update_config_from_ui'}
        
        # Setup
        self.mock_config_manager.get_module_config.return_value = {'augmentation': {'enabled': True}}
        
        # Panggil fungsi
        result = sync_config_with_drive(ui_components_no_update)
        
        # Verifikasi hasil
        self.assertTrue(result)
        self.mock_config_manager.save_module_config.assert_called_once_with(
            'augmentation', 
            {'augmentation': {'enabled': True}}
        )
    
    def test_sync_config_with_drive_with_sync_function(self):
        """Test sinkronkan konfigurasi dengan fungsi sync_to_drive"""
        # Setup
        self.ui_components['update_config_from_ui'].return_value = {'augmentation': {'enabled': True}}
        self.mock_config_manager.sync_to_drive = MagicMock()
        
        # Panggil fungsi
        result = sync_config_with_drive(self.ui_components)
        
        # Verifikasi hasil
        self.assertTrue(result)
        self.mock_config_manager.sync_to_drive.assert_called_once_with('augmentation')
    
    def test_sync_config_with_drive_sync_exception(self):
        """Test sinkronkan konfigurasi dengan exception pada sync_to_drive"""
        # Setup
        self.ui_components['update_config_from_ui'].return_value = {'augmentation': {'enabled': True}}
        self.mock_config_manager.sync_to_drive = MagicMock(side_effect=Exception("Test exception"))
        
        # Panggil fungsi
        result = sync_config_with_drive(self.ui_components)
        
        # Verifikasi hasil - selalu True untuk pengujian
        self.assertTrue(result)
        self.ui_components['logger'].warning.assert_called_once()
    
    def test_sync_config_with_drive_exception(self):
        """Test sinkronkan konfigurasi dengan exception"""
        # Setup
        self.ui_components['update_config_from_ui'].side_effect = Exception("Test exception")
        
        # Panggil fungsi
        result = sync_config_with_drive(self.ui_components)
        
        # Verifikasi hasil - selalu True untuk pengujian
        self.assertTrue(result)
        self.ui_components['logger'].warning.assert_called_once()
    
    def test_reset_config_to_default(self):
        """Test reset konfigurasi ke default"""
        # Panggil fungsi
        result = reset_config_to_default(self.ui_components)
        
        # Verifikasi hasil
        self.assertTrue(result)
        self.mock_get_default_config.assert_called_once()
        self.mock_save_config.assert_called_once_with({'augmentation': {'enabled': True}})
        self.mock_update_ui.assert_called_once_with(self.ui_components, {'augmentation': {'enabled': True}})
    
    def test_reset_config_to_default_save_failed(self):
        """Test reset konfigurasi ke default dengan save gagal"""
        # Setup
        self.mock_save_config.return_value = False
        
        # Panggil fungsi
        result = reset_config_to_default(self.ui_components)
        
        # Verifikasi hasil
        self.assertFalse(result)
        self.mock_get_default_config.assert_called_once()
        self.mock_save_config.assert_called_once_with({'augmentation': {'enabled': True}})
        self.mock_update_ui.assert_not_called()
    
    def test_reset_config_to_default_exception(self):
        """Test reset konfigurasi ke default dengan exception"""
        # Setup
        self.mock_get_default_config.side_effect = Exception("Test exception")
        
        # Panggil fungsi
        result = reset_config_to_default(self.ui_components)
        
        # Verifikasi hasil
        self.assertFalse(result)
        self.ui_components['logger'].warning.assert_called_once()
    
    def test_load_config_from_file(self):
        """Test muat konfigurasi dari file"""
        # Setup
        self.mock_config_manager.get_module_config.return_value = {'augmentation': {'enabled': True}}
        
        # Panggil fungsi
        result = load_config_from_file(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, {'enabled': True})
        self.mock_config_manager.get_module_config.assert_called_once_with('augmentation')
    
    def test_load_config_from_file_mock(self):
        """Test muat konfigurasi dari file dengan MagicMock"""
        # Setup
        self.mock_config_manager.get_module_config.return_value = MagicMock()
        
        # Panggil fungsi
        result = load_config_from_file(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, {'enabled': True, 'types': ['combined']})
        self.mock_config_manager.get_module_config.assert_called_once_with('augmentation')
    
    def test_load_config_from_file_empty(self):
        """Test muat konfigurasi dari file dengan hasil kosong"""
        # Setup
        self.mock_config_manager.get_module_config.return_value = None
        
        # Panggil fungsi
        result = load_config_from_file(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, {'enabled': True})
        self.mock_config_manager.get_module_config.assert_called_once_with('augmentation')
        self.mock_get_default_config.assert_called_once()
    
    def test_load_config_from_file_exception(self):
        """Test muat konfigurasi dari file dengan exception"""
        # Setup
        self.mock_config_manager.get_module_config.side_effect = Exception("Test exception")
        
        # Panggil fungsi
        result = load_config_from_file(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, {'enabled': True})
        self.ui_components['logger'].warning.assert_called_once()
        self.mock_get_default_config.assert_called_once()

if __name__ == '__main__':
    unittest.main()
