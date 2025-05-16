"""
File: smartcash/ui/dataset/augmentation/tests/test_config_sync.py
Deskripsi: Test untuk sinkronisasi konfigurasi augmentasi dengan Google Drive
"""

import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys
from pathlib import Path

class TestConfigSync(unittest.TestCase):
    """Test untuk sinkronisasi konfigurasi augmentasi dengan Google Drive"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock modules untuk menghindari import error
        sys.modules['smartcash.ui.utils.constants'] = MagicMock()
        sys.modules['smartcash.ui.utils.alert_utils'] = MagicMock()
        sys.modules['smartcash.common.logger'] = MagicMock()
        sys.modules['smartcash.components.observer'] = MagicMock()
        
        # Mock untuk logger
        self.mock_logger = MagicMock()
        
        # Mock untuk UI components
        self.ui_components = {
            'logger': self.mock_logger,
            'status': MagicMock(),
            'update_config_from_ui': MagicMock(),
        }
        
        # Mock untuk config_manager
        self.mock_config_manager = MagicMock()
        self.mock_get_config_manager = patch('smartcash.common.config.manager.get_config_manager')
        self.mock_get_config_manager_func = self.mock_get_config_manager.start()
        self.mock_get_config_manager_func.return_value = self.mock_config_manager
        
        # Mock untuk update_config_from_ui
        self.ui_components['update_config_from_ui'].return_value = {'augmentation': {'enabled': True}}
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        self.mock_get_config_manager.stop()
    
    def test_sync_config_with_drive_success(self):
        """Test sinkronisasi konfigurasi dengan Google Drive berhasil"""
        from smartcash.ui.dataset.augmentation.handlers.persistence_handler import sync_config_with_drive
        
        # Setup mock
        self.mock_config_manager.save_module_config.return_value = True
        self.mock_config_manager.sync_to_drive.return_value = (True, "Konfigurasi berhasil disinkronkan dengan Google Drive")
        
        # Panggil fungsi
        result = sync_config_with_drive(self.ui_components)
        
        # Verifikasi hasil
        self.assertTrue(result)
        self.mock_config_manager.save_module_config.assert_called_once_with('augmentation', {'augmentation': {'enabled': True}})
        self.mock_config_manager.sync_to_drive.assert_called_once_with('augmentation')
        self.mock_logger.info.assert_called_once()
    
    def test_sync_config_with_drive_save_failed(self):
        """Test sinkronisasi konfigurasi dengan Google Drive gagal saat menyimpan"""
        from smartcash.ui.dataset.augmentation.handlers.persistence_handler import sync_config_with_drive
        
        # Setup mock
        self.mock_config_manager.save_module_config.return_value = False
        
        # Panggil fungsi
        result = sync_config_with_drive(self.ui_components)
        
        # Verifikasi hasil
        self.assertFalse(result)
        self.mock_config_manager.save_module_config.assert_called_once_with('augmentation', {'augmentation': {'enabled': True}})
        self.mock_config_manager.sync_to_drive.assert_not_called()
        self.mock_logger.warning.assert_called_once()
    
    def test_sync_config_with_drive_sync_failed(self):
        """Test sinkronisasi konfigurasi dengan Google Drive gagal saat sinkronisasi"""
        from smartcash.ui.dataset.augmentation.handlers.persistence_handler import sync_config_with_drive
        
        # Setup mock
        self.mock_config_manager.save_module_config.return_value = True
        self.mock_config_manager.sync_to_drive.return_value = (False, "Error saat sinkronisasi dengan Google Drive")
        
        # Panggil fungsi
        result = sync_config_with_drive(self.ui_components)
        
        # Verifikasi hasil
        self.assertFalse(result)
        self.mock_config_manager.save_module_config.assert_called_once_with('augmentation', {'augmentation': {'enabled': True}})
        self.mock_config_manager.sync_to_drive.assert_called_once_with('augmentation')
        self.mock_logger.warning.assert_called_once()
    
    def test_sync_config_with_drive_no_sync_function(self):
        """Test sinkronisasi konfigurasi dengan Google Drive tanpa fungsi sync"""
        from smartcash.ui.dataset.augmentation.handlers.persistence_handler import sync_config_with_drive
        
        # Setup mock
        self.mock_config_manager.save_module_config.return_value = True
        del self.mock_config_manager.sync_to_drive  # Hapus fungsi sync_to_drive
        
        # Panggil fungsi
        result = sync_config_with_drive(self.ui_components)
        
        # Verifikasi hasil
        self.assertTrue(result)
        self.mock_config_manager.save_module_config.assert_called_once_with('augmentation', {'augmentation': {'enabled': True}})
        self.mock_logger.info.assert_called_once()
    
    def test_sync_config_with_drive_exception(self):
        """Test sinkronisasi konfigurasi dengan Google Drive dengan exception"""
        from smartcash.ui.dataset.augmentation.handlers.persistence_handler import sync_config_with_drive
        
        # Setup mock
        self.mock_config_manager.save_module_config.side_effect = Exception("Test exception")
        
        # Panggil fungsi
        result = sync_config_with_drive(self.ui_components)
        
        # Verifikasi hasil
        self.assertFalse(result)
        self.mock_logger.warning.assert_called_once()
        self.mock_logger.debug.assert_called_once()

if __name__ == '__main__':
    unittest.main()
