"""
File: smartcash/ui/setup/env_config/tests/test_setup_handler.py
Deskripsi: Integration test untuk SetupHandler
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from smartcash.ui.setup.env_config.handlers import SetupHandler
from smartcash.common.config.manager import SimpleConfigManager

class TestSetupHandler(unittest.TestCase):
    """
    Integration test untuk SetupHandler
    """
    
    def setUp(self):
        """
        Setup untuk test
        """
        # Mock callbacks untuk UI
        self.log_messages = []
        self.status_updates = []
        
        def mock_log_message(message):
            self.log_messages.append(message)
        
        def mock_update_status(message, status_type="info"):
            self.status_updates.append((message, status_type))
        
        self.ui_callback = {
            'log_message': mock_log_message,
            'update_status': mock_update_status
        }
        
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.EnvironmentSetupHandler.setup_environment')
    def test_perform_setup_success(self, mock_setup_environment):
        """
        Test integration perform_setup dengan EnvironmentSetupHandler untuk kasus sukses
        """
        # Mock setup_environment
        mock_config_manager = MagicMock(spec=SimpleConfigManager)
        mock_base_dir = Path('/mock/base/dir')
        mock_config_dir = Path('/mock/config/dir')
        mock_setup_environment.return_value = (mock_config_manager, mock_base_dir, mock_config_dir)
        
        # Buat handler dengan UI callbacks
        handler = SetupHandler(self.ui_callback)
        
        # Panggil perform_setup
        config_manager, base_dir, config_dir = handler.perform_setup()
        
        # Verifikasi hasil
        self.assertEqual(config_manager, mock_config_manager)
        self.assertEqual(base_dir, mock_base_dir)
        self.assertEqual(config_dir, mock_config_dir)
        
        # Verifikasi interaksi dengan UI
        self.assertTrue(any("Memulai setup environment" in msg for msg in self.log_messages), 
                       "Log message memulai setup tidak ditemukan")
        self.assertTrue(any("Setup environment berhasil" in msg for msg in self.log_messages), 
                       "Log message sukses tidak ditemukan")
        
        # Verifikasi status updates
        self.assertTrue(any("Setup environment sedang berjalan" in msg for msg, _ in self.status_updates), 
                       "Status update 'sedang berjalan' tidak ditemukan")
        self.assertTrue(any(status == "success" for msg, status in self.status_updates if "berhasil" in msg), 
                       "Status update success tidak ditemukan")
    
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.EnvironmentSetupHandler.setup_environment')
    def test_perform_setup_error(self, mock_setup_environment):
        """
        Test integration perform_setup dengan EnvironmentSetupHandler untuk kasus error
        """
        # Mock setup_environment untuk throw exception
        mock_error = Exception("Mock setup error")
        mock_setup_environment.side_effect = mock_error
        
        # Buat handler dengan UI callbacks
        handler = SetupHandler(self.ui_callback)
        
        # Panggil perform_setup dan expect exception
        with self.assertRaises(Exception) as context:
            handler.perform_setup()
        
        # Verifikasi exception yang dihasilkan
        self.assertEqual(str(context.exception), "Mock setup error")
        
        # Verifikasi interaksi dengan UI
        self.assertTrue(any("Error saat setup environment" in msg for msg in self.log_messages), 
                       "Log message error tidak ditemukan")
        
        # Verifikasi status updates
        self.assertTrue(any(status == "error" for msg, status in self.status_updates if "Error" in msg), 
                       "Status update error tidak ditemukan")
    
    @patch('smartcash.ui.setup.env_config.handlers.setup_handler.get_config_manager')
    def test_handle_error(self, mock_get_config_manager):
        """
        Test integration handle_error dengan sistem config manager
        """
        # Mock get_config_manager
        mock_config_manager = MagicMock(spec=SimpleConfigManager)
        mock_get_config_manager.return_value = mock_config_manager
        
        # Buat handler dengan UI callbacks
        handler = SetupHandler(self.ui_callback)
        
        # Panggil handle_error
        error = Exception("Test error message")
        result = handler.handle_error(error)
        
        # Verifikasi hasil
        self.assertEqual(result, mock_config_manager)
        
        # Verifikasi interaksi dengan UI
        self.assertTrue(any("Error saat inisialisasi environment" in msg for msg in self.log_messages), 
                       "Log message error tidak ditemukan")
        self.assertTrue(any("Berhasil mendapatkan config manager fallback" in msg for msg in self.log_messages), 
                       "Log message fallback sukses tidak ditemukan")
        
        # Verifikasi status updates
        self.assertTrue(any(status == "error" for msg, status in self.status_updates if "Error" in msg), 
                       "Status update error tidak ditemukan")
    
    @patch('smartcash.ui.setup.env_config.handlers.setup_handler.get_config_manager')
    def test_handle_error_fallback_fails(self, mock_get_config_manager):
        """
        Test integration handle_error dengan sistem config manager untuk kasus fallback gagal
        """
        # Mock get_config_manager untuk throw exception
        mock_get_config_manager.side_effect = Exception("Fallback error")
        
        # Buat handler dengan UI callbacks
        handler = SetupHandler(self.ui_callback)
        
        # Panggil handle_error
        error = Exception("Test error message")
        result = handler.handle_error(error)
        
        # Verifikasi hasil
        self.assertIsNone(result)
        
        # Verifikasi interaksi dengan UI
        self.assertTrue(any("Error saat inisialisasi environment" in msg for msg in self.log_messages), 
                       "Log message error tidak ditemukan")
        self.assertTrue(any("Error mendapatkan config manager fallback" in msg for msg in self.log_messages), 
                       "Log message fallback error tidak ditemukan")
        
        # Verifikasi status updates
        self.assertTrue(any(status == "error" for msg, status in self.status_updates if "Error" in msg), 
                       "Status update error tidak ditemukan")

if __name__ == '__main__':
    unittest.main() 