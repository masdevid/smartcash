"""
File: smartcash/ui/setup/env_config/tests/test_environment_setup_handler.py
Deskripsi: Unit test untuk environment_setup_handler.py
"""

import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from smartcash.ui.setup.env_config.handlers.environment_setup_handler import EnvironmentSetupHandler, setup_environment

class TestEnvironmentSetupHandler(unittest.TestCase):
    """
    Test untuk environment_setup_handler.py
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
    
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.is_colab')
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.LocalSetupHandler')
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.get_config_manager')
    @patch('pathlib.Path.exists')
    def test_setup_environment_local(self, mock_exists, mock_get_config_manager, mock_local_handler_class, mock_is_colab):
        """
        Test setup_environment di environment local
        """
        # Setup mocks
        mock_is_colab.return_value = False
        mock_exists.return_value = True
        
        # Mock local handler
        mock_local_handler = MagicMock()
        mock_local_handler_class.return_value = mock_local_handler
        
        # Mock hasil setup_local_environment
        mock_base_dir = Path('/path/to/base')
        mock_config_dir = Path('/path/to/config')
        mock_local_handler.setup_local_environment.return_value = (mock_base_dir, mock_config_dir)
        
        # Mock config manager
        mock_config_manager = MagicMock()
        mock_get_config_manager.return_value = mock_config_manager
        
        # Initialize handler
        handler = EnvironmentSetupHandler(self.ui_callback)
        
        # Call method to test
        result = handler.setup_environment()
        
        # Verify hasil
        self.assertEqual(result, (mock_config_manager, mock_base_dir, mock_config_dir))
        
        # Verify log messages
        self.assertTrue(any("Mendeteksi environment lokal" in msg for msg in self.log_messages), 
                      "Log message 'Mendeteksi environment lokal' tidak ditemukan")
        
        # Verify status updates
        self.assertTrue(any("Environment berhasil dikonfigurasi" in msg[0] for msg in self.status_updates), 
                      "Status message 'Environment berhasil dikonfigurasi' tidak ditemukan")
        
        # Verify mock method calls
        mock_local_handler.setup_local_environment.assert_called_once()
        mock_get_config_manager.assert_called_once()
    
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.is_colab')
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.ColabSetupHandler')
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.get_config_manager')
    @patch('pathlib.Path.exists')
    def test_setup_environment_colab(self, mock_exists, mock_get_config_manager, mock_colab_handler_class, mock_is_colab):
        """
        Test setup_environment di environment Colab
        """
        # Setup mocks
        mock_is_colab.return_value = True
        mock_exists.return_value = True
        
        # Mock colab handler
        mock_colab_handler = MagicMock()
        mock_colab_handler_class.return_value = mock_colab_handler
        
        # Mock hasil setup_colab_environment
        mock_base_dir = Path('/content')
        mock_config_dir = Path('/content/drive/MyDrive/SmartCash/configs')
        mock_colab_handler.setup_colab_environment.return_value = (mock_base_dir, mock_config_dir)
        
        # Mock config manager
        mock_config_manager = MagicMock()
        mock_get_config_manager.return_value = mock_config_manager
        
        # Initialize handler
        handler = EnvironmentSetupHandler(self.ui_callback)
        
        # Call method to test
        result = handler.setup_environment()
        
        # Verify hasil
        self.assertEqual(result, (mock_config_manager, mock_base_dir, mock_config_dir))
        
        # Verify log messages
        self.assertTrue(any("Mendeteksi environment Google Colab" in msg for msg in self.log_messages), 
                      "Log message 'Mendeteksi environment Google Colab' tidak ditemukan")
        
        # Verify status updates
        self.assertTrue(any("Environment berhasil dikonfigurasi" in msg[0] for msg in self.status_updates), 
                      "Status message 'Environment berhasil dikonfigurasi' tidak ditemukan")
        
        # Verify mock method calls
        mock_colab_handler.setup_colab_environment.assert_called_once()
        mock_get_config_manager.assert_called_once()
    
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.is_colab')
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.LocalSetupHandler')
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.get_config_manager')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    def test_setup_environment_config_dir_not_exists(self, mock_mkdir, mock_exists, 
                                                  mock_get_config_manager, mock_local_handler_class, mock_is_colab):
        """
        Test setup_environment ketika config_dir tidak ada
        """
        # Setup mocks
        mock_is_colab.return_value = False
        mock_exists.return_value = False  # Config dir tidak ada
        
        # Mock local handler
        mock_local_handler = MagicMock()
        mock_local_handler_class.return_value = mock_local_handler
        
        # Mock hasil setup_local_environment
        mock_base_dir = Path('/path/to/base')
        mock_config_dir = Path('/path/to/config')
        mock_local_handler.setup_local_environment.return_value = (mock_base_dir, mock_config_dir)
        
        # Mock config manager
        mock_config_manager = MagicMock()
        mock_get_config_manager.return_value = mock_config_manager
        
        # Initialize handler
        handler = EnvironmentSetupHandler(self.ui_callback)
        
        # Call method to test
        result = handler.setup_environment()
        
        # Verify hasil
        self.assertEqual(result, (mock_config_manager, mock_base_dir, mock_config_dir))
        
        # Verify log messages
        self.assertTrue(any("Mendeteksi environment lokal" in msg for msg in self.log_messages), 
                      "Log message 'Mendeteksi environment lokal' tidak ditemukan")
        self.assertTrue(any("Gagal membuat symlink atau direktori konfigurasi" in msg for msg in self.log_messages), 
                      "Log message tentang kegagalan membuat symlink tidak ditemukan")
        
        # Verify mkdir dipanggil
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.is_colab')
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.LocalSetupHandler')
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.get_config_manager')
    @patch('pathlib.Path.exists')
    def test_setup_environment_config_manager_exception(self, mock_exists, 
                                                     mock_get_config_manager, mock_local_handler_class, mock_is_colab):
        """
        Test setup_environment ketika get_config_manager melempar exception
        """
        # Setup mocks
        mock_is_colab.return_value = False
        mock_exists.return_value = True
        
        # Mock local handler
        mock_local_handler = MagicMock()
        mock_local_handler_class.return_value = mock_local_handler
        
        # Mock hasil setup_local_environment
        mock_base_dir = Path('/path/to/base')
        mock_config_dir = Path('/path/to/config')
        mock_local_handler.setup_local_environment.return_value = (mock_base_dir, mock_config_dir)
        
        # Mock get_config_manager untuk melempar exception
        mock_exception = Exception("Config manager error")
        mock_get_config_manager.side_effect = mock_exception
        
        # Initialize handler
        handler = EnvironmentSetupHandler(self.ui_callback)
        
        # Call method to test dan expect exception
        with self.assertRaises(Exception) as context:
            handler.setup_environment()
        
        # Verify log messages
        self.assertTrue(any("Error saat inisialisasi config manager" in msg for msg in self.log_messages), 
                      "Log message tentang error saat inisialisasi config manager tidak ditemukan")
        
        # Verify status updates
        self.assertTrue(any("Error saat inisialisasi config manager" in msg[0] and msg[1] == "error" 
                         for msg in self.status_updates), 
                      "Status error message tidak ditemukan")
    
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.EnvironmentSetupHandler')
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.get_fallback_logger')
    def test_setup_environment_function(self, mock_get_fallback_logger, mock_handler_class):
        """
        Test fungsi setup_environment (compatibility function)
        """
        # Mock logger
        mock_logger = MagicMock()
        
        # Mock EnvironmentSetupHandler
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        
        # Mock hasil setup_environment
        expected_result = (MagicMock(), Path('/path/to/base'), Path('/path/to/config'))
        mock_handler.setup_environment.return_value = expected_result
        
        # Buat ui_components dengan logger
        ui_components = {
            'logger': mock_logger,
            'log_message': MagicMock(),
            'update_status': MagicMock()
        }
        
        # Call function to test
        result = setup_environment(ui_components)
        
        # Verify hasil
        self.assertEqual(result, expected_result)
        
        # Verify EnvironmentSetupHandler dibuat dengan callback yang benar
        mock_handler_class.assert_called_once()
        
        # Ekstrak ui_callbacks yang diberikan ke EnvironmentSetupHandler
        ui_callbacks = mock_handler_class.call_args[0][0]  # Ambil positional argument pertama
        
        # Verifikasi bahwa ui_callbacks memiliki kunci yang diperlukan
        self.assertIn('log_message', ui_callbacks)
        self.assertIn('update_status', ui_callbacks)
    
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.EnvironmentSetupHandler')
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.get_fallback_logger')
    def test_setup_environment_function_no_logger(self, mock_get_fallback_logger, mock_handler_class):
        """
        Test fungsi setup_environment tanpa logger
        """
        # Mock fallback logger
        mock_logger = MagicMock()
        mock_get_fallback_logger.return_value = mock_logger
        
        # Mock EnvironmentSetupHandler
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        
        # Mock hasil setup_environment
        expected_result = (MagicMock(), Path('/path/to/base'), Path('/path/to/config'))
        mock_handler.setup_environment.return_value = expected_result
        
        # Buat ui_components tanpa logger
        ui_components = {}
        
        # Call function to test
        result = setup_environment(ui_components)
        
        # Verify hasil
        self.assertEqual(result, expected_result)
        
        # Verify get_fallback_logger dipanggil
        mock_get_fallback_logger.assert_called_once_with("env_config_setup")

if __name__ == '__main__':
    unittest.main() 