"""
File: smartcash/ui/setup/env_config/tests/test_integration.py
Deskripsi: Integration test untuk integrasi penuh env_config
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from smartcash.common.config.manager import SimpleConfigManager
from smartcash.ui.setup.env_config.env_config_initializer import initialize_env_config_ui
from smartcash.ui.setup.env_config.components import UIFactory
from smartcash.ui.setup.env_config.handlers import SetupHandler

class TestEnvConfigIntegration(unittest.TestCase):
    """
    Integration test untuk interaksi penuh antara EnvConfigInitializer, 
    UIFactory, dan SetupHandler
    """
    
    @patch('smartcash.ui.setup.env_config.env_config_initializer.SetupHandler')
    def test_initialize_env_config_ui_success(self, mock_setup_handler_class):
        """
        Test integrasi penuh initialize_env_config_ui untuk kasus sukses
        """
        # Setup mock
        mock_setup_handler = MagicMock()
        mock_setup_handler_class.return_value = mock_setup_handler
        
        # Mock perform_setup untuk mengembalikan konfigurasi dan direktori
        mock_config_manager = MagicMock(spec=SimpleConfigManager)
        mock_base_dir = Path('/mock/base/dir')
        mock_config_dir = Path('/mock/config/dir')
        mock_setup_handler.perform_setup.return_value = (mock_config_manager, mock_base_dir, mock_config_dir)
        
        # Panggil initialize_env_config_ui
        ui_components = initialize_env_config_ui()
        
        # Verifikasi SetupHandler dibuat dengan callback UI yang sesuai
        mock_setup_handler_class.assert_called_once()
        
        # Periksa argumen yang diberikan ke konstruktor SetupHandler
        args, kwargs = mock_setup_handler_class.call_args
        self.assertEqual(len(args), 1, "SetupHandler harus dibuat dengan 1 argumen positional")
        ui_callbacks = args[0]
        self.assertIn('log_message', ui_callbacks, "Callback log_message tidak ada")
        self.assertIn('update_status', ui_callbacks, "Callback update_status tidak ada")
        
        # Verifikasi perform_setup dipanggil
        mock_setup_handler.perform_setup.assert_called_once()
        
        # Verifikasi hasil initialize_env_config_ui berisi semua yang diharapkan
        self.assertIsNotNone(ui_components, "UI components tidak boleh None")
        self.assertIn('config_manager', ui_components, "Config manager tidak ada di UI components")
        self.assertEqual(ui_components['config_manager'], mock_config_manager)
        
        # Verifikasi komponen UI hasil dari UIFactory ada di hasil
        self.assertIn('header', ui_components, "Header tidak ada di UI components")
        self.assertIn('setup_button', ui_components, "Setup button tidak ada di UI components")
        self.assertIn('status_panel', ui_components, "Status panel tidak ada di UI components")
        self.assertIn('log_panel', ui_components, "Log panel tidak ada di UI components")
        self.assertIn('ui_layout', ui_components, "UI layout tidak ada di UI components")
    
    @patch('smartcash.ui.setup.env_config.env_config_initializer.SetupHandler')
    @patch('smartcash.ui.setup.env_config.env_config_initializer.UIFactory')
    def test_initialize_env_config_ui_error(self, mock_ui_factory, mock_setup_handler_class):
        """
        Test integrasi penuh initialize_env_config_ui untuk kasus error
        """
        # Setup mock
        mock_setup_handler = MagicMock()
        mock_setup_handler_class.return_value = mock_setup_handler
        
        # Mock perform_setup untuk menghasilkan exception
        mock_error = Exception("Test setup error")
        mock_setup_handler.perform_setup.side_effect = mock_error
        
        # Mock handle_error untuk mengembalikan config manager fallback
        mock_config_manager = MagicMock(spec=SimpleConfigManager)
        mock_setup_handler.handle_error.return_value = mock_config_manager
        
        # Mock UIFactory.create_error_ui_components
        mock_error_ui_components = {
            'header': MagicMock(),
            'error_alert': MagicMock(),
            'log_panel': MagicMock(),
            'log_output': MagicMock(),
            'ui_layout': MagicMock()
        }
        mock_ui_factory.create_error_ui_components.return_value = mock_error_ui_components
        
        # Panggil initialize_env_config_ui
        ui_components = initialize_env_config_ui()
        
        # Verifikasi UIFactory.create_error_ui_components dipanggil dengan pesan error
        mock_ui_factory.create_error_ui_components.assert_called_once()
        args, kwargs = mock_ui_factory.create_error_ui_components.call_args
        self.assertEqual(args[0], "Test setup error", "Pesan error tidak sesuai")
        
        # Verifikasi handle_error dipanggil
        mock_setup_handler.handle_error.assert_called_once_with(mock_error)
        
        # Verifikasi hasil initialize_env_config_ui berisi semua yang diharapkan
        self.assertIsNotNone(ui_components, "UI components tidak boleh None")
        self.assertEqual(ui_components, mock_error_ui_components, "UI components tidak sama dengan mock_error_ui_components")
        self.assertIn('config_manager', ui_components, "Config manager tidak ada di UI components")
        self.assertEqual(ui_components['config_manager'], mock_config_manager)
    
    @patch('smartcash.ui.setup.env_config.handlers.environment_setup_handler.EnvironmentSetupHandler.setup_environment')
    def test_real_integration_between_ui_factory_and_setup_handler(self, mock_setup_environment):
        """
        Test integrasi nyata antara UIFactory dan SetupHandler tanpa mock berlebihan
        """
        # Setup mock
        mock_config_manager = MagicMock(spec=SimpleConfigManager)
        mock_base_dir = Path('/mock/base/dir')
        mock_config_dir = Path('/mock/config/dir')
        mock_setup_environment.return_value = (mock_config_manager, mock_base_dir, mock_config_dir)
        
        # Buat komponen UI
        ui_components = UIFactory.create_ui_components()
        
        # Extract callbacks
        ui_callbacks = {}
        
        # Buat mock untuk callback log_message
        log_messages = []
        def mock_log_message(message):
            log_messages.append(message)
        
        # Buat mock untuk callback update_status
        status_updates = []
        def mock_update_status(message, status_type="info"):
            status_updates.append((message, status_type))
        
        # Setup callbacks
        ui_callbacks['log_message'] = mock_log_message
        ui_callbacks['update_status'] = mock_update_status
        
        # Buat handler dengan UI callbacks
        handler = SetupHandler(ui_callbacks)
        
        # Panggil perform_setup
        config_manager, base_dir, config_dir = handler.perform_setup()
        
        # Verifikasi hasil
        self.assertEqual(config_manager, mock_config_manager)
        self.assertEqual(base_dir, mock_base_dir)
        self.assertEqual(config_dir, mock_config_dir)
        
        # Verifikasi interaksi dengan UI melalui callbacks
        self.assertTrue(any("Memulai setup environment" in msg for msg in log_messages), 
                       "Log message memulai setup tidak ditemukan")
        self.assertTrue(any("Setup environment berhasil" in msg for msg in log_messages), 
                       "Log message sukses tidak ditemukan")
        
        # Verifikasi status updates
        self.assertTrue(any("Setup environment sedang berjalan" in msg for msg, _ in status_updates), 
                       "Status update 'sedang berjalan' tidak ditemukan")
        self.assertTrue(any(status == "success" for msg, status in status_updates if "berhasil" in msg), 
                       "Status update success tidak ditemukan")

if __name__ == '__main__':
    unittest.main() 