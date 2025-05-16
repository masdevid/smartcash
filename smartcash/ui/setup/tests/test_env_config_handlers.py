"""
File: smartcash/ui/setup/tests/test_env_config_handlers.py
Deskripsi: Test untuk handler UI konfigurasi environment
"""

import unittest
from unittest.mock import MagicMock, patch, call
import ipywidgets as widgets
from smartcash.ui.setup.tests.test_helper import WarningTestCase, ignore_layout_warnings

class TestEnvConfigHandlers(WarningTestCase):
    """Test case untuk env_config_handlers.py"""
    
    def setUp(self):
        """Setup untuk test"""
        # Buat mock UI components
        # Gunakan mock untuk layout untuk menghindari deprecation warning
        mock_layout = MagicMock()
        mock_layout.visibility = 'hidden'
        
        self.ui_components = {
            'drive_button': widgets.Button(),
            'directory_button': widgets.Button(),
            'progress_bar': MagicMock(
                value=0,
                min=0,
                max=10,
                layout=mock_layout
            ),
            'progress_message': MagicMock(
                value="",
                layout=mock_layout
            ),
            'status': widgets.Output(),
            'logger': MagicMock()
        }
    
    @patch('smartcash.ui.setup.environment_detector.detect_environment')
    @patch('smartcash.ui.utils.logging_utils.create_cleanup_function')
    def test_setup_env_config_handlers(self, mock_create_cleanup, mock_detect_env):
        """Test setup handler untuk UI konfigurasi environment"""
        from smartcash.ui.setup.env_config_handlers import setup_env_config_handlers
        
        # Setup mock
        mock_detect_env.return_value = self.ui_components
        mock_create_cleanup.return_value = lambda: None
        
        # Mock environment dan config
        mock_env = MagicMock()
        mock_config = {}
        
        # Patch fungsi internal
        with patch('smartcash.ui.setup.env_config_handlers._setup_drive_button_handler') as mock_setup_drive, \
             patch('smartcash.ui.setup.env_config_handlers._setup_directory_button_handler') as mock_setup_dir, \
             patch('smartcash.ui.setup.env_config_handlers._register_cleanup_event') as mock_register:
            
            # Panggil fungsi
            result = setup_env_config_handlers(self.ui_components, mock_env, mock_config)
            
            # Verifikasi hasil
            self.assertEqual(result, self.ui_components)
            mock_detect_env.assert_called_once_with(self.ui_components, mock_env)
            mock_setup_drive.assert_called_once_with(self.ui_components)
            mock_setup_dir.assert_called_once_with(self.ui_components)
            mock_create_cleanup.assert_called_once()
            mock_register.assert_called_once()
            self.ui_components['logger'].info.assert_called_once()
    
    @patch('smartcash.ui.utils.ui_logger.log_to_ui')
    @patch('smartcash.common.environment.get_environment_manager')
    @patch('smartcash.ui.setup.environment_detector.detect_environment')
    def test_drive_button_handler(self, mock_detect_env, mock_get_env, mock_log_to_ui):
        """Test handler untuk tombol connect drive"""
        from smartcash.ui.setup.env_config_handlers import _setup_drive_button_handler
        
        # Setup mock
        mock_env_manager = MagicMock()
        mock_env_manager.mount_drive.return_value = (True, "Drive berhasil terhubung")
        mock_get_env.return_value = mock_env_manager
        
        # Panggil fungsi setup
        _setup_drive_button_handler(self.ui_components)
        
        # Verifikasi handler terdaftar
        self.assertTrue(len(self.ui_components['drive_button']._click_handlers.callbacks) > 0)
        
        # Simulasi klik tombol
        self.ui_components['drive_button']._click_handlers(self.ui_components['drive_button'])
        
        # Verifikasi pemanggilan fungsi
        mock_log_to_ui.assert_has_calls([
            call(self.ui_components, "Menghubungkan ke Google Drive...", "info", "🔄"),
            call(self.ui_components, "Drive berhasil terhubung", "success", "✅")
        ])
        mock_get_env.assert_called_once()
        mock_env_manager.mount_drive.assert_called_once()
        mock_detect_env.assert_called_once()
        
        # Verifikasi tombol disembunyikan
        self.assertEqual(self.ui_components['drive_button'].layout.display, 'none')
    
    @patch('smartcash.ui.utils.ui_logger.log_to_ui')
    @patch('smartcash.common.environment.get_environment_manager')
    def test_directory_button_handler(self, mock_get_env, mock_log_to_ui):
        """Test handler untuk tombol setup direktori"""
        from smartcash.ui.setup.env_config_handlers import _setup_directory_button_handler
        
        # Setup mock
        mock_env_manager = MagicMock()
        mock_env_manager.is_drive_mounted = False
        mock_env_manager.is_colab = False
        mock_env_manager.setup_project_structure.return_value = {'created': 5, 'existing': 2}
        mock_get_env.return_value = mock_env_manager
        
        # Panggil fungsi setup
        _setup_directory_button_handler(self.ui_components)
        
        # Verifikasi handler terdaftar
        self.assertTrue(len(self.ui_components['directory_button']._click_handlers.callbacks) > 0)
        
        # Simulasi klik tombol
        self.ui_components['directory_button']._click_handlers(self.ui_components['directory_button'])
        
        # Verifikasi pemanggilan fungsi
        mock_get_env.assert_called_once()
        mock_env_manager.setup_project_structure.assert_called_once()
        mock_log_to_ui.assert_called_with(
            self.ui_components,
            "Berhasil membuat struktur direktori: 5 direktori baru, 2 sudah ada",
            "success",
            "✅"
        )
    
    def test_register_cleanup_event(self):
        """Test register cleanup function ke IPython event"""
        from smartcash.ui.setup.env_config_handlers import _register_cleanup_event
        
        # Mock IPython
        mock_ipython = MagicMock()
        mock_ipython.events._events = {'pre_run_cell': []}
        
        with patch('IPython.get_ipython', return_value=mock_ipython):
            # Mock cleanup function
            mock_cleanup = MagicMock()
            mock_cleanup.__qualname__ = 'cleanup'
            
            # Panggil fungsi
            result = _register_cleanup_event(mock_cleanup)
            
            # Verifikasi hasil
            self.assertTrue(result)
            mock_ipython.events.register.assert_called_once_with('pre_run_cell', mock_cleanup)

if __name__ == '__main__':
    unittest.main()
