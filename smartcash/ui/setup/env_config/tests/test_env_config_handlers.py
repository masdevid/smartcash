"""
File: smartcash/ui/setup/env_config/tests/test_env_config_handlers.py
Deskripsi: Test untuk handler UI konfigurasi environment
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from smartcash.ui.setup.env_config.tests.test_helper import WarningTestCase, ignore_layout_warnings

class TestEnvConfigHandlers(WarningTestCase):
    """Test case untuk env_config_handlers.py"""
    
    def setUp(self):
        """Setup untuk test"""
        # Buat mock UI components
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
    
    def test_setup_handlers_import(self):
        """Test import setup_handlers berhasil"""
        from smartcash.ui.setup.env_config.handlers.setup_handlers import setup_env_config_handlers
        self.assertTrue(callable(setup_env_config_handlers))
    
    def test_drive_button_handler_import(self):
        """Test import drive_button_handler berhasil"""
        from smartcash.ui.setup.env_config.handlers.drive_button_handler import setup_drive_button_handler
        self.assertTrue(callable(setup_drive_button_handler))
    
    def test_directory_button_handler_import(self):
        """Test import directory_button_handler berhasil"""
        from smartcash.ui.setup.env_config.handlers.directory_button_handler import setup_directory_button_handler
        self.assertTrue(callable(setup_directory_button_handler))
    
    def test_register_cleanup_event(self):
        """Test register cleanup function ke IPython event"""
        from smartcash.ui.setup.env_config.handlers.setup_handlers import _register_cleanup_event
        
        # Panggil fungsi dengan dummy cleanup function
        result = _register_cleanup_event(lambda: None)
        
        # Verifikasi hasil
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
