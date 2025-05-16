"""
File: smartcash/ui/setup/tests/test_env_config_component.py
Deskripsi: Test untuk komponen UI konfigurasi environment
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from smartcash.ui.setup.tests.test_helper import WarningTestCase, ignore_layout_warnings

class TestEnvConfigComponent(WarningTestCase):
    """Test case untuk env_config_component.py"""
    
    @ignore_layout_warnings
    @patch('smartcash.ui.utils.header_utils.create_header')
    @patch('smartcash.ui.utils.alert_utils.create_info_alert')
    @patch('smartcash.ui.info_boxes.get_environment_info')
    def test_create_env_config_ui(self, mock_get_env_info, mock_create_alert, mock_create_header):
        """Test pembuatan komponen UI konfigurasi environment"""
        from smartcash.ui.setup.env_config_component import create_env_config_ui
        
        # Setup mock
        mock_create_header.return_value = widgets.HTML(value="Mock Header")
        mock_create_alert.return_value = widgets.HTML(value="Mock Alert")
        mock_get_env_info.return_value = widgets.VBox()
        
        # Mock environment dan config
        mock_env = MagicMock()
        mock_config = {}
        
        # Panggil fungsi
        ui_components = create_env_config_ui(mock_env, mock_config)
        
        # Verifikasi hasil
        self.assertIsInstance(ui_components, dict)
        self.assertIn('ui', ui_components)
        self.assertIn('drive_button', ui_components)
        self.assertIn('directory_button', ui_components)
        self.assertIn('progress_bar', ui_components)
        self.assertIn('status', ui_components)
        self.assertIn('reset_progress', ui_components)
        self.assertEqual(ui_components['module_name'], 'env_config')
        
        # Verifikasi fungsi reset_progress
        progress_bar = ui_components['progress_bar']
        progress_message = ui_components['progress_message']
        
        # Simpan nilai awal
        initial_visibility_bar = progress_bar.layout.visibility
        initial_visibility_msg = progress_message.layout.visibility
        
        # Set nilai untuk diuji
        progress_bar.layout.visibility = 'visible'
        progress_message.layout.visibility = 'visible'
        progress_bar.value = 5
        progress_message.value = "Test"
        
        # Panggil reset_progress
        ui_components['reset_progress']()
        
        # Verifikasi reset berhasil
        self.assertEqual(progress_bar.layout.visibility, 'hidden')
        self.assertEqual(progress_message.layout.visibility, 'hidden')
        self.assertEqual(progress_bar.value, 0)
        self.assertEqual(progress_message.value, "")

if __name__ == '__main__':
    unittest.main()
