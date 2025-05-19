"""
File: smartcash/ui/setup/env_config/tests/test_env_config_component.py
Deskripsi: Test untuk komponen UI konfigurasi environment
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from smartcash.ui.setup.env_config.tests.test_helper import WarningTestCase, ignore_layout_warnings

class TestEnvConfigComponent(WarningTestCase):
    """Test case untuk env_config_component.py"""
    
    def test_create_env_config_ui_import(self):
        """Test import create_env_config_ui berhasil"""
        from smartcash.ui.setup.env_config.components.env_config_component import create_env_config_ui
        self.assertTrue(callable(create_env_config_ui))
    
    @ignore_layout_warnings
    def test_create_env_config_ui_basic(self):
        """Test pembuatan komponen UI konfigurasi environment (dasar)"""
        from smartcash.ui.setup.env_config.components.env_config_component import create_env_config_ui
        
        # Mock environment dan config
        mock_env = MagicMock()
        mock_config = {}
        
        # Panggil fungsi dengan patch minimal
        with patch('smartcash.ui.utils.header_utils.create_header', return_value=widgets.HTML(value="Mock Header")), \
             patch('smartcash.ui.utils.alert_utils.create_info_box', return_value=widgets.HTML(value="Mock Alert")), \
             patch('smartcash.ui.info_boxes.get_environment_info', return_value=widgets.VBox()):
            
            # Panggil fungsi
            ui_components = create_env_config_ui(mock_env, mock_config)
            
            # Verifikasi hasil dasar
            self.assertIsInstance(ui_components, dict)
            self.assertIn('ui', ui_components)
            self.assertIn('module_name', ui_components)
            self.assertEqual(ui_components['module_name'], 'env_config')

if __name__ == '__main__':
    unittest.main()
