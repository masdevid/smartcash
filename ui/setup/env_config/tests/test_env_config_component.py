"""
File: smartcash/ui/setup/env_config/tests/test_env_config_component.py
Deskripsi: Test untuk komponen environment config
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import ipywidgets as widgets

from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
from smartcash.ui.setup.env_config.components.ui_creator import create_env_config_ui

class TestEnvConfigComponent(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        self.mock_component = MagicMock(spec=EnvConfigComponent)
        self.mock_component.ui_components = {
            'directory_button': widgets.Button(description='Select Directory'),
            'drive_button': widgets.Button(description='Connect Drive'),
            'status_output': widgets.Output(),
            'progress_bar': widgets.FloatProgress()
        }

    def test_create_env_config_ui_basic(self):
        """Test pembuatan UI dasar"""
        ui_components = create_env_config_ui()
        self.assertIsInstance(ui_components, dict)
        self.assertIn('directory_button', ui_components)
        self.assertIn('drive_button', ui_components)
        self.assertIn('status_panel', ui_components)
        self.assertIn('log_panel', ui_components)
        self.assertIn('progress_bar', ui_components)
        self.assertIn('ui_layout', ui_components)

    def test_create_env_config_ui_import(self):
        """Test import dan inisialisasi UI"""
        mock_colab_manager = MagicMock()
        mock_colab_manager._local_base_path = Path('/tmp/SmartCash/configs')
        mock_colab_manager._drive_base_path = Path('/tmp/drive/SmartCash/configs')
        with patch('smartcash.ui.setup.env_config.components.manager_setup.ColabConfigManager', return_value=mock_colab_manager), \
             patch('smartcash.ui.setup.env_config.components.manager_setup.setup_managers', return_value=(MagicMock(), mock_colab_manager, Path('/tmp/SmartCash'), Path('/tmp/SmartCash/configs'))), \
             patch('asyncio.create_task', return_value=None):
            component = EnvConfigComponent()
            self.assertIsInstance(component, EnvConfigComponent)
            self.assertIsInstance(component.ui_components, dict)
            self.assertIn('directory_button', component.ui_components)
            self.assertIn('drive_button', component.ui_components)
            self.assertIn('status_panel', component.ui_components)
            self.assertIn('log_panel', component.ui_components)
            self.assertIn('progress_bar', component.ui_components)
            self.assertIn('ui_layout', component.ui_components)

if __name__ == '__main__':
    unittest.main()
