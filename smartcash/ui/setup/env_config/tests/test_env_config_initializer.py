"""
File: smartcash/ui/setup/env_config/tests/test_env_config_initializer.py
Deskripsi: Test untuk inisialisasi environment config
"""

import unittest
import os
import tempfile
from unittest.mock import MagicMock, patch
from pathlib import Path
import ipywidgets as widgets

from smartcash.ui.setup.env_config.env_config_initializer import initialize_env_config_ui
from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
from smartcash.ui.setup.env_config.tests.test_helper import ignore_layout_warnings, MockColabEnvironment

class TestEnvConfigInitializer(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        self.mock_component = MagicMock(spec=EnvConfigComponent)
        self.mock_component.ui_components = {
            'setup_button': widgets.Button(description='Konfigurasi Environment'),
            'status_panel': widgets.Output(),
            'progress_bar': widgets.FloatProgress(),
            'log_output': widgets.Output()
        }
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.drive_dir = os.path.join(self.temp_dir, 'drive')
        os.makedirs(os.path.join(self.temp_dir, 'SmartCash', 'configs'), exist_ok=True)
        os.makedirs(os.path.join(self.drive_dir, 'SmartCash', 'configs'), exist_ok=True)
        
    def tearDown(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_initializer_import(self):
        """Test import initializer"""
        self.assertIsNotNone(initialize_env_config_ui)
        
    @ignore_layout_warnings
    def test_initializer_creation(self):
        """Test pembuatan initializer"""
        # Mock config managers
        mock_config_manager = MagicMock()
        mock_colab_manager = MagicMock()
        
        with MockColabEnvironment(), \
             patch('smartcash.ui.setup.env_config.components.manager_setup.ColabConfigManager', return_value=mock_colab_manager), \
             patch('smartcash.ui.setup.env_config.components.manager_setup.ConfigManager', return_value=mock_config_manager), \
             patch('smartcash.ui.setup.env_config.components.manager_setup.setup_managers', 
                  return_value=(mock_config_manager, mock_colab_manager, Path('/content'), Path('/content/configs'))):
            
            component = initialize_env_config_ui()
            self.assertIsInstance(component, EnvConfigComponent)
            self.assertIsInstance(component.ui_components, dict)
            self.assertIn('setup_button', component.ui_components)
            self.assertIn('status_panel', component.ui_components)
            self.assertIn('log_panel', component.ui_components)
            self.assertIn('progress_bar', component.ui_components)
            self.assertIn('ui_layout', component.ui_components)
            
            # Verify button properties
            setup_button = component.ui_components['setup_button']
            self.assertEqual(setup_button.description, "Konfigurasi Environment")
            self.assertEqual(setup_button.button_style, "primary")

if __name__ == '__main__':
    unittest.main()
