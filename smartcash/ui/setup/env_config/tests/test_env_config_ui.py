"""
File: smartcash/ui/setup/env_config/tests/test_env_config_ui.py
Deskripsi: Test untuk UI environment config
"""

import unittest
from unittest.mock import MagicMock, patch
from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
from smartcash.ui.setup.env_config.components.ui_creator import create_env_config_ui
from pathlib import Path
from ipywidgets import VBox, Button
from smartcash.ui.setup.env_config.tests.test_helper import ignore_layout_warnings, MockColabEnvironment

class TestEnvConfigUI(unittest.TestCase):
    @ignore_layout_warnings
    def setUp(self):
        """Set up test environment"""
        # Mock config managers
        mock_config_manager = MagicMock()
        mock_colab_manager = MagicMock()
        
        # Create the component with mocked setup_managers
        with MockColabEnvironment(), \
             patch('smartcash.ui.setup.env_config.components.manager_setup.ColabConfigManager', return_value=mock_colab_manager), \
             patch('smartcash.ui.setup.env_config.components.manager_setup.ConfigManager', return_value=mock_config_manager), \
             patch('smartcash.ui.setup.env_config.components.manager_setup.setup_managers', 
                  return_value=(mock_config_manager, mock_colab_manager, Path('/content'), Path('/content/configs'))):
            
            self.env_config = EnvConfigComponent()

    @ignore_layout_warnings
    def test_ui_components_exist(self):
        """Test that all required UI components are created and exist"""
        # Check if all expected UI components exist
        self.assertIn('header', self.env_config.ui_components)
        self.assertIn('setup_button', self.env_config.ui_components)
        self.assertIn('status_panel', self.env_config.ui_components)
        self.assertIn('log_panel', self.env_config.ui_components)
        self.assertIn('progress_bar', self.env_config.ui_components)
        self.assertIn('progress_message', self.env_config.ui_components)
        self.assertIn('button_layout', self.env_config.ui_components)

    @ignore_layout_warnings
    def test_button_layout(self):
        """Test that setup button is properly laid out in a VBox"""
        button_layout = self.env_config.ui_components['button_layout']
        
        # Check if button_layout is a VBox
        self.assertIsInstance(button_layout, VBox)
        
        # Check if it contains the setup button
        self.assertIn(self.env_config.ui_components['setup_button'], button_layout.children)

    @ignore_layout_warnings
    def test_setup_button_properties(self):
        """Test that setup button has correct properties"""
        setup_button = self.env_config.ui_components['setup_button']
        
        # Check button properties
        self.assertIsInstance(setup_button, Button)
        self.assertEqual(setup_button.description, "Konfigurasi Environment")
        self.assertEqual(setup_button.button_style, "primary")
        self.assertEqual(setup_button.icon, "cog")

    @ignore_layout_warnings
    def test_create_env_config_ui(self):
        """Test create_env_config_ui function directly"""
        ui_components = create_env_config_ui()
        
        # Check if all expected UI components exist
        self.assertIn('header', ui_components)
        self.assertIn('setup_button', ui_components)
        self.assertIn('status_panel', ui_components)
        self.assertIn('log_panel', ui_components)
        self.assertIn('progress_bar', ui_components)
        self.assertIn('progress_message', ui_components)
        self.assertIn('button_layout', ui_components)
        
        # Check if setup button has correct properties
        setup_button = ui_components['setup_button']
        self.assertEqual(setup_button.description, "Konfigurasi Environment")
        self.assertEqual(setup_button.button_style, "primary")
        self.assertEqual(setup_button.icon, "cog")

if __name__ == '__main__':
    unittest.main() 