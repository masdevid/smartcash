import unittest
from unittest.mock import MagicMock, patch
from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
from pathlib import Path

class TestEnvConfigUI(unittest.TestCase):
    @patch('asyncio.create_task')
    @patch('smartcash.common.config.colab_manager.ColabConfigManager._setup_colab_environment')
    @patch('smartcash.common.config.colab_manager.ColabConfigManager._get_config_files')
    @patch('smartcash.common.config.colab_manager.ColabConfigManager._initialize_first_time_config')
    def setUp(self, mock_init_config, mock_get_config_files, mock_setup_colab_environment, mock_create_task):
        # Mock the config files
        mock_get_config_files.return_value = ['config1.json', 'config2.json']
        
        # Mock the base path
        mock_base_path = MagicMock()
        mock_base_path.__truediv__.return_value = Path('/mock/path')
        
        # Patch asyncio.create_task to do nothing
        mock_create_task.return_value = None
        
        # Create the component
        self.env_config = EnvConfigComponent()
        
        # Store mocks for later use
        self.mock_init_config = mock_init_config
        self.mock_get_config_files = mock_get_config_files
        self.mock_setup_colab_environment = mock_setup_colab_environment
        self.mock_create_task = mock_create_task

    def test_ui_components_exist(self):
        """Test that all required UI components are created and exist"""
        # Check if all expected UI components exist
        self.assertIn('header', self.env_config.ui_components)
        self.assertIn('drive_button', self.env_config.ui_components)
        self.assertIn('directory_button', self.env_config.ui_components)
        self.assertIn('status_panel', self.env_config.ui_components)
        self.assertIn('log_panel', self.env_config.ui_components)
        self.assertIn('progress_bar', self.env_config.ui_components)
        self.assertIn('progress_message', self.env_config.ui_components)
        self.assertIn('button_layout', self.env_config.ui_components)

    def test_button_layout(self):
        """Test that buttons are properly laid out in an HBox"""
        from ipywidgets import HBox
        button_layout = self.env_config.ui_components['button_layout']
        
        # Check if button_layout is an HBox
        self.assertIsInstance(button_layout, HBox)
        
        # Check if it contains both buttons
        self.assertIn(self.env_config.ui_components['drive_button'], button_layout.children)
        self.assertIn(self.env_config.ui_components['directory_button'], button_layout.children)

if __name__ == '__main__':
    unittest.main() 