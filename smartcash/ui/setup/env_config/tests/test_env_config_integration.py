"""
File: smartcash/ui/setup/env_config/tests/test_env_config_integration.py
Deskripsi: Test integrasi untuk environment config
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import ipywidgets as widgets

from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
from smartcash.ui.setup.env_config.tests.test_helper import ignore_layout_warnings, MockColabEnvironment

class TestEnvConfigIntegration(unittest.TestCase):
    @ignore_layout_warnings
    def setUp(self):
        """Set up test environment"""
        # Mock config managers
        self.mock_config_manager = MagicMock()
        self.mock_colab_manager = MagicMock()
        
        # Create the component with mocked handlers and managers
        with MockColabEnvironment(), \
             patch('smartcash.ui.setup.env_config.components.manager_setup.ColabConfigManager', return_value=self.mock_colab_manager), \
             patch('smartcash.ui.setup.env_config.components.manager_setup.ConfigManager', return_value=self.mock_config_manager), \
             patch('smartcash.ui.setup.env_config.components.manager_setup.setup_managers', 
                  return_value=(self.mock_config_manager, self.mock_colab_manager, Path('/content'), Path('/content/configs'))):
            
            # Mock AutoCheckHandler before creating EnvConfigComponent
            with patch('smartcash.ui.setup.env_config.handlers.auto_check_handler.AutoCheckHandler') as self.mock_auto_check_class:
                self.mock_auto_check_instance = MagicMock()
                self.mock_auto_check_class.return_value = self.mock_auto_check_instance
                
                self.env_config = EnvConfigComponent()
                
                # Mock internal methods
                self.env_config._connect_drive = MagicMock(return_value=True)
                self.env_config._setup_directories = MagicMock(return_value=True)
                self.env_config._setup_config_files = MagicMock(return_value=True)
                self.env_config._initialize_singletons = MagicMock(return_value=True)
                self.env_config._update_status = MagicMock()
                self.env_config._update_progress = MagicMock()
                self.env_config._log_message = MagicMock()

    @ignore_layout_warnings
    def test_setup_button_click(self):
        """Test klik tombol setup"""
        # Simulate setup button click
        self.mock_colab_manager.is_drive_connected.return_value = False
        mock_button = MagicMock()
        
        # Execute setup click handler
        self.env_config._handle_setup_click(mock_button)
        
        # Verify button is disabled during setup
        self.assertTrue(mock_button.disabled)
        
        # Verify method calls
        self.env_config._connect_drive.assert_called_once()
        self.env_config._setup_directories.assert_called_once()
        self.env_config._setup_config_files.assert_called_once()
        self.env_config._initialize_singletons.assert_called_once()
        
        # Verify status update
        self.env_config._update_status.assert_called_with("Environment berhasil dikonfigurasi", "success")

    @ignore_layout_warnings
    def test_setup_button_click_drive_already_connected(self):
        """Test klik tombol setup dengan drive sudah terhubung"""
        # Simulate setup button click with drive already connected
        self.mock_colab_manager.is_drive_connected.return_value = True
        mock_button = MagicMock()
        
        # Execute setup click handler
        self.env_config._handle_setup_click(mock_button)
        
        # Verify connect_drive not called
        self.env_config._connect_drive.assert_not_called()
        
        # Verify other methods called
        self.env_config._setup_directories.assert_called_once()
        self.env_config._setup_config_files.assert_called_once()
        self.env_config._initialize_singletons.assert_called_once()

    @ignore_layout_warnings
    def test_display_with_existing_environment(self):
        """Test display dengan environment yang sudah ada"""
        # Mock _check_required_dirs to return True
        self.env_config._check_required_dirs = MagicMock(return_value=True)
        
        # Reset mock untuk _update_status
        self.env_config._update_status.reset_mock()
        
        # Call display with patched display
        with patch('IPython.display.display') as mock_display:
            # Patch auto_check method
            with patch.object(self.env_config.auto_check, 'auto_check') as mock_auto_check_method:
                self.env_config.display()
                
                # Verify button is disabled
                self.assertTrue(self.env_config.ui_components['setup_button'].disabled)
                
                # Verify status update - updated to match actual behavior
                self.env_config._update_status.assert_any_call("Environment sudah terkonfigurasi", "success")
                
                # Verify singletons initialized
                self.env_config._initialize_singletons.assert_called_once()
                
                # Verify auto check called
                mock_auto_check_method.assert_called_once()

    @ignore_layout_warnings
    def test_display_with_new_environment(self):
        """Test display dengan environment baru"""
        # Mock _check_required_dirs to return False
        self.env_config._check_required_dirs = MagicMock(return_value=False)
        
        # Reset mock untuk _update_status
        self.env_config._update_status.reset_mock()
        
        # Call display with patched display
        with patch('IPython.display.display') as mock_display:
            # Patch auto_check method
            with patch.object(self.env_config.auto_check, 'auto_check') as mock_auto_check_method:
                self.env_config.display()
                
                # Verify button is enabled
                self.assertFalse(self.env_config.ui_components['setup_button'].disabled)
                
                # Verify status update - updated to match actual behavior
                self.env_config._update_status.assert_any_call("Environment perlu dikonfigurasi", "info")
                
                # Verify auto check called
                mock_auto_check_method.assert_called_once()

if __name__ == '__main__':
    unittest.main() 