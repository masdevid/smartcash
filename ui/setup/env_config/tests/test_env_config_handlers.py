"""
File: smartcash/ui/setup/env_config/tests/test_env_config_handlers.py
Deskripsi: Test untuk handlers environment config
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from smartcash.ui.setup.env_config.handlers.auto_check_handler import AutoCheckHandler
from smartcash.ui.setup.env_config.handlers.setup_handlers import setup_env_config_handlers
from smartcash.ui.setup.env_config.handlers.state_manager import StateManager

class TestEnvConfigHandlers(unittest.TestCase):
    """
    Test untuk handlers environment config
    """
    
    def setUp(self):
        """
        Setup test
        """
        # Create mock component
        self.mock_component = MagicMock()
        self.mock_component.ui_components = {
            'progress_bar': MagicMock(),
            'log_output': MagicMock(),
            'status_panel': MagicMock()
        }
        self.mock_component.config_manager = MagicMock()
        self.mock_component.config_manager.base_dir = Path('/dummy/base/dir')
        self.mock_component.config_dir = Path('/dummy/config/dir')
        self.mock_component._update_status = MagicMock()
    
    def test_auto_check_handler(self):
        """
        Test AutoCheckHandler lifecycle
        """
        handler = AutoCheckHandler(self.mock_component)
        handler.auto_check()
        
        # Verify progress updates
        self.mock_component.ui_components['progress_bar'].value = 0.2
        self.mock_component.ui_components['progress_bar'].value = 1.0
        
        # Verify status update
        self.mock_component._update_status.assert_called_with("Environment check completed", "success")
    
    def test_setup_env_config_handlers(self):
        """
        Test setup_env_config_handlers
        """
        mock_colab_manager = MagicMock()
        mock_ui_components = {
            'drive_button': MagicMock(),
            'directory_button': MagicMock(),
            'status_panel': MagicMock()
        }
        
        setup_env_config_handlers(mock_ui_components, mock_colab_manager)
        
        # Verify button handlers are set
        mock_ui_components['drive_button'].on_click.assert_called_once()
        mock_ui_components['directory_button'].on_click.assert_called_once()
    
    def test_state_manager_initial_state(self):
        """
        Test StateManager initial state
        """
        state_manager = StateManager()
        
        # Verify initial state
        self.assertFalse(state_manager.is_drive_connected)
        self.assertFalse(state_manager.is_directory_setup)
        self.assertEqual(state_manager.progress, 0)
    
    def test_state_manager_drive_connection(self):
        """
        Test StateManager drive connection
        """
        state_manager = StateManager()
        
        # Test drive connection
        state_manager.set_drive_connected(True)
        self.assertTrue(state_manager.is_drive_connected)
        
        state_manager.set_drive_connected(False)
        self.assertFalse(state_manager.is_drive_connected)
    
    def test_state_manager_directory_setup(self):
        """
        Test StateManager directory setup
        """
        state_manager = StateManager()
        
        # Test directory setup
        state_manager.set_directory_setup(True)
        self.assertTrue(state_manager.is_directory_setup)
        
        state_manager.set_directory_setup(False)
        self.assertFalse(state_manager.is_directory_setup)
    
    def test_state_manager_progress_tracking(self):
        """
        Test StateManager progress tracking
        """
        state_manager = StateManager()
        
        # Test progress updates
        state_manager.update_progress(0.5, "Progress at 50%")
        self.assertEqual(state_manager.progress, 0.5)
        
        state_manager.update_progress(1.0, "Progress completed")
        self.assertEqual(state_manager.progress, 1.0)
    
    def test_state_manager_drive_sync(self):
        """
        Test StateManager drive sync
        """
        state_manager = StateManager()
        
        # Test drive sync
        state_manager.set_drive_synced(True)
        self.assertTrue(state_manager.is_drive_synced)
        
        state_manager.set_drive_synced(False)
        self.assertFalse(state_manager.is_drive_synced)

if __name__ == '__main__':
    unittest.main()
