"""
File: smartcash/ui/setup/env_config/tests/test_env_config_handlers.py
Deskripsi: Test untuk handlers environment config
"""

import unittest
import asyncio
from unittest.mock import MagicMock, patch
from smartcash.ui.setup.env_config.handlers.setup_handlers import setup_env_config_handlers
from smartcash.ui.setup.env_config.handlers.auto_check_handler import AutoCheckHandler
from smartcash.ui.setup.env_config.components.state_manager import EnvConfigStateManager

class TestEnvConfigHandlers(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        # Create a mock/dummy ui_components dict with required keys
        self.ui_components = {
            'drive_button': MagicMock(),
            'directory_button': MagicMock(),
            'status_panel': MagicMock(),
            'log': MagicMock(),  # for log context manager
            'progress': MagicMock(),  # for progress bar
            'progress_bar': MagicMock(),
            'progress_message': MagicMock(),
            'log_panel': MagicMock(),
        }
        self.colab_manager = MagicMock()
        # Create a mock component for AutoCheckHandler
        self.mock_component = MagicMock()
        self.mock_component.ui_components = self.ui_components
        self.mock_component.config_manager.base_dir = '/dummy/base/dir'
        self.mock_component.config_dir = '/dummy/config/dir'
        self.mock_component._update_status = MagicMock()
        
    def test_setup_env_config_handlers(self):
        """Test setup_env_config_handlers function"""
        with patch('smartcash.ui.setup.env_config.handlers.setup_handlers.setup_drive_handler') as mock_drive, \
             patch('smartcash.ui.setup.env_config.handlers.setup_handlers.setup_directory_handler') as mock_dir:
            
            setup_env_config_handlers(self.ui_components, self.colab_manager)
            
            mock_drive.assert_called_once_with(self.ui_components, self.colab_manager)
            mock_dir.assert_called_once_with(self.ui_components, self.colab_manager)
            
    async def _test_auto_check_handler_async(self):
        """Async test helper for auto_check_handler"""
        handler = AutoCheckHandler(self.mock_component)
        await handler.auto_check()
        
        # Verify progress was updated
        self.assertEqual(self.mock_component.ui_components['progress'].value, 1.0)
        # Verify status was updated
        self.mock_component._update_status.assert_called_once()
        
    def test_auto_check_handler(self):
        """Test AutoCheckHandler lifecycle"""
        asyncio.run(self._test_auto_check_handler_async())
        
    def test_state_manager_initial_state(self):
        """Test EnvConfigStateManager initial state"""
        state_manager = EnvConfigStateManager(self.ui_components, self.colab_manager)
        
        # Test initial state update
        state_manager._update_initial_state()
        
        # Verify UI components are in correct initial state
        self.assertFalse(self.ui_components['drive_button'].disabled)
        self.assertFalse(self.ui_components['directory_button'].disabled)
        self.assertEqual(self.ui_components['status_panel'].value, "")
        self.assertEqual(self.ui_components['log_panel'].value, "")
        
    def test_state_manager_drive_connection(self):
        """Test EnvConfigStateManager drive connection states"""
        state_manager = EnvConfigStateManager(self.ui_components, self.colab_manager)
        
        # Test connection start
        state_manager.handle_drive_connection_start()
        self.assertTrue(self.ui_components['drive_button'].disabled)
        self.assertTrue(self.ui_components['directory_button'].disabled)
        self.assertIn("Connecting to Google Drive", self.ui_components['status_panel'].value)
        
        # Test connection success
        state_manager.handle_drive_connection_success()
        self.assertTrue(self.ui_components['drive_button'].disabled)
        self.assertFalse(self.ui_components['directory_button'].disabled)
        self.assertIn("Connected to Google Drive", self.ui_components['status_panel'].value)
        
        # Test connection error
        error_msg = "Connection failed"
        state_manager.handle_drive_connection_error(error_msg)
        self.assertFalse(self.ui_components['drive_button'].disabled)
        self.assertFalse(self.ui_components['directory_button'].disabled)
        self.assertIn(error_msg, self.ui_components['status_panel'].value)
        
    def test_state_manager_directory_setup(self):
        """Test EnvConfigStateManager directory setup states"""
        state_manager = EnvConfigStateManager(self.ui_components, self.colab_manager)
        
        # Test setup start
        state_manager.handle_directory_setup_start()
        self.assertTrue(self.ui_components['directory_button'].disabled)
        self.assertIn("Setting up directories", self.ui_components['status_panel'].value)
        
        # Test setup success
        state_manager.handle_directory_setup_success()
        self.assertTrue(self.ui_components['directory_button'].disabled)
        self.assertIn("Directories setup complete", self.ui_components['status_panel'].value)
        
        # Test setup error
        error_msg = "Setup failed"
        state_manager.handle_directory_setup_error(error_msg)
        self.assertFalse(self.ui_components['directory_button'].disabled)
        self.assertIn(error_msg, self.ui_components['status_panel'].value)
        
    def test_state_manager_drive_sync(self):
        """Test EnvConfigStateManager drive sync states"""
        state_manager = EnvConfigStateManager(self.ui_components, self.colab_manager)
        
        # Test sync start
        state_manager.handle_drive_sync_start()
        self.assertTrue(self.ui_components['drive_button'].disabled)
        self.assertTrue(self.ui_components['directory_button'].disabled)
        self.assertIn("Syncing with Google Drive", self.ui_components['status_panel'].value)
        
        # Test sync success
        state_manager.handle_drive_sync_success()
        self.assertTrue(self.ui_components['drive_button'].disabled)
        self.assertFalse(self.ui_components['directory_button'].disabled)
        self.assertIn("Sync complete", self.ui_components['status_panel'].value)
        
        # Test sync error
        error_msg = "Sync failed"
        state_manager.handle_drive_sync_error(error_msg)
        self.assertFalse(self.ui_components['drive_button'].disabled)
        self.assertFalse(self.ui_components['directory_button'].disabled)
        self.assertIn(error_msg, self.ui_components['status_panel'].value)
        
    def test_state_manager_progress_tracking(self):
        """Test EnvConfigStateManager progress tracking"""
        state_manager = EnvConfigStateManager(self.ui_components, self.colab_manager)
        
        # Test progress update
        state_manager.update_progress(50, "Halfway done")
        self.assertEqual(self.ui_components['progress_bar'].value, 50)
        self.assertEqual(self.ui_components['progress_message'].value, "Halfway done")
        
        # Test progress completion
        state_manager.update_progress(100, "Complete")
        self.assertEqual(self.ui_components['progress_bar'].value, 100)
        self.assertEqual(self.ui_components['progress_message'].value, "Complete")
        
        # Test progress reset
        state_manager.reset_progress()
        self.assertEqual(self.ui_components['progress_bar'].value, 0)
        self.assertEqual(self.ui_components['progress_message'].value, "")

if __name__ == '__main__':
    unittest.main()
