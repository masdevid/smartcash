import unittest
from unittest.mock import MagicMock
from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
from smartcash.ui.setup.env_config.components.state_manager import EnvConfigStateManager
from smartcash.ui.setup.env_config.handlers.setup_handlers import setup_env_config_handlers

class TestEnvConfigIntegration(unittest.TestCase):
    def setUp(self):
        # Setup mock UI components
        self.ui_components = {
            'drive_button': MagicMock(),
            'directory_button': MagicMock(),
            'status_panel': MagicMock(),
            'log_panel': MagicMock(),
            'progress_bar': MagicMock(),
            'progress_message': MagicMock(),
            'log': MagicMock(),
        }
        self.colab_manager = MagicMock()
        self.colab_manager.is_drive_connected.return_value = False
        self.state_manager = EnvConfigStateManager(self.ui_components, self.colab_manager)
        setup_env_config_handlers(self.ui_components, self.colab_manager)

    def test_drive_button_click(self):
        # Simulate drive button click handler
        self.state_manager.handle_drive_connection_start()
        self.assertTrue(self.ui_components['drive_button'].disabled)
        self.state_manager.handle_drive_connection_success()
        self.assertTrue(self.ui_components['drive_button'].disabled)
        self.assertFalse(self.ui_components['directory_button'].disabled)

    def test_directory_button_click(self):
        # Simulate directory button click handler
        self.state_manager.handle_directory_setup_start()
        self.assertTrue(self.ui_components['directory_button'].disabled)
        self.state_manager.handle_directory_setup_success()
        self.assertTrue(self.ui_components['directory_button'].disabled)

    def test_drive_sync(self):
        # Simulate drive sync handler
        self.state_manager.handle_drive_sync_start()
        self.assertTrue(self.ui_components['drive_button'].disabled)
        self.assertTrue(self.ui_components['directory_button'].disabled)
        self.state_manager.handle_drive_sync_success()
        self.assertTrue(self.ui_components['drive_button'].disabled)
        self.assertFalse(self.ui_components['directory_button'].disabled)

if __name__ == '__main__':
    unittest.main() 