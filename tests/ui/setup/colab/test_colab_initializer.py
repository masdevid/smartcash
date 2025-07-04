import pytest
from unittest.mock import Mock, patch

from smartcash.ui.setup.colab.colab_initializer import ColabEnvInitializer
from smartcash.ui.setup.colab.handlers.setup_handler import SetupHandler


@pytest.fixture
def mock_colab_initializer():
    initializer = ColabEnvInitializer()
    initializer.logger = Mock()
    initializer._env_manager = Mock()
    initializer._ui_components = {"test": "component"}
    initializer._handlers = {"setup": Mock(spec=SetupHandler)}
    initializer.__dict__['_initialized'] = False
    return initializer


class TestColabEnvInitializer:
    def test_post_checks_with_setup_handler(self, mock_colab_initializer):
        # Arrange
        initializer = mock_colab_initializer
        setup_handler = initializer._handlers["setup"]
        setup_handler.perform_initial_status_check = Mock()
        setup_handler.should_sync_config_templates = Mock(return_value=True)
        setup_handler.sync_config_templates = Mock()
        
        # Act
        initializer._post_checks()
        
        # Assert
        initializer.logger.info.assert_any_call("🔍 Post-initialization checks…")
        setup_handler.perform_initial_status_check.assert_called_with(initializer._ui_components)
        setup_handler.should_sync_config_templates.assert_called_once()
        initializer.logger.info.assert_any_call("📋 Drive mounted dan setup complete, syncing templates…")
        setup_handler.sync_config_templates.assert_called_with(
            force_overwrite=False,
            update_ui=True,
            ui_components=initializer._ui_components
        )
        initializer.logger.info.assert_any_call("✅ Post-checks selesai")

    def test_post_checks_without_setup_handler(self, mock_colab_initializer):
        # Arrange
        initializer = mock_colab_initializer
        initializer._handlers = {}
        
        # Act
        initializer._post_checks()
        
        # Assert
        initializer.logger.info.assert_any_call("🔍 Post-initialization checks…")
        initializer.logger.warning.assert_any_call("⚠️ Setup handler tidak tersedia")

    def test_post_checks_with_exception(self, mock_colab_initializer):
        # Arrange
        initializer = mock_colab_initializer
        setup_handler = initializer._handlers["setup"]
        exception = Exception("Test error")
        setup_handler.perform_initial_status_check = Mock(side_effect=exception)
        
        # Act
        initializer._post_checks()
        
        # Assert
        initializer.logger.info.assert_any_call("🔍 Post-initialization checks…")
        initializer.logger.error.assert_any_call("❌ Error post-checks: %s", exception, exc_info=True)
