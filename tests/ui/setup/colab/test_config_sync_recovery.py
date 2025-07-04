# file_path: /Users/masdevid/Projects/smartcash/tests/ui/setup/colab/test_config_sync_recovery.py
# Deskripsi: Unit test untuk memverifikasi pemulihan konfigurasi yang hilang selama post-checks di ColabEnvInitializer.

import pytest
from unittest.mock import Mock
import os

from smartcash.ui.setup.colab.colab_initializer import ColabEnvInitializer
from smartcash.ui.setup.colab.handlers.setup_handler import SetupHandler


@pytest.fixture
def mock_initializer():
    initializer = ColabEnvInitializer()
    initializer.logger = Mock()
    initializer._handlers = {'setup': Mock(spec=SetupHandler)}
    initializer._ui_components = Mock()
    initializer.__dict__['_initialized'] = True
    yield initializer


def test_post_checks_recognizes_missing_config_and_recovers(mock_initializer):
    """
    Test untuk memastikan post-checks mengenali konfigurasi yang hilang dan memulihkannya dari repo.
    """
    # Arrange
    initializer = mock_initializer
    setup_handler = initializer._handlers['setup']
    
    # Mock methods to simulate missing config and recovery
    setup_handler.perform_initial_status_check = Mock()
    setup_handler.should_sync_config_templates = Mock(return_value=True)
    setup_handler.sync_config_templates = Mock()
    
    # Act
    initializer._post_checks()
    
    # Assert
    initializer.logger.info.assert_any_call('🔍 Post-initialization checks…')
    setup_handler.perform_initial_status_check.assert_called_with(initializer._ui_components)
    setup_handler.should_sync_config_templates.assert_called_once()
    initializer.logger.info.assert_any_call('📋 Drive mounted dan setup complete, syncing templates…')
    setup_handler.sync_config_templates.assert_called_with(
        force_overwrite=False,
        update_ui=True,
        ui_components=initializer._ui_components
    )
    initializer.logger.info.assert_any_call('✅ Post-checks selesai')
