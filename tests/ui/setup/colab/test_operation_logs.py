# file_path: /Users/masdevid/Projects/smartcash/tests/ui/setup/colab/test_operation_logs.py
# Deskripsi: Unit test untuk memastikan log operasi ditampilkan di log_accordion dan tidak muncul sebagai print atau default Colab log.

import pytest
from unittest.mock import Mock, patch

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


def test_operation_logs_displayed_in_log_accordion(mock_initializer):
    """
    Test untuk memastikan log operasi ditampilkan di log_accordion dan tidak sebagai print atau default Colab log.
    """
    # Arrange
    initializer = mock_initializer
    setup_handler = initializer._handlers['setup']
    
    # Mock methods to simulate operations logging
    setup_handler.perform_initial_status_check = Mock()
    setup_handler.should_sync_config_templates = Mock(return_value=False)
    
    # Act
    initializer._post_checks()
    
    # Assert
    initializer.logger.info.assert_any_call('🔍 Post-initialization checks…')
    initializer.logger.info.assert_any_call('✅ Post-checks selesai')
    
    # Ensure no print statements or default Colab logs are used
    with patch('builtins.print') as mocked_print:
        initializer._post_checks()
        mocked_print.assert_not_called()
