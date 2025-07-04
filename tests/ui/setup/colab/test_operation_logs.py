# file_path: /Users/masdevid/Projects/smartcash/tests/ui/setup/colab/test_operation_logs.py
# Deskripsi: Unit test untuk memastikan log operasi ditampilkan di log_accordion dan tidak muncul sebagai print atau default Colab log.

import sys
import pytest
from unittest.mock import MagicMock, patch, Mock
import os

# Import test_helpers dan setup mocks sebelum import apapun
from . import test_helpers
test_helpers.setup_mocks(sys.modules)

# Sekarang import module yang di-test
try:
    from smartcash.ui.setup.colab.colab_initializer import ColabEnvInitializer
    from smartcash.ui.setup.colab.handlers.setup_handler import SetupHandler
except ImportError as e:
    print(f"Import error: {e}. Menggunakan mock sebagai fallback.")
    ColabEnvInitializer = MagicMock()
    SetupHandler = MagicMock()

# Setup mocks sebelum test dijalankan
sys.modules['smartcash.ui.setup.env_config'] = MagicMock()
sys.modules['smartcash.ui.setup.env_config.handlers'] = MagicMock()
sys.modules['smartcash.ui.setup.env_config.handlers.env_config_handler'] = MagicMock()
sys.modules['smartcash.ui.setup.env_config.handlers.setup_handler'] = MagicMock()

# Import test helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def colab_initializer(mocker):
    try:
        from smartcash.ui.setup.colab.colab_initializer import ColabEnvInitializer
    except ImportError:
        ColabEnvInitializer = mock.Mock()
    
    init_instance = ColabEnvInitializer()
    mocker.patch.object(init_instance, 'initialize', return_value={'success': True})
    mocker.patch.object(init_instance, '_post_checks', return_value=None)
    return init_instance


def test_operation_logs_displayed_in_log_accordion(colab_initializer, mocker):
    """
    Test untuk memastikan log operasi ditampilkan di log_accordion dan tidak sebagai print atau default Colab log.
    """
    # Arrange
    mocker.patch.object(colab_initializer, 'logger', autospec=True)
    mocker.patch.object(colab_initializer, 'operation_handler', autospec=True)
    mocker.patch.object(colab_initializer, '_ui_components', autospec=True)
    mocker.patch.object(colab_initializer._ui_components, 'log_accordion', autospec=True)
    operation_name = 'Test Operation'
    operation_log = '🛠️ Log operasi tes'
    mocker.patch.object(colab_initializer.operation_handler, 'run_operation', return_value={'status': 'success', 'logs': [operation_log]})
    
    # Act
    result = colab_initializer.operation_handler.run_operation(operation_name)
    
    # Assert
    assert result['status'] == 'success'
    colab_initializer._ui_components.log_accordion.log.assert_called_with(operation_log)
    colab_initializer.logger.info.assert_called_with(f'✅ Operasi berhasil: {operation_name}')
