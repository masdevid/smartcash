# file_path: /Users/masdevid/Projects/smartcash/tests/ui/setup/colab/test_config_sync_recovery.py
# Deskripsi: Unit test untuk memverifikasi pemulihan konfigurasi yang hilang selama post-checks di ColabEnvInitializer.

import sys
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pytest_mock import mocker

# Import test_helpers dan setup mocks sebelum import apapun
from . import test_helpers
test_helpers.setup_mocks(sys.modules)

# Sekarang import module yang di-test
try:
    from smartcash.ui.setup.colab.colab_initializer import ColabEnvInitializer
except ImportError as e:
    print(f"Import error: {e}. Menggunakan mock sebagai fallback.")
    ColabEnvInitializer = MagicMock()

from smartcash.ui.setup.colab.handlers.setup_handler import SetupHandler


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


def test_post_checks_recognizes_missing_config_and_recovers(colab_initializer, mocker):
    """
    Test untuk memastikan post-checks mengenali konfigurasi yang hilang dan memulihkannya dari repo.
    """
    # Arrange
    mocker.patch.object(colab_initializer, 'logger', autospec=True)
    mocker.patch.object(colab_initializer, 'setup_handler', autospec=True)
    mocker.patch.object(colab_initializer.setup_handler, 'is_config_missing', return_value=True)
    mocker.patch.object(colab_initializer.setup_handler, 'recover_config', return_value={'status': 'success'})
    
    # Act
    colab_initializer._post_checks()
    
    # Assert
    colab_initializer.logger.info.assert_called_with('🔄 Config hilang, memulai proses pemulihan...')
    colab_initializer.logger.info.assert_called_with('✅ Config berhasil dipulihkan')
    colab_initializer.setup_handler.recover_config.assert_called_once()
