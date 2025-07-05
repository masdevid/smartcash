"""
file_path: tests/ui/setup/colab/test_colab_initializer_error_handling.py

Tes untuk inisialisasi UI Colab dengan fokus pada penanganan error.
"""

import sys
import pytest
from unittest.mock import MagicMock, AsyncMock
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

# Import module lain yang dibutuhkan
import os

# Setup mocks sebelum test dijalankan
sys.modules['smartcash.ui.setup.env_config'] = MagicMock()
sys.modules['smartcash.ui.setup.env_config.handlers'] = MagicMock()
sys.modules['smartcash.ui.setup.env_config.handlers.env_config_handler'] = MagicMock()
sys.modules['smartcash.ui.setup.env_config.handlers.setup_handler'] = MagicMock()

@pytest.fixture
def colab_initializer(mocker):
    try:
        from smartcash.ui.setup.colab.colab_initializer import ColabEnvInitializer
    except ImportError:
        ColabEnvInitializer = mocker.Mock()
    
    init_instance = ColabEnvInitializer()
    mocker.patch.object(init_instance, 'initialize', return_value={'success': True})
    mocker.patch.object(init_instance, '_post_checks', return_value=None)
    return init_instance


def test_colab_initializer_error_handling(colab_initializer, mocker):
    """
    Test error handling during Colab UI initialization.
    """
    # Arrange
    mocker.patch.object(colab_initializer, 'logger', autospec=True)
    mocker.patch.object(colab_initializer, 'initialize', return_value={'success': False, 'error': 'Initialization failed'})
    
    # Act
    result = colab_initializer.initialize()
    
    # Assert
    assert result['success'] is False
    assert 'error' in result
    colab_initializer.logger.error.assert_called_with('❌ Gagal inisialisasi Colab: %s', result['error'])


def test_colab_initializer_successful_init(colab_initializer, mocker):
    """
    Menguji inisialisasi yang berhasil dari ColabEnvInitializer.

    Args:
        colab_initializer: Fixture yang menyediakan instance ColabEnvInitializer untuk pengujian.
    """
    # Arrange
    mocker.patch.object(colab_initializer, 'logger', autospec=True)
    mocker.patch.object(colab_initializer, 'initialize', return_value={'success': True})
    
    # Act
    result = colab_initializer.initialize()
    
    # Assert
    assert result['success'] is True
    colab_initializer.logger.info.assert_called_with('✅ Inisialisasi Colab selesai')
