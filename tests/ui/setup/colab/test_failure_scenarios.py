"""
file_path: /Users/masdevid/Projects/smartcash/tests/ui/setup/colab/test_failure_scenarios.py

File ini berisi pengujian untuk skenario kegagalan dan non-happy path
dalam proses setup dan operasi Colab environment.
"""

import sys
import os
import asyncio
import pytest
from unittest.mock import MagicMock, patch, Mock

# Setup mocks sebelum test dijalankan
def setup_mocks(modules):
    # Pastikan semua module yang mungkin bermasalah sudah dimock sebelum import
    modules['smartcash.ui.setup.env_config'] = MagicMock()
    modules['smartcash.ui.setup.env_config.handlers'] = MagicMock()
    modules['smartcash.ui.setup.env_config.handlers.env_config_handler'] = MagicMock()
    modules['smartcash.ui.setup.env_config.handlers.setup_handler'] = MagicMock()

# Import test helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from . import test_helpers

# Setup mocks
test_helpers.setup_mocks(sys.modules)

# Sekarang import module yang di-test
try:
    from smartcash.ui.setup.colab.colab_initializer import ColabEnvInitializer
except ImportError as e:
    print(f"Import error: {e}. Menggunakan mock sebagai fallback.")
    ColabEnvInitializer = MagicMock()

# Import module yang akan diuji setelah mocks disetup
try:
    from smartcash.ui.setup.colab.handlers.setup_handler import SetupHandler, SetupPhase
except ImportError as e:
    print(f"Import error: {e}")
    # Jika masih ada error, mock module yang bermasalah
    sys.modules['smartcash.ui.setup.colab.handlers'] = MagicMock()
    sys.modules['smartcash.ui.setup.colab.handlers.setup_handler'] = MagicMock()
    sys.modules['smartcash.ui.setup.env_config.handlers'] = MagicMock()
    sys.modules['smartcash.ui.setup.env_config.handlers.setup_handler'] = MagicMock()
    sys.modules['smartcash.ui.setup.env_config.handlers.env_config_handler'] = MagicMock()
    SetupHandler = MagicMock()
    SetupPhase = MagicMock()

# Mock fixture for SetupHandler
@pytest.fixture
async def mock_setup_handler():
    handler = SetupHandler()
    handler._setup_in_progress = False
    handler._ui_components = {'progress_bar': MockWidget(), 'status_label': MockWidget(), 'summary_container': MockWidget()}
    return handler


class MockWidget:
    def __init__(self):
        self.value = None


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


class TestSetupFailureScenarios:
    """Kelas untuk menguji skenario kegagalan setup."""

    def test_drive_not_mounted(self, colab_initializer, mocker):
        """
        Test untuk skenario ketika drive tidak ter-mount.
        """
        # Arrange
        mocker.patch.object(colab_initializer, 'logger', autospec=True)
        mocker.patch.object(colab_initializer, 'setup_handler', autospec=True)
        mocker.patch.object(colab_initializer.setup_handler, 'is_drive_mounted', return_value=False)
        
        # Act
        result = colab_initializer.initialize()
        
        # Assert
        assert result['success'] is False
        colab_initializer.logger.error.assert_called_with('❌ Google Drive belum di-mount. Mount Drive terlebih dahulu.')

    def test_mounting_success_then_unmounted(self, colab_initializer, mocker):
        """
        Test untuk skenario ketika mounting berhasil tetapi kemudian di-unmount.
        """
        # Arrange
        mocker.patch.object(colab_initializer, 'logger', autospec=True)
        mocker.patch.object(colab_initializer, 'setup_handler', autospec=True)
        mocker.patch.object(colab_initializer.setup_handler, 'is_drive_mounted', return_value=True)
        mocker.patch.object(colab_initializer.setup_handler, '_stage_drive_mount', return_value={'status': 'success'})
        mocker.patch.object(colab_initializer.setup_handler, 'is_drive_mounted', side_effect=[True, False])
        
        # Act
        result = colab_initializer.initialize()
        
        # Assert
        assert result['success'] is False
        colab_initializer.logger.error.assert_called_with('❌ Google Drive terdeteksi tidak di-mount setelah pemeriksaan awal.')

    def test_mounted_but_no_write_access(self, colab_initializer, mocker):
        """
        Test untuk skenario ketika drive mounted tetapi tidak ada akses tulis.
        """
        # Arrange
        mocker.patch.object(colab_initializer, 'logger', autospec=True)
        mocker.patch.object(colab_initializer, 'setup_handler', autospec=True)
        mocker.patch.object(colab_initializer.setup_handler, 'is_drive_mounted', return_value=True)
        mocker.patch.object(colab_initializer.setup_handler, 'has_write_access', return_value=False)
        
        # Act
        result = colab_initializer.initialize()
        
        # Assert
        assert result['success'] is False
        colab_initializer.logger.error.assert_called_with('❌ Tidak memiliki akses tulis ke Google Drive. Periksa izin Anda.')

    def test_permission_denied_during_folder_setup(self, colab_initializer, mocker):
        """
        Test untuk skenario ketika izin ditolak saat setup folder.
        """
        # Arrange
        mocker.patch.object(colab_initializer, 'logger', autospec=True)
        mocker.patch.object(colab_initializer, 'setup_handler', autospec=True)
        mocker.patch.object(colab_initializer.setup_handler, 'is_drive_mounted', return_value=True)
        mocker.patch.object(colab_initializer.setup_handler, 'has_write_access', return_value=True)
        mocker.patch.object(colab_initializer.setup_handler, 'setup_folders', return_value={'status': 'error', 'error': 'Permission denied'})
        
        # Act
        result = colab_initializer.initialize()
        
        # Assert
        assert result['success'] is False
        colab_initializer.logger.error.assert_called_with('❌ Gagal membuat folder: Permission denied')

    def test_symlink_creation_failure(self, colab_initializer, mocker):
        """
        Test untuk skenario ketika pembuatan symlink gagal.
        """
        # Arrange
        mocker.patch.object(colab_initializer, 'logger', autospec=True)
        mocker.patch.object(colab_initializer, 'setup_handler', autospec=True)
        mocker.patch.object(colab_initializer.setup_handler, 'is_drive_mounted', return_value=True)
        mocker.patch.object(colab_initializer.setup_handler, 'has_write_access', return_value=True)
        mocker.patch.object(colab_initializer.setup_handler, 'setup_folders', return_value={'status': 'success'})
        mocker.patch.object(colab_initializer.setup_handler, 'create_symlinks', return_value={'status': 'error', 'error': 'Failed to create symlink'})
        
        # Act
        result = colab_initializer.initialize()
        
        # Assert
        assert result['success'] is False
        colab_initializer.logger.error.assert_called_with('❌ Gagal membuat symlink: Failed to create symlink')


class TestOperationFailureScenarios:
    """Kelas untuk menguji skenario kegagalan operasi."""

    def test_operation_timeout(self, colab_initializer, mocker):
        """
        Test untuk skenario ketika operasi timeout.
        """
        # Arrange
        mocker.patch.object(colab_initializer, 'logger', autospec=True)
        mocker.patch.object(colab_initializer, 'operation_handler', autospec=True)
        mocker.patch.object(colab_initializer.operation_handler, 'run_operation', return_value={'status': 'error', 'error': 'Operation timed out'})
        
        # Act
        result = colab_initializer.operation_handler.run_operation('Test operation')
        
        # Assert
        assert result['status'] == 'error'
        colab_initializer.logger.error.assert_called_with('❌ Operasi gagal (timeout): Test operation')

    def test_operation_unexpected_error(self, colab_initializer, mocker):
        """
        Test untuk skenario ketika terjadi error tak terduga selama operasi.
        """
        # Arrange
        mocker.patch.object(colab_initializer, 'logger', autospec=True)
        mocker.patch.object(colab_initializer, 'operation_handler', autospec=True)
        mocker.patch.object(colab_initializer.operation_handler, 'run_operation', return_value={'status': 'error', 'error': 'Unexpected error'})
        
        # Act
        result = colab_initializer.operation_handler.run_operation('Test operation')
        
        # Assert
        assert result['status'] == 'error'
        colab_initializer.logger.error.assert_called_with('❌ Operasi gagal (kesalahan tidak terduga): Test operation')
