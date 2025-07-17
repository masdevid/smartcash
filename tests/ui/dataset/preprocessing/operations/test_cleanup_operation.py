"""
File: tests/ui/dataset/preprocessing/operations/test_cleanup_operation.py
Description: Unit tests for the CleanupOperationHandler.
"""
import pytest
from unittest.mock import MagicMock, patch, ANY, call

from smartcash.ui.dataset.preprocessing.operations.cleanup_operation import CleanupOperationHandler


@pytest.fixture
def mock_ui_module():
    """Fixture for a mocked PreprocessingUIModule."""
    return MagicMock()


@pytest.fixture
def mock_config():
    """Fixture for a sample configuration dictionary."""
    return {"dataset": {"path": "/fake/path"}}


@pytest.fixture
def mock_callbacks():
    """Fixture for mocked callbacks."""
    return {
        'on_success': MagicMock(),
        'on_failure': MagicMock(),
        'on_complete': MagicMock()
    }


def test_cleanup_operation_execute_success(mock_ui_module, mock_config, mock_callbacks):
    """Test execute when the cleanup operation is successful."""
    handler = CleanupOperationHandler(ui_module=mock_ui_module, config=mock_config, callbacks=mock_callbacks)
    backend_result = {
        "success": True,
        "files_deleted": 50,
        "space_reclaimed_mb": 120.5,
        "message": "Cleanup complete."
    }

    with patch.object(handler, 'log_operation') as mock_log_operation, \
         patch('smartcash.ui.dataset.preprocessing.operations.cleanup_operation.cleanup_preprocessing_files', return_value=backend_result) as mock_cleanup:
        handler.execute()

        mock_cleanup.assert_called_once_with(config=mock_config)
        mock_log_operation.assert_has_calls([
            call("🧹 Menghubungkan ke backend untuk pembersihan...", level='info'),
            call(" Pembersihan berhasil. 50 file telah dihapus.", level='success')
        ])
        
        mock_callbacks['on_success'].assert_called_once()
        summary_arg = mock_callbacks['on_success'].call_args[0][0]
        assert "| **Status Operasi** | ✅ Berhasil |" in summary_arg
        assert "| **File Dihapus** | 🗑️ 50 |" in summary_arg
        assert "| **Ruang Kosong** | 💾 120.50 MB |" in summary_arg
        assert "**Pesan dari Backend:** *Cleanup complete.*" in summary_arg

        mock_callbacks['on_failure'].assert_not_called()
        mock_callbacks['on_complete'].assert_called_once()


def test_cleanup_operation_execute_failure(mock_ui_module, mock_config, mock_callbacks):
    """Test execute when the cleanup operation fails."""
    handler = CleanupOperationHandler(ui_module=mock_ui_module, config=mock_config, callbacks=mock_callbacks)
    backend_result = {"success": False, "message": "Permission denied."}

    with patch.object(handler, 'log_operation') as mock_log_operation, \
         patch('smartcash.ui.dataset.preprocessing.operations.cleanup_operation.cleanup_preprocessing_files', return_value=backend_result) as mock_cleanup:
        handler.execute()

        mock_cleanup.assert_called_once_with(config=mock_config)
        mock_log_operation.assert_any_call(" Gagal melakukan pembersihan: Permission denied.", level='error')
        
        mock_callbacks['on_failure'].assert_called_once()
        summary_arg = mock_callbacks['on_failure'].call_args[0][0]
        assert "| **Status Operasi** | ❌ Gagal |" in summary_arg
        assert "**Pesan dari Backend:** *Permission denied.*" in summary_arg

        mock_callbacks['on_success'].assert_not_called()
        mock_callbacks['on_complete'].assert_called_once()


def test_cleanup_operation_execute_exception(mock_ui_module, mock_config, mock_callbacks):
    """Test execute when an exception occurs during the backend call."""
    handler = CleanupOperationHandler(ui_module=mock_ui_module, config=mock_config, callbacks=mock_callbacks)

    with patch.object(handler, 'log_operation') as mock_log_operation, \
         patch('smartcash.ui.dataset.preprocessing.operations.cleanup_operation.cleanup_preprocessing_files', side_effect=Exception("Disk is full")):
        handler.execute()

        mock_log_operation.assert_any_call("❌ Gagal memanggil backend pembersihan: Disk is full", level='error')
        mock_callbacks['on_failure'].assert_called_once_with("Gagal memanggil backend pembersihan: Disk is full")
        mock_callbacks['on_success'].assert_not_called()
        mock_callbacks['on_complete'].assert_called_once()


@pytest.mark.parametrize("result, expected_strings", [
    ({"success": True, "files_deleted": 100, "space_reclaimed_mb": 200.25, "message": "Done"}, 
     ["✅ Berhasil", "🗑️ 100", "💾 200.25 MB", "*Done*"]),
    ({"success": False, "files_deleted": 0, "space_reclaimed_mb": 0, "message": "Error"}, 
     ["❌ Gagal", "🗑️ 0", "💾 0.00 MB", "*Error*"])
])
def test_format_cleanup_summary(result, expected_strings):
    """Test the summary formatting for various cleanup results."""
    handler = CleanupOperationHandler(ui_module=MagicMock(), config={}, callbacks={})
    summary = handler._format_cleanup_summary(result)

    assert "### Ringkasan Operasi Pembersihan" in summary
    for expected_string in expected_strings:
        assert expected_string in summary
