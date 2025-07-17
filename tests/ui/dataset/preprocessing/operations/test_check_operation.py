"""
File: tests/ui/dataset/preprocessing/operations/test_check_operation.py
Description: Unit tests for the CheckOperationHandler.
"""
import pytest
from unittest.mock import MagicMock, patch, call

from smartcash.ui.dataset.preprocessing.operations.check_operation import CheckOperationHandler


@pytest.fixture
def mock_ui_module():
    """Fixture for a mocked PreprocessingUIModule."""
    return MagicMock()


@pytest.fixture
def mock_config():
    """Fixture for a sample configuration dictionary."""
    return {"dataset": {"path": "/fake/path"}}


@pytest.fixture
def success_callbacks():
    """Fixture for mocked callbacks."""
    return {
        'on_success': MagicMock(),
        'on_failure': MagicMock(),
        'on_complete': MagicMock()
    }


def test_check_operation_execute_service_ready_and_data_exists(mock_ui_module, mock_config, success_callbacks):
    """Test execute when the backend service is ready and dataset exists."""
    handler = CheckOperationHandler(ui_module=mock_ui_module, config=mock_config, callbacks=success_callbacks)
    backend_status = {
        "service_ready": True,
        "file_statistics": {
            "train": {
                "raw_images": 200,
                "preprocessed_files": 150,
                "missing_files": 10
            }
        },
        "paths": {
            "raw_data_path": "/fake/path/raw",
            "preprocessed_data_path": "/fake/path/preprocessed"
        },
        "message": "Status check complete."
    }

    with patch.object(handler, 'log_operation') as mock_log_operation, \
         patch('smartcash.ui.dataset.preprocessing.operations.check_operation.get_preprocessing_status', return_value=backend_status) as mock_get_status:
        
        handler.execute()

        mock_get_status.assert_called_once_with(config=mock_config)
        mock_log_operation.assert_has_calls([
            call("🔍 Menghubungkan ke backend untuk memeriksa status...", level='info'),
            call("✅ Pemeriksaan berhasil. Ringkasan status dibuat.", level='success')
        ])

        success_callbacks['on_success'].assert_called_once()
        summary_arg = success_callbacks['on_success'].call_args[0][0]
        assert "| **Status Layanan** | ✅ Siap |" in summary_arg
        assert "| **Gambar Mentah** | 🖼️ 200 |" in summary_arg
        assert "| **File Diproses** | ✨ 150 |" in summary_arg
        assert "| **File Hilang** | ❓ 10 |" in summary_arg
        assert "- **Data Mentah:** `/fake/path/raw`" in summary_arg
        assert "- **Data Diproses:** `/fake/path/preprocessed`" in summary_arg
        assert "**Pesan dari Backend:** *Status check complete.*" in summary_arg

        success_callbacks['on_failure'].assert_not_called()
        success_callbacks['on_complete'].assert_called_once()


def test_check_operation_execute_service_not_ready(mock_ui_module, mock_config, success_callbacks):
    """Test execute when the backend service is not ready."""
    handler = CheckOperationHandler(ui_module=mock_ui_module, config=mock_config, callbacks=success_callbacks)
    backend_status = {"service_ready": False, "message": "Service is currently offline for maintenance."}

    with patch.object(handler, 'log_operation') as mock_log_operation, \
         patch('smartcash.ui.dataset.preprocessing.operations.check_operation.get_preprocessing_status', return_value=backend_status):
        handler.execute()

        mock_log_operation.assert_any_call("❌ Backend melaporkan layanan belum siap.", level='error')
        success_callbacks['on_failure'].assert_called_once()
        summary_arg = success_callbacks['on_failure'].call_args[0][0]
        assert "| **Status Layanan** | ❌ Tidak Siap |" in summary_arg
        assert "**Pesan dari Backend:** *Service is currently offline for maintenance.*" in summary_arg
        success_callbacks['on_success'].assert_not_called()
        success_callbacks['on_complete'].assert_called_once()


def test_check_operation_execute_exception(mock_ui_module, mock_config, success_callbacks):
    """Test execute when an exception occurs during the backend call."""
    handler = CheckOperationHandler(ui_module=mock_ui_module, config=mock_config, callbacks=success_callbacks)

    with patch.object(handler, 'log_operation') as mock_log_operation, \
         patch('smartcash.ui.dataset.preprocessing.operations.check_operation.get_preprocessing_status', side_effect=Exception("Connection timed out")):
        handler.execute()

        mock_log_operation.assert_any_call("❌ Gagal memanggil backend pemeriksaan status: Connection timed out", level='error')
        success_callbacks['on_failure'].assert_called_once_with("Gagal memanggil backend pemeriksaan status: Connection timed out")
        success_callbacks['on_success'].assert_not_called()
        success_callbacks['on_complete'].assert_called_once()


@pytest.mark.parametrize("status, expected_strings", [
    ({
        "service_ready": True,
        "file_statistics": {"train": {"raw_images": 120, "preprocessed_files": 100, "missing_files": 5}},
        "paths": {"raw_data_path": "/path/a", "preprocessed_data_path": "/path/b"},
        "message": "OK"
    }, 
     ["| ✅ Siap |", "| 🖼️ 120 |", "| ✨ 100 |", "| ❓ 5 |", "`/path/a`", "`/path/b`", "*OK*"]),
    ({
        "service_ready": False,
        "message": "Offline for maintenance"
    }, 
     ["| ❌ Tidak Siap |", "*Offline for maintenance*"])
])
def test_format_status_summary(status, expected_strings):
    """Test the summary formatting for various status responses."""
    handler = CheckOperationHandler(ui_module=MagicMock(), config={}, callbacks={})
    summary = handler._format_status_summary(status)

    assert "### Ringkasan Status Pra-pemrosesan" in summary
    for expected_string in expected_strings:
        assert expected_string in summary
