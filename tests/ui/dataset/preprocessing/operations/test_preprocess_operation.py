"""
File: tests/ui/dataset/preprocessing/operations/test_preprocess_operation.py
Description: Unit tests for the PreprocessOperationHandler.
"""
import pytest
from unittest.mock import MagicMock, patch, ANY, call

from smartcash.ui.dataset.preprocessing.operations.preprocess_operation import PreprocessOperationHandler



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


def test_preprocess_operation_execute_success(mock_ui_module, mock_config, mock_callbacks):
    """Test execute when the preprocessing operation is successful."""
    handler = PreprocessOperationHandler(ui_module=mock_ui_module, config=mock_config, callbacks=mock_callbacks)
    backend_result = {
        "success": True,
        "statistics": {
            "files_processed": 100,
            "files_skipped": 10,
            "files_failed": 0
        },
        "total_time_seconds": 45.5,
        "message": "Preprocessing complete."
    }

    with patch.object(handler, 'log_operation') as mock_log_operation, \
         patch('smartcash.ui.dataset.preprocessing.operations.preprocess_operation.preprocess_dataset', return_value=backend_result) as mock_preprocess:
        handler.execute()

        mock_preprocess.assert_called_once_with(config=mock_config, progress_callback=ANY)
        mock_log_operation.assert_has_calls([
            call("🚀 Menghubungkan ke backend untuk pra-pemrosesan...", level='info'),
            call("✅ Pra-pemrosesan berhasil.", level='success')
        ])

        mock_callbacks['on_success'].assert_called_once()
        summary_arg = mock_callbacks['on_success'].call_args[0][0]
        assert "| **Status Operasi** | ✅ Berhasil |" in summary_arg
        assert "| **File Diproses** | ✔️ 100 |" in summary_arg
        assert "| **File Dilewati** | ⏭️ 10 |" in summary_arg
        assert "| **File Gagal** | ❌ 0 |" in summary_arg
        assert "| **Total Waktu** | ⏱️ 45.50 detik |" in summary_arg
        assert "**Pesan dari Backend:** *Preprocessing complete.*" in summary_arg

        mock_callbacks['on_failure'].assert_not_called()
        mock_callbacks['on_complete'].assert_called_once()


def test_preprocess_operation_execute_failure(mock_ui_module, mock_config, mock_callbacks):
    """Test execute when the preprocessing operation fails."""
    handler = PreprocessOperationHandler(ui_module=mock_ui_module, config=mock_config, callbacks=mock_callbacks)
    backend_result = {"success": False, "message": "Invalid file format found."}

    with patch.object(handler, 'log_operation') as mock_log_operation, \
         patch('smartcash.ui.dataset.preprocessing.operations.preprocess_operation.preprocess_dataset', return_value=backend_result) as mock_preprocess:
        handler.execute()

        mock_preprocess.assert_called_once_with(config=mock_config, progress_callback=ANY)
        mock_log_operation.assert_any_call("❌ Gagal melakukan pra-pemrosesan: Invalid file format found.", level='error')
        
        mock_callbacks['on_failure'].assert_called_once()
        summary_arg = mock_callbacks['on_failure'].call_args[0][0]
        assert "| **Status Operasi** | ❌ Gagal |" in summary_arg
        assert "**Pesan dari Backend:** *Invalid file format found.*" in summary_arg

        mock_callbacks['on_success'].assert_not_called()
        mock_callbacks['on_complete'].assert_called_once()


def test_preprocess_operation_execute_exception(mock_ui_module, mock_config, mock_callbacks):
    """Test execute when an exception occurs during the backend call."""
    handler = PreprocessOperationHandler(ui_module=mock_ui_module, config=mock_config, callbacks=mock_callbacks)

    with patch.object(handler, 'log_operation') as mock_log_operation, \
         patch('smartcash.ui.dataset.preprocessing.operations.preprocess_operation.preprocess_dataset', side_effect=Exception("Critical error")):
        handler.execute()

        mock_log_operation.assert_any_call("❌ Gagal memanggil backend pra-pemrosesan: Critical error", level='error')
        mock_callbacks['on_failure'].assert_called_once_with("Gagal memanggil backend pra-pemrosesan: Critical error")
        mock_callbacks['on_success'].assert_not_called()
        mock_callbacks['on_complete'].assert_called_once()


@pytest.mark.parametrize("result, expected_strings", [
    ({"success": True, "statistics": {"files_processed": 200, "files_skipped": 5, "files_failed": 1}, "total_time_seconds": 120.1, "message": "Done"}, 
     ["✅ Berhasil", "✔️ 200", "⏭️ 5", "❌ 1", "⏱️ 120.10 detik", "*Done*"]),
    ({"success": False, "statistics": {}, "total_time_seconds": 0, "message": "Error occurred"}, 
     ["❌ Gagal", "✔️ 0", "⏭️ 0", "❌ 0", "⏱️ 0.00 detik", "*Error occurred*"])
])
def test_format_preprocess_summary(result, expected_strings):
    """Test the summary formatting for various preprocessing results."""
    handler = PreprocessOperationHandler(ui_module=MagicMock(), config={}, callbacks={})
    summary = handler._format_preprocess_summary(result)

    assert "### Ringkasan Operasi Pra-pemrosesan" in summary
    for expected_string in expected_strings:
        assert expected_string in summary
