"""
file_path: tests/ui/setup/colab/test_colab_initializer_error_handling.py

Tes untuk inisialisasi UI Colab dengan fokus pada penanganan error.
"""

import pytest
from unittest.mock import patch, MagicMock

# Import yang diperlukan
from smartcash.ui.setup.colab.colab_initializer import ColabEnvInitializer


def test_colab_initializer_error_handling():
    """
    Test penanganan error selama inisialisasi UI Colab.
    Memastikan error handler terpusat digunakan saat terjadi exception.
    """
    # Mock create_colab_ui untuk melempar exception
    with patch('smartcash.ui.setup.colab.components.ui_components.create_colab_ui') as mock_create_ui:
        mock_create_ui.side_effect = Exception("name 'ß' is not defined")

        # Mock get_error_handler dan handle_exception untuk memastikan dipanggil
        with patch('smartcash.ui.core.shared.error_handler.get_error_handler') as mock_get_handler:
            mock_error_handler = MagicMock()
            mock_error_component = MagicMock()
            mock_error_handler.handle_exception.return_value = mock_error_component
            mock_get_handler.return_value = mock_error_handler

            initializer = ColabEnvInitializer()
            result = initializer.initialize()

            # Verifikasi hasil
            assert result["status"] is False
            assert "error" in result
            assert "ß" in result["error"]
            assert "ui" in result
            assert "ui" in result["ui"]
            assert result["ui"]["ui"] == mock_error_component

            # Verifikasi bahwa get_error_handler dan handle_exception dipanggil dengan parameter yang benar
            mock_get_handler.assert_called_once_with("colab")
            mock_error_handler.handle_exception.assert_called_once()
            call_args = mock_error_handler.handle_exception.call_args[0]
            assert "ß" in str(call_args[0])
            assert len(mock_error_handler.handle_exception.call_args[1]) >= 1
            assert mock_error_handler.handle_exception.call_args[1]["context"] == "UI Initialization"


def test_colab_initializer_successful_init():
    """
    Test inisialisasi UI Colab yang berhasil.
    Memastikan handler dan UI components dikembalikan dengan benar.
    """
    initializer = ColabEnvInitializer()
    with patch('smartcash.ui.setup.colab.components.ui_components.create_colab_ui') as mock_create_ui:
        mock_ui_components = {"main_container": MagicMock()}
        mock_create_ui.return_value = mock_ui_components

        result = initializer.initialize()

        assert result["status"] is True
        assert result["ui"] == mock_ui_components
        assert "handlers" in result
        assert isinstance(result["handlers"], dict)
        assert "env_config" in result["handlers"]
        assert "setup" in result["handlers"]
