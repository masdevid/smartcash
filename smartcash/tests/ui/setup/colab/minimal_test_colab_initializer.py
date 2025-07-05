# -*- coding: utf-8 -*-
"""
file_path: /Users/masdevid/Projects/smartcash/tests/ui/setup/colab/minimal_test_colab_initializer.py

Tes minimal untuk ColabEnvInitializer untuk mengisolasi masalah dengan _post_checks.
"""

import pytest
from unittest.mock import MagicMock
from smartcash.ui.setup.colab.colab_initializer import ColabEnvInitializer

# Kelas logger mock sederhana
class SimpleMockLogger:
    def info(self, *args, **kwargs):
        print(f"[INFO] {args}")
    def warning(self, *args, **kwargs):
        print(f"[WARNING] {args}")
    def error(self, *args, **kwargs):
        print(f"[ERROR] {args}")
    def debug(self, *args, **kwargs):
        print(f"[DEBUG] {args}")
    def critical(self, *args, **kwargs):
        print(f"[CRITICAL] {args}")
    def exception(self, *args, **kwargs):
        print(f"[EXCEPTION] {args}")
    def log(self, level, *args, **kwargs):
        print(f"[LOG LEVEL {level}] {args}")

@pytest.fixture
def minimal_colab_initializer(mocker):
    """
    Fixture untuk membuat instance minimal dari ColabEnvInitializer.
    """
    # Mock semua dependensi eksternal
    # Tidak lagi mencoba mock fungsi spesifik yang mungkin tidak ada
    mocker.patch("smartcash.ui.setup.colab.components.create_colab_ui", return_value=MagicMock())
    mocker.patch("smartcash.ui.setup.colab.handlers.colab_config_handler.ColabConfigHandler", return_value=MagicMock())
    mocker.patch("smartcash.ui.handlers.config_handlers.ConfigHandler", return_value=MagicMock())
    mocker.patch("smartcash.common.environment.get_environment_manager", return_value=MagicMock())
    mocker.patch("smartcash.ui.core.shared.logger.get_enhanced_logger", return_value=SimpleMockLogger())
    mocker.patch("smartcash.ui.core.shared.logger.get_module_logger", return_value=SimpleMockLogger())
    
    # Buat instance
    init_instance = ColabEnvInitializer()
    
    # Set logger mock secara langsung tanpa mengakses fungsi get_enhanced_logger
    mock_logger = SimpleMockLogger()
    setattr(init_instance, "logger", mock_logger)
    
    # Set handler setup mock
    setup_handler = MagicMock()
    setup_handler.perform_initial_status_check = MagicMock(return_value={"status": "ok"})
    setup_handler.should_sync_config_templates = MagicMock(return_value=False)
    init_instance._handlers = {"setup": setup_handler}
    
    # Set komponen UI mock
    init_instance._ui_components = MagicMock()
    
    return init_instance

def test_minimal_post_checks(minimal_colab_initializer):
    """
    Tes minimal untuk memanggil _post_checks pada ColabEnvInitializer.
    """
    init_instance = minimal_colab_initializer
    print("\n[DEBUG] Memulai tes minimal _post_checks")
    print(f"[DEBUG] Instance: {init_instance}")
    print(f"[DEBUG] Handlers: {init_instance._handlers}")
    print(f"[DEBUG] UI Components: {init_instance._ui_components}")
    
    # Mock initialize() to set _initialized attribute
    def mock_initialize(self, config=None, **kwargs):
        self._initialized = True
        print(f"[DEBUG] Mocked initialize() called, set _initialized to True")
        return {"status": True, "ui": self._ui_components, "handlers": self._handlers}

    # Patch the initialize method
    import types
    init_instance.initialize = types.MethodType(mock_initialize, init_instance)
    print(f"[DEBUG] Patched initialize() method")

    # Manually set both possible attributes to True to bypass potential issues
    setattr(init_instance, "_initialized", True)
    setattr(init_instance, "_is_initialized", True)
    print(f"[DEBUG] Set _initialized to True")
    print(f"[DEBUG] Set _is_initialized to True")
    # Additional debug output to inspect object state
    print(f"[DEBUG] Object attributes before _post_checks: {dir(init_instance)}")
    
    # Mock SetupHandler methods to prevent errors in _post_checks
    class MockSetupHandler:
        def __init__(self):
            pass

        def perform_initial_status_check(self, ui_components):
            print(f"[DEBUG] Mocked perform_initial_status_check called")
            return None

        def should_sync_config_templates(self):
            print(f"[DEBUG] Mocked should_sync_config_templates called")
            return False

        def sync_config_templates(self, force_overwrite=False, update_ui=True, ui_components=None):
            print(f"[DEBUG] Mocked sync_config_templates called")
            return None

    # Set up mock handler
    mock_setup_handler = MockSetupHandler()
    init_instance._handlers["setup"] = mock_setup_handler
    print(f"[DEBUG] Set up mock SetupHandler in handlers")

    # Call _post_checks independently
    try:
        print(f"[DEBUG] Memanggil _post_checks...")
        init_instance._post_checks()
        print(f"[DEBUG] _post_checks selesai tanpa error")
    except Exception as e:
        print(f"[DEBUG] Kesalahan saat memanggil _post_checks: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    print("[DEBUG] Tes minimal selesai")
