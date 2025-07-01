"""
Tests for importing the environment configuration module.

This module verifies that all necessary components can be imported correctly.
"""

def test_import_env_config_initializer():
    """Test that env_config_initializer can be imported."""
    from smartcash.ui.setup.env_config import env_config_initializer
    assert env_config_initializer is not None

def test_import_handlers():
    """Test that all handlers can be imported."""
    from smartcash.ui.setup.env_config.handlers import (
        ConfigHandler,
        DriveHandler,
        FolderHandler,
        StatusChecker,
        SetupHandler,
        BaseEnvHandler,
        EnvConfigErrorHandler
    )
    assert all((
        ConfigHandler,
        DriveHandler,
        FolderHandler,
        StatusChecker,
        SetupHandler,
        BaseEnvHandler,
        EnvConfigErrorHandler
    ))

def test_import_utils():
    """Test that utility modules can be imported."""
    from smartcash.ui.setup.env_config.utils import (
        ui_state,
        dual_progress_tracker,
        env_detector,
        handler_utils,
        ui_updater
    )
    assert all([ui_state, dual_progress_tracker, env_detector, handler_utils, ui_updater])

def test_import_components():
    """Test that UI components can be imported."""
    from smartcash.ui.setup.env_config.components import ui_components
    assert ui_components is not None
