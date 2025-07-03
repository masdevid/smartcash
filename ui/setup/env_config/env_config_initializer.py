"""Environment Configuration Initializer

Provides `EnvConfigInitializer`, responsible for boot-strapping the Environment
Configuration UI.  All log suppression and buffering is centrally handled by
`EnhancedUILogger`, so this class just logs; the logger decides whether to emit
immediately or buffer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from smartcash.ui.core.initializers.module_initializer import ModuleInitializer

# Handlers / components
from smartcash.ui.setup.env_config.handlers.env_config_handler import EnvConfigHandler
from smartcash.ui.setup.env_config.configs.config_handler import ConfigHandler
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
from smartcash.ui.setup.env_config.components.ui_components import create_env_config_ui
from smartcash.ui.setup.env_config.configs.defaults import DEFAULT_CONFIG


class EnvConfigInitializer(ModuleInitializer):
    """Boot-straps the Environment Configuration UI."""

    def __init__(self) -> None:
        try:
            super().__init__(module_name="env_config", parent_module="setup")

            from smartcash.common.environment import get_environment_manager

            self._env_manager = get_environment_manager()
            self._handlers: Dict[str, Any] = {}
            self._ui_components: Dict[str, Any] = {}

            self.logger.info("🔧 Environment configuration initializer siap")
        except Exception as exc:  # pragma: no cover
            msg = f"❌ Gagal menginisialisasi environment manager: {exc}"
            self.logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(self, config: Dict[str, Any] | None = None, **kwargs) -> Dict[str, Any]:
        """Run the full initialization workflow and return a result dict."""
        try:
            self._config = {**self._get_default_config(), **(config or {})}

            self.pre_initialize_checks()
            self._ui_components = self.create_ui_components()
            self.setup_handlers()
            self.post_initialization_checks()
            self._initialized = True

            self.logger.info("✅ Environment configuration UI initialized successfully")
            return {"success": True, "ui": self._ui_components, "handlers": self._handlers}
        except Exception as exc:  # pragma: no cover – surfaced via UI
            from smartcash.ui.core.shared.error_handler import get_error_handler

            msg = f"Failed to initialize environment configuration UI: {exc}"
            self.logger.error(msg, exc_info=True)
            error_handler = get_error_handler("env_config")
            error_ui = error_handler.handle_error(
                msg,
                level="error",
                exc_info=True,
                fail_fast=False,
                create_ui_error=True,
            )
            if error_ui is None:
                error_ui = {}
            return {"success": False, "error": msg, "ui": error_ui}

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        return DEFAULT_CONFIG.copy()

    def pre_initialize_checks(self) -> None:
        if not self._env_manager:
            raise RuntimeError("Environment manager tidak tersedia")

        for path in (Path.home(), Path.cwd()):
            if not path.exists():
                self.logger.warning("⚠️ Path tidak ditemukan: %s", path)

    def create_ui_components(self) -> Dict[str, Any]:
        self.logger.info("🎨 Membuat UI components…")
        try:
            ui = create_env_config_ui()
            self.logger.info("✅ UI components berhasil dibuat")
            return ui
        except Exception as exc:
            msg = f"❌ Gagal create UI components: {exc}"
            self.logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from exc

    def setup_handlers(self) -> None:
        self.logger.info("🔧 Setup handlers…")
        try:
            config_handler = ConfigHandler()
            config_handler.config = self._config

            setup_handler = SetupHandler()
            setup_handler.config = self._config

            env_config_handler = EnvConfigHandler(ui_components=self._ui_components)
            env_config_handler.config = self._config
            env_config_handler.setup_dependencies(
                config_handler=config_handler,
                setup_handler=setup_handler,
            )

            self._handlers = {
                "env_config": env_config_handler,
                "config": config_handler,
                "setup": setup_handler,
            }
            self.logger.info("✅ Handlers berhasil di-setup")
        except Exception as exc:
            msg = f"❌ Gagal setup handlers: {exc}"
            self.logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from exc

    def post_initialization_checks(self) -> None:
        self.logger.info("🔍 Melakukan post-initialization checks…")
        try:
            setup_handler: SetupHandler | None = self._handlers.get("setup")
            if not setup_handler:
                self.logger.warning("⚠️ Setup handler tidak tersedia")
                return

            setup_handler.perform_initial_status_check(self._ui_components)

            if setup_handler.should_sync_config_templates():
                self.logger.info("📋 Drive mounted dan setup complete, syncing config templates…")
                setup_handler.sync_config_templates(
                    force_overwrite=False,
                    update_ui=True,
                    ui_components=self._ui_components,
                )

            self.logger.info("✅ Post-initialization checks selesai")
        except Exception as exc:
            self.logger.error("❌ Error during post-initialization checks: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def handlers(self) -> Dict[str, Any]:
        return self._handlers

    def get_handler(self, name: str):
        return self._handlers.get(name)


def initialize_env_config_ui(config: Dict[str, Any] | None = None, **kwargs):  # noqa: ANN001
    """Notebook/CLI helper that returns the rendered widget via `safe_display`."""

    initializer = EnvConfigInitializer()
    result = initializer.initialize(config=config, **kwargs)

    # Check if initialization was successful (using 'status' key for API consistency)
    if not result.get('status', False):
        # If there was an error, display the error UI
        return result.get('ui', {}).get('ui', {})
    
    # Return the main UI widget directly
    return result.get('ui', {}).get('ui', {})
