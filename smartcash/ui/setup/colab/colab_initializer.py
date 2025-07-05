"""
file_path: smartcash/ui/setup/colab/colab_initializer.py
Deskripsi: Inisialisasi konfigurasi lingkungan khusus Google Colab.

File ini menyediakan entry-point untuk menginisialisasi UI konfigurasi lingkungan
khusus Google Colab di SmartCash.

Contoh penggunaan:
    from smartcash.ui.setup.colab.colab_initializer import initialize_colab_ui
    ui = initialize_colab_ui()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from smartcash.ui.core.initializers.module_initializer import ModuleInitializer

# ---------------------------------------------------------------------------
# Untuk sementara, gunakan handler/komponen lama dari env_config
# ---------------------------------------------------------------------------
# Import local modules
from smartcash.ui.setup.colab.handlers.colab_config_handler import ColabConfigHandler
from smartcash.ui.setup.colab.handlers.setup_handler import SetupHandler
from smartcash.ui.setup.colab.components.colab_ui import create_colab_ui_components
from smartcash.ui.setup.colab.configs.defaults import DEFAULT_CONFIG



class ColabEnvInitializer(ModuleInitializer):
    """Boot-straps Environment Configuration UI khusus Colab."""

    def __init__(self) -> None:
        super().__init__(module_name="colab", parent_module="setup")

        self._handlers: Dict[str, Any] = {}
        self._ui_components: Dict[str, Any] = {}

        try:
            from smartcash.common.environment import get_environment_manager

            self._env_manager = get_environment_manager()
            self.logger.info("🔧 Colab environment initializer siap")
        except ImportError:
            self.logger.warning("⚠️ smartcash.common.environment tidak tersedia, fallback")
            self._env_manager = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(self, config: Dict[str, Any] | None = None, **kwargs) -> Dict[str, Any]:
        """Inisialisasi UI konfigurasi environment.

        Args:
            config: Konfigurasi opsional.
            **kwargs: Argumen tambahan.

        Returns:
            Dict berisi status, ui, dan handlers.
        """
        try:
            self._config = {**self._get_default_config(), **(config or {})}
            self._pre_checks()

            # Initialize UI components with the new container structure
            ui_components = create_colab_ui_components()
            self._ui_components = ui_components

            # Initialize handlers with the UI components
            env_handler = ColabConfigHandler(ui_components)
            self._handlers["env_config"] = env_handler

            setup_handler = SetupHandler(ui_components)
            self._handlers["setup"] = setup_handler

            # Connect the setup button handler
            action_container = ui_components.get("action_container", {})
            if action_container and hasattr(action_container, 'get'):
                setup_button = action_container.get('setup_button')
                if setup_button:
                    # Remove any existing click handlers
                    if hasattr(setup_button, '_click_callbacks'):
                        for cb in list(setup_button._click_callbacks):
                            setup_button.on_click(cb, remove=True)
                    # Add our handler
                    setup_button.on_click(env_handler.handle_setup_button_click)
                    self.logger.debug("✅ Tombol setup terhubung ke handler")

            # The main container is already created in create_colab_ui_components()
            main_container = ui_components.get("main_container")
            if main_container:
                ui_components["ui"] = main_container.container
            else:
                self.logger.warning("Main container not found in UI components")

            self._post_checks()
            self._initialized = True

            self.logger.info("✅ Colab environment UI berhasil di-inisialisasi")

            return {"status": True, "ui": ui_components, "handlers": self._handlers}
        except Exception as exc:  # pragma: no cover
            from smartcash.ui.core.shared.error_handler import get_error_handler

            msg = f"❌ Gagal inisialisasi UI: {exc}"
            self.logger.error(msg, exc_info=True)
            error_handler = get_error_handler("colab")
            error_component = error_handler.handle_exception(exc, context="UI Initialization")
            return {
                "status": False,
                "error": str(exc),
                "ui": {"ui": error_component if error_component else None},
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        return DEFAULT_CONFIG.copy()

    def _pre_checks(self) -> None:
        if not self._env_manager:
            raise RuntimeError("Environment manager tidak tersedia")

        for path in (Path.home(), Path.cwd()):
            if not path.exists():
                self.logger.warning("⚠️ Path tidak ditemukan: %s", path)

    def _post_checks(self) -> None:
        self.logger.info("🔍 Post-initialization checks…")
        try:
            setup_handler: SetupHandler | None = self._handlers.get("setup")
            if not setup_handler:
                self.logger.warning("⚠️ Setup handler tidak tersedia")
                return

            setup_handler.perform_initial_status_check(self._ui_components)

            if setup_handler.should_sync_config_templates():
                self.logger.info("📋 Drive mounted dan setup complete, syncing templates…")
                setup_handler.sync_config_templates(
                    force_overwrite=False,
                    update_ui=True,
                    ui_components=self._ui_components,
                )

            self.logger.info("✅ Post-checks selesai")
        except Exception as exc:  # pragma: no cover
            self.logger.error("❌ Error post-checks: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def handlers(self) -> Dict[str, Any]:
        return self._handlers

    def get_handler(self, name: str):
        return self._handlers.get(name)


def initialize_colab_ui(config: Dict[str, Any] | None = None, **kwargs):
    """
    Helper Notebook/CLI untuk langsung menampilkan widget UI.
    
    Args:
        config: Konfigurasi opsional untuk inisialisasi
        **kwargs: Argumen tambahan untuk inisialisasi
        
    Returns:
        Dict[str, Any]: Komponen UI yang dihasilkan
    """
    initializer = ColabEnvInitializer()
    result = initializer.initialize(config=config, **kwargs)

    if not result.get("status", False):
        return result.get("ui", {}).get("ui", {})

    return result.get("ui", {}).get("ui", {})
