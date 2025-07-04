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
        # Initialize the base class first to set up the logger
        super().__init__(module_name="env_config", parent_module="setup")
        
        self._handlers: Dict[str, Any] = {}
        self._ui_components: Dict[str, Any] = {}
        
        try:
            # Try to import environment manager
            from smartcash.common.environment import get_environment_manager
            self._env_manager = get_environment_manager()
            self.logger.info("🔧 Environment configuration initializer siap")
        except ImportError as ie:
            # Handle case where environment module is not available
            self.logger.warning("⚠️ Environment module not available, using fallback")
            self._env_manager = None
        except Exception as exc:  # pragma: no cover
            # For all other exceptions, log and re-raise
            msg = f"❌ Gagal menginisialisasi environment manager: {exc}"
            self.logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(self, config: Dict[str, Any] | None = None, **kwargs) -> Dict[str, Any]:
        """Initialize environment configuration UI and handlers.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with status and UI components
        """
        try:
            # Set config
            self._config = {**self._get_default_config(), **(config or {})}
            
            # Run pre-initialization checks
            self.pre_initialize_checks()
            
            # Create UI components
            ui_components = create_env_config_ui()
            self._ui_components = ui_components
            
            # Initialize handlers
            handler = EnvConfigHandler(ui_components)
            self._handlers['env_config'] = handler
            
            # Initialize setup handler
            setup_handler = SetupHandler(ui_components)
            self._handlers['setup'] = setup_handler
            
            # Connect setup button to handler
            if 'setup_button' in ui_components:
                setup_button = ui_components['setup_button']
                # Replace placeholder handler with actual handler
                for callback in list(setup_button._click_callbacks):
                    setup_button.on_click(callback, remove=True)
                setup_button.on_click(handler.handle_setup_button_click)
                self.logger.debug("✅ Connected setup button to handler")
            
            # Create main container
            main_container = MainContainer(
                header=ui_components.get('header_container'),
                content=[
                    ui_components.get('summary_container'),
                    ui_components.get('progress_tracker').widget,
                    ui_components.get('env_info_panel'),
                    ui_components.get('form_container'),
                    ui_components.get('tips_requirements')
                ],
                footer=ui_components.get('footer_container')
            )
            
            # Store the main container in UI components
            ui_components['main_container'] = main_container
            ui_components['ui'] = main_container.widget
            
            # Run post-initialization checks
            self.post_initialization_checks()
            self._initialized = True
            
            self.logger.info("✅ Environment configuration UI initialized successfully")
            
            # Return success with status key (not success) for API consistency
            return {
                'status': True,
                'ui': ui_components,
                'handlers': self._handlers
            }
            
        except Exception as e:
            # Log error and return failure
            msg = f"Failed to initialize environment configuration UI: {e}"
            self.logger.error(msg, exc_info=True)
            
            # Create simple error UI
            from ipywidgets import HTML
            error_ui = {'ui': HTML(f"<div style='color:red;padding:10px;'>❌ {msg}</div>")}
            
            return {
                'status': False,
                'error': str(e),
                'ui': error_ui
            }

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


def initialize_env_config_ui(config: Dict[str, Any] | None = None, **kwargs):  
    """Notebook/CLI helper that returns the rendered widget via `safe_display`."""
    
    # Initialize the UI with a fresh instance
    initializer = EnvConfigInitializer()
    result = initializer.initialize(config=config, **kwargs)
    
    # Check for success using 'status' key for API consistency
    # This follows the memory to use 'status' instead of 'success'
    if not result.get('status', False):
        return result.get('ui', {}).get('ui', {})
    
    # Return the main UI widget directly
    return result.get('ui', {}).get('ui', {})
