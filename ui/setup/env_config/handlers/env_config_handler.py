"""
File: smartcash/ui/setup/env_config/handlers/env_config_handler.py

Environment Configuration Handler - Refactored dengan arsitektur baru.

Handler utama untuk environment configuration yang mengkoordinasikan
berbagai handlers dan mengelola proses setup environment secara keseluruhan.
"""

from typing import Dict, Any, Optional, List, Callable
from smartcash.ui.core.shared.logger import get_enhanced_logger, UILogger
from pathlib import Path

# Import core handlers dari arsitektur baru
from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from smartcash.ui.core.handlers import ConfigurableHandler

# Import module-specific handlers
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
from smartcash.ui.setup.env_config.configs.config_handler import ConfigHandler
from smartcash.ui.setup.env_config.constants import SetupStage


class EnvConfigHandler(ModuleUIHandler, ConfigurableHandler):
    """Main orchestrator untuk environment configuration.
    
    Handler ini mengkoordinasikan berbagai environment configuration components:
    - SetupHandler: Mengelola setup workflow
    - ConfigHandler: Mengelola configuration (in-memory saja)
    - UI updates dan status management
    
    Menyediakan unified interface untuk UI berinteraksi dengan environment
    configuration system.
    """
    
    # ------------------------------------------------------------------
    # BaseHandler abstract method implementation
    # ------------------------------------------------------------------
    def initialize(self) -> Dict[str, Any]:
        """Concrete initializer to satisfy :class:`BaseHandler`.

        This handler is primarily driven externally (UI components call
        `setup_dependencies` etc.), so there is no heavy lifting here.
        We simply mark the handler as initialized and return a status
        dictionary that can be used by callers to verify success.
        """
        if not self._is_initialized:
            self._is_initialized = True
            self.logger.debug("✅ EnvConfigHandler marked as initialized")
        return {"status": True, "message": "EnvConfigHandler initialized"}

    def __init__(
        self,
        ui_components: Dict[str, Any],
        logger: Optional[UILogger] = None
    ):
        """Initialize environment configuration handler.
        
        Args:
            ui_components: Dictionary berisi UI components
            logger: Optional logger instance
        """
        # Initialize parent classes
        ModuleUIHandler.__init__(
            self,
            module_name='env_config',
            parent_module='setup'
        )
        ConfigurableHandler.__init__(
            self,
            module_name='env_config',
            parent_module='setup'
        )
        
        # Set UI components
        self.ui_components = ui_components
        
        # Initialize state
        self._current_stage = SetupStage.INIT
        self._status = 'idle'
        self._progress = 0.0
        
        # Handlers akan di-inject via setup_dependencies
        self._config_handler = None
        self._setup_handler = None
        
        self.logger = get_enhanced_logger(__name__)
        self.logger.info("🔧 Environment configuration handler initialized")
    
    def setup_dependencies(
        self,
        config_handler: ConfigHandler,
        setup_handler: SetupHandler
    ):
        """Setup dependency injection.
        
        Args:
            config_handler: ConfigHandler instance
            setup_handler: SetupHandler instance
        """
        self._config_handler = config_handler
        self._setup_handler = setup_handler
        
        # Setup handlers dengan UI components
        self._setup_handler.set_ui_components(self.ui_components)
        
        self.logger.info("🔗 Dependencies berhasil di-setup")
    
    @property
    def config_handler(self) -> ConfigHandler:
        """Get config handler.
        
        Returns:
            ConfigHandler instance
            
        Raises:
            RuntimeError: Jika config handler belum di-setup
        """
        if not self._config_handler:
            raise RuntimeError("Config handler belum di-setup")
        return self._config_handler
    
    @property
    def setup_handler(self) -> SetupHandler:
        """Get setup handler.
        
        Returns:
            SetupHandler instance
            
        Raises:
            RuntimeError: Jika setup handler belum di-setup
        """
        if not self._setup_handler:
            raise RuntimeError("Setup handler belum di-setup")
        return self._setup_handler
    
    @property
    def current_stage(self) -> SetupStage:
        """Get current setup stage.
        
        Returns:
            Current SetupStage
        """
        return self._current_stage
    
    def update_status(self, message: str, status_type: str = 'info'):
        """Update status dengan container-aware access."""
        try:
            # Try direct access first
            if 'status_panel' in self.ui_components:
                panel = self.ui_components['status_panel']
                if hasattr(panel, 'update'):
                    panel.update(message, status_type)
                    return
            
            # Try container-based access
            container_keys = ['action_container', 'summary_container', 'main_container']
            for container_key in container_keys:
                if container_key in self.ui_components:
                    container = self.ui_components[container_key]
                    # Check for status panel in container
                    if hasattr(container, 'status_panel'):
                        container.status_panel.update(message, status_type)
                        return
                    # Check container children
                    elif hasattr(container, 'children'):
                        for child in container.children:
                            if hasattr(child, 'status_panel') or getattr(child, '__class__', None).__name__ == 'StatusPanel':
                                status_panel = child.status_panel if hasattr(child, 'status_panel') else child
                                if hasattr(status_panel, 'update'):
                                    status_panel.update(message, status_type)
                                    return
            
            # Fallback: log the status
            self.logger.info(f"📢 Status: {message} ({status_type})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update status: {str(e)}")
    
    def update_progress(self, value: float, message: str = None):
        """Update progress dengan container-aware access."""
        try:
            # Try direct access first
            if 'progress_tracker' in self.ui_components:
                tracker = self.ui_components['progress_tracker']
                if hasattr(tracker, 'update'):
                    tracker.update(value, message)
                    return
            
            # Try container-based access
            container_keys = ['main_container', 'summary_container', 'action_container']
            for container_key in container_keys:
                if container_key in self.ui_components:
                    container = self.ui_components[container_key]
                    # Check for progress tracker in container
                    if hasattr(container, 'progress_tracker'):
                        container.progress_tracker.update(value, message)
                        return
                    # Check container children
                    elif hasattr(container, 'children'):
                        for child in container.children:
                            if hasattr(child, 'progress_tracker') or getattr(child, '__class__', None).__name__ == 'ProgressTracker':
                                tracker = child.progress_tracker if hasattr(child, 'progress_tracker') else child
                                if hasattr(tracker, 'update'):
                                    tracker.update(value, message)
                                    return
            
            # Fallback: log the progress  
            self.logger.info(f"📊 Progress: {value*100:.1f}% - {message or 'Processing...'}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update progress: {str(e)}")
    
    def handle_setup_button_click(self, button):
        """Handle setup button click event.
        
        Args:
            button: Button widget yang diklik
        """
        try:
            self.logger.info("🚀 Setup button clicked")
            
            # Update UI state
            self.update_status("🔄 Memulai environment setup...", 'info')
            self.update_progress(0.1, "Initializing...")
            
            # Perform setup action
            result = self.perform_setup_action('setup_environment')
            # Show error UI if needed
            self.show_error_ui(result)
            
            # Update progress berdasarkan hasil
            if result.get('status', False):
                self.update_progress(1.0, "Setup completed successfully!")
            else:
                self.update_progress(0.0, "Setup failed")
                
        except Exception as e:
            error_msg = f"❌ Error handling setup button click: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.update_status(error_msg, 'error')
            self.update_progress(0.0, "Error occurred")
    
    def perform_setup_action(self, action: str, **kwargs) -> Dict[str, Any]:
        """Perform setup action dengan UI updates.
        
        Args:
            action: Action yang akan dilakukan
            **kwargs: Additional arguments untuk action
            
        Returns:
            Dictionary berisi hasil action
        """
        try:
            self.logger.info(f"🚀 Performing setup action: {action}")
            
            # Update status
            self._status = 'running'
            self.update_status(f"Menjalankan {action}...", 'info')
            
            # Delegate ke setup handler
            result = self.setup_handler.perform_action(action, **kwargs)
            
            # Update status berdasarkan hasil
            if result.get('status', False):
                self._status = 'success'
                self.update_status(f"✅ {action} berhasil", 'success')
            else:
                self._status = 'error'
                self.update_status(f"❌ {action} gagal", 'error')
            
            return result
            
        except Exception as e:
            self._status = 'error'
            error_msg = f"❌ Error performing {action}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.update_status(error_msg, 'error')
            
            return {
                'status': False,
                'error': str(e),
                'message': error_msg
            }
    
    def perform_setup_action(self, action: str, **kwargs) -> Dict[str, Any]:
        """Perform setup action dengan UI updates.
        
        Args:
            action: Action yang akan dilakukan
            **kwargs: Additional arguments untuk action
            
        Returns:
            Dictionary berisi hasil action
        """
        try:
            self.logger.info(f"🚀 Performing setup action: {action}")
            
            # Update status
            self._status = 'running'
            self.update_status(f"Menjalankan {action}...", 'info')
            
            # Delegate ke setup handler
            result = self.setup_handler.perform_action(action, **kwargs)
            
            # Update status berdasarkan hasil
            if result.get('status', False):
                self._status = 'success'
                self.update_status(f"✅ {action} berhasil", 'success')
            else:
                self._status = 'error'
                self.update_status(f"❌ {action} gagal", 'error')
            
            return result
            
        except Exception as e:
            self._status = 'error'
            error_msg = f"❌ Error performing {action}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.update_status(error_msg, 'error')
            
            return {
                'status': False,
                'error': str(e),
                'message': error_msg
            }
    
    def sync_config_templates(
        self,
        force_overwrite: bool = False,
        update_ui: bool = True
    ) -> Dict[str, Any]:
        """Sync config templates dengan UI updates.
        
        Args:
            force_overwrite: Force overwrite existing templates
            update_ui: Update UI components
            
        Returns:
            Dictionary berisi hasil sync
        """
        try:
            if update_ui:
                self.update_status("📋 Syncing config templates...", 'info')
            
            # Delegate ke setup handler
            result = self.setup_handler.sync_config_templates(
                force_overwrite=force_overwrite,
                update_ui=update_ui,
                ui_components=self.ui_components if update_ui else None
            )
            
            if update_ui:
                if result.get('status', False):
                    self.update_status("✅ Config templates synced", 'success')
                else:
                    self.update_status("❌ Config templates sync failed", 'error')
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error syncing config templates: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            if update_ui:
                self.update_status(error_msg, 'error')
            
            return {
                'status': False,
                'error': str(e),
                'message': error_msg
            }
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate environment configuration.
        
        Returns:
            Dictionary berisi validation results
        """
        try:
            self.update_status("🔍 Validating environment...", 'info')
            
            # Delegate ke setup handler
            result = self.setup_handler.validate_environment()
            
            if result.get('status', False):
                self.update_status("✅ Environment validation passed", 'success')
            else:
                self.update_status("❌ Environment validation failed", 'error')
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error validating environment: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.update_status(error_msg, 'error')
            
            return {
                'status': False,
                'error': str(e),
                'message': error_msg
            }
    
    def reset_environment(self) -> Dict[str, Any]:
        """Reset environment configuration.
        
        Returns:
            Dictionary berisi reset results
        """
        try:
            self.update_status("🔄 Resetting environment...", 'info')
            
            # Reset handlers
            self._status = 'idle'
            self._progress = 0.0
            self._current_stage = SetupStage.INIT
            
            # Reset config ke defaults
            self.reset_to_defaults()
            
            # Delegate ke setup handler
            result = self.setup_handler.reset_environment()
            
            if result.get('status', False):
                self.update_status("✅ Environment reset completed", 'success')
            else:
                self.update_status("❌ Environment reset failed", 'error')
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error resetting environment: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.update_status(error_msg, 'error')
            
            return {
                'status': False,
                'error': str(e),
                'message': error_msg
            }
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information.
        
        Returns:
            Dictionary berisi environment info
        """
        try:
            # Get info dari setup handler
            setup_info = self.setup_handler.get_environment_info()
            
            # Combine dengan handler info
            info = {
                'handler_info': {
                    'current_stage': self._current_stage.name,
                    'status': self._status,
                    'progress': self._progress
                },
                'setup_info': setup_info
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"❌ Error getting environment info: {str(e)}")
            return {
                'handler_info': {
                    'current_stage': self._current_stage.name,
                    'status': 'error',
                    'progress': 0.0,
                    'error': str(e)
                },
                'setup_info': {}
            }