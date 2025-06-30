"""
File: smartcash/ui/handlers/config_handlers.py
Deskripsi: ConfigHandler with integrated UILogger and error handling
"""

from typing import Dict, Any, Optional, Callable, TypeVar, List
from abc import ABC, abstractmethod
from functools import partial
from dataclasses import dataclass, field
from datetime import datetime

from smartcash.common.config.manager import get_config_manager
from smartcash.ui.utils.ui_logger import get_module_logger
from smartcash.ui.handlers.error_handler import handle_ui_errors, create_error_response, ErrorContext
from smartcash.ui.config_cell.managers.shared_config_manager import (
    get_shared_config_manager,
    subscribe_to_config,
    broadcast_config_update
)

# Type variable for generic return types
T = TypeVar('T')

@dataclass
class ConfigState:
    """Simple configuration state management."""
    config: Dict[str, Any] = field(default_factory=dict)
    last_updated: Optional[datetime] = None
    
    def update(self, new_config: Dict[str, Any]) -> None:
        self.config = new_config.copy()
        self.last_updated = datetime.now()
    
    def get(self) -> Dict[str, Any]:
        return self.config.copy()


class ConfigHandler(ABC):
    """ConfigHandler with shared configuration management and proper lifecycle handling.
    
    Features:
    - Shared configuration across components using SharedConfigManager
    - Thread-safe operations
    - Automatic config change notifications
    - Fallback to local config when shared config is not available
    """
    
    @handle_ui_errors(error_component_title="Config Error", log_error=True)
    def __init__(self, module_name: str, parent_module: str = None, use_shared_config: bool = True):
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        self.use_shared_config = use_shared_config
        
        # Initialize logger with module-level logging
        self.logger = get_module_logger(f"smartcash.ui.{self.full_module_name}.config")
        self.config_manager = get_config_manager()
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Initialize shared config manager if enabled
        self.shared_manager = None
        self._unsubscribe = None
        if self.use_shared_config and self.parent_module:
            try:
                self.shared_manager = get_shared_config_manager(self.parent_module)
                self._unsubscribe = subscribe_to_config(
                    self.parent_module, 
                    self.module_name,
                    self._on_shared_config_updated
                )
                self.logger.debug(f"Using shared config for {self.full_module_name}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize shared config: {e}",
                    exc_info=True
                )
        
        # Local config state
        self._config_state = ConfigState()
        
        self.logger.debug(f"Initialized ConfigHandler for {self.full_module_name}")
    
    def __del__(self):
        """Clean up resources."""
        if self._unsubscribe:
            self._unsubscribe()
    
    def _on_shared_config_updated(self, config: Dict[str, Any]) -> None:
        """Handle updates from shared config manager."""
        if not self.use_shared_config:
            return
            
        self.logger.debug(f"Received shared config update for {self.module_name}")
        self._config_state.update(config)
        self._notify_callbacks(config)
        
    @abstractmethod
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Ekstrak konfigurasi dari komponen UI"""
        pass
        
    @abstractmethod
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari konfigurasi yang dimuat"""
        pass
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dari defaults.py untuk reset scenarios"""
        try:
            module_path = (f"smartcash.ui.{self.parent_module}.{self.module_name}.handlers.defaults" 
                          if hasattr(self, 'parent_module') and self.parent_module 
                          else f"smartcash.ui.{self.module_name}.handlers.defaults")
            
            module = __import__(module_path, fromlist=['DEFAULT_CONFIG'])
            default_config = getattr(module, 'DEFAULT_CONFIG', {})
            
            return (self.logger.info(f"📋 Loaded defaults.py for {self.module_name}") or default_config 
                   if default_config else self._get_fallback_config())
            
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"🔍 defaults.py not found for {self.module_name}: {str(e)}")
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Fallback minimal default structure"""
        return {'module_name': self.module_name, 'version': '1.0.0', 'created_by': 'SmartCash', 'settings': {}}
    
    @handle_ui_errors(
        error_component_title="Config Load Error",
        log_error=True,
        return_type=dict
    )
    def load_config(self, config_name: Optional[str] = None, use_base_config: bool = True) -> Dict[str, Any]:
        """Load config with fallback to shared config and base_config.yaml.
        
        Priority order:
        1. Specific config from file
        2. Shared config (if enabled)
        3. Base config from file
        4. Default config
        """
        config_name = config_name or f"{self.module_name}_config"
        
        # Try to load from shared config first if enabled
        if self.use_shared_config and self.shared_manager:
            try:
                if shared_config := self.shared_manager.get_config(self.module_name):
                    self.logger.info(f"Loaded shared config for {self.module_name}")
                    self._config_state.update(shared_config)
                    return shared_config
            except Exception as e:
                self.logger.warning(
                    f"Failed to load shared config: {e}",
                    exc_info=True
                )
        
        # Fallback to file-based config
        try:
            # Try to load specific config
            if specific_config := self.config_manager.get_config(config_name):
                self.logger.info(f"Loaded config: {config_name}")
                resolved = self._resolve_config_inheritance(specific_config, config_name)
                self._config_state.update(resolved)
                return resolved
            
            # Fallback to base config if enabled
            if use_base_config and (base_config := self.config_manager.get_config('base_config')):
                self.logger.info(f"Using base_config.yaml for {config_name}")
                resolved = self._resolve_config_inheritance(base_config, 'base_config')
                self._config_state.update(resolved)
                return resolved
                
        except Exception as e:
            self.logger.error(
                f"Error loading config: {e}",
                exc_info=True
            )
        
        # Final fallback to defaults
        self.logger.warning(f"Using defaults for {config_name}")
        default_config = self.get_default_config()
        self._config_state.update(default_config)
        return default_config
    
    @handle_ui_errors(
        error_component_title="Config Save Error",
        log_error=True,
        return_type=bool
    )
    def save_config(self, ui_components: Dict[str, Any], config_name: Optional[str] = None, 
                   update_shared: bool = True) -> bool:
        """Save config with lifecycle hooks and proper error handling.
        
        Args:
            ui_components: Dictionary of UI components
            config_name: Optional custom config name
            update_shared: Whether to update shared config if enabled
            
        Returns:
            bool: True if save was successful
        """
        self.before_save(ui_components)
        
        # Extract and validate config
        config = self.extract_config(ui_components)
        if not config:
            raise ValueError("Failed to extract configuration from UI components")
        
        # Update local state
        self._config_state.update(config)
        
        # Update shared config if enabled
        if self.use_shared_config and update_shared and self.parent_module:
            try:
                broadcast_config_update(
                    parent_module=self.parent_module,
                    module_name=self.module_name,
                    config=config,
                    persist=True
                )
                self.logger.info(f"Updated shared config for {self.module_name}")
                return self._handle_save_success(ui_components, config)
            except Exception as e:
                self.logger.error(
                    f"Failed to update shared config: {e}",
                    exc_info=True
                )
                # Fall through to file-based save
        
        # Fallback to file-based save
        config_name = config_name or f"{self.module_name}_config"
        try:
            success = self.config_manager.save_config(config, config_name)
            if success:
                return self._handle_save_success(ui_components, config)
            
            self._handle_save_failure(ui_components, "Failed to save configuration to file")
            return False
            
        except Exception as e:
            self._handle_save_failure(ui_components, f"Error saving config: {str(e)}")
            return False
    
    @handle_ui_errors(
        error_component_title="Config Reset Error",
        log_error=True,
        return_type=bool
    )
    def reset_config(self, ui_components: Dict[str, Any], config_name: Optional[str] = None) -> bool:
        """Reset config to defaults with proper error handling."""
        self.before_reset(ui_components)
        
        # Get and validate default config
        default_config = self.get_default_config()
        if not default_config:
            raise ValueError("Failed to get default configuration")
        
        # Update UI and save
        self.update_ui(ui_components, default_config)
        config_name = config_name or f"{self.module_name}_config"
        success = self.config_manager.save_config(default_config, config_name)
        
        # Handle success/failure
        if success:
            return self._handle_reset_success(ui_components, default_config)
        
        self._handle_reset_failure(ui_components, "Failed to reset configuration")
        return False
    
    def _execute_callbacks(self, config: Dict[str, Any], operation: str) -> None:
        """Safely execute callbacks with error handling.
        
        Args:
            config: The configuration to pass to callbacks
            operation: Operation name for logging (e.g., 'save' or 'reset')
        """
        self._notify_callbacks(config, operation)
    
    def _notify_callbacks(self, config: Dict[str, Any], operation: str = 'update') -> None:
        """Notify all registered callbacks with the new config.
        
        Args:
            config: The configuration to pass to callbacks
            operation: Operation name for logging
        """
        for callback in self.callbacks:
            try:
                callback(config)
            except Exception as e:
                self.logger.error(
                    f"Error in {operation} callback: {str(e)}", 
                    exc_info=True,
                    extra={
                        'component': self.__class__.__name__,
                        'operation': f'{operation}_callback'
                    }
                )
    
    def _handle_success(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                       callback_type: str) -> None:
        """Common success handler for save/reset operations.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration data
            callback_type: Either 'save' or 'reset'
        """
        ui_components['config'] = config
        getattr(self, f'after_{callback_type}_success')(ui_components, config)
        self._execute_callbacks(config, callback_type)
    
    def _handle_failure(self, ui_components: Dict[str, Any], error: str, 
                       callback_type: str) -> None:
        """Common failure handler for save/reset operations.
        
        Args:
            ui_components: Dictionary of UI components
            error: Error message
            callback_type: Either 'save' or 'reset'
        """
        getattr(self, f'after_{callback_type}_failure')(ui_components, error)
    
    def _handle_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Handle save success with callback execution"""
        self._handle_success(ui_components, config, 'save')
    
    def _handle_save_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Handle save failure"""
        self._handle_failure(ui_components, error, 'save')
    
    def _handle_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Handle reset success with callback execution"""
        self._handle_success(ui_components, config, 'reset')
    
    def _handle_reset_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Handle reset failure"""
        self._handle_failure(ui_components, error, 'reset')
    
    @handle_ui_errors(
        error_component_title="Config Inheritance Error",
        log_error=True,
        return_type=dict
    )
    def _resolve_config_inheritance(self, config: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Resolve config inheritance with _base_ support."""
        if not config or '_base_' not in config:
            return config
        
        # Handle single base config as string
        base_configs = [config.pop('_base_')] if isinstance(
            base_configs := config.pop('_base_'), str
        ) else base_configs
        
        # Merge all base configs
        merged_config = {}
        for base_name in base_configs:
            if base_config := self.config_manager.get_config(base_name):
                merged_config.update(self._resolve_config_inheritance(base_config, base_name))
            else:
                self.logger.warning(f"Base config not found: {base_name}")
        
        # Apply current config on top of merged base configs
        merged_config.update(config)
        self.logger.debug(f"Resolved inheritance for {config_name} with {len(base_configs)} base configs")
        return merged_config
    
    # Lifecycle hooks
    def before_save(self, ui_components: Dict[str, Any]) -> None:
        """Hook called before saving configuration."""
        self._clear_ui_outputs(ui_components)
        self._update_status_panel(ui_components, "Saving configuration...", "info")
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook called after successful save."""
        self._update_status_panel(ui_components, "Configuration saved successfully", "success")
        self.logger.info(f"Configuration {self.module_name} saved successfully")
    
    def after_save_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Hook called when save fails."""
        self._update_status_panel(ui_components, f"Failed to save: {error}", "error")
        self.logger.error(f"Error saving configuration: {error}", exc_info=True)
    
    def before_reset(self, ui_components: Dict[str, Any]) -> None:
        """Hook called before resetting configuration."""
        self._clear_ui_outputs(ui_components)
        self._reset_progress_bars(ui_components)
        self._update_status_panel(ui_components, "Resetting configuration...", "info")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook called after successful reset."""
        self._update_status_panel(ui_components, "Configuration reset successfully", "success")
        self.logger.info(f"Configuration {self.module_name} reset successfully")
    
    def after_reset_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Hook called when reset fails."""
        self._update_status_panel(ui_components, f"Failed to reset: {error}", "error")
        self.logger.error(f"Error resetting configuration: {error}", exc_info=True)
    
    # Helper methods with proper error handling
    def _clear_ui_outputs(self, ui_components: Dict[str, Any]) -> None:
        """Clear UI outputs with safe widget access."""
        output_keys = ['log_output', 'status', 'confirmation_area']
        
        for key in output_keys:
            widget = ui_components.get(key)
            if widget and hasattr(widget, 'clear_output'):
                try:
                    widget.clear_output(wait=True)
                except Exception as e:
                    self.logger.debug(f"Error clearing {key}: {str(e)}", exc_info=True)
    
    def _reset_progress_bars(self, ui_components: Dict[str, Any]) -> None:
        """Reset progress bars with safe widget access."""
        progress_keys = ['progress_bar', 'progress_container', 'current_progress', 'progress_tracker']
        
        for key in progress_keys:
            widget = ui_components.get(key)
            if widget:
                try:
                    # Try to hide widget
                    if hasattr(widget, 'layout'):
                        widget.layout.visibility = 'hidden'
                        widget.layout.display = 'none'
                    
                    # Try to reset value
                    if hasattr(widget, 'value'):
                        widget.value = 0
                    
                    # Try to reset progress tracker
                    if hasattr(widget, 'reset'):
                        widget.reset()
                        
                except Exception as e:
                    self.logger.debug(f"Error resetting {key}: {str(e)}", exc_info=True)
    
    def _update_status_panel(self, ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
        """Update status panel with safe fallback"""
        try:
            if 'status_panel' in ui_components and hasattr(ui_components['status_panel'], 'update'):
                ui_components['status_panel'].update(
                    create_error_response(
                        error_message=message,
                        title="Status Update",
                        error_type=status_type,
                        include_traceback=False
                    )
                )
            elif 'logger' in ui_components:
                log_method = getattr(ui_components['logger'], status_type, ui_components['logger'].info)
                log_method(f"Status: {message}")
            else:
                print(f"[{status_type.upper()}] {message}")
        except Exception as e:
            self.logger.error(
                f"Failed to update status panel: {str(e)}",
                exc_info=True,
                extra={
                    'component': self.__class__.__name__,
                    'operation': 'update_status_panel'
                }
            )
    
    # Callback management dengan one-liner checks
    def add_callback(self, cb: Callable) -> None:
        """Add callback jika belum ada"""
        cb not in self.callbacks and self.callbacks.append(cb)
    
    def remove_callback(self, cb: Callable) -> None:
        """Remove callback jika ada"""
        cb in self.callbacks and self.callbacks.remove(cb)
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """Get config summary untuk display"""
        return f"📊 {self.module_name}: {len(config)} konfigurasi dimuat"
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config - override untuk custom validation"""
        return {'valid': True, 'errors': []}


# Factory functions
def create_config_handler(
    module_name: str, 
    parent_module: str = None, 
    use_shared_config: bool = True,
    **kwargs
) -> ConfigHandler:
    """Create a new ConfigHandler instance with shared config support.
    
    Args:
        module_name: Name of the module
        parent_module: Optional parent module name for shared config
        use_shared_config: Whether to enable shared config
        **kwargs: Additional arguments for ConfigHandler
        
    Returns:
        ConfigHandler: New instance
    """
    return ConfigHandler(
        module_name=module_name,
        parent_module=parent_module,
        use_shared_config=use_shared_config,
        **kwargs
    )

def get_or_create_handler(
    ui_components: Dict[str, Any], 
    module_name: str, 
    parent_module: str = None,
    **kwargs
) -> ConfigHandler:
    """Get existing handler or create a new one with shared config support.
    
    Args:
        ui_components: Dictionary of UI components
        module_name: Name of the module
        parent_module: Optional parent module name
        **kwargs: Additional arguments for ConfigHandler
        
    Returns:
        ConfigHandler: Existing or new instance
    """
    if 'config_handler' in ui_components:
        return ui_components['config_handler']
        
    handler = create_config_handler(
        module_name=module_name,
        parent_module=parent_module,
        **kwargs
    )
    ui_components['config_handler'] = handler
    return handler