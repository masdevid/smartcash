"""
File: smartcash/ui/handlers/config_handlers.py
Description: ConfigHandler with shared configuration management, BaseHandler integration,
and support for both persistent and non-persistent configuration handling.
"""

from typing import Dict, Any, Optional, Callable, TypeVar, List
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from smartcash.common.config.manager import get_config_manager
from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.handlers.error_handler import handle_ui_errors
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


class ConfigHandler(BaseHandler):
    """ConfigHandler with shared configuration management and proper lifecycle handling.
    
    Features:
    - Shared configuration across components using SharedConfigManager
    - Thread-safe operations with proper lifecycle hooks
    - Automatic config change notifications
    - Fallback to local config when shared config is not available
    - Support for non-persistent configuration (in-memory only)
    - Inherits from BaseHandler for centralized functionality:
      - Consistent logging and log redirection to UI
      - Standardized error handling with UI feedback
      - Confirmation dialog utilities
      - Button state management (enable/disable)
      - Status panel updates
      - Progress tracker integration
      - UI component clearing and reset
    
    This handler standardizes API response success checking using the 'status' key
    for consistency across the application.
    
    For handlers that don't require persistence (where load/save operations are irrelevant),
    set persistence_enabled=False when initializing the handler. This will make the handler
    operate in memory-only mode without attempting to load from or save to disk.
    
    When persistence_enabled=False:
    - extract_config and update_ui methods become optional
    - If extract_config is not implemented, in-memory state is used
    - If update_ui is not implemented, UI updates are skipped
    - No file operations are performed
    - Shared config operations are skipped
    
    This allows for simpler handlers that only need to maintain state in memory
    without the overhead of implementing UI extraction/update methods when they
    aren't needed.
    """
    
    @handle_ui_errors(error_component_title="Config Error", log_error=True)
    def __init__(self, module_name: str, parent_module: str = None, use_shared_config: bool = True,
                 persistence_enabled: bool = True):
        # Initialize BaseHandler first
        super().__init__(module_name, parent_module)
        
        # Config-specific attributes
        self.use_shared_config = use_shared_config
        self.persistence_enabled = persistence_enabled
        self.config_manager = get_config_manager() if persistence_enabled else None
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Initialize shared config manager if enabled and persistence is enabled
        self.shared_manager = None
        self._unsubscribe = None
        if self.persistence_enabled and self.use_shared_config and self.parent_module:
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
        
        # Local config state (always available even for non-persistent config)
        self._config_state = ConfigState()
        
        self.logger.debug(
            f"Initialized ConfigHandler for {self.full_module_name} " +
            f"(persistence {'enabled' if self.persistence_enabled else 'disabled'})"
        )
    
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
        """Ekstrak konfigurasi dari komponen UI
        
        Note: Please create a dedicated Single Responsibility Principle (SRP) file for this method,
        e.g. `config_extractor.py` to avoid cluttering this file with unrelated code.
        """
        pass
        
    @abstractmethod
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari konfigurasi yang dimuat
        
        Note: Please create a dedicated Single Responsibility Principle (SRP) file for this method,
        e.g. `config_updater.py` to avoid cluttering this file with unrelated code.
        """
        pass
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dari defaults.py untuk reset scenarios
        
        Note: Please create a dedicated file `defaults.py` for this hardcoded configuration
        without cluttering with unrelated functions.
        """
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
        1. Specific config from file (if persistence enabled)
        2. Shared config (if enabled and persistence enabled)
        3. Base config from file (if persistence enabled)
        4. Default config
        
        For non-persistent handlers, this will always return the default config or
        the current in-memory config state.
        """
        # If persistence is disabled, just return current state or default config
        if not self.persistence_enabled:
            if self._config_state.config:
                self.logger.debug(f"Using in-memory config for {self.module_name} (persistence disabled)")
                return self._config_state.config
            else:
                default_config = self.get_default_config()
                self.logger.debug(f"Using default config for {self.module_name} (persistence disabled)")
                self._config_state.update(default_config)
                return default_config
        
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
                if use_base_config:
                    specific_config = self._resolve_config_inheritance(specific_config, config_name)
                self.logger.info(f"Loaded specific config: {config_name}")
                self._config_state.update(specific_config)
                return specific_config
                
            # Try to load base config
            if use_base_config:
                if base_config := self.config_manager.get_config('base_config'):
                    self.logger.info("Loaded base config")
                    self._config_state.update(base_config)
                    return base_config
        except Exception as e:
            self.logger.warning(
                f"Failed to load file-based config: {e}",
                exc_info=True
            )
            
        # Use default config as last resort
        default_config = self.get_default_config()
        self.logger.info("Using default config")
        self._config_state.update(default_config)
        return default_config
    
    def _try_extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Try to extract config from UI components, respecting persistence setting.
        
        For persistent handlers, extract_config is required.
        For non-persistent handlers, extract_config is optional.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Dict[str, Any]: Extracted or current config
            
        Raises:
            ValueError: If extraction fails for persistent handlers
        """
        if not self.persistence_enabled:
            # For non-persistent handlers, extract_config is optional
            extract_method = getattr(self, 'extract_config', None)
            if callable(extract_method):
                try:
                    config = extract_method(ui_components)
                    if config:
                        return config
                except NotImplementedError:
                    self.logger.debug(f"Using current config for {self.module_name} (extract_config not implemented)")
            else:
                self.logger.debug(f"Using current config for {self.module_name} (extract_config not implemented)")
            
            # Use current config if extract_config is not implemented or fails
            return self._config_state.get() or {}
        else:
            # For persistent handlers, extract_config is required
            config = self.extract_config(ui_components)
            if not config:
                raise ValueError("Failed to extract configuration from UI components")
            return config
    
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
            
        For non-persistent handlers, this will only update the in-memory state
        and will not attempt to save to disk or shared config.
        
        For non-persistent handlers, extract_config is optional. If not implemented,
        the current in-memory config will be used.
        """
        self.before_save(ui_components)
        
        # Try to extract config based on persistence setting
        config = self._try_extract_config(ui_components)
        
        # Always update local state
        self._config_state.update(config)
        
        # For non-persistent handlers, just update in-memory state and return
        if not self.persistence_enabled:
            self.logger.debug(f"Updated in-memory config for {self.module_name} (persistence disabled)")
            return self._handle_save_success(ui_components, config)
        
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
        
        # Save to file
        config_name = config_name or f"{self.module_name}_config"
        try:
            self.config_manager.save_config(config_name, config)
            self.logger.info(f"Saved config to {config_name}.yaml")
            return self._handle_save_success(ui_components, config)
        except Exception as e:
            error_msg = f"Failed to save config: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return self._handle_save_failure(ui_components, error_msg)
    
    @handle_ui_errors(
        error_component_title="Config Reset Error",
        log_error=True,
        return_type=bool
    )
    def reset_config(self, ui_components: Dict[str, Any], config_name: Optional[str] = None, 
                    update_shared: bool = True) -> bool:
        """Reset config to defaults with lifecycle hooks.
        
        Args:
            ui_components: Dictionary of UI components
            config_name: Optional custom config name
            update_shared: Whether to update shared config if enabled
            
        Returns:
            bool: True if reset was successful
            
        For non-persistent handlers, this will only reset the in-memory state
        and will not attempt to save to disk or shared config.
        
        For non-persistent handlers, update_ui is optional. If implemented, it will
        be called to update the UI with the default configuration.
        """
        self.before_reset(ui_components)
        
        # Get default config
        default_config = self.get_default_config()
        if not default_config:
            error_msg = "Failed to get default configuration"
            self.logger.error(error_msg)
            self._handle_reset_failure(ui_components, error_msg)
            return False
            
        # Always update local state
        self._config_state.update(default_config)
        
        # For non-persistent handlers, just update in-memory state and return
        if not self.persistence_enabled:
            self.logger.debug(f"Reset in-memory config for {self.module_name} (persistence disabled)")
            self._handle_reset_success(ui_components, default_config)
            return True
        
        # Update shared config if enabled
        if self.use_shared_config and update_shared and self.parent_module:
            try:
                broadcast_config_update(
                    parent_module=self.parent_module,
                    module_name=self.module_name,
                    config=default_config,
                    persist=True
                )
                self.logger.info(f"Reset shared config for {self.module_name}")
                self._handle_reset_success(ui_components, default_config)
                return True
            except Exception as e:
                self.logger.error(
                    f"Failed to reset shared config: {e}",
                    exc_info=True
                )
                # Fall through to file-based save
        
        # Save default config to file
        config_name = config_name or f"{self.module_name}_config"
        try:
            self.config_manager.save_config(config_name, default_config)
            self.logger.info(f"Reset config saved to {config_name}.yaml")
            self._handle_reset_success(ui_components, default_config)
            return True
        except Exception as e:
            error_msg = f"Failed to reset config: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._handle_reset_failure(ui_components, error_msg)
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
            
        For non-persistent handlers, update_ui is optional. If implemented, it will
        be called to update the UI with the configuration.
        """
        ui_components['config'] = config
        
        # Try to update UI based on persistence setting
        self._try_update_ui(ui_components, config)
        
        # Call lifecycle hook and execute callbacks
        getattr(self, f'after_{callback_type}_success')(ui_components, config)
        self._execute_callbacks(config, callback_type)
    
    def _try_update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Try to update UI components with config data, respecting persistence setting.
        
        For persistent handlers, update_ui is required.
        For non-persistent handlers, update_ui is optional.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration data to apply to UI
        """
        if self.persistence_enabled or not hasattr(self, 'persistence_enabled'):
            # For persistent handlers, update_ui is required
            self.update_ui(ui_components, config)
        else:
            # For non-persistent handlers, update_ui is optional
            update_method = getattr(self, 'update_ui', None)
            if callable(update_method):
                try:
                    update_method(ui_components, config)
                except NotImplementedError:
                    self.logger.debug(f"UI not updated for {self.module_name} (update_ui not implemented)")
    
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
        self.clear_ui_outputs(ui_components)
        self.update_status_panel(ui_components, "Saving configuration...", "info")
    
    def before_reset(self, ui_components: Dict[str, Any]) -> None:
        """Hook called before resetting configuration."""
        self.clear_ui_outputs(ui_components)
        self.update_status_panel(ui_components, "Resetting configuration...", "info")
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook called after successful save."""
        self.update_status_panel(ui_components, "Configuration saved successfully", "success")
        self.logger.info(f"Configuration {self.module_name} saved successfully")
    
    def after_save_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Hook called when save fails."""
        self.update_status_panel(ui_components, f"Failed to save: {error}", "error")
        self.logger.error(f"Error saving configuration: {error}", exc_info=True)
    
    def before_reset(self, ui_components: Dict[str, Any]) -> None:
        """Hook called before resetting configuration."""
        self.clear_ui_outputs(ui_components)
        self.reset_progress_bars(ui_components)
        self.update_status_panel(ui_components, "Resetting configuration...", "info")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook called after successful reset."""
        self.update_status_panel(ui_components, "Configuration reset successfully", "success")
        self.logger.info(f"Configuration {self.module_name} reset successfully")
    
    def after_reset_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Hook called when reset fails."""
        self.update_status_panel(ui_components, f"Failed to reset: {error}", "error")
        self.logger.error(f"Error resetting configuration: {error}", exc_info=True)
    
    # Use BaseHandler's UI output clearing and progress bar reset methods instead of custom implementation
    
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