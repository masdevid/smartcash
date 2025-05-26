"""
File: smartcash/ui/utils/common_initializer.py
Deskripsi: Base class untuk UI initializers dengan shared functionality dan error handling
"""

from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod
import ipywidgets as widgets
from IPython.display import display
import datetime
import logging

from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.utils.button_state_manager import get_button_state_manager


class CommonInitializer(ABC):
    """
    Base class untuk UI initializers dengan shared functionality.
    
    Provides common patterns for:
    - Configuration management
    - Logger setup
    - Error handling
    - UI component management
    - Handler registration
    """
    
    def __init__(self, module_name: str, logger_namespace: str):
        """
        Initialize common initializer.
        
        Args:
            module_name: Name of the module (e.g., 'dataset_augmentation')
            logger_namespace: Logger namespace for this module
        """
        self.module_name = module_name
        self.logger_namespace = logger_namespace
        self._initialized = False
        self._cached_components = None
        self._initialization_timestamp = None
        
        # Setup basic logger
        self.logger = get_logger(logger_namespace)
    
    def initialize(self, env=None, config=None, force_refresh=False, **kwargs) -> Any:
        """
        Main initialization method with caching and error recovery.
        
        Args:
            env: Environment context
            config: Custom configuration
            force_refresh: Force refresh UI components
            **kwargs: Additional initialization parameters
            
        Returns:
            UI widget or components
        """
        # Return cached if available and not forcing refresh
        if self._initialized and self._cached_components and not force_refresh:
            return self._get_cached_or_refresh(config)
        
        try:
            # Setup log suppression for clean initialization
            self._setup_log_suppression()
            
            # Get merged configuration
            merged_config = self._get_merged_config(config)
            
            # Create UI components
            ui_components = self._create_ui_components_safe(merged_config, env, **kwargs)
            if not ui_components:
                return self._create_error_fallback_ui("Failed to create UI components")
            
            # Setup logger bridge
            logger_bridge = self._setup_logger_bridge_safe(ui_components)
            self._enhance_components_with_logger(ui_components, logger_bridge)
            
            # Setup handlers and additional functionality
            ui_components = self._setup_handlers_comprehensive(ui_components, merged_config, env, **kwargs)
            
            # Validate setup
            validation_result = self._validate_setup(ui_components)
            if not validation_result['valid']:
                return self._create_error_fallback_ui(validation_result['message'])
            
            # Finalize setup
            self._finalize_setup(ui_components, merged_config)
            
            # Cache components
            self._cached_components = ui_components
            self._initialized = True
            self._initialization_timestamp = self._get_timestamp()
            
            # Log success
            if logger_bridge and 'log_output' in ui_components:
                with ui_components['log_output']:
                    logger_bridge.success(f"✅ {self.module_name} UI berhasil diinisialisasi")
            
            return self._get_return_value(ui_components)
            
        except Exception as e:
            self.logger.error(f"❌ Error inisialisasi {self.module_name}: {str(e)}")
            return self._create_error_fallback_ui(f"Initialization error: {str(e)}")
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """
        Create UI components specific to the module.
        
        Args:
            config: Configuration dictionary
            env: Environment context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of UI components
        """
        pass
    
    @abstractmethod
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """
        Setup handlers specific to the module.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
            env: Environment context
            **kwargs: Additional parameters
            
        Returns:
            Updated UI components dictionary
        """
        pass
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for the module.
        
        Returns:
            Default configuration dictionary
        """
        pass
    
    @abstractmethod
    def _get_critical_components(self) -> List[str]:
        """
        Get list of critical component keys that must exist.
        
        Returns:
            List of critical component keys
        """
        pass
    
    # Common implementation methods
    
    def _get_merged_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get merged configuration with safe error handling."""
        try:
            config_manager = get_config_manager()
            
            # Load saved config
            saved_config = {}
            try:
                if hasattr(config_manager, 'get_config'):
                    saved_config = config_manager.get_config(self.module_name) or {}
            except Exception:
                pass
            
            # Start with default config
            merged_config = self._get_default_config()
            
            # Merge with saved config
            if saved_config:
                merged_config.update(saved_config)
            
            # Merge with parameter config
            if config:
                merged_config.update(config)
            
            return merged_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error merging config, using default: {str(e)}")
            return self._get_default_config()
    
    def _create_ui_components_safe(self, config: Dict[str, Any], env=None, **kwargs) -> Optional[Dict[str, Any]]:
        """Create UI components with comprehensive error handling."""
        try:
            return self._create_ui_components(config, env, **kwargs)
        except Exception as e:
            self.logger.error(f"❌ Error creating UI components: {str(e)}")
            # Try with minimal config
            try:
                minimal_config = self._get_default_config()
                return self._create_ui_components(minimal_config, env, **kwargs)
            except Exception as e2:
                self.logger.error(f"❌ Error with minimal config: {str(e2)}")
                return None
    
    def _setup_logger_bridge_safe(self, ui_components: Dict[str, Any]) -> Optional[Any]:
        """Setup logger bridge with error handling."""
        try:
            return create_ui_logger_bridge(ui_components, self.logger_namespace)
        except Exception as e:
            self.logger.warning(f"⚠️ Logger bridge setup failed: {str(e)}")
            return None
    
    def _enhance_components_with_logger(self, ui_components: Dict[str, Any], logger_bridge) -> None:
        """Enhance UI components with logger and metadata."""
        ui_components['logger'] = logger_bridge or self.logger
        ui_components['logger_namespace'] = self.logger_namespace
        ui_components['module_name'] = self.module_name
        ui_components[f'{self.module_name}_initialized'] = True
    
    def _setup_handlers_comprehensive(self, ui_components: Dict[str, Any], 
                                    config: Dict[str, Any], 
                                    env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers with comprehensive error recovery."""
        
        # Setup shared button state manager
        try:
            if 'button_state_manager' not in ui_components:
                button_state_manager = get_button_state_manager(ui_components)
                ui_components['button_state_manager'] = button_state_manager
        except Exception as e:
            ui_components['logger'].warning(f"⚠️ Button state manager setup failed: {str(e)}")
        
        # Setup module-specific handlers
        try:
            ui_components = self._setup_module_handlers(ui_components, config, env, **kwargs)
        except Exception as e:
            ui_components['logger'].error(f"❌ Module handlers setup failed: {str(e)}")
        
        # Setup common button handlers
        try:
            self._setup_common_button_handlers(ui_components)
        except Exception as e:
            ui_components['logger'].warning(f"⚠️ Common button handlers setup failed: {str(e)}")
        
        # Store config
        ui_components['config'] = config
        
        return ui_components
    
    def _setup_common_button_handlers(self, ui_components: Dict[str, Any]) -> None:
        """Setup common button handlers with safe checking."""
        logger = ui_components.get('logger', self.logger)
        
        # Debug: Log available buttons
        available_buttons = [k for k in ui_components.keys() if 'button' in k and ui_components[k] is not None]
        logger.debug(f"🔍 Available buttons: {available_buttons}")
        
        # Common button patterns
        common_buttons = {
            'reset_button': self._handle_reset_button,
            'save_button': self._handle_save_button,
            'cleanup_button': self._handle_cleanup_button
        }
        
        for button_key, handler in common_buttons.items():
            button = ui_components.get(button_key)
            if button is not None and hasattr(button, 'on_click'):
                button.on_click(lambda b, h=handler: h(ui_components, b))
                logger.debug(f"✅ {button_key} handler registered")
            else:
                logger.debug(f"⚠️ {button_key} tidak ditemukan atau tidak functional")
    
    def _validate_setup(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validate setup and critical components."""
        
        # Check critical components
        critical_components = self._get_critical_components()
        missing_critical = [comp for comp in critical_components if comp not in ui_components]
        
        if missing_critical:
            return {
                'valid': False,
                'message': f"Critical components missing: {', '.join(missing_critical)}"
            }
        
        # Additional validation can be implemented by subclasses
        additional_validation = self._additional_validation(ui_components)
        if not additional_validation.get('valid', True):
            return additional_validation
        
        return {'valid': True, 'message': 'Setup validation passed'}
    
    def _additional_validation(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Additional validation that can be overridden by subclasses."""
        return {'valid': True}
    
    def _finalize_setup(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Finalize setup with common configurations."""
        ui_components['module_initialized'] = True
        ui_components['initialization_timestamp'] = self._get_timestamp()
        ui_components['config'] = config
    
    def _get_return_value(self, ui_components: Dict[str, Any]) -> Any:
        """Get return value from UI components. Override if needed."""
        return ui_components.get('ui', ui_components)
    
    def _get_cached_or_refresh(self, config=None) -> Any:
        """Get cached UI atau refresh dengan config baru."""
        try:
            if not self._cached_components:
                return self.initialize(config=config, force_refresh=True)
            
            # Update config if provided
            if config:
                self._update_cached_config(config)
            
            return self._get_return_value(self._cached_components)
            
        except Exception as e:
            return self._create_error_fallback_ui(f"Cache refresh error: {str(e)}")
    
    def _update_cached_config(self, new_config: Dict[str, Any]) -> None:
        """Update cached UI components with new config. Override if needed."""
        if not self._cached_components:
            return
        
        try:
            # Basic config update - can be overridden by subclasses
            self._cached_components['config'].update(new_config)
        except Exception:
            pass
    
    def _setup_log_suppression(self) -> None:
        """Setup log suppression for clean initialization."""
        loggers_to_suppress = [
            'smartcash.common.environment',
            'smartcash.common.config.manager',
            'smartcash.common.logger',
            'smartcash.ui.utils.logger_bridge',
            'requests', 'urllib3', 'http.client',
            'ipywidgets', 'traitlets'
        ]
        
        for logger_name in loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
    
    def _create_error_fallback_ui(self, error_message: str):
        """Create error fallback UI with actionable information."""
        error_html = f"""
        <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffc107; 
                    border-radius: 8px; color: #856404; margin: 10px 0; max-width: 800px;">
            <h4 style="color: #721c24; margin-top: 0;">⚠️ Error Inisialisasi {self.module_name}</h4>
            <div style="margin: 15px 0;">
                <strong>Error Detail:</strong><br>
                <code style="background: #f8f9fa; padding: 5px; border-radius: 3px; font-size: 12px;">
                    {error_message}
                </code>
            </div>
            <div style="margin: 15px 0;">
                <strong>🔧 Solusi yang Bisa Dicoba:</strong>
                <ol style="margin: 10px 0; padding-left: 20px;">
                    <li>Restart kernel dan jalankan ulang cell</li>
                    <li>Clear output semua cell dan jalankan dari awal</li>
                    <li>Periksa koneksi internet dan dependencies</li>
                    <li>Pastikan tidak ada error pada cell-cell sebelumnya</li>
                </ol>
            </div>
            <div style="margin: 15px 0; padding: 10px; background: #e8f4fd; border-radius: 5px;">
                <strong>💡 Quick Fix:</strong> Jalankan <code>reset_{self.module_name.replace('_', '_').lower()}_module()</code> kemudian coba lagi
            </div>
        </div>
        """
        
        return widgets.HTML(error_html)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Common button handlers (can be overridden)
    
    def _handle_reset_button(self, ui_components: Dict[str, Any], button) -> None:
        """Default reset button handler."""
        logger = ui_components.get('logger', self.logger)
        logger.info("🔄 Reset button clicked")
        # Implementation can be overridden by subclasses
    
    def _handle_save_button(self, ui_components: Dict[str, Any], button) -> None:
        """Default save button handler."""
        logger = ui_components.get('logger', self.logger)
        logger.info("💾 Save button clicked")
        # Implementation can be overridden by subclasses
    
    def _handle_cleanup_button(self, ui_components: Dict[str, Any], button) -> None:
        """Default cleanup button handler."""
        logger = ui_components.get('logger', self.logger)
        logger.info("🧹 Cleanup button clicked")
        # Implementation can be overridden by subclasses
    
    # Public utility methods
    
    def reset_module(self) -> None:
        """Reset module initialization."""
        self._initialized = False
        self._cached_components = None
        self._initialization_timestamp = None
        self.logger.info(f"🔄 {self.module_name} module reset")
    
    def get_module_status(self) -> Dict[str, Any]:
        """Get module status for debugging."""
        status = {
            'module_name': self.module_name,
            'initialized': self._initialized,
            'cached_available': self._cached_components is not None,
            'timestamp': self._get_timestamp(),
            'initialization_time': self._initialization_timestamp
        }
        
        if self._cached_components:
            status.update({
                'logger_available': 'logger' in self._cached_components,
                'ui_available': 'ui' in self._cached_components,
                'config_available': 'config' in self._cached_components
            })
        
        return status
    
    def get_cached_components(self) -> Optional[Dict[str, Any]]:
        """Get cached components if available."""
        return self._cached_components