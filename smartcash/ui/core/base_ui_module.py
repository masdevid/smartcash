"""
Base UI Module class that acts as a config orchestrator.

This class provides a standard base for all UI modules focused on configuration
orchestration and delegates implementation to separate config_handler classes.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from smartcash.ui.core.mixins import (
    ConfigurationMixin,
    OperationMixin,
    LoggingMixin,
    ButtonHandlerMixin,
    ValidationMixin,
    DisplayMixin
)
from smartcash.ui.logger import get_module_logger


class BaseUIModule(
    ConfigurationMixin,
    OperationMixin,
    LoggingMixin,
    ButtonHandlerMixin,
    ValidationMixin,
    DisplayMixin,
    ABC
):
    """
    Base UI Module class that acts as a config orchestrator.
    
    This class provides:
    - Configuration orchestration through config_handler delegation
    - UI component management through mixins
    - Environment support when enabled
    - Operation handling through dedicated handlers
    
    The main responsibility is to orchestrate configurations and delegate
    implementation details to separate config_handler classes in each module.
    BaseUIModule uses composition over inheritance and acts as a config 
    orchestrator where all config operations are delegated to config_handler.
    """
    
    @property
    def has_environment_support(self) -> bool:
        """Check if environment features are enabled."""
        return self._enable_environment and hasattr(self, '_environment_paths')
    
    @property
    def environment_paths(self) -> Optional[Dict[str, str]]:
        """Get environment paths if environment support is enabled."""
        if not self.has_environment_support:
            self.logger.warning("Environment support is not enabled for this module")
            return None
        return getattr(self, '_environment_paths', None)
    
    @property
    def is_colab(self) -> Optional[bool]:
        """Check if running in Google Colab (if environment support is enabled)."""
        return getattr(self, '_is_colab', None) if self.has_environment_support else None
    
    @property
    def is_drive_mounted(self) -> Optional[bool]:
        """Check if Google Drive is mounted (if environment support is enabled)."""
        return getattr(self, '_is_drive_mounted', None) if self.has_environment_support else None
    
    def __init__(self, module_name: str, parent_module: str = None, enable_environment: bool = False, **kwargs):
        """
        Initialize base UI module.
        
        Args:
            module_name: Name of the module
            parent_module: Parent module name
            enable_environment: Whether to enable environment management features
            **kwargs: Additional keyword arguments for parent classes
        """
        # Initialize module identification
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name

        # Initialize module identification first
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        
        # Initialize mixins directly. This is the correct pattern.
        LoggingMixin.__init__(self)
        OperationMixin.__init__(self)
        
        # Initialize environment support if enabled
        self._enable_environment = enable_environment
        if self._enable_environment:
            from smartcash.ui.core.mixins.environment_mixin import EnvironmentMixin
            EnvironmentMixin.__init__(self)
        
        # Set up logger after module name is set
        self.logger = get_module_logger(f"smartcash.ui.{self.full_module_name}")
        
        # Initialize UI components dictionary
        self._ui_components = {}
        
        # Initialize common attributes
        self._is_initialized = False
        self._ui_components = {}
        self._required_components = getattr(self, '_required_components', [])
        
        # Ensure logging mixin has the correct module info
        if hasattr(self, '_update_logging_context'):
            self._update_logging_context()
        
        self.logger.debug(f"✅ BaseUIModule diinisialisasi: {self.full_module_name}")
    
    # === Config Orchestration Methods ===
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current configuration from config_handler.
        
        This method delegates to the config_handler to get the current
        processed configuration state.
        
        Returns:
            Current configuration dictionary
        """
        if hasattr(self, '_config_handler') and self._config_handler:
            if hasattr(self._config_handler, 'get_current_config'):
                return self._config_handler.get_current_config()
            elif hasattr(self._config_handler, 'config'):
                return self._config_handler.config
        
        # Fallback to default config
        return self.get_default_config()
    
    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration through config_handler.
        
        This method delegates configuration updates to the config_handler.
        All configuration logic is handled by the config_handler instance.
        
        Args:
            updates: Configuration updates to apply
            
        Returns:
            Result dictionary with success status
        """
        if not hasattr(self, '_config_handler') or not self._config_handler:
            return {'success': False, 'message': 'Config handler belum diinisialisasi'}
        
        try:
            if hasattr(self._config_handler, 'update_config'):
                return self._config_handler.update_config(updates)
            else:
                # Fallback update
                if hasattr(self._config_handler, 'config'):
                    self._config_handler.config.update(updates)
                return {'success': True, 'message': 'Konfigurasi berhasil diperbarui'}
        except Exception as e:
            return {'success': False, 'message': f'Pembaruan konfigurasi gagal: {e}'}
    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self.get_current_config()
    
    def save_config(self) -> Dict[str, Any]:
        """
        Save configuration through config_handler.
        
        This method delegates configuration saving to the config_handler.
        All configuration persistence is handled by the config_handler instance.
        
        Returns:
            Result dictionary with success status
        """
        if not hasattr(self, '_config_handler') or not self._config_handler:
            return {'success': False, 'message': 'Config handler belum diinisialisasi'}
        
        try:
            if hasattr(self._config_handler, 'save_config'):
                return self._config_handler.save_config()
            else:
                return {'success': True, 'message': 'Konfigurasi disimpan (tidak ada operasi)'}
        except Exception as e:
            return {'success': False, 'message': f'Penyimpanan konfigurasi gagal: {e}'}
    
    def reset_config(self) -> Dict[str, Any]:
        """
        Reset configuration through config_handler.
        
        This method delegates configuration reset to the config_handler.
        All configuration reset logic is handled by the config_handler instance.
        
        Returns:
            Result dictionary with success status
        """
        if not hasattr(self, '_config_handler') or not self._config_handler:
            return {'success': False, 'message': 'Config handler belum diinisialisasi'}
        
        try:
            if hasattr(self._config_handler, 'reset_config'):
                return self._config_handler.reset_config()
            else:
                # Fallback reset
                if hasattr(self._config_handler, 'config'):
                    self._config_handler.config = self.get_default_config()
                return {'success': True, 'message': 'Konfigurasi berhasil direset ke pengaturan awal'}
        except Exception as e:
            return {'success': False, 'message': f'Reset konfigurasi gagal: {e}'}
    
    def _initialize_config_handler(self) -> None:
        """
        Initialize the config handler for this module.
        
        This method creates and sets up the config handler that will handle
        all configuration operations for this module. The BaseUIModule
        acts as a config orchestrator and delegates implementation to
        separate config_handler classes.
        """
        try:
            # Get default configuration
            default_config = self.get_default_config()
            
            # Create config handler with default config
            self._config_handler = self.create_config_handler(default_config)
            
            # Additional setup if config handler has initialization method
            if hasattr(self._config_handler, 'initialize'):
                self._config_handler.initialize()
            
            self.logger.debug(f"✅ Config handler diinisialisasi untuk {self.full_module_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menginisialisasi config handler: {e}")
            raise
    
    def get_config_handler(self) -> Any:
        """
        Get the config handler instance.
        
        The config handler is responsible for all configuration-related
        operations. The BaseUIModule delegates to this handler.
        
        Returns:
            Config handler instance or None if not initialized
        """
        return getattr(self, '_config_handler', None)
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for this module.
        
        This method should return the default configuration that will be
        passed to the config_handler for processing.
        
        Returns:
            Default configuration dictionary
        """
        pass
    
    @abstractmethod
    def create_config_handler(self, config: Dict[str, Any]) -> Any:
        """
        Create config handler instance for this module.
        
        The config handler is responsible for all configuration-related
        operations including validation, merging, saving, and loading.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Config handler instance (e.g., DependencyConfigHandler)
        """
        pass
    
    @abstractmethod
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create UI components for this module.
        
        This method should create and return all UI components needed
        for the module. The config parameter contains the processed
        configuration from the config_handler.
        
        Args:
            config: Processed configuration dictionary from config_handler
            
        Returns:
            UI components dictionary
        """
        pass
    
    def initialize(self) -> bool:
        """
        Initialize the module with all components.
        
        Returns:
            True if initialization was successful
        """
        try:
            if self._is_initialized:
                return True
            
            self.logger.debug(f"🔄 Menginisialisasi {self.full_module_name}")
            
            # Initialize configuration handler (delegates to config_handler)
            self._initialize_config_handler()
            
            # Create UI components
            self._ui_components = self.create_ui_components(self.get_current_config())
            
            # Initialize operation manager if method exists
            if hasattr(self, '_initialize_operation_manager'):
                self._initialize_operation_manager()
            
            # Register default operation handlers FIRST
            self._register_default_operations()
            
            # Register dynamic button handlers (includes setup)
            self._register_dynamic_button_handlers()
            
            # Validate button-handler integrity AFTER setup
            self._validate_button_handler_integrity()
            
            # Setup UI logging bridge
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    self._setup_ui_logging_bridge(operation_container)
            
            # Initialize progress display
            self._initialize_progress_display()
            
            # Flush any buffered logs now that operation container is ready
            if hasattr(self, '_flush_log_buffer'):
                self._flush_log_buffer()
            
            self._is_initialized = True
            
            self.logger.info(f"✅ {self.full_module_name} berhasil diinisialisasi")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menginisialisasi {self.full_module_name}: {e}")
            return False
    
    def _register_default_operations(self) -> None:
        """Register default operation handlers."""
        # Register common operations
        self.register_operation_handler('save_config', self.save_config)
        self.register_operation_handler('reset_config', self.reset_config)
        self.register_operation_handler('get_status', self.get_status)
        self.register_operation_handler('validate_all', self.validate_all)
        self.register_operation_handler('display_ui', self.display_ui)
        
        # Register button handlers with status updates
        self.register_button_handler('save', self._handle_save_config)
        self.register_button_handler('reset', self._handle_reset_config)
    
    def _handle_save_config(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle save config button click with status updates."""
        try:
            result = self.save_config()
            if result.get('success'):
                success_msg = result.get('message', 'Konfigurasi berhasil disimpan')
                self._update_header_status(f"💾 {success_msg}", "success")
                self.log(f"💾 {success_msg}", 'success')
            else:
                error_msg = result.get('message', 'Penyimpanan gagal')
                self._update_header_status(f"❌ Penyimpanan gagal: {error_msg}", "error")
                self.log(f"❌ Penyimpanan gagal: {error_msg}", 'error')
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan penyimpanan konfigurasi: {e}"
            self._update_header_status(f"❌ {error_msg}", "error")
            self.log(f"❌ {error_msg}", 'error')
            return {'success': False, 'message': error_msg}
    
    def _handle_reset_config(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle reset config button click with status updates."""
        try:
            result = self.reset_config()
            if result.get('success'):
                success_msg = result.get('message', 'Konfigurasi berhasil direset ke pengaturan awal')
                self._update_header_status(f"🔄 {success_msg}", "success")
                self.log(f"🔄 {success_msg}", 'success')
            else:
                error_msg = result.get('message', 'Reset gagal')
                self._update_header_status(f"❌ Reset gagal: {error_msg}", "error")
                self.log(f"❌ Reset gagal: {error_msg}", 'error')
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan reset konfigurasi: {e}"
            self._update_header_status(f"❌ {error_msg}", "error")
            self.log(f"❌ {error_msg}", 'error')
            return {'success': False, 'message': error_msg}
    
    def get_module_info(self) -> Dict[str, Any]:
        """
        Get comprehensive module information.
        
        Returns:
            Module information dictionary
        """
        return {
            'module_name': self.module_name,
            'parent_module': self.parent_module,
            'full_module_name': self.full_module_name,
            'is_initialized': self._is_initialized,
            'has_config_handler': self._config_handler is not None,
            'has_ui_components': self._ui_components is not None,
            'ui_components_count': len(self._ui_components) if self._ui_components else 0,
            'registered_operations': len(getattr(self, '_operation_handlers', {})),
            'registered_button_handlers': len(getattr(self, '_button_handlers', {}))
        }
    
    def __repr__(self) -> str:
        """String representation of the module."""
        status = "initialized" if self._is_initialized else "not initialized"
        return f"BaseUIModule({self.full_module_name}, {status})"
    
    def __str__(self) -> str:
        """String representation of the module."""
        return f"{self.full_module_name} UI Module"
    
    def _initialize_progress_display(self) -> None:
        """
        Initialize progress display components.
        
        This method ensures that progress tracking components are properly
        initialized and visible. Subclasses can override this method to
        implement module-specific progress display logic.
        """
        try:
            # Ensure progress visibility for operations
            self._ensure_progress_visibility()
            
            # Initialize progress bars if available
            if hasattr(self, '_ui_components') and self._ui_components:
                # Look for progress tracker in operation container
                operation_container = self._ui_components.get('operation_container')
                if operation_container and isinstance(operation_container, dict):
                    progress_tracker = operation_container.get('progress_tracker')
                    if progress_tracker and hasattr(progress_tracker, 'initialize'):
                        progress_tracker.initialize()
                        self.logger.debug("✅ Progress tracker initialized")
                
                # Look for standalone progress components
                progress_tracker = self._ui_components.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'initialize'):
                    progress_tracker.initialize()
                    self.logger.debug("✅ Standalone progress tracker initialized")
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to initialize progress display: {e}")
    
    def _ensure_progress_visibility(self) -> None:
        """
        Ensure progress components are visible and properly configured.
        
        This method handles the visibility and configuration of progress
        tracking components. Subclasses can override this method to
        implement module-specific progress visibility logic.
        """
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                # Ensure operation container progress is visible
                operation_container = self._ui_components.get('operation_container')
                if operation_container and isinstance(operation_container, dict):
                    # Enable progress tracking if available
                    if 'update_progress' in operation_container:
                        self.logger.debug("✅ Progress tracking enabled in operation container")
                    
                    # Ensure progress tracker is visible
                    progress_tracker = operation_container.get('progress_tracker')
                    if progress_tracker:
                        # Make progress tracker visible if it has visibility controls
                        if hasattr(progress_tracker, 'layout'):
                            progress_tracker.layout.visibility = 'visible'
                        if hasattr(progress_tracker, 'visible'):
                            progress_tracker.visible = True
                        self.logger.debug("✅ Progress tracker visibility ensured")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to ensure progress visibility: {e}")
    
    def _validate_button_handler_integrity(self) -> None:
        """
        Validate button-handler integrity during module setup.
        
        This method ensures that all buttons have corresponding handlers
        and follows naming conventions. It can auto-fix common issues.
        """
        try:
            from smartcash.ui.core.validation.button_validator import validate_button_handlers
            
            # Validate with auto-fix enabled
            result = validate_button_handlers(self, auto_fix=True)
            
            # Log validation results
            if result.has_errors:
                self.logger.error(f"Button validation errors in {self.full_module_name}: {len([i for i in result.issues if i.level.value == 'error'])} errors")
                for issue in result.issues:
                    if issue.level.value == 'error':
                        self.logger.error(f"🔘 {issue.message}")
                        if issue.suggestion:
                            self.logger.error(f"   💡 Suggestion: {issue.suggestion}")
            
            if result.has_warnings:
                self.logger.warning(f"Button validation warnings in {self.full_module_name}: {len([i for i in result.issues if i.level.value == 'warning'])} warnings")
                for issue in result.issues:
                    if issue.level.value == 'warning':
                        self.logger.warning(f"🔘 {issue.message}")
                        if issue.suggestion:
                            self.logger.warning(f"   💡 Suggestion: {issue.suggestion}")
            
            # Log auto-fixes
            if result.auto_fixes_applied:
                self.logger.info(f"Button validation auto-fixes applied: {len(result.auto_fixes_applied)}")
                for fix in result.auto_fixes_applied:
                    self.logger.info(f"🔧 {fix}")
            
            # Log success if no issues
            if result.is_valid and not result.has_warnings:
                self.logger.debug(f"✅ Button validation passed for {self.full_module_name}")
            
        except Exception as e:
            self.logger.warning(f"Button validation failed: {e}")
    
    # === Core Module Methods ===
    
    def get_component(self, component_type: str) -> Optional[Any]:
        """
        Get a UI component from the current components.
        
        This method provides access to UI components created by the module.
        Components are created by the create_ui_components method.
        
        Args:
            component_type: Type of component to retrieve
            
        Returns:
            Component instance or None if not found
        """
        if hasattr(self, '_ui_components') and self._ui_components:
            return self._ui_components.get(component_type)
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current module status."""
        return {
            'module_name': self.module_name,
            'full_module_name': self.full_module_name,
            'is_initialized': self._is_initialized,
            'has_config_handler': hasattr(self, '_config_handler') and self._config_handler is not None,
            'has_ui_components': hasattr(self, '_ui_components') and self._ui_components is not None,
            'component_count': len(self._ui_components) if hasattr(self, '_ui_components') and self._ui_components else 0
        }
    
    def cleanup(self) -> None:
        """Cleanup module resources."""
        try:
            # Clear UI components
            if hasattr(self, '_ui_components') and self._ui_components:
                self._ui_components.clear()
            
            # Reset initialization state
            self._is_initialized = False
            
            self.logger.debug(f"🧹 Cleaned up BaseUIModule: {self.full_module_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")
    
    def _register_dynamic_button_handlers(self) -> None:
        """Register module-specific button handlers using the mixin's functionality.
        
        This method delegates button discovery and registration to the ButtonHandlerMixin,
        maintaining clear separation of concerns. The module only provides its specific
        button handler mappings.
        """
        try:
            # Get module-specific button handlers
            button_handlers = self._get_module_button_handlers()
            
            if not button_handlers:
                self.logger.debug("No module-specific button handlers defined")
                return
            
            # Register each module-specific handler with the mixin
            for button_id, handler in button_handlers.items():
                self.register_button_handler(button_id, handler)
                self.logger.debug(f"🔘 Registered module handler: {button_id}")
            
            # Let the mixin handle the actual button discovery and setup
            # This eliminates duplication and uses the mixin's comprehensive logic
            self._setup_button_handlers()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register dynamic button handlers: {e}")
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get module-specific button handlers.
        
        This method should be overridden by subclasses to provide their specific button mappings.
        Default implementation includes common save/reset handlers.
        
        Returns:
            Dictionary mapping button IDs to handler functions
        """
        return {
            'save': self._handle_save_config,
            'reset': self._handle_reset_config
        }
        
    def get_button_validation_status(self) -> Dict[str, Any]:
        """
        Get current button validation status.
        
        Returns:
            Dictionary with validation status information
        """
        return {
            'is_valid': self._is_initialized and hasattr(self, '_ui_components') and bool(self._ui_components),
            'missing_components': [
                comp for comp in self._required_components 
                if not (hasattr(self, '_ui_components') and self._ui_components.get(comp))
            ],
            'total_required': len(self._required_components),
            'initialized': self._is_initialized
        }
        
    def ensure_components_ready(self) -> bool:
        """Ensure all required UI components are ready for operations.
        
        Returns:
            bool: True if all components are ready, False otherwise
        """
        # Check if UI components are initialized
        if not hasattr(self, '_ui_components') or not self._ui_components:
            self.log("⚠️ UI components not initialized", 'warning')
            return False
            
        # Check if operation container is available
        if 'operation_container' not in self._ui_components:
            self.log("⚠️ Required UI component not available: operation_container", 'warning')
            return False
            
        # Check if progress tracker is available within operation container
        operation_container = self._ui_components.get('operation_container')
        if operation_container and isinstance(operation_container, dict):
            progress_tracker = operation_container.get('progress_tracker')
            if not progress_tracker:
                self.log("⚠️ Progress tracker not available in operation container", 'warning')
                return False
        
        # Ensure progress tracker is ready if the method exists
        if hasattr(self, 'ensure_progress_ready') and not self.ensure_progress_ready():
            self.log("⚠️ Progress tracker not ready", 'warning')
            return False
            
        return True
    
    def _update_header_status(self, message: str, status_type: str = "info") -> None:
        """
        Update header container status.
        
        Args:
            message: Status message
            status_type: Status type (info, success, warning, error)
        """
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                header_container = self._ui_components.get('header_container')
                if header_container:
                    # Handle object-style header container
                    if hasattr(header_container, 'update_status'):
                        header_container.update_status(message, status_type)
                        return
                    # Handle dict-style header container
                    elif isinstance(header_container, dict) and 'update_status' in header_container:
                        header_container['update_status'](message, status_type)
                        return
                        
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to update header status: {e}")
    
    def _execute_operation_with_wrapper(self, 
                                       operation_name: str,
                                       operation_func: callable,
                                       button: Any = None,
                                       validation_func: callable = None,
                                       success_message: str = None,
                                       error_message: str = None) -> Dict[str, Any]:
        """
        Common wrapper for executing operations with standard logging, progress, and button management.
        
        Args:
            operation_name: Display name for the operation
            operation_func: Function to execute the actual operation
            button: Button that triggered the operation
            validation_func: Optional validation function to run before operation
            success_message: Custom success message (optional)
            error_message: Custom error message prefix (optional)
            
        Returns:
            Operation result dictionary
        """
        # Extract button ID for individual button management
        button_id = getattr(button, 'description', operation_name.lower()).lower().replace(' ', '_')
        
        try:
            # Start operation logging and progress tracking
            self.log_operation_start(operation_name)
            self.start_progress(f"Memulai {operation_name.lower()}...", 0)
            self.log(f"Memulai {operation_name.lower()}...", "info")
            
            # Only disable the clicked button, not all buttons
            self.disable_all_buttons(f"⏳ {operation_name}...", button_id=button_id)
            
            # Run validation if provided
            if validation_func:
                validation_result = validation_func()
                if not validation_result.get('valid', True):
                    warning_msg = validation_result.get('message', f'Validasi {operation_name.lower()} gagal')
                    self.log(f"⚠️ {warning_msg}", 'warning')
                    self.error_progress(warning_msg)
                    return {'success': False, 'message': warning_msg}
            
            # Update progress for execution
            self.update_progress(25, f"Memproses {operation_name.lower()}...")
            
            # Execute the actual operation
            result = operation_func()
            
            # Handle result
            if result.get('success'):
                success_msg = success_message or result.get('message', f'{operation_name} berhasil diselesaikan')
                self.log_operation_complete(operation_name)
                self.log(f"✅ {success_msg}", "info")
                self.log(f"✅ {success_msg}", 'success')
                self.complete_progress(f"{operation_name} selesai")
            else:
                error_prefix = error_message or f'{operation_name} gagal'
                error_msg = result.get('message', 'Operasi gagal')
                full_error = f"{error_prefix}: {error_msg}"
                self.log_operation_error(operation_name, error_msg)
                self.log(f"❌ {full_error}", "error")
                self.error_progress(full_error)
                
            return result
            
        except Exception as e:
            error_prefix = error_message or f'Kesalahan {operation_name.lower()}'
            error_msg = f"{error_prefix}: {e}"
            self.log_operation_error(operation_name, str(e))
            self.update_operation_status(error_msg, "error")
            self.error_progress(error_msg)
            return {'success': False, 'message': error_msg}
        finally:
            # Re-enable only the specific button that was disabled
            if button_id:
                self.enable_all_buttons(button_id=button_id)