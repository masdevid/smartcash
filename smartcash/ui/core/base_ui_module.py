"""
Base UI Module class that acts as a config orchestrator.

This class provides a standard base for all UI modules focused on configuration
orchestration and delegates implementation to separate config_handler classes.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import os
from smartcash.ui.core.mixins import (
    ConfigurationMixin,
    OperationMixin,
    LoggingMixin,
    ButtonHandlerMixin,
    ValidationMixin,
    DisplayMixin,
    EnvironmentMixin
)

# Default configuration paths
DEFAULT_CONFIG_PATHS = {
    'colab': '/content/configs',
    'local': './configs',
    'default': './configs'
}

# Default config file names
DEFAULT_CONFIG_FILES = {
    'training': 'training_config.yaml',
    'inference': 'inference_config.yaml',
    'evaluation': 'evaluation_config.yaml',
    'default': 'config.yaml'
}

class BaseUIModule(
    ConfigurationMixin,
    OperationMixin,
    LoggingMixin,
    ButtonHandlerMixin,
    ValidationMixin,
    DisplayMixin,
    EnvironmentMixin,
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
        return self._enable_environment
    
    def __init__(self, 
                 module_name: str, 
                 parent_module: str = None, 
                 enable_environment: bool = True,
                 show_environment_indicator: bool = True,
                 config_type: str = 'default',
                 **kwargs):
        """
        Initialize base UI module.
        
        Args:
            module_name: Name of the module
            parent_module: Parent module name
            enable_environment: Whether to enable environment management features
            show_environment_indicator: Whether to show the environment indicator in the header
            config_type: Type of configuration ('training', 'inference', 'evaluation', etc.)
            **kwargs: Additional keyword arguments for parent classes
        """
        # Initialize module identification
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        self.show_environment_indicator = show_environment_indicator
        self.config_type = config_type
        
        # Initialize parent classes using super() to respect MRO
        super().__init__(**kwargs)
        # Initialize UI components dictionary
        self._ui_components = {}
        # Initialize components first
        self.initialize_components()
        # Initialize environment support flag
        self._enable_environment = enable_environment
        # Environment detection is handled by EnvironmentMixin
        self._environment = self._detect_environment()
        self._config_path = self._get_default_config_path()
        
        
        self._required_components = getattr(self, '_required_components', [])
        self._is_initialized = False
        
        # Update logging context with module info
        if hasattr(self, '_update_logging_context'):
            self._update_logging_context()
        
        self.log_debug(f"✅ BaseUIModule diinisialisasi: {self.full_module_name}")
        
       
    
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
        
        # Return default configuration if config handler is not available
        return self.get_default_config()
        

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
            
            self.log_debug(f"✅ Config handler diinisialisasi untuk {self.full_module_name}")
            
        except Exception as e:
            self.log_error(f"❌ Gagal menginisialisasi config handler: {e}")
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
    
    def _initialize_core(self, *, is_initial_setup: bool = False) -> bool:
        """
        Core initialization logic shared between initialize() and initialize_components().
        
        Args:
            is_initial_setup: If True, performs additional setup only needed during object initialization
            
        Returns:
            bool: True if initialization was successful
        """
        try:
            if self._is_initialized and not is_initial_setup:
                return True

            self.log_debug(f"🔄 {'Menginisialisasi' if is_initial_setup else 'Memuat ulang'} {self.full_module_name}")

            # 1. Initialize configuration handler
            self._initialize_config_handler()
            
            # 2. Create/update UI components
            config = self.get_default_config() if is_initial_setup else self.get_current_config()
            self._ui_components = self.create_ui_components(config)

            # 3. Setup environment if this is the initial setup
            if is_initial_setup and hasattr(self, 'refresh_environment_detection'):
                self.refresh_environment_detection()

            # 4. Register operations and handlers
            self._register_default_operations()
            self._register_dynamic_button_handlers()
            self._validate_button_handler_integrity()
            
            # 5. Link action container for unified button state management
            self._link_action_container()
            
            # 6. Setup UI logging bridge if operation container exists
            if 'operation_container' in self._ui_components:
                self._setup_ui_logging_bridge(self._ui_components['operation_container'])
            
            # 7. Initialize progress display
            self._initialize_progress_display()
            
            # 8. Mark as initialized
            self._is_initialized = True
            
            # 9. Log initialization complete
            if is_initial_setup:
                if hasattr(self, '_flush_log_buffer'):
                    self._flush_log_buffer()
                self._log_initialization_complete()
                self.log_info(f"✅ {self.full_module_name} berhasil diinisialisasi")
            
            return True
            
        except Exception as e:
            error_msg = f"❌ Gagal {'menginisialisasi' if is_initial_setup else 'memuat ulang'} {self.full_module_name}: {e}"
            self.log_error(error_msg)
            self._is_initialized = False
            return False

    def initialize(self) -> bool:
        """
        Inisialisasi ulang modul dengan semua komponen.
        
        Returns:
            bool: True jika inisialisasi berhasil
        """
        return self._initialize_core(is_initial_setup=False)

    def initialize_components(self) -> None:
        """
        Inisialisasi komponen UI selama inisialisasi objek.
        
        Method ini dipanggil otomatis selama __init__ dan seharusnya hanya dipanggil sekali.
        Untuk inisialisasi ulang, gunakan initialize().
        """
        self._initialize_core(is_initial_setup=True)
    
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
    
    def _handle_save_config(self, _button=None) -> Dict[str, Any]:
        """Handle save config button click with status updates."""
        try:
            result = self.save_config()
            if result.get('success'):
                success_msg = result.get('message', 'Konfigurasi berhasil disimpan')
                self.log(f"💾 {success_msg}", 'success')
            else:
                error_msg = result.get('message', 'Penyimpanan gagal')
                self.log(f"❌ Penyimpanan gagal: {error_msg}", 'error')
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan penyimpanan konfigurasi: {e}"
            self.log(f"❌ {error_msg}", 'error')
            return {'success': False, 'message': error_msg}
    
    def _handle_reset_config(self, _button=None) -> Dict[str, Any]:
        """Handle reset config button click with status updates."""
        try:
            result = self.reset_config()
            if result.get('success'):
                success_msg = result.get('message', 'Konfigurasi berhasil direset ke pengaturan awal')
                self.log(f"🔄 {success_msg}", 'success')
            else:
                error_msg = result.get('message', 'Reset gagal')
                self.log(f"❌ Reset gagal: {error_msg}", 'error')
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan reset konfigurasi: {e}"
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
   
    def _log_initialization_complete(self) -> None:
        """Log initialization completion to operation container (after it's ready)."""
        try:
            # Log environment info if environment support is enabled
            if self.has_environment_support:
                # Refresh environment detection to ensure latest values
                self.refresh_environment_detection()
                
                # Update header indicator with current environment
                self.update_header_indicator()
                
                # Safely access environment_paths attributes
                if hasattr(self, 'environment_paths') and self.environment_paths is not None:
                    if hasattr(self.environment_paths, 'data_root') and self.environment_paths.data_root:
                        self.log_info(f"📁 Direktori kerja: {self.environment_paths.data_root}")
                    else:
                        self.log_info("ℹ️ Direktori kerja default akan digunakan")
            
            # Update status panel
            self.log_info(f"📊 Status: Siap untuk {self.full_module_name}")
            
        except Exception as e:
            # Log the error using the logging mixin
            self.log_error(f"Gagal mencatat inisialisasi selesai: {e}", exc_info=True)
            self.log_error(f"⚠️ Terjadi kesalahan saat inisialisasi: {str(e)}")
     
    def update_header_indicator(self) -> None:
        """Update header container environment indicator with current detected environment."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                header_container = self._ui_components.get('header_container')
                if header_container and hasattr(header_container, 'update'):
                    # Get current environment info
                    env_info = self.get_environment_info() if hasattr(self, 'get_environment_info') else {}
                    
                    # Update the header environment indicator
                    header_container.update(
                        environment=self._environment,
                        config_path=self._config_path
                    )
                    
                    self.log_debug(f"Header environment indicator updated: {self._environment}")
                else:
                    self.log_debug("Header container not found or no update method available")
        except Exception as e:
            self.log_debug(f"Failed to update header environment indicator: {e}")
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
            
            # Initialize progress bars if available and enabled
            if hasattr(self, '_ui_components') and self._ui_components:
                # Look for progress tracker in operation container
                operation_container = self._ui_components.get('operation_container')
                if operation_container and isinstance(operation_container, dict):
                    # Check if progress tracking is enabled by looking at progress_tracker key
                    progress_tracker = operation_container.get('progress_tracker')
                    
                    if progress_tracker is not None:
                        # Progress tracking is enabled
                        if hasattr(progress_tracker, 'initialize'):
                            progress_tracker.initialize()
                            self.log_debug("✅ Progress tracker initialized")
                    else:
                        self.log_debug("Progress tracking disabled, skipping progress tracker initialization")
                
                # Look for standalone progress components (always check these as they may be independently configured)
                progress_tracker = self._ui_components.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'initialize'):
                    progress_tracker.initialize()
                    self.log_debug("✅ Standalone progress tracker initialized")
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.log_debug(f"Failed to initialize progress display: {e}")
    
    def _ensure_progress_visibility(self) -> None:
        """
        Ensure progress components are visible and properly configured.
        
        This method handles the visibility and configuration of progress
        tracking components. Subclasses can override this method to
        implement module-specific progress visibility logic.
        """
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                # Ensure operation container progress is visible (only if progress tracking is enabled)
                operation_container = self._ui_components.get('operation_container')
                if operation_container and isinstance(operation_container, dict):
                    # Check if progress tracking is enabled by looking at progress_tracker key
                    progress_tracker = operation_container.get('progress_tracker')
                    
                    if progress_tracker is not None:
                        # Progress tracking is enabled
                        if 'update_progress' in operation_container:
                            self.log_debug("✅ Progress tracking enabled in operation container")
                        
                        # Ensure progress tracker is visible
                        if hasattr(progress_tracker, 'layout'):
                            progress_tracker.layout.visibility = 'visible'
                        if hasattr(progress_tracker, 'visible'):
                            progress_tracker.visible = True
                        self.log_debug("✅ Progress tracker visibility ensured")
                    else:
                        self.log_debug("Progress tracking disabled, skipping visibility setup")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.log_debug(f"Failed to ensure progress visibility: {e}")
    
    def _link_action_container(self) -> None:
        """Link ActionContainer to this module for unified button state management.
        
        This enables ActionContainer to delegate button state operations to
        the module's ButtonHandlerMixin, creating a single source of truth
        for button state management.
        """
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                action_container = self._ui_components.get('action_container')
                if action_container:
                    # Handle both dict-style and object-style action containers
                    if isinstance(action_container, dict):
                        container_obj = action_container.get('container')
                        if container_obj and hasattr(container_obj, 'set_parent_module'):
                            container_obj.set_parent_module(self)
                            self.log_debug("✅ ActionContainer linked to module for unified button state management")
                    elif hasattr(action_container, 'set_parent_module'):
                        action_container.set_parent_module(self)
                        self.log_debug("✅ ActionContainer linked to module for unified button state management")
                        
        except Exception as e:
            self.log_debug(f"Failed to link ActionContainer: {e}")
    
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
                self.log_error(f"Button validation errors in {self.full_module_name}: {len([i for i in result.issues if i.level.value == 'error'])} errors")
                for issue in result.issues:
                    if issue.level.value == 'error':
                        self.log_error(f"🔘 {issue.message}")
                        if issue.suggestion:
                            self.log_error(f"   💡 Suggestion: {issue.suggestion}")
            
            if result.has_warnings:
                self.log_warning(f"Button validation warnings in {self.full_module_name}: {len([i for i in result.issues if i.level.value == 'warning'])} warnings")
                for issue in result.issues:
                    if issue.level.value == 'warning':
                        self.log_warning(f"🔘 {issue.message}")
                        if issue.suggestion:
                            self.log_warning(f"   💡 Suggestion: {issue.suggestion}")
            
            # Log auto-fixes
            if result.auto_fixes_applied:
                self.log_info(f"Button validation auto-fixes applied: {len(result.auto_fixes_applied)}")
                for fix in result.auto_fixes_applied:
                    self.log_info(f"🔧 {fix}")
            
            # Log success if no issues
            if result.is_valid and not result.has_warnings:
                self.log_debug(f"✅ Button validation passed for {self.full_module_name}")
            
        except Exception as e:
            self.log_warning(f"Button validation failed: {e}")
    
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
            
            self.log_debug(f"🧹 Cleaned up BaseUIModule: {self.full_module_name}")
            
        except Exception as e:
            self.log_error(f"❌ Error during cleanup: {e}")
    
    def _register_dynamic_button_handlers(self) -> None:
        """Register module-specific button handlers using the mixin's functionality.
        
        This method delegates button discovery and registration to the ButtonHandlerMixin,
        maintaining clear separation of concerns. The module only provides its specific
        button handler mappings.
        """
        # Prevent duplicate registration
        if getattr(self, '_button_handlers_registered', False):
            return
            
        try:
            # Get module-specific button handlers
            button_handlers = self._get_module_button_handlers()
            
            if not button_handlers:
                return
            
            # Register each module-specific handler with the mixin
            for button_id, handler in button_handlers.items():
                self.register_button_handler(button_id, handler)
            
            # Let the mixin handle the actual button discovery and setup
            self._setup_button_handlers()
            
            # Mark as registered to prevent duplicates
            self._button_handlers_registered = True
            
        except Exception as e:
            self.log_error(f"❌ Failed to register dynamic button handlers: {e}")
    
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
        
    def _extract_button_id(self, button: Any, operation_name: str) -> Optional[str]:
        """
        Extract button ID from button widget with improved detection logic.
        
        Args:
            button: Button widget that triggered the operation
            operation_name: Name of the operation (fallback for ID generation)
            
        Returns:
            Button ID string or None if not determinable
        """
        if button is None:
            return None
            
        # Method 1: Check for explicit button ID attribute
        for attr in ['button_id', 'id', '_button_id']:
            if hasattr(button, attr):
                button_id = getattr(button, attr)
                if button_id and isinstance(button_id, str):
                    return self._normalize_button_id(button_id)
        
        # Method 2: Look for button in action container and find matching ID
        if hasattr(self, '_ui_components') and self._ui_components:
            action_container = self._ui_components.get('action_container')
            if action_container:
                buttons = {}
                
                # Extract buttons from action container
                if isinstance(action_container, dict) and 'buttons' in action_container:
                    buttons = action_container['buttons']
                elif hasattr(action_container, 'buttons'):
                    buttons = getattr(action_container, 'buttons', {})
                
                # Find matching button by reference
                for btn_id, btn_widget in buttons.items():
                    if btn_widget is button:
                        return self._normalize_button_id(btn_id)
        
        # Method 3: Extract from button description (fallback)
        if hasattr(button, 'description') and button.description:
            desc = button.description.strip()
            # Remove emoji and common prefixes
            import re
            # Remove emoji (Unicode > 127)
            desc_clean = ''.join(char for char in desc if ord(char) <= 127)
            desc_clean = desc_clean.strip()
            
            # Remove common button text patterns
            patterns_to_remove = [
                r'^(run|execute|start|begin)\s+',
                r'\s+(button|btn)$',
                r'^(button|btn)\s+',
            ]
            for pattern in patterns_to_remove:
                desc_clean = re.sub(pattern, '', desc_clean, flags=re.IGNORECASE)
            
            if desc_clean:
                return self._normalize_button_id(desc_clean)
        
        # Method 4: Fallback to operation name
        return self._normalize_button_id(operation_name)
    
    def _normalize_button_id(self, button_id: str) -> str:
        """Normalize button ID by removing common suffixes and prefixes."""
        if not button_id:
            return 'unknown'
            
        button_id = str(button_id).strip().lower()
        
        # Remove common suffixes
        for suffix in ['_button', '_btn', 'button', 'btn']:
            if button_id.endswith(suffix):
                button_id = button_id[:-len(suffix)]
                break
        
        # Remove common prefixes  
        for prefix in ['btn_', 'button_']:
            if button_id.startswith(prefix):
                button_id = button_id[len(prefix):]
                break
                
        # Clean up spaces and special characters
        import re
        button_id = re.sub(r'[^\w]+', '_', button_id)
        button_id = button_id.strip('_')
        
        return button_id or 'unknown'
        
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
        # Extract button ID for individual button management with improved detection
        button_id = self._extract_button_id(button, operation_name)
        
        try:
            # CRITICAL: Ensure logging bridge is ready BEFORE any operation executes
            # This prevents backend service logs from leaking to console
            self._ensure_logging_bridge_ready()
            
            # Clear operation logs at start of each operation
            self._clear_operation_logs()
            
            # Check if we should reduce logging for one-click operations
            reduce_logging = hasattr(self, 'should_reduce_operation_logging') and self.should_reduce_operation_logging(operation_name)
            
            # Start operation logging and progress tracking (optimized for one-click)
            if not reduce_logging:
                self.log_operation_start(operation_name)
                self.log(f"Memulai {operation_name.lower()}...", "info")
            
            self.start_progress(f"Memulai {operation_name.lower()}...", 0)
            
            # Disable all buttons during operation to prevent overlapping operations
            self.disable_all_buttons(f"⏳ {operation_name}...")
            
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
            
            # Handle result (optimized logging for one-click operations)
            if result.get('success'):
                # Check if this is a dialog display - don't show success message for dialog displays
                # The actual success will be handled by the dialog callback
                if result.get('dialog_shown'):
                    # Just update progress to indicate dialog is shown, don't show success message
                    self.update_progress(50, "⏳ Menunggu konfirmasi user...")
                else:
                    success_msg = success_message or result.get('message', f'{operation_name} berhasil diselesaikan')
                    
                    if not reduce_logging:
                        self.log_operation_complete(operation_name)
                        self.log(f"✅ {success_msg}", "info")
                        
                    # Always log success to operation container for user feedback
                    self.log(f"✅ {success_msg}", 'success')
                    self.complete_progress(f"{operation_name} selesai")
            else:
                error_prefix = error_message or f'{operation_name} gagal'
                error_msg = result.get('message', 'Operasi gagal')
                full_error = f"{error_prefix}: {error_msg}"
                
                # Always log errors regardless of one-click mode
                self.log_operation_error(operation_name, error_msg)
                self.log(f"❌ {full_error}", "error")
                self.error_progress(full_error)
                
            return result
            
        except Exception as e:
            error_prefix = error_message or f'Kesalahan {operation_name.lower()}'
            error_msg = f"{error_prefix}: {e}"
            
            # Always log errors regardless of one-click mode
            self.log_operation_error(operation_name, str(e))
            self.error_progress(error_msg)
            return {'success': False, 'message': error_msg}
        finally:
            # Re-enable all buttons after operation completes
            self.enable_all_buttons()

    def _clear_operation_logs(self) -> None:
        """Clear operation container logs at the start of each operation."""
        try:
            operation_container = self.get_component('operation_container')
            if operation_container and isinstance(operation_container, dict):
                # Try to clear logs if the operation container has a clear method
                if 'clear_logs' in operation_container and callable(operation_container['clear_logs']):
                    operation_container['clear_logs']()
                elif 'log_accordion' in operation_container:
                    log_accordion = operation_container['log_accordion']
                    if hasattr(log_accordion, 'clear'):
                        log_accordion.clear()
        except Exception as e:
            # Ignore errors in log clearing - it's not critical
            pass

    def _ensure_logging_bridge_ready(self) -> None:
        """
        Ensure logging bridge is ready before operations execute.
        
        This is critical to prevent backend service logs from leaking to console.
        Must be called before any operation that might use backend services.
        """
        try:
            # Check if operation container exists and bridge is setup
            if (hasattr(self, '_ui_components') and 
                'operation_container' in self._ui_components and
                not getattr(self, '_ui_logging_bridge_setup', False)):
                
                self._setup_ui_logging_bridge(self._ui_components['operation_container'])
                self.log_debug("✅ Logging bridge activated before operation")
                
        except Exception as e:
            self.log_debug(f"Could not ensure logging bridge ready: {e}")

    def _detect_environment(self) -> str:
        """
        Detect the current environment using EnvironmentMixin.
        
        Returns:
            String indicating the environment type
        """
        try:
            if self.has_environment_support and hasattr(self, 'get_environment_info'):
                # Use EnvironmentMixin's comprehensive environment detection
                env_info = self.get_environment_info()
                return env_info.get('environment_type', 'local')
            else:
                # Fallback to basic detection if environment support is disabled
                from smartcash.common.environment import is_colab_environment
                
                if is_colab_environment():
                    return 'colab'
                else:
                    # Check if running in Jupyter
                    import sys
                    if 'ipykernel' in sys.modules:
                        return 'jupyter'
                    else:
                        return 'local'
        except Exception as e:
            # Log the error if logger is available
            if hasattr(self, 'log_warning'):
                self.log_warning(f"Environment detection failed: {e}")
            return 'local'
    
    def _get_default_config_path(self) -> str:
        """
        Get the default configuration path based on environment and config type.
        
        Returns:
            Default configuration path
        """
        env_path = DEFAULT_CONFIG_PATHS.get(self._environment, DEFAULT_CONFIG_PATHS['default'])
        return os.path.join(env_path, DEFAULT_CONFIG_FILES.get(self.config_type, DEFAULT_CONFIG_FILES['default']))
    
    def refresh_environment_detection(self) -> str:
        """
        Refresh environment detection and update related attributes.
        
        This method re-runs environment detection and updates the environment
        information in the header container if it exists.
        
        Returns:
            String indicating the detected environment type
        """
        try:
            # Re-detect environment
            old_environment = self._environment
            self._environment = self._detect_environment()
            
            # Update config path based on new environment
            self._config_path = self._get_default_config_path()
            
            # Update environment mixin if available
            if hasattr(self, '_environment_mixin') and self._environment_mixin:
                # Force re-setup of environment mixin to get updated values
                self._environment_mixin._setup_environment()
                if hasattr(self._environment_mixin, 'get_environment_info'):
                    env_info = self._environment_mixin.get_environment_info()
                    detected_env = env_info.get('environment_type', self._environment)
                    if detected_env != self._environment:
                        self._environment = detected_env
            
            # Update header container if it exists
            if hasattr(self, '_ui_components') and self._ui_components:
                header_container = self._ui_components.get('header_container')
                if header_container and hasattr(header_container, 'update'):
                    header_container.update(
                        environment=self._environment,
                        config_path=self._config_path
                    )
            
            # Log environment change if it occurred
            if old_environment != self._environment:
                if hasattr(self, 'log_info'):
                    self.log_info(f"🔄 Environment updated: {old_environment} → {self._environment}")
            
            return self._environment
            
        except Exception as e:
            if hasattr(self, 'log_error'):
                self.log_error(f"Failed to refresh environment detection: {e}")
            return self._environment
    