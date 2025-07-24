# -*- coding: utf-8 -*-
"""
File: smartcash/ui/model/backbone/backbone_uimodule.py
Description: Refactored implementation of the Backbone Module using the modern BaseUIModule pattern.
"""

from typing import Dict, Any, Optional, List

from smartcash.ui.core.base_ui_module import BaseUIModule
# Enhanced UI Module Factory removed - use direct instantiation

from smartcash.ui.core.decorators import suppress_ui_init_logs
from .components.backbone_ui import create_backbone_ui
from .configs.backbone_config_handler import BackboneConfigHandler
from .configs.backbone_defaults import get_default_backbone_config

# Import mixins for shared functionality
from smartcash.ui.model.mixins import (
    ModelDiscoveryMixin,
    ModelConfigSyncMixin,
    BackendServiceMixin
)

# Import the operation service for backend API integration
from .services import BackboneOperationService


class BackboneUIModule(BaseUIModule, ModelDiscoveryMixin, ModelConfigSyncMixin, BackendServiceMixin):
    """
    Backbone UI Module with proper inheritance structure.
    
    Inherits from:
    - BaseUIModule: Core UI module functionality with proper MRO
    - ModelDiscoveryMixin: Standardized checkpoint discovery and file scanning
    - ModelConfigSyncMixin: Cross-module configuration synchronization
    - BackendServiceMixin: Backend service integration and management
    
    The inheritance order ensures proper Method Resolution Order (MRO) with
    BaseUIModule as the primary base class and mixins providing additional functionality.
    """
    # Define required UI components at class level
    _required_components = [
        'main_container',
        'header_container',
        'form_container',
        'action_container',
        'operation_container'
    ]

    def __init__(self, enable_environment: bool = True):
        """
        Initialize the Backbone UI module.
        
        Args:
            enable_environment: Whether to enable environment management features
        """
        # Call parent initializer with required parameters
        super().__init__(
            module_name='backbone',
            parent_module='model',
            enable_environment=enable_environment
        )
        
        # Initialize log buffer for pre-operation-container logs
        self._log_buffer = []
        
        # Operation container reference for logging
        self._operation_container = None
        
        # Button registration tracking
        self._buttons_registered = False
        
        # Initialize operation service for backend API integration
        self._operation_service = None
        
        self.log_debug("BackboneUIModule initialized with proper mixin inheritance.")

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this module."""
        return get_default_backbone_config()

    def create_config_handler(self, config: Dict[str, Any]) -> BackboneConfigHandler:
        """Creates a configuration handler instance."""
        return BackboneConfigHandler(config)

    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Creates the UI components for the module."""
        try:
            # Create the UI components
            ui_components = create_backbone_ui(config=config)
            
            # Store the UI components in the instance
            self._ui_components = ui_components
            
            # Set the main container as the main widget
            if 'main_container' in ui_components and ui_components['main_container'] is not None:
                self._display_state['main_component'] = ui_components['main_container']
            
            self.log_debug("UI components created successfully")
            return ui_components
            
        except Exception as e:
            self.log_error(f"Failed to create UI components: {e}", exc_info=True)
            raise

    def _register_default_operations(self) -> None:
        """Register default operation handlers including backbone-specific operations."""
        # Call parent method to register base operations
        super()._register_default_operations()
        
        # Note: Dynamic button handler registration is now handled by BaseUIModule
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get Backbone module-specific button handlers."""
        # Start with base handlers (save, reset)
        handlers = super()._get_module_button_handlers()
        
        # Add Backbone-specific handlers 
        backbone_handlers = {
            'validate': self._operation_validate,
            'build': self._operation_build,
            'save': self._handle_save_config,
            'reset': self._handle_reset_config,
            'rescan_models': self._handle_rescan_models,
        }
        
        handlers.update(backbone_handlers)
        return handlers
    
    def _register_module_button_handlers(self) -> None:
        """Register module-specific button handlers."""
        if self._buttons_registered:
            self.log_debug("⏭️ Skipping button registration - already registered")
            return
            
        try:
            # Get module-specific handlers
            module_handlers = self._get_module_button_handlers()
            
            # Register each handler
            for button_id, handler in module_handlers.items():
                self.register_button_handler(button_id, handler)
                self.log_debug(f"✅ Registered backbone button handler: {button_id}")
            
            # Setup button handlers after registering them
            self._setup_button_handlers()
            
            # Mark as registered
            self._buttons_registered = True
            
            self.log_info(f"🎯 Registered {len(module_handlers)} backbone button handlers")
            
        except Exception as e:
            self.log_error(f"Failed to register module button handlers: {e}", exc_info=True)
        
    def _setup_operation_container(self) -> bool:
        """
        Set up the operation container for the module.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            # Get the operation container from UI components
            if not hasattr(self, '_ui_components') or not self._ui_components:
                self.log_error("UI components not available for operation container setup")
                return False
                
            # Store reference to operation container
            self._operation_container = self._ui_components.get('operation_container')
            
            if not self._operation_container:
                self.log_warning("Operation container not found in UI components")
                return False
                
            # Flush any buffered logs to the operation container
            self._flush_log_buffer()
            
            self.log_debug("✅ Operation container setup complete")
            return True
            
        except Exception as e:
            self.log_error(f"Failed to setup operation container: {e}", exc_info=True)
            return False
    
    def get_available_models_via_mixin(self) -> Dict[str, Any]:
        """
        Demonstrate proper usage of ModelDiscoveryMixin for model discovery.
        
        This method shows how to leverage the mixin functionality instead of
        implementing custom discovery logic.
        
        Returns:
            Dict containing discovered models organized by type
        """
        try:
            # Use the ModelDiscoveryMixin's standardized discovery
            discovered_checkpoints = self.discover_checkpoints(
                discovery_paths=['data/models', 'data/checkpoints'],
                filename_patterns=[
                    '*backbone*smartcash*.pt', '*backbone*smartcash*.pth',
                    '*backbone*.pt', '*backbone*.pth',
                    '*.pt', '*.pth'
                ],
                validation_requirements={
                    'min_size_mb': 1,
                    'required_extensions': ['.pt', '.pth']
                }
            )
            
            # Organize results by backbone type using mixin's normalization
            by_type = {}
            for checkpoint in discovered_checkpoints:
                backbone = checkpoint.get('backbone', 'unknown')
                normalized = self._normalize_backbone_name(backbone)
                
                if normalized not in by_type:
                    by_type[normalized] = []
                
                by_type[normalized].append(checkpoint)
            
            return {
                'success': True,
                'total_found': len(discovered_checkpoints),
                'by_type': by_type,
                'mixin_used': True  # Indicate this uses proper mixin functionality
            }
            
        except Exception as e:
            self.log_error(f"Mixin-based model discovery failed: {e}")
            return {'success': False, 'error': str(e), 'mixin_used': True}

    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Initialize the Backbone module.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
            
        Returns:
            True if initialization was successful
        """
        try:
            # Set config if provided before initialization
            if config:
                self._user_config = config
            
            # Initialize using base class which handles everything
            success = super().initialize()
            
            if success:
                # Set UI components in config handler for extraction
                if self._config_handler and hasattr(self._config_handler, 'set_ui_components'):
                    self._config_handler.set_ui_components(self._ui_components)
                
                # Setup operation container reference for logging
                self._setup_operation_container()
                
                # Initialize operation service after UI components are ready
                self._operation_service = BackboneOperationService(ui_module=self, logger=self.logger)
                
                # Setup button dependencies for backbone-specific requirements
                self._setup_backbone_button_dependencies()
                
                # Register module-specific button handlers
                self._register_module_button_handlers()
                
                # Register rescan models button with BaseUIModule system
                self._register_rescan_button()
                
                # Setup rescan models button handler
                self._setup_rescan_button_handler()
                
                # Flush any buffered logs to operation container
                self._flush_log_buffer()
                
                # Perform initial model scan
                self._perform_initial_model_scan()
                
                # Log initialization completion (Operation Checklist 3.2)
                self.log("🧬 Backbone module siap digunakan", 'info')
                self.log("✅ Semua fitur backbone tersedia", 'success')
                
                # Log module readiness
                self.log("🏗️ Modul backbone siap digunakan", 'info')
                
                self.log_debug("✅ Backbone module initialization completed")
            
            return success
            
        except Exception as e:
            self.log_error(f"Failed to initialize Backbone module: {e}")
            return False

    def _operation_validate(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle validation operation with backend integration."""
        def validate_data():
            # Ensure UI components are ready first
            if not hasattr(self, '_ui_components') or not self._ui_components:
                return {'valid': False, 'message': "Komponen UI belum siap, silakan coba lagi"}
            
            # Check data prerequisites using operation service
            if self._operation_service:
                prereq_check = self._operation_service.validate_data_prerequisites()
                if not prereq_check.get('prerequisites_ready', False):
                    return {'valid': False, 'message': f"Prerequisites missing: {prereq_check.get('message', 'Unknown error')}"}
            else:
                return {'valid': False, 'message': "Operation service not available"}
            
            return {'valid': True}
        
        def execute_validate():
            self.log("🔍 Memulai validasi konfigurasi backbone...", 'info')
            return self._execute_validate_operation()
        
        return self._execute_operation_with_wrapper(
            operation_name="Validasi Backbone",
            operation_func=execute_validate,
            button=button,
            validation_func=validate_data,
            success_message="Validasi backbone berhasil diselesaikan",
            error_message="Kesalahan validasi backbone"
        )

    def _operation_build(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle build operation with backend integration."""
        def validate_build():
            # Ensure UI components are ready first
            if not hasattr(self, '_ui_components') or not self._ui_components:
                return {'valid': False, 'message': "Komponen UI belum siap, silakan coba lagi"}
            
            # Check data prerequisites using operation service
            if self._operation_service:
                prereq_check = self._operation_service.validate_data_prerequisites()
                if not prereq_check.get('prerequisites_ready', False):
                    return {'valid': False, 'message': f"Prerequisites missing: {prereq_check.get('message', 'Unknown error')}"}
            else:
                return {'valid': False, 'message': "Operation service not available"}
            
            return {'valid': True}
        
        def execute_build():
            self.log("🏗️ Memulai pembangunan model backbone...", 'info')
            
            # Disable UI during operation
            self._disable_ui_during_operation()
            
            try:
                result = self._execute_build_operation()
                
                # Update model status if successful
                if result.get('success'):
                    self._update_built_model_indicators()
                    
                return result
            finally:
                # Re-enable UI after operation completes
                self._enable_ui_after_operation()
        
        return self._execute_operation_with_wrapper(
            operation_name="Pembangunan Model",
            operation_func=execute_build,
            button=button,
            validation_func=validate_build,
            success_message="Pembangunan model berhasil diselesaikan",
            error_message="Kesalahan pembangunan model"
        )

    # Data validation methods removed - now handled by backend API via operation service

    # ==================== OPERATION EXECUTION METHODS ====================

    def _execute_validate_operation(self) -> Dict[str, Any]:
        """Execute the validation operation using operation handler."""
        try:
            from .operations.backbone_validate_operation import BackboneValidateOperationHandler
            
            # Create handler with current UI components and config
            handler = BackboneValidateOperationHandler(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={'on_success': self._update_operation_summary}
            )
            
            # Execute the operation
            result = handler.execute()
            
            # Return standardized result
            if result and result.get('success'):
                return {'success': True, 'message': 'Validasi berhasil diselesaikan'}
            else:
                error_msg = result.get('message', 'Validasi gagal') if result else 'Validasi gagal'
                return {'success': False, 'message': error_msg}
            
        except Exception as e:
            return {'success': False, 'message': f"Error in validation operation: {e}"}

    def _execute_build_operation(self) -> Dict[str, Any]:
        """Execute the build operation using operation handler."""
        try:
            from .operations.backbone_build_operation import BackboneBuildOperationHandler
            
            # Create handler with current UI components and config
            handler = BackboneBuildOperationHandler(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={'on_success': self._update_operation_summary}
            )
            
            # Execute the operation
            result = handler.execute()
            
            # Return standardized result
            if result and result.get('success'):
                return {'success': True, 'message': 'Pembangunan model berhasil diselesaikan'}
            else:
                error_msg = result.get('message', 'Pembangunan model gagal') if result else 'Pembangunan model gagal'
                return {'success': False, 'message': error_msg}
            
        except Exception as e:
            return {'success': False, 'message': f"Error in build operation: {e}"}

    def _update_operation_summary(self, content: str) -> None:
        """Updates the operation summary container with new content."""
        try:
            # Try to get operation container summary updater
            operation_container = self._ui_components.get('operation_container')
            if operation_container and isinstance(operation_container, dict):
                # Check for summary update function
                if 'update_summary' in operation_container:
                    operation_container['update_summary'](content)
                    self.log("✅ Ringkasan operasi diperbarui", 'debug')
                    return
                
                # Check for summary container in operation container
                if 'summary_container' in operation_container:
                    summary_container = operation_container['summary_container']
                    if hasattr(summary_container, 'update_content'):
                        summary_container.update_content({'title': 'Operation Summary', 'content': content})
                        self.log("✅ Ringkasan operasi diperbarui melalui summary container", 'debug')
                        return
            
            # Fallback: use summary container directly
            summary_container = self._ui_components.get('summary_container')
            if summary_container and hasattr(summary_container, 'update_content'):
                summary_container.update_content({'title': 'Operation Summary', 'content': content})
                self.log("✅ Ringkasan operasi diperbarui melalui summary container utama", 'debug')
                return
            
            # If all else fails, just log the content
            self.log("📝 " + content, 'info')
            
        except Exception as e:
            self.log_error(f"Failed to update operation summary: {e}")
            # Fallback: just log the content
            self.log("📝 " + content, 'info')

    def _flush_log_buffer(self) -> None:
        """Flush buffered logs to operation container."""
        try:
            if not hasattr(self, '_log_buffer') or not self._log_buffer:
                return
                
            # Ensure operation container is available
            if not hasattr(self, '_operation_container') or not self._operation_container:
                self.log_warning("⚠️ Operation container not available for log buffer flush")
                return
            
            # Flush all buffered logs
            for log_entry in self._log_buffer:
                message, level = log_entry
                self.log(message, level)
            
            # Clear the buffer after flushing
            buffered_logs = len(self._log_buffer)
            self._log_buffer.clear()
            self.log_debug(f"✅ Flushed {buffered_logs} logs to operation container")
            
        except Exception as e:
            self.log_error(f"Failed to flush log buffer: {e}")

    def get_ui_components(self) -> Dict[str, Any]:
        """
        Get UI components dictionary.
        
        Returns:
            UI components dictionary
        """
        return self._ui_components or {}

    def _handle_save_config(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle save configuration button click."""
        try:
            self.log("💾 Menyimpan konfigurasi backbone...", 'info')
            result = self.save_config()
            if result.get('success', True):
                self.log("✅ Konfigurasi backbone berhasil disimpan", 'success')
                return {'success': True, 'message': 'Configuration saved successfully'}
            else:
                self.log("❌ Gagal menyimpan konfigurasi backbone", 'error')
                return {'success': False, 'message': result.get('message', 'Save failed')}
        except Exception as e:
            self.log(f"❌ Error menyimpan konfigurasi: {e}", 'error')
            return {'success': False, 'message': str(e)}

    def _handle_reset_config(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle reset configuration button click."""
        try:
            self.log("🔄 Reset konfigurasi backbone ke default...", 'info')
            result = self.reset_config()
            if result.get('success', True):
                self.log("✅ Konfigurasi backbone berhasil direset", 'success')
                return {'success': True, 'message': 'Configuration reset successfully'}
            else:
                self.log("❌ Gagal reset konfigurasi backbone", 'error')
                return {'success': False, 'message': result.get('message', 'Reset failed')}
        except Exception as e:
            self.log(f"❌ Error reset konfigurasi: {e}", 'error')
            return {'success': False, 'message': str(e)}

    def _disable_ui_during_operation(self) -> None:
        """Disable UI components during operations to prevent interference."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                widgets = self._ui_components.get('widgets', {})
                
                # Disable form widgets
                for widget_name, widget in widgets.items():
                    if hasattr(widget, 'disabled'):
                        widget.disabled = True
                        self.log_debug(f"Disabled widget: {widget_name}")
                
                # Disable action buttons
                action_container = self._ui_components.get('action_container')
                if action_container:
                    # Try multiple methods to disable buttons
                    if isinstance(action_container, dict):
                        # Method 1: Use disable_buttons function if available
                        if 'disable_buttons' in action_container:
                            action_container['disable_buttons'](['validate', 'build', 'save', 'reset'])
                        
                        # Method 2: Direct button access
                        if 'buttons' in action_container:
                            buttons = action_container['buttons']
                            if hasattr(buttons, 'children'):
                                for button in buttons.children:
                                    if hasattr(button, 'disabled'):
                                        button.disabled = True
                        
                        # Method 3: Container-level disable
                        if 'container' in action_container:
                            container = action_container['container']
                            if hasattr(container, 'children'):
                                for child in container.children:
                                    if hasattr(child, 'disabled'):
                                        child.disabled = True
                    
                    # Method 4: Direct widget access if action_container is a widget
                    elif hasattr(action_container, 'children'):
                        for child in action_container.children:
                            if hasattr(child, 'disabled'):
                                child.disabled = True
                
                self.log("🔒 UI disabled during operation", 'info')
                
        except Exception as e:
            self.log_error(f"Failed to disable UI: {e}")

    def _enable_ui_after_operation(self) -> None:
        """Re-enable UI components after operations complete."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                widgets = self._ui_components.get('widgets', {})
                
                # Re-enable form widgets
                for widget_name, widget in widgets.items():
                    if hasattr(widget, 'disabled'):
                        widget.disabled = False
                        self.log_debug(f"Enabled widget: {widget_name}")
                
                # Re-enable action buttons
                action_container = self._ui_components.get('action_container')
                if action_container:
                    # Try multiple methods to enable buttons
                    if isinstance(action_container, dict):
                        # Method 1: Use enable_buttons function if available
                        if 'enable_buttons' in action_container:
                            action_container['enable_buttons'](['validate', 'build', 'save', 'reset'])
                        
                        # Method 2: Direct button access
                        if 'buttons' in action_container:
                            buttons = action_container['buttons']
                            if hasattr(buttons, 'children'):
                                for button in buttons.children:
                                    if hasattr(button, 'disabled'):
                                        button.disabled = False
                        
                        # Method 3: Container-level enable
                        if 'container' in action_container:
                            container = action_container['container']
                            if hasattr(container, 'children'):
                                for child in container.children:
                                    if hasattr(child, 'disabled'):
                                        child.disabled = False
                    
                    # Method 4: Direct widget access if action_container is a widget
                    elif hasattr(action_container, 'children'):
                        for child in action_container.children:
                            if hasattr(child, 'disabled'):
                                child.disabled = False
                
                self.log("🔓 UI re-enabled after operation", 'info')
                
        except Exception as e:
            self.log_error(f"Failed to enable UI: {e}")

    def _check_built_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Check for built models using operation service.
        
        Returns:
            Dictionary mapping backbone types to lists of model information
        """
        try:
            if self._operation_service:
                scan_result = self._operation_service.rescan_built_models()
                if scan_result.get('success', False):
                    return scan_result.get('by_backbone', {})
                else:
                    self.log_debug(f"Model scan failed: {scan_result.get('message', 'Unknown error')}")
                    return {}
            else:
                self.log_warning("Operation service not available for model checking")
                return {}
        except Exception as e:
            self.log_error(f"Failed to check built models: {e}")
            return {}
    
    def _update_built_model_indicators(self) -> None:
        """Update UI to show built model indicators."""
        try:
            # Check for built models using operation service
            built_models = self._check_built_models()
            
            # Get current backbone type safely
            current_config = self.get_current_config()
            backbone_type = current_config.get('backbone', {}).get('model_type', 'efficientnet_b4')
            
            # Calculate model statistics
            model_count = len(built_models.get(backbone_type, []))
            status_text = f"✅ Built" if model_count > 0 else "⚠️ Not Built"
            last_built = "Recently" if model_count > 0 else "Never"
            available_text = f"{model_count} model(s) found" if model_count > 0 else "No models found"
            
            # Update the display using the existing method
            self._update_model_status_display(status_text, last_built, available_text)
            
            self.log(f"📊 Model indicators updated: {available_text}", 'info')
                
        except Exception as e:
            self.log_error(f"Failed to update built model indicators: {e}")

    # Model discovery methods removed - now handled by backend API via operation service

    def _register_rescan_button(self) -> None:
        """Register rescan models button with BaseUIModule system."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                # Try multiple ways to find the rescan button
                rescan_button = None
                
                # Method 1: Direct UI component reference (stored in backbone_ui.py line 133)
                rescan_button = self._ui_components.get('rescan_models')
                
                # Method 2: Check in widgets dict with different naming patterns
                if not rescan_button:
                    widgets = self._ui_components.get('widgets', {})
                    rescan_button = widgets.get('rescan_models_button') or widgets.get('rescan_models')
                
                # Method 3: Check in action_container buttons directly
                if not rescan_button:
                    action_container = self._ui_components.get('action_container')
                    if action_container and isinstance(action_container, dict):
                        buttons = action_container.get('buttons', {})
                        rescan_button = buttons.get('rescan_models')
                
                if rescan_button:
                    # Register with BaseUIModule button system
                    self.register_button_handler('rescan_models', self._handle_rescan_models)
                    self.log_debug("✅ Rescan models button registered with BaseUIModule")
                else:
                    self.log_warning("Rescan models button widget not found")
        except Exception as e:
            self.log_error(f"Failed to register rescan button: {e}")

    def _setup_rescan_button_handler(self) -> None:
        """Setup the rescan models button click handler."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                # Try multiple ways to find the rescan button
                rescan_button = None
                
                # Method 1: Direct UI component reference (stored in backbone_ui.py line 133)
                rescan_button = self._ui_components.get('rescan_models')
                
                # Method 2: Check in widgets dict with different naming patterns
                if not rescan_button:
                    widgets = self._ui_components.get('widgets', {})
                    rescan_button = widgets.get('rescan_models_button') or widgets.get('rescan_models')
                
                # Method 3: Check in action_container buttons directly
                if not rescan_button:
                    action_container = self._ui_components.get('action_container')
                    if action_container and isinstance(action_container, dict):
                        buttons = action_container.get('buttons', {})
                        rescan_button = buttons.get('rescan_models')
                
                if rescan_button and hasattr(rescan_button, 'on_click'):
                    rescan_button.on_click(self._handle_rescan_models)
                    self.log_debug("✅ Rescan models button handler registered")
                else:
                    self.log_warning("Rescan models button not found or invalid")
        except Exception as e:
            self.log_error(f"Failed to setup rescan button handler: {e}")

    def _handle_rescan_models(self, _=None) -> None:
        """Handle rescan models button click using operation service."""
        try:
            # Suppress button registration logs during discovery
            self._suppress_button_logs = True
            
            self.log("🔄 Rescanning for existing models...", 'info')
            
            if self._operation_service:
                # Use operation service for model rescanning
                result = self._operation_service.rescan_built_models()
                
                if result.get('success'):
                    total_models = result.get('total_models', 0)
                    self.log(f"✅ Model rescan completed - found {total_models} models", 'success')
                    
                    # Update UI display based on results
                    self._update_ui_from_scan_results(result)
                else:
                    error_msg = result.get('discovery_summary', 'Unknown error')
                    self.log(f"❌ Model rescan failed: {error_msg}", 'error')
            else:
                self.log("❌ Operation service not available", 'error')
                
        except Exception as e:
            self.log(f"❌ Error during model rescan: {e}", 'error')
            # Log additional debug information
            self.log_error(f"Rescan error details: {e}", exc_info=True)
        finally:
            # Always reset the log suppression flag
            self._suppress_button_logs = False

    def _perform_initial_model_scan(self) -> None:
        """Perform initial model scan during module initialization using operation service."""
        try:
            self.log("🔍 Scanning for existing backbone models...", 'info')
            
            if self._operation_service:
                result = self._operation_service.rescan_built_models()
                if result.get('success'):
                    total_models = result.get('total_models', 0)
                    self.log(f"✅ Found {total_models} existing models", 'info')
                    self._update_ui_from_scan_results(result)
                else:
                    self.log("⚠️ No existing models found", 'warning')
            else:
                self.log("⚠️ Operation service not available for initial scan", 'warning')
                
        except Exception as e:
            self.log_error(f"Failed to perform initial model scan: {e}")
            self.log("⚠️ Failed to scan for existing models", 'warning')
    
    def _update_ui_from_scan_results(self, scan_result: Dict[str, Any]) -> None:
        """
        Update UI components based on model scan results from backend API.
        
        Args:
            scan_result: Results from backend API model discovery
        """
        try:
            by_backbone = scan_result.get('by_backbone', {})
            total_models = scan_result.get('total_models', 0)
            
            # Get current backbone type
            current_config = self.get_current_config()
            backbone_type = current_config.get('backbone', {}).get('model_type', 'efficientnet_b4')
            
            # Count models for current backbone
            current_backbone_models = by_backbone.get(backbone_type, [])
            model_count = len(current_backbone_models)
            
            # Determine status and button states
            has_models = model_count > 0
            if has_models:
                status_text = "✅ Built"
                last_built = "Recently"
                available_text = f"{model_count} model(s) found"
            else:
                status_text = "⚠️ Not Built"
                last_built = "Never"
                available_text = "No models found"
            
            # Update button states using enhanced mixin functionality
            button_conditions = {
                'validate': has_models,
                # Other buttons can be added here as needed
            }
            
            button_reasons = {
                'validate': "No built models available" if not has_models else None,
            }
            
            self.update_button_states_based_on_condition(button_conditions, button_reasons)
            
            # Update UI indicators
            self._update_model_status_display(status_text, last_built, available_text)
            
            # Log results
            if total_models > 0:
                self.log_debug(f"Found {total_models} built models across all backbones")
                self.log_debug(f"Current backbone ({backbone_type}): {model_count} models")
            else:
                self.log_debug("No built models found")
                
        except Exception as e:
            self.log_error(f"Failed to update UI from scan results: {e}")
            # Set error state
            self._update_model_status_display("❌ Error", "Scan Failed", "Error processing results")
            
            # Disable all buttons in error state
            self.update_button_states_based_on_condition(
                {'validate': False, 'build': False},
                {'validate': "Error scanning models", 'build': "Error scanning models"}
            )

    # Old model scan method removed - now handled by backend API via operation service

    def _update_model_status_display(self, status: str, last_built: str, available: str) -> None:
        """Update the model status display in the UI with enhanced error handling."""
        try:
            from IPython.display import Javascript, display
            
            # Sanitize inputs to prevent JavaScript injection
            status = str(status).replace("'", "\\'").replace('"', '\\"')
            last_built = str(last_built).replace("'", "\\'").replace('"', '\\"')
            available = str(available).replace("'", "\\'").replace('"', '\\"')
            
            js_code = f"""
            try {{
                if (document.getElementById('model-status')) {{
                    document.getElementById('model-status').innerText = '{status}';
                }}
                if (document.getElementById('last-built')) {{
                    document.getElementById('last-built').innerText = '{last_built}';
                }}
                if (document.getElementById('available-models')) {{
                    document.getElementById('available-models').innerText = '{available}';
                }}
            }} catch (e) {{
                console.log('Model status update elements not found:', e);
            }}
            """
            display(Javascript(js_code))
            self.log_debug(f"Model status display updated: {available}")
            
        except Exception as e:
            self.log_error(f"Failed to update model status display: {e}")

    def _setup_backbone_button_dependencies(self) -> None:
        """
        Setup button dependencies specific to backbone module requirements.
        
        This uses the enhanced button mixin functionality to define when
        buttons should be enabled/disabled based on module state.
        """
        try:
            # Validate button depends on having built models
            self.set_button_dependency('validate', self._check_validate_button_dependency)
            
            # Build button depends on data prerequisites  
            self.set_button_dependency('build', self._check_build_button_dependency)
            
            self.log_debug("✅ Backbone button dependencies configured")
            
        except Exception as e:
            self.log_error(f"Failed to setup backbone button dependencies: {e}")
    
    def _check_validate_button_dependency(self) -> bool:
        """
        Check if validate button should be enabled.
        
        Returns:
            True if validate button should be enabled (models are available)
        """
        try:
            if not self._operation_service:
                return False
                
            # Quick check: if we have any built models, enable validate
            result = self._operation_service.rescan_built_models()
            return result.get('success', False) and result.get('total_models', 0) > 0
            
        except Exception as e:
            self.log_debug(f"Validate button dependency check failed: {e}")
            return False
    
    def _check_build_button_dependency(self) -> bool:
        """
        Check if build button should be enabled.
        
        Returns:
            True if build button should be enabled (data prerequisites are met)
        """
        try:
            if not self._operation_service:
                return False
                
            # Check data prerequisites
            result = self._operation_service.validate_data_prerequisites()
            return result.get('prerequisites_ready', False)
            
        except Exception as e:
            self.log_debug(f"Build button dependency check failed: {e}")
            return False
    
    def _update_validate_button_visibility(self, visible: bool) -> None:
        """Update validate button visibility using enhanced button mixin functionality."""
        try:
            reason = "No built models available" if not visible else None
            self.set_button_visibility('validate', visible, reason)
            
            self.log_debug(f"Validate button visibility set to: {visible}")
        except Exception as e:
            self.log_error(f"Failed to update validate button visibility: {e}")

    def cleanup(self) -> None:
        """Widget lifecycle cleanup - optimization.md compliance."""
        try:
            # Cleanup UI components if they have cleanup methods
            if hasattr(self, '_ui_components') and self._ui_components:
                # Call component-specific cleanup if available
                if hasattr(self._ui_components, '_cleanup'):
                    self._ui_components._cleanup()
                
                # Close individual widgets
                for component in self._ui_components.values():
                    if hasattr(component, 'close'):
                        try:
                            component.close()
                        except Exception:
                            pass  # Ignore cleanup errors
            
            # Call parent cleanup
            if hasattr(super(), 'cleanup'):
                super().cleanup()
            
            # Minimal logging for cleanup completion
            if hasattr(self, 'logger'):
                self.logger.info("Backbone module cleanup completed")
                
        except Exception as e:
            # Critical errors always logged
            if hasattr(self, 'logger'):
                self.logger.error(f"Backbone module cleanup failed: {e}")

    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during deletion

def get_backbone_uimodule(auto_initialize: bool = True) -> BackboneUIModule:
    """
    Factory function to get a BackboneUIModule instance.
    
    Args:
        auto_initialize: Whether to automatically initialize the module
        
    Returns:
        BackboneUIModule instance
    """
    module = BackboneUIModule()
    if auto_initialize:
        module.initialize()
    return module
