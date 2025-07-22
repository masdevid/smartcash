# -*- coding: utf-8 -*-
"""
File: smartcash/ui/model/backbone/backbone_uimodule.py
Description: Refactored implementation of the Backbone Module using the modern BaseUIModule pattern.
"""

from typing import Dict, Any, Optional

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


class BackboneUIModule(ModelDiscoveryMixin, ModelConfigSyncMixin, BackendServiceMixin, BaseUIModule):
    """
    Backbone UI Module.
    """
    # Define required UI components at class level
    _required_components = [
        'main_container',
        'header_container',
        'form_container',
        'action_container',
        'operation_container'
    ]

    def __init__(self, enable_environment: bool = False):
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
        
        self.log_debug("BackboneUIModule initialized.")

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
        try:
            # Get module-specific handlers
            module_handlers = self._get_module_button_handlers()
            
            # Register each handler
            for button_id, handler in module_handlers.items():
                self.register_button_handler(button_id, handler)
                self.log_debug(f"✅ Registered backbone button handler: {button_id}")
            
            # Setup button handlers after registering them
            self._setup_button_handlers()
            
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
            
            # Check data prerequisites
            prereq_check = self._check_data_prerequisites()
            if not prereq_check['success']:
                return {'valid': False, 'message': f"Prerequisites missing: {prereq_check['message']}"}
            
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
            
            # Check data prerequisites
            prereq_check = self._check_data_prerequisites()
            if not prereq_check['success']:
                return {'valid': False, 'message': f"Prerequisites missing: {prereq_check['message']}"}
            
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

    def _check_data_prerequisites(self) -> Dict[str, Any]:
        """Check if all required data is available."""
        try:
            missing = []
            warnings = []
            
            # Check pretrained models
            pretrained_check = self._check_pretrained_models()
            if not pretrained_check['available']:
                warnings.append("Pretrained models not found")
            
            # Check raw data using preprocessor API
            raw_data_check = self._check_raw_data()
            if not raw_data_check['available']:
                missing.append("Raw data not found")
            
            # Check preprocessed data
            preprocessed_check = self._check_preprocessed_data()
            if not preprocessed_check['available']:
                warnings.append("Preprocessed data not found")
            
            if missing:
                return {
                    'success': False,
                    'message': f"Missing required data: {', '.join(missing)}",
                    'missing': missing,
                    'warnings': warnings
                }
            elif warnings:
                return {
                    'success': True,
                    'message': f"Some optional data missing: {', '.join(warnings)}",
                    'missing': [],
                    'warnings': warnings
                }
            else:
                return {
                    'success': True,
                    'message': "All data prerequisites available",
                    'missing': [],
                    'warnings': []
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Error checking prerequisites: {e}",
                'missing': [],
                'warnings': []
            }

    def _check_pretrained_models(self) -> Dict[str, Any]:
        """Check if pretrained models are available."""
        try:
            from pathlib import Path
            
            pretrained_dir = Path('/data/pretrained')
            backbone_config = self.get_current_config().get('backbone', {})
            backbone_type = backbone_config.get('model_type', 'efficientnet_b4')
            
            # Check for backbone-specific pretrained models
            if backbone_type == 'efficientnet_b4':
                efficientnet_dir = pretrained_dir / 'efficientnet'
                if efficientnet_dir.exists():
                    model_files = list(efficientnet_dir.glob('*.pth')) + list(efficientnet_dir.glob('*.pt'))
                    if model_files:
                        return {'available': True, 'path': str(efficientnet_dir), 'files': len(model_files)}
            elif backbone_type == 'cspdarknet':
                yolov5_dir = pretrained_dir / 'yolov5'
                if yolov5_dir.exists():
                    model_files = list(yolov5_dir.glob('*.pt'))
                    if model_files:
                        return {'available': True, 'path': str(yolov5_dir), 'files': len(model_files)}
            
            return {'available': False, 'path': str(pretrained_dir), 'files': 0}
            
        except Exception as e:
            self.log_error(f"Error checking pretrained models: {e}")
            return {'available': False, 'path': '', 'files': 0}

    def _check_raw_data(self) -> Dict[str, Any]:
        """Check raw data using preprocessor API."""
        try:
            # Use preprocessor API to check raw data
            from smartcash.dataset.preprocessor.api.preprocessing_api import get_preprocessing_status
            
            status = get_preprocessing_status(config=self.get_current_config())
            if status.get('success') and status.get('service_ready'):
                file_stats = status.get('file_statistics', {})
                
                # Check if any split has raw images
                total_raw = sum(
                    split_data.get('raw_images', 0) 
                    for split_data in file_stats.values()
                )
                
                return {
                    'available': total_raw > 0,
                    'total_files': total_raw,
                    'details': file_stats
                }
            
            return {'available': False, 'total_files': 0, 'details': {}}
            
        except Exception as e:
            self.log_error(f"Error checking raw data: {e}")
            return {'available': False, 'total_files': 0, 'details': {}}

    def _check_preprocessed_data(self) -> Dict[str, Any]:
        """Check preprocessed data using preprocessor API."""
        try:
            # Use preprocessor API to check preprocessed data
            from smartcash.dataset.preprocessor.api.preprocessing_api import get_preprocessing_status
            
            status = get_preprocessing_status(config=self.get_current_config())
            if status.get('success') and status.get('service_ready'):
                file_stats = status.get('file_statistics', {})
                
                # Check if any split has preprocessed files
                total_preprocessed = sum(
                    split_data.get('preprocessed_files', 0) 
                    for split_data in file_stats.values()
                )
                
                return {
                    'available': total_preprocessed > 0,
                    'total_files': total_preprocessed,
                    'details': file_stats
                }
            
            return {'available': False, 'total_files': 0, 'details': {}}
            
        except Exception as e:
            self.log_error(f"Error checking preprocessed data: {e}")
            return {'available': False, 'total_files': 0, 'details': {}}

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

    def _update_built_model_indicators(self) -> None:
        """Update UI to show built model indicators."""
        try:
            # Check for built models
            built_models = self._check_built_models()
            
            if hasattr(self, '_ui_components') and self._ui_components:
                widgets = self._ui_components.get('widgets', {})
                
                # Update model status info if available
                if 'model_status_info' in widgets:
                    status_widget = widgets['model_status_info']
                    if hasattr(status_widget, 'value'):
                        # Update the HTML content to show built model status
                        current_config = self.get_current_config()
                        backbone_type = current_config.get('backbone', {}).get('model_type', 'efficientnet_b4')
                        
                        model_count = len(built_models.get(backbone_type, []))
                        status_text = f"✅ Built" if model_count > 0 else "⚠️ Not Built"
                        last_built = "Recently" if model_count > 0 else "Never"
                        available_text = f"{model_count} model(s) found" if model_count > 0 else "No models found"
                        
                        # Use JavaScript to update the spans
                        from IPython.display import Javascript, display
                        js_code = f"""
                        if (document.getElementById('model-status')) {{
                            document.getElementById('model-status').innerText = '{status_text}';
                            document.getElementById('last-built').innerText = '{last_built}';
                            document.getElementById('available-models').innerText = '{available_text}';
                        }}
                        """
                        display(Javascript(js_code))
                        
                        self.log(f"📊 Model indicators updated: {available_text}", 'info')
                
        except Exception as e:
            self.log_error(f"Failed to update built model indicators: {e}")

    def _check_built_models(self) -> Dict[str, Any]:
        """Check for built models by backbone type using mixin's configurable checkpoint discovery."""
        try:
            # Use ModelDiscoveryMixin's discover_checkpoints method
            discovered_checkpoints = self.discover_checkpoints(
                discovery_paths=[
                    'data/checkpoints',                     # Primary checkpoint directory
                    'runs/train/*/weights',                 # YOLOv5 training output pattern  
                    'experiments/*/checkpoints'             # Alternative experiment directory
                ],
                filename_patterns=[
                    'best_*.pt',                           # Standard best model format
                    'last.pt',                             # Latest checkpoint fallback
                    'epoch_*.pt'                           # Epoch-based checkpoints
                ],
                validation_requirements={
                    'min_val_map': 0.3,
                    'supported_backbones': ['cspdarknet', 'efficientnet_b4'],
                    'required_keys': ['model_state_dict', 'config']
                }
            )
            
            # Group by backbone type
            built_models = {}
            for checkpoint in discovered_checkpoints:
                backbone_type = checkpoint.get('backbone', 'unknown')
                if backbone_type not in built_models:
                    built_models[backbone_type] = []
                built_models[backbone_type].append(checkpoint['path'])
            
            self.log_debug(f"Found {len(discovered_checkpoints)} built models across {len(built_models)} backbone types")
            return built_models
            
        except Exception as e:
            self.log_error(f"Failed to check built models: {e}")
            return {}

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

    def _handle_rescan_models(self, button=None) -> None:
        """Handle rescan models button click."""
        try:
            self.log("🔄 Rescanning for existing models...", 'info')
            self._perform_model_scan()
        except Exception as e:
            self.log(f"❌ Error during model rescan: {e}", 'error')

    def _perform_initial_model_scan(self) -> None:
        """Perform initial model scan during module initialization."""
        try:
            self.log("🔍 Scanning for existing backbone models...", 'info')
            self._perform_model_scan()
        except Exception as e:
            self.log_error(f"Failed to perform initial model scan: {e}")
            self.log("⚠️ Failed to scan for existing models", 'warning')

    def _perform_model_scan(self) -> None:
        """Perform model scan and update UI indicators."""
        try:
            # Check for built models
            built_models = self._check_built_models()
            
            # Get current backbone type
            current_config = self.get_current_config()
            backbone_type = current_config.get('backbone', {}).get('model_type', 'efficientnet_b4')
            
            # Count models for current backbone
            current_backbone_models = built_models.get(backbone_type, [])
            model_count = len(current_backbone_models)
            
            # Determine status
            if model_count > 0:
                status_text = "✅ Built"
                last_built = "Recently"
                available_text = f"{model_count} model(s) found"
                status_level = 'success'
                
                # Update validate button visibility
                self._update_validate_button_visibility(True)
            else:
                status_text = "⚠️ Not Built"
                last_built = "Never"
                available_text = "No models found"
                status_level = 'warning'
                
                # Hide validate button if no models
                self._update_validate_button_visibility(False)
            
            # Update UI indicators
            self._update_model_status_display(status_text, last_built, available_text)
            
            # Log results
            total_models = sum(len(models) for models in built_models.values())
            if total_models > 0:
                self.log(f"✅ Found {total_models} built models across all backbones", status_level)
                self.log(f"📊 Current backbone ({backbone_type}): {model_count} models", 'info')
            else:
                self.log("❌ No built models found - build models first", 'warning')
                
        except Exception as e:
            self.log_error(f"Failed to perform model scan: {e}")
            self.log("❌ Model scan failed", 'error')
            # Set error state
            self._update_model_status_display("❌ Error", "Scan Failed", "Error scanning models")

    def _update_model_status_display(self, status: str, last_built: str, available: str) -> None:
        """Update the model status display in the UI."""
        try:
            from IPython.display import Javascript, display
            js_code = f"""
            if (document.getElementById('model-status')) {{
                document.getElementById('model-status').innerText = '{status}';
                document.getElementById('last-built').innerText = '{last_built}';
                document.getElementById('available-models').innerText = '{available}';
            }}
            """
            display(Javascript(js_code))
        except Exception as e:
            self.log_error(f"Failed to update model status display: {e}")

    def _update_validate_button_visibility(self, visible: bool) -> None:
        """Update validate button visibility based on model availability."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                action_container = self._ui_components.get('action_container')
                if action_container and isinstance(action_container, dict):
                    # Update button visibility through action container
                    if 'update_button_visibility' in action_container:
                        action_container['update_button_visibility']('validate', visible)
                    elif 'buttons' in action_container:
                        # Fallback: directly disable/enable button
                        buttons = action_container['buttons']
                        # Check if buttons is a dict (button_id: button_widget) or list
                        if isinstance(buttons, dict):
                            # Handle dict structure
                            if 'validate' in buttons:
                                validate_button = buttons['validate']
                                if hasattr(validate_button, 'disabled'):
                                    validate_button.disabled = not visible
                        elif isinstance(buttons, list):
                            # Handle list structure
                            for button_info in buttons:
                                # Check if button_info is a dict with 'id' key
                                if isinstance(button_info, dict) and button_info.get('id') == 'validate':
                                    button_info['disabled'] = not visible
                                    break
                                
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
                for component_name, component in self._ui_components.items():
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
