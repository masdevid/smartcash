"""
File: smartcash/ui/model/backbone/handlers/backbone_ui_handler.py
Description: UI handler for backbone management following dependency pattern
"""

from typing import Dict, Any, Optional
import asyncio
from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from ..configs.backbone_config_handler import BackboneConfigHandler
from .operation_manager import BackboneOperationManager
from ..constants import BackboneOperation


class BackboneUIHandler(ModuleUIHandler):
    """Handler for backbone UI management following dependency pattern."""
    
    def __init__(self, module_name: str = 'backbone', parent_module: str = 'model', **kwargs):
        """Initialize backbone UI handler.
        
        Args:
            module_name: Module name
            parent_module: Parent module name
            **kwargs: Additional arguments
        """
        super().__init__(module_name, parent_module)
        self.config_handler = BackboneConfigHandler()
        self.operation_manager = None  # Will be initialized with operation container
        self._backbone_factory = None
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract configuration from UI components."""
        try:
            config = self.config_handler.get_config().copy()
            
            # Extract from form components
            if 'backbone_dropdown' in self._ui_components:
                backbone_value = self._ui_components['backbone_dropdown'].value
                config['model']['backbone'] = backbone_value
            
            if 'detection_layers_select' in self._ui_components:
                layers_value = list(self._ui_components['detection_layers_select'].value)
                config['model']['detection_layers'] = layers_value
            
            if 'layer_mode_dropdown' in self._ui_components:
                mode_value = self._ui_components['layer_mode_dropdown'].value
                config['model']['layer_mode'] = mode_value
            
            if 'feature_optimization_checkbox' in self._ui_components:
                opt_enabled = self._ui_components['feature_optimization_checkbox'].value
                config['model']['feature_optimization']['enabled'] = opt_enabled
                config['model']['feature_optimization']['use_attention'] = opt_enabled
            
            if 'mixed_precision_checkbox' in self._ui_components:
                mixed_precision = self._ui_components['mixed_precision_checkbox'].value
                config['model']['mixed_precision'] = mixed_precision
            
            # Extract advanced options if available
            if 'advanced_options_accordion' in self._ui_components:
                accordion = self._ui_components['advanced_options_accordion']
                if hasattr(accordion, 'selected_index') and accordion.selected_index is not None:
                    config['ui']['show_advanced_options'] = True
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting config from UI: {e}")
            return self.config_handler.get_config()
    
    def update_ui_from_config(self, config: Dict[str, Any]) -> None:
        """Update UI components from configuration."""
        try:
            model_config = config.get('model', {})
            
            # Update backbone dropdown
            if 'backbone_dropdown' in self._ui_components:
                backbone = model_config.get('backbone', 'efficientnet_b4')
                self._ui_components['backbone_dropdown'].value = backbone
            
            # Update detection layers
            if 'detection_layers_select' in self._ui_components:
                layers = model_config.get('detection_layers', ['banknote'])
                self._ui_components['detection_layers_select'].value = tuple(layers)
            
            # Update layer mode
            if 'layer_mode_dropdown' in self._ui_components:
                mode = model_config.get('layer_mode', 'single')
                self._ui_components['layer_mode_dropdown'].value = mode
            
            # Update feature optimization
            if 'feature_optimization_checkbox' in self._ui_components:
                feature_opt = model_config.get('feature_optimization', {})
                enabled = feature_opt.get('enabled', True)
                self._ui_components['feature_optimization_checkbox'].value = enabled
            
            # Update mixed precision
            if 'mixed_precision_checkbox' in self._ui_components:
                mixed_precision = model_config.get('mixed_precision', True)
                self._ui_components['mixed_precision_checkbox'].value = mixed_precision
            
            # Update config summary if available
            if 'config_summary' in self._ui_components:
                self._update_config_summary(config)
            
            self.logger.info("✅ UI successfully updated from config")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating UI from config: {e}")
    
    def _update_config_summary(self, config: Dict[str, Any]) -> None:
        """Update configuration summary widget."""
        try:
            from ..components.config_summary import update_config_summary
            update_config_summary(self._ui_components['config_summary'], config)
        except ImportError:
            # Fallback if config_summary module is not available
            self.logger.warning("⚠️ Config summary module not available")
    
    def setup(self, ui_components: Dict[str, Any]) -> None:
        """Set up the handler with UI components.
        
        Args:
            ui_components: Dictionary of UI components to be managed by this handler
        """
        self.logger.info("🖥️ Setting up UI components for Backbone UI Handler")
        self._ui_components = ui_components
        
        # Initialize operation manager with operation container
        operation_container = ui_components.get('progress_tracker')
        self.operation_manager = BackboneOperationManager(operation_container)
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Initialize with default config
        self.sync_ui_with_config()
        
        self.logger.info("✅ UI components setup complete for Backbone UI Handler")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        # Form change handlers
        form_widgets = [
            'backbone_dropdown', 'detection_layers_select', 
            'layer_mode_dropdown', 'feature_optimization_checkbox',
            'mixed_precision_checkbox'
        ]
        
        for widget_name in form_widgets:
            if widget_name in self._ui_components:
                widget = self._ui_components[widget_name]
                widget.observe(
                    lambda change, w=widget_name: self._on_form_change(w, change), 
                    names='value'
                )
        
        # Button handlers - only include buttons that exist in the UI
        button_handlers = {
            'validate_btn': self._handle_validate_operation,
            'load_btn': self._handle_load_operation,
            'build_btn': self._handle_build_operation,
            'summary_btn': self._handle_summary_operation,
            'save_button': self._handle_save_config,
            'reset_button': self._handle_reset_config
        }
        
        # Log available UI components for debugging
        available_buttons = [name for name in button_handlers.keys() 
                           if name in self._ui_components and self._ui_components[name] is not None]
        self.logger.info(f"Available buttons: {available_buttons}")
        
        # Set up handlers for existing buttons
        for button_name, handler in button_handlers.items():
            button = self._ui_components.get(button_name)
            if button is not None and hasattr(button, 'on_click'):
                button.on_click(lambda b, h=handler: h())
            else:
                self.logger.warning(f"Button {button_name} not found or invalid in UI components")
        
        self.logger.info("✅ Event handlers setup complete")
    
    def _on_form_change(self, widget_name: str, change) -> None:
        """Handle form widget changes."""
        try:
            # Extract current config
            current_config = self.extract_config_from_ui()
            
            # Update config handler
            self.config_handler.update_config(current_config)
            
            # Update config summary
            if 'config_summary' in self._ui_components:
                self._update_config_summary(current_config)
            
            # Validate form state
            self._validate_form_state(current_config)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Form change handling error: {e}")
    
    def _validate_form_state(self, config: Dict[str, Any]) -> None:
        """Validate current form state and show warnings."""
        try:
            model_config = config.get('model', {})
            
            # Check layer mode compatibility
            layer_mode = model_config.get('layer_mode', 'single')
            detection_layers = model_config.get('detection_layers', [])
            
            if layer_mode == 'single' and len(detection_layers) > 1:
                self.track_status(
                    "⚠️ Single layer mode with multiple detection layers - consider multilayer mode",
                    "warning"
                )
            
            # Validate backbone availability
            backbone = model_config.get('backbone')
            if backbone:
                available_backbones = self.config_handler.get_available_backbones()
                if backbone not in available_backbones:
                    self.track_status(f"❌ Backbone '{backbone}' not available", "error")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Form validation error: {e}")
    
    def sync_config_with_ui(self) -> None:
        """Sync configuration with UI state."""
        try:
            # Extract current config from UI
            current_config = self.extract_config_from_ui()
            
            # Update config handler
            self.config_handler.update_config(current_config)
            
            self.logger.info("✅ Config successfully synced with UI")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing config with UI: {e}")
    
    def sync_ui_with_config(self) -> None:
        """Sync UI with configuration."""
        try:
            # Get current config
            current_config = self.config_handler.get_config()
            
            # Update UI
            self.update_ui_from_config(current_config)
            
            self.logger.info("✅ UI successfully synced with config")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing UI with config: {e}")
    
    def get_backbone_factory(self):
        """Get backbone factory instance."""
        if self._backbone_factory is None:
            try:
                import sys
                sys.path.append('.')
                from model.utils.backbone_factory import BackboneFactory
                self._backbone_factory = BackboneFactory()
                self.logger.info("✅ Backbone factory initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize backbone factory: {e}")
                self._backbone_factory = None
        return self._backbone_factory
    
    # === Operation Handlers ===
    
    def _handle_validate_operation(self) -> None:
        """Handle validate operation."""
        config = self.extract_config_from_ui()
        self._run_async_operation(BackboneOperation.VALIDATE.value, config)
    
    def _handle_load_operation(self) -> None:
        """Handle load operation."""
        config = self.extract_config_from_ui()
        self._run_async_operation(BackboneOperation.LOAD.value, config)
    
    def _handle_build_operation(self) -> None:
        """Handle build operation."""
        config = self.extract_config_from_ui()
        self._run_async_operation(BackboneOperation.BUILD.value, config)
    
    def _handle_summary_operation(self) -> None:
        """Handle summary operation."""
        config = self.extract_config_from_ui()
        self._run_async_operation(BackboneOperation.SUMMARY.value, config)
    
    def _run_async_operation(self, operation_type: str, config: Dict[str, Any]) -> None:
        """Run an async operation in a thread to avoid blocking the UI."""
        try:
            # Check if operation manager is initialized
            if not self.operation_manager:
                self.track_status("❌ Operation manager not initialized", "error")
                return
            
            # Create a new event loop for the thread
            import threading
            
            def run_operation():
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Run the async operation
                    result = loop.run_until_complete(
                        self.operation_manager.execute_operation(
                            operation_type=operation_type,
                            config=config,
                            progress_callback=self._create_progress_callback(),
                            log_callback=self._create_log_callback()
                        )
                    )
                    
                    # Handle result
                    self._handle_operation_result(operation_type, result)
                    
                except Exception as e:
                    self.logger.error(f"Operation {operation_type} failed: {e}")
                    self.track_status(f"❌ {operation_type} failed: {str(e)}", "error")
                finally:
                    loop.close()
            
            # Start operation in a separate thread
            thread = threading.Thread(target=run_operation, daemon=True)
            thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start operation {operation_type}: {e}")
            self.track_status(f"❌ Failed to start {operation_type}: {str(e)}", "error")
    
    def _create_progress_callback(self):
        """Create progress callback for operations."""
        def callback(current: int, total: int, message: str):
            # Update progress if operation container is available
            if hasattr(self.operation_manager, 'operation_container') and self.operation_manager.operation_container:
                progress_percent = int((current / total) * 100) if total > 0 else 0
                self.operation_manager.operation_container.update_progress(
                    value=progress_percent,
                    message=message,
                    level='primary'
                )
        return callback
    
    def _create_log_callback(self):
        """Create log callback for operations."""
        def callback(level: str, message: str):
            self.track_status(message, level.lower())
        return callback
    
    def _handle_operation_result(self, operation_type: str, result: Dict[str, Any]) -> None:
        """Handle the result of an operation."""
        try:
            if operation_type == BackboneOperation.VALIDATE.value:
                if result.get('valid', False):
                    self.track_status("✅ Configuration validation completed successfully", "success")
                else:
                    errors = result.get('errors', [])
                    self.track_status(f"❌ Validation failed: {', '.join(errors)}", "error")
            
            elif operation_type == BackboneOperation.LOAD.value:
                if result.get('success', False):
                    info = result.get('info', {})
                    params = info.get('total_parameters', 0)
                    size_mb = info.get('model_size_mb', 0)
                    self.track_status(f"✅ Model loaded: {params:,} parameters ({size_mb:.1f} MB)", "success")
                else:
                    self.track_status(f"❌ Loading failed: {result.get('error', 'Unknown error')}", "error")
            
            elif operation_type == BackboneOperation.BUILD.value:
                if result.get('success', False):
                    stats = result.get('stats', {})
                    params = stats.get('total_parameters', 0)
                    self.track_status(f"✅ Architecture built: {params:,} parameters", "success")
                else:
                    self.track_status(f"❌ Build failed: {result.get('error', 'Unknown error')}", "error")
            
            elif operation_type == BackboneOperation.SUMMARY.value:
                if result.get('success', False):
                    summary = result.get('summary', {})
                    backbone_type = summary.get('backbone_type', 'Unknown')
                    self.track_status(f"✅ Summary generated for {backbone_type} backbone", "success")
                else:
                    self.track_status(f"❌ Summary failed: {result.get('error', 'Unknown error')}", "error")
                    
        except Exception as e:
            self.logger.error(f"Error handling operation result: {e}")
            self.track_status(f"❌ Error processing operation result", "error")
    
    def _handle_save_config(self) -> None:
        """Handle configuration save."""
        try:
            config = self.extract_config_from_ui()
            self.config_handler.update_config(config)
            self.track_status("💾 Configuration saved", "success")
        except Exception as e:
            self.track_status(f"❌ Save failed: {str(e)}", "error")
    
    def _handle_reset_config(self) -> None:
        """Handle configuration reset."""
        try:
            self.config_handler.reset_to_defaults()
            config = self.config_handler.get_config()
            self.update_ui_from_config(config)
            self.track_status("🔄 Configuration reset to defaults", "info")
        except Exception as e:
            self.track_status(f"❌ Reset failed: {str(e)}", "error")
    
    def initialize(self) -> None:
        """Initialize the UI handler for backbone module.
        
        Implements the abstract method required by ModuleUIHandler.
        """
        self.logger.info("🚀 Initializing Backbone UI Handler")
        # Perform any necessary initialization for the backbone UI handler
        self.logger.info("✅ Backbone UI Handler initialized successfully")
    
    def get_selected_backbone(self) -> str:
        """Get selected backbone type."""
        config = self.extract_config_from_ui()
        return config.get('model', {}).get('backbone', 'efficientnet_b4')
    
    def get_detection_layers(self) -> list:
        """Get selected detection layers."""
        config = self.extract_config_from_ui()
        return config.get('model', {}).get('detection_layers', ['banknote'])
    
    def get_layer_mode(self) -> str:
        """Get selected layer mode."""
        config = self.extract_config_from_ui()
        return config.get('model', {}).get('layer_mode', 'single')
    
    def set_backbone(self, backbone: str) -> bool:
        """Set backbone type and update UI."""
        if self.config_handler.set_backbone(backbone):
            self.sync_ui_with_config()
            return True
        return False
    
    def set_detection_layers(self, layers: list) -> bool:
        """Set detection layers and update UI."""
        if self.config_handler.set_detection_layers(layers):
            self.sync_ui_with_config()
            return True
        return False
    
    def set_layer_mode(self, mode: str) -> bool:
        """Set layer mode and update UI."""
        if self.config_handler.set_layer_mode(mode):
            self.sync_ui_with_config()
            return True
        return False