"""
File: smartcash/ui/model/backbone/handlers/model_handler.py
Deskripsi: Handler untuk model backbone operations dengan API integration
"""

from typing import Dict, Any, Optional, Callable
import asyncio
from smartcash.ui.utils import with_error_handling, ErrorHandler
from smartcash.ui.model.backbone.utils.ui_utils import (
    extract_model_config, update_model_ui, reset_model_ui, 
    validate_model_config, get_default_model_config
)
from smartcash.ui.model.backbone.components.config_summary import update_config_summary

class BackboneModelHandler:
    """Handler untuk backbone model configuration dan operations"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger_bridge = ui_components.get('logger_bridge')
        self.error_handler = ErrorHandler()
        self.model_api = None
        self.shared_config_manager = None
        self.config_handler = None
        self.api_handler = None
        
        if not self.logger_bridge:
            raise ValueError("Logger bridge required in UI components")
    
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup event handlers untuk UI components"""
        handlers = {}
        
        # Button handlers
        if 'build_btn' in self.ui_components:
            self.ui_components['build_btn'].on_click(
                lambda b: self._handle_build_model()
            )
            handlers['build_btn'] = self.ui_components['build_btn']
        
        if 'validate_btn' in self.ui_components:
            self.ui_components['validate_btn'].on_click(
                lambda b: self._handle_validate_config()
            )
            handlers['validate_btn'] = self.ui_components['validate_btn']
        
        if 'info_btn' in self.ui_components:
            self.ui_components['info_btn'].on_click(
                lambda b: self._handle_model_info()
            )
            handlers['info_btn'] = self.ui_components['info_btn']
        
        # Save/Reset handlers
        if 'save_button' in self.ui_components:
            self.ui_components['save_button'].on_click(
                lambda b: self._handle_save_config()
            )
            handlers['save_button'] = self.ui_components['save_button']
        
        if 'reset_button' in self.ui_components:
            self.ui_components['reset_button'].on_click(
                lambda b: self._handle_reset_config()
            )
            handlers['reset_button'] = self.ui_components['reset_button']
        
        # Form change handlers untuk auto-update summary
        form_widgets = ['backbone_dropdown', 'detection_layers_select', 
                       'layer_mode_dropdown', 'feature_optimization_checkbox',
                       'mixed_precision_checkbox']
        
        for widget_name in form_widgets:
            if widget_name in self.ui_components:
                widget = self.ui_components[widget_name]
                widget.observe(lambda change: self._update_summary(), names='value')
        
        # Initialize shared config manager
        self._setup_shared_config()
        
        return handlers
    
    def _setup_shared_config(self) -> None:
        """Setup shared configuration manager"""
        try:
            from smartcash.ui.config_cell.managers.shared_config_manager import get_shared_config_manager
            self.shared_config_manager = get_shared_config_manager('model')
            
            # Subscribe to config updates
            self.shared_config_manager.subscribe('backbone', self._on_config_update)
            
            # Load existing config if available
            existing_config = self.shared_config_manager.get_config('backbone')
            if existing_config:
                update_model_ui(self.ui_components, existing_config)
                self.logger.info("📡 Loaded existing backbone configuration")
        except Exception as e:
            self.logger.warning(f"⚠️ Shared config not available: {e}")
    
    def _on_config_update(self, config: Dict[str, Any]) -> None:
        """Handle config updates from shared manager"""
        try:
            update_model_ui(self.ui_components, config)
            self.logger_bridge.info("🔄 Updated from shared configuration")
        except Exception as e:
            self.logger_bridge.error(f"❌ Failed to apply shared config: {e}")
    
    @with_error_handling
    def _handle_build_model(self) -> None:
        """Handle build model operation"""
        self.logger.info("🏗️ Starting model build...")
        
        # Extract configuration
        config = extract_model_config(self.ui_components)
        
        # Validate configuration
        is_valid, message = validate_model_config(config)
        if not is_valid:
            self.logger.error(f"❌ Configuration invalid: {message}")
            self._update_status(f"❌ {message}", "error")
            return
        
        # Disable buttons during operation
        self._set_buttons_state(True)
        
        # Get progress tracker
        progress_tracker = self._get_progress_tracker()
        
        # Create progress callback
        progress_callback = self._create_progress_callback()
        
        try:
            # Use API handler for model building
            if not self.api_handler:
                from smartcash.ui.model.backbone.handlers.api_handler import BackboneAPIHandler
                self.api_handler = BackboneAPIHandler(self.logger_bridge)
            
            # Build model dengan API handler
            result = self.api_handler.build_model_async(config, progress_callback)
            
            if result['success']:
                self.logger_bridge.success(f"✅ Model built successfully!")
                if 'total_params' in result:
                    self.logger_bridge.info(f"📊 Total parameters: {result['total_params']:,}")
                if 'build_time' in result:
                    self.logger_bridge.info(f"⏱️ Build time: {result['build_time']:.2f}s")
                self._update_status("✅ Model built successfully", "success")
                
                # Save config to both handlers
                if self.config_handler:
                    self.config_handler.save_config(config)
                if self.shared_config_manager:
                    self.shared_config_manager.update_config('backbone', config)
            else:
                raise Exception(result.get('error', 'Unknown error'))
                
        except Exception as e:
            self.logger_bridge.error(f"❌ Model build failed: {str(e)}")
            self._update_status(f"❌ Build failed: {str(e)}", "error")
            raise
        finally:
            self._set_buttons_state(False)
            if progress_tracker:
                progress_tracker.hide()
    
    @with_error_handling
    def _handle_validate_config(self) -> None:
        """Handle configuration validation"""
        self.logger_bridge.info("📊 Validating configuration...")
        
        # Extract configuration
        config = extract_model_config(self.ui_components)
        
        # Validate
        is_valid, message = validate_model_config(config)
        
        if is_valid:
            self.logger_bridge.success("✅ Configuration is valid")
            self._update_status("✅ Configuration valid", "success")
            
            # Log configuration details
            model_config = config['model']
            self.logger_bridge.info(f"📋 Backbone: {model_config['backbone']}")
            self.logger_bridge.info(f"📋 Detection Layers: {', '.join(model_config['detection_layers'])}")
            self.logger_bridge.info(f"📋 Layer Mode: {model_config['layer_mode']}")
            self.logger_bridge.info(f"📋 Feature Optimization: {'Enabled' if model_config['feature_optimization']['enabled'] else 'Disabled'}")
        else:
            self.logger_bridge.error(f"❌ {message}")
            self._update_status(f"❌ {message}", "error")
    
    @with_error_handling
    def _handle_model_info(self) -> None:
        """Handle model information display"""
        self.logger_bridge.info("🔍 Getting model information...")
        
        try:
            # Use API handler for model info
            if not self.api_handler:
                self.logger_bridge.warning("⚠️ Model not built yet")
                self._update_status("⚠️ Build model first", "warning")
                return
            
            # Get model info
            info = self.api_handler.get_model_info()
            
            if info:
                self.logger_bridge.info("📊 Model Information:")
                self.logger_bridge.info(f"   • Architecture: {info.get('architecture', 'N/A')}")
                self.logger_bridge.info(f"   • Total Parameters: {info.get('total_parameters', 0):,}")
                self.logger_bridge.info(f"   • Trainable Parameters: {info.get('trainable_parameters', 0):,}")
                self.logger_bridge.info(f"   • Input Size: {info.get('input_size', 'N/A')}")
                self.logger_bridge.info(f"   • Output Classes: {info.get('num_classes', 'N/A')}")
                self.logger_bridge.info(f"   • Detection Layers: {', '.join(info.get('detection_layers', []))}")
                self._update_status("✅ Model info displayed", "success")
            else:
                self.logger_bridge.warning("⚠️ No model information available")
                self._update_status("⚠️ No model info available", "warning")
                
        except Exception as e:
            self.logger_bridge.error(f"❌ Failed to get model info: {str(e)}")
            self._update_status(f"❌ Error: {str(e)}", "error")
    
    @with_error_handling
    def _handle_save_config(self) -> None:
        """Handle configuration save"""
        self.logger_bridge.info("💾 Saving configuration...")
        
        try:
            # Extract configuration
            config = extract_model_config(self.ui_components)
            
            # Validate first
            is_valid, message = validate_model_config(config)
            if not is_valid:
                self.logger_bridge.error(f"❌ Cannot save invalid config: {message}")
                self._update_status(f"❌ {message}", "error")
                return
            
            # Initialize config handler if not exists
            if not hasattr(self, 'config_handler'):
                from smartcash.ui.model.backbone.handlers.config_handler import BackboneConfigHandler
                self.config_handler = BackboneConfigHandler(self.logger_bridge)
            
            # Save using config handler
            success = self.config_handler.save_config(config)
            
            if success:
                # Also update shared config manager if available
                if self.shared_config_manager:
                    self.shared_config_manager.update_config('backbone', config, persist=True)
                
                self._update_status("✅ Configuration saved", "success")
            else:
                self._update_status("❌ Failed to save configuration", "error")
                
        except Exception as e:
            self.logger_bridge.error(f"❌ Failed to save config: {str(e)}")
            self._update_status(f"❌ Save failed: {str(e)}", "error")
    
    def _handle_reset_config(self) -> None:
        """Handle configuration reset"""
        self.logger_bridge.info("🔄 Resetting configuration...")
        
        try:
            reset_model_ui(self.ui_components)
            # Update the config summary if available
            if 'config_summary' in self.ui_components:
                from smartcash.ui.model.backbone.components.config_summary import update_config_summary
                config = self.extract_model_config(self.ui_components)
                update_config_summary(self.ui_components['config_summary'], config)
            self.logger_bridge.success("✅ Configuration reset to defaults")
        except Exception as e:
            self.logger_bridge.error(f"❌ Failed to reset: {str(e)}")