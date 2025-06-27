
"""
Event handlers for dependency management UI.

This module contains functions to set up and manage event handlers for the dependency management UI.
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.config_handlers import ConfigHandler


def setup_event_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup event handlers for UI components.
    
    Args:
        ui_components: Dictionary containing all UI components
    """
    if not ui_components:
        return
    
    # Setup logger bridge if available
    logger_bridge = ui_components.get('logger_bridge')
    
    # Setup operation handlers
    from .operation_handlers import setup_operation_handlers
    
    try:
        # Setup operation handlers
        operation_handlers = setup_operation_handlers(ui_components, {})
        
        # Store reference to handlers if needed
        ui_components['_operation_handlers'] = operation_handlers
        
        if logger_bridge:
            logger_bridge.info("✅ Operation handlers berhasil disetup")
    except Exception as e:
        if logger_bridge:
            logger_bridge.error(f"❌ Gagal setup operation handlers: {str(e)}")
    
    # Setup logger bridge for components that need it
    if logger_bridge:
        if 'log_accordion' in ui_components and ui_components['log_accordion']:
            ui_components['log_accordion'].logger_bridge = logger_bridge
        
        if 'progress_tracker' in ui_components and ui_components['progress_tracker']:
            ui_components['progress_tracker'].logger_bridge = logger_bridge
    
    # Initialize save status
    if 'update_save_status' in ui_components and callable(ui_components['update_save_status']):
        ui_components['update_save_status']()


def setup_all_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], 
                      config_handler: Optional[ConfigHandler] = None) -> Dict[str, Any]:
    """Setup semua handlers untuk dependency management"""
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi di UI components")
    
    handlers = {}
    
    try:
        # Setup config handlers
        from .config_event_handlers import setup_config_handlers
        handlers['config'] = setup_config_handlers(ui_components, config_handler)
        
        # Setup operation handlers
        from .operation_handlers import setup_operation_handlers
        handlers['operations'] = setup_operation_handlers(ui_components, config)
        
        # Setup selection handlers
        from .selection_handlers import setup_selection_handlers
        handlers['selections'] = setup_selection_handlers(ui_components, config)
        
        logger_bridge.info("✅ Semua dependency handlers berhasil disetup")
        return handlers
        
    except Exception as e:
        logger_bridge.error(f"❌ Error setup handlers: {str(e)}")
        raise ValueError(f"Gagal setup handlers: {str(e)}") from e

def setup_config_handlers(ui_components: Dict[str, Any], config_handler: ConfigHandler) -> Dict[str, Any]:
    """Setup config event handlers"""
    from .base_handler import BaseDependencyHandler
    
    class ConfigEventHandler(BaseDependencyHandler):
        def save_config(self, *args):
            try:
                config = config_handler.extract_config(ui_components)
                config_handler.save_config(config)
                self.log_success("💾 Konfigurasi berhasil disimpan")
            except Exception as e:
                self.log_error(f"❌ Error save config: {str(e)}")
        
        def reset_config(self, *args):
            try:
                default_config = config_handler.get_default_config()
                config_handler.update_ui(ui_components, default_config)
                self.log_success("🔄 Konfigurasi direset ke default")
            except Exception as e:
                self.log_error(f"❌ Error reset config: {str(e)}")
    
    handler = ConfigEventHandler(ui_components)
    return {'save': handler.save_config, 'reset': handler.reset_config}