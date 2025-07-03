"""
File: smartcash/ui/setup/dependency/handlers/event_handlers.py
Description: Event handlers for dependency management UI with centralized error handling.

This module contains functions to set up and manage event handlers for the dependency management UI.
"""

from typing import Dict, Any, Optional
from .base_dependency_handler import BaseDependencyHandler


def setup_event_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup event handlers for UI components.
    
    Args:
        ui_components: Dictionary containing all UI components
    """
    if not ui_components:
        return
    
    # Create handler for centralized error handling
    handler = BaseDependencyHandler(ui_components=ui_components)
    
    # Setup operation handlers
    from .operation_handlers import setup_operation_handlers
    
    try:
        # Setup operation handlers
        operation_handlers = setup_operation_handlers(ui_components, {})
        
        # Store reference to handlers if needed
        ui_components['_operation_handlers'] = operation_handlers
        
        handler.logger.info("✅ Operation handlers berhasil disetup")
    except Exception as e:
        handler.handle_error(e, "Gagal setup operation handlers")
    
    # Initialize save status
    if 'update_save_status' in ui_components and callable(ui_components['update_save_status']):
        try:
            ui_components['update_save_status']()
        except Exception as e:
            handler.handle_error(e, "Gagal menginisialisasi status penyimpanan")


def setup_all_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], 
                      config_handler: Optional['BaseConfigHandler'] = None) -> Dict[str, Any]:
    """Setup semua handlers untuk dependency management.
    
    Args:
        ui_components: Dictionary containing UI components
        config: Configuration dictionary
        config_handler: Optional config handler instance
        
    Returns:
        Dictionary containing all handlers
        
    Raises:
        ValueError: If setup fails
    """
    # Create handler for centralized error handling
    handler = BaseDependencyHandler(ui_components=ui_components)
    
    handlers = {}
    
    try:
        # Setup config handlers
        handlers['config'] = setup_config_handlers(ui_components, config_handler)
        
        # Setup operation handlers
        from .operation_handlers import setup_operation_handlers
        handlers['operations'] = setup_operation_handlers(ui_components, config)
        
        # Setup selection handlers
        from .selection_handlers import setup_selection_handlers
        handlers['selections'] = setup_selection_handlers(ui_components, config)
        
        handler.logger.info("✅ Semua dependency handlers berhasil disetup")
        return handlers
        
    except Exception as e:
        result = handler.handle_error(e, "Error setup handlers")
        raise ValueError(f"Gagal setup handlers: {str(e)}") from e

def setup_config_handlers(ui_components: Dict[str, Any], config_handler) -> Dict[str, Any]:
    """Setup config event handlers with centralized error handling.
    
    Args:
        ui_components: Dictionary containing UI components
        config_handler: Configuration handler instance
        
    Returns:
        Dictionary containing config event handlers
    """
    class ConfigEventHandler(BaseDependencyHandler):
        def save_config(self, *args):
            try:
                config = config_handler.extract_config(ui_components)
                config_handler.save_config(config)
                self.logger.info("💾 Konfigurasi berhasil disimpan")
            except Exception as e:
                self.handle_error(e, "Error saving configuration")
        
        def reset_config(self, *args):
            try:
                default_config = config_handler.get_default_config()
                config_handler.update_ui(ui_components, default_config)
                self.logger.info("🔄 Konfigurasi direset ke default")
            except Exception as e:
                self.handle_error(e, "Error resetting configuration")
    
    handler = ConfigEventHandler(ui_components=ui_components)
    return {'save': handler.save_config, 'reset': handler.reset_config}