"""
File: smartcash/ui/dataset/split/handlers/button_handlers.py
Deskripsi: Handler untuk button events split dataset
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.split.handlers.config_handlers import get_split_config, update_config_from_ui, update_ui_from_config
from smartcash.common.config import get_config_manager

logger = get_logger(__name__)

def get_default_base_dir():
    import os
    from pathlib import Path
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def handle_split_button_click(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle click event untuk split button.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Get config
        config = get_split_config(ui_components)
        
        # Update config from UI
        updated_config = update_config_from_ui(ui_components)
        
        # Update UI from config
        update_ui_from_config(ui_components, updated_config)
        
        logger.info("✅ Split button berhasil dihandle")
        
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error saat handle split button: {str(e)}")
        return ui_components

def handle_reset_button_click(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle click event untuk reset button.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Get default config
        from smartcash.ui.dataset.split.handlers.config_handlers import get_default_split_config
        default_config = get_default_split_config()
        
        # Update UI from default config
        update_ui_from_config(ui_components, default_config)
        
        logger.info("✅ Reset button berhasil dihandle")
        
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error saat handle reset button: {str(e)}")
        return ui_components
