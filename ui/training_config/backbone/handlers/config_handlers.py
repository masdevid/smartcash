"""
File: smartcash/ui/training_config/backbone/handlers/config_handlers.py
Deskripsi: Handler utama untuk konfigurasi backbone model
"""

from typing import Dict, Any
import os
from pathlib import Path
import ipywidgets as widgets

from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger, LogLevel
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.training_config.backbone.handlers.default_config import get_default_backbone_config

# Setup logger dengan level INFO
logger = get_logger(__name__)
logger.set_level(LogLevel.INFO)

def get_default_base_dir():
    """Mendapatkan direktori default berdasarkan environment"""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def get_backbone_config(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi backbone dari config manager.
    
    Args:
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Dictionary konfigurasi backbone
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = config_manager.get_config('model')
        if config and 'model' in config:
            return config
        logger.warning(f"{ICONS.get('warning', '⚠️')} Konfigurasi backbone tidak ditemukan, menggunakan default")
        return get_default_backbone_config()
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mengambil konfigurasi backbone: {str(e)}")
        return get_default_backbone_config()

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi backbone dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = config_manager.get_config('model') or get_default_backbone_config()
        
        logger.info("Mengupdate konfigurasi dari UI components")
        
        # Update backbone settings
        if 'backbone_dropdown' in ui_components:
            config['model']['backbone'] = ui_components['backbone_dropdown'].value
            
        # Update model-specific settings
        if 'model_type_dropdown' in ui_components:
            config['model']['model_type'] = ui_components['model_type_dropdown'].value
            
        # Update feature adapter settings
        if 'use_attention_checkbox' in ui_components:
            config['model']['use_attention'] = ui_components['use_attention_checkbox'].value
            
        if 'use_residual_checkbox' in ui_components:
            config['model']['use_residual'] = ui_components['use_residual_checkbox'].value
            
        if 'use_ciou_checkbox' in ui_components:
            config['model']['use_ciou'] = ui_components['use_ciou_checkbox'].value
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil diupdate dari UI")
        
        return config
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat update konfigurasi backbone dari UI: {str(e)}")
        return get_default_backbone_config()

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """
    Update UI dari konfigurasi backbone.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan digunakan (opsional)
    """
    try:
        # Get config if not provided
        if config is None:
            logger.info("Mengambil konfigurasi backbone dari config manager")
            config = get_backbone_config(ui_components)
            
        # Ensure config has model key
        if 'model' not in config:
            logger.info("Menambahkan key 'model' ke konfigurasi")
            config['model'] = get_default_backbone_config()['model']
            
        logger.info(f"Memperbarui UI dari konfigurasi backbone")
            
        # Update UI components for backbone
        if 'backbone_dropdown' in ui_components and 'backbone' in config['model']:
            ui_components['backbone_dropdown'].value = config['model']['backbone']
            
        # Update UI for model type
        if 'model_type_dropdown' in ui_components and 'model_type' in config['model']:
            ui_components['model_type_dropdown'].value = config['model']['model_type']
            
        # Update UI for feature adapters
        if 'use_attention_checkbox' in ui_components and 'use_attention' in config['model']:
            ui_components['use_attention_checkbox'].value = config['model']['use_attention']
            
        if 'use_residual_checkbox' in ui_components and 'use_residual' in config['model']:
            ui_components['use_residual_checkbox'].value = config['model']['use_residual']
            
        if 'use_ciou_checkbox' in ui_components and 'use_ciou' in config['model']:
            ui_components['use_ciou_checkbox'].value = config['model']['use_ciou']
            
        # Force update UI by triggering change events for boolean widgets
        _trigger_ui_update_events(ui_components)
            
        logger.info(f"{ICONS.get('success', '✅')} UI berhasil diupdate dari konfigurasi backbone")
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mengupdate UI dari konfigurasi: {str(e)}")
        if 'info_panel' in ui_components:
            with ui_components['info_panel']:
                from IPython.display import display, HTML
                display(HTML(f"<div style='color: red;'>Error: {str(e)}</div>"))

def _trigger_ui_update_events(ui_components: Dict[str, Any]) -> None:
    """
    Memicu event perubahan pada checkbox untuk mengupdate UI terkait.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    for widget_name, widget in ui_components.items():
        if isinstance(widget, widgets.Checkbox):
            # Trigger a change event to update dependent widgets
            old_value = widget.value
            # Toggle boolean value to trigger change event
            widget.value = not old_value
            widget.value = old_value

def update_backbone_info(ui_components: Dict[str, Any], message: str = None) -> None:
    """
    Update info panel dengan informasi backbone yang dipilih.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan tambahan yang akan ditampilkan (opsional)
    """
    from smartcash.ui.training_config.backbone.handlers.info_panel import update_backbone_info_panel
    update_backbone_info_panel(ui_components, message)