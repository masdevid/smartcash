"""
File: smartcash/ui/training_config/backbone/handlers/form_handlers.py
Deskripsi: Handler untuk form pada UI pemilihan backbone model SmartCash
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def on_backbone_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan pada dropdown backbone.
    
    Args:
        change: Dictionary berisi informasi perubahan
        ui_components: Dictionary berisi komponen UI
    """
    backbone = change.get('new')
    model_type_dropdown = ui_components.get('model_type_dropdown')
    
    if not backbone or not model_type_dropdown:
        return
    
    # Jika backbone adalah cspdarknet_s, set model type ke yolov5s
    if backbone == 'cspdarknet_s':
        model_type_dropdown.value = 'yolov5s'
        
        # Disable checkbox optimasi untuk CSPDarknet
        for checkbox_name in ['use_attention_checkbox', 'use_residual_checkbox', 'use_ciou_checkbox']:
            checkbox = ui_components.get(checkbox_name)
            if checkbox:
                checkbox.value = False
                checkbox.disabled = True
    
    # Jika backbone adalah efficientnet_b4, enable checkbox optimasi
    elif backbone == 'efficientnet_b4':
        model_type_dropdown.value = 'efficient_basic'
        
        # Enable checkbox optimasi untuk EfficientNet
        for checkbox_name in ['use_attention_checkbox', 'use_residual_checkbox', 'use_ciou_checkbox']:
            checkbox = ui_components.get(checkbox_name)
            if checkbox:
                checkbox.disabled = False
    
    logger.debug(f"{ICONS.get('info', 'ℹ️')} Backbone diubah ke {backbone}")

def on_model_type_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan pada dropdown model type.
    
    Args:
        change: Dictionary berisi informasi perubahan
        ui_components: Dictionary berisi komponen UI
    """
    model_type = change.get('new')
    backbone_dropdown = ui_components.get('backbone_dropdown')
    
    if not model_type or not backbone_dropdown:
        return
    
    # Jika model type adalah yolov5s, set backbone ke cspdarknet_s
    if model_type == 'yolov5s':
        backbone_dropdown.value = 'cspdarknet_s'
        
        # Disable checkbox optimasi untuk YOLOv5s
        for checkbox_name in ['use_attention_checkbox', 'use_residual_checkbox', 'use_ciou_checkbox']:
            checkbox = ui_components.get(checkbox_name)
            if checkbox:
                checkbox.value = False
                checkbox.disabled = True
    
    # Jika model type adalah efficient_basic, set backbone ke efficientnet_b4
    elif model_type == 'efficient_basic':
        backbone_dropdown.value = 'efficientnet_b4'
        
        # Enable checkbox optimasi untuk EfficientNet
        for checkbox_name in ['use_attention_checkbox', 'use_residual_checkbox', 'use_ciou_checkbox']:
            checkbox = ui_components.get(checkbox_name)
            if checkbox:
                checkbox.disabled = False
    
    logger.debug(f"{ICONS.get('info', 'ℹ️')} Model type diubah ke {model_type}")

def on_attention_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan pada checkbox attention.
    
    Args:
        change: Dictionary berisi informasi perubahan
        ui_components: Dictionary berisi komponen UI
    """
    use_attention = change.get('new')
    logger.debug(f"{ICONS.get('info', 'ℹ️')} Penggunaan FeatureAdapter (Attention) diubah ke {use_attention}")

def on_residual_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan pada checkbox residual.
    
    Args:
        change: Dictionary berisi informasi perubahan
        ui_components: Dictionary berisi komponen UI
    """
    use_residual = change.get('new')
    logger.debug(f"{ICONS.get('info', 'ℹ️')} Penggunaan ResidualAdapter diubah ke {use_residual}")

def on_ciou_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan pada checkbox ciou.
    
    Args:
        change: Dictionary berisi informasi perubahan
        ui_components: Dictionary berisi komponen UI
    """
    use_ciou = change.get('new')
    logger.debug(f"{ICONS.get('info', 'ℹ️')} Penggunaan CIoU Loss diubah ke {use_ciou}")
