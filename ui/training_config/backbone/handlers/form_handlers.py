"""
File: smartcash/ui/training_config/backbone/handlers/form_handlers.py
Deskripsi: Handler untuk form pada UI pemilihan backbone model SmartCash
"""

from typing import Dict, Any
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
    
    if not backbone:
        return
    
    # Update UI berdasarkan backbone yang dipilih
    _update_ui_based_on_backbone(backbone, ui_components)
    
    # Log perubahan
    logger.debug(f"{ICONS.get('info', 'ℹ️')} Backbone diubah ke {backbone}")
    
    # Update info panel
    _update_info_after_change(ui_components)

def on_model_type_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan pada dropdown model type.
    
    Args:
        change: Dictionary berisi informasi perubahan
        ui_components: Dictionary berisi komponen UI
    """
    model_type = change.get('new')
    
    if not model_type:
        return
    
    # Update UI berdasarkan model type yang dipilih
    _update_ui_based_on_model_type(model_type, ui_components)
    
    # Log perubahan
    logger.debug(f"{ICONS.get('info', 'ℹ️')} Model type diubah ke {model_type}")
    
    # Update info panel
    _update_info_after_change(ui_components)

def on_attention_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan pada checkbox attention.
    
    Args:
        change: Dictionary berisi informasi perubahan
        ui_components: Dictionary berisi komponen UI
    """
    use_attention = change.get('new')
    logger.debug(f"{ICONS.get('info', 'ℹ️')} Penggunaan FeatureAdapter (Attention) diubah ke {use_attention}")
    
    # Update info panel
    _update_info_after_change(ui_components)

def on_residual_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan pada checkbox residual.
    
    Args:
        change: Dictionary berisi informasi perubahan
        ui_components: Dictionary berisi komponen UI
    """
    use_residual = change.get('new')
    logger.debug(f"{ICONS.get('info', 'ℹ️')} Penggunaan ResidualAdapter diubah ke {use_residual}")
    
    # Update info panel
    _update_info_after_change(ui_components)

def on_ciou_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan pada checkbox ciou.
    
    Args:
        change: Dictionary berisi informasi perubahan
        ui_components: Dictionary berisi komponen UI
    """
    use_ciou = change.get('new')
    logger.debug(f"{ICONS.get('info', 'ℹ️')} Penggunaan CIoU Loss diubah ke {use_ciou}")
    
    # Update info panel
    _update_info_after_change(ui_components)

def _update_ui_based_on_backbone(backbone: str, ui_components: Dict[str, Any]) -> None:
    """
    Update UI berdasarkan backbone yang dipilih.
    
    Args:
        backbone: Nilai backbone yang dipilih
        ui_components: Dictionary berisi komponen UI
    """
    model_type_dropdown = ui_components.get('model_type_dropdown')
    
    if not model_type_dropdown:
        return
    
    # Jika backbone adalah cspdarknet_s, set model type ke yolov5s
    if backbone == 'cspdarknet_s':
        # Hindari trigger event change jika nilai sudah sama
        if model_type_dropdown.value != 'yolov5s':
            model_type_dropdown.value = 'yolov5s'
        
        # Disable checkbox optimasi untuk CSPDarknet
        _update_optimization_checkboxes(ui_components, False, False, False, True)
    
    # Jika backbone adalah efficientnet_b4, enable checkbox optimasi
    elif backbone == 'efficientnet_b4':
        # Hindari trigger event change jika nilai sudah sama
        if model_type_dropdown.value != 'efficient_basic':
            model_type_dropdown.value = 'efficient_basic'
        
        # Enable checkbox optimasi untuk EfficientNet
        _update_optimization_checkboxes(ui_components, True, True, False, False)

def _update_ui_based_on_model_type(model_type: str, ui_components: Dict[str, Any]) -> None:
    """
    Update UI berdasarkan model type yang dipilih.
    
    Args:
        model_type: Nilai model type yang dipilih
        ui_components: Dictionary berisi komponen UI
    """
    backbone_dropdown = ui_components.get('backbone_dropdown')
    
    if not backbone_dropdown:
        return
    
    # Jika model type adalah yolov5s, set backbone ke cspdarknet_s
    if model_type == 'yolov5s':
        # Hindari trigger event change jika nilai sudah sama
        if backbone_dropdown.value != 'cspdarknet_s':
            backbone_dropdown.value = 'cspdarknet_s'
        
        # Disable checkbox optimasi untuk YOLOv5s
        _update_optimization_checkboxes(ui_components, False, False, False, True)
    
    # Jika model type adalah efficient_basic, set backbone ke efficientnet_b4
    elif model_type == 'efficient_basic':
        # Hindari trigger event change jika nilai sudah sama
        if backbone_dropdown.value != 'efficientnet_b4':
            backbone_dropdown.value = 'efficientnet_b4'
        
        # Enable checkbox optimasi untuk EfficientNet
        _update_optimization_checkboxes(ui_components, True, True, False, False)

def _update_optimization_checkboxes(ui_components: Dict[str, Any], 
                                  attention_value: bool, 
                                  residual_value: bool, 
                                  ciou_value: bool,
                                  disable: bool) -> None:
    """
    Update checkbox optimasi dengan nilai dan status disabled.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        attention_value: Nilai untuk checkbox attention
        residual_value: Nilai untuk checkbox residual
        ciou_value: Nilai untuk checkbox ciou
        disable: Apakah perlu disable checkbox
    """
    # Update checkbox attention
    attention_checkbox = ui_components.get('use_attention_checkbox')
    if attention_checkbox:
        attention_checkbox.disabled = disable
        # Set nilai jika checkbox di-enable
        if not disable:
            attention_checkbox.value = attention_value
    
    # Update checkbox residual
    residual_checkbox = ui_components.get('use_residual_checkbox')
    if residual_checkbox:
        residual_checkbox.disabled = disable
        # Set nilai jika checkbox di-enable
        if not disable:
            residual_checkbox.value = residual_value
    
    # Update checkbox ciou
    ciou_checkbox = ui_components.get('use_ciou_checkbox')
    if ciou_checkbox:
        ciou_checkbox.disabled = disable
        # Set nilai jika checkbox di-enable
        if not disable:
            ciou_checkbox.value = ciou_value

def _update_info_after_change(ui_components: Dict[str, Any]) -> None:
    """
    Update info panel setelah perubahan pada form.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Import hanya jika diperlukan untuk menghindari circular import
    if 'update_backbone_info' in ui_components and callable(ui_components['update_backbone_info']):
        ui_components['update_backbone_info'](ui_components)
    else:
        # Fallback jika fungsi tidak tersedia
        from smartcash.ui.training_config.backbone.handlers.info_panel import update_backbone_info_panel
        update_backbone_info_panel(ui_components)