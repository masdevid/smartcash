"""
File: smartcash/ui/training_config/hyperparameters/handlers/form_handlers.py
Deskripsi: Handler untuk form dan input di UI konfigurasi hyperparameters
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.training_config.hyperparameters.handlers.config_handlers import (
    update_config_from_ui,
    update_hyperparameters_info
)

logger = get_logger(__name__)

def setup_hyperparameters_form_handlers(ui_components: Dict[str, Any], env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Setup handler untuk form dan input di UI konfigurasi hyperparameters.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        config: Konfigurasi hyperparameters (opsional)
        
    Returns:
        Dictionary komponen UI yang telah diupdate dengan handler
    """
    # Handler untuk optimizer type dropdown
    if 'optimizer_type_dropdown' in ui_components:
        ui_components['optimizer_type_dropdown'].observe(
            lambda change: optimizer_type_changed(change, ui_components),
            names='value'
        )
    
    # Handler untuk scheduler enabled checkbox
    if 'scheduler_enabled_checkbox' in ui_components:
        ui_components['scheduler_enabled_checkbox'].observe(
            lambda change: scheduler_enabled_changed(change, ui_components),
            names='value'
        )
    
    # Handler untuk scheduler type dropdown
    if 'scheduler_type_dropdown' in ui_components:
        ui_components['scheduler_type_dropdown'].observe(
            lambda change: scheduler_type_changed(change, ui_components),
            names='value'
        )
    
    # Handler untuk loss type dropdown
    if 'loss_type_dropdown' in ui_components:
        ui_components['loss_type_dropdown'].observe(
            lambda change: loss_type_changed(change, ui_components),
            names='value'
        )
    
    # Handler untuk augmentation enabled checkbox
    if 'augmentation_enabled_checkbox' in ui_components:
        ui_components['augmentation_enabled_checkbox'].observe(
            lambda change: augmentation_enabled_changed(change, ui_components),
            names='value'
        )
    
    return ui_components

def optimizer_type_changed(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan tipe optimizer.
    
    Args:
        change: Objek perubahan dari widget
        ui_components: Dictionary komponen UI
    """
    try:
        optimizer_type = change['new']
        
        # Tampilkan/sembunyikan parameter sesuai dengan tipe optimizer
        if optimizer_type == 'SGD':
            # Tampilkan parameter SGD
            toggle_widget_visibility(ui_components, 'momentum_slider', True)
            toggle_widget_visibility(ui_components, 'beta1_slider', False)
            toggle_widget_visibility(ui_components, 'beta2_slider', False)
            toggle_widget_visibility(ui_components, 'eps_slider', False)
        elif optimizer_type == 'Adam':
            # Tampilkan parameter Adam
            toggle_widget_visibility(ui_components, 'momentum_slider', False)
            toggle_widget_visibility(ui_components, 'beta1_slider', True)
            toggle_widget_visibility(ui_components, 'beta2_slider', True)
            toggle_widget_visibility(ui_components, 'eps_slider', True)
        elif optimizer_type == 'AdamW':
            # Tampilkan parameter AdamW
            toggle_widget_visibility(ui_components, 'momentum_slider', False)
            toggle_widget_visibility(ui_components, 'beta1_slider', True)
            toggle_widget_visibility(ui_components, 'beta2_slider', True)
            toggle_widget_visibility(ui_components, 'eps_slider', True)
        
        # Update konfigurasi dan info panel
        update_config_from_ui(ui_components)
        update_hyperparameters_info(ui_components)
        
    except Exception as e:
        logger.error(f"Error saat mengubah tipe optimizer: {str(e)}")

def scheduler_enabled_changed(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan status scheduler.
    
    Args:
        change: Objek perubahan dari widget
        ui_components: Dictionary komponen UI
    """
    try:
        scheduler_enabled = change['new']
        
        # Tampilkan/sembunyikan parameter scheduler
        scheduler_widgets = [
            'scheduler_type_dropdown',
            'warmup_epochs_slider',
            'min_lr_slider',
            'patience_slider',
            'factor_slider',
            'threshold_slider'
        ]
        
        for widget_name in scheduler_widgets:
            toggle_widget_visibility(ui_components, widget_name, scheduler_enabled)
        
        # Update konfigurasi dan info panel
        update_config_from_ui(ui_components)
        update_hyperparameters_info(ui_components)
        
    except Exception as e:
        logger.error(f"Error saat mengubah status scheduler: {str(e)}")

def scheduler_type_changed(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan tipe scheduler.
    
    Args:
        change: Objek perubahan dari widget
        ui_components: Dictionary komponen UI
    """
    try:
        scheduler_type = change['new']
        
        # Tampilkan/sembunyikan parameter sesuai dengan tipe scheduler
        if scheduler_type == 'CosineAnnealingLR':
            # Tampilkan parameter CosineAnnealingLR
            toggle_widget_visibility(ui_components, 'warmup_epochs_slider', True)
            toggle_widget_visibility(ui_components, 'min_lr_slider', True)
            toggle_widget_visibility(ui_components, 'patience_slider', False)
            toggle_widget_visibility(ui_components, 'factor_slider', False)
            toggle_widget_visibility(ui_components, 'threshold_slider', False)
        elif scheduler_type == 'ReduceLROnPlateau':
            # Tampilkan parameter ReduceLROnPlateau
            toggle_widget_visibility(ui_components, 'warmup_epochs_slider', False)
            toggle_widget_visibility(ui_components, 'min_lr_slider', True)
            toggle_widget_visibility(ui_components, 'patience_slider', True)
            toggle_widget_visibility(ui_components, 'factor_slider', True)
            toggle_widget_visibility(ui_components, 'threshold_slider', True)
        elif scheduler_type == 'StepLR':
            # Tampilkan parameter StepLR
            toggle_widget_visibility(ui_components, 'warmup_epochs_slider', True)
            toggle_widget_visibility(ui_components, 'min_lr_slider', False)
            toggle_widget_visibility(ui_components, 'patience_slider', False)
            toggle_widget_visibility(ui_components, 'factor_slider', True)
            toggle_widget_visibility(ui_components, 'threshold_slider', False)
        
        # Update konfigurasi dan info panel
        update_config_from_ui(ui_components)
        update_hyperparameters_info(ui_components)
        
    except Exception as e:
        logger.error(f"Error saat mengubah tipe scheduler: {str(e)}")

def loss_type_changed(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan tipe loss function.
    
    Args:
        change: Objek perubahan dari widget
        ui_components: Dictionary komponen UI
    """
    try:
        loss_type = change['new']
        
        # Tampilkan/sembunyikan parameter sesuai dengan tipe loss
        if loss_type == 'CrossEntropyLoss':
            # Tampilkan parameter CrossEntropyLoss
            toggle_widget_visibility(ui_components, 'label_smoothing_slider', True)
            toggle_widget_visibility(ui_components, 'alpha_slider', False)
            toggle_widget_visibility(ui_components, 'gamma_slider', False)
        elif loss_type == 'FocalLoss':
            # Tampilkan parameter FocalLoss
            toggle_widget_visibility(ui_components, 'label_smoothing_slider', True)
            toggle_widget_visibility(ui_components, 'alpha_slider', True)
            toggle_widget_visibility(ui_components, 'gamma_slider', True)
        
        # Update konfigurasi dan info panel
        update_config_from_ui(ui_components)
        update_hyperparameters_info(ui_components)
        
    except Exception as e:
        logger.error(f"Error saat mengubah tipe loss function: {str(e)}")

def augmentation_enabled_changed(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan status augmentation.
    
    Args:
        change: Objek perubahan dari widget
        ui_components: Dictionary komponen UI
    """
    try:
        augmentation_enabled = change['new']
        
        # Tampilkan/sembunyikan parameter augmentation
        augmentation_widgets = [
            'mosaic_checkbox',
            'mixup_checkbox',
            'hsv_h_slider',
            'hsv_s_slider',
            'hsv_v_slider',
            'degrees_slider',
            'translate_slider',
            'scale_slider',
            'shear_slider',
            'perspective_slider',
            'flipud_slider',
            'fliplr_slider',
            'mosaic_prob_slider',
            'mixup_prob_slider'
        ]
        
        for widget_name in augmentation_widgets:
            toggle_widget_visibility(ui_components, widget_name, augmentation_enabled)
        
        # Update konfigurasi dan info panel
        update_config_from_ui(ui_components)
        update_hyperparameters_info(ui_components)
        
    except Exception as e:
        logger.error(f"Error saat mengubah status augmentation: {str(e)}")

def toggle_widget_visibility(ui_components: Dict[str, Any], widget_name: str, visible: bool) -> None:
    """
    Toggle visibilitas widget.
    
    Args:
        ui_components: Dictionary komponen UI
        widget_name: Nama widget yang akan diubah visibilitasnya
        visible: Status visibilitas (True/False)
    """
    if widget_name in ui_components:
        if visible:
            ui_components[widget_name].layout.display = 'flex'
        else:
            ui_components[widget_name].layout.display = 'none' 