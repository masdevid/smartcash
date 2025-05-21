"""
File: smartcash/ui/training_config/hyperparameters/handlers/form_handlers.py
Deskripsi: Handler untuk form dan input di UI konfigurasi hyperparameters
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.training_config.hyperparameters.handlers.config_reader import update_config_from_ui
from smartcash.ui.training_config.hyperparameters.handlers.status_handlers import show_error_status

logger = get_logger(__name__)

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
    # Handler untuk optimizer dropdown
    if 'optimizer_dropdown' in ui_components:
        ui_components['optimizer_dropdown'].observe(
            lambda change: on_optimizer_change(change, ui_components),
            names='value'
        )
    
    # Handler untuk scheduler checkbox
    if 'scheduler_checkbox' in ui_components:
        ui_components['scheduler_checkbox'].observe(
            lambda change: on_scheduler_change(change, ui_components),
            names='value'
        )
    
    # Handler untuk scheduler dropdown
    if 'scheduler_dropdown' in ui_components:
        ui_components['scheduler_dropdown'].observe(
            lambda change: on_scheduler_type_change(change, ui_components),
            names='value'
        )
    
    # Handler untuk early stopping checkbox
    if 'early_stopping_checkbox' in ui_components:
        ui_components['early_stopping_checkbox'].observe(
            lambda change: on_early_stopping_change(change, ui_components),
            names='value'
        )
    
    # Handler untuk augmentation checkbox
    if 'augment_checkbox' in ui_components:
        ui_components['augment_checkbox'].observe(
            lambda change: on_augmentation_change(change, ui_components),
            names='value'
        )
    
    # Simpan referensi ke function handler untuk testing
    ui_components['on_optimizer_change'] = lambda change: on_optimizer_change(change, ui_components)
    ui_components['on_scheduler_change'] = lambda change: on_scheduler_change(change, ui_components)
    ui_components['on_scheduler_type_change'] = lambda change: on_scheduler_type_change(change, ui_components)
    ui_components['on_early_stopping_change'] = lambda change: on_early_stopping_change(change, ui_components)
    ui_components['on_augmentation_change'] = lambda change: on_augmentation_change(change, ui_components)
    
    return ui_components

def on_optimizer_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan optimizer.
    
    Args:
        change: Objek perubahan dari widget
        ui_components: Dictionary komponen UI
    """
    try:
        optimizer_type = change['new']
        logger.info(f"Optimizer diubah ke {optimizer_type}")
        
        # Toggle visibilitas parameter sesuai dengan optimizer
        if optimizer_type == 'SGD':
            # Tampilkan parameter SGD
            toggle_widget_visibility(ui_components, 'momentum_slider', True)
            toggle_widget_visibility(ui_components, 'beta1_slider', False) if 'beta1_slider' in ui_components else None
            toggle_widget_visibility(ui_components, 'beta2_slider', False) if 'beta2_slider' in ui_components else None
            toggle_widget_visibility(ui_components, 'eps_slider', False) if 'eps_slider' in ui_components else None
        elif optimizer_type in ['Adam', 'AdamW']:
            # Tampilkan parameter Adam/AdamW
            toggle_widget_visibility(ui_components, 'momentum_slider', False)
            toggle_widget_visibility(ui_components, 'beta1_slider', True) if 'beta1_slider' in ui_components else None
            toggle_widget_visibility(ui_components, 'beta2_slider', True) if 'beta2_slider' in ui_components else None
            toggle_widget_visibility(ui_components, 'eps_slider', True) if 'eps_slider' in ui_components else None
        elif optimizer_type == 'RMSprop':
            # Tampilkan parameter RMSprop
            toggle_widget_visibility(ui_components, 'momentum_slider', False)
            toggle_widget_visibility(ui_components, 'beta1_slider', False) if 'beta1_slider' in ui_components else None
            toggle_widget_visibility(ui_components, 'beta2_slider', False) if 'beta2_slider' in ui_components else None
            toggle_widget_visibility(ui_components, 'eps_slider', True) if 'eps_slider' in ui_components else None
        
        # Update config dari UI
        update_config_from_ui(ui_components)
        
        # Update info panel jika ada
        if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
            ui_components['update_hyperparameters_info'](ui_components)
    except Exception as e:
        logger.error(f"Error saat mengubah optimizer: {str(e)}")
        show_error_status(ui_components, f"Error saat mengubah optimizer: {str(e)}")

def on_scheduler_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan status scheduler.
    
    Args:
        change: Objek perubahan dari widget
        ui_components: Dictionary komponen UI
    """
    try:
        enabled = change['new']
        logger.info(f"Scheduler {'diaktifkan' if enabled else 'dinonaktifkan'}")
        
        # Toggle visibilitas komponen scheduler
        scheduler_widgets = [
            'scheduler_dropdown',
            'warmup_epochs_slider',
            'warmup_momentum_slider',
            'warmup_bias_lr_slider'
        ]
        
        for widget_name in scheduler_widgets:
            if widget_name in ui_components:
                toggle_widget_visibility(ui_components, widget_name, enabled)
        
        # Update config dari UI
        update_config_from_ui(ui_components)
        
        # Update info panel jika ada
        if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
            ui_components['update_hyperparameters_info'](ui_components)
    except Exception as e:
        logger.error(f"Error saat mengubah status scheduler: {str(e)}")
        show_error_status(ui_components, f"Error saat mengubah status scheduler: {str(e)}")

def on_scheduler_type_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan tipe scheduler.
    
    Args:
        change: Objek perubahan dari widget
        ui_components: Dictionary komponen UI
    """
    try:
        scheduler_type = change['new']
        logger.info(f"Tipe scheduler diubah ke {scheduler_type}")
        
        # Sesuaikan parameter berdasarkan tipe scheduler
        if scheduler_type == 'cosine':
            # Cosine annealing scheduler
            if 'warmup_epochs_slider' in ui_components:
                ui_components['warmup_epochs_slider'].max = 10
        elif scheduler_type == 'step':
            # Step scheduler
            if 'warmup_epochs_slider' in ui_components:
                ui_components['warmup_epochs_slider'].max = 5
        elif scheduler_type == 'exp':
            # Exponential scheduler
            if 'warmup_epochs_slider' in ui_components:
                ui_components['warmup_epochs_slider'].max = 3
        
        # Update config dari UI
        update_config_from_ui(ui_components)
        
        # Update info panel jika ada
        if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
            ui_components['update_hyperparameters_info'](ui_components)
    except Exception as e:
        logger.error(f"Error saat mengubah tipe scheduler: {str(e)}")
        show_error_status(ui_components, f"Error saat mengubah tipe scheduler: {str(e)}")

def on_early_stopping_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan status early stopping.
    
    Args:
        change: Objek perubahan dari widget
        ui_components: Dictionary komponen UI
    """
    try:
        enabled = change['new']
        logger.info(f"Early stopping {'diaktifkan' if enabled else 'dinonaktifkan'}")
        
        # Toggle visibilitas komponen early stopping
        early_stopping_widgets = [
            'patience_slider',
            'min_delta_slider'
        ]
        
        for widget_name in early_stopping_widgets:
            if widget_name in ui_components:
                toggle_widget_visibility(ui_components, widget_name, enabled)
        
        # Update config dari UI
        update_config_from_ui(ui_components)
        
        # Update info panel jika ada
        if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
            ui_components['update_hyperparameters_info'](ui_components)
    except Exception as e:
        logger.error(f"Error saat mengubah status early stopping: {str(e)}")
        show_error_status(ui_components, f"Error saat mengubah status early stopping: {str(e)}")

def on_augmentation_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan status augmentasi.
    
    Args:
        change: Objek perubahan dari widget
        ui_components: Dictionary komponen UI
    """
    try:
        enabled = change['new']
        logger.info(f"Data augmentation {'diaktifkan' if enabled else 'dinonaktifkan'}")
        
        # Update config dari UI
        update_config_from_ui(ui_components)
        
        # Update info panel jika ada
        if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
            ui_components['update_hyperparameters_info'](ui_components)
    except Exception as e:
        logger.error(f"Error saat mengubah status augmentasi: {str(e)}")
        show_error_status(ui_components, f"Error saat mengubah status augmentasi: {str(e)}")