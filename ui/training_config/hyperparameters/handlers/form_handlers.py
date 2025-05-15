"""
File: smartcash/ui/training_config/hyperparameters/handlers/form_handlers.py
Deskripsi: Handler untuk form pada komponen UI hyperparameter
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager

logger = get_logger(__name__)

def setup_hyperparameters_form_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk form pada komponen UI hyperparameter.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        # Handler untuk perubahan optimizer
        def on_optimizer_type_change(change):
            optimizer_type = change['new']
            
            # Aktifkan/nonaktifkan parameter yang relevan berdasarkan optimizer
            if optimizer_type == 'SGD':
                # SGD memerlukan momentum
                ui_components['momentum_slider'].disabled = False
                ui_components['weight_decay_slider'].disabled = False
            elif optimizer_type == 'Adam' or optimizer_type == 'AdamW':
                # Adam dan AdamW tidak memerlukan momentum
                ui_components['momentum_slider'].disabled = True
                ui_components['weight_decay_slider'].disabled = False
            elif optimizer_type == 'RMSprop':
                # RMSprop memerlukan momentum
                ui_components['momentum_slider'].disabled = False
                ui_components['weight_decay_slider'].disabled = False
            
            # Update info panel jika ada
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info']()
        
        # Handler untuk perubahan scheduler
        def on_scheduler_type_change(change):
            scheduler_type = change['new']
            
            # Aktifkan/nonaktifkan parameter yang relevan berdasarkan scheduler
            if scheduler_type == 'none':
                # Tidak ada scheduler, nonaktifkan semua parameter terkait
                ui_components['warmup_epochs_slider'].disabled = True
                ui_components['warmup_momentum_slider'].disabled = True
                ui_components['warmup_bias_lr_slider'].disabled = True
            else:
                # Ada scheduler, aktifkan parameter terkait
                ui_components['warmup_epochs_slider'].disabled = False
                ui_components['warmup_momentum_slider'].disabled = False
                ui_components['warmup_bias_lr_slider'].disabled = False
            
            # Update info panel jika ada
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info']()
        
        # Handler untuk perubahan early stopping
        def on_early_stopping_enabled_change(change):
            early_stopping_enabled = change['new']
            
            # Aktifkan/nonaktifkan parameter yang relevan berdasarkan early stopping
            ui_components['early_stopping_patience_slider'].disabled = not early_stopping_enabled
            ui_components['early_stopping_min_delta_slider'].disabled = not early_stopping_enabled
            
            # Update info panel jika ada
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info']()
        
        # Handler untuk perubahan save best
        def on_save_best_change(change):
            save_best = change['new']
            
            # Aktifkan/nonaktifkan parameter yang relevan berdasarkan save best
            ui_components['checkpoint_metric_dropdown'].disabled = not save_best
            
            # Update info panel jika ada
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info']()
        
        # Handler untuk perubahan augment
        def on_augment_change(change):
            augment = change['new']
            
            # Update info panel jika ada
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info']()
        
        # Handler untuk perubahan batch size
        def on_batch_size_change(change):
            # Update info panel jika ada
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info']()
        
        # Handler untuk perubahan image size
        def on_image_size_change(change):
            # Update info panel jika ada
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info']()
        
        # Handler untuk perubahan epochs
        def on_epochs_change(change):
            # Update info panel jika ada
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info']()
        
        # Handler untuk perubahan learning rate
        def on_learning_rate_change(change):
            # Update info panel jika ada
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info']()
        
        # Pasang handler ke komponen UI
        if 'optimizer_dropdown' in ui_components:
            ui_components['optimizer_dropdown'].observe(on_optimizer_type_change, names='value')
        
        if 'scheduler_dropdown' in ui_components:
            ui_components['scheduler_dropdown'].observe(on_scheduler_type_change, names='value')
        
        if 'early_stopping_enabled_checkbox' in ui_components:
            ui_components['early_stopping_enabled_checkbox'].observe(on_early_stopping_enabled_change, names='value')
        
        if 'save_best_checkbox' in ui_components:
            ui_components['save_best_checkbox'].observe(on_save_best_change, names='value')
        
        if 'augment_checkbox' in ui_components:
            ui_components['augment_checkbox'].observe(on_augment_change, names='value')
        
        if 'batch_size_slider' in ui_components:
            ui_components['batch_size_slider'].observe(on_batch_size_change, names='value')
        
        if 'image_size_slider' in ui_components:
            ui_components['image_size_slider'].observe(on_image_size_change, names='value')
        
        if 'epochs_slider' in ui_components:
            ui_components['epochs_slider'].observe(on_epochs_change, names='value')
        
        if 'learning_rate_slider' in ui_components:
            ui_components['learning_rate_slider'].observe(on_learning_rate_change, names='value')
        
        # Tambahkan handler ke ui_components
        ui_components.update({
            'on_optimizer_type_change': on_optimizer_type_change,
            'on_scheduler_type_change': on_scheduler_type_change,
            'on_early_stopping_enabled_change': on_early_stopping_enabled_change,
            'on_save_best_change': on_save_best_change,
            'on_augment_change': on_augment_change,
            'on_batch_size_change': on_batch_size_change,
            'on_image_size_change': on_image_size_change,
            'on_epochs_change': on_epochs_change,
            'on_learning_rate_change': on_learning_rate_change
        })
        
        # Set state awal komponen berdasarkan nilai saat ini
        if 'optimizer_dropdown' in ui_components:
            on_optimizer_type_change({'new': ui_components['optimizer_dropdown'].value})
        
        if 'scheduler_dropdown' in ui_components:
            on_scheduler_type_change({'new': ui_components['scheduler_dropdown'].value})
        
        if 'early_stopping_enabled_checkbox' in ui_components:
            on_early_stopping_enabled_change({'new': ui_components['early_stopping_enabled_checkbox'].value})
        
        if 'save_best_checkbox' in ui_components:
            on_save_best_change({'new': ui_components['save_best_checkbox'].value})
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup form handlers: {str(e)}")
        return ui_components
