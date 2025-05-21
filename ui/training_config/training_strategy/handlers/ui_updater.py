"""
File: smartcash/ui/training_config/training_strategy/handlers/ui_updater.py
Deskripsi: Fungsi untuk mengupdate UI dari konfigurasi training strategy
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """
    Update UI dari konfigurasi training strategy.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan digunakan (opsional)
    """
    try:
        # Get config if not provided
        if config is None:
            from smartcash.ui.training_config.training_strategy.handlers.config_loader import get_training_strategy_config
            logger.info("Mengambil konfigurasi training strategy dari config manager")
            config = get_training_strategy_config(ui_components)
            
        # Ensure config has training_strategy key
        if 'training_strategy' not in config:
            logger.info("Menambahkan key 'training_strategy' ke konfigurasi")
            config = {'training_strategy': config}
            
        logger.info(f"Memperbarui UI dari konfigurasi training strategy")
            
        # Helper untuk update value jika key ada
        def safe_set(key, value):
            if key in ui_components:
                try:
                    # Simpan nilai lama untuk debugging
                    old_value = ui_components[key].value
                    # Set nilai baru
                    ui_components[key].value = value
                    # Log jika nilai tidak berubah
                    if old_value == ui_components[key].value and old_value != value:
                        logger.warning(f"Nilai untuk '{key}' tidak berubah: {old_value} -> {ui_components[key].value}, seharusnya {value}")
                except Exception as e:
                    logger.warning(f"Error saat mengubah nilai untuk '{key}': {str(e)}")
            else:
                logger.warning(f"Key '{key}' tidak ditemukan di ui_components, skip update komponen.")
                
        # Update UI components dengan nilai dari config
        ts_config = config['training_strategy']
        
        # Parameter utama
        safe_set('enabled_checkbox', ts_config.get('enabled', True))
        safe_set('batch_size_slider', ts_config.get('batch_size', 16))
        safe_set('epochs_slider', ts_config.get('epochs', 100))
        safe_set('learning_rate_slider', ts_config.get('learning_rate', 0.001))
        
        # Optimizer
        optimizer = ts_config.get('optimizer', {})
        safe_set('optimizer_dropdown', optimizer.get('type', 'adam'))
        safe_set('weight_decay_slider', optimizer.get('weight_decay', 0.0005))
        safe_set('momentum_slider', optimizer.get('momentum', 0.9))
        
        # Scheduler
        scheduler = ts_config.get('scheduler', {})
        safe_set('scheduler_checkbox', scheduler.get('enabled', True))
        safe_set('scheduler_dropdown', scheduler.get('type', 'cosine'))
        safe_set('warmup_epochs_slider', scheduler.get('warmup_epochs', 5))
        safe_set('min_lr_slider', scheduler.get('min_lr', 0.00001))
        
        # Early stopping
        early_stopping = ts_config.get('early_stopping', {})
        safe_set('early_stopping_checkbox', early_stopping.get('enabled', True))
        safe_set('patience_slider', early_stopping.get('patience', 10))
        safe_set('min_delta_slider', early_stopping.get('min_delta', 0.001))
        
        # Checkpoint
        checkpoint = ts_config.get('checkpoint', {})
        safe_set('checkpoint_checkbox', checkpoint.get('enabled', True))
        safe_set('save_best_only_checkbox', checkpoint.get('save_best_only', True))
        safe_set('save_freq_slider', checkpoint.get('save_freq', 1))
        
        # Utils
        utils = ts_config.get('utils', {})
        safe_set('experiment_name', utils.get('experiment_name', 'efficientnet_b4_training'))
        safe_set('checkpoint_dir', utils.get('checkpoint_dir', '/content/runs/train/checkpoints'))
        safe_set('tensorboard', utils.get('tensorboard', True))
        safe_set('log_metrics_every', utils.get('log_metrics_every', 10))
        safe_set('visualize_batch_every', utils.get('visualize_batch_every', 100))
        safe_set('gradient_clipping', utils.get('gradient_clipping', 1.0))
        safe_set('mixed_precision', utils.get('mixed_precision', True))
        safe_set('layer_mode', utils.get('layer_mode', 'single'))
        
        # Validation
        validation = ts_config.get('validation', {})
        safe_set('validation_frequency', validation.get('validation_frequency', 1))
        safe_set('iou_threshold', validation.get('iou_threshold', 0.6))
        safe_set('conf_threshold', validation.get('conf_threshold', 0.001))
        
        # Multiscale
        multiscale = ts_config.get('multiscale', {})
        safe_set('multi_scale', multiscale.get('enabled', True))
        
        # Force update UI by triggering change events for boolean widgets
        for widget_name, widget in ui_components.items():
            if isinstance(widget, widgets.Checkbox) and widget_name in ['enabled_checkbox', 'scheduler_checkbox', 
                                                                      'early_stopping_checkbox', 'checkpoint_checkbox', 
                                                                      'tensorboard', 'mixed_precision', 'multi_scale']:
                try:
                    # Trigger a change event to update dependent widgets
                    old_value = widget.value
                    # Toggle boolean value to trigger change event
                    widget.value = not old_value
                    widget.value = old_value
                except Exception as e:
                    logger.warning(f"Error saat trigger event untuk '{widget_name}': {str(e)}")
                
        logger.info(f"{ICONS.get('success', '✅')} UI berhasil diupdate dari konfigurasi training strategy")
        
        # Update info panel
        from smartcash.ui.training_config.training_strategy.handlers.info_updater import update_training_strategy_info
        update_training_strategy_info(ui_components)
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mengupdate UI dari konfigurasi: {str(e)}")
        if 'info_panel' in ui_components:
            with ui_components['info_panel']:
                clear_output(wait=True)
                display(widgets.HTML(f"<div style='color:red'>{ICONS.get('error', '❌')} Error: {str(e)}</div>"))