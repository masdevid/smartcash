"""
File: smartcash/ui/training_config/hyperparameters/handlers/config_handlers.py
Deskripsi: Handler untuk konfigurasi hyperparameter
"""

from typing import Dict, Any, Optional, List, Callable
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager

logger = get_logger(__name__)

def update_ui_from_config(ui_components: Dict[str, Any], config_to_use: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Update UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config_to_use: Konfigurasi yang akan digunakan (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Dapatkan config manager
    config_manager = get_config_manager()
    
    # Dapatkan konfigurasi yang akan digunakan
    if config_to_use is None:
        # Gunakan config dari ui_components jika ada
        if 'config' in ui_components:
            current_config = ui_components['config']
        else:
            # Jika tidak ada, ambil dari config manager
            current_config = config_manager.get_module_config('hyperparameters', {})
    else:
        current_config = config_to_use
    
    try:
        
        # Update slider batch size
        if 'batch_size_slider' in ui_components:
            ui_components['batch_size_slider'].value = hp_config.get('batch_size', 16)
        
        # Update slider image size
        if 'image_size_slider' in ui_components:
            ui_components['image_size_slider'].value = hp_config.get('image_size', 640)
        
        # Update slider epochs
        if 'epochs_slider' in ui_components:
            ui_components['epochs_slider'].value = hp_config.get('epochs', 50)
        
        # Update slider learning rate
        if 'learning_rate_slider' in ui_components:
            ui_components['learning_rate_slider'].value = hp_config.get('learning_rate', 0.001)
        
        # Update dropdown optimizer
        if 'optimizer_dropdown' in ui_components:
            optimizer = hp_config.get('optimizer', 'Adam')
            if optimizer in ui_components['optimizer_dropdown'].options:
                ui_components['optimizer_dropdown'].value = optimizer
        
        # Update slider weight decay
        if 'weight_decay_slider' in ui_components:
            ui_components['weight_decay_slider'].value = hp_config.get('weight_decay', 0.0005)
        
        # Update slider momentum
        if 'momentum_slider' in ui_components:
            ui_components['momentum_slider'].value = hp_config.get('momentum', 0.937)
        
        # Update dropdown scheduler
        if 'scheduler_dropdown' in ui_components:
            scheduler = hp_config.get('lr_scheduler', 'cosine')
            if scheduler in ui_components['scheduler_dropdown'].options:
                ui_components['scheduler_dropdown'].value = scheduler
        
        # Update slider warmup epochs
        if 'warmup_epochs_slider' in ui_components:
            ui_components['warmup_epochs_slider'].value = hp_config.get('warmup_epochs', 3)
        
        # Update slider warmup momentum
        if 'warmup_momentum_slider' in ui_components:
            ui_components['warmup_momentum_slider'].value = hp_config.get('warmup_momentum', 0.8)
        
        # Update slider warmup bias lr
        if 'warmup_bias_lr_slider' in ui_components:
            ui_components['warmup_bias_lr_slider'].value = hp_config.get('warmup_bias_lr', 0.1)
        
        # Update checkbox augment
        if 'augment_checkbox' in ui_components:
            ui_components['augment_checkbox'].value = hp_config.get('augment', True)
        
        # Update slider dropout
        if 'dropout_slider' in ui_components:
            ui_components['dropout_slider'].value = hp_config.get('dropout', 0.0)
        
        # Update slider box loss gain
        if 'box_loss_gain_slider' in ui_components:
            ui_components['box_loss_gain_slider'].value = hp_config.get('box_loss_gain', 0.05)
        
        # Update slider cls loss gain
        if 'cls_loss_gain_slider' in ui_components:
            ui_components['cls_loss_gain_slider'].value = hp_config.get('cls_loss_gain', 0.5)
        
        # Update slider obj loss gain
        if 'obj_loss_gain_slider' in ui_components:
            ui_components['obj_loss_gain_slider'].value = current_config.get('obj_loss_gain', 1.0)
        
        # Update early stopping
        if 'early_stopping_enabled_checkbox' in ui_components:
            early_stopping = current_config.get('early_stopping', {})
            ui_components['early_stopping_enabled_checkbox'].value = early_stopping.get('enabled', True)
            
            if 'early_stopping_patience_slider' in ui_components:
                ui_components['early_stopping_patience_slider'].value = early_stopping.get('patience', 10)
            
            if 'early_stopping_min_delta_slider' in ui_components:
                ui_components['early_stopping_min_delta_slider'].value = early_stopping.get('min_delta', 0.001)
        
        # Update checkpoint
        if 'save_best_checkbox' in ui_components:
            save_best = current_config.get('save_best', {})
            ui_components['save_best_checkbox'].value = save_best.get('enabled', True)
            
            if 'checkpoint_metric_dropdown' in ui_components:
                metric = save_best.get('metric', 'mAP_0.5')
                if metric in ui_components['checkpoint_metric_dropdown'].options:
                    ui_components['checkpoint_metric_dropdown'].value = metric
        
        # Simpan referensi config di ui_components
        ui_components['config'] = current_config
        
        # Update info panel jika ada
        if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
            ui_components['update_hyperparameters_info']()
        
        logger.info(f"{ICONS.get('success', '✅')} UI hyperparameter berhasil diupdate dari konfigurasi")
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error update UI: {str(e)}")
    
    return ui_components

def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari nilai UI.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi hyperparameter
        
    Returns:
        Konfigurasi yang diupdate
    """
    try:
        # Pastikan struktur config ada
        if 'hyperparameters' not in config:
            config['hyperparameters'] = {}
        
        # Update batch size
        if 'batch_size_slider' in ui_components:
            config['batch_size'] = ui_components['batch_size_slider'].value
        
        # Update image size
        if 'image_size_slider' in ui_components:
            config['image_size'] = ui_components['image_size_slider'].value
        
        # Update epochs
        if 'epochs_slider' in ui_components:
            config['epochs'] = ui_components['epochs_slider'].value
        
        # Update optimizer
        if 'optimizer_dropdown' in ui_components:
            config['optimizer'] = ui_components['optimizer_dropdown'].value
        
        # Update learning rate
        if 'learning_rate_slider' in ui_components:
            config['learning_rate'] = ui_components['learning_rate_slider'].value
        
        # Update weight decay
        if 'weight_decay_slider' in ui_components:
            config['weight_decay'] = ui_components['weight_decay_slider'].value
        
        # Update momentum
        if 'momentum_slider' in ui_components:
            config['momentum'] = ui_components['momentum_slider'].value
        
        # Update scheduler
        if 'scheduler_dropdown' in ui_components:
            config['scheduler'] = ui_components['scheduler_dropdown'].value
        
        # Update warmup epochs
        if 'warmup_epochs_slider' in ui_components:
            config['warmup_epochs'] = ui_components['warmup_epochs_slider'].value
        
        # Update warmup momentum
        if 'warmup_momentum_slider' in ui_components:
            config['warmup_momentum'] = ui_components['warmup_momentum_slider'].value
        
        # Update warmup bias lr
        if 'warmup_bias_lr_slider' in ui_components:
            config['warmup_bias_lr'] = ui_components['warmup_bias_lr_slider'].value
        
        # Update augment
        if 'augment_checkbox' in ui_components:
            config['augment'] = ui_components['augment_checkbox'].value
        
        if 'dropout_slider' in ui_components:
            config['hyperparameters']['dropout'] = ui_components['dropout_slider'].value
        
        if 'box_loss_gain_slider' in ui_components:
            config['hyperparameters']['box_loss_gain'] = ui_components['box_loss_gain_slider'].value
        
        if 'cls_loss_gain_slider' in ui_components:
            config['hyperparameters']['cls_loss_gain'] = ui_components['cls_loss_gain_slider'].value
        
        if 'obj_loss_gain_slider' in ui_components:
            config['hyperparameters']['obj_loss_gain'] = ui_components['obj_loss_gain_slider'].value
        
        # Update early stopping
        if 'early_stopping_enabled_checkbox' in ui_components or 'early_stopping_patience_slider' in ui_components or 'early_stopping_min_delta_slider' in ui_components:
            if 'early_stopping' not in config:
                config['early_stopping'] = {}
            
            if 'early_stopping_enabled_checkbox' in ui_components:
                config['early_stopping']['enabled'] = ui_components['early_stopping_enabled_checkbox'].value
            
            if 'early_stopping_patience_slider' in ui_components:
                config['early_stopping']['patience'] = ui_components['early_stopping_patience_slider'].value
            
            if 'early_stopping_min_delta_slider' in ui_components:
                config['early_stopping']['min_delta'] = ui_components['early_stopping_min_delta_slider'].value
        
        # Update save best
        if 'save_best_checkbox' in ui_components or 'checkpoint_metric_dropdown' in ui_components:
            if 'save_best' not in config:
                config['save_best'] = {}
            
            if 'save_best_checkbox' in ui_components:
                config['save_best']['enabled'] = ui_components['save_best_checkbox'].value
            
            if 'checkpoint_metric_dropdown' in ui_components:
                config['save_best']['metric'] = ui_components['checkpoint_metric_dropdown'].value
        
        # Simpan konfigurasi di ui_components
        ui_components['config'] = config
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil diupdate dari UI")
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error update config: {str(e)}")
    
    return config

def update_hyperparameters_info(ui_components: Dict[str, Any]) -> None:
    """
    Update informasi hyperparameter pada panel info.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    if 'info_panel' not in ui_components:
        return
    
    try:
        # Dapatkan konfigurasi
        config = ui_components.get('config', {})
        
        # Buat HTML untuk info panel
        info_html = f"""
        <div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>
            <h4 style='margin-top: 0;'>{ICONS.get('info', 'ℹ️')} Informasi Hyperparameter</h4>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                <div>
                    <p><strong>Batch Size:</strong> {config.get('batch_size', 16)}</p>
                    <p><strong>Image Size:</strong> {config.get('image_size', 640)}</p>
                    <p><strong>Epochs:</strong> {config.get('epochs', 100)}</p>
                    <p><strong>Optimizer:</strong> {config.get('optimizer', 'SGD')}</p>
                    <p><strong>Learning Rate:</strong> {config.get('learning_rate', 0.01)}</p>
                    <p><strong>Weight Decay:</strong> {config.get('weight_decay', 0.0005)}</p>
                </div>
                <div>
                    <p><strong>Scheduler:</strong> {config.get('scheduler', 'cosine')}</p>
                    <p><strong>Augmentation:</strong> {'Aktif' if config.get('augment', True) else 'Nonaktif'}</p>
                    <p><strong>Early Stopping:</strong> {'Aktif' if config.get('early_stopping', {}).get('enabled', True) else 'Nonaktif'}</p>
                    <p><strong>Save Best Model:</strong> {'Ya' if config.get('save_best', {}).get('enabled', True) else 'Tidak'}</p>
                    <p><strong>Checkpoint Metric:</strong> {config.get('save_best', {}).get('metric', 'mAP_0.5')}</p>
                </div>
            </div>
        </div>
        """
        
        # Update info panel
        with ui_components['info_panel']:
            clear_output(wait=True)
            display(widgets.HTML(info_html))
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat update info hyperparameter: {str(e)}")
        with ui_components['info_panel']:
            clear_output(wait=True)
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Gagal memperbarui informasi hyperparameter: {str(e)}",
                alert_type='error'
            ))
