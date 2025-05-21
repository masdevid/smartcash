"""
File: smartcash/ui/training_config/hyperparameters/handlers/config_reader.py
Deskripsi: Fungsi untuk membaca konfigurasi hyperparameters dari UI
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.common.logger import get_logger, LogLevel
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.training_config.hyperparameters.handlers.default_config import get_default_hyperparameters_config
from smartcash.ui.training_config.hyperparameters.handlers.config_manager import get_hyperparameters_config as get_saved_config

# Setup logger dengan level INFO
logger = get_logger(__name__)
logger.set_level(LogLevel.INFO)

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi hyperparameters dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        # Dapatkan konfigurasi yang sudah ada
        config = get_saved_config()
        
        # Pastikan config memiliki struktur yang benar
        if 'hyperparameters' not in config:
            config = {'hyperparameters': {}}
            
        hp_config = config['hyperparameters']
        
        # Update training parameters
        if 'training' not in hp_config:
            hp_config['training'] = {}
            
        if 'batch_size_slider' in ui_components:
            hp_config['training']['batch_size'] = ui_components['batch_size_slider'].value
            
        if 'image_size_slider' in ui_components:
            hp_config['training']['image_size'] = ui_components['image_size_slider'].value
            
        if 'epochs_slider' in ui_components:
            hp_config['training']['epochs'] = ui_components['epochs_slider'].value
            
        if 'dropout_slider' in ui_components:
            hp_config['training']['dropout'] = ui_components['dropout_slider'].value
        
        # Update optimizer settings
        if 'optimizer' not in hp_config:
            hp_config['optimizer'] = {}
            
        if 'optimizer_dropdown' in ui_components:
            hp_config['optimizer']['type'] = ui_components['optimizer_dropdown'].value
            
        if 'learning_rate_slider' in ui_components:
            hp_config['optimizer']['learning_rate'] = ui_components['learning_rate_slider'].value
            
        if 'weight_decay_slider' in ui_components:
            hp_config['optimizer']['weight_decay'] = ui_components['weight_decay_slider'].value
            
        if 'momentum_slider' in ui_components:
            hp_config['optimizer']['momentum'] = ui_components['momentum_slider'].value
        
        # Update scheduler settings
        if 'scheduler' not in hp_config:
            hp_config['scheduler'] = {}
            
        if 'scheduler_checkbox' in ui_components:
            hp_config['scheduler']['enabled'] = ui_components['scheduler_checkbox'].value
            
        if 'scheduler_dropdown' in ui_components:
            hp_config['scheduler']['type'] = ui_components['scheduler_dropdown'].value
            
        if 'warmup_epochs_slider' in ui_components:
            hp_config['scheduler']['warmup_epochs'] = ui_components['warmup_epochs_slider'].value
            
        if 'warmup_momentum_slider' in ui_components:
            hp_config['scheduler']['warmup_momentum'] = ui_components['warmup_momentum_slider'].value
            
        if 'warmup_bias_lr_slider' in ui_components:
            hp_config['scheduler']['warmup_bias_lr'] = ui_components['warmup_bias_lr_slider'].value
        
        # Update early stopping settings
        if 'early_stopping' not in hp_config:
            hp_config['early_stopping'] = {}
            
        if 'early_stopping_checkbox' in ui_components:
            hp_config['early_stopping']['enabled'] = ui_components['early_stopping_checkbox'].value
            
        if 'patience_slider' in ui_components:
            hp_config['early_stopping']['patience'] = ui_components['patience_slider'].value
            
        if 'min_delta_slider' in ui_components:
            hp_config['early_stopping']['min_delta'] = ui_components['min_delta_slider'].value
        
        # Update loss settings
        if 'loss' not in hp_config:
            hp_config['loss'] = {}
            
        if 'box_loss_gain_slider' in ui_components:
            hp_config['loss']['box_loss_gain'] = ui_components['box_loss_gain_slider'].value
            
        if 'cls_loss_gain_slider' in ui_components:
            hp_config['loss']['cls_loss_gain'] = ui_components['cls_loss_gain_slider'].value
            
        if 'obj_loss_gain_slider' in ui_components:
            hp_config['loss']['obj_loss_gain'] = ui_components['obj_loss_gain_slider'].value
        
        # Update augmentation settings
        if 'augmentation' not in hp_config:
            hp_config['augmentation'] = {}
            
        if 'augment_checkbox' in ui_components:
            hp_config['augmentation']['enabled'] = ui_components['augment_checkbox'].value
        
        # Update checkpoint settings
        if 'checkpoint' not in hp_config:
            hp_config['checkpoint'] = {}
            
        if 'save_best_checkbox' in ui_components:
            hp_config['checkpoint']['save_best'] = ui_components['save_best_checkbox'].value
            
        if 'checkpoint_metric_dropdown' in ui_components:
            hp_config['checkpoint']['metric'] = ui_components['checkpoint_metric_dropdown'].value
        
        logger.info("✅ Konfigurasi berhasil diupdate dari UI")
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update konfigurasi dari UI: {str(e)}")
        return get_default_hyperparameters_config()