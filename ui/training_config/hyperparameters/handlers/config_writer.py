"""
File: smartcash/ui/training_config/hyperparameters/handlers/config_writer.py
Deskripsi: Fungsi untuk mengupdate UI dengan konfigurasi hyperparameters
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.common.logger import get_logger, LogLevel
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.training_config.hyperparameters.handlers.default_config import get_default_hyperparameters_config
from smartcash.ui.training_config.hyperparameters.handlers.config_manager import get_hyperparameters_config

# Setup logger dengan level INFO
logger = get_logger(__name__)
logger.set_level(LogLevel.INFO)

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """
    Update UI dari konfigurasi hyperparameters.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan digunakan (opsional)
    """
    try:
        # Get config if not provided
        if config is None:
            logger.info("Mengambil konfigurasi hyperparameters dari config manager")
            config = get_hyperparameters_config()
            
        # Ensure config has hyperparameters key
        if 'hyperparameters' not in config:
            logger.info("Menambahkan key 'hyperparameters' ke konfigurasi")
            config = {'hyperparameters': config}
            
        # Get hyperparameters config with safe defaults
        hp_config = config['hyperparameters']
        default_config = get_default_hyperparameters_config()['hyperparameters']
        
        logger.info(f"Memperbarui UI dari konfigurasi: {len(hp_config.keys())} kategori")
        
        # Update training parameters
        training = hp_config.get('training', default_config.get('training', {}))
        logger.info(f"Memperbarui parameter training")
        
        if 'batch_size_slider' in ui_components:
            ui_components['batch_size_slider'].value = training.get('batch_size', 16)
            
        if 'image_size_slider' in ui_components:
            ui_components['image_size_slider'].value = training.get('image_size', 640)
            
        if 'epochs_slider' in ui_components:
            ui_components['epochs_slider'].value = training.get('epochs', 100)
            
        if 'dropout_slider' in ui_components:
            ui_components['dropout_slider'].value = training.get('dropout', 0.0)
            
        # Update optimizer UI components
        optimizer = hp_config.get('optimizer', default_config['optimizer'])
        logger.info(f"Memperbarui komponen optimizer: {optimizer.get('type', 'unknown')}")
        
        if 'optimizer_dropdown' in ui_components:
            ui_components['optimizer_dropdown'].value = optimizer.get('type', default_config['optimizer']['type'])
            
        if 'learning_rate_slider' in ui_components:
            ui_components['learning_rate_slider'].value = optimizer.get('learning_rate', default_config['optimizer']['learning_rate'])
            
        if 'weight_decay_slider' in ui_components:
            ui_components['weight_decay_slider'].value = optimizer.get('weight_decay', default_config['optimizer']['weight_decay'])
            
        if 'momentum_slider' in ui_components:
            ui_components['momentum_slider'].value = optimizer.get('momentum', default_config['optimizer']['momentum'])
            
        # Update scheduler UI components
        scheduler = hp_config.get('scheduler', default_config['scheduler'])
        logger.info(f"Memperbarui komponen scheduler: {scheduler.get('type', 'unknown')}, enabled: {scheduler.get('enabled', False)}")
        
        if 'scheduler_checkbox' in ui_components:
            ui_components['scheduler_checkbox'].value = scheduler.get('enabled', default_config['scheduler']['enabled'])
            
        if 'scheduler_dropdown' in ui_components:
            ui_components['scheduler_dropdown'].value = scheduler.get('type', default_config['scheduler']['type'])
            
        if 'warmup_epochs_slider' in ui_components:
            ui_components['warmup_epochs_slider'].value = scheduler.get('warmup_epochs', default_config['scheduler']['warmup_epochs'])
            
        if 'warmup_momentum_slider' in ui_components:
            ui_components['warmup_momentum_slider'].value = scheduler.get('warmup_momentum', 0.8)
            
        if 'warmup_bias_lr_slider' in ui_components:
            ui_components['warmup_bias_lr_slider'].value = scheduler.get('warmup_bias_lr', 0.1)
            
        # Update loss UI components
        loss = hp_config.get('loss', default_config['loss'])
        logger.info(f"Memperbarui komponen loss")
        
        if 'box_loss_gain_slider' in ui_components:
            ui_components['box_loss_gain_slider'].value = loss.get('box_loss_gain', default_config['loss']['box_loss_gain'])
            
        if 'cls_loss_gain_slider' in ui_components:
            ui_components['cls_loss_gain_slider'].value = loss.get('cls_loss_gain', default_config['loss']['cls_loss_gain'])
            
        if 'obj_loss_gain_slider' in ui_components:
            ui_components['obj_loss_gain_slider'].value = loss.get('obj_loss_gain', default_config['loss']['obj_loss_gain'])
        
        # Update early stopping UI components
        early_stopping = hp_config.get('early_stopping', default_config.get('early_stopping', {}))
        logger.info(f"Memperbarui komponen early stopping")
        
        if 'early_stopping_checkbox' in ui_components:
            ui_components['early_stopping_checkbox'].value = early_stopping.get('enabled', True)
            
        if 'patience_slider' in ui_components:
            ui_components['patience_slider'].value = early_stopping.get('patience', 10)
            
        if 'min_delta_slider' in ui_components:
            ui_components['min_delta_slider'].value = early_stopping.get('min_delta', 0.001)
            
        # Update augmentation UI components
        augmentation = hp_config.get('augmentation', default_config['augmentation'])
        logger.info(f"Memperbarui komponen augmentation: enabled: {augmentation.get('enabled', False)}")
        
        if 'augment_checkbox' in ui_components:
            ui_components['augment_checkbox'].value = augmentation.get('enabled', default_config['augmentation']['enabled'])
            
        # Update checkpoint UI components
        checkpoint = hp_config.get('checkpoint', default_config.get('checkpoint', {}))
        logger.info(f"Memperbarui komponen checkpoint")
        
        if 'save_best_checkbox' in ui_components:
            ui_components['save_best_checkbox'].value = checkpoint.get('save_best', True)
            
        if 'checkpoint_metric_dropdown' in ui_components:
            ui_components['checkpoint_metric_dropdown'].value = checkpoint.get('metric', 'mAP_0.5')
            
        logger.info("✅ UI berhasil diupdate dari konfigurasi hyperparameters")
        
    except Exception as e:
        logger.error(f"❌ Error saat mengupdate UI dari konfigurasi: {str(e)}")