"""
File: smartcash/ui/training_config/hyperparameters/handlers/config_handlers.py
Deskripsi: Handler konfigurasi untuk hyperparameters training
"""

from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

def get_default_hyperparameters_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk hyperparameters training.
    
    Returns:
        Dictionary konfigurasi default
    """
    return {
        "hyperparameters": {
            "enabled": True,
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0005,
                "momentum": 0.9,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8
            },
            "scheduler": {
                "enabled": True,
                "type": "cosine",
                "warmup_epochs": 5,
                "min_lr": 0.00001,
                "patience": 10,
                "factor": 0.1,
                "threshold": 0.001
            },
            "loss": {
                "type": "focal",
                "alpha": 0.25,
                "gamma": 2.0,
                "label_smoothing": 0.1,
                "box_loss_gain": 0.05,
                "cls_loss_gain": 0.5,
                "obj_loss_gain": 1.0
            },
            "augmentation": {
                "enabled": True,
                "mosaic": True,
                "mixup": True,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic_prob": 1.0,
                "mixup_prob": 0.0
            }
        }
    }

def get_hyperparameters_config(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi hyperparameters dari config manager.
    
    Args:
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Dictionary konfigurasi hyperparameters
    """
    try:
        # Get config manager
        config_manager = get_config_manager()
        
        # Get config
        config = config_manager.get_module_config('hyperparameters')
        
        if config:
            return config
            
        # Jika tidak ada config, gunakan default
        logger.warning("⚠️ Konfigurasi hyperparameters tidak ditemukan, menggunakan default")
        return get_default_hyperparameters_config()
        
    except Exception as e:
        logger.error(f"❌ Error saat mengambil konfigurasi hyperparameters: {str(e)}")
        return get_default_hyperparameters_config()

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi hyperparameters dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        # Get config manager
        config_manager = get_config_manager()
        
        # Get current config
        config = config_manager.get_module_config('hyperparameters') or get_default_hyperparameters_config()
        
        # Update config from UI
        if 'enabled_checkbox' in ui_components:
            config['hyperparameters']['enabled'] = ui_components['enabled_checkbox'].value
            
        # Optimizer
        if 'optimizer_dropdown' in ui_components:
            config['hyperparameters']['optimizer']['type'] = ui_components['optimizer_dropdown'].value
            
        if 'learning_rate_slider' in ui_components:
            config['hyperparameters']['optimizer']['learning_rate'] = ui_components['learning_rate_slider'].value
            
        if 'weight_decay_slider' in ui_components:
            config['hyperparameters']['optimizer']['weight_decay'] = ui_components['weight_decay_slider'].value
            
        if 'momentum_slider' in ui_components:
            config['hyperparameters']['optimizer']['momentum'] = ui_components['momentum_slider'].value
            
        if 'beta1_slider' in ui_components:
            config['hyperparameters']['optimizer']['beta1'] = ui_components['beta1_slider'].value
            
        if 'beta2_slider' in ui_components:
            config['hyperparameters']['optimizer']['beta2'] = ui_components['beta2_slider'].value
            
        if 'eps_slider' in ui_components:
            config['hyperparameters']['optimizer']['eps'] = ui_components['eps_slider'].value
            
        # Scheduler
        if 'scheduler_checkbox' in ui_components:
            config['hyperparameters']['scheduler']['enabled'] = ui_components['scheduler_checkbox'].value
            
        if 'scheduler_dropdown' in ui_components:
            config['hyperparameters']['scheduler']['type'] = ui_components['scheduler_dropdown'].value
            
        if 'warmup_epochs_slider' in ui_components:
            config['hyperparameters']['scheduler']['warmup_epochs'] = ui_components['warmup_epochs_slider'].value
            
        if 'min_lr_slider' in ui_components:
            config['hyperparameters']['scheduler']['min_lr'] = ui_components['min_lr_slider'].value
            
        if 'patience_slider' in ui_components:
            config['hyperparameters']['scheduler']['patience'] = ui_components['patience_slider'].value
            
        if 'factor_slider' in ui_components:
            config['hyperparameters']['scheduler']['factor'] = ui_components['factor_slider'].value
            
        if 'threshold_slider' in ui_components:
            config['hyperparameters']['scheduler']['threshold'] = ui_components['threshold_slider'].value
            
        # Loss
        if 'loss_dropdown' in ui_components:
            config['hyperparameters']['loss']['type'] = ui_components['loss_dropdown'].value
            
        if 'alpha_slider' in ui_components:
            config['hyperparameters']['loss']['alpha'] = ui_components['alpha_slider'].value
            
        if 'gamma_slider' in ui_components:
            config['hyperparameters']['loss']['gamma'] = ui_components['gamma_slider'].value
            
        if 'label_smoothing_slider' in ui_components:
            config['hyperparameters']['loss']['label_smoothing'] = ui_components['label_smoothing_slider'].value
            
        if 'box_loss_gain_slider' in ui_components:
            config['hyperparameters']['loss']['box_loss_gain'] = ui_components['box_loss_gain_slider'].value
            
        if 'cls_loss_gain_slider' in ui_components:
            config['hyperparameters']['loss']['cls_loss_gain'] = ui_components['cls_loss_gain_slider'].value
            
        if 'obj_loss_gain_slider' in ui_components:
            config['hyperparameters']['loss']['obj_loss_gain'] = ui_components['obj_loss_gain_slider'].value
            
        # Augmentation
        if 'augmentation_checkbox' in ui_components:
            config['hyperparameters']['augmentation']['enabled'] = ui_components['augmentation_checkbox'].value
            
        if 'mosaic_checkbox' in ui_components:
            config['hyperparameters']['augmentation']['mosaic'] = ui_components['mosaic_checkbox'].value
            
        if 'mixup_checkbox' in ui_components:
            config['hyperparameters']['augmentation']['mixup'] = ui_components['mixup_checkbox'].value
            
        if 'hsv_h_slider' in ui_components:
            config['hyperparameters']['augmentation']['hsv_h'] = ui_components['hsv_h_slider'].value
            
        if 'hsv_s_slider' in ui_components:
            config['hyperparameters']['augmentation']['hsv_s'] = ui_components['hsv_s_slider'].value
            
        if 'hsv_v_slider' in ui_components:
            config['hyperparameters']['augmentation']['hsv_v'] = ui_components['hsv_v_slider'].value
            
        if 'degrees_slider' in ui_components:
            config['hyperparameters']['augmentation']['degrees'] = ui_components['degrees_slider'].value
            
        if 'translate_slider' in ui_components:
            config['hyperparameters']['augmentation']['translate'] = ui_components['translate_slider'].value
            
        if 'scale_slider' in ui_components:
            config['hyperparameters']['augmentation']['scale'] = ui_components['scale_slider'].value
            
        if 'shear_slider' in ui_components:
            config['hyperparameters']['augmentation']['shear'] = ui_components['shear_slider'].value
            
        if 'perspective_slider' in ui_components:
            config['hyperparameters']['augmentation']['perspective'] = ui_components['perspective_slider'].value
            
        if 'flipud_slider' in ui_components:
            config['hyperparameters']['augmentation']['flipud'] = ui_components['flipud_slider'].value
            
        if 'fliplr_slider' in ui_components:
            config['hyperparameters']['augmentation']['fliplr'] = ui_components['fliplr_slider'].value
            
        if 'mosaic_prob_slider' in ui_components:
            config['hyperparameters']['augmentation']['mosaic_prob'] = ui_components['mosaic_prob_slider'].value
            
        if 'mixup_prob_slider' in ui_components:
            config['hyperparameters']['augmentation']['mixup_prob'] = ui_components['mixup_prob_slider'].value
            
        # Save config
        config_manager.save_module_config('hyperparameters', config)
        
        logger.info("✅ Konfigurasi hyperparameters berhasil diupdate")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update konfigurasi hyperparameters: {str(e)}")
        return get_default_hyperparameters_config()

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
            config = get_hyperparameters_config(ui_components)
            
        # Update UI components
        if 'enabled_checkbox' in ui_components:
            ui_components['enabled_checkbox'].value = config['hyperparameters']['enabled']
            
        # Optimizer
        if 'optimizer_dropdown' in ui_components:
            ui_components['optimizer_dropdown'].value = config['hyperparameters']['optimizer']['type']
            
        if 'learning_rate_slider' in ui_components:
            ui_components['learning_rate_slider'].value = config['hyperparameters']['optimizer']['learning_rate']
            
        if 'weight_decay_slider' in ui_components:
            ui_components['weight_decay_slider'].value = config['hyperparameters']['optimizer']['weight_decay']
            
        if 'momentum_slider' in ui_components:
            ui_components['momentum_slider'].value = config['hyperparameters']['optimizer']['momentum']
            
        if 'beta1_slider' in ui_components:
            ui_components['beta1_slider'].value = config['hyperparameters']['optimizer']['beta1']
            
        if 'beta2_slider' in ui_components:
            ui_components['beta2_slider'].value = config['hyperparameters']['optimizer']['beta2']
            
        if 'eps_slider' in ui_components:
            ui_components['eps_slider'].value = config['hyperparameters']['optimizer']['eps']
            
        # Scheduler
        if 'scheduler_checkbox' in ui_components:
            ui_components['scheduler_checkbox'].value = config['hyperparameters']['scheduler']['enabled']
            
        if 'scheduler_dropdown' in ui_components:
            ui_components['scheduler_dropdown'].value = config['hyperparameters']['scheduler']['type']
            
        if 'warmup_epochs_slider' in ui_components:
            ui_components['warmup_epochs_slider'].value = config['hyperparameters']['scheduler']['warmup_epochs']
            
        if 'min_lr_slider' in ui_components:
            ui_components['min_lr_slider'].value = config['hyperparameters']['scheduler']['min_lr']
            
        if 'patience_slider' in ui_components:
            ui_components['patience_slider'].value = config['hyperparameters']['scheduler']['patience']
            
        if 'factor_slider' in ui_components:
            ui_components['factor_slider'].value = config['hyperparameters']['scheduler']['factor']
            
        if 'threshold_slider' in ui_components:
            ui_components['threshold_slider'].value = config['hyperparameters']['scheduler']['threshold']
            
        # Loss
        if 'loss_dropdown' in ui_components:
            ui_components['loss_dropdown'].value = config['hyperparameters']['loss']['type']
            
        if 'alpha_slider' in ui_components:
            ui_components['alpha_slider'].value = config['hyperparameters']['loss']['alpha']
            
        if 'gamma_slider' in ui_components:
            ui_components['gamma_slider'].value = config['hyperparameters']['loss']['gamma']
            
        if 'label_smoothing_slider' in ui_components:
            ui_components['label_smoothing_slider'].value = config['hyperparameters']['loss']['label_smoothing']
            
        if 'box_loss_gain_slider' in ui_components:
            ui_components['box_loss_gain_slider'].value = config['hyperparameters']['loss']['box_loss_gain']
            
        if 'cls_loss_gain_slider' in ui_components:
            ui_components['cls_loss_gain_slider'].value = config['hyperparameters']['loss']['cls_loss_gain']
            
        if 'obj_loss_gain_slider' in ui_components:
            ui_components['obj_loss_gain_slider'].value = config['hyperparameters']['loss']['obj_loss_gain']
            
        # Augmentation
        if 'augmentation_checkbox' in ui_components:
            ui_components['augmentation_checkbox'].value = config['hyperparameters']['augmentation']['enabled']
            
        if 'mosaic_checkbox' in ui_components:
            ui_components['mosaic_checkbox'].value = config['hyperparameters']['augmentation']['mosaic']
            
        if 'mixup_checkbox' in ui_components:
            ui_components['mixup_checkbox'].value = config['hyperparameters']['augmentation']['mixup']
            
        if 'hsv_h_slider' in ui_components:
            ui_components['hsv_h_slider'].value = config['hyperparameters']['augmentation']['hsv_h']
            
        if 'hsv_s_slider' in ui_components:
            ui_components['hsv_s_slider'].value = config['hyperparameters']['augmentation']['hsv_s']
            
        if 'hsv_v_slider' in ui_components:
            ui_components['hsv_v_slider'].value = config['hyperparameters']['augmentation']['hsv_v']
            
        if 'degrees_slider' in ui_components:
            ui_components['degrees_slider'].value = config['hyperparameters']['augmentation']['degrees']
            
        if 'translate_slider' in ui_components:
            ui_components['translate_slider'].value = config['hyperparameters']['augmentation']['translate']
            
        if 'scale_slider' in ui_components:
            ui_components['scale_slider'].value = config['hyperparameters']['augmentation']['scale']
            
        if 'shear_slider' in ui_components:
            ui_components['shear_slider'].value = config['hyperparameters']['augmentation']['shear']
            
        if 'perspective_slider' in ui_components:
            ui_components['perspective_slider'].value = config['hyperparameters']['augmentation']['perspective']
            
        if 'flipud_slider' in ui_components:
            ui_components['flipud_slider'].value = config['hyperparameters']['augmentation']['flipud']
            
        if 'fliplr_slider' in ui_components:
            ui_components['fliplr_slider'].value = config['hyperparameters']['augmentation']['fliplr']
            
        if 'mosaic_prob_slider' in ui_components:
            ui_components['mosaic_prob_slider'].value = config['hyperparameters']['augmentation']['mosaic_prob']
            
        if 'mixup_prob_slider' in ui_components:
            ui_components['mixup_prob_slider'].value = config['hyperparameters']['augmentation']['mixup_prob']
            
        logger.info("✅ UI berhasil diupdate dari konfigurasi hyperparameters")
        
    except Exception as e:
        logger.error(f"❌ Error saat mengupdate UI dari konfigurasi: {str(e)}")
