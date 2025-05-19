"""
File: smartcash/ui/training_config/training_strategy/handlers/config_handlers.py
Deskripsi: Handler konfigurasi untuk training strategy
"""

from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

def get_default_training_strategy_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk training strategy.
    
    Returns:
        Dictionary konfigurasi default
    """
    return {
        "training_strategy": {
            "enabled": True,
            "batch_size": 16,
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": {
                "type": "adam",
                "weight_decay": 0.0005,
                "momentum": 0.9
            },
            "scheduler": {
                "enabled": True,
                "type": "cosine",
                "warmup_epochs": 5,
                "min_lr": 0.00001
            },
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "min_delta": 0.001
            },
            "checkpoint": {
                "enabled": True,
                "save_best_only": True,
                "save_freq": 1
            }
        }
    }

def get_training_strategy_config(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi training strategy dari config manager.
    
    Args:
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Dictionary konfigurasi training strategy
    """
    try:
        # Get config manager
        config_manager = get_config_manager()
        
        # Get config
        config = config_manager.get_module_config('training_strategy')
        
        if config:
            return config
            
        # Jika tidak ada config, gunakan default
        logger.warning("⚠️ Konfigurasi training strategy tidak ditemukan, menggunakan default")
        return get_default_training_strategy_config()
        
    except Exception as e:
        logger.error(f"❌ Error saat mengambil konfigurasi training strategy: {str(e)}")
        return get_default_training_strategy_config()

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi training strategy dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        # Get config manager
        config_manager = get_config_manager()
        
        # Get current config
        config = config_manager.get_module_config('training_strategy') or get_default_training_strategy_config()
        
        # Update config from UI
        if 'enabled_checkbox' in ui_components:
            config['training_strategy']['enabled'] = ui_components['enabled_checkbox'].value
            
        if 'batch_size_slider' in ui_components:
            config['training_strategy']['batch_size'] = ui_components['batch_size_slider'].value
            
        if 'epochs_slider' in ui_components:
            config['training_strategy']['epochs'] = ui_components['epochs_slider'].value
            
        if 'learning_rate_slider' in ui_components:
            config['training_strategy']['learning_rate'] = ui_components['learning_rate_slider'].value
            
        if 'optimizer_dropdown' in ui_components:
            config['training_strategy']['optimizer']['type'] = ui_components['optimizer_dropdown'].value
            
        if 'weight_decay_slider' in ui_components:
            config['training_strategy']['optimizer']['weight_decay'] = ui_components['weight_decay_slider'].value
            
        if 'momentum_slider' in ui_components:
            config['training_strategy']['optimizer']['momentum'] = ui_components['momentum_slider'].value
            
        if 'scheduler_checkbox' in ui_components:
            config['training_strategy']['scheduler']['enabled'] = ui_components['scheduler_checkbox'].value
            
        if 'scheduler_dropdown' in ui_components:
            config['training_strategy']['scheduler']['type'] = ui_components['scheduler_dropdown'].value
            
        if 'warmup_epochs_slider' in ui_components:
            config['training_strategy']['scheduler']['warmup_epochs'] = ui_components['warmup_epochs_slider'].value
            
        if 'min_lr_slider' in ui_components:
            config['training_strategy']['scheduler']['min_lr'] = ui_components['min_lr_slider'].value
            
        if 'early_stopping_checkbox' in ui_components:
            config['training_strategy']['early_stopping']['enabled'] = ui_components['early_stopping_checkbox'].value
            
        if 'patience_slider' in ui_components:
            config['training_strategy']['early_stopping']['patience'] = ui_components['patience_slider'].value
            
        if 'min_delta_slider' in ui_components:
            config['training_strategy']['early_stopping']['min_delta'] = ui_components['min_delta_slider'].value
            
        if 'checkpoint_checkbox' in ui_components:
            config['training_strategy']['checkpoint']['enabled'] = ui_components['checkpoint_checkbox'].value
            
        if 'save_best_only_checkbox' in ui_components:
            config['training_strategy']['checkpoint']['save_best_only'] = ui_components['save_best_only_checkbox'].value
            
        if 'save_freq_slider' in ui_components:
            config['training_strategy']['checkpoint']['save_freq'] = ui_components['save_freq_slider'].value
            
        # Save config
        config_manager.save_module_config('training_strategy', config)
        
        logger.info("✅ Konfigurasi training strategy berhasil diupdate")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update konfigurasi training strategy: {str(e)}")
        return get_default_training_strategy_config()

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
            config = get_training_strategy_config(ui_components)
            
        # Update UI components
        if 'enabled_checkbox' in ui_components:
            ui_components['enabled_checkbox'].value = config['training_strategy']['enabled']
            
        if 'batch_size_slider' in ui_components:
            ui_components['batch_size_slider'].value = config['training_strategy']['batch_size']
            
        if 'epochs_slider' in ui_components:
            ui_components['epochs_slider'].value = config['training_strategy']['epochs']
            
        if 'learning_rate_slider' in ui_components:
            ui_components['learning_rate_slider'].value = config['training_strategy']['learning_rate']
            
        if 'optimizer_dropdown' in ui_components:
            ui_components['optimizer_dropdown'].value = config['training_strategy']['optimizer']['type']
            
        if 'weight_decay_slider' in ui_components:
            ui_components['weight_decay_slider'].value = config['training_strategy']['optimizer']['weight_decay']
            
        if 'momentum_slider' in ui_components:
            ui_components['momentum_slider'].value = config['training_strategy']['optimizer']['momentum']
            
        if 'scheduler_checkbox' in ui_components:
            ui_components['scheduler_checkbox'].value = config['training_strategy']['scheduler']['enabled']
            
        if 'scheduler_dropdown' in ui_components:
            ui_components['scheduler_dropdown'].value = config['training_strategy']['scheduler']['type']
            
        if 'warmup_epochs_slider' in ui_components:
            ui_components['warmup_epochs_slider'].value = config['training_strategy']['scheduler']['warmup_epochs']
            
        if 'min_lr_slider' in ui_components:
            ui_components['min_lr_slider'].value = config['training_strategy']['scheduler']['min_lr']
            
        if 'early_stopping_checkbox' in ui_components:
            ui_components['early_stopping_checkbox'].value = config['training_strategy']['early_stopping']['enabled']
            
        if 'patience_slider' in ui_components:
            ui_components['patience_slider'].value = config['training_strategy']['early_stopping']['patience']
            
        if 'min_delta_slider' in ui_components:
            ui_components['min_delta_slider'].value = config['training_strategy']['early_stopping']['min_delta']
            
        if 'checkpoint_checkbox' in ui_components:
            ui_components['checkpoint_checkbox'].value = config['training_strategy']['checkpoint']['enabled']
            
        if 'save_best_only_checkbox' in ui_components:
            ui_components['save_best_only_checkbox'].value = config['training_strategy']['checkpoint']['save_best_only']
            
        if 'save_freq_slider' in ui_components:
            ui_components['save_freq_slider'].value = config['training_strategy']['checkpoint']['save_freq']
            
        logger.info("✅ UI berhasil diupdate dari konfigurasi training strategy")
        
    except Exception as e:
        logger.error(f"❌ Error saat mengupdate UI dari konfigurasi: {str(e)}")
