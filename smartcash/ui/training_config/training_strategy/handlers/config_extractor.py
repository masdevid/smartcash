"""
File: smartcash/ui/training_config/training_strategy/handlers/config_extractor.py
Deskripsi: Fungsi untuk mengekstrak konfigurasi dari UI components
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

from smartcash.ui.training_config.training_strategy.handlers.default_config import (
    get_default_training_strategy_config
)
from smartcash.ui.training_config.training_strategy.handlers.config_loader import (
    get_default_base_dir,
    save_training_strategy_config
)

logger = get_logger(__name__)

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi training strategy dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        # Dapatkan konfigurasi yang sudah ada atau gunakan default
        from smartcash.common.config import get_config_manager
        from smartcash.ui.training_config.training_strategy.handlers.config_loader import get_training_strategy_config
        
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = get_training_strategy_config(ui_components)
        
        logger.info(f"{ICONS.get('info', 'ℹ️')} Mengupdate konfigurasi dari UI components")
        
        # Helper untuk update value jika key ada
        def safe_get(key, default_value):
            if key in ui_components:
                try:
                    return ui_components[key].value
                except Exception as e:
                    logger.warning(f"Error saat mengambil nilai '{key}': {str(e)}")
                    return default_value
            else:
                logger.warning(f"Key '{key}' tidak ditemukan di ui_components, menggunakan nilai default.")
                return default_value
        
        # Referensi ke training_strategy untuk kemudahan akses
        ts_config = config['training_strategy']
        
        # Pastikan semua struktur nested dictionary ada
        if 'utils' not in ts_config:
            ts_config['utils'] = {}
        if 'validation' not in ts_config:
            ts_config['validation'] = {}
        if 'multiscale' not in ts_config:
            ts_config['multiscale'] = {}
            
        # Update parameter utama
        ts_config['enabled'] = safe_get('enabled_checkbox', ts_config.get('enabled', True))
        ts_config['batch_size'] = safe_get('batch_size_slider', ts_config.get('batch_size', 16))
        ts_config['epochs'] = safe_get('epochs_slider', ts_config.get('epochs', 100))
        ts_config['learning_rate'] = safe_get('learning_rate_slider', ts_config.get('learning_rate', 0.001))
        
        # Update optimizer
        if 'optimizer' not in ts_config:
            ts_config['optimizer'] = {}
        ts_config['optimizer']['type'] = safe_get('optimizer_dropdown', ts_config['optimizer'].get('type', 'adam'))
        ts_config['optimizer']['weight_decay'] = safe_get('weight_decay_slider', ts_config['optimizer'].get('weight_decay', 0.0005))
        ts_config['optimizer']['momentum'] = safe_get('momentum_slider', ts_config['optimizer'].get('momentum', 0.9))
        
        # Update scheduler
        if 'scheduler' not in ts_config:
            ts_config['scheduler'] = {}
        ts_config['scheduler']['enabled'] = safe_get('scheduler_checkbox', ts_config['scheduler'].get('enabled', True))
        ts_config['scheduler']['type'] = safe_get('scheduler_dropdown', ts_config['scheduler'].get('type', 'cosine'))
        ts_config['scheduler']['warmup_epochs'] = safe_get('warmup_epochs_slider', ts_config['scheduler'].get('warmup_epochs', 5))
        ts_config['scheduler']['min_lr'] = safe_get('min_lr_slider', ts_config['scheduler'].get('min_lr', 0.00001))
        
        # Update early stopping
        if 'early_stopping' not in ts_config:
            ts_config['early_stopping'] = {}
        ts_config['early_stopping']['enabled'] = safe_get('early_stopping_checkbox', ts_config['early_stopping'].get('enabled', True))
        ts_config['early_stopping']['patience'] = safe_get('patience_slider', ts_config['early_stopping'].get('patience', 10))
        ts_config['early_stopping']['min_delta'] = safe_get('min_delta_slider', ts_config['early_stopping'].get('min_delta', 0.001))
        
        # Update checkpoint
        if 'checkpoint' not in ts_config:
            ts_config['checkpoint'] = {}
        ts_config['checkpoint']['enabled'] = safe_get('checkpoint_checkbox', ts_config['checkpoint'].get('enabled', True))
        ts_config['checkpoint']['save_best_only'] = safe_get('save_best_only_checkbox', ts_config['checkpoint'].get('save_best_only', True))
        ts_config['checkpoint']['save_freq'] = safe_get('save_freq_slider', ts_config['checkpoint'].get('save_freq', 1))
        
        # Update utils
        ts_config['utils']['experiment_name'] = safe_get('experiment_name', ts_config['utils'].get('experiment_name', 'efficientnet_b4_training'))
        ts_config['utils']['checkpoint_dir'] = safe_get('checkpoint_dir', ts_config['utils'].get('checkpoint_dir', '/content/runs/train/checkpoints'))
        ts_config['utils']['tensorboard'] = safe_get('tensorboard', ts_config['utils'].get('tensorboard', True))
        ts_config['utils']['log_metrics_every'] = safe_get('log_metrics_every', ts_config['utils'].get('log_metrics_every', 10))
        ts_config['utils']['visualize_batch_every'] = safe_get('visualize_batch_every', ts_config['utils'].get('visualize_batch_every', 100))
        ts_config['utils']['gradient_clipping'] = safe_get('gradient_clipping', ts_config['utils'].get('gradient_clipping', 1.0))
        ts_config['utils']['mixed_precision'] = safe_get('mixed_precision', ts_config['utils'].get('mixed_precision', True))
        ts_config['utils']['layer_mode'] = safe_get('layer_mode', ts_config['utils'].get('layer_mode', 'single'))
        
        # Update validation
        ts_config['validation']['validation_frequency'] = safe_get('validation_frequency', ts_config['validation'].get('validation_frequency', 1))
        ts_config['validation']['iou_threshold'] = safe_get('iou_threshold', ts_config['validation'].get('iou_threshold', 0.6))
        ts_config['validation']['conf_threshold'] = safe_get('conf_threshold', ts_config['validation'].get('conf_threshold', 0.001))
        
        # Update multiscale
        ts_config['multiscale']['enabled'] = safe_get('multi_scale', ts_config['multiscale'].get('enabled', True))
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi training strategy berhasil diupdate")
        
        return config
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat update konfigurasi training strategy: {str(e)}")
        return get_default_training_strategy_config()