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
from IPython.display import display, clear_output
import ipywidgets as widgets

logger = get_logger(__name__)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

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

# Alias for backward compatibility
get_default_config = get_default_training_strategy_config

def get_training_strategy_config(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi training strategy dari config manager.
    
    Args:
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Dictionary konfigurasi training strategy
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = config_manager.get_module_config('training_strategy')
        if config:
            return config
        logger.warning("⚠️ Konfigurasi training strategy tidak ditemukan, menggunakan default")
        return get_default_training_strategy_config()
    except Exception as e:
        logger.error(f"❌ Error saat mengambil konfigurasi training strategy: {str(e)}")
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
                # Simpan nilai lama untuk debugging
                old_value = ui_components[key].value
                # Set nilai baru
                ui_components[key].value = value
                # Log jika nilai tidak berubah
                if old_value == ui_components[key].value and old_value != value:
                    logger.warning(f"Nilai untuk '{key}' tidak berubah: {old_value} -> {ui_components[key].value}, seharusnya {value}")
            else:
                logger.warning(f"Key '{key}' tidak ditemukan di ui_components, skip update komponen.")
                
        # Update UI components dengan nilai dari config
        ts_config = config['training_strategy']
        
        # Update komponen UI satu per satu dengan nilai dari config
        safe_set('enabled_checkbox', ts_config['enabled'])
        safe_set('batch_size_slider', ts_config['batch_size'])
        
        # Khusus untuk epochs_slider, pastikan nilai diupdate dengan benar
        if 'epochs_slider' in ui_components:
            # Force update untuk epochs_slider
            ui_components['epochs_slider'].value = ts_config['epochs']
            
        safe_set('learning_rate_slider', ts_config['learning_rate'])
        safe_set('optimizer_dropdown', ts_config['optimizer']['type'])
        safe_set('weight_decay_slider', ts_config['optimizer']['weight_decay'])
        safe_set('momentum_slider', ts_config['optimizer']['momentum'])
        safe_set('scheduler_checkbox', ts_config['scheduler']['enabled'])
        safe_set('scheduler_dropdown', ts_config['scheduler']['type'])
        safe_set('warmup_epochs_slider', ts_config['scheduler']['warmup_epochs'])
        safe_set('min_lr_slider', ts_config['scheduler']['min_lr'])
        safe_set('early_stopping_checkbox', ts_config['early_stopping']['enabled'])
        safe_set('patience_slider', ts_config['early_stopping']['patience'])
        safe_set('min_delta_slider', ts_config['early_stopping']['min_delta'])
        safe_set('checkpoint_checkbox', ts_config['checkpoint']['enabled'])
        safe_set('save_best_only_checkbox', ts_config['checkpoint']['save_best_only'])
        safe_set('save_freq_slider', ts_config['checkpoint']['save_freq'])
        
        # Force update UI by triggering change events for boolean widgets
        for widget_name, widget in ui_components.items():
            if isinstance(widget, widgets.Checkbox):
                # Trigger a change event to update dependent widgets
                old_value = widget.value
                # Toggle boolean value to trigger change event
                widget.value = not old_value
                widget.value = old_value
                
        logger.info("✅ UI berhasil diupdate dari konfigurasi training strategy")
        
        # Update info panel
        update_training_strategy_info(ui_components)
        
    except Exception as e:
        logger.error(f"❌ Error saat mengupdate UI dari konfigurasi: {str(e)}")
        if 'info_panel' in ui_components:
            ui_components['info_panel'].value = f"Error: {str(e)}"

def update_training_strategy_info(ui_components: Dict[str, Any], message: str = None) -> None:
    """
    Update info panel dengan informasi training strategy yang dipilih.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan tambahan yang akan ditampilkan (opsional)
    """
    try:
        info_panel = ui_components.get('info_panel')
        if not info_panel:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Info panel tidak ditemukan")
            return
            
        # Get current config
        config = get_training_strategy_config(ui_components)
        
        # Ensure config has training_strategy key
        if 'training_strategy' not in config:
            config = {'training_strategy': config}
        
        # Get training_strategy config with safe defaults
        ts_config = config['training_strategy']
        default_config = get_default_training_strategy_config()['training_strategy']
        
        # Safely get values with defaults
        enabled = ts_config.get('enabled', default_config['enabled'])
        batch_size = ts_config.get('batch_size', default_config['batch_size'])
        epochs = ts_config.get('epochs', default_config['epochs'])
        learning_rate = ts_config.get('learning_rate', default_config['learning_rate'])
        
        # Safely get optimizer values
        optimizer = ts_config.get('optimizer', default_config['optimizer'])
        optimizer_type = optimizer.get('type', default_config['optimizer']['type'])
        weight_decay = optimizer.get('weight_decay', default_config['optimizer']['weight_decay'])
        momentum = optimizer.get('momentum', default_config['optimizer']['momentum'])
        
        # Safely get scheduler values
        scheduler = ts_config.get('scheduler', default_config['scheduler'])
        scheduler_enabled = scheduler.get('enabled', default_config['scheduler']['enabled'])
        scheduler_type = scheduler.get('type', default_config['scheduler']['type'])
        warmup_epochs = scheduler.get('warmup_epochs', default_config['scheduler']['warmup_epochs'])
        min_lr = scheduler.get('min_lr', default_config['scheduler']['min_lr'])
        
        # Safely get early stopping values
        early_stopping = ts_config.get('early_stopping', default_config['early_stopping'])
        early_stopping_enabled = early_stopping.get('enabled', default_config['early_stopping']['enabled'])
        patience = early_stopping.get('patience', default_config['early_stopping']['patience'])
        min_delta = early_stopping.get('min_delta', default_config['early_stopping']['min_delta'])
        
        # Safely get checkpoint values
        checkpoint = ts_config.get('checkpoint', default_config['checkpoint'])
        checkpoint_enabled = checkpoint.get('enabled', default_config['checkpoint']['enabled'])
        save_best_only = checkpoint.get('save_best_only', default_config['checkpoint']['save_best_only'])
        save_freq = checkpoint.get('save_freq', default_config['checkpoint']['save_freq'])
        
        # Update info panel dengan informasi training strategy
        info_text = f"""
        <div style='font-family: monospace;'>
        <h4>Training Strategy Configuration:</h4>
        <ul>
            <li>Enabled: {enabled}</li>
            <li>Batch Size: {batch_size}</li>
            <li>Epochs: {epochs}</li>
            <li>Learning Rate: {learning_rate}</li>
            <li>Optimizer: {optimizer_type}</li>
            <li>Weight Decay: {weight_decay}</li>
            <li>Momentum: {momentum}</li>
            <li>Scheduler Enabled: {scheduler_enabled}</li>
            <li>Scheduler Type: {scheduler_type}</li>
            <li>Warmup Epochs: {warmup_epochs}</li>
            <li>Min Learning Rate: {min_lr}</li>
            <li>Early Stopping Enabled: {early_stopping_enabled}</li>
            <li>Patience: {patience}</li>
            <li>Min Delta: {min_delta}</li>
            <li>Checkpoint Enabled: {checkpoint_enabled}</li>
            <li>Save Best Only: {save_best_only}</li>
            <li>Save Frequency: {save_freq}</li>
        </ul>
        """
        
        # Add message if provided
        if message:
            info_text += f"<p>{message}</p>"
            
        info_text += "</div>"
        
        info_panel.value = info_text
        
        logger.info("✅ Info panel berhasil diupdate")
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}")
        if info_panel:
            info_panel.value = f"{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}"

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi training strategy dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = config_manager.get_module_config('training_strategy') or get_default_training_strategy_config()
        
        # Pastikan config memiliki struktur yang benar
        if 'training_strategy' not in config:
            config = {'training_strategy': config}
        
        logger.info("Mengupdate konfigurasi dari UI components")
        
        # Helper untuk update value jika key ada
        def safe_get(key, default_value):
            if key in ui_components:
                return ui_components[key].value
            else:
                logger.warning(f"Key '{key}' tidak ditemukan di ui_components, menggunakan nilai default.")
                return default_value
        
        # Referensi ke training_strategy untuk kemudahan akses
        ts_config = config['training_strategy']
                
        # Update config from UI
        ts_config['enabled'] = safe_get('enabled_checkbox', ts_config['enabled'])
        ts_config['batch_size'] = safe_get('batch_size_slider', ts_config['batch_size'])
        
        # Khusus untuk epochs, pastikan nilai diambil dengan benar
        if 'epochs_slider' in ui_components:
            ts_config['epochs'] = ui_components['epochs_slider'].value
        
        ts_config['learning_rate'] = safe_get('learning_rate_slider', ts_config['learning_rate'])
        ts_config['optimizer']['type'] = safe_get('optimizer_dropdown', ts_config['optimizer']['type'])
        ts_config['optimizer']['weight_decay'] = safe_get('weight_decay_slider', ts_config['optimizer']['weight_decay'])
        ts_config['optimizer']['momentum'] = safe_get('momentum_slider', ts_config['optimizer']['momentum'])
        ts_config['scheduler']['enabled'] = safe_get('scheduler_checkbox', ts_config['scheduler']['enabled'])
        ts_config['scheduler']['type'] = safe_get('scheduler_dropdown', ts_config['scheduler']['type'])
        ts_config['scheduler']['warmup_epochs'] = safe_get('warmup_epochs_slider', ts_config['scheduler']['warmup_epochs'])
        ts_config['scheduler']['min_lr'] = safe_get('min_lr_slider', ts_config['scheduler']['min_lr'])
        ts_config['early_stopping']['enabled'] = safe_get('early_stopping_checkbox', ts_config['early_stopping']['enabled'])
        ts_config['early_stopping']['patience'] = safe_get('patience_slider', ts_config['early_stopping']['patience'])
        ts_config['early_stopping']['min_delta'] = safe_get('min_delta_slider', ts_config['early_stopping']['min_delta'])
        ts_config['checkpoint']['enabled'] = safe_get('checkpoint_checkbox', ts_config['checkpoint']['enabled'])
        ts_config['checkpoint']['save_best_only'] = safe_get('save_best_only_checkbox', ts_config['checkpoint']['save_best_only'])
        ts_config['checkpoint']['save_freq'] = safe_get('save_freq_slider', ts_config['checkpoint']['save_freq'])
        
        # Save config
        config_manager.save_module_config('training_strategy', config)
        
        logger.info("✅ Konfigurasi training strategy berhasil diupdate")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update konfigurasi training strategy: {str(e)}")
        return get_default_training_strategy_config()
