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
        # Helper untuk update value jika key ada
        def safe_set(key, value):
            if key in ui_components:
                ui_components[key].value = value
            else:
                logger.warning(f"Key '{key}' tidak ditemukan di ui_components, skip update komponen.")
        # Update UI components
        safe_set('enabled_checkbox', config['training_strategy']['enabled'])
        safe_set('batch_size_slider', config['training_strategy']['batch_size'])
        safe_set('epochs_slider', config['training_strategy']['epochs'])
        safe_set('learning_rate_slider', config['training_strategy']['learning_rate'])
        safe_set('optimizer_dropdown', config['training_strategy']['optimizer']['type'])
        safe_set('weight_decay_slider', config['training_strategy']['optimizer']['weight_decay'])
        safe_set('momentum_slider', config['training_strategy']['optimizer']['momentum'])
        safe_set('scheduler_checkbox', config['training_strategy']['scheduler']['enabled'])
        safe_set('scheduler_dropdown', config['training_strategy']['scheduler']['type'])
        safe_set('warmup_epochs_slider', config['training_strategy']['scheduler']['warmup_epochs'])
        safe_set('min_lr_slider', config['training_strategy']['scheduler']['min_lr'])
        safe_set('early_stopping_checkbox', config['training_strategy']['early_stopping']['enabled'])
        safe_set('patience_slider', config['training_strategy']['early_stopping']['patience'])
        safe_set('min_delta_slider', config['training_strategy']['early_stopping']['min_delta'])
        safe_set('checkpoint_checkbox', config['training_strategy']['checkpoint']['enabled'])
        safe_set('save_best_only_checkbox', config['training_strategy']['checkpoint']['save_best_only'])
        safe_set('save_freq_slider', config['training_strategy']['checkpoint']['save_freq'])
        logger.info("✅ UI berhasil diupdate dari konfigurasi training strategy")
        
        # Update info panel
        update_training_strategy_info(ui_components)
    except Exception as e:
        logger.error(f"❌ Error saat mengupdate UI dari konfigurasi: {str(e)}")

def update_training_strategy_info(ui_components: Optional[Dict[str, Any]] = None) -> None:
    """
    Update informasi strategi pelatihan di panel info.
    
    Args:
        ui_components: Komponen UI
    """
    info_panel = ui_components.get('training_strategy_info')
    if not info_panel:
        logger.warning(f"{ICONS.get('warning', '⚠️')} Info panel tidak ditemukan")
        return
        
    with info_panel:
        clear_output(wait=True)
        
        if not ui_components:
            display(widgets.HTML(
                f"<h3>{ICONS.get('info', 'ℹ️')} Informasi Strategi Pelatihan</h3>"
                f"<p>Tidak ada informasi yang tersedia.</p>"
            ))
            return
        
        try:
            # Dapatkan nilai dari komponen UI
            experiment_name = ui_components.get('experiment_name', widgets.Text()).value
            checkpoint_dir = ui_components.get('checkpoint_dir', widgets.Text()).value
            tensorboard = ui_components.get('tensorboard', widgets.Checkbox()).value
            log_metrics_every = ui_components.get('log_metrics_every', widgets.IntSlider()).value
            visualize_batch_every = ui_components.get('visualize_batch_every', widgets.IntSlider()).value
            gradient_clipping = ui_components.get('gradient_clipping', widgets.FloatSlider()).value
            mixed_precision = ui_components.get('mixed_precision', widgets.Checkbox()).value
            layer_mode = ui_components.get('layer_mode', widgets.RadioButtons()).value
            
            # Validasi
            validation_frequency = ui_components.get('validation_frequency', widgets.IntSlider()).value
            iou_threshold = ui_components.get('iou_threshold', widgets.FloatSlider()).value
            conf_threshold = ui_components.get('conf_threshold', widgets.FloatSlider()).value
            
            # Multi-scale
            multi_scale = ui_components.get('multi_scale', widgets.Checkbox()).value
            
            # Buat konten HTML
            html_content = f"""
            <h3>{ICONS.get('info', 'ℹ️')} Informasi Strategi Pelatihan</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 10px;">
                <div>
                    <h4>Parameter Utilitas</h4>
                    <ul>
                        <li><b>Experiment Name:</b> {experiment_name}</li>
                        <li><b>Checkpoint Dir:</b> {checkpoint_dir}</li>
                        <li><b>TensorBoard:</b> {'Aktif' if tensorboard else 'Nonaktif'}</li>
                        <li><b>Log Metrics Every:</b> {log_metrics_every} batch</li>
                        <li><b>Visualize Batch Every:</b> {visualize_batch_every} batch</li>
                        <li><b>Gradient Clipping:</b> {gradient_clipping}</li>
                        <li><b>Mixed Precision:</b> {'Aktif' if mixed_precision else 'Nonaktif'}</li>
                        <li><b>Layer Mode:</b> {layer_mode}</li>
                    </ul>
                </div>
                <div>
                    <h4>Validasi & Multi-scale</h4>
                    <ul>
                        <li><b>Validation Frequency:</b> {validation_frequency} epoch</li>
                        <li><b>IoU Threshold:</b> {iou_threshold}</li>
                        <li><b>Conf Threshold:</b> {conf_threshold}</li>
                        <li><b>Multi-scale Training:</b> {'Aktif' if multi_scale else 'Nonaktif'}</li>
                    </ul>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;">
                <p><b>{ICONS.get('info', 'ℹ️')} Catatan:</b> Konfigurasi strategi pelatihan ini akan digunakan untuk melatih model YOLOv5 dengan EfficientNet backbone.</p>
            </div>
            """
            
            display(widgets.HTML(html_content))
            
            # Sinkronkan dengan drive
            try:
                config_manager = get_config_manager(base_dir=get_default_base_dir())
                config_manager.sync_config_with_drive('training_strategy')
                logger.info("✅ Konfigurasi berhasil disinkronkan dengan drive")
            except Exception as sync_error:
                logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat sinkronisasi dengan drive: {str(sync_error)}")
            
        except Exception as e:
            logger.error(f"{ICONS.get('error', '❌')} Error update info panel: {str(e)}")
            display(widgets.HTML(
                f"<h3>{ICONS.get('error', '❌')} Error</h3>"
                f"<p>Terjadi error saat memperbarui informasi: {str(e)}</p>"
            ))
