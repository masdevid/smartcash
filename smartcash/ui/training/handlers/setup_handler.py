"""
File: smartcash/ui/training/handlers/setup_handler.py
Deskripsi: Setup handler untuk komponen UI training
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert

def setup_training_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI training.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        # Dapatkan logger
        logger = ui_components.get('logger', None) or get_logger()
        
        # Dapatkan environment manager jika belum tersedia
        env = env or get_environment_manager()
        
        # Dapatkan ConfigManager
        config_manager = get_config_manager()
        
        # Log informasi setup
        logger.info(f"{ICONS.get('settings', '⚙️')} Memulai setup handler training...")
        
        # Daftarkan handler untuk tombol
        register_button_handlers(ui_components, env, config)
        
        # Update informasi training dari konfigurasi yang sudah ada
        update_training_info_from_config(ui_components, config_manager)
        
        # Pastikan persistensi UI components
        config_manager.register_ui_components('training', ui_components)
        
        logger.info(f"{ICONS.get('check', '✓')} Setup handler training berhasil")
        return ui_components
    
    except Exception as e:
        # Fallback jika terjadi error
        logger = get_logger()
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup training handlers: {str(e)}")
        
        # Tampilkan pesan error
        if 'status_panel' in ui_components:
            with ui_components['status_panel']:
                ui_components['status_panel'].clear_output()
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Error saat setup training handlers: {str(e)}",
                    alert_type='error'
                ))
        
        return ui_components

def register_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> None:
    """
    Mendaftarkan handler untuk tombol-tombol UI training.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
    """
    # Import handler untuk tombol
    from smartcash.ui.training.handlers.button_handlers import (
        on_start_training,
        on_stop_training,
        on_reset_training,
        on_cleanup_training,
        on_save_config,
        on_reset_config
    )
    
    # Daftarkan handler untuk tombol start
    if 'start_button' in ui_components:
        ui_components['start_button'].on_click(
            lambda b: on_start_training(b, ui_components, env, config)
        )
    
    # Daftarkan handler untuk tombol stop
    if 'stop_button' in ui_components:
        ui_components['stop_button'].on_click(
            lambda b: on_stop_training(b, ui_components, env, config)
        )
    
    # Daftarkan handler untuk tombol reset training (bukan reset config)
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(
            lambda b: on_reset_training(b, ui_components, env, config)
        )
    
    # Daftarkan handler untuk tombol cleanup
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].on_click(
            lambda b: on_cleanup_training(b, ui_components, env, config)
        )
        
    # Training UI tidak lagi menggunakan save_reset_buttons
    # Fokus pada menjalankan training dan menampilkan metrik realtime F1 dan mAP plot

def update_training_info_from_config(ui_components: Dict[str, Any], config_manager) -> None:
    """
    Memperbarui informasi training dari konfigurasi yang sudah ada.
    
    Args:
        ui_components: Komponen UI
        config_manager: Config manager
    """
    try:
        # Dapatkan konfigurasi dari modul-modul sebelumnya
        model_config = config_manager.get_module_config('model', {})
        hyperparameters_config = config_manager.get_module_config('hyperparameters', {})
        training_strategy_config = config_manager.get_module_config('training_strategy', {})
        
        # Update informasi di status panel
        if 'status_panel' in ui_components:
            with ui_components['status_panel']:
                ui_components['status_panel'].clear_output()
                display(HTML(f"""
                <div style="padding: 10px;">
                    <p><strong>{ICONS.get('info', 'ℹ️')} Informasi Konfigurasi Training</strong></p>
                    <ul>
                        <li><strong>Model:</strong> {model_config.get('backbone', 'EfficientNet-B4')}</li>
                        <li><strong>Learning Rate:</strong> {hyperparameters_config.get('learning_rate', 0.001)}</li>
                        <li><strong>Batch Size:</strong> {hyperparameters_config.get('batch_size', 16)}</li>
                        <li><strong>Epochs:</strong> {training_strategy_config.get('epochs', 100)}</li>
                    </ul>
                    <p>Konfigurasi di atas diambil dari modul-modul sebelumnya. Anda dapat mengubahnya di form konfigurasi training.</p>
                </div>
                """))
        
        # Update nilai di form konfigurasi jika ada
        if 'backbone_dropdown' in ui_components and 'backbone' in model_config:
            # Konversi nilai backbone ke format yang sesuai dengan dropdown
            backbone_value = model_config.get('backbone', 'efficientnet_b4')
            ui_components['backbone_dropdown'].value = backbone_value
        
        if 'epochs_input' in ui_components and 'epochs' in training_strategy_config:
            ui_components['epochs_input'].value = training_strategy_config.get('epochs', 100)
        
        if 'batch_size_input' in ui_components and 'batch_size' in hyperparameters_config:
            ui_components['batch_size_input'].value = hyperparameters_config.get('batch_size', 16)
        
        if 'learning_rate_input' in ui_components and 'learning_rate' in hyperparameters_config:
            ui_components['learning_rate_input'].value = hyperparameters_config.get('learning_rate', 0.001)
    
    except Exception as e:
        # Log error tapi jangan gagalkan inisialisasi UI
        logger = get_logger()
        logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat update training info: {str(e)}")

