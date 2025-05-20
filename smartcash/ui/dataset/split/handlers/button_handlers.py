"""
File: smartcash/ui/dataset/split/handlers/button_handlers.py
Deskripsi: Handler untuk button di split dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.common.logger import get_logger
from smartcash.common.constants.log_messages import OPERATION_FAILED
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert
from smartcash.ui.dataset.split.handlers.config_handlers import (
    load_config, save_config, update_ui_from_config, update_config_from_ui, is_colab_environment
)
from smartcash.ui.dataset.split.handlers.sync_logger import (
    log_sync_success, log_sync_error, log_sync_warning, update_sync_status_only, add_sync_status_panel
)
from smartcash.ui.dataset.split.handlers.status_handlers import add_status_panel, update_status_panel
from smartcash.ui.dataset.split.handlers.save_handlers import create_save_handler
from smartcash.ui.dataset.split.handlers.reset_handlers import create_reset_handler

logger = get_logger(__name__)

def handle_split_button_click(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle click event untuk split button.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Update status panel
        update_sync_status_only(ui_components, "Memproses split dataset...", 'info')
        
        # Get current config
        config = load_config()
        
        # Simpan nilai UI sebelum diupdate
        if 'enabled_checkbox' in ui_components:
            enabled_before = ui_components['enabled_checkbox'].value
        if 'train_ratio_slider' in ui_components:
            train_ratio_before = ui_components['train_ratio_slider'].value
        if 'val_ratio_slider' in ui_components:
            val_ratio_before = ui_components['val_ratio_slider'].value
        if 'test_ratio_slider' in ui_components:
            test_ratio_before = ui_components['test_ratio_slider'].value
        
        # Update UI from config
        update_ui_from_config(ui_components, config)
        
        # Verifikasi bahwa nilai UI setelah diupdate sesuai dengan nilai config
        is_consistent = True
        if 'enabled_checkbox' in ui_components and 'split' in config:
            if ui_components['enabled_checkbox'].value != config['split']['enabled']:
                is_consistent = False
                log_sync_warning(ui_components, "Nilai enabled tidak konsisten setelah update")
        if 'train_ratio_slider' in ui_components and 'split' in config:
            if ui_components['train_ratio_slider'].value != config['split']['train_ratio']:
                is_consistent = False
                log_sync_warning(ui_components, "Nilai train_ratio tidak konsisten setelah update")
        
        # Show success message if consistent
        if is_consistent:
            update_sync_status_only(ui_components, "Dataset berhasil di-split", 'success')
            logger.info(f"{ICONS.get('success', '✅')} Dataset berhasil di-split")
        else:
            update_sync_status_only(ui_components, "Dataset berhasil di-split, namun nilai UI tidak konsisten dengan config", 'warning')
            logger.warning(f"{ICONS.get('warning', '⚠️')} Dataset berhasil di-split, namun nilai UI tidak konsisten dengan config")
        
        logger.info(f"{ICONS.get('success', '✅')} Split button berhasil dihandle")
        return ui_components
        
    except Exception as e:
        log_sync_error(ui_components, f"Error saat split dataset: {str(e)}")
        logger.error(f"{ICONS.get('error', '❌')} Error saat handle split button: {str(e)}")
        return ui_components

def handle_reset_button_click(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler untuk reset button split dataset (untuk keperluan unit test).
    Args:
        ui_components: Dictionary komponen UI
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Update status panel
        update_sync_status_only(ui_components, "Mereset konfigurasi ke default...", 'info')
        
        # Load default config
        config = load_config()
        
        # Simpan nilai sebelum reset untuk verifikasi
        pre_reset_values = {}
        if 'enabled_checkbox' in ui_components:
            pre_reset_values['enabled'] = ui_components['enabled_checkbox'].value
        if 'train_ratio_slider' in ui_components:
            pre_reset_values['train_ratio'] = ui_components['train_ratio_slider'].value
        if 'val_ratio_slider' in ui_components:
            pre_reset_values['val_ratio'] = ui_components['val_ratio_slider'].value
        if 'test_ratio_slider' in ui_components:
            pre_reset_values['test_ratio'] = ui_components['test_ratio_slider'].value
        if 'random_seed_input' in ui_components:
            pre_reset_values['random_seed'] = ui_components['random_seed_input'].value
        if 'stratify_checkbox' in ui_components:
            pre_reset_values['stratify'] = ui_components['stratify_checkbox'].value
        
        # Update UI
        update_ui_from_config(ui_components, config)
        
        # Save config
        saved_config = save_config(config, ui_components)
        
        # Verifikasi nilai UI setelah reset
        is_consistent = True
        if 'enabled_checkbox' in ui_components and 'split' in saved_config:
            if ui_components['enabled_checkbox'].value != saved_config['split']['enabled']:
                is_consistent = False
                logger.warning(f"⚠️ Nilai enabled tidak konsisten setelah reset: " 
                             f"{ui_components['enabled_checkbox'].value} vs {saved_config['split']['enabled']}")
        if 'train_ratio_slider' in ui_components and 'split' in saved_config:
            if ui_components['train_ratio_slider'].value != saved_config['split']['train_ratio']:
                is_consistent = False
                logger.warning(f"⚠️ Nilai train_ratio tidak konsisten setelah reset: " 
                             f"{ui_components['train_ratio_slider'].value} vs {saved_config['split']['train_ratio']}")
        
        # Log hasil verifikasi
        if is_consistent:
            log_sync_success(ui_components, "Konfigurasi berhasil direset ke default (unit test)")
            logger.info(f"{ICONS.get('success', '✅')} Konfigurasi berhasil direset ke default (unit test)")
        else:
            log_sync_warning(ui_components, "Konfigurasi direset tapi nilai tidak konsisten (unit test)")
            logger.warning(f"⚠️ Konfigurasi direset tapi nilai tidak konsisten (unit test)")
        
        return ui_components
    except Exception as e:
        log_sync_error(ui_components, f"Error saat reset konfigurasi (unit test): {str(e)}")
        logger.error(f"{ICONS.get('error', '❌')} Error saat reset konfigurasi (unit test): {str(e)}")
        return ui_components

def setup_button_handlers(ui_components: Dict[str, Any], config: Dict[str, Any] = None, env: Any = None) -> Dict[str, Any]:
    """
    Setup handler untuk button di split dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi untuk dataset
        env: Environment manager
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Pastikan ada panel status
        ui_components = add_status_panel(ui_components)
        
        # Buat handler untuk tombol
        save_handler = create_save_handler(ui_components)
        reset_handler = create_reset_handler(ui_components)
        
        # Bind handler ke button
        if 'reset_button' in ui_components:
            ui_components['reset_button'].on_click(reset_handler)
            
        if 'save_button' in ui_components:
            ui_components['save_button'].on_click(save_handler)
        
        # Tampilkan informasi lingkungan
        is_colab = is_colab_environment()
        if is_colab:
            logger.info("🔄 Sinkronisasi dengan Google Drive aktif")
        else:
            logger.info("ℹ️ Berjalan di lingkungan lokal (tanpa sinkronisasi Drive)")
        
        return ui_components
    except Exception as e:
        logger.error(OPERATION_FAILED.format(operation="Setup button handlers", reason=str(e)))
        return ui_components
