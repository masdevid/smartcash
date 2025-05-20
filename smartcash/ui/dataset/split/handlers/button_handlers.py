"""
File: smartcash/ui/dataset/split/handlers/button_handlers.py
Deskripsi: Handler untuk button di split dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert
from smartcash.ui.dataset.split.handlers.config_handlers import load_config, save_config, update_ui_from_config
from smartcash.ui.dataset.split.handlers.sync_logger import log_sync_success, log_sync_error

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
        # Get current config
        config = load_config()
        
        # Update UI from config
        update_ui_from_config(ui_components, config)
        
        # Show success message
        display(create_info_alert(
            f"{ICONS.get('success', '✅')} Dataset berhasil di-split",
            alert_type='success'
        ))
        
        log_sync_success(ui_components, "Dataset berhasil di-split")
        logger.info(f"{ICONS.get('success', '✅')} Split button berhasil dihandle")
        return ui_components
        
    except Exception as e:
        log_sync_error(ui_components, f"Error saat split dataset: {str(e)}")
        logger.error(f"{ICONS.get('error', '❌')} Error saat handle split button: {str(e)}")
        display(create_info_alert(
            f"{ICONS.get('error', '❌')} Error saat split dataset: {str(e)}",
            alert_type='error'
        ))
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
        # Load default config
        config = load_config()
        # Update UI
        update_ui_from_config(ui_components, config)
        # Save config
        save_config(config)
        log_sync_success(ui_components, "Konfigurasi berhasil direset ke default (unit test)")
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi berhasil direset ke default (unit test)")
        return ui_components
    except Exception as e:
        log_sync_error(ui_components, f"Error saat reset konfigurasi (unit test): {str(e)}")
        logger.error(f"{ICONS.get('error', '❌')} Error saat reset konfigurasi (unit test): {str(e)}")
        return ui_components

def setup_button_handlers(ui_components: Dict[str, Any], env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Setup handler untuk button di split dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        config: Konfigurasi (opsional)
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Split button handler
        if 'split_button' in ui_components:
            ui_components['split_button'].on_click(lambda b: handle_split_button_click(ui_components))
            
        # Reset button handler
        if 'reset_button' in ui_components:
            def on_reset_clicked(b):
                try:
                    # Load default config
                    config = load_config()
                    
                    # Simpan config sebelum reset untuk verifikasi
                    pre_reset_config = config.copy() if config else None
                    
                    # Update UI
                    update_ui_from_config(ui_components, config)
                    
                    # Save config
                    save_config(config)
                    
                    # Verifikasi reset berhasil dengan memuat ulang config
                    post_reset_config = load_config()
                    
                    # Bandingkan nilai config sebelum dan sesudah reset
                    is_consistent = True
                    for key in post_reset_config['split']:
                        if pre_reset_config and key in pre_reset_config['split']:
                            # Skip perbandingan jika key tidak ada di salah satu config
                            if post_reset_config['split'][key] != pre_reset_config['split'][key]:
                                is_consistent = False
                                break
                            
                    if is_consistent:
                        # Log sync status
                        log_sync_success(ui_components, "Konfigurasi berhasil direset ke default")
                        
                        # Show success message
                        display(create_info_alert(
                            f"{ICONS.get('success', '✅')} Konfigurasi berhasil direset ke default",
                            alert_type='success'
                        ))
                    else:
                        log_sync_error(ui_components, "Reset konfigurasi tidak konsisten")
                        display(create_info_alert(
                            f"{ICONS.get('error', '❌')} Reset konfigurasi tidak konsisten",
                            alert_type='error'
                        ))
                        
                except Exception as e:
                    log_sync_error(ui_components, f"Error saat reset konfigurasi: {str(e)}")
                    logger.error(f"{ICONS.get('error', '❌')} Error saat reset konfigurasi: {str(e)}")
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Error saat reset konfigurasi: {str(e)}",
                        alert_type='error'
                    ))
            
            ui_components['reset_button'].on_click(on_reset_clicked)
            
        # Save button handler
        if 'save_button' in ui_components:
            def on_save_clicked(b):
                try:
                    # Get current config dan simpan nilai UI
                    ui_values = {
                        'enabled': ui_components['enabled_checkbox'].value if 'enabled_checkbox' in ui_components else True,
                        'train_ratio': ui_components['train_ratio_slider'].value if 'train_ratio_slider' in ui_components else 0.7,
                        'val_ratio': ui_components['val_ratio_slider'].value if 'val_ratio_slider' in ui_components else 0.15,
                        'test_ratio': ui_components['test_ratio_slider'].value if 'test_ratio_slider' in ui_components else 0.15,
                        'random_seed': ui_components['random_seed_input'].value if 'random_seed_input' in ui_components else 42,
                        'stratify': ui_components['stratify_checkbox'].value if 'stratify_checkbox' in ui_components else True
                    }
                    
                    # Get current config
                    config = load_config()
                    
                    # Update config from UI
                    if 'enabled_checkbox' in ui_components:
                        config['split']['enabled'] = ui_components['enabled_checkbox'].value
                        
                    if 'train_ratio_slider' in ui_components:
                        config['split']['train_ratio'] = ui_components['train_ratio_slider'].value
                        
                    if 'val_ratio_slider' in ui_components:
                        config['split']['val_ratio'] = ui_components['val_ratio_slider'].value
                        
                    if 'test_ratio_slider' in ui_components:
                        config['split']['test_ratio'] = ui_components['test_ratio_slider'].value
                        
                    if 'random_seed_input' in ui_components:
                        config['split']['random_seed'] = ui_components['random_seed_input'].value
                        
                    if 'stratify_checkbox' in ui_components:
                        config['split']['stratify'] = ui_components['stratify_checkbox'].value
                    
                    # Save config
                    save_config(config)
                    
                    # Verifikasi perubahan telah disimpan dengan benar
                    loaded_config = load_config()
                    
                    # Periksa apakah nilai yang disimpan sesuai dengan nilai di UI
                    is_consistent = (
                        ui_values['enabled'] == loaded_config['split']['enabled'] and
                        ui_values['train_ratio'] == loaded_config['split']['train_ratio'] and
                        ui_values['val_ratio'] == loaded_config['split']['val_ratio'] and
                        ui_values['test_ratio'] == loaded_config['split']['test_ratio'] and
                        ui_values['random_seed'] == loaded_config['split']['random_seed'] and
                        ui_values['stratify'] == loaded_config['split']['stratify']
                    )
                    
                    if is_consistent:
                        # Log sync status
                        log_sync_success(ui_components, "Konfigurasi berhasil disimpan")
                        
                        # Show success message
                        display(create_info_alert(
                            f"{ICONS.get('success', '✅')} Konfigurasi berhasil disimpan",
                            alert_type='success'
                        ))
                    else:
                        log_sync_error(ui_components, "Konfigurasi tidak disimpan dengan benar, ada perbedaan data")
                        display(create_info_alert(
                            f"{ICONS.get('error', '❌')} Konfigurasi tidak disimpan dengan benar",
                            alert_type='error'
                        ))
                        
                except Exception as e:
                    log_sync_error(ui_components, f"Error saat menyimpan konfigurasi: {str(e)}")
                    logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi: {str(e)}")
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi: {str(e)}",
                        alert_type='error'
                    ))
            
            ui_components['save_button'].on_click(on_save_clicked)
            
        return ui_components
        
    except Exception as e:
        log_sync_error(ui_components, f"Error saat setup button handlers: {str(e)}")
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup button handlers: {str(e)}")
        return ui_components
