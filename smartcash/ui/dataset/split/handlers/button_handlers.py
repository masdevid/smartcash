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
from smartcash.ui.dataset.split.handlers.config_handlers import (
    load_config, save_config, update_ui_from_config, update_config_from_ui, is_colab_environment
)
from smartcash.ui.dataset.split.handlers.sync_logger import (
    log_sync_success, log_sync_error, log_sync_warning, update_sync_status_only, add_sync_status_panel
)

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
            display(create_info_alert(
                f"{ICONS.get('success', '✅')} Dataset berhasil di-split",
                alert_type='success'
            ))
            log_sync_success(ui_components, "Dataset berhasil di-split")
        else:
            display(create_info_alert(
                f"{ICONS.get('warning', '⚠️')} Dataset berhasil di-split, namun nilai UI tidak konsisten dengan config",
                alert_type='warning'
            ))
            log_sync_warning(ui_components, "Dataset berhasil di-split, namun nilai UI tidak konsisten dengan config")
        
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
        # Import dependency
        from smartcash.ui.dataset.split.handlers.config_handlers import (
            load_config, save_config, update_ui_from_config, update_config_from_ui, is_colab_environment
        )
        from smartcash.ui.dataset.split.handlers.sync_logger import (
            update_sync_status_only, add_sync_status_panel
        )
        from smartcash.ui.utils.constants import ICONS
        from smartcash.ui.utils.alert_utils import create_info_alert
        
        # Pastikan ada panel status
        ui_components = add_sync_status_panel(ui_components)
        
        # Deteksi apakah berjalan di Colab
        is_colab = is_colab_environment()
        
        # Handler untuk tombol Reset
        def on_reset_clicked(b):
            try:
                # Update status
                update_sync_status_only(ui_components, "Memuat ulang konfigurasi...", 'info')
                
                # Load konfigurasi
                config = load_config()
                
                # Update UI dari konfigurasi
                update_ui_from_config(ui_components, config)
                
                # Update status
                update_sync_status_only(ui_components, "Konfigurasi berhasil dimuat ulang", 'success')
                
                # Tampilkan pesan sukses
                display(create_info_alert(
                    f"{ICONS.get('success', '✅')} Konfigurasi berhasil dimuat ulang",
                    alert_type='success'
                ))
            except Exception as e:
                # Update status
                update_sync_status_only(ui_components, f"Error saat reset: {str(e)}", 'error')
                
                # Tampilkan pesan error
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Error saat reset: {str(e)}",
                    alert_type='danger'
                ))
                logger.error(f"{ICONS.get('error', '❌')} Error saat reset: {str(e)}")
        
        # Handler untuk tombol Save
        def on_save_clicked(b):
            try:
                # Update status
                update_sync_status_only(ui_components, "Menyimpan konfigurasi...", 'info')
                
                # Dapatkan nilai dari UI
                ui_values = {
                    'enabled': ui_components['enabled_checkbox'].value,
                    'train_ratio': ui_components['train_ratio_slider'].value,
                    'val_ratio': ui_components['val_ratio_slider'].value,
                    'test_ratio': ui_components['test_ratio_slider'].value,
                    'random_seed': ui_components['random_seed_input'].value,
                    'stratify': ui_components['stratify_checkbox'].value
                }
                
                # Buat konfigurasi dari UI
                config = {
                    'split': ui_values
                }
                
                # Simpan konfigurasi
                saved_config = save_config(config, ui_components)
                
                # Verifikasi konfigurasi tersimpan dengan benar
                if saved_config and 'split' in saved_config:
                    # Muat ulang konfigurasi untuk verifikasi
                    loaded_config = load_config()
                    
                    # Periksa apakah data yang disimpan sesuai
                    if (ui_values['train_ratio'] == loaded_config['split']['train_ratio'] and
                        ui_values['val_ratio'] == loaded_config['split']['val_ratio'] and
                        ui_values['test_ratio'] == loaded_config['split']['test_ratio'] and
                        ui_values['random_seed'] == loaded_config['split']['random_seed'] and
                        ui_values['stratify'] == loaded_config['split']['stratify'] and
                        ui_values['enabled'] == loaded_config['split']['enabled']):
                        
                        # Pesan sukses yang berbeda berdasarkan lingkungan
                        if is_colab:
                            success_message = "Konfigurasi berhasil disimpan dan disinkronkan dengan Google Drive"
                        else:
                            success_message = "Konfigurasi berhasil disimpan"
                            
                        # Update status
                        update_sync_status_only(ui_components, success_message, 'success')
                        
                        # Tampilkan pesan sukses
                        display(create_info_alert(
                            f"{ICONS.get('success', '✅')} {success_message}",
                            alert_type='success'
                        ))
                    else:
                        # Coba deteksi key mana yang tidak konsisten
                        inconsistent_keys = []
                        if ui_values['enabled'] != loaded_config['split']['enabled']:
                            inconsistent_keys.append('enabled')
                        if ui_values['train_ratio'] != loaded_config['split']['train_ratio']:
                            inconsistent_keys.append('train_ratio')
                        if ui_values['val_ratio'] != loaded_config['split']['val_ratio']:
                            inconsistent_keys.append('val_ratio')
                        if ui_values['test_ratio'] != loaded_config['split']['test_ratio']:
                            inconsistent_keys.append('test_ratio')
                        if ui_values['random_seed'] != loaded_config['split']['random_seed']:
                            inconsistent_keys.append('random_seed')
                        if ui_values['stratify'] != loaded_config['split']['stratify']:
                            inconsistent_keys.append('stratify')
                        
                        # Update status
                        update_sync_status_only(ui_components, f"Inkonsistensi data pada: {', '.join(inconsistent_keys)}", 'warning')
                        
                        # Tampilkan pesan warning
                        display(create_info_alert(
                            f"{ICONS.get('warning', '⚠️')} Inkonsistensi data terdeteksi pada: {', '.join(inconsistent_keys)}",
                            alert_type='warning'
                        ))
                else:
                    # Update status
                    update_sync_status_only(ui_components, "Gagal menyimpan konfigurasi", 'error')
                    
                    # Tampilkan pesan error
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi",
                        alert_type='danger'
                    ))
            except Exception as e:
                # Update status
                update_sync_status_only(ui_components, f"Error saat menyimpan: {str(e)}", 'error')
                
                # Tampilkan pesan error
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Error saat menyimpan: {str(e)}",
                    alert_type='danger'
                ))
                logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan: {str(e)}")
        
        # Bind handler ke button
        if 'reset_button' in ui_components:
            ui_components['reset_button'].on_click(on_reset_clicked)
            
        if 'save_button' in ui_components:
            ui_components['save_button'].on_click(on_save_clicked)
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup button handlers: {str(e)}")
        return ui_components
