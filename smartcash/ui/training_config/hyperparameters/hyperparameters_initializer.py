"""
File: smartcash/ui/training_config/hyperparameters/hyperparameters_initializer.py
Deskripsi: Initializer untuk UI konfigurasi hyperparameter model
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output
import os
import yaml
import copy

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager

logger = get_logger(__name__)

def initialize_hyperparameters_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI untuk konfigurasi hyperparameter model.
    
    Args:
        env: Environment manager
        config: Konfigurasi untuk model
        
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {'module_name': 'hyperparameters'}
    
    try:
        # Dapatkan environment dan config jika belum tersedia
        env = env or get_environment_manager()
        config_manager = get_config_manager()
        
        # Dapatkan konfigurasi dari ConfigManager
        if config is None:
            config = config_manager.get_module_config('hyperparameters', {})
        
        # Buat komponen UI
        from smartcash.ui.training_config.hyperparameters.components.hyperparameters_components import create_hyperparameters_ui
        ui_components.update(create_hyperparameters_ui(config))
        
        # Tambahkan tombol konfigurasi dari komponen standar
        from smartcash.ui.components.config_buttons import create_config_buttons
        config_buttons = create_config_buttons()
        ui_components.update({
            'save_button': config_buttons['save_button'],
            'reset_button': config_buttons['reset_button'],
            'config_buttons': config_buttons['container']
        })
        
        # Setup multi-progress tracking
        from smartcash.ui.handlers.multi_progress import setup_multi_progress_tracking
        setup_multi_progress_tracking(ui_components, "hyperparameters", "hyperparameters_step")
        
        # Setup handlers untuk tombol save dan reset
        def on_save_config(b):
            from smartcash.ui.utils.alert_utils import create_info_alert
            
            # Update config dari UI
            updated_config = update_config_from_ui(ui_components, copy.deepcopy(config))
            
            # Simpan konfigurasi menggunakan ConfigManager
            success = config_manager.save_module_config('hyperparameters', updated_config)
            
            # Pastikan UI components teregistrasi untuk persistensi
            config_manager.register_ui_components('hyperparameters', ui_components)
            
            # Update status
            if 'status' in ui_components:
                with ui_components['status']:
                    clear_output(wait=True)
                    if success:
                        display(create_info_alert(
                            f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil disimpan",
                            alert_type='success'
                        ))
                    else:
                        display(create_info_alert(
                            f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi hyperparameter",
                            alert_type='error'
                        ))
            
            # Update info panel
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info']()
                
            # Sinkronisasi ke Google Drive jika diaktifkan
            try:
                if env.is_drive_mounted:
                    # Sinkronisasi ke Google Drive
                    from smartcash.ui.training_config.hyperparameters.handlers.drive_handlers import sync_to_drive
                    logger.info(f"{ICONS.get('info', 'ℹ️')} Menyinkronkan konfigurasi hyperparameter ke Google Drive...")
                    sync_to_drive(None, ui_components)
            except Exception as e:
                logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyinkronkan ke Google Drive: {str(e)}")
                
            logger.info(f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil disimpan")
        
        def on_reset_config(b):
            # Deklarasikan ui_components sebagai nonlocal untuk mencegah UnboundLocalError
            nonlocal ui_components
            
            # Dapatkan default config
            default_config = config_manager.get_module_config('hyperparameters', {})
            
            # Update UI dari default config
            ui_components = update_ui_from_config(ui_components, default_config)
            
            # Simpan default config menggunakan ConfigManager
            success = config_manager.save_module_config('hyperparameters', default_config)
            
            # Pastikan UI components teregistrasi untuk persistensi
            config_manager.register_ui_components('hyperparameters', ui_components)
            
            # Tampilkan status
            from smartcash.ui.utils.alert_utils import create_info_alert
            if 'status' in ui_components:
                with ui_components['status']:
                    clear_output(wait=True)
                    if success:
                        display(create_info_alert(
                            f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil direset ke default",
                            alert_type='success'
                        ))
                    else:
                        display(create_info_alert(
                            f"{ICONS.get('error', '❌')} Gagal mereset konfigurasi hyperparameter",
                            alert_type='error'
                        ))
            
            # Update info panel
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info']()
                
            # Sinkronisasi ke Google Drive jika diaktifkan
            try:
                if env.is_drive_mounted:
                    # Sinkronisasi ke Google Drive
                    from smartcash.ui.training_config.hyperparameters.handlers.drive_handlers import sync_to_drive
                    logger.info(f"{ICONS.get('info', 'ℹ️')} Menyinkronkan konfigurasi hyperparameter ke Google Drive...")
                    sync_to_drive(None, ui_components)
            except Exception as e:
                logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyinkronkan ke Google Drive: {str(e)}")
                
            logger.info(f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil direset ke default")
        
        # Register handler untuk tombol
        ui_components['save_button'].on_click(on_save_config)
        ui_components['reset_button'].on_click(on_reset_config)
        
        # Tambahkan fungsi helper ke ui_components
        ui_components.update({
            'update_config_from_ui': update_config_from_ui,
            'update_ui_from_config': update_ui_from_config,
            'config': config
        })
        
        # Setup handlers lainnya
        from smartcash.ui.training_config.hyperparameters.handlers.button_handlers import setup_hyperparameters_button_handlers
        from smartcash.ui.training_config.hyperparameters.handlers.form_handlers import setup_hyperparameters_form_handlers
        
        ui_components = setup_hyperparameters_button_handlers(ui_components, env, config)
        ui_components = setup_hyperparameters_form_handlers(ui_components, env, config)
        
        # Update UI dari config yang tersimpan
        ui_components = update_ui_from_config(ui_components, config)
        
        # Tampilkan container utama
        if 'main_container' in ui_components:
            # Ganti placeholder tombol dengan config_buttons
            if 'buttons_placeholder' in ui_components and hasattr(ui_components['main_container'], 'children'):
                # Tambahkan tombol dan info sinkronisasi ke placeholder
                children_list = list(ui_components['main_container'].children)
                placeholder_idx = next((i for i, child in enumerate(children_list) 
                                    if child is ui_components['buttons_placeholder']), -1)
                
                # Buat container untuk tombol dan info sinkronisasi
                buttons_container = widgets.VBox([
                    config_buttons['container'],
                    ui_components['sync_info']
                ])
                
                if placeholder_idx >= 0:
                    # Ganti placeholder dengan tombol konfigurasi dan info sinkronisasi
                    children_list[placeholder_idx] = buttons_container
                    ui_components['main_container'].children = tuple(children_list)
                else:
                    # Jika placeholder tidak ditemukan, tambahkan sebelum status
                    if 'status' in ui_components:
                        status_idx = next((i for i, child in enumerate(children_list) 
                                        if child is ui_components['status']), len(children_list))
                        children_list.insert(status_idx, buttons_container)
                        ui_components['main_container'].children = tuple(children_list)
            
            # Perbarui informasi sinkronisasi berdasarkan status drive
            try:
                if not env.is_drive_mounted:
                    # Jika drive tidak diaktifkan, perbarui pesan
                    ui_components['sync_info'].value = f"<div style='margin-top: 5px; font-style: italic; color: #666;'>{ICONS.get('warning', '⚠️')} Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi otomatis.</div>"
            except Exception as e:
                logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal memeriksa status drive: {str(e)}")
            
            # Inisialisasi UI dari konfigurasi
            try:
                # Dapatkan konfigurasi
                current_config = config_manager.get_module_config('hyperparameters')
                
                # Update UI dari konfigurasi
                update_ui_from_config(ui_components, current_config)
                
                # Update info panel
                if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                    ui_components['update_hyperparameters_info']()
                
                # Register UI components untuk persistensi
                config_manager.register_ui_components('hyperparameters', ui_components)
                
                logger.info(f"{ICONS.get('success', '✅')} UI hyperparameter berhasil diinisialisasi")
            except Exception as e:
                logger.error(f"{ICONS.get('error', '❌')} Error saat inisialisasi UI hyperparameter: {str(e)}")
            
            display(ui_components['main_container'])
        else:
            # Fallback jika main_container tidak ada
            from smartcash.ui.utils.alert_utils import create_info_alert
            display(create_info_alert(
                f"{ICONS.get('warning', '⚠️')} Container utama tidak ditemukan. Mencoba menampilkan komponen yang tersedia.",
                alert_type='warning'
            ))
            # Coba tampilkan komponen yang tersedia
            if 'form' in ui_components:
                display(ui_components['form'])
            display(config_buttons['container'])
            display(ui_components['sync_info'])
            if 'status' in ui_components:
                display(ui_components['status'])
        
    except Exception as e:
        # Gunakan utilitas fallback yang ada
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        ui_components = create_fallback_ui(ui_components, str(e), "error")
    
    return ui_components

# Fungsi untuk mengupdate konfigurasi dari UI
def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari nilai UI.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi hyperparameter
        
    Returns:
        Konfigurasi yang diupdate
    """
    # Pastikan struktur config ada
    if 'hyperparameters' not in config:
        config['hyperparameters'] = {}
    
    # Ekstrak nilai dari form hyperparameters
    if 'batch_size_slider' in ui_components:
        config['hyperparameters']['batch_size'] = ui_components['batch_size_slider'].value
    
    if 'epochs_slider' in ui_components:
        config['hyperparameters']['epochs'] = ui_components['epochs_slider'].value
    
    if 'learning_rate_slider' in ui_components:
        config['hyperparameters']['learning_rate'] = ui_components['learning_rate_slider'].value
    
    if 'optimizer_dropdown' in ui_components:
        config['hyperparameters']['optimizer'] = ui_components['optimizer_dropdown'].value
    
    # Simpan konfigurasi di ui_components
    ui_components['config'] = config
    
    return config

# Fungsi untuk mengupdate UI dari konfigurasi
def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi hyperparameter
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Dapatkan konfigurasi hyperparameters
    hp_config = config.get('hyperparameters', {})
    
    # Update slider batch size
    if 'batch_size_slider' in ui_components:
        ui_components['batch_size_slider'].value = hp_config.get('batch_size', 16)
    
    # Update slider epochs
    if 'epochs_slider' in ui_components:
        ui_components['epochs_slider'].value = hp_config.get('epochs', 50)
    
    # Update slider learning rate
    if 'learning_rate_slider' in ui_components:
        ui_components['learning_rate_slider'].value = hp_config.get('learning_rate', 0.001)
    
    # Update dropdown optimizer
    if 'optimizer_dropdown' in ui_components:
        optimizer = hp_config.get('optimizer', 'Adam')
        if optimizer in ui_components['optimizer_dropdown'].options:
            ui_components['optimizer_dropdown'].value = optimizer
    
    # Simpan referensi config di ui_components
    ui_components['config'] = config
    
    return ui_components
