"""
File: smartcash/ui/dataset/augmentation_handler.py
Deskripsi: Handler terpadu untuk augmentasi dataset menggunakan shared modules
"""

from typing import Dict, Any, Optional
import time, os, sys
from pathlib import Path
from IPython.display import display, clear_output, HTML

def setup_augmentation_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup semua handler terpadu untuk augmentasi dataset menggunakan shared modules."""
    
    # Import dan inisialisasi utilitas standar
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.alert_utils import create_status_indicator
    from smartcash.ui.utils.logging_utils import setup_ipython_logging, reset_logging
    
    try:
        # Setup logging dengan integrasi UI
        logger = setup_ipython_logging(ui_components, "augmentation_handler")
        ui_components['logger'] = logger  # Pastikan logger selalu tersedia di ui_components
        
        # Catat lokasi awal
        if logger: logger.debug(f"Inisialisasi augmentation handler dengan cwd: {os.getcwd()}")
        
        # Load konfigurasi terlebih dahulu, sehingga UI dibuat dengan konfigurasi yang benar
        from smartcash.ui.dataset.augmentation_config_handler import load_augmentation_config
        saved_config = load_augmentation_config(ui_components=ui_components)
        # Store config di ui_components agar bisa diakses oleh semua handler
        ui_components['config'] = saved_config
        
        # Inisialisasi dan deteksi state augmentasi
        from smartcash.ui.dataset.shared.setup_utils import detect_module_state
        ui_components = detect_module_state(ui_components, 'augmentation')
        
        # Inisialisasi AugmentationManager
        from smartcash.ui.dataset.shared.setup_utils import setup_manager
        augmentation_manager = setup_manager(ui_components, saved_config, 'augmentation')
        ui_components['augmentation_manager'] = augmentation_manager
        
        # Setup handlers spesifik augmentasi (yang tidak di-share)
        from smartcash.ui.dataset.augmentation_click_handler import setup_click_handlers
        ui_components = setup_click_handlers(ui_components, env, saved_config)
        
        # Gunakan shared handlers untuk progress, visualization, cleanup dan summary
        from smartcash.ui.dataset.shared.integration import apply_shared_handlers, create_cleanup_function
        ui_components = apply_shared_handlers(ui_components, env, saved_config, 'augmentation')
        create_cleanup_function(ui_components, 'augmentation')
        
        # Setup event handler untuk save button (spesifik augmentasi)
        ui_components['save_button'].on_click(lambda b: save_config_handler(ui_components, saved_config))
        
        # Update UI dari konfigurasi yang telah dimuat
        from smartcash.ui.dataset.augmentation_config_handler import update_ui_from_config
        if saved_config:
            ui_components = update_ui_from_config(ui_components, saved_config)
            logger.info(f"{ICONS['success']} Konfigurasi augmentasi dimuat")
            
        # Sembunyikan tombol yang tidak relevan pada awal
        if not ui_components.get('augmentation_running', False):
            ui_components['stop_button'].layout.display = 'none'
            ui_components['progress_bar'].layout.visibility = 'hidden'
            ui_components['current_progress'].layout.visibility = 'hidden'
            
        # Jika tidak terdeteksi data augmentasi, sembunyikan container visualisasi
        if not ui_components.get('is_augmented', False):
            # Sembunyikan summary dan visualization container pada awal
            ui_components['summary_container'].layout.display = 'none'
            ui_components['visualization_container'].layout.display = 'none'
            ui_components['visualization_buttons'].layout.display = 'none'
            
            # Sembunyikan tombol cleanup
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'none'
        
        # Tampilkan Log accordion terbuka secara default
        ui_components['log_accordion'].selected_index = 0
        
        logger.info(f"{ICONS['success']} Augmentation handler berhasil diinisialisasi")
        
    except Exception as e:
        # Pastikan logging dikembalikan ke default
        reset_logging()
        
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error inisialisasi handler: {str(e)}"))
        
        # Log error
        if logger: logger.error(f"{ICONS.get('error', '❌')} Error inisialisasi augmentation handler: {str(e)}")
    
    return ui_components

def save_config_handler(ui_components: Dict[str, Any], config: Dict[str, Any] = None):
    """Handler untuk menyimpan konfigurasi dari UI ke file."""
    from smartcash.ui.utils.constants import ICONS
    from smartcash.ui.utils.alert_utils import create_status_indicator
    from smartcash.ui.dataset.augmentation_config_handler import update_config_from_ui, save_augmentation_config
    
    logger = ui_components.get('logger')
    
    try:
        # Update konfigurasi dari UI dengan config yang sebelumnya diload
        updated_config = update_config_from_ui(ui_components, ui_components.get('config', config))
        
        # Tambahkan logger ke config untuk digunakan oleh fungsi save
        updated_config['logger'] = logger
        
        # Simpan konfigurasi yang sudah diupdate
        config_path = "configs/augmentation_config.yaml"
        success = save_augmentation_config(updated_config, config_path)
        
        # Simpan konfigurasi yang sudah diupdate ke ui_components
        ui_components['config'] = updated_config
        
        # Update UI
        if success:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("success", f"{ICONS.get('success', '✅')} Konfigurasi berhasil disimpan ke {config_path}"))
            
            # Update status panel
            from smartcash.ui.dataset.shared.status_panel import update_status_panel
            update_status_panel(
                ui_components, 
                "success", 
                f"{ICONS.get('success', '✅')} Konfigurasi augmentasi berhasil disimpan ke {config_path}"
            )
            
            # Log
            if logger: logger.success(f"{ICONS.get('success', '✅')} Konfigurasi augmentasi berhasil disimpan ke {config_path}")
        else:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Gagal menyimpan konfigurasi"))
            
            if logger: logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyimpan konfigurasi")
    except Exception as e:
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
        
        if logger: logger.error(f"{ICONS.get('error', '❌')} Error menyimpan konfigurasi: {str(e)}")