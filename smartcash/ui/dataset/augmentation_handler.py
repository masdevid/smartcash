"""
File: smartcash/ui/dataset/augmentation_handler.py
Deskripsi: Handler terintegrasi untuk augmentasi dataset dengan perbaikan num_workers dan deteksi data augmentasi
"""

from typing import Dict, Any, Optional
import time, os, sys
from pathlib import Path
from IPython.display import display, clear_output, HTML

def setup_augmentation_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup semua handler terintegrasi untuk augmentasi dataset."""
    
    # Import dan inisialisasi utilitas standar
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.alert_utils import create_status_indicator
    from smartcash.ui.utils.logging_utils import setup_ipython_logging, reset_logging
    
    # Import handlers terpisah yang dikonsolidasikan di sini
    from smartcash.ui.dataset.augmentation_initialization import detect_augmentation_state
    from smartcash.ui.dataset.augmentation_config_handler import load_augmentation_config, update_ui_from_config
    from smartcash.ui.dataset.augmentation_click_handler import setup_click_handlers
    from smartcash.ui.dataset.augmentation_progress_handler import setup_progress_handler
    from smartcash.ui.dataset.augmentation_visualization_handler import setup_visualization_handler
    from smartcash.ui.dataset.augmentation_cleanup_handler import setup_cleanup_handler
    
    try:
        # Setup logging dengan integrasi UI
        logger = setup_ipython_logging(ui_components, "augmentation_handler")
        
        # Inisialisasi dan deteksi state augmentasi
        ui_components = detect_augmentation_state(ui_components, env, config)
        
        # Inisialisasi AugmentationManager
        augmentation_manager = setup_augmentation_manager(ui_components, config)
        ui_components['augmentation_manager'] = augmentation_manager
        
        # Setup handlers spesifik (dengan order yang benar untuk interdependensi)
        ui_components = setup_progress_handler(ui_components, env, config)
        ui_components = setup_click_handlers(ui_components, env, config)
        ui_components = setup_visualization_handler(ui_components, env, config)
        ui_components = setup_cleanup_handler(ui_components, env, config)
        
        # Setup event handler untuk save button
        ui_components['save_button'].on_click(lambda b: save_config_handler(ui_components, config))
        
        # Muat konfigurasi yang tersimpan dan update UI
        saved_config = load_augmentation_config()
        if saved_config:
            ui_components = update_ui_from_config(ui_components, saved_config)
            logger.info(f"{ICONS['success']} Konfigurasi augmentasi dimuat")
        
        # Setup observer integration jika tersedia
        try:
            from smartcash.ui.handlers.observer_handler import setup_observer_handlers
            ui_components = setup_observer_handlers(ui_components, "augmentation_observers")
            logger.debug(f"{ICONS['info']} Observer berhasil diinisialisasi")
        except ImportError:
            pass
            
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
        
        # Callback untuk update summary setelah augmentasi
        def update_summary(result: Dict[str, Any]):
            import ipywidgets as widgets
            
            with ui_components['summary_container']:
                clear_output(wait=True)
                display(widgets.HTML(
                    f"""<div style="padding:10px; background-color:{COLORS['light']}; border-radius:5px; border-left:4px solid {COLORS['primary']}; color: black">
                    <h4 style="color:{COLORS['dark']}; margin-top:5px;">📊 Hasil Augmentasi</h4>
                    <ul>
                        <li><b>File asli:</b> {result.get('original', 0)}</li>
                        <li><b>File augmentasi:</b> {result.get('generated', 0)}</li>
                        <li><b>Total file:</b> {result.get('total_files', 0)}</li>
                        <li><b>Durasi:</b> {result.get('duration', 0):.2f} detik</li>
                        <li><b>Jenis augmentasi:</b> {', '.join(result.get('augmentation_types', []))}</li>
                    </ul>
                    </div>"""
                ))
            
            # Tampilkan summary container
            ui_components['summary_container'].layout.display = 'block'
        
        # Tambahkan update_summary ke UI components
        ui_components['update_summary'] = update_summary
        
        # Cleanup function untuk dijalankan saat cell di-reset
        def cleanup_resources():
            """Bersihkan resources yang digunakan oleh augmentation handler."""
            # Unregister observer group
            if 'observer_manager' in ui_components:
                ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
            
            # Reset flags
            ui_components['augmentation_running'] = False
            
            # Kembalikan logging ke default
            reset_logging()
        
        # Register cleanup function
        ui_components['cleanup'] = cleanup_resources
        
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
        # Update lokasi dari input jika tersedia
        if 'data_dir_input' in ui_components and 'output_dir_input' in ui_components:
            ui_components['data_dir'] = ui_components['data_dir_input'].value
            ui_components['augmented_dir'] = ui_components['output_dir_input'].value
            
        # Update konfigurasi dari UI
        updated_config = update_config_from_ui(ui_components, config)
        
        # Simpan konfigurasi ke file
        success = save_augmentation_config(updated_config)
        
        # Update UI
        if success:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("success", f"{ICONS.get('success', '✅')} Konfigurasi berhasil disimpan"))
            
            # Update status panel
            from smartcash.ui.dataset.augmentation_initialization import update_status_panel
            update_status_panel(
                ui_components, 
                "success", 
                f"{ICONS.get('success', '✅')} Konfigurasi augmentasi berhasil disimpan"
            )
            
            # Log
            if logger: logger.success(f"{ICONS.get('success', '✅')} Konfigurasi augmentasi berhasil disimpan")
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

def setup_augmentation_manager(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Any:
    """Setup dan inisialisasi AugmentationManager dengan perbaikan parameter."""
    from smartcash.ui.utils.constants import ICONS
    
    logger = ui_components.get('logger')
    
    try:
        # Import kelas augmentation service
        from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
        
        # Dapatkan paths dan config yang diperlukan
        data_dir = ui_components.get('data_dir', 'data')
        
        # Ambil num_workers dari UI jika tersedia
        num_workers = 4  # Default value
        if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 5:
            num_workers = ui_components['aug_options'].children[5].value
        
        # Buat instance AugmentationService dengan explicit parameter num_workers
        augmentation_manager = AugmentationService(config, data_dir, logger, num_workers)
        
        # Simpan instance di ui_components
        ui_components['augmentation_manager'] = augmentation_manager
        
        # Log
        if logger: logger.info(f"{ICONS.get('success', '✅')} AugmentationService berhasil diinisialisasi dengan {num_workers} workers")
        
        # Register progress callback jika tersedia
        if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
            ui_components['register_progress_callback'](augmentation_manager)
        
        return augmentation_manager
    except ImportError as e:
        # Tangani error import
        if logger: logger.warning(f"{ICONS.get('warning', '⚠️')} AugmentationService tidak tersedia: {str(e)}")
        return None
    except Exception as e:
        # Tangani error lainnya
        if logger: logger.error(f"{ICONS.get('error', '❌')} Error inisialisasi AugmentationManager: {str(e)}")
        return None