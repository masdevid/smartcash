"""
File: smartcash/ui/dataset/augmentation/handlers/button_handlers.py
Deskripsi: Handler tombol untuk augmentasi dataset dengan pendekatan SRP
"""

from typing import Dict, Any, Optional  # List, Union tidak digunakan langsung
from IPython.display import display, clear_output
import traceback
import ipywidgets as widgets
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
# get_logger diimpor tapi tidak digunakan langsung (logger diambil dari ui_components)
# update_status_panel diimpor tapi tidak digunakan langsung (digunakan dalam execution_handler)

# Import handler terpisah untuk SRP
from smartcash.ui.dataset.augmentation.handlers.config_persistence import (
    ensure_ui_persistence, save_augmentation_config, reset_config_to_default, sync_config_with_drive
)
# validate_augmentation_config tidak digunakan langsung
from smartcash.ui.dataset.augmentation.handlers.config_mapper import (
    map_ui_to_config  # map_config_to_ui, extract_augmentation_params tidak digunakan langsung
)
# observer_handler tidak digunakan langsung (menggunakan import langsung dari smartcash.components.observer)
from smartcash.ui.dataset.augmentation.handlers.execution_handler import run_augmentation
from smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler import stop_augmentation

def setup_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol UI augmentasi dengan sistem progress yang ditingkatkan.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Pastikan persistensi UI components
    ensure_ui_persistence(ui_components)
    logger = ui_components.get('logger')
    
    # Import handler terpisah untuk SRP
    from smartcash.ui.dataset.augmentation.handlers.progress_handler import (
        register_progress_callback, reset_progress_tracking
    )
    from smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler import (
        get_augmentation_service, execute_augmentation, stop_augmentation
    )
    
    # Penanganan error dengan decorator standar
    from smartcash.ui.handlers.error_handler import try_except_decorator
    
    @try_except_decorator(ui_components.get('status'))
    def on_augment_click(b):
        """Handler tombol augmentasi dengan dukungan progress tracking yang dioptimalkan."""
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger')
        if logger:
            logger.info(f"{ICONS['info']} Tombol augmentasi diklik")
        
        # Update UI: menampilkan proses dimulai
        with ui_components['status']: 
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai augmentasi dataset..."))
        
        # Tampilkan log panel dan progress bar
        ui_components['log_accordion'].selected_index = 0  # Expand log
        # Tampilkan progress bar dan label
        for element in ['progress_bar', 'current_progress', 'overall_label', 'step_label']:
            if element in ui_components:
                ui_components[element].layout.visibility = 'visible'
        
        # Disable semua komponen UI selama proses
        disable_ui_during_processing(ui_components, True)
        
        # Update UI tombol
        ui_components['augment_button'].layout.display = 'none'
        ui_components['stop_button'].layout.display = 'block'
        
        # Tandai bahwa augmentasi sedang berjalan
        ui_components['augmentation_running'] = True
        
        # Update konfigurasi dari UI dan simpan dengan ConfigManager
        try:
            # Gunakan config mapper untuk update konfigurasi dari UI
            updated_config = map_ui_to_config(ui_components, config)
            if logger:
                logger.info(f"{ICONS['info']} Konfigurasi berhasil diupdate dari UI")
                
            success = save_augmentation_config(updated_config)
            if success and logger:
                logger.info(f"{ICONS['success']} Konfigurasi augmentasi berhasil disimpan")
            else:
                logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi augmentasi")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Jalankan augmentasi dengan handler terpisah
        try:
            # Log proses yang akan dilakukan
            if logger:
                logger.info(f"{ICONS['info']} Memulai proses augmentasi dataset")
            
            # Gunakan execution_handler untuk menjalankan augmentasi
            from smartcash.ui.dataset.augmentation.handlers.initialization_handler import initialize_augmentation_directories
            
            # Inisialisasi direktori terlebih dahulu
            if logger:
                logger.info(f"{ICONS['info']} Menginisialisasi direktori augmentasi")
                
            init_result = initialize_augmentation_directories(ui_components)
            if not init_result['success']:
                # Jika inisialisasi gagal, tampilkan error dan hentikan proses
                error_message = init_result.get('message', 'Gagal menginisialisasi direktori augmentasi')
                if logger:
                    logger.error(f"{ICONS['error']} {error_message}")
                    
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_status_indicator("error", f"{ICONS['error']} {error_message}"))
                cleanup_ui(ui_components)
                ui_components['augmentation_running'] = False
                return
            
            # Jalankan augmentasi dengan execution_handler
            if logger:
                logger.info(f"{ICONS['info']} Menjalankan proses augmentasi dengan execution_handler")
                
            # Import di sini untuk menghindari circular import
            from smartcash.ui.dataset.augmentation.handlers.execution_handler import run_augmentation
            result = run_augmentation(ui_components, config)
            
            # Tampilkan hasil jika berhasil
            if result and isinstance(result, dict):
                # Tampilkan summary
                if 'summary_container' in ui_components:
                    ui_components['summary_container'].layout.display = 'block'
                    with ui_components['summary_container']:
                        clear_output(wait=True)
                        display(create_status_indicator("success", f"{ICONS['success']} Augmentasi selesai"))
                        
                        # Tampilkan summary dengan format yang lebih baik
                        display(widgets.HTML(f"""<div style='padding:10px; background-color:#f8f9fa; border-radius:4px;'>
                            <h4>Hasil Augmentasi:</h4>
                            <ul>
                                <li><b>Total gambar yang diproses:</b> {result.get('total_processed', 0)}</li>
                                <li><b>Total gambar yang dihasilkan:</b> {result.get('total_generated', 0)}</li>
                                <li><b>Direktori output:</b> {result.get('output_dir', 'data/augmented')}</li>
                                <li><b>Waktu eksekusi:</b> {result.get('execution_time', 0):.2f} detik</li>
                            </ul>
                        </div>"""))
                
                # Tampilkan tombol visualisasi
                if 'visualization_buttons' in ui_components:
                    ui_components['visualization_buttons'].layout.display = 'block'
                
                # Tampilkan tombol cleanup
                if 'cleanup_button' in ui_components:
                    ui_components['cleanup_button'].layout.display = 'block'
        except Exception as e:
            # Tangani error
            if logger: logger.error(f"{ICONS['error']} Error saat augmentasi: {str(e)}")
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS['error']} Error saat augmentasi: {str(e)}"))
                display(widgets.HTML(f"<pre>{traceback.format_exc()}</pre>"))
        finally:
            # Cleanup UI
            cleanup_ui(ui_components)
            ui_components['augmentation_running'] = False
            
            # Restore UI
            cleanup_ui(ui_components)
    
    # Handler untuk tombol stop dengan notifikasi yang ditingkatkan
    def on_stop_click(b):
        """Handler untuk menghentikan augmentasi."""
        # Set flag untuk menghentikan augmentasi
        ui_components['augmentation_running'] = False
        
        # Tampilkan pesan di status
        with ui_components['status']: 
            display(create_status_indicator("warning", f"{ICONS['warning']} Menghentikan augmentasi..."))
        
        # Update status panel
        update_status_panel(ui_components, "warning", f"{ICONS['warning']} Augmentasi dihentikan oleh pengguna")
        
        # Hentikan augmentasi menggunakan handler terpisah
        try:
            # Dapatkan augmentation service dari ui_components
            augmentation_service = ui_components.get('augmentation_service')
            if augmentation_service:
                # Gunakan handler untuk menghentikan augmentasi
                stop_augmentation(augmentation_service)
                if logger: logger.info(f"{ICONS['stop']} Augmentasi dihentikan oleh pengguna")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat menghentikan augmentasi: {str(e)}")
        
        # Notifikasi observer tentang pembatalan
        notify_process_error(ui_components, "Augmentasi dihentikan oleh pengguna")
        
        # Reset UI
        cleanup_ui(ui_components)
    
    # Reset handler untuk reset button
    @try_except_decorator(ui_components.get('status'))
    def on_reset_click(b):
        """Reset UI dan konfigurasi ke default."""
        # Tampilkan status
        with ui_components['status']: 
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Mereset konfigurasi ke default..."))
        
        try:
            # Reset UI terlebih dahulu
            reset_ui(ui_components)
            
            # Gunakan fungsi reset_config_to_default dari config_persistence
            success = reset_config_to_default(ui_components)
            
            # Pastikan UI components terdaftar untuk persistensi
            ensure_ui_persistence(ui_components)
            
            # Sinkronkan dengan drive
            try:
                drive_sync = sync_config_with_drive(ui_components)
                if drive_sync and logger:
                    logger.info(f"{ICONS['success']} Konfigurasi default berhasil disinkronkan dengan drive")
            except Exception as e:
                if logger:
                    logger.warning(f"{ICONS['warning']} Gagal menyinkronkan default dengan drive: {str(e)}")
            
            # Update status panel
            status_type = 'success' if success else 'warning'
            message = f"{ICONS['success' if success else 'warning']} Konfigurasi {'berhasil' if success else 'sebagian'} direset ke default"
            
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator(status_type, message))
            
            # Update status panel
            try:
                update_status_panel(ui_components, status_type, message)
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Gagal update status panel: {str(e)}")
            
            # Log success
            if logger:
                log_method = logger.success if success else logger.warning
                log_method(message)
            
        except Exception as e:
            # Update status
            with ui_components['status']: 
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS['error']} Error saat mereset konfigurasi: {str(e)}"))
            
            if logger: logger.error(f"{ICONS['error']} Error saat mereset konfigurasi: {str(e)}")
            
            # Update status panel dengan validasi
            try:
                update_status_panel(ui_components, "warning", f"{ICONS['warning']} Reset UI sebagian: {str(e)}")
            except Exception:
                pass
            
            if logger: logger.warning(f"{ICONS['warning']} Error saat reset konfigurasi: {str(e)}")
            
            # Fallback: coba reset dengan cara lama
            try:
                # Dapatkan konfigurasi default
                from smartcash.ui.dataset.augmentation.handlers.config_persistence import get_default_augmentation_config
                default_config = get_default_augmentation_config()
                
                # Update UI dari konfigurasi default
                map_config_to_ui(ui_components, default_config)
            except Exception:
                pass

    # Register handlers untuk tombol-tombol dengan one-liner
    button_handlers = {
        'augment_button': on_augment_click,
        'stop_button': on_stop_click,
        'reset_button': on_reset_click
    }
    
    [ui_components[button].on_click(handler) for button, handler in button_handlers.items() if button in ui_components]
    
    # Tambahkan referensi ke handlers di ui_components
    ui_components.update({
        'on_augment_click': on_augment_click,
        'on_stop_click': on_stop_click,
        'on_reset_click': on_reset_click,
        'augmentation_running': False
    })
    
    return ui_components

def disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """
    Enable/disable komponen UI saat augmentasi berjalan.
    
    Args:
        ui_components: Dictionary komponen UI 
        disable: Apakah perlu di-disable
    """
    # Disable komponen utama
    aug_options = ui_components.get('aug_options')
    if aug_options is not None and hasattr(aug_options, 'children'):
        # Disable tab container (children[1])
        if len(aug_options.children) >= 2 and hasattr(aug_options.children[1], 'children'):
            tab = aug_options.children[1]
            # Disable semua tab
            for tab_child in tab.children:
                if hasattr(tab_child, 'children'):
                    for child in tab_child.children:
                        if hasattr(child, 'disabled'):
                            child.disabled = disable
    
    # Disable tombol-tombol
    for btn in ['save_button', 'reset_button', 'cleanup_button']:
        if btn in ui_components:
            ui_components[btn].disabled = disable

def cleanup_ui(ui_components: Dict[str, Any]) -> None:
    """
    Kembalikan UI ke kondisi awal setelah augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Enable kembali semua UI component
    disable_ui_during_processing(ui_components, False)
    
    # Kembalikan tampilan tombol
    ui_components['augment_button'].layout.display = 'block'
    ui_components['stop_button'].layout.display = 'none'
    
    # Reset progress bar menggunakan handler terpisah
    try:
        from smartcash.ui.dataset.augmentation.handlers.progress_handler import reset_progress_tracking
        reset_progress_tracking(ui_components)
    except Exception as e:
        # Fallback jika import gagal
        for element in ['progress_bar', 'current_progress', 'overall_label', 'step_label']:
            if element in ui_components:
                if hasattr(ui_components[element], 'layout') and hasattr(ui_components[element].layout, 'visibility'):
                    ui_components[element].layout.visibility = 'hidden'
                if hasattr(ui_components[element], 'value') and element in ['progress_bar', 'current_progress']:
                    ui_components[element].value = 0

def reset_ui(ui_components: Dict[str, Any]) -> None:
    """
    Reset UI ke kondisi default.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Reset tombol dan progress
    cleanup_ui(ui_components)
    
    # Hide containers
    for component in ['visualization_container', 'summary_container']:
        if component in ui_components:
            ui_components[component].layout.display = 'none'
            with ui_components[component]: 
                clear_output()
    
    # Hide buttons
    for btn in ['visualization_buttons', 'cleanup_button']:
        if btn in ui_components:
            ui_components[btn].layout.display = 'none'
    
    # Reset logs
    if 'status' in ui_components:
        with ui_components['status']: 
            clear_output()
    
    # Reset accordion
    if 'log_accordion' in ui_components:
        ui_components['log_accordion'].selected_index = None
# Fungsi get_dataset_manager telah dipindahkan ke augmentation_service_handler.py
# Gunakan get_augmentation_service dari augmentation_service_handler.py sebagai gantinya