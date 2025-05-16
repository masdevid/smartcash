"""
File: smartcash/ui/dataset/preprocessing/handlers/button_handlers.py
Deskripsi: Handler tombol untuk preprocessing dataset dengan pendekatan SRP
"""

from typing import Dict, Any  # Optional, List, Union tidak digunakan langsung
from IPython.display import display, clear_output
import os
import traceback
import ipywidgets as widgets
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.preprocessing.handlers.status_handler import update_status_panel
# get_logger diimpor tapi tidak digunakan langsung (logger diambil dari ui_components)

# Import handler terpisah untuk SRP
from smartcash.ui.dataset.preprocessing.handlers.config_handler import save_preprocessing_config
from smartcash.ui.dataset.preprocessing.handlers.service_handler import run_preprocessing
from smartcash.ui.dataset.preprocessing.handlers.parameter_handler import validate_preprocessing_params
from smartcash.ui.dataset.preprocessing.handlers.persistence_handler import sync_config_with_drive, ensure_ui_persistence, reset_config_to_default
from smartcash.ui.dataset.preprocessing.handlers.initialization_handler import initialize_preprocessing_directories

def setup_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol UI preprocessing dengan sistem progress yang ditingkatkan.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Penanganan error dengan decorator standar
    from smartcash.ui.handlers.error_handler import try_except_decorator
    
    @try_except_decorator(ui_components.get('status'))
    def on_preprocess_click(b):
        """Handler tombol preprocessing dengan dukungan progress tracking yang dioptimalkan."""
        # Update UI: menampilkan proses dimulai
        with ui_components['status']: 
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai preprocessing dataset..."))
        
        # Tampilkan log panel dan progress bar
        ui_components['log_accordion'].selected_index = 0  # Expand log
        # Tampilkan progress bar dan label
        for element in ['progress_bar', 'current_progress', 'overall_label', 'step_label']:
            if element in ui_components:
                ui_components[element].layout.visibility = 'visible'
        
        # Disable semua komponen UI selama proses
        disable_ui_during_processing(ui_components, True)
        
        # Update UI tombol
        ui_components['preprocess_button'].layout.display = 'none'
        ui_components['stop_button'].layout.display = 'block'
        
        # Update konfigurasi dari UI dan simpan dengan ConfigManager
        try:
            # Gunakan config handler untuk update konfigurasi dari UI
            updated_config = ui_components['update_config_from_ui'](ui_components, config)
            success = save_preprocessing_config(updated_config)
            
            # Sinkronkan konfigurasi dengan drive
            drive_sync_success = sync_config_with_drive(ui_components)
            
            if success and drive_sync_success and logger:
                logger.info(f"{ICONS['success']} Konfigurasi preprocessing berhasil disimpan dan tersinkron dengan drive")
            else:
                logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi preprocessing atau menyinkronkan dengan drive")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Tandai preprocessing sedang berjalan
        ui_components['preprocessing_running'] = True
        
        # Notifikasi observer tentang mulai preprocessing
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(event_type=EventTopics.PREPROCESSING_START, sender="preprocessing_handler", 
                  message=f"Memulai preprocessing dataset", split=None, split_info="Semua")
        except ImportError: 
            pass
        
        # Jalankan preprocessing dengan handler terpisah
        try:
            # Gunakan initialization_handler untuk inisialisasi direktori terlebih dahulu
            init_result = initialize_preprocessing_directories(ui_components)
            if not init_result['success']:
                # Jika inisialisasi gagal, tampilkan error dan hentikan proses
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_status_indicator("error", f"{ICONS['error']} {init_result['message']}"))
                cleanup_ui(ui_components)
                ui_components['preprocessing_running'] = False
                return
            
            # Jalankan preprocessing dengan execution_handler
            result = run_preprocessing(ui_components, config)
            
            # Tampilkan hasil jika berhasil
            if result and isinstance(result, dict):
                # Tampilkan summary
                if 'summary_container' in ui_components:
                    ui_components['summary_container'].layout.display = 'block'
                    with ui_components['summary_container']:
                        clear_output(wait=True)
                        display(create_status_indicator("success", f"{ICONS['success']} Preprocessing selesai"))
                        
                        # Tampilkan summary dengan format yang lebih baik
                        display(widgets.HTML(f"""<div style='padding:10px; background-color:#f8f9fa; border-radius:4px;'>
                            <h4>Hasil Preprocessing:</h4>
                            <ul>
                                <li><b>Total gambar yang diproses:</b> {result.get('total_processed', 0)}</li>
                                <li><b>Direktori output:</b> {result.get('output_dir', 'data/preprocessed')}</li>
                                <li><b>Split:</b> {result.get('split', 'Semua')}</li>
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
            if logger: logger.error(f"{ICONS['error']} Error saat preprocessing: {str(e)}")
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS['error']} Error saat preprocessing: {str(e)}"))
                display(widgets.HTML(f"<pre>{traceback.format_exc()}</pre>"))
        finally:
            # Cleanup UI
            cleanup_ui(ui_components)
            ui_components['preprocessing_running'] = False
            
            # Restore UI
            cleanup_ui(ui_components)
    
    # Handler untuk tombol stop dengan notifikasi yang ditingkatkan
    def on_stop_click(b):
        """Handler untuk menghentikan preprocessing."""
        # Set flag untuk menghentikan preprocessing
        ui_components['preprocessing_running'] = False
        
        # Tampilkan pesan di status
        with ui_components['status']: 
            display(create_status_indicator("warning", f"{ICONS['warning']} Menghentikan preprocessing..."))
        
        # Update status panel
        update_status_panel(ui_components, "warning", f"{ICONS['warning']} Preprocessing dihentikan oleh pengguna")
        
        # Notifikasi observer
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(
                event_type=EventTopics.PREPROCESSING_END,
                sender="preprocessing_handler",
                message=f"Preprocessing dihentikan oleh pengguna",
                status="cancelled"
            )
        except ImportError:
            pass
        
        # Reset UI
        cleanup_ui(ui_components)
    
    # Reset handler untuk reset button
    def on_reset_click(b):
        """Reset UI dan konfigurasi ke default."""
        # Reset UI
        reset_ui(ui_components)
        
        # Reset konfigurasi ke default dan update UI
        try:
            # Gunakan fungsi reset_config_to_default dari persistence_handler
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
            
            # Deteksi state preprocessing
            from smartcash.ui.dataset.preprocessing.handlers.state_handler import detect_preprocessing_state
            detect_preprocessing_state(ui_components)
            
            # Tampilkan pesan sukses
            status_type = 'success' if success else 'warning'
            message = f"{ICONS['success' if success else 'warning']} UI dan konfigurasi {'berhasil' if success else 'sebagian'} direset ke nilai default"
            
            with ui_components['status']: 
                display(create_status_indicator(status_type, message))
            
            # Update status panel
            update_status_panel(ui_components, status_type, message)
            
            # Log success
            if logger:
                log_method = logger.success if success else logger.warning
                log_method(message)
        except Exception as e:
            # Jika gagal reset konfigurasi, tampilkan pesan error
            with ui_components['status']: 
                display(create_status_indicator("warning", f"{ICONS['warning']} Reset sebagian: {str(e)}"))
            
            # Update status panel
            update_status_panel(ui_components, "warning", f"{ICONS['warning']} Reset UI sebagian: {str(e)}")
            
            if logger: logger.warning(f"{ICONS['warning']} Error saat reset konfigurasi: {str(e)}")
            
            # Fallback: coba load default config dengan cara lama
            try:
                default_config = ui_components['load_preprocessing_config']()
                ui_components['update_ui_from_config'](ui_components, default_config)
            except Exception:
                pass
    
    # Register handlers untuk tombol-tombol dengan one-liner
    button_handlers = {
        'preprocess_button': on_preprocess_click,
        'stop_button': on_stop_click,
        'reset_button': on_reset_click
    }
    
    [ui_components[button].on_click(handler) for button, handler in button_handlers.items() if button in ui_components]
    
    # Tambahkan referensi ke handlers di ui_components
    ui_components.update({
        'on_preprocess_click': on_preprocess_click,
        'on_stop_click': on_stop_click,
        'on_reset_click': on_reset_click,
        'preprocessing_running': False
    })
    
    return ui_components

def disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """
    Enable/disable komponen UI saat preprocessing berjalan.
    
    Args:
        ui_components: Dictionary komponen UI 
        disable: Apakah perlu di-disable
    """
    # Disable semua komponen utama
    for component_name in ['preprocess_options', 'split_selector', 'advanced_accordion']:
        component = ui_components.get(component_name)
        if component is not None and hasattr(component, 'disabled'):
            component.disabled = disable
    
    # Disable children dari container widgets
    for component_name in ['preprocess_options', 'validation_options']:
        component = ui_components.get(component_name)
        if component is not None and hasattr(component, 'children'):
            for child in component.children:
                if hasattr(child, 'disabled'):
                    child.disabled = disable
    
    # Disable tombol-tombol
    for btn in ['save_button', 'reset_button', 'cleanup_button']:
        if btn in ui_components:
            ui_components[btn].disabled = disable

def cleanup_ui(ui_components: Dict[str, Any]) -> None:
    """
    Kembalikan UI ke kondisi awal setelah preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Enable kembali semua UI component
    disable_ui_during_processing(ui_components, False)
    
    # Kembalikan tampilan tombol
    ui_components['preprocess_button'].layout.display = 'block'
    ui_components['stop_button'].layout.display = 'none'
    
    # Reset progress bar
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar']()
    else:
        # Fallback jika fungsi reset tidak tersedia
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
    
    # Buka log accordion secara default
    if 'log_accordion' in ui_components:
        ui_components['log_accordion'].selected_index = 0

def get_dataset_manager(ui_components: Dict[str, Any], config: Dict[str, Any] = None, logger=None):
    """
    Dapatkan dataset manager dengan fallback yang terstandarisasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        logger: Logger untuk logging
        
    Returns:
        Dataset manager instance
    """
    # Cek apakah sudah ada di ui_components
    if 'dataset_manager' in ui_components and ui_components['dataset_manager']:
        return ui_components['dataset_manager']
    
    # Gunakan service_handler untuk konsistensi
    try:
        from smartcash.ui.dataset.preprocessing.handlers.service_handler import get_dataset_manager as get_manager
        dataset_manager = get_manager(ui_components, config, logger)
        
        # Register progress callback jika tersedia
        try:
            if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
                ui_components['register_progress_callback'](dataset_manager)
        except Exception:
            pass
        
        # Simpan ke ui_components
        ui_components['dataset_manager'] = dataset_manager
        
        return dataset_manager
    except Exception as e:
        if logger: logger.error(f"{ICONS['error']} Gagal membuat dataset manager: {str(e)}")
        return None