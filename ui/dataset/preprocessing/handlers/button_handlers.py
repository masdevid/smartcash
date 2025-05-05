"""
File: smartcash/ui/dataset/preprocessing/handlers/button_handlers.py
Deskripsi: Handler tombol untuk preprocessing dataset dengan perbaikan PosixPath error
"""

from typing import Dict, Any
from IPython.display import display, clear_output
import os
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.preprocessing.handlers.status_handler import update_status_panel

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
        # Dapatkan split dari UI
        split_option = ui_components['split_selector'].value
        split_map = {'All Splits': None, 'Train Only': 'train', 'Validation Only': 'valid', 'Test Only': 'test'}
        split = split_map.get(split_option)
        
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
        
        # Update konfigurasi dari UI dan simpan
        try:
            updated_config = ui_components['update_config_from_ui'](ui_components, config)
            ui_components['save_preprocessing_config'](updated_config)
            if logger: logger.info(f"{ICONS['success']} Konfigurasi preprocessing berhasil disimpan")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Tandai preprocessing sedang berjalan
        ui_components['preprocessing_running'] = True
        
        # Update status panel dengan informasi awal
        split_info = f"Split {split}" if split else "Semua split"
        update_status_panel(ui_components, "info", f"{ICONS['processing']} Memulai preprocessing {split_info}...")
        
        # PERBAIKAN: Cek apakah path identik untuk menghindari masalah symlink
        data_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        if os.path.realpath(data_dir) == os.path.realpath(preprocessed_dir):
            error_msg = f"Path data input dan output sama: {data_dir}, ini akan menyebabkan masalah"
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS['error']} {error_msg}"))
            update_status_panel(ui_components, "error", f"{ICONS['error']} {error_msg}")
            cleanup_ui(ui_components)
            return
        
        # Notifikasi observer tentang mulai preprocessing
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(event_type=EventTopics.PREPROCESSING_START, sender="preprocessing_handler", 
                  message=f"Memulai preprocessing dataset {split_info}", split=split, split_info=split_info)
        except ImportError: 
            pass
        
        # Dapatkan dataset manager
        dataset_manager = get_dataset_manager(ui_components, config, logger)
        if not dataset_manager:
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS['error']} Dataset Manager tidak tersedia"))
            cleanup_ui(ui_components)
            return
        
        # Dapatkan opsi preprocessing dari UI dengan one-liner
        normalize, preserve_aspect_ratio = [ui_components['preprocess_options'].children[i].value for i in [1, 2]]
        
        # Jalankan preprocessing
        try:
            # Update konfigurasi dataset manager dengan path yang benar
            if hasattr(dataset_manager, 'config'): 
                dataset_manager.config['dataset_dir'] = data_dir
                if 'preprocessing' in dataset_manager.config:
                    dataset_manager.config['preprocessing']['output_dir'] = preprocessed_dir
            
            # Update konfigurasi preproc jika ada
            if hasattr(dataset_manager, 'preprocess_config'):
                dataset_manager.preprocess_config.preprocessed_dir = preprocessed_dir
                dataset_manager.preprocess_config.raw_dataset_dir = data_dir
            
            # Log awal preprocessing
            if logger: logger.info(f"{ICONS['start']} Memulai preprocessing {split_info}")
            
            # PERBAIKAN: Pastikan data_dir dan preprocessed_dir adalah string, bukan PosixPath
            data_dir_str = str(data_dir) if isinstance(data_dir, Path) else data_dir
            preprocessed_dir_str = str(preprocessed_dir) if isinstance(preprocessed_dir, Path) else preprocessed_dir
            
            # Jalankan preprocessing dengan parameter yang ditingkatkan
            preprocess_result = dataset_manager.preprocess_dataset(
                split=split, 
                force_reprocess=True,
                normalize=normalize,
                preserve_aspect_ratio=preserve_aspect_ratio,
                # PERBAIKAN: Jika dataset_manager mendukung parameter berikut, tambahkan secara eksplisit
                raw_dataset_dir=data_dir_str,
                preprocessed_dir=preprocessed_dir_str
            )
            
            # Tambahkan path output jika tidak ada
            if 'output_dir' not in preprocess_result: 
                preprocess_result['output_dir'] = preprocessed_dir
            
            # Setelah selesai, update UI dengan status sukses
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("success", f"{ICONS['success']} Preprocessing {split_info} selesai"))
            
            # Update status panel
            update_status_panel(ui_components, "success", 
                               f"{ICONS['success']} Preprocessing dataset berhasil diselesaikan")
            
            # Update UI state - tampilkan summary dan visualisasi
            for component in ['visualization_container', 'summary_container']:
                if component in ui_components:
                    ui_components[component].layout.display = 'block'
            
            # Tampilkan tombol visualisasi dan cleanup
            ui_components['visualization_buttons'].layout.display = 'flex'
            ui_components['cleanup_button'].layout.display = 'block'
            
            # Update summary dengan hasil preprocessing
            from smartcash.ui.dataset.preprocessing.handlers.state_handler import generate_preprocessing_summary
            generate_preprocessing_summary(ui_components, preprocessed_dir)
            
            # Notifikasi observer tentang selesai
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.PREPROCESSING_END,
                    sender="preprocessing_handler",
                    message=f"Preprocessing dataset {split_info} selesai",
                    result=preprocess_result,
                    duration=preprocess_result.get('processing_time', 0),
                    total_images=preprocess_result.get('total_images', 0)
                )
            except ImportError:
                pass
            
        except Exception as e:
            # Handle error dengan notifikasi
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
            
            # Update status panel
            update_status_panel(ui_components, "error", f"{ICONS['error']} Preprocessing gagal: {str(e)}")
            
            # Notifikasi observer tentang error
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.PREPROCESSING_ERROR,
                    sender="preprocessing_handler",
                    message=f"Error saat preprocessing: {str(e)}",
                    error=str(e)
                )
            except ImportError:
                pass
            
            # Log error
            if logger: logger.error(f"{ICONS['error']} Error saat preprocessing dataset: {str(e)}")
        
        finally:
            # Tandai preprocessing selesai
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
        
        # Load konfigurasi default dan update UI
        try:
            default_config = ui_components['load_preprocessing_config']()
            ui_components['update_ui_from_config'](ui_components, default_config)
            
            # Deteksi state preprocessing
            from smartcash.ui.dataset.preprocessing.handlers.state_handler import detect_preprocessing_state
            detect_preprocessing_state(ui_components)
            
            # Tampilkan pesan sukses
            with ui_components['status']: 
                display(create_status_indicator("success", f"{ICONS['success']} UI dan konfigurasi berhasil direset ke nilai default"))
            
            # Update status panel
            update_status_panel(ui_components, "success", f"{ICONS['success']} Konfigurasi direset ke nilai default")
            
            # Log success
            if logger: logger.success(f"{ICONS['success']} Konfigurasi preprocessor berhasil direset ke nilai default")
        except Exception as e:
            # Jika gagal reset konfigurasi, tampilkan pesan error
            with ui_components['status']: 
                display(create_status_indicator("warning", f"{ICONS['warning']} Reset sebagian: {str(e)}"))
            
            # Update status panel
            update_status_panel(ui_components, "warning", f"{ICONS['warning']} Reset UI sebagian: {str(e)}")
            
            if logger: logger.warning(f"{ICONS['warning']} Error saat reset konfigurasi: {str(e)}")
    
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
    
    # Reset accordion
    if 'log_accordion' in ui_components:
        ui_components['log_accordion'].selected_index = None

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
    
    # Gunakan fallback_utils untuk konsistensi
    try:
        from smartcash.ui.utils.fallback_utils import get_dataset_manager
        dataset_manager = get_dataset_manager(config, logger)
        
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