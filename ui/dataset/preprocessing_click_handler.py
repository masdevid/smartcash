"""
File: smartcash/ui/dataset/preprocessing_click_handler.py
Deskripsi: Handler tombol dan interaksi UI untuk preprocessing dataset dengan button standar dan pendekatan DRY
"""

from typing import Dict, Any
from IPython.display import display, clear_output
import ipywidgets as widgets
import os
from pathlib import Path

def setup_click_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk tombol UI preprocessing dengan sistem progress yang ditingkatkan."""
    
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS
    
    # Penanganan error dengan decorator standar
    from smartcash.ui.handlers.error_handler import try_except_decorator
    
    @try_except_decorator(ui_components.get('status'))
    def on_preprocess_click(b):
        """Handler tombol preprocessing dengan dukungan progress tracking yang dioptimalkan."""
        # Dapatkan split dari UI
        split_option = ui_components['split_selector'].value
        split_map = {'All Splits': None, 'Train Only': 'train', 'Validation Only': 'valid', 'Test Only': 'test'}
        split = split_map.get(split_option)
        
        # Persiapkan preprocessing dengan utilitas UI standar
        from smartcash.ui.dataset.shared.status_panel import update_status_panel
        from smartcash.ui.utils.alert_utils import create_status_indicator

        # Update UI: menampilkan proses dimulai dengan one-liner
        with ui_components['status']: clear_output(wait=True); display(create_status_indicator("info", f"{ICONS['processing']} Memulai preprocessing dataset..."))
        
        # Tampilkan log panel dan progress bar
        ui_components['log_accordion'].selected_index = 0  # Expand log
        [setattr(ui_components[p_bar].layout, 'visibility', 'visible') for p_bar in ['progress_bar', 'current_progress'] if p_bar in ui_components]
        
        # Disable semua komponen UI dengan one-liner
        disable_ui_during_processing(ui_components, True)
        
        # Update UI tombol dengan komponen standar
        ui_components['preprocess_button'].layout.display = 'none'
        ui_components['stop_button'].layout.display = 'block'
        
        # Update konfigurasi dari UI dan simpan dengan one-liner
        try:
            from smartcash.ui.dataset.preprocessing_config_handler import update_config_from_ui, save_preprocessing_config
            updated_config = update_config_from_ui(ui_components, config); save_preprocessing_config(updated_config)
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
            with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} {error_msg}"))
            update_status_panel(ui_components, "error", f"{ICONS['error']} {error_msg}")
            cleanup_ui(); return
        
        # Notifikasi observer tentang mulai preprocessing dengan one-liner
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(event_type=EventTopics.PREPROCESSING_START, sender="preprocessing_handler", 
                  message=f"Memulai preprocessing dataset {split_info}", split=split, split_info=split_info)
        except ImportError: pass
        
        # Dapatkan dataset manager
        dataset_manager = ui_components.get('dataset_manager')
        if not dataset_manager:
            with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} Dataset Manager tidak tersedia"))
            cleanup_ui(); return
        
        # Dapatkan opsi preprocessing dari UI dengan one-liner
        normalize, preserve_aspect_ratio = [ui_components['preprocess_options'].children[i].value for i in [1, 2]]
        
        # Register progress callback jika tersedia
        if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
            ui_components['register_progress_callback'](dataset_manager)
        
        # Jalankan preprocessing
        try:
            # Dapatkan paths dari ui_components
            data_dir = ui_components.get('data_dir', 'data')
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            
            # Update konfigurasi dataset manager dengan path yang benar dengan one-liner
            if hasattr(dataset_manager, 'config'): 
                dataset_manager.config['dataset_dir'] = data_dir
                [dataset_manager.config['preprocessing'].update({k: v}) for k, v in {'output_dir': preprocessed_dir}.items() if 'preprocessing' in dataset_manager.config]
            
            # Update konfigurasi preproc jika ada dengan one-liner
            if hasattr(dataset_manager, 'preprocess_config'):
                [setattr(dataset_manager.preprocess_config, k, v) for k, v in {'preprocessed_dir': preprocessed_dir, 'raw_dataset_dir': data_dir}.items()]
            
            # Log awal preprocessing dengan pesan ringkas
            if logger: logger.info(f"{ICONS['start']} Memulai preprocessing {split_info}")
            
            # PERBAIKAN: Jalankan preprocessing dengan parameter yang ditingkatkan
            preprocess_result = dataset_manager.preprocess_dataset(
                split=split, 
                force_reprocess=True,
                normalize=normalize,
                preserve_aspect_ratio=preserve_aspect_ratio
            )
            
            # PERBAIKAN: Tambahkan path output jika tidak ada
            if 'output_dir' not in preprocess_result: preprocess_result['output_dir'] = preprocessed_dir
            
            # Setelah selesai, update UI dengan status sukses
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("success", f"{ICONS['success']} Preprocessing {split_info} selesai"))
            
            # Update summary jika function tersedia
            if 'update_summary' in ui_components and callable(ui_components['update_summary']):
                ui_components['update_summary'](preprocess_result)
            
            # Update status panel
            update_status_panel(ui_components, "success", f"{ICONS['success']} Preprocessing dataset berhasil diselesaikan")
            
            # Tampilkan tombol visualisasi dan cleanup dengan one-liner
            [setattr(ui_components[component].layout, 'display', 'flex' if component == 'visualization_buttons' else 'block') 
             for component in ['visualization_buttons', 'cleanup_button', 'visualization_container', 'summary_container'] if component in ui_components]
            
            # PERBAIKAN: Notifikasi observer dengan detail yang lebih lengkap
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
            except ImportError: pass
            
        except Exception as e:
            # Handle error dengan notifikasi yang lebih baik
            with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
            
            # Update status panel
            update_status_panel(ui_components, "error", f"{ICONS['error']} Preprocessing gagal: {str(e)}")
            
            # PERBAIKAN: Notifikasi observer dengan detail error
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.PREPROCESSING_ERROR,
                    sender="preprocessing_handler",
                    message=f"Error saat preprocessing: {str(e)}",
                    error=str(e)
                )
            except ImportError: pass
            
            # Log error
            if logger: logger.error(f"{ICONS['error']} Error saat preprocessing dataset: {str(e)}")
        
        finally:
            # Tandai preprocessing selesai
            ui_components['preprocessing_running'] = False
            
            # Restore UI
            cleanup_ui()
    
    # Handler untuk tombol stop dengan notifikasi yang ditingkatkan
    def on_stop_click(b):
        """Handler untuk menghentikan preprocessing."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        from smartcash.ui.dataset.shared.status_panel import update_status_panel
        
        # Set flag untuk menghentikan preprocessing
        ui_components['preprocessing_running'] = False
        
        # Tampilkan pesan di status
        with ui_components['status']: display(create_status_indicator("warning", f"{ICONS['warning']} Menghentikan preprocessing..."))
        
        # Update status panel
        update_status_panel(ui_components, "warning", f"{ICONS['warning']} Preprocessing dihentikan oleh pengguna")
        
        # PERBAIKAN: Notifikasi observer dengan status yang jelas
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(
                event_type=EventTopics.PREPROCESSING_END,
                sender="preprocessing_handler",
                message=f"Preprocessing dihentikan oleh pengguna",
                status="cancelled"
            )
        except ImportError: pass
        
        # Reset UI
        cleanup_ui()
    
    # Function untuk disable/enable UI components selama processing
    def disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True):
        """Enable/disable komponen UI saat preprocessing berjalan dengan one-liner."""
        # PERBAIKAN: Gunakan one-liner untuk disable semua komponen utama
        [setattr(component, 'disabled', disable) for component_name in ['preprocess_options', 'split_selector', 'advanced_accordion'] 
         for component in [ui_components.get(component_name)] if component is not None and hasattr(component, 'disabled')]
        
        # PERBAIKAN: Disable children dari container widgets dengan one-liner
        [[setattr(child, 'disabled', disable) for child in component.children if hasattr(child, 'disabled')] 
         for component_name in ['preprocess_options', 'split_selector'] 
         for component in [ui_components.get(component_name)] if component is not None and hasattr(component, 'children')]
        
        # Disable tombol-tombol dengan one-liner
        [setattr(ui_components[btn], 'disabled', disable) for btn in ['save_button', 'reset_button', 'cleanup_button'] if btn in ui_components]
    
    # Function untuk cleanup UI setelah preprocessing
    def cleanup_ui():
        """Kembalikan UI ke kondisi awal setelah preprocessing dengan pendekatan one-liner."""
        # Enable kembali semua UI component
        disable_ui_during_processing(ui_components, False)
        
        # Kembalikan tampilan tombol
        ui_components['preprocess_button'].layout.display = 'block'
        ui_components['stop_button'].layout.display = 'none'
        
        # Reset progress bar dengan one-liner
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        else:
            # Fallback jika fungsi reset tidak tersedia, gunakan one-liner
            [setattr(ui_components[p_bar].layout, 'visibility', 'hidden') for p_bar in ['progress_bar', 'current_progress'] if p_bar in ui_components]
            [setattr(ui_components[p_bar], 'value', 0) for p_bar in ['progress_bar', 'current_progress'] if p_bar in ui_components]
    
    # Reset handler untuk reset button dengan fungsionalitas yang ditingkatkan
    def on_reset_click(b):
        """Reset UI dan konfigurasi ke default dengan pendekatan DRY."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        from smartcash.ui.dataset.shared.status_panel import update_status_panel
        
        # Reset UI
        cleanup_ui()
        
        # Hide visualization dan summary dengan one-liner
        [setattr(ui_components[component].layout, 'display', 'none') for component in ['visualization_container', 'summary_container', 'visualization_buttons', 'cleanup_button'] if component in ui_components]
        
        # Reset logs
        if 'status' in ui_components: clear_output(wait=True)
        
        # PERBAIKAN: Reset konfigurasi ke default jika tersedia
        try:
            from smartcash.ui.dataset.preprocessing_config_handler import load_preprocessing_config, update_ui_from_config
            default_config = load_preprocessing_config(); update_ui_from_config(ui_components, default_config)
            
            # Re-detect state preprocessing
            from smartcash.ui.dataset.shared.setup_utils import detect_module_state
            detect_module_state(ui_components, 'preprocessing')
            
            # Tampilkan pesan sukses
            with ui_components['status']: display(create_status_indicator("success", f"{ICONS['success']} UI dan konfigurasi berhasil direset ke nilai default"))
            
            # Update status panel
            update_status_panel(ui_components, "success", f"{ICONS['success']} Konfigurasi direset ke nilai default")
            
            # Log success
            if logger: logger.success(f"{ICONS['success']} Konfigurasi preprocessor berhasil direset ke nilai default")
        except Exception as e:
            # Jika gagal reset konfigurasi, tampilkan pesan error
            with ui_components['status']: display(create_status_indicator("warning", f"{ICONS['warning']} Reset sebagian: {str(e)}"))
            
            # Update status panel
            update_status_panel(ui_components, "warning", f"{ICONS['warning']} Reset UI sebagian: {str(e)}")
            
            if logger: logger.warning(f"{ICONS['warning']} Error saat reset konfigurasi: {str(e)}")
    
    # Register handlers untuk tombol-tombol dengan one-liner
    [ui_components[button].on_click(handler) for button, handler in [('preprocess_button', on_preprocess_click), ('stop_button', on_stop_click), ('reset_button', on_reset_click)] if button in ui_components]
    
    # Tambahkan referensi ke handlers di ui_components
    ui_components.update({
        'on_preprocess_click': on_preprocess_click,
        'on_stop_click': on_stop_click,
        'on_reset_click': on_reset_click,
        'cleanup_ui': cleanup_ui,
        'disable_ui_during_processing': disable_ui_during_processing,
        'preprocessing_running': False
    })
    
    return ui_components