"""
File: smartcash/ui/dataset/preprocessing_click_handler.py
Deskripsi: Handler tombol dan interaksi UI untuk preprocessing dataset dengan integrasi progress tracking yang ditingkatkan
"""

from typing import Dict, Any
from IPython.display import display, clear_output
import ipywidgets as widgets

def setup_click_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk tombol UI preprocessing dengan integrasi progress tracking yang ditingkatkan."""
    
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS
    
    # Penanganan error dengan decorator
    from smartcash.ui.handlers.error_handler import try_except_decorator
    
    @try_except_decorator(ui_components.get('status'))
    def on_preprocess_click(b):
        """Handler tombol preprocessing dengan konfigurasi progress steps yang lebih akurat."""
        # Dapatkan split dari UI
        split_option = ui_components['split_selector'].value if 'split_selector' in ui_components else 'All Splits'
        split_map = {'All Splits': None, 'Train Only': 'train', 'Validation Only': 'valid', 'Test Only': 'test'}
        split = split_map.get(split_option)
        
        # Persiapkan preprocessing dengan utilitas UI standar
        from smartcash.ui.dataset.preprocessing_initialization import update_status_panel
        from smartcash.ui.utils.alert_utils import create_status_indicator

        # Update UI untuk menunjukkan proses dimulai
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai preprocessing dataset..."))
        
        # Tampilkan log panel
        ui_components['log_accordion'].selected_index = 0  # Expand log
        
        # Update UI: sembunyikan tombol preprocess, tampilkan tombol stop
        ui_components['preprocess_button'].layout.display = 'none'
        ui_components['stop_button'].layout.display = 'block'
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['current_progress'].layout.visibility = 'visible'
        
        # Update konfigurasi dari UI
        try:
            from smartcash.ui.dataset.preprocessing_config_handler import update_config_from_ui, save_preprocessing_config
            updated_config = update_config_from_ui(ui_components, config)
            save_preprocessing_config(updated_config)
            if logger: logger.info(f"{ICONS['success']} Konfigurasi preprocessing berhasil disimpan")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Update status panel
        update_status_panel(ui_components, "info", f"{ICONS['processing']} Preprocessing dataset {split or 'All Splits'}...")
        
        # Notifikasi observer tentang mulai preprocessing
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(
                event_type=EventTopics.PREPROCESSING_START,
                sender="preprocessing_handler",
                message=f"Memulai preprocessing dataset {split or 'All Splits'}"
            )
        except ImportError:
            pass
        
        # Dapatkan dataset manager
        dataset_manager = ui_components.get('dataset_manager')
        if not dataset_manager:
            with ui_components['status']:
                display(create_status_indicator("error", f"{ICONS['error']} Dataset Manager tidak tersedia"))
            cleanup_ui()
            return
        
        # Dapatkan opsi preprocessing dari UI
        normalize = ui_components['preprocess_options'].children[1].value if len(ui_components['preprocess_options'].children) > 1 else True
        preserve_aspect_ratio = ui_components['preprocess_options'].children[2].value if len(ui_components['preprocess_options'].children) > 2 else True
        
        # Konfigurasi langkah-langkah progress tracking
        if 'configure_progress_steps' in ui_components and callable(ui_components['configure_progress_steps']):
            # Tentukan langkah-langkah preprocessing
            steps = 3  # Persiapan, Pemrosesan, Finalisasi
            
            # Konfigurasi bobot setiap langkah (total harus 1.0)
            step_weights = [0.1, 0.8, 0.1]  # 10% Persiapan, 80% Pemrosesan, 10% Finalisasi
            
            # Nama langkah untuk display
            step_names = {
                0: "Persiapan dataset",
                1: "Preprocessing gambar",
                2: "Penyimpanan hasil"
            }
            
            # Konfigurasi steps di progress handler
            ui_components['configure_progress_steps'](steps, step_weights, step_names)
        
        # Register progress callback jika tersedia
        if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
            ui_components['register_progress_callback'](dataset_manager)
        
        # Tandai preprocessing sedang berjalan
        ui_components['preprocessing_running'] = True
        
        # Advance ke langkah pertama (persiapan)
        if 'advance_to_step' in ui_components and callable(ui_components['advance_to_step']):
            ui_components['advance_to_step'](0, "Persiapan dataset", "Mempersiapkan dataset untuk preprocessing...")
        
        # Jalankan preprocessing - Gunakan path dari UI components yang sudah diupdate
        try:
            # Dapatkan path dari UI components
            data_dir = ui_components.get('data_dir')
            preprocessed_dir = ui_components.get('preprocessed_dir')
            
            # Update konfigurasi dataset manager dengan path yang benar
            if hasattr(dataset_manager, 'config'):
                if data_dir: dataset_manager.config['dataset_dir'] = data_dir
                
                # Update konfigurasi preprocessing jika ada
                if preprocessed_dir and 'preprocessing' in dataset_manager.config:
                    dataset_manager.config['preprocessing']['output_dir'] = preprocessed_dir
            
            # Update konfigurasi preproc jika ada
            if hasattr(dataset_manager, 'preprocess_config'):
                if preprocessed_dir: dataset_manager.preprocess_config['preprocessed_dir'] = preprocessed_dir
                if data_dir: dataset_manager.preprocess_config['raw_dataset_dir'] = data_dir
            
            # Advance ke langkah pemrosesan gambar
            if 'advance_to_step' in ui_components and callable(ui_components['advance_to_step']):
                ui_components['advance_to_step'](1, "Preprocessing gambar", "Mulai memproses gambar...")
            
            # Jalankan preprocessing dengan parameter yang benar - tidak perlu lagi mengirim img_size
            preprocess_result = dataset_manager.preprocess_dataset(
                split=split, 
                force_reprocess=True,
                normalize=normalize,
                preserve_aspect_ratio=preserve_aspect_ratio
            )
            
            # Advance ke langkah finalisasi
            if 'advance_to_step' in ui_components and callable(ui_components['advance_to_step']):
                ui_components['advance_to_step'](2, "Finalisasi hasil", "Menyimpan hasil dan membersihkan resources...")
            
            # Setelah selesai, update UI dengan status sukses
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("success", f"Preprocessing dataset selesai"))
            
            # Update summary jika function tersedia
            if 'update_summary' in ui_components and callable(ui_components['update_summary']):
                ui_components['update_summary'](preprocess_result)
            
            # Update status panel dan UI elements
            update_status_panel(ui_components, "success", f"Preprocessing dataset berhasil")
            
            # Tampilkan tombol visualisasi dan cleanup
            if 'visualization_buttons' in ui_components:
                ui_components['visualization_buttons'].layout.display = 'flex'
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'block'
            
            # Notifikasi observer tentang selesai preprocessing
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.PREPROCESSING_END,
                    sender="preprocessing_handler",
                    message=f"Preprocessing dataset {split or 'All Splits'} selesai"
                )
            except ImportError:
                pass
            
        except Exception as e:
            # Handle error
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
                    message=f"Error saat preprocessing: {str(e)}"
                )
            except ImportError:
                pass
            
            # Log error
            if logger: logger.error(f"{ICONS['error']} Error saat preprocessing dataset: {str(e)}")
        
        finally:
            # Tandai preprocessing selesai
            ui_components['preprocessing_running'] = False
            
            # Restore UI
            cleanup_ui()
    
    # Handler untuk tombol stop
    def on_stop_click(b):
        """Handler untuk menghentikan preprocessing."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        from smartcash.ui.dataset.preprocessing_initialization import update_status_panel
        
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
                message=f"Preprocessing dihentikan oleh pengguna"
            )
        except ImportError:
            pass
        
        # Reset UI
        cleanup_ui()
    
    # Function untuk cleanup UI setelah preprocessing
    def cleanup_ui():
        """Kembalikan UI ke kondisi awal setelah preprocessing."""
        ui_components['preprocess_button'].layout.display = 'block'
        ui_components['stop_button'].layout.display = 'none'
        
        # Reset progress bar
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        else:
            # Fallback jika fungsi reset tidak tersedia
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].layout.visibility = 'hidden'
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].layout.visibility = 'hidden'
    
    # Register handlers untuk tombol-tombol
    if 'preprocess_button' in ui_components:
        ui_components['preprocess_button'].on_click(on_preprocess_click)
    
    if 'stop_button' in ui_components:
        ui_components['stop_button'].on_click(on_stop_click)
    
    # Tambahkan referensi ke handlers di ui_components
    ui_components.update({
        'on_preprocess_click': on_preprocess_click,
        'on_stop_click': on_stop_click,
        'cleanup_ui': cleanup_ui,
        'preprocessing_running': False
    })
    
    return ui_components