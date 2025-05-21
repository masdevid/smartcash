"""
File: smartcash/ui/dataset/preprocessing/handlers/executor.py
Deskripsi: Executor untuk menjalankan proses preprocessing dataset dengan perbaikan tombol stop
"""

from typing import Dict, Any
import time

from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import (
    update_ui_before_preprocessing, update_ui_state, reset_ui_after_preprocessing, set_preprocessing_state
)
from smartcash.ui.dataset.preprocessing.utils.progress_manager import (
    update_progress, reset_progress_bar, start_progress
)

def execute_preprocessing(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Eksekusi preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi preprocessing
    """
    # Update UI status
    update_ui_before_preprocessing(ui_components)
    
    # Set flag bahwa preprocessing sedang berjalan
    set_preprocessing_state(ui_components, True)
    
    # Tampilkan progress bar
    start_progress(ui_components, "Mempersiapkan preprocessing dataset...")
    
    try:
        # Import komponen yang dibutuhkan
        from smartcash.common.config import get_config_manager
        from smartcash.dataset.services.preprocessor.preprocessing_service import PreprocessingService
        from smartcash.ui.dataset.preprocessing.utils.notification_manager import get_notification_manager
        from smartcash.ui.utils.constants import ICONS
        
        # Get config manager
        config_manager = get_config_manager()
        
        # Get notification manager
        notification_manager = get_notification_manager(ui_components)
        
        # Dapatkan parameter
        split = config.get('split', 'all')
        normalization = config.get('normalization', 'minmax')
        augmentation = config.get('augmentation', False)
        num_workers = config.get('num_workers', 4)
        
        # Dapatkan resolusi
        resolution = config.get('resolution')
        if isinstance(resolution, tuple) and len(resolution) == 2:
            img_size = resolution
        elif isinstance(resolution, str) and 'x' in resolution:
            try:
                width, height = map(int, resolution.split('x'))
                img_size = (width, height)
            except ValueError:
                img_size = (640, 640) # Default size
        else:
            img_size = (640, 640) # Default size
        
        # Siapkan config untuk DatasetPreprocessor
        preprocessor_config = {
            'preprocessing': {
                'img_size': img_size,
                'normalize': normalization != 'none',
                'normalization': normalization,
                'augmentation': augmentation,
                'num_workers': num_workers, # Untuk Google Colab sebaiknya gunakan 1 worker karena tidak multi-threading
                'output_dir': config.get('preprocessed_dir', 'data/preprocessed')
            },
            'data': {
                'dir': config.get('data_dir', 'data')
            }
        }
        
        # Log parameters
        log_message(ui_components, f"Preprocessing dataset dengan resolusi {img_size}, normalisasi {normalization}, augmentasi {augmentation}, split {split}, workers {num_workers}", "info", "🔄")
        
        # Tampilkan confirmation area jika tersedia
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'layout'):
            ui_components['confirmation_area'].layout.display = 'block'
            ui_components['confirmation_area'].clear_output(wait=True)
            with ui_components['confirmation_area']:
                from IPython.display import display, HTML
                display(HTML(f"""
                <div style='padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;'>
                    <h3 style='margin-top: 0;'>{ICONS.get('info', 'ℹ️')} Preprocessing sedang berjalan</h3>
                    <p>Memproses dataset dengan konfigurasi:</p>
                    <ul>
                        <li><b>Resolusi:</b> {img_size[0]}x{img_size[1]}</li>
                        <li><b>Normalisasi:</b> {normalization}</li>
                        <li><b>Split:</b> {split}</li>
                        <li><b>Workers:</b> {num_workers}</li>
                        <li><b>Output:</b> {config.get('preprocessed_dir', 'data/preprocessed')}</li>
                    </ul>
                    <p>Perkiraan waktu: <i>Tergantung ukuran dataset</i></p>
                </div>
                """))
        
        # Notify process start menggunakan notification manager
        notification_manager.notify_process_start("preprocessing", f"split: {split}", split)
        
        # Update progress
        update_progress(ui_components, 10, 100, "Memulai preprocessing dataset...", f"Menganalisis dataset untuk split: {split}")
        
        # Setup observer manager jika belum ada di ui_components
        if 'observer_manager' not in ui_components:
            from smartcash.ui.dataset.preprocessing.utils.ui_observers import create_observer_manager
            ui_components['observer_manager'] = create_observer_manager(ui_components)
            ui_components['observer_manager'].notification_manager = notification_manager
        
        # Buat preprocessing service dengan observer_manager dari UI
        preprocessing_service = PreprocessingService(
            config=preprocessor_config,
            logger=ui_components.get('logger'),
            observer_manager=ui_components.get('observer_manager')
        )
        
        # Dapatkan preprocessor
        preprocessor = preprocessing_service.preprocessor
        
        # Buat dan register progress callback untuk preprocessing service
        def progress_callback(**kwargs):
            # Gunakan notification manager untuk menangani progress
            try:
                notification_manager.notify_progress(**kwargs)
                
                # Cek jika stop_requested
                if ui_components.get('stop_requested', False):
                    return False  # Mengirim signal stop ke processor
                    
                return True  # Lanjutkan processing
            except Exception as e:
                log_message(ui_components, f"Error saat update progress: {str(e)}", "error", "❌")
                return True  # Default tetap lanjutkan
        
        # Register progress callback ke preprocessor
        try:
            preprocessor.register_progress_callback(progress_callback)
            log_message(ui_components, "Progress callback berhasil diregister ke preprocessor", "debug", "🔄")
        except Exception as e:
            log_message(ui_components, f"Gagal register callback: {str(e)}", "warning", "⚠️")
        
        # Jalankan preprocessing
        result = preprocessing_service.preprocess_dataset(
            split=split,
            force_reprocess=config.get('force_reprocess', False),
            show_progress=True,
            normalize=normalization != 'none',
            preserve_aspect_ratio=config.get('preserve_aspect_ratio', True)
        )
        
        # Log hasil preprocessing
        if result:
            total_processed = result.get('processed', 0)
            total_skipped = result.get('skipped', 0)
            total_failed = result.get('failed', 0)
            success_message = f"Preprocessing selesai: {total_processed} gambar diproses, {total_skipped} dilewati, {total_failed} gagal"
            log_message(ui_components, success_message, "success", "✅")
            
            # Update confirmation area dengan hasil
            if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
                ui_components['confirmation_area'].clear_output(wait=True)
                with ui_components['confirmation_area']:
                    from IPython.display import display, HTML
                    display(HTML(f"""
                    <div style='padding: 10px; background-color: #f0fff0; border-left: 4px solid #28a745;'>
                        <h3 style='margin-top: 0;'>{ICONS.get('success', '✅')} Preprocessing Selesai</h3>
                        <p>Hasil preprocessing:</p>
                        <ul>
                            <li><b>Diproses:</b> {total_processed} gambar</li>
                            <li><b>Dilewati:</b> {total_skipped} gambar</li>
                            <li><b>Gagal:</b> {total_failed} gambar</li>
                        </ul>
                        <p><b>Output:</b> {config.get('preprocessed_dir', 'data/preprocessed')}</p>
                    </div>
                    """))
            
            # Notifikasi selesai
            notification_manager.notify_process_complete(result, f"split {split}")
        
        # Reset UI setelah selesai
        reset_ui_after_preprocessing(ui_components)
        
    except Exception as e:
        # Log error
        error_message = f"Error saat preprocessing dataset: {str(e)}"
        log_message(ui_components, error_message, "error", "❌")
        
        # Update UI state
        update_ui_state(ui_components, "error", error_message)
        
        # Reset progress
        reset_progress_bar(ui_components)
        
        # Reset UI components
        reset_ui_after_preprocessing(ui_components)
        
        # Notify error dengan notification manager
        try:
            notification_manager = get_notification_manager(ui_components)
            notification_manager.notify_process_error(error_message)
        except Exception:
            # Fallback jika notification manager tidak tersedia
            update_ui_state(ui_components, "error", error_message)