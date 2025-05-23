"""
File: smartcash/ui/dataset/preprocessing/handlers/main_handler.py
Deskripsi: Handler utama untuk operasi preprocessing dengan integrasi backend service dan confirmation dialog
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.common.environment import get_environment_manager
from smartcash.dataset.services.preprocessing_manager import PreprocessingManager
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog


def setup_main_handler(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handler untuk operasi preprocessing utama."""
    logger = ui_components.get('logger')
    env_manager = env or get_environment_manager()
    
    # Initialize preprocessing manager
    ui_components['preprocessing_manager'] = None
    ui_components['processing_running'] = False
    ui_components['stop_requested'] = False
    
    def get_preprocessing_config() -> Dict[str, Any]:
        """Ambil konfigurasi preprocessing dari UI components."""
        return {
            'img_size': _parse_resolution(ui_components['resolution_dropdown'].value),
            'normalization': ui_components['normalization_dropdown'].value,
            'num_workers': ui_components['worker_slider'].value,
            'split': ui_components['split_dropdown'].value,
            'raw_dataset_dir': ui_components.get('data_dir', 'data'),
            'preprocessed_dir': ui_components.get('preprocessed_dir', 'data/preprocessed'),
            'normalize': ui_components['normalization_dropdown'].value != 'none',
            'preserve_aspect_ratio': True
        }
    
    def _get_preprocessing_manager() -> PreprocessingManager:
        """Dapatkan preprocessing manager dengan lazy initialization."""
        if not ui_components['preprocessing_manager']:
            config = {
                'preprocessing': get_preprocessing_config(),
                'data': {'dir': ui_components.get('data_dir', 'data')}
            }
            ui_components['preprocessing_manager'] = PreprocessingManager(config, logger)
            
            # Register progress callback
            def progress_callback(**kwargs):
                _update_progress_ui(**kwargs)
            
            ui_components['preprocessing_manager'].register_progress_callback(progress_callback)
        
        return ui_components['preprocessing_manager']
    
    def _update_progress_ui(**kwargs):
        """Update progress UI berdasarkan callback dari preprocessing manager."""
        try:
            progress = kwargs.get('progress', 0)
            total = kwargs.get('total', 100)
            message = kwargs.get('message', 'Processing...')
            status = kwargs.get('status', 'info')
            step = kwargs.get('step', 0)
            
            # Show progress container
            if 'progress_helpers' in ui_components:
                ui_components['progress_helpers']['show_container']()
            
            # Update overall progress
            if 'progress_bar' in ui_components and total > 0:
                ui_components['progress_bar'].value = min((progress / total) * 100, 100)
            
            # Update overall label
            if 'overall_label' in ui_components:
                ui_components['overall_label'].value = f"Progress: {progress}/{total} - {message}"
            
            # Update step progress
            step_name = kwargs.get('split_step', kwargs.get('step_name', ''))
            if 'step_label' in ui_components and step_name:
                ui_components['step_label'].value = f"Langkah: {step_name}"
            
            # Update current split progress
            current_progress = kwargs.get('current_progress', 0)
            current_total = kwargs.get('current_total', 0)
            if current_total > 0 and 'current_progress' in ui_components:
                ui_components['current_progress'].value = min((current_progress / current_total) * 100, 100)
            
            # Log ke UI output saja (tidak ke console)
            if logger and message.strip():
                if status == 'success':
                    logger.success(message)
                elif status == 'error':
                    logger.error(message)
                elif status == 'warning':
                    logger.warning(message)
                else:
                    logger.info(message)
                    
        except Exception:
            # Silent fail untuk prevent console logs
            pass
    
    def _parse_resolution(resolution_str: str) -> tuple:
        """Parse resolution string ke tuple."""
        try:
            width, height = resolution_str.split('x')
            return (int(width), int(height))
        except:
            return (640, 640)
    
    def _check_dataset_exists() -> tuple[bool, str]:
        """Check apakah dataset mentah exists."""
        data_dir = Path(ui_components.get('data_dir', 'data'))
        
        if not data_dir.exists():
            return False, f"Direktori dataset tidak ditemukan: {data_dir}"
        
        # Check splits
        splits = ['train', 'val', 'test']
        missing_splits = []
        
        for split in splits:
            split_dir = data_dir / split
            if not split_dir.exists():
                missing_splits.append(split)
                continue
            
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                missing_splits.append(f"{split} (missing images/labels)")
        
        if missing_splits:
            return False, f"Split tidak lengkap: {', '.join(missing_splits)}"
        
        return True, "Dataset tersedia"
    
    def _on_preprocess_click(b):
        """Handler untuk tombol preprocessing."""
        if ui_components['processing_running']:
            logger.warning("⚠️ Preprocessing sedang berjalan")
            return
        
        # Check dataset exists
        dataset_exists, dataset_msg = _check_dataset_exists()
        if not dataset_exists:
            logger.error(f"❌ {dataset_msg}")
            return
        
        # Check preprocessed exists
        preprocessed_dir = Path(ui_components.get('preprocessed_dir', 'data/preprocessed'))
        split_config = ui_components['split_dropdown'].value
        
        existing_data = []
        if split_config == 'all':
            for split in ['train', 'val', 'test']:
                split_path = preprocessed_dir / split
                if split_path.exists() and len(list(split_path.glob('**/*.jpg'))) > 0:
                    existing_data.append(split)
        else:
            split_path = preprocessed_dir / split_config
            if split_path.exists() and len(list(split_path.glob('**/*.jpg'))) > 0:
                existing_data.append(split_config)
        
        if existing_data:
            # Show confirmation dialog
            existing_str = ', '.join(existing_data)
            confirmation_msg = f"""Data preprocessing sudah ada untuk: {existing_str}
            
Apakah Anda ingin memproses ulang?
Data yang ada akan ditimpa."""
            
            dialog = create_confirmation_dialog(
                title="Konfirmasi Preprocessing",
                message=confirmation_msg,
                on_confirm=lambda b: _start_preprocessing_confirmed(),
                on_cancel=lambda b: _hide_dialog(),
                confirm_text="Ya, Proses Ulang",
                cancel_text="Batal"
            )
            
            ui_components['confirmation_dialog'] = dialog
            _show_dialog(dialog)
        else:
            _start_preprocessing_confirmed()
    
    def _start_preprocessing_confirmed():
        """Mulai preprocessing setelah konfirmasi."""
        if 'confirmation_dialog' in ui_components:
            _hide_dialog()
        
        ui_components['processing_running'] = True
        ui_components['stop_requested'] = False
        
        # Disable button dan update status
        ui_components['preprocess_button'].disabled = True
        ui_components['preprocess_button'].description = "Processing..."
        
        if logger:
            logger.info("🚀 Memulai preprocessing dataset")
        
        # Update status panel
        from smartcash.ui.components.status_panel import update_status_panel
        update_status_panel(ui_components['status_panel'], "Memulai preprocessing dataset...", "info")
        
        # Start preprocessing dalam thread
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_run_preprocessing)
        ui_components['preprocessing_future'] = future
        ui_components['preprocessing_executor'] = executor
        
        # Monitor completion
        def check_completion():
            if future.done():
                try:
                    result = future.result()
                    _on_preprocessing_complete(result)
                except Exception as e:
                    _on_preprocessing_error(e)
                finally:
                    executor.shutdown(wait=False)
            else:
                # Check again in 1 second
                import threading
                threading.Timer(1.0, check_completion).start()
        
        check_completion()
    
    def _run_preprocessing():
        """Jalankan preprocessing dalam thread terpisah."""
        try:
            preprocessing_manager = _get_preprocessing_manager()
            config = get_preprocessing_config()
            
            result = preprocessing_manager.preprocess_dataset(
                split=config['split'],
                force_reprocess=True,
                show_progress=True,
                **config
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing: {str(e)}")
            raise
    
    def _on_preprocessing_complete(result):
        """Handler saat preprocessing selesai."""
        ui_components['processing_running'] = False
        ui_components['preprocess_button'].disabled = False
        ui_components['preprocess_button'].description = "Mulai Preprocessing"
        
        if result and result.get('success', False):
            total_images = result.get('total_images', 0)
            processing_time = result.get('processing_time', 0)
            
            success_msg = f"✅ Preprocessing selesai: {total_images} gambar dalam {processing_time:.1f} detik"
            logger.success(success_msg)
            
            update_status_panel(ui_components['status_panel'], success_msg, "success")
        else:
            error_msg = "❌ Preprocessing gagal"
            logger.error(error_msg)
            update_status_panel(ui_components['status_panel'], error_msg, "error")
        
        # Reset progress
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
        if 'overall_label' in ui_components:
            ui_components['overall_label'].value = "Siap memulai preprocessing"
    
    def _on_preprocessing_error(error):
        """Handler saat preprocessing error."""
        ui_components['processing_running'] = False
        ui_components['preprocess_button'].disabled = False
        ui_components['preprocess_button'].description = "Mulai Preprocessing"
        
        error_msg = f"❌ Error preprocessing: {str(error)}"
        logger.error(error_msg)
        update_status_panel(ui_components['status_panel'], error_msg, "error")
        
        # Reset progress
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
    
    def _show_dialog(dialog):
        """Show confirmation dialog."""
        from IPython.display import display
        display(dialog)
    
    def _hide_dialog():
        """Hide confirmation dialog."""
        if 'confirmation_dialog' in ui_components:
            try:
                ui_components['confirmation_dialog'].close()
            except:
                pass
            del ui_components['confirmation_dialog']
    
    # Setup event handlers
    ui_components['preprocess_button'].on_click(_on_preprocess_click)
    
    if logger:
        logger.debug("✅ Main handler preprocessing setup selesai")
    
    return ui_components