"""
File: smartcash/ui/dataset/preprocessing/handlers/main_handler.py
Deskripsi: Fixed handler utama dengan progress tracking 3-level yang konsisten
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.common.environment import get_environment_manager
from smartcash.dataset.services.preprocessing_manager import PreprocessingManager
from smartcash.dataset.utils.path_validator import get_path_validator
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.components.progress_tracking import (
    update_overall_progress, update_step_progress, update_current_progress
)


def setup_main_handler(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handler utama dengan fixed progress tracking."""
    logger = ui_components.get('logger')
    env_manager = env or get_environment_manager()
    path_validator = get_path_validator(logger)
    
    ui_components['preprocessing_manager'] = None
    ui_components['processing_running'] = False
    ui_components['stop_requested'] = False
    
    def get_preprocessing_config() -> Dict[str, Any]:
        """Ambil konfigurasi preprocessing dari UI."""
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
            ui_components['preprocessing_manager'].register_progress_callback(_update_progress_ui)
        
        return ui_components['preprocessing_manager']
    
    def _update_progress_ui(**kwargs):
        """Update progress UI dengan 3-level hierarchy yang konsisten."""
        try:
            # Show progress container
            if 'progress_components' in ui_components:
                ui_components['progress_components']['show_container']()
            
            # Extract progress data
            split_name = kwargs.get('split', kwargs.get('step_name', ''))
            split_step = kwargs.get('split_step', 0)
            total_splits = kwargs.get('total_splits', 3)  # train, valid, test
            
            # File progress dalam split
            file_progress = kwargs.get('current_progress', kwargs.get('progress', 0))
            file_total = kwargs.get('current_total', kwargs.get('total', 0))
            
            # Overall progress calculation
            overall_progress = kwargs.get('overall_progress', 0)
            overall_total = kwargs.get('overall_total', 100)
            message = kwargs.get('message', 'Processing...')
            
            # === UPDATE OVERALL PROGRESS (Keseluruhan preprocessing) ===
            if overall_progress > 0 or overall_total > 0:
                update_overall_progress(
                    ui_components['progress_components'],
                    overall_progress, overall_total, message
                )
            
            # === UPDATE STEP PROGRESS (Per split: train, valid, test) ===
            if split_step > 0 or total_splits > 0:
                step_message = f"Processing {split_name}" if split_name else "Processing"
                update_step_progress(
                    ui_components['progress_components'],
                    split_step, total_splits, step_message
                )
            
            # === UPDATE CURRENT PROGRESS (Per file dalam split) ===
            if file_total > 0:
                current_message = f"{split_name} files" if split_name else "Files"
                update_current_progress(
                    ui_components['progress_components'],
                    file_progress, file_total, current_message
                )
            
            # Log ke UI (tanpa console spam)
            if logger and message.strip():
                status = kwargs.get('status', 'info')
                if status == 'success':
                    logger.success(message)
                elif status == 'error':
                    logger.error(message)
                elif status == 'warning':
                    logger.warning(message)
                else:
                    logger.info(message)
                    
        except Exception:
            # Silent fail untuk prevent recursive errors
            pass
    
    def _parse_resolution(resolution_str: str) -> tuple:
        """Parse resolution string ke tuple."""
        try:
            width, height = resolution_str.split('x')
            return (int(width), int(height))
        except:
            return (640, 640)
    
    def _check_dataset_exists() -> tuple[bool, str]:
        """Check dataset dengan path validator."""
        data_dir = ui_components.get('data_dir', 'data')
        validation_result = path_validator.validate_dataset_structure(data_dir)
        
        if not validation_result['valid']:
            return False, f"Dataset tidak ditemukan: {data_dir}"
        
        # Check critical issues
        critical_issues = [i for i in validation_result['issues'] if '❌' in i]
        if critical_issues:
            return False, f"Critical issues: {', '.join(critical_issues)}"
        
        if validation_result['total_images'] == 0:
            return False, "Dataset kosong, tidak ada gambar ditemukan"
        
        return True, f"Dataset valid: {validation_result['total_images']} gambar"
    
    def _on_preprocess_click(b):
        """Handler untuk tombol preprocessing."""
        if ui_components['processing_running']:
            logger.warning("⚠️ Preprocessing sedang berjalan")
            return
        
        # Check dataset dengan path validator
        dataset_exists, dataset_msg = _check_dataset_exists()
        if not dataset_exists:
            logger.error(f"❌ {dataset_msg}")
            return
        
        # Check existing preprocessed data
        preprocessed_dir = Path(ui_components.get('preprocessed_dir', 'data/preprocessed'))
        split_config = ui_components['split_dropdown'].value
        
        # Get preprocessed validation
        preprocessed_validation = path_validator.validate_preprocessed_structure(str(preprocessed_dir))
        existing_data = []
        
        if split_config == 'all':
            for split in ['train', 'valid', 'test']:
                if (preprocessed_validation['splits'].get(split, {}).get('processed', 0) > 0):
                    existing_data.append(split)
        else:
            if (preprocessed_validation['splits'].get(split_config, {}).get('processed', 0) > 0):
                existing_data.append(split_config)
        
        if existing_data:
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
        
        # Update UI state
        ui_components['preprocess_button'].disabled = True
        ui_components['preprocess_button'].description = "Processing..."
        
        if logger:
            logger.info("🚀 Memulai preprocessing dataset")
        
        # Update status panel
        from smartcash.ui.components.status_panel import update_status_panel
        update_status_panel(ui_components['status_panel'], "Memulai preprocessing dataset...", "info")
        
        # Show progress container
        if 'progress_components' in ui_components:
            ui_components['progress_components']['show_container']()
            ui_components['progress_components']['reset_all']()
        
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
            
            # Update overall progress to 100%
            if 'progress_components' in ui_components:
                update_overall_progress(
                    ui_components['progress_components'],
                    100, 100, "Preprocessing selesai"
                )
            
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(ui_components['status_panel'], success_msg, "success")
        else:
            error_msg = "❌ Preprocessing gagal"
            logger.error(error_msg)
            update_status_panel(ui_components['status_panel'], error_msg, "error")
        
        # Hide progress setelah delay
        import threading
        def hide_progress():
            if 'progress_components' in ui_components:
                ui_components['progress_components']['hide_container']()
        threading.Timer(3.0, hide_progress).start()
    
    def _on_preprocessing_error(error):
        """Handler saat preprocessing error."""
        ui_components['processing_running'] = False
        ui_components['preprocess_button'].disabled = False
        ui_components['preprocess_button'].description = "Mulai Preprocessing"
        
        error_msg = f"❌ Error preprocessing: {str(error)}"
        logger.error(error_msg)
        
        from smartcash.ui.components.status_panel import update_status_panel
        update_status_panel(ui_components['status_panel'], error_msg, "error")
        
        # Reset progress
        if 'progress_components' in ui_components:
            ui_components['progress_components']['reset_all']()
            ui_components['progress_components']['hide_container']()
    
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