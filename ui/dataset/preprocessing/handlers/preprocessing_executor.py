"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_executor.py
Deskripsi: Handler khusus untuk execute preprocessing dengan progress tracking
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from smartcash.common.environment import get_environment_manager
from smartcash.dataset.services.preprocessing_manager import PreprocessingManager
from smartcash.ui.components.status_panel import update_status_panel
from smartcash.ui.dataset.preprocessing.utils import (
    get_config_extractor, get_validation_helper, 
    get_dialog_manager, get_ui_state_manager, get_progress_bridge
)

def setup_preprocessing_executor(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handler untuk execute preprocessing dengan comprehensive management."""
    logger = ui_components.get('logger')
    env_manager = env or get_environment_manager()
    
    # Get utilities
    config_extractor = get_config_extractor(ui_components)
    validation_helper = get_validation_helper(ui_components, logger)
    dialog_manager = get_dialog_manager(ui_components)
    ui_state = get_ui_state_manager(ui_components)
    progress_bridge = get_progress_bridge(ui_components, logger)
    
    # State management
    ui_components['preprocessing_manager'] = None
    
    def _get_preprocessing_manager() -> PreprocessingManager:
        """Get preprocessing manager dengan lazy initialization."""
        if not ui_components['preprocessing_manager']:
            config = {
                'preprocessing': config_extractor.get_preprocessing_config(),
                'data': {'dir': ui_components.get('data_dir', 'data')}
            }
            ui_components['preprocessing_manager'] = PreprocessingManager(config, logger)
            ui_components['preprocessing_manager'].register_progress_callback(progress_bridge.update_progress)
        
        return ui_components['preprocessing_manager']
    
    def _on_preprocess_click(b):
        """Handler untuk tombol preprocessing dengan comprehensive validation."""
        # Check operation state
        can_start, message = ui_state.can_start_operation('preprocessing', exclude_operations=['validation'])
        if not can_start:
            logger and logger.warning(f"⚠️ {message}")
            update_status_panel(ui_components['status_panel'], message, "warning")
            return
        
        # Check dataset existence
        dataset_exists, dataset_msg = validation_helper.check_dataset_exists()
        if not dataset_exists:
            logger and logger.error(f"❌ {dataset_msg}")
            update_status_panel(ui_components['status_panel'], dataset_msg, "error")
            return
        
        # Check existing preprocessed data
        split_config = ui_components['split_dropdown'].value
        existing_data = validation_helper.check_existing_preprocessed_for_split(split_config)
        
        # Show confirmation jika ada data existing
        if existing_data:
            existing_str = ', '.join(existing_data)
            confirmation_msg = f"""Data preprocessing sudah ada untuk: {existing_str}

Apakah Anda ingin memproses ulang?
Data yang ada akan ditimpa dan tidak dapat dikembalikan."""
            
            dialog_manager.show_confirmation_dialog(
                title="Konfirmasi Preprocessing Ulang",
                message=confirmation_msg,
                on_confirm=_start_preprocessing_confirmed,
                confirm_text="Ya, Proses Ulang",
                cancel_text="Batal",
                danger_mode=True
            )
        else:
            _start_preprocessing_confirmed()
    
    def _start_preprocessing_confirmed():
        """Start preprocessing setelah konfirmasi."""
        # Setup progress
        progress_bridge.setup_for_operation('preprocessing')
        
        # Set UI state
        ui_state.set_button_processing('preprocess_button', True, "Processing...")
        
        logger and logger.info("🚀 Memulai preprocessing dataset")
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
                import threading
                threading.Timer(1.0, check_completion).start()
        
        check_completion()
    
    def _run_preprocessing():
        """Execute preprocessing dalam thread terpisah."""
        try:
            preprocessing_manager = _get_preprocessing_manager()
            config = config_extractor.get_preprocessing_config()
            
            # Extract split untuk avoid duplicate keyword argument
            split_value = config.pop('split', 'all')
            
            # Map 'val' ke 'valid' untuk consistency
            if split_value == 'val':
                split_value = 'valid'
            
            # Execute preprocessing dengan full config
            result = preprocessing_manager.preprocess_dataset(
                split=split_value,
                force_reprocess=True,
                show_progress=False,  # Disable tqdm, use UI progress
                **config
            )
            
            return result
            
        except Exception as e:
            logger and logger.error(f"❌ Error preprocessing: {str(e)}")
            raise
    
    def _on_preprocessing_complete(result):
        """Handler saat preprocessing selesai."""
        # Reset UI state
        ui_state.set_button_processing('preprocess_button', False, 
                                     success_text="Mulai Preprocessing", 
                                     success_style='success')
        
        if result and result.get('success', False):
            total_images = result.get('total_images', 0)
            processing_time = result.get('processing_time', 0)
            
            success_msg = f"✅ Preprocessing selesai: {total_images:,} gambar dalam {processing_time:.1f} detik"
            logger and logger.success(success_msg)
            
            progress_bridge.complete_operation("Preprocessing selesai")
            update_status_panel(ui_components['status_panel'], success_msg, "success")
            
            # Log detail statistics
            if 'split_stats' in result:
                stats_detail = []
                for split, stats in result['split_stats'].items():
                    images = stats.get('images', 0)
                    if images > 0:
                        stats_detail.append(f"{split}: {images:,} gambar")
                
                if stats_detail and logger:
                    logger.info(f"📊 Detail hasil: {', '.join(stats_detail)}")
        else:
            error_msg = result.get('message', 'Preprocessing gagal') if result else "Preprocessing gagal"
            logger and logger.error(f"❌ {error_msg}")
            
            progress_bridge.error_operation("Preprocessing gagal")
            update_status_panel(ui_components['status_panel'], error_msg, "error")
    
    def _on_preprocessing_error(error):
        """Handler saat preprocessing error."""
        # Reset UI state
        ui_state.set_button_processing('preprocess_button', False)
        
        error_msg = f"❌ Error preprocessing: {str(error)}"
        logger and logger.error(error_msg)
        
        progress_bridge.error_operation("Error preprocessing")
        update_status_panel(ui_components['status_panel'], error_msg, "error")
    
    # Setup event handler
    ui_components['preprocess_button'].on_click(_on_preprocess_click)
    
    logger and logger.debug("✅ Preprocessing executor setup selesai")
    
    return ui_components