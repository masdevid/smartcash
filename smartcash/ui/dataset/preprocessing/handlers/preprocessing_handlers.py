"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Simplified handlers tanpa complex confirmation untuk fix import error
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.preprocessing.utils.ui_utils import (
    clear_outputs, handle_ui_error, show_ui_success, log_to_accordion
)
from smartcash.ui.dataset.preprocessing.utils.progress_utils import (
    create_dual_progress_callback, setup_progress_tracking,
    complete_progress_tracking, error_progress_tracking
)
from smartcash.ui.dataset.preprocessing.utils.button_manager import with_button_management
from smartcash.ui.dataset.preprocessing.utils.backend_utils import (
    validate_dataset_ready, check_preprocessed_exists,
    create_backend_preprocessor, create_backend_checker, create_backend_cleanup_service,
    _convert_ui_to_backend_config
)
from smartcash.ui.dataset.preprocessing.utils.confirmation_utils import (
    get_preprocessing_confirmation, get_cleanup_confirmation, clear_confirmation_area
)

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan simplified confirmation system"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Setup progress integration
        _setup_progress_integration(ui_components)
        
        # Setup config handlers (tanpa progress tracking)
        _setup_config_handlers(ui_components)
        
        # Setup operation handlers (dengan simplified confirmation)
        _setup_operation_handlers(ui_components, config)
        
        logger.info("✅ Preprocessing handlers setup completed")
        return ui_components
        
    except Exception as e:
        error_msg = f"❌ Error setup handlers: {str(e)}"
        logger.error(error_msg)
        handle_ui_error(ui_components, error_msg)
        return ui_components

def _setup_progress_integration(ui_components: Dict[str, Any]):
    """Setup progress callback integration"""
    if 'progress_callback' not in ui_components:
        progress_callback = create_dual_progress_callback(ui_components)
        ui_components['progress_callback'] = progress_callback

def _setup_config_handlers(ui_components: Dict[str, Any]):
    """Setup config handlers tanpa progress tracking"""
    
    def simple_save_config(button=None):
        """Simple save config tanpa progress tracking"""
        clear_outputs(ui_components, clear_logs=False, clear_confirm=True)
        
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "❌ Config handler tidak tersedia")
                return
            
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            success = config_handler.save_config(ui_components)
            
            if success:
                log_to_accordion(ui_components, "✅ Konfigurasi berhasil disimpan", "success")
            else:
                handle_ui_error(ui_components, "❌ Gagal menyimpan konfigurasi")
                
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error save config: {str(e)}")
    
    def simple_reset_config(button=None):
        """Simple reset config tanpa progress tracking"""
        clear_outputs(ui_components, clear_logs=False, clear_confirm=True)
        
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "❌ Config handler tidak tersedia")
                return
            
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            success = config_handler.reset_config(ui_components)
            
            if success:
                log_to_accordion(ui_components, "🔄 Konfigurasi berhasil direset ke default", "success")
            else:
                handle_ui_error(ui_components, "❌ Gagal reset konfigurasi")
                
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error reset config: {str(e)}")
    
    # Bind handlers
    if save_button := ui_components.get('save_button'):
        save_button.on_click(simple_save_config)
    if reset_button := ui_components.get('reset_button'):
        reset_button.on_click(simple_reset_config)

def _setup_operation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup operation handlers dengan simplified confirmation"""
    
    def preprocessing_handler(button=None):
        return _execute_preprocessing_simplified(ui_components, config)
    
    def check_handler(button=None):
        return _execute_check_operation(ui_components, config)
    
    def cleanup_handler(button=None):
        return _execute_cleanup_simplified(ui_components, config)
    
    # Bind handlers
    if preprocess_button := ui_components.get('preprocess_button'):
        preprocess_button.on_click(preprocessing_handler)
    if check_button := ui_components.get('check_button'):
        check_button.on_click(check_handler)
    if cleanup_button := ui_components.get('cleanup_button'):
        cleanup_button.on_click(cleanup_handler)

@with_button_management('preprocessing')
def _execute_preprocessing_simplified(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Execute preprocessing dengan simplified confirmation"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        clear_outputs(ui_components)
        
        # Simplified confirmation
        confirmed = get_preprocessing_confirmation(ui_components, "preprocessing dataset")
        if not confirmed:
            log_to_accordion(ui_components, "🚫 Preprocessing dibatalkan", "info")
            return False
        
        # Setup progress
        setup_progress_tracking(ui_components, "Dataset Preprocessing")
        
        # Get backend config
        backend_config = _convert_ui_to_backend_config(ui_components)
        
        # Validation
        log_to_accordion(ui_components, "🔍 Memvalidasi dataset...", "info")
        is_valid, validation_msg = validate_dataset_ready(backend_config)
        if not is_valid:
            error_progress_tracking(ui_components, "Validation failed")
            handle_ui_error(ui_components, f"Pre-validation failed: {validation_msg}")
            return False
        
        log_to_accordion(ui_components, f"✅ {validation_msg}", "success")
        
        # Create preprocessor
        log_to_accordion(ui_components, "🏗️ Membuat preprocessing service...", "info")
        preprocessor = create_backend_preprocessor(backend_config, progress_callback=ui_components.get('progress_callback'))
        if not preprocessor:
            error_progress_tracking(ui_components, "Service creation failed")
            handle_ui_error(ui_components, "Failed to create preprocessing service")
            return False
        
        # Execute preprocessing
        log_to_accordion(ui_components, "🚀 Memulai proses preprocessing dataset...", "info")
        result = preprocessor.preprocess_dataset()
        
        if result.get('success', False):
            stats = result.get('stats', {})
            processed_count = stats.get('total_processed', 0)
            processing_time = stats.get('processing_time', 0)
            
            message = f"Preprocessing berhasil: {processed_count:,} gambar diproses dalam {processing_time:.1f} detik"
            complete_progress_tracking(ui_components, message)
            show_ui_success(ui_components, message)
            
            # Log additional stats jika ada
            if success_rate := stats.get('success_rate'):
                log_to_accordion(ui_components, f"📊 Success rate: {success_rate}%", "info")
            
            return True
        else:
            error_msg = result.get('message', 'Preprocessing failed')
            error_progress_tracking(ui_components, error_msg)
            handle_ui_error(ui_components, error_msg)
            return False
            
    except Exception as e:
        error_msg = f"Preprocessing error: {str(e)}"
        logger.error(error_msg)
        error_progress_tracking(ui_components, error_msg)
        handle_ui_error(ui_components, error_msg)
        return False

@with_button_management('validation')
def _execute_check_operation(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Execute dataset check operation"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        clear_outputs(ui_components)
        setup_progress_tracking(ui_components, "Dataset Validation")
        
        # Get backend config
        backend_config = _convert_ui_to_backend_config(ui_components)
        
        # Check source dataset
        log_to_accordion(ui_components, "🔍 Memeriksa dataset sumber...", "info")
        is_valid, source_msg = validate_dataset_ready(backend_config)
        
        # Check preprocessed data
        log_to_accordion(ui_components, "💾 Memeriksa data preprocessed...", "info")
        preprocessed_exists, preprocessed_msg = check_preprocessed_exists(backend_config)
        
        # Compile results
        results = []
        if is_valid:
            results.append(f"✅ Dataset sumber: {source_msg}")
        else:
            results.append(f"❌ Dataset sumber: {source_msg}")
        
        if preprocessed_exists:
            results.append(f"💾 Preprocessed: {preprocessed_msg}")
        else:
            results.append("ℹ️ Belum ada data preprocessed")
        
        # Show results
        final_message = " | ".join(results)
        
        if is_valid:
            complete_progress_tracking(ui_components, "Dataset check completed")
            show_ui_success(ui_components, final_message)
            
            # Log detailed info
            for result in results:
                log_to_accordion(ui_components, result, "info")
            
            return True
        else:
            error_progress_tracking(ui_components, source_msg)
            handle_ui_error(ui_components, final_message)
            return False
            
    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        logger.error(error_msg)
        error_progress_tracking(ui_components, error_msg)
        handle_ui_error(ui_components, error_msg)
        return False

@with_button_management('cleanup')
def _execute_cleanup_simplified(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Execute cleanup dengan simplified confirmation"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        clear_outputs(ui_components)
        
        # Get backend config
        backend_config = _convert_ui_to_backend_config(ui_components)
        
        # Check files exist first
        log_to_accordion(ui_components, "🔍 Memeriksa data preprocessed yang ada...", "info")
        exists, detailed_msg = check_preprocessed_exists(backend_config)
        if not exists:
            log_to_accordion(ui_components, "ℹ️ Tidak ada data untuk dibersihkan", "info")
            return True
        
        # Log what will be deleted
        log_to_accordion(ui_components, f"📊 Data yang akan dihapus: {detailed_msg}", "info")
        
        # Simplified confirmation
        confirmed = get_cleanup_confirmation(ui_components, detailed_msg)
        if not confirmed:
            log_to_accordion(ui_components, "🚫 Cleanup dibatalkan", "info")
            return False
        
        # Setup progress
        setup_progress_tracking(ui_components, "Dataset Cleanup")
        
        # Create cleanup service
        log_to_accordion(ui_components, "🏗️ Membuat cleanup service...", "info")
        cleanup_service = create_backend_cleanup_service(backend_config, ui_components=ui_components)
        if not cleanup_service:
            error_progress_tracking(ui_components, "Service creation failed")
            handle_ui_error(ui_components, "Failed to create cleanup service")
            return False
        
        # Execute cleanup
        log_to_accordion(ui_components, "🗑️ Menghapus data preprocessed...", "info")
        result = cleanup_service.cleanup_preprocessed_data()
        
        if result.get('success', False):
            stats = result.get('stats', {})
            files_removed = stats.get('files_removed', 0)
            dirs_removed = stats.get('dirs_removed', 0)
            
            message = f"Cleanup berhasil: {files_removed:,} file dan {dirs_removed} direktori dihapus"
            complete_progress_tracking(ui_components, message)
            show_ui_success(ui_components, message)
            
            # Additional cleanup info
            log_to_accordion(ui_components, f"🗑️ Files removed: {files_removed:,}", "info")
            log_to_accordion(ui_components, f"📁 Directories removed: {dirs_removed}", "info")
            
            return True
        else:
            error_msg = result.get('message', 'Cleanup failed')
            error_progress_tracking(ui_components, error_msg)
            handle_ui_error(ui_components, error_msg)
            return False
            
    except Exception as e:
        error_msg = f"Cleanup error: {str(e)}"
        logger.error(error_msg)
        error_progress_tracking(ui_components, error_msg)
        handle_ui_error(ui_components, error_msg)
        return False