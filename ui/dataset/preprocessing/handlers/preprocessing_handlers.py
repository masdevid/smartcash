"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Integrated handlers dengan working confirmation dialog dan progress tracker
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
from smartcash.ui.dataset.preprocessing.utils.button_manager import (
    disable_operation_buttons, enable_operation_buttons
)
from smartcash.ui.dataset.preprocessing.utils.backend_utils import (
    validate_dataset_ready, check_preprocessed_exists,
    create_backend_preprocessor, create_backend_checker, create_backend_cleanup_service,
    _convert_ui_to_backend_config
)
from smartcash.ui.dataset.preprocessing.utils.confirmation_utils import (
    show_preprocessing_confirmation, show_cleanup_confirmation, 
    should_execute_operation, is_confirmation_pending, clear_confirmation_area
)

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan working confirmation dan progress integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Setup progress integration
        _setup_progress_integration(ui_components)
        
        # Setup config handlers (tanpa progress tracking)
        _setup_config_handlers(ui_components)
        
        # Setup operation handlers (dengan working confirmation)
        _setup_operation_handlers(ui_components, config)
        
        logger.info("✅ Preprocessing handlers setup dengan working confirmation")
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
    """Setup operation handlers dengan working confirmation system"""
    
    def preprocessing_handler(button=None):
        return _handle_preprocessing_request(ui_components, config)
    
    def check_handler(button=None):
        return _execute_check_operation(ui_components, config)
    
    def cleanup_handler(button=None):
        return _handle_cleanup_request(ui_components, config)
    
    # Bind handlers
    if preprocess_button := ui_components.get('preprocess_button'):
        preprocess_button.on_click(preprocessing_handler)
    if check_button := ui_components.get('check_button'):
        check_button.on_click(check_handler)
    if cleanup_button := ui_components.get('cleanup_button'):
        cleanup_button.on_click(cleanup_handler)

def _handle_preprocessing_request(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Handle preprocessing request dengan confirmation dialog"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Clear outputs
        clear_outputs(ui_components)
        
        # Check jika sudah ada konfirmasi sebelumnya
        if should_execute_operation(ui_components, 'preprocessing'):
            return _execute_preprocessing_confirmed(ui_components, config)
        
        # Check jika sedang menunggu konfirmasi
        if is_confirmation_pending(ui_components):
            log_to_accordion(ui_components, "⏳ Menunggu konfirmasi user...", "info")
            return False
        
        # Disable buttons
        disable_operation_buttons(ui_components)
        
        # Pre-validation sebelum konfirmasi
        log_to_accordion(ui_components, "🔍 Memeriksa dataset sebelum konfirmasi...", "info")
        backend_config = _convert_ui_to_backend_config(ui_components)
        
        is_valid, validation_msg = validate_dataset_ready(backend_config)
        if not is_valid:
            enable_operation_buttons(ui_components)
            handle_ui_error(ui_components, f"Pre-validation failed: {validation_msg}")
            return False
        
        log_to_accordion(ui_components, f"✅ Pre-validation: {validation_msg}", "success")
        
        # Show confirmation dialog
        show_preprocessing_confirmation(ui_components, 
            f"Dataset siap untuk diproses.<br>📊 {validation_msg}<br><br>Apakah Anda yakin ingin memulai preprocessing?")
        
        return True
        
    except Exception as e:
        error_msg = f"Error handling preprocessing request: {str(e)}"
        logger.error(error_msg)
        enable_operation_buttons(ui_components)
        handle_ui_error(ui_components, error_msg)
        return False

def _execute_preprocessing_confirmed(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Execute preprocessing setelah konfirmasi diterima"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Setup progress tracking
        setup_progress_tracking(ui_components, "Dataset Preprocessing")
        
        # Get backend config
        backend_config = _convert_ui_to_backend_config(ui_components)
        
        # Create preprocessor dengan progress callback
        log_to_accordion(ui_components, "🏗️ Membuat preprocessing service...", "info")
        preprocessor = create_backend_preprocessor(backend_config, progress_callback=ui_components.get('progress_callback'))
        if not preprocessor:
            error_progress_tracking(ui_components, "Service creation failed")
            handle_ui_error(ui_components, "Failed to create preprocessing service")
            enable_operation_buttons(ui_components)
            return False
        
        # Execute preprocessing dengan progress tracking
        log_to_accordion(ui_components, "🚀 Memulai proses preprocessing dataset...", "info")
        result = preprocessor.preprocess_dataset()
        
        if result.get('success', False):
            stats = result.get('stats', {})
            processed_count = stats.get('total_processed', 0)
            processing_time = stats.get('processing_time', 0)
            success_rate = stats.get('success_rate', 0)
            
            message = f"Preprocessing berhasil: {processed_count:,} gambar diproses dalam {processing_time:.1f} detik (Success rate: {success_rate}%)"
            complete_progress_tracking(ui_components, message)
            show_ui_success(ui_components, message)
            
            # Log detailed stats
            if normalized_count := stats.get('total_normalized', 0):
                log_to_accordion(ui_components, f"🎨 Normalisasi: {normalized_count:,} gambar", "info")
            
            enable_operation_buttons(ui_components)
            return True
        else:
            error_msg = result.get('message', 'Preprocessing failed')
            error_progress_tracking(ui_components, error_msg)
            handle_ui_error(ui_components, error_msg)
            enable_operation_buttons(ui_components)
            return False
            
    except Exception as e:
        error_msg = f"Preprocessing execution error: {str(e)}"
        logger.error(error_msg)
        error_progress_tracking(ui_components, error_msg)
        handle_ui_error(ui_components, error_msg)
        enable_operation_buttons(ui_components)
        return False

def _handle_cleanup_request(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Handle cleanup request dengan confirmation dialog"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Clear outputs
        clear_outputs(ui_components)
        
        # Check jika sudah ada konfirmasi sebelumnya
        if should_execute_operation(ui_components, 'cleanup'):
            return _execute_cleanup_confirmed(ui_components, config)
        
        # Check jika sedang menunggu konfirmasi
        if is_confirmation_pending(ui_components):
            log_to_accordion(ui_components, "⏳ Menunggu konfirmasi user...", "info")
            return False
        
        # Disable buttons
        disable_operation_buttons(ui_components)
        
        # Check files exist
        log_to_accordion(ui_components, "🔍 Memeriksa data preprocessed yang ada...", "info")
        backend_config = _convert_ui_to_backend_config(ui_components)
        
        exists, detailed_msg = check_preprocessed_exists(backend_config)
        if not exists:
            enable_operation_buttons(ui_components)
            log_to_accordion(ui_components, "ℹ️ Tidak ada data untuk dibersihkan", "info")
            return True
        
        log_to_accordion(ui_components, f"📊 Data ditemukan: {detailed_msg}", "info")
        
        # Show confirmation dialog dengan detailed stats
        show_cleanup_confirmation(ui_components, detailed_msg)
        
        return True
        
    except Exception as e:
        error_msg = f"Error handling cleanup request: {str(e)}"
        logger.error(error_msg)
        enable_operation_buttons(ui_components)
        handle_ui_error(ui_components, error_msg)
        return False

def _execute_cleanup_confirmed(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Execute cleanup setelah konfirmasi diterima dengan progress tracking"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Setup progress tracking
        setup_progress_tracking(ui_components, "Dataset Cleanup")
        
        # Get backend config
        backend_config = _convert_ui_to_backend_config(ui_components)
        
        # Create cleanup service
        log_to_accordion(ui_components, "🏗️ Membuat cleanup service...", "info")
        cleanup_service = create_backend_cleanup_service(backend_config, ui_components=ui_components)
        if not cleanup_service:
            error_progress_tracking(ui_components, "Service creation failed")
            handle_ui_error(ui_components, "Failed to create cleanup service")
            enable_operation_buttons(ui_components)
            return False
        
        # Execute cleanup dengan progress tracking
        log_to_accordion(ui_components, "🗑️ Menghapus data preprocessed...", "info")
        
        # Update progress manually karena cleanup service mungkin tidak provide callback
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            if hasattr(progress_tracker, 'update_current'):
                progress_tracker.update_current(25, "Memulai proses cleanup...")
        
        result = cleanup_service.cleanup_preprocessed_data()
        
        # Update progress
        if progress_tracker and hasattr(progress_tracker, 'update_current'):
            progress_tracker.update_current(75, "Finalisasi cleanup...")
        
        if result.get('success', False):
            stats = result.get('stats', {})
            files_removed = stats.get('files_removed', 0)
            dirs_removed = stats.get('dirs_removed', 0)
            errors = stats.get('errors', 0)
            
            message = f"Cleanup berhasil: {files_removed:,} file dan {dirs_removed} direktori dihapus"
            if errors > 0:
                message += f" ({errors} error)"
            
            complete_progress_tracking(ui_components, message)
            show_ui_success(ui_components, message)
            
            # Log detailed cleanup stats
            log_to_accordion(ui_components, f"🗑️ Files removed: {files_removed:,}", "info")
            log_to_accordion(ui_components, f"📁 Directories removed: {dirs_removed}", "info")
            if errors > 0:
                log_to_accordion(ui_components, f"⚠️ Errors encountered: {errors}", "warning")
            
            enable_operation_buttons(ui_components)
            return True
        else:
            error_msg = result.get('message', 'Cleanup failed')
            error_progress_tracking(ui_components, error_msg)
            handle_ui_error(ui_components, error_msg)
            enable_operation_buttons(ui_components)
            return False
            
    except Exception as e:
        error_msg = f"Cleanup execution error: {str(e)}"
        logger.error(error_msg)
        error_progress_tracking(ui_components, error_msg)
        handle_ui_error(ui_components, error_msg)
        enable_operation_buttons(ui_components)
        return False

def _execute_check_operation(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Execute dataset check operation dengan progress tracking"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        clear_outputs(ui_components)
        disable_operation_buttons(ui_components)
        setup_progress_tracking(ui_components, "Dataset Validation")
        
        # Get backend config
        backend_config = _convert_ui_to_backend_config(ui_components)
        
        # Check source dataset
        log_to_accordion(ui_components, "🔍 Memeriksa dataset sumber...", "info")
        is_valid, source_msg = validate_dataset_ready(backend_config)
        
        # Update progress
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update_current'):
            progress_tracker.update_current(50, "Memeriksa data preprocessed...")
        
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
            
            enable_operation_buttons(ui_components)
            return True
        else:
            error_progress_tracking(ui_components, source_msg)
            handle_ui_error(ui_components, final_message)
            enable_operation_buttons(ui_components)
            return False
            
    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        logger.error(error_msg)
        error_progress_tracking(ui_components, error_msg)
        handle_ui_error(ui_components, error_msg)
        enable_operation_buttons(ui_components)
        return False