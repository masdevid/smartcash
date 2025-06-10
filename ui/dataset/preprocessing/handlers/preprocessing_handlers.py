"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Fixed handlers dengan proper progress integration mengikuti pola augmentation
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.preprocessing.utils.ui_utils import (
    clear_outputs, handle_ui_error, show_ui_success, log_to_accordion
)
from smartcash.ui.dataset.preprocessing.utils.progress_utils import (
    setup_progress_tracking, complete_progress_tracking, error_progress_tracking
)
from smartcash.ui.dataset.preprocessing.utils.button_manager import (
    disable_operation_buttons, enable_operation_buttons
)
from smartcash.ui.dataset.preprocessing.utils.backend_utils import (
    validate_dataset_ready, check_preprocessed_exists,
    create_backend_preprocessor_with_progress, create_backend_checker, 
    create_backend_cleanup_service_with_progress, _convert_ui_to_backend_config
)
from smartcash.ui.dataset.preprocessing.utils.confirmation_utils import (
    show_preprocessing_confirmation, show_cleanup_confirmation, 
    should_execute_operation, is_confirmation_pending, clear_confirmation_area
)

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """🔧 Setup handlers dengan fixed progress integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Setup config handlers (tanpa progress tracking)
        _setup_config_handlers_fixed(ui_components)
        
        # Setup operation handlers (dengan working progress integration)
        _setup_operation_handlers_fixed(ui_components, config)
        
        logger.info("✅ Preprocessing handlers setup dengan fixed progress integration")
        return ui_components
        
    except Exception as e:
        error_msg = f"❌ Error setup handlers: {str(e)}"
        logger.error(error_msg)
        handle_ui_error(ui_components, error_msg)
        return ui_components

def _setup_config_handlers_fixed(ui_components: Dict[str, Any]):
    """Setup config handlers dengan UI logging yang benar"""
    
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
            # ✅ Log sudah ditangani di config_handler via _log_to_ui
                
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
            # ✅ Log sudah ditangani di config_handler via _log_to_ui
                
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error reset config: {str(e)}")
    
    # Bind handlers
    if save_button := ui_components.get('save_button'):
        save_button.on_click(simple_save_config)
    if reset_button := ui_components.get('reset_button'):
        reset_button.on_click(simple_reset_config)

def _setup_operation_handlers_fixed(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """🔧 Setup operation handlers dengan fixed progress integration pattern"""
    
    def preprocessing_handler(button=None):
        return _handle_preprocessing_request_fixed(ui_components, config)
    
    def check_handler(button=None):
        return _execute_check_operation_fixed(ui_components, config)
    
    def cleanup_handler(button=None):
        return _handle_cleanup_request_fixed(ui_components, config)
    
    # Bind handlers
    if preprocess_button := ui_components.get('preprocess_button'):
        preprocess_button.on_click(preprocessing_handler)
    if check_button := ui_components.get('check_button'):
        check_button.on_click(check_handler)
    if cleanup_button := ui_components.get('cleanup_button'):
        cleanup_button.on_click(cleanup_handler)

def _handle_preprocessing_request_fixed(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🚀 Handle preprocessing request dengan fixed confirmation dan progress"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        clear_outputs(ui_components)
        
        # Check confirmation state
        if should_execute_operation(ui_components, 'preprocessing'):
            return _execute_preprocessing_with_progress(ui_components, config)
        
        if is_confirmation_pending(ui_components):
            log_to_accordion(ui_components, "⏳ Menunggu konfirmasi user...", "info")
            return False
        
        # Disable buttons dan validate
        disable_operation_buttons(ui_components)
        
        log_to_accordion(ui_components, "🔍 Validating dataset before confirmation...", "info")
        backend_config = _convert_ui_to_backend_config(ui_components)
        
        is_valid, validation_msg = validate_dataset_ready(backend_config)
        if not is_valid:
            enable_operation_buttons(ui_components)
            handle_ui_error(ui_components, f"Pre-validation failed: {validation_msg}")
            return False
        
        log_to_accordion(ui_components, f"✅ Pre-validation: {validation_msg}", "success")
        
        # Show confirmation
        show_preprocessing_confirmation(ui_components, 
            f"Dataset siap untuk diproses.<br>📊 {validation_msg}<br><br>Apakah Anda yakin ingin memulai preprocessing?")
        
        return True
        
    except Exception as e:
        error_msg = f"Error handling preprocessing request: {str(e)}"
        logger.error(error_msg)
        enable_operation_buttons(ui_components)
        handle_ui_error(ui_components, error_msg)
        return False

def _execute_preprocessing_with_progress(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🔧 Execute preprocessing dengan fixed progress integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # 🎯 CRITICAL: Setup progress tracking SEBELUM create service
        setup_progress_tracking(ui_components, "Dataset Preprocessing")
        
        # Show progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'show'):
            progress_tracker.show()
        
        log_to_accordion(ui_components, "🏗️ Creating preprocessing service dengan progress integration...", "info")
        
        # 🔑 KEY: Create service dengan UI components untuk progress integration
        service = create_backend_preprocessor_with_progress(ui_components)
        if not service:
            error_progress_tracking(ui_components, "Service creation failed")
            handle_ui_error(ui_components, "Failed to create preprocessing service")
            enable_operation_buttons(ui_components)
            return False
        
        # 🚀 Execute preprocessing dengan integrated progress
        log_to_accordion(ui_components, "🚀 Starting preprocessing pipeline...", "info")
        
        # ✅ Service sudah memiliki progress callback terintegrasi
        result = service.preprocess_dataset()
        
        if result.get('success', False):
            stats = result.get('stats', {})
            processed_count = stats.get('output', {}).get('total_processed', 0)
            processing_time = stats.get('processing_time', 0)
            success_rate = stats.get('output', {}).get('success_rate', '0%')
            
            success_message = f"Preprocessing berhasil: {processed_count:,} gambar diproses dalam {processing_time:.1f} detik (Success rate: {success_rate})"
            
            # Complete progress
            complete_progress_tracking(ui_components, success_message)
            show_ui_success(ui_components, success_message)
            
            # Log detailed stats (TANPA backend log untuk avoid double)
            if normalized_count := stats.get('output', {}).get('total_normalized', 0):
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

def _handle_cleanup_request_fixed(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🧹 Handle cleanup request dengan fixed progress integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        clear_outputs(ui_components)
        
        # Check confirmation state
        if should_execute_operation(ui_components, 'cleanup'):
            return _execute_cleanup_with_progress(ui_components, config)
        
        if is_confirmation_pending(ui_components):
            log_to_accordion(ui_components, "⏳ Menunggu konfirmasi user...", "info")
            return False
        
        # Disable buttons dan check files
        disable_operation_buttons(ui_components)
        
        log_to_accordion(ui_components, "🔍 Checking preprocessed data...", "info")
        backend_config = _convert_ui_to_backend_config(ui_components)
        
        exists, detailed_msg = check_preprocessed_exists(backend_config)
        if not exists:
            enable_operation_buttons(ui_components)
            log_to_accordion(ui_components, "ℹ️ Tidak ada data untuk dibersihkan", "info")
            return True
        
        log_to_accordion(ui_components, f"📊 Data ditemukan: {detailed_msg}", "info")
        
        # Show confirmation dengan detailed stats
        show_cleanup_confirmation(ui_components, detailed_msg)
        
        return True
        
    except Exception as e:
        error_msg = f"Error handling cleanup request: {str(e)}"
        logger.error(error_msg)
        enable_operation_buttons(ui_components)
        handle_ui_error(ui_components, error_msg)
        return False

def _execute_cleanup_with_progress(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🔧 Execute cleanup dengan fixed progress integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Setup progress tracking
        setup_progress_tracking(ui_components, "Dataset Cleanup")
        
        # Show progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'show'):
            progress_tracker.show()
        
        log_to_accordion(ui_components, "🏗️ Creating cleanup service dengan progress integration...", "info")
        
        # 🔑 KEY: Create cleanup service dengan UI components untuk progress
        cleanup_service = create_backend_cleanup_service_with_progress(ui_components)
        if not cleanup_service:
            error_progress_tracking(ui_components, "Service creation failed")
            handle_ui_error(ui_components, "Failed to create cleanup service")
            enable_operation_buttons(ui_components)
            return False
        
        # Execute cleanup dengan integrated progress
        log_to_accordion(ui_components, "🗑️ Starting cleanup process...", "info")
        
        result = cleanup_service.cleanup_preprocessed_data()
        
        if result.get('success', False):
            stats = result.get('stats', {})
            files_removed = stats.get('files_removed', 0)
            
            success_message = f"Cleanup berhasil: {files_removed:,} file dihapus"
            
            # Complete progress
            complete_progress_tracking(ui_components, success_message)
            show_ui_success(ui_components, success_message)
            
            # Log cleanup details
            log_to_accordion(ui_components, f"🗑️ Files removed: {files_removed:,}", "info")
            
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

def _execute_check_operation_fixed(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🔍 Execute dataset check dengan fixed progress integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        clear_outputs(ui_components)
        disable_operation_buttons(ui_components)
        setup_progress_tracking(ui_components, "Dataset Validation")
        
        # Show progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'show'):
            progress_tracker.show()
        
        # Get backend config
        backend_config = _convert_ui_to_backend_config(ui_components)
        
        # Check source dataset
        log_to_accordion(ui_components, "🔍 Checking source dataset...", "info")
        if progress_tracker and hasattr(progress_tracker, 'update_current'):
            progress_tracker.update_current(25, "Validating source dataset...")
        
        is_valid, source_msg = validate_dataset_ready(backend_config)
        
        # Check preprocessed data
        if progress_tracker and hasattr(progress_tracker, 'update_current'):
            progress_tracker.update_current(75, "Checking preprocessed data...")
        
        log_to_accordion(ui_components, "💾 Checking preprocessed data...", "info")
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