"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Fixed handlers dengan working progress integration dan proper UI state management
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.preprocessing.utils.ui_utils import (
    clear_outputs, handle_ui_error, show_ui_success, log_to_accordion
)
from smartcash.ui.dataset.preprocessing.utils.button_manager import (
    disable_operation_buttons, enable_operation_buttons
)
from smartcash.ui.dataset.preprocessing.utils.backend_utils import (
    validate_dataset_ready, check_preprocessed_exists,
    create_backend_preprocessor_with_progress, create_backend_checker, 
    create_backend_cleanup_service_with_progress, _extract_and_enhance_config
)
from smartcash.ui.dataset.preprocessing.utils.confirmation_utils import (
    show_preprocessing_confirmation, show_cleanup_confirmation, 
    should_execute_operation, is_confirmation_pending, clear_confirmation_area
)

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """🔧 Setup handlers dengan working progress integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Setup config handlers (tanpa progress tracking)
        _setup_config_handlers_fixed(ui_components)
        
        # Setup operation handlers (dengan working progress integration)
        _setup_operation_handlers_fixed(ui_components, config)
        
        logger.info("✅ Preprocessing handlers setup dengan working progress integration")
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
    """🔧 Setup operation handlers dengan working progress integration"""
    
    def preprocessing_handler(button=None):
        return _handle_preprocessing_request_working(ui_components, config)
    
    def check_handler(button=None):
        return _execute_check_operation_working(ui_components, config)
    
    def cleanup_handler(button=None):
        return _handle_cleanup_request_working(ui_components, config)
    
    # Bind handlers
    if preprocess_button := ui_components.get('preprocess_button'):
        preprocess_button.on_click(preprocessing_handler)
    if check_button := ui_components.get('check_button'):
        check_button.on_click(check_handler)
    if cleanup_button := ui_components.get('cleanup_button'):
        cleanup_button.on_click(cleanup_handler)

def _handle_preprocessing_request_working(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🚀 Handle preprocessing request dengan working confirmation dan progress"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        clear_outputs(ui_components)
        
        # Check confirmation state
        if should_execute_operation(ui_components, 'preprocessing'):
            return _execute_preprocessing_with_working_progress(ui_components, config)
        
        if is_confirmation_pending(ui_components):
            log_to_accordion(ui_components, "⏳ Menunggu konfirmasi user...", "info")
            return False
        
        # Disable buttons dan validate
        disable_operation_buttons(ui_components)
        
        log_to_accordion(ui_components, "🔍 Validating dataset before confirmation...", "info")
        backend_config = _extract_and_enhance_config(ui_components)
        
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

def _execute_preprocessing_with_working_progress(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🔧 Execute preprocessing dengan WORKING progress integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # 🎯 CRITICAL: Force show progress tracker dengan visibility fix
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            # Ensure container is visible
            if hasattr(progress_tracker, 'container'):
                container = progress_tracker.container
                if hasattr(container, 'layout'):
                    container.layout.visibility = 'visible'
                    container.layout.display = 'flex'
                    container.layout.height = 'auto'
            
            # Call show method if available
            if hasattr(progress_tracker, 'show'):
                progress_tracker.show()
            
            # Initialize progress
            if hasattr(progress_tracker, 'update_overall'):
                progress_tracker.update_overall(0, "🚀 Memulai preprocessing...")
        
        log_to_accordion(ui_components, "🏗️ Creating preprocessing service dengan progress integration...", "info")
        
        # 🔑 KEY: Create service dengan UI components untuk progress integration
        service = create_backend_preprocessor_with_progress(ui_components)
        if not service:
            _error_cleanup(ui_components, "Failed to create preprocessing service")
            return False
        
        # 🚀 Execute preprocessing dengan integrated progress
        log_to_accordion(ui_components, "🚀 Starting preprocessing pipeline...", "info")
        
        # Update progress manually untuk start
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(5, "Pipeline starting...")
        
        # ✅ Service sudah memiliki progress callback terintegrasi
        result = service.preprocess_dataset()
        
        if result.get('success', False):
            stats = result.get('stats', {})
            processed_count = stats.get('output', {}).get('total_processed', 0)
            processing_time = stats.get('processing_time', 0)
            success_rate = stats.get('output', {}).get('success_rate', '0%')
            
            success_message = f"Preprocessing berhasil: {processed_count:,} gambar diproses dalam {processing_time:.1f} detik (Success rate: {success_rate})"
            
            # Complete progress
            if progress_tracker:
                if hasattr(progress_tracker, 'complete'):
                    progress_tracker.complete(success_message)
                elif hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(100, "✅ Selesai")
            
            show_ui_success(ui_components, success_message)
            
            # Log detailed stats
            if normalized_count := stats.get('output', {}).get('total_normalized', 0):
                log_to_accordion(ui_components, f"🎨 Normalisasi: {normalized_count:,} gambar", "info")
            
            enable_operation_buttons(ui_components)
            return True
        else:
            error_msg = result.get('message', 'Preprocessing failed')
            _error_cleanup(ui_components, error_msg)
            return False
            
    except Exception as e:
        error_msg = f"Preprocessing execution error: {str(e)}"
        logger.error(error_msg)
        _error_cleanup(ui_components, error_msg)
        return False

def _handle_cleanup_request_working(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🧹 Handle cleanup request dengan working progress integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        clear_outputs(ui_components)
        
        # Check confirmation state
        if should_execute_operation(ui_components, 'cleanup'):
            return _execute_cleanup_with_working_progress(ui_components, config)
        
        if is_confirmation_pending(ui_components):
            log_to_accordion(ui_components, "⏳ Menunggu konfirmasi user...", "info")
            return False
        
        # Disable buttons dan check files
        disable_operation_buttons(ui_components)
        
        log_to_accordion(ui_components, "🔍 Checking preprocessed data...", "info")
        backend_config = _extract_and_enhance_config(ui_components)
        
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

def _execute_cleanup_with_working_progress(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🔧 Execute cleanup dengan WORKING progress integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # 🎯 CRITICAL: Force show progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            # Ensure container is visible
            if hasattr(progress_tracker, 'container'):
                container = progress_tracker.container
                if hasattr(container, 'layout'):
                    container.layout.visibility = 'visible'
                    container.layout.display = 'flex'
                    container.layout.height = 'auto'
            
            # Call show method if available
            if hasattr(progress_tracker, 'show'):
                progress_tracker.show()
            
            # Initialize progress
            if hasattr(progress_tracker, 'update_overall'):
                progress_tracker.update_overall(0, "🗑️ Memulai cleanup...")
        
        log_to_accordion(ui_components, "🏗️ Creating cleanup service dengan progress integration...", "info")
        
        # 🔑 KEY: Create cleanup service dengan UI components untuk progress
        cleanup_service = create_backend_cleanup_service_with_progress(ui_components)
        if not cleanup_service:
            _error_cleanup(ui_components, "Failed to create cleanup service")
            return False
        
        # Execute cleanup dengan integrated progress
        log_to_accordion(ui_components, "🗑️ Starting cleanup process...", "info")
        
        # Update progress manually untuk start
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(5, "Cleanup starting...")
        
        result = cleanup_service.cleanup_preprocessed_data()
        
        if result.get('success', False):
            stats = result.get('stats', {})
            files_removed = stats.get('files_removed', 0)
            
            success_message = f"Cleanup berhasil: {files_removed:,} file dihapus"
            
            # Complete progress
            if progress_tracker:
                if hasattr(progress_tracker, 'complete'):
                    progress_tracker.complete(success_message)
                elif hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(100, "✅ Cleanup selesai")
            
            show_ui_success(ui_components, success_message)
            
            # Log cleanup details
            log_to_accordion(ui_components, f"🗑️ Files removed: {files_removed:,}", "info")
            
            enable_operation_buttons(ui_components)
            return True
        else:
            error_msg = result.get('message', 'Cleanup failed')
            _error_cleanup(ui_components, error_msg)
            return False
            
    except Exception as e:
        error_msg = f"Cleanup execution error: {str(e)}"
        logger.error(error_msg)
        _error_cleanup(ui_components, error_msg)
        return False

def _execute_check_operation_working(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🔍 Execute dataset check dengan working progress integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        clear_outputs(ui_components)
        disable_operation_buttons(ui_components)
        
        # 🎯 CRITICAL: Force show progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            # Ensure container is visible
            if hasattr(progress_tracker, 'container'):
                container = progress_tracker.container
                if hasattr(container, 'layout'):
                    container.layout.visibility = 'visible'
                    container.layout.display = 'flex'
                    container.layout.height = 'auto'
            
            # Call show method if available
            if hasattr(progress_tracker, 'show'):
                progress_tracker.show()
            
            # Initialize progress
            if hasattr(progress_tracker, 'update_overall'):
                progress_tracker.update_overall(0, "🔍 Memulai validasi...")
        
        # Get backend config
        backend_config = _extract_and_enhance_config(ui_components)
        
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
            # Complete progress
            if progress_tracker:
                if hasattr(progress_tracker, 'complete'):
                    progress_tracker.complete("Dataset check completed")
                elif hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(100, "✅ Check selesai")
            
            show_ui_success(ui_components, final_message)
            
            # Log detailed info
            for result in results:
                log_to_accordion(ui_components, result, "info")
            
            enable_operation_buttons(ui_components)
            return True
        else:
            _error_cleanup(ui_components, source_msg)
            return False
            
    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        logger.error(error_msg)
        _error_cleanup(ui_components, error_msg)
        return False

def _error_cleanup(ui_components: Dict[str, Any], error_msg: str):
    """Cleanup UI state setelah error"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'error'):
            progress_tracker.error(error_msg)
        elif hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(0, f"❌ Error")
    
    handle_ui_error(ui_components, error_msg)
    enable_operation_buttons(ui_components)