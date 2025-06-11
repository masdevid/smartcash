"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Updated handlers dengan integrasi API preprocessor yang dikonsolidasi
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
    create_backend_cleanup_service_with_progress, _extract_and_enhance_config,
    get_preprocessing_samples, get_system_status
)
from smartcash.ui.dataset.preprocessing.utils.confirmation_utils import (
    show_preprocessing_confirmation, show_cleanup_confirmation, 
    should_execute_operation, is_confirmation_pending, clear_confirmation_area
)

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """🔧 Setup handlers dengan API integration dan improved error handling"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Check API compatibility first
        _log_api_compatibility_status(ui_components)
        
        # Setup config handlers
        _setup_config_handlers_fixed(ui_components)
        
        # Setup operation handlers dengan API integration
        _setup_operation_handlers_with_api(ui_components, config)
        
        logger.info("✅ Preprocessing handlers setup dengan API integration")
        return ui_components
        
    except Exception as e:
        error_msg = f"❌ Error setup handlers: {str(e)}"
        logger.error(error_msg)
        handle_ui_error(ui_components, error_msg)
        return ui_components

def _log_api_compatibility_status(ui_components: Dict[str, Any]):
    """Log API compatibility status ke UI"""
    try:
        from smartcash.ui.dataset.preprocessing.utils.backend_integration import validate_api_compatibility
        
        compatibility = validate_api_compatibility()
        if compatibility['compatible']:
            log_to_accordion(ui_components, f"🚀 {compatibility['message']}", "success")
            
            # Log available features
            enhanced_features = [
                "✅ Consolidated preprocessing pipeline",
                "✅ Real-time progress tracking", 
                "✅ YOLO-specific normalization",
                "✅ Enhanced validation with statistics",
                "✅ Automatic file organization"
            ]
            
            for feature in enhanced_features:
                log_to_accordion(ui_components, feature, "info")
        else:
            log_to_accordion(ui_components, f"⚠️ {compatibility['message']}", "warning")
            log_to_accordion(ui_components, "Menggunakan fallback implementation", "warning")
            
    except Exception as e:
        log_to_accordion(ui_components, f"⚠️ Cannot check API compatibility: {str(e)}", "warning")

def _setup_config_handlers_fixed(ui_components: Dict[str, Any]):
    """Setup config handlers dengan simplified approach"""
    
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
                
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error reset config: {str(e)}")
    
    # Bind handlers
    if save_button := ui_components.get('save_button'):
        save_button.on_click(simple_save_config)
    if reset_button := ui_components.get('reset_button'):
        reset_button.on_click(simple_reset_config)

def _setup_operation_handlers_with_api(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """🔧 Setup operation handlers dengan API integration"""
    
    def preprocessing_handler(button=None):
        return _handle_preprocessing_request_with_api(ui_components, config)
    
    def check_handler(button=None):
        return _execute_check_operation_with_api(ui_components, config)
    
    def cleanup_handler(button=None):
        return _handle_cleanup_request_with_api(ui_components, config)
    
    def samples_handler(button=None):
        return _execute_samples_operation_with_api(ui_components, config)
    
    # Bind handlers
    if preprocess_button := ui_components.get('preprocess_button'):
        preprocess_button.on_click(preprocessing_handler)
    if check_button := ui_components.get('check_button'):
        check_button.on_click(check_handler)
    if cleanup_button := ui_components.get('cleanup_button'):
        cleanup_button.on_click(cleanup_handler)
    
    # Add samples handler ke UI components untuk optional usage
    ui_components['_samples_handler'] = samples_handler

def _handle_preprocessing_request_with_api(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🚀 Handle preprocessing request dengan API integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        clear_outputs(ui_components)
        
        # Check confirmation state
        if should_execute_operation(ui_components, 'preprocessing'):
            return _execute_preprocessing_with_api(ui_components, config)
        
        if is_confirmation_pending(ui_components):
            log_to_accordion(ui_components, "⏳ Menunggu konfirmasi user...", "info")
            return False
        
        # Disable buttons dan validate
        disable_operation_buttons(ui_components)
        
        log_to_accordion(ui_components, "🔍 Validating dataset menggunakan API...", "info")
        backend_config = _extract_and_enhance_config(ui_components)
        
        is_valid, validation_msg = validate_dataset_ready(backend_config)
        if not is_valid:
            enable_operation_buttons(ui_components)
            handle_ui_error(ui_components, f"Pre-validation failed: {validation_msg}")
            return False
        
        log_to_accordion(ui_components, f"✅ Pre-validation: {validation_msg}", "success")
        
        # Show confirmation dengan enhanced info
        show_preprocessing_confirmation(ui_components, 
            f"📊 Dataset siap untuk diproses dengan API baru<br>"
            f"✅ {validation_msg}<br>"
            f"🎯 Features: YOLO normalization, real-time progress, enhanced validation<br><br>"
            f"Apakah Anda yakin ingin memulai preprocessing?")
        
        return True
        
    except Exception as e:
        error_msg = f"Error handling preprocessing request: {str(e)}"
        logger.error(error_msg)
        enable_operation_buttons(ui_components)
        handle_ui_error(ui_components, error_msg)
        return False

def _execute_preprocessing_with_api(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🔧 Execute preprocessing menggunakan consolidated API"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        from smartcash.dataset.preprocessor import preprocess_dataset
        
        # Setup progress tracker
        _force_show_progress_tracker(ui_components, "🚀 Memulai preprocessing dengan API...")
        
        log_to_accordion(ui_components, "🏗️ Creating preprocessing service dengan API integration...", "info")
        
        # Extract config yang compatible dengan API
        backend_config = _extract_and_enhance_config(ui_components)
        
        # Create progress callback
        progress_callback = _create_api_progress_callback(ui_components)
        
        # 🎯 CRITICAL: Direct API call dengan UI integration
        log_to_accordion(ui_components, "🚀 Starting consolidated preprocessing pipeline...", "info")
        
        result = preprocess_dataset(
            config=backend_config,
            ui_components=ui_components,
            progress_callback=progress_callback
        )
        
        if result.get('success', False):
            stats = result.get('stats', {})
            processed_count = stats.get('total_processed', 0)
            processing_time = result.get('processing_time', 0)
            success_rate = stats.get('success_rate', '100%')
            
            success_message = f"Preprocessing berhasil: {processed_count:,} gambar diproses dalam {processing_time:.1f} detik (Success rate: {success_rate})"
            
            _complete_progress_tracker(ui_components, success_message)
            show_ui_success(ui_components, success_message)
            
            # Log detailed stats dari API
            _log_api_processing_stats(ui_components, stats)
            
            enable_operation_buttons(ui_components)
            return True
        else:
            error_msg = result.get('message', 'Preprocessing failed')
            _error_cleanup(ui_components, error_msg)
            return False
            
    except Exception as e:
        error_msg = f"API preprocessing error: {str(e)}"
        logger.error(error_msg)
        _error_cleanup(ui_components, error_msg)
        return False

def _execute_check_operation_with_api(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🔍 Execute dataset check menggunakan API"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        from smartcash.dataset.preprocessor import validate_dataset, get_preprocessing_status
        
        clear_outputs(ui_components)
        disable_operation_buttons(ui_components)
        
        _force_show_progress_tracker(ui_components, "🔍 Memulai validasi dengan API...")
        
        backend_config = _extract_and_enhance_config(ui_components)
        
        # Validate source dataset menggunakan API
        log_to_accordion(ui_components, "🔍 Checking source dataset dengan API validation...", "info")
        _update_progress_tracker(ui_components, 'current', 25, "Validating source dataset...")
        
        target_split = backend_config.get('preprocessing', {}).get('target_splits', ['train'])[0]
        validation_result = validate_dataset(
            config=backend_config,
            target_split=target_split,
            ui_components=ui_components
        )
        
        # Check preprocessed data status
        _update_progress_tracker(ui_components, 'current', 75, "Checking preprocessed data...")
        log_to_accordion(ui_components, "💾 Checking preprocessed data dengan API status...", "info")
        
        status_result = get_preprocessing_status(
            config=backend_config,
            ui_components=ui_components
        )
        
        # Compile results
        results = []
        if validation_result.get('success', False):
            summary = validation_result.get('summary', {})
            total_images = summary.get('total_images', 0)
            validation_rate = summary.get('validation_rate', '0%')
            results.append(f"✅ Dataset sumber: {total_images:,} gambar (rate: {validation_rate})")
        else:
            results.append(f"❌ Dataset sumber: {validation_result.get('message', 'Invalid')}")
        
        if status_result.get('success', False):
            preprocessed_info = status_result.get('preprocessed_data', {})
            if preprocessed_info.get('exists', False):
                total_preprocessed = preprocessed_info.get('total_files', 0)
                results.append(f"💾 Preprocessed: {total_preprocessed:,} files tersedia")
            else:
                results.append("ℹ️ Belum ada data preprocessed")
        else:
            results.append("⚠️ Status preprocessed tidak dapat diperiksa")
        
        # Show results
        final_message = " | ".join(results)
        
        if validation_result.get('success', False):
            _complete_progress_tracker(ui_components, "Dataset check completed")
            show_ui_success(ui_components, final_message)
            
            # Log detailed info
            for result in results:
                log_to_accordion(ui_components, result, "info")
            
            enable_operation_buttons(ui_components)
            return True
        else:
            _error_cleanup(ui_components, validation_result.get('message', 'Validation failed'))
            return False
            
    except Exception as e:
        error_msg = f"API validation error: {str(e)}"
        logger.error(error_msg)
        _error_cleanup(ui_components, error_msg)
        return False

def _handle_cleanup_request_with_api(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🧹 Handle cleanup request menggunakan API"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        clear_outputs(ui_components)
        
        # Check confirmation state
        if should_execute_operation(ui_components, 'cleanup'):
            return _execute_cleanup_with_api(ui_components, config)
        
        if is_confirmation_pending(ui_components):
            log_to_accordion(ui_components, "⏳ Menunggu konfirmasi user...", "info")
            return False
        
        # Disable buttons dan check files
        disable_operation_buttons(ui_components)
        
        log_to_accordion(ui_components, "🔍 Checking preprocessed data dengan API...", "info")
        backend_config = _extract_and_enhance_config(ui_components)
        
        exists, detailed_msg = check_preprocessed_exists(backend_config)
        if not exists:
            enable_operation_buttons(ui_components)
            log_to_accordion(ui_components, "ℹ️ Tidak ada data untuk dibersihkan", "info")
            return True
        
        log_to_accordion(ui_components, f"📊 Data ditemukan: {detailed_msg}", "info")
        
        # Show confirmation dengan API-enhanced stats
        show_cleanup_confirmation(ui_components, f"API Detection: {detailed_msg}")
        
        return True
        
    except Exception as e:
        error_msg = f"Error handling cleanup request: {str(e)}"
        logger.error(error_msg)
        enable_operation_buttons(ui_components)
        handle_ui_error(ui_components, error_msg)
        return False

def _execute_cleanup_with_api(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🔧 Execute cleanup menggunakan consolidated API"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        from smartcash.dataset.preprocessor import cleanup_preprocessed_data
        
        _force_show_progress_tracker(ui_components, "🗑️ Memulai cleanup dengan API...")
        
        log_to_accordion(ui_components, "🏗️ Starting API cleanup process...", "info")
        
        backend_config = _extract_and_enhance_config(ui_components)
        
        # Execute cleanup menggunakan API
        result = cleanup_preprocessed_data(
            config=backend_config,
            ui_components=ui_components
        )
        
        if result.get('success', False):
            stats = result.get('stats', {})
            files_removed = stats.get('files_removed', 0)
            
            success_message = f"Cleanup berhasil: {files_removed:,} file dihapus menggunakan API"
            
            _complete_progress_tracker(ui_components, success_message)
            show_ui_success(ui_components, success_message)
            
            # Log cleanup details
            log_to_accordion(ui_components, f"🗑️ Files removed: {files_removed:,}", "info")
            
            enable_operation_buttons(ui_components)
            return True
        else:
            error_msg = result.get('message', 'API cleanup failed')
            _error_cleanup(ui_components, error_msg)
            return False
            
    except Exception as e:
        error_msg = f"API cleanup error: {str(e)}"
        logger.error(error_msg)
        _error_cleanup(ui_components, error_msg)
        return False

def _execute_samples_operation_with_api(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🎲 Execute get samples menggunakan API"""
    try:
        clear_outputs(ui_components)
        
        log_to_accordion(ui_components, "🎲 Getting dataset samples dengan API...", "info")
        
        samples_result = get_preprocessing_samples(ui_components, target_split="train", max_samples=5)
        
        if samples_result.get('success', False):
            samples = samples_result.get('samples', [])
            log_to_accordion(ui_components, f"📊 Found {len(samples)} samples:", "success")
            
            for i, sample in enumerate(samples, 1):
                filename = sample.get('filename', 'Unknown')
                dimensions = sample.get('dimensions', 'Unknown')
                log_to_accordion(ui_components, f"  {i}. {filename} - {dimensions}", "info")
            
            return True
        else:
            handle_ui_error(ui_components, samples_result.get('message', 'Failed to get samples'))
            return False
            
    except Exception as e:
        handle_ui_error(ui_components, f"Samples error: {str(e)}")
        return False

def _create_api_progress_callback(ui_components: Dict[str, Any]) -> callable:
    """Create progress callback yang compatible dengan API"""
    def api_progress_callback(level: str, current: int, total: int, message: str):
        try:
            progress_tracker = ui_components.get('progress_tracker')
            if not progress_tracker:
                return
            
            # Map API level ke UI tracker level
            if level in ['overall', 'primary']:
                if hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(current, message)
            elif level in ['step', 'current', 'batch', 'file']:
                if hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(current, message)
            
            # Log milestone progress to UI
            if _is_milestone_progress(current, total):
                log_to_accordion(ui_components, f"📊 {message} ({current}/{total})", "info")
                
        except Exception:
            pass  # Silent fail
    
    return api_progress_callback

def _log_api_processing_stats(ui_components: Dict[str, Any], stats: Dict[str, Any]):
    """Log detailed processing stats dari API"""
    try:
        if 'output' in stats:
            output_stats = stats['output']
            normalized_count = output_stats.get('total_normalized', 0)
            success_rate = output_stats.get('success_rate', '100%')
            
            if normalized_count > 0:
                log_to_accordion(ui_components, f"🎨 Normalisasi YOLO: {normalized_count:,} gambar", "info")
            
            log_to_accordion(ui_components, f"📈 Success rate: {success_rate}", "info")
        
        if 'validation' in stats:
            validation_stats = stats['validation']
            valid_files = validation_stats.get('valid_files', 0)
            invalid_files = validation_stats.get('invalid_files', 0)
            
            log_to_accordion(ui_components, f"✅ Valid files: {valid_files:,}", "info")
            if invalid_files > 0:
                log_to_accordion(ui_components, f"⚠️ Invalid files: {invalid_files:,}", "warning")
        
    except Exception:
        pass  # Silent fail

def _force_show_progress_tracker(ui_components: Dict[str, Any], initial_message: str):
    """Force show progress tracker dengan visibility fix"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'container'):
            container = progress_tracker.container
            if hasattr(container, 'layout'):
                container.layout.visibility = 'visible'
                container.layout.display = 'flex'
                container.layout.height = 'auto'
        
        if hasattr(progress_tracker, 'show'):
            progress_tracker.show()
        
        if hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(0, initial_message)

def _update_progress_tracker(ui_components: Dict[str, Any], level: str, progress: int, message: str):
    """Update progress tracker dengan level yang sesuai"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if level == 'overall' and hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(progress, message)
        elif level == 'current' and hasattr(progress_tracker, 'update_current'):
            progress_tracker.update_current(progress, message)

def _complete_progress_tracker(ui_components: Dict[str, Any], message: str):
    """Complete progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'complete'):
            progress_tracker.complete(message)
        elif hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(100, f"✅ {message}")

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

def _is_milestone_progress(current: int, total: int) -> bool:
    """Check if progress adalah milestone yang perlu di-log"""
    if total <= 10:
        return True
    
    milestones = [0, 10, 25, 50, 75, 90, 100]
    progress_pct = (current / total) * 100 if total > 0 else 0
    return any(abs(progress_pct - milestone) < 1 for milestone in milestones) or current == total