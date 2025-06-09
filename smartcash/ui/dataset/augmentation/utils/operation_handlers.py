"""
File: smartcash/ui/dataset/augmentation/utils/operation_handlers.py
Deskripsi: Fixed operation handlers dengan proper imports dan error handling
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

def handle_augmentation_execution(ui_components: Dict[str, Any]):
    """Handle augmentation execution dengan comprehensive pipeline"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import validate_form_inputs, clear_ui_outputs
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _execute_pipeline(ui_components):
        clear_ui_outputs(ui_components)
        logger = get_logger('smartcash.ui.dataset.augmentation')
        
        # Form validation
        validation = validate_form_inputs(ui_components)
        if not validation['valid']:
            _show_validation_errors(ui_components, validation)
            return
        
        # Start progress tracking
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.show()
        
        try:
            from smartcash.dataset.augmentor import augment_and_normalize
            from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
            
            # Extract config
            ui_config = extract_augmentation_config(ui_components)
            
            # Create progress callback
            def progress_callback(level, current, total, message):
                if progress_tracker:
                    if level == 'overall':
                        progress_tracker.update_overall(int((current/total)*100), message)
                    elif level == 'step':
                        progress_tracker.update_step(int((current/total)*100), message)
                    elif level == 'current':
                        progress_tracker.update_current(int((current/total)*100), message)
            
            # Execute augmentation pipeline
            result = augment_and_normalize(
                config=ui_config,
                target_split=ui_config['augmentation']['target_split'],
                progress_tracker=progress_tracker,
                progress_callback=progress_callback
            )
            
            # Handle result
            _handle_pipeline_result(ui_components, result, 'augmentation')
            
        except Exception as e:
            logger.error(f"❌ Augmentation pipeline error: {str(e)}")
            _handle_operation_error(ui_components, e, 'augmentation pipeline')
    
    _execute_pipeline(ui_components)

def handle_dataset_check(ui_components: Dict[str, Any]):
    """Handle comprehensive dataset check"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import clear_ui_outputs
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _execute_check(ui_components):
        clear_ui_outputs(ui_components)
        logger = get_logger('smartcash.ui.dataset.augmentation')
        
        # Start progress tracking
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.show()
            progress_tracker.update_overall(0, "Memulai pengecekan dataset...")
        
        try:
            from smartcash.dataset.augmentor import get_augmentation_status
            from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
            
            ui_config = extract_augmentation_config(ui_components)
            
            if progress_tracker:
                progress_tracker.update_overall(25, "Menganalisis dataset mentah...")
            
            # Execute check using status function
            result = get_augmentation_status(ui_config, progress_tracker)
            
            if progress_tracker:
                progress_tracker.update_overall(100, "Pengecekan selesai")
            
            # Handle result
            _handle_check_result(ui_components, result)
            
        except Exception as e:
            logger.error(f"❌ Dataset check error: {str(e)}")
            _handle_operation_error(ui_components, e, 'dataset check')
    
    _execute_check(ui_components)

def handle_cleanup_with_confirmation(ui_components: Dict[str, Any]):
    """Handle cleanup dengan confirmation dialog"""
    from smartcash.ui.components.dialogs import show_destructive_confirmation
    from smartcash.ui.dataset.augmentation.utils.ui_utils import clear_ui_outputs
    
    def on_confirm_cleanup(button):
        clear_ui_outputs(ui_components)
        _execute_cleanup(ui_components)
    
    def on_cancel_cleanup(button):
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, "❌ Cleanup dibatalkan", "info")
    
    # Show confirmation dialog
    show_destructive_confirmation(
        title="Konfirmasi Cleanup Dataset",
        message="Apakah Anda yakin ingin menghapus semua file augmented dan preprocessed?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!",
        item_name="file augmented",
        on_confirm=on_confirm_cleanup,
        on_cancel=on_cancel_cleanup
    )

def _execute_cleanup(ui_components: Dict[str, Any]):
    """Execute cleanup operation dengan progress tracking"""
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _cleanup_operation(ui_components):
        logger = get_logger('smartcash.ui.dataset.augmentation')
        
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.show()
            progress_tracker.update_overall(0, "Memulai cleanup...")
        
        try:
            from smartcash.dataset.augmentor import cleanup_augmented_data
            from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
            
            ui_config = extract_augmentation_config(ui_components)
            
            if progress_tracker:
                progress_tracker.update_overall(25, "Mencari file yang akan dihapus...")
            
            # Execute cleanup
            result = cleanup_augmented_data(ui_config, progress_tracker=progress_tracker)
            
            if progress_tracker:
                progress_tracker.update_overall(100, "Cleanup selesai")
            
            # Handle result
            _handle_cleanup_result(ui_components, result)
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {str(e)}")
            _handle_operation_error(ui_components, e, 'cleanup')
    
    _cleanup_operation(ui_components)

def _show_validation_errors(ui_components: Dict[str, Any], validation: Dict[str, Any]):
    """Show validation errors menggunakan dialog system"""
    from smartcash.ui.components.dialogs import show_warning
    from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
    
    error_messages = validation.get('errors', []) + validation.get('warnings', [])
    message = "Form validation menemukan masalah:\n\n" + "\n".join(error_messages)
    
    show_warning(
        title="Validation Error",
        message=message,
        on_close=lambda btn: log_to_ui(ui_components, "⚠️ Silakan perbaiki input form", "warning")
    )

def _handle_pipeline_result(ui_components: Dict[str, Any], result: Dict[str, Any], operation: str):
    """Handle pipeline result dengan detailed feedback"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
    
    progress_tracker = ui_components.get('progress_tracker')
    
    if result.get('status') == 'success':
        total_generated = result.get('total_generated', 0)
        total_normalized = result.get('total_normalized', 0)
        processing_time = result.get('processing_time', 0)
        
        success_msg = f"✅ Pipeline berhasil: {total_generated} generated, {total_normalized} normalized ({processing_time:.1f}s)"
        log_to_ui(ui_components, success_msg, "success")
        
        if progress_tracker:
            progress_tracker.complete(success_msg)
            
        # Log additional details
        if 'phases' in result:
            phases = result['phases']
            if 'augmentation' in phases:
                aug_result = phases['augmentation']
                log_to_ui(ui_components, f"📈 Augmentation success rate: {aug_result.get('success_rate', 0):.1f}%", "info")
            
            if 'symlinks' in phases:
                symlink_result = phases['symlinks']
                log_to_ui(ui_components, f"🔗 Symlinks created: {symlink_result.get('total_created', 0)}", "info")
    else:
        error_msg = f"❌ {operation.title()} gagal: {result.get('message', 'Unknown error')}"
        log_to_ui(ui_components, error_msg, "error")
        
        if progress_tracker:
            progress_tracker.error(error_msg)

def _handle_check_result(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Handle check result dengan comprehensive feedback"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
    
    progress_tracker = ui_components.get('progress_tracker')
    
    if result.get('service_ready'):
        # Log service info
        paths = result.get('paths', {})
        log_to_ui(ui_components, f"📁 Data directory: {paths.get('data_dir', 'Unknown')}", "info")
        
        # Check augmented files
        train_aug = result.get('train_augmented', 0)
        train_prep = result.get('train_preprocessed', 0)
        
        if train_aug > 0:
            log_to_ui(ui_components, f"✅ Found {train_aug} augmented files, {train_prep} preprocessed", "success")
        else:
            log_to_ui(ui_components, "💡 No augmented files found - ready for augmentation", "info")
        
        success_msg = "✅ Dataset check completed"
        log_to_ui(ui_components, success_msg, "success")
        if progress_tracker:
            progress_tracker.complete(success_msg)
    else:
        error_msg = f"❌ Check gagal: {result.get('error', 'Service not ready')}"
        log_to_ui(ui_components, error_msg, "error")
        if progress_tracker:
            progress_tracker.error(error_msg)

def _handle_cleanup_result(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Handle cleanup result dengan feedback"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
    
    progress_tracker = ui_components.get('progress_tracker')
    
    if result.get('status') == 'success':
        total_removed = result.get('total_removed', 0)
        if total_removed > 0:
            success_msg = f"✅ Cleanup berhasil: {total_removed} file dihapus"
            log_to_ui(ui_components, success_msg, "success")
            if progress_tracker:
                progress_tracker.complete(success_msg)
        else:
            info_msg = "💡 Cleanup selesai: tidak ada file untuk dihapus"
            log_to_ui(ui_components, info_msg, "info")
            if progress_tracker:
                progress_tracker.complete(info_msg)
    else:
        error_msg = f"❌ Cleanup gagal: {result.get('message', 'Unknown error')}"
        log_to_ui(ui_components, error_msg, "error")
        if progress_tracker:
            progress_tracker.error(error_msg)

def _handle_operation_error(ui_components: Dict[str, Any], error: Exception, operation: str):
    """Handle operation error dengan comprehensive logging"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
    
    error_msg = f"❌ {operation.title()} error: {str(error)}"
    log_to_ui(ui_components, error_msg, "error")
    
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        progress_tracker.error(error_msg)