"""
File: smartcash/ui/dataset/augmentation/utils/operation_handlers.py
Deskripsi: Fixed operation handlers dengan proper cleanup binding dan backend integration
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

def handle_augmentation_execution(ui_components: Dict[str, Any]):
    """Handle augmentation execution dengan smart cleanup detection"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import validate_form_inputs, clear_ui_outputs
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _execute_pipeline(ui_components):
        clear_ui_outputs(ui_components)
        
        # Form validation
        validation = validate_form_inputs(ui_components)
        if not validation['valid']:
            _show_validation_errors(ui_components, validation)
            return
        
        # Smart cleanup detection
        if _should_offer_cleanup_before_augmentation(ui_components):
            _show_cleanup_before_augmentation_dialog(ui_components)
            return
        
        # Execute augmentation directly
        _execute_augmentation_pipeline(ui_components)
    
    _execute_pipeline(ui_components)

def handle_cleanup_with_confirmation(ui_components: Dict[str, Any], skip_confirmation: bool = False):
    """FIXED: Handle cleanup dengan proper confirmation dan execution"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import clear_ui_outputs
    
    if skip_confirmation:
        clear_ui_outputs(ui_components)
        _execute_cleanup(ui_components)
        return
    
    # Show confirmation dialog menggunakan Dialog API
    from smartcash.ui.components.dialogs import show_destructive_confirmation
    
    def on_confirm_cleanup(button):
        clear_ui_outputs(ui_components)
        _execute_cleanup(ui_components)
    
    def on_cancel_cleanup(button):
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, "❌ Cleanup dibatalkan", "info")
    
    show_destructive_confirmation(
        title="Konfirmasi Cleanup Dataset",
        message="Apakah Anda yakin ingin menghapus semua file augmented dan preprocessed?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!",
        item_name="file augmented",
        on_confirm=on_confirm_cleanup,
        on_cancel=on_cancel_cleanup
    )

def _execute_cleanup(ui_components: Dict[str, Any]):
    """FIXED: Execute cleanup operation dengan proper backend integration"""
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _cleanup_operation(ui_components):
        logger = get_logger('smartcash.ui.dataset.augmentation')
        
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.show()
            progress_tracker.update_overall(0, "Memulai cleanup...")
        
        try:
            # CRITICAL FIX: Use proper service import
            from smartcash.dataset.augmentor.service import create_augmentation_service
            from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
            
            ui_config = extract_augmentation_config(ui_components)
            
            # Create service dengan proper config
            service = create_augmentation_service(ui_config, progress_tracker)
            
            if progress_tracker:
                progress_tracker.update_overall(25, "Mencari file yang akan dihapus...")
            
            # Execute cleanup via service
            result = service.cleanup_augmented_data()
            
            if progress_tracker:
                progress_tracker.update_overall(100, "Cleanup selesai")
            
            _handle_cleanup_result(ui_components, result)
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {str(e)}")
            _handle_operation_error(ui_components, e, 'cleanup')
    
    _cleanup_operation(ui_components)

def _execute_augmentation_pipeline(ui_components: Dict[str, Any]):
    """FIXED: Execute augmentation pipeline dengan proper backend integration"""
    logger = get_logger('smartcash.ui.dataset.augmentation')
    
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        progress_tracker.show()
    
    try:
        # CRITICAL FIX: Use proper service import
        from smartcash.dataset.augmentor.service import create_augmentation_service
        from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
        
        ui_config = extract_augmentation_config(ui_components)
        
        # Create service dengan proper config
        service = create_augmentation_service(ui_config, progress_tracker)
        
        def progress_callback(level, current, total, message):
            if progress_tracker:
                progress_pct = int((current/total)*100) if total > 0 else current
                if level == 'overall':
                    progress_tracker.update_overall(progress_pct, message)
                elif level == 'step':
                    progress_tracker.update_step(progress_pct, message)
                elif level == 'current':
                    progress_tracker.update_current(progress_pct, message)
        
        result = service.run_augmentation_pipeline(
            target_split=ui_config['augmentation']['target_split'],
            progress_callback=progress_callback
        )
        
        _handle_pipeline_result(ui_components, result, 'augmentation')
        
    except Exception as e:
        logger.error(f"❌ Augmentation pipeline error: {str(e)}")
        _handle_operation_error(ui_components, e, 'augmentation pipeline')

def handle_dataset_check(ui_components: Dict[str, Any]):
    """FIXED: Handle dataset check dengan proper backend integration"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import clear_ui_outputs
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _execute_check(ui_components):
        clear_ui_outputs(ui_components)
        
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.show()
            progress_tracker.update_overall(0, "Memulai comprehensive dataset check...")
        
        try:
            # CRITICAL FIX: Use proper service import
            from smartcash.dataset.augmentor.service import create_augmentation_service
            from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
            
            ui_config = extract_augmentation_config(ui_components)
            
            if progress_tracker:
                progress_tracker.update_overall(25, "Menganalisis raw data...")
            
            # Create service dan get status
            service = create_augmentation_service(ui_config, progress_tracker)
            result = service.get_augmentation_status()
            
            if progress_tracker:
                progress_tracker.update_overall(100, "Dataset analysis completed")
            
            _handle_enhanced_check_result(ui_components, result)
            
        except Exception as e:
            logger = get_logger('smartcash.ui.dataset.augmentation')
            logger.error(f"❌ Dataset check error: {str(e)}")
            _handle_operation_error(ui_components, e, 'dataset check')
    
    _execute_check(ui_components)

def _should_offer_cleanup_before_augmentation(ui_components: Dict[str, Any]) -> bool:
    """Check if cleanup should be offered before augmentation"""
    try:
        from smartcash.dataset.augmentor.service import create_augmentation_service
        from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
        
        ui_config = extract_augmentation_config(ui_components)
        target_split = ui_config['augmentation']['target_split']
        
        # Create service dan check status
        service = create_augmentation_service(ui_config)
        status = service.get_augmentation_status()
        
        aug_files = status.get(f'{target_split}_augmented', 0)
        prep_files = status.get(f'{target_split}_preprocessed', 0)
        
        return (aug_files > 0 or prep_files > 0)
    except Exception:
        return False

def _show_cleanup_before_augmentation_dialog(ui_components: Dict[str, Any]):
    """Show cleanup dialog menggunakan Dialog API"""
    from smartcash.ui.components.dialogs import show_confirmation
    from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
    
    ui_config = extract_augmentation_config(ui_components)
    target_split = ui_config['augmentation']['target_split']
    
    def on_cleanup_and_continue():
        """Cleanup dan continue dengan augmentation"""
        _execute_cleanup_then_augmentation(ui_components)
    
    def on_cancel():
        """Cancel operation"""
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, "❌ Augmentasi dibatalkan", "info")
    
    message = f"""File augmented ditemukan untuk split '{target_split}'.

Apakah Anda ingin menghapus file existing dan melanjutkan augmentasi?

⚠️ File yang akan dihapus:
- Augmented images (.jpg) dan labels (.txt)
- Preprocessed images (.npy) dan labels (.txt)"""
    
    show_confirmation(
        title="File Augmented Ditemukan",
        message=message,
        on_confirm=on_cleanup_and_continue,
        on_cancel=on_cancel
    )

def _execute_cleanup_then_augmentation(ui_components: Dict[str, Any]):
    """Execute cleanup then continue dengan augmentation"""
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _cleanup_then_augment(ui_components):
        logger = get_logger('smartcash.ui.dataset.augmentation')
        
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.show()
            progress_tracker.update_overall(0, "Memulai cleanup...")
        
        try:
            from smartcash.dataset.augmentor.service import create_augmentation_service
            from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            
            ui_config = extract_augmentation_config(ui_components)
            target_split = ui_config['augmentation']['target_split']
            
            # Create service
            service = create_augmentation_service(ui_config, progress_tracker)
            
            # Phase 1: Cleanup
            if progress_tracker:
                progress_tracker.update_overall(25, "Menghapus file existing...")
            
            log_to_ui(ui_components, f"🧹 Menghapus file augmented untuk {target_split}...", "info")
            cleanup_result = service.cleanup_augmented_data(target_split)
            
            if cleanup_result.get('status') != 'success':
                error_msg = f"❌ Cleanup gagal: {cleanup_result.get('message', 'Unknown error')}"
                logger.error(error_msg)
                if progress_tracker:
                    progress_tracker.error(error_msg)
                return
            
            # Log cleanup success
            total_removed = cleanup_result.get('total_removed', 0)
            log_to_ui(ui_components, f"✅ Cleanup berhasil: {total_removed} file dihapus", "success")
            
            # Phase 2: Continue dengan augmentation
            if progress_tracker:
                progress_tracker.update_overall(50, "Memulai augmentasi...")
            
            log_to_ui(ui_components, "🚀 Melanjutkan dengan augmentasi pipeline...", "info")
            _execute_augmentation_pipeline(ui_components)
            
        except Exception as e:
            error_msg = f"❌ Error cleanup & augmentation: {str(e)}"
            logger.error(error_msg)
            if progress_tracker:
                progress_tracker.error(error_msg)
    
    _cleanup_then_augment(ui_components)

def _show_validation_errors(ui_components: Dict[str, Any], validation: Dict[str, Any]):
    """Show validation errors menggunakan Dialog API"""
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
    """Handle pipeline result dengan enhanced feedback"""
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
        
        # Log enhanced summary
        pipeline_summary = result.get('pipeline_summary', {})
        if pipeline_summary:
            overall = pipeline_summary.get('overall', {})
            files_flow = overall.get('files_flow', '')
            if files_flow:
                log_to_ui(ui_components, f"🔄 Files flow: {files_flow}", "info")
    else:
        error_msg = f"❌ {operation.title()} gagal: {result.get('message', 'Unknown error')}"
        log_to_ui(ui_components, error_msg, "error")
        
        if progress_tracker:
            progress_tracker.error(error_msg)

def _handle_enhanced_check_result(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Handle enhanced check result dengan comprehensive feedback"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
    
    progress_tracker = ui_components.get('progress_tracker')
    
    if result.get('service_ready'):
        # Log enhanced dataset analysis
        for split in ['train', 'valid', 'test']:
            raw_imgs = result.get(f'{split}_raw_images', 0)
            raw_status = result.get(f'{split}_raw_status', 'unknown')
            aug_imgs = result.get(f'{split}_augmented', 0)
            aug_status = result.get(f'{split}_aug_status', 'unknown')
            prep_imgs = result.get(f'{split}_preprocessed', 0)
            prep_status = result.get(f'{split}_prep_status', 'unknown')
            
            # Status icons dan messages
            raw_icon = "✅" if raw_status == 'available' else "❌" if raw_status == 'not_found' else "⚠️"
            aug_icon = "✅" if aug_status == 'available' else "❌" if aug_status == 'not_found' else "⚠️"
            prep_icon = "✅" if prep_status == 'available' else "❌" if prep_status == 'not_found' else "⚠️"
            
            log_to_ui(ui_components, f"📂 {split.upper()}: {raw_icon} {raw_imgs} raw | {aug_icon} {aug_imgs} augmented | {prep_icon} {prep_imgs} preprocessed", "info")
        
        success_msg = "✅ Enhanced dataset check completed"
        log_to_ui(ui_components, success_msg, "success")
        if progress_tracker:
            progress_tracker.complete(success_msg)
    else:
        error_msg = f"❌ Check gagal: {result.get('error', 'Service not ready')}"
        log_to_ui(ui_components, error_msg, "error")
        if progress_tracker:
            progress_tracker.error(error_msg)

def _handle_cleanup_result(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Handle cleanup result dengan enhanced feedback"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
    
    progress_tracker = ui_components.get('progress_tracker')
    
    if result.get('status') == 'success':
        total_removed = result.get('total_removed', 0)
        if total_removed > 0:
            success_msg = f"✅ Cleanup berhasil: {total_removed} file dihapus"
            log_to_ui(ui_components, success_msg, "success")
            
            # Log cleanup summary if available
            cleanup_summary = result.get('cleanup_summary', {})
            if cleanup_summary:
                files_removed = cleanup_summary.get('files_removed', {})
                for split, counts in files_removed.items():
                    if counts.get('total', 0) > 0:
                        log_to_ui(ui_components, f"  📂 {split}: {counts['augmented']} aug + {counts['preprocessed']} prep = {counts['total']} files", "info")
            
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