"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handlers.py
Deskripsi: Optimized handlers dengan consolidated operations dan simplified binding
"""

from typing import Dict, Any, Callable
from smartcash.common.logger import get_logger

def setup_augmentation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan optimized operations dan consolidated logic"""
    
    # Setup config handler dengan UI integration
    config_handler = ui_components.get('config_handler')
    if config_handler and hasattr(config_handler, 'set_ui_components'):
        config_handler.set_ui_components(ui_components)
    
    # Setup handlers dengan consolidated approach
    _setup_operation_handlers_optimized(ui_components, config)
    _setup_config_handlers_optimized(ui_components, config)
    
    return ui_components

def _setup_operation_handlers_optimized(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Optimized operation handlers dengan consolidated operations"""
    
    # Base operation handler dengan shared logic
    def base_operation_handler(operation_type: str):
        """Base handler dengan shared preparation dan cleanup logic"""
        from smartcash.ui.dataset.augmentation.utils.ui_utils import clear_ui_outputs, validate_form_inputs
        from smartcash.ui.dataset.augmentation.utils.dialog_utils import clear_confirmation_area
        
        clear_ui_outputs(ui_components)
        clear_confirmation_area(ui_components)
        
        if operation_type == 'augment':
            validation = validate_form_inputs(ui_components)
            if not validation['valid']:
                _show_validation_errors_in_area(ui_components, validation)
                return
            
            if _should_offer_cleanup_before_augmentation(ui_components):
                _show_cleanup_before_augmentation_dialog(ui_components)
                return
            
            _execute_backend_operation(ui_components, 'augmentation_pipeline')
        
        elif operation_type == 'check':
            _execute_backend_operation(ui_components, 'dataset_check')
        
        elif operation_type == 'cleanup':
            _show_cleanup_confirmation_dialog(ui_components)
    
    # Operation handlers dengan base handler
    augment_handler = lambda btn=None: base_operation_handler('augment')
    check_handler = lambda btn=None: base_operation_handler('check')
    cleanup_handler = lambda btn=None: base_operation_handler('cleanup')
    
    # Bind handlers dengan error handling
    handlers = {
        'augment_button': augment_handler,
        'check_button': check_handler,
        'cleanup_button': cleanup_handler
    }
    
    _bind_handlers_safe(ui_components, handlers)

def _setup_config_handlers_optimized(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Optimized config handlers - save dengan validation, reset langsung"""
    from smartcash.ui.dataset.augmentation.utils.config_handlers import handle_save_config, handle_reset_config
    
    # Config handlers dengan consolidated operations
    config_handlers = {
        'save_button': handle_save_config,
        'reset_button': handle_reset_config
    }
    
    _bind_handlers_safe(ui_components, config_handlers)

def _bind_handlers_safe(ui_components: Dict[str, Any], handlers: Dict[str, Callable]):
    """Safe handler binding dengan error handling"""
    for button_key, handler in handlers.items():
        button = ui_components.get(button_key)
        if button and hasattr(button, 'on_click'):
            try:
                # Langsung pasang handler baru, on_click akan menangani penggantian handler lama
                button.on_click(handler)
                
                # Log successful binding
                from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
                log_to_ui(ui_components, f"✅ {button_key.replace('_', ' ').title()} handler bound", "debug")
                
            except Exception as e:
                from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
                log_to_ui(ui_components, f"⚠️ Error binding {button_key}: {str(e)}", "warning")

def _execute_backend_operation(ui_components: Dict[str, Any], operation_type: str):
    """Consolidated backend operation execution"""
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _backend_operation(ui_components):
        logger = get_logger('smartcash.ui.dataset.augmentation')
        progress_tracker = ui_components.get('progress_tracker')
        
        try:
            # Import dan setup service
            from smartcash.dataset.augmentor.service import create_augmentation_service
            from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            
            ui_config = extract_augmentation_config(ui_components)
            service = create_augmentation_service(ui_config, progress_tracker)
            
            if progress_tracker:
                progress_tracker.show()
            
            # Execute operation berdasarkan type
            if operation_type == 'augmentation_pipeline':
                log_to_ui(ui_components, "🚀 Memulai augmentation pipeline...", "info")
                
                def progress_callback(level, current, total, message):
                    if progress_tracker:
                        progress_pct = int((current/total)*100) if total > 0 else current
                        getattr(progress_tracker, f'update_{level}', lambda p, m: None)(progress_pct, message)
                
                result = service.run_augmentation_pipeline(
                    target_split=ui_config['augmentation']['target_split'],
                    progress_callback=progress_callback
                )
                _handle_pipeline_result(ui_components, result)
            
            elif operation_type == 'dataset_check':
                if progress_tracker:
                    progress_tracker.update_overall(25, "Menganalisis dataset...")
                
                log_to_ui(ui_components, "🔍 Memulai comprehensive dataset check...", "info")
                result = service.get_augmentation_status()
                
                if progress_tracker:
                    progress_tracker.update_overall(100, "Check completed")
                
                _handle_check_result(ui_components, result)
            
            elif operation_type == 'cleanup':
                if progress_tracker:
                    progress_tracker.update_overall(25, "Mencari file untuk dihapus...")
                
                log_to_ui(ui_components, "🧹 Memulai cleanup augmented files...", "info")
                # Membersihkan file augmented saja untuk kompatibilitas dengan kode yang ada
                result = service.cleanup_augmented_data(target_split=ui_config['augmentation']['target_split'])
                
                if progress_tracker:
                    progress_tracker.update_overall(100, "Cleanup selesai")
                
                _handle_cleanup_result(ui_components, result)
            
        except Exception as e:
            error_msg = f"❌ {operation_type.replace('_', ' ').title()} error: {str(e)}"
            logger.error(error_msg)
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            log_to_ui(ui_components, error_msg, "error")
            if progress_tracker:
                progress_tracker.error(error_msg)
    
    _backend_operation(ui_components)

def _execute_config_operation(ui_components: Dict[str, Any], operation_type: str):
    """Consolidated config operation execution"""
    try:
        config_handler = ui_components.get('config_handler')
        if not config_handler:
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            log_to_ui(ui_components, "❌ Config handler tidak tersedia", "error")
            return
        
        if hasattr(config_handler, 'set_ui_components'):
            config_handler.set_ui_components(ui_components)
        
        if operation_type == 'save':
            success = config_handler.save_config(ui_components)
            if success:
                _show_success_info(ui_components, "Save Config Berhasil", 
                    "✅ Konfigurasi berhasil disimpan!\n\nKonfigurasi telah disimpan ke file dan UI telah direfresh.")
        
        elif operation_type == 'reset':
            success = config_handler.reset_config(ui_components)
            if success:
                _show_success_info(ui_components, "Reset Config Berhasil",
                    "✅ Konfigurasi berhasil direset!\n\nSemua pengaturan telah dikembalikan ke nilai default.")
            
    except Exception as e:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, f"❌ Error {operation_type} config: {str(e)}", "error")

def _show_cleanup_confirmation_dialog(ui_components: Dict[str, Any]):
    """Show cleanup confirmation dialog"""
    from smartcash.ui.dataset.augmentation.utils.dialog_utils import show_confirmation_in_area
    
    def on_confirm_cleanup(btn):
        from smartcash.ui.dataset.augmentation.utils.dialog_utils import clear_confirmation_area
        clear_confirmation_area(ui_components)
        _execute_backend_operation(ui_components, 'cleanup')
    
    def on_cancel_cleanup(btn):
        from smartcash.ui.dataset.augmentation.utils.dialog_utils import clear_confirmation_area
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        clear_confirmation_area(ui_components)
        log_to_ui(ui_components, "❌ Cleanup dibatalkan", "info")
    
    show_confirmation_in_area(
        ui_components,
        title="Konfirmasi Cleanup Dataset",
        message="Apakah Anda yakin ingin menghapus semua file augmented dan preprocessed?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!",
        on_confirm=on_confirm_cleanup,
        on_cancel=on_cancel_cleanup,
        confirm_text="Ya, Hapus",
        cancel_text="Batal",
        danger_mode=True
    )



def _should_offer_cleanup_before_augmentation(ui_components: Dict[str, Any]) -> bool:
    """Check if cleanup should be offered before augmentation"""
    try:
        from smartcash.dataset.augmentor.service import create_augmentation_service
        from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
        
        ui_config = extract_augmentation_config(ui_components)
        target_split = ui_config['augmentation']['target_split']
        
        service = create_augmentation_service(ui_config)
        status = service.get_augmentation_status()
        
        aug_files = status.get(f'{target_split}_augmented', 0)
        prep_files = status.get(f'{target_split}_preprocessed', 0)
        
        return (aug_files > 0 or prep_files > 0)
    except Exception:
        return False

def _show_cleanup_before_augmentation_dialog(ui_components: Dict[str, Any]):
    """Show cleanup before augmentation dialog"""
    from smartcash.ui.dataset.augmentation.utils.dialog_utils import show_confirmation_in_area
    from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
    
    ui_config = extract_augmentation_config(ui_components)
    target_split = ui_config['augmentation']['target_split']
    
    def on_cleanup_and_continue(btn):
        from smartcash.ui.dataset.augmentation.utils.dialog_utils import clear_confirmation_area
        clear_confirmation_area(ui_components)
        _execute_cleanup_then_augmentation(ui_components)
    
    def on_cancel(btn):
        from smartcash.ui.dataset.augmentation.utils.dialog_utils import clear_confirmation_area
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        clear_confirmation_area(ui_components)
        log_to_ui(ui_components, "❌ Augmentasi dibatalkan", "info")
    
    message = f"""File augmented ditemukan untuk split '{target_split}'.

Apakah Anda ingin menghapus file existing dan melanjutkan augmentasi?

⚠️ File yang akan dihapus:
• Augmented images (.jpg) dan labels (.txt)
• Preprocessed images (.npy) dan labels (.txt)"""
    
    show_confirmation_in_area(
        ui_components,
        title="File Augmented Ditemukan",
        message=message,
        on_confirm=on_cleanup_and_continue,
        on_cancel=on_cancel,
        confirm_text="Ya, Hapus & Lanjutkan",
        cancel_text="Batal"
    )

def _execute_cleanup_then_augmentation(ui_components: Dict[str, Any]):
    """Execute cleanup then augmentation dengan sequential operations"""
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _cleanup_then_augment(ui_components):
        logger = get_logger('smartcash.ui.dataset.augmentation')
        progress_tracker = ui_components.get('progress_tracker')
        
        try:
            from smartcash.dataset.augmentor.service import create_augmentation_service
            from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            
            ui_config = extract_augmentation_config(ui_components)
            target_split = ui_config['augmentation']['target_split']
            service = create_augmentation_service(ui_config, progress_tracker)
            
            if progress_tracker:
                progress_tracker.show()
                progress_tracker.update_overall(0, "Memulai cleanup...")
            
            # Phase 1: Cleanup
            log_to_ui(ui_components, f"🧹 Menghapus file augmented untuk {target_split}...", "info")
            # Membersihkan file augmented saja untuk kompatibilitas dengan kode yang ada
            cleanup_result = service.cleanup_augmented_data(target_split=target_split)
            
            if cleanup_result.get('status') != 'success':
                error_msg = f"❌ Cleanup gagal: {cleanup_result.get('message', 'Unknown error')}"
                logger.error(error_msg)
                log_to_ui(ui_components, error_msg, "error")
                if progress_tracker:
                    progress_tracker.error(error_msg)
                return
            
            total_removed = cleanup_result.get('total_removed', 0)
            log_to_ui(ui_components, f"✅ Cleanup berhasil: {total_removed} file dihapus", "success")
            
            # Phase 2: Augmentation
            if progress_tracker:
                progress_tracker.update_overall(50, "Memulai augmentasi...")
            
            log_to_ui(ui_components, "🚀 Melanjutkan dengan augmentasi pipeline...", "info")
            
            def progress_callback(level, current, total, message):
                if progress_tracker:
                    # Map progress ke 50-100% range
                    mapped_progress = 50 + int((current/total)*50) if total > 0 else current
                    if level == 'overall':
                        progress_tracker.update_overall(mapped_progress, message)
                    else:
                        getattr(progress_tracker, f'update_{level}', lambda p, m: None)(current, message)
            
            result = service.run_augmentation_pipeline(
                target_split=target_split,
                progress_callback=progress_callback
            )
            
            _handle_pipeline_result(ui_components, result)
            
        except Exception as e:
            error_msg = f"❌ Error cleanup & augmentation: {str(e)}"
            logger.error(error_msg)
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            log_to_ui(ui_components, error_msg, "error")
            if progress_tracker:
                progress_tracker.error(error_msg)
    
    _cleanup_then_augment(ui_components)



def _show_success_info(ui_components: Dict[str, Any], title: str, message: str):
    """Show success info di confirmation area"""
    from smartcash.ui.dataset.augmentation.utils.dialog_utils import show_info_in_area
    
    show_info_in_area(
        ui_components,
        title=title,
        message=message,
        on_close=lambda btn: None
    )

# Result handlers optimized
def _handle_pipeline_result(ui_components: Dict[str, Any], result: Dict[str, Any]):
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
        
        # Show success info
        pipeline_summary = result.get('pipeline_summary', {})
        overall = pipeline_summary.get('overall', {})
        files_flow = overall.get('files_flow', f"{total_generated} → {total_normalized}")
        
        _show_success_info(
            ui_components,
            "Pipeline Berhasil!",
            f"✅ Augmentation pipeline selesai!\n\n📊 Results:\n• {files_flow}\n• Processing time: {processing_time:.1f}s"
        )
    else:
        error_msg = f"❌ Pipeline gagal: {result.get('message', 'Unknown error')}"
        log_to_ui(ui_components, error_msg, "error")
        if progress_tracker:
            progress_tracker.error(error_msg)

def _handle_check_result(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Handle check result dengan comprehensive feedback"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
    
    progress_tracker = ui_components.get('progress_tracker')
    
    if result.get('service_ready'):
        # Log dataset analysis
        for split in ['train', 'valid', 'test']:
            raw_imgs = result.get(f'{split}_raw_images', 0)
            raw_status = result.get(f'{split}_raw_status', 'unknown')
            aug_imgs = result.get(f'{split}_augmented', 0)
            prep_imgs = result.get(f'{split}_preprocessed', 0)
            
            status_icon = "✅" if raw_status == 'available' else "❌"
            log_to_ui(ui_components, f"📂 {split.upper()}: {status_icon} {raw_imgs} raw | {aug_imgs} aug | {prep_imgs} prep", "info")
        
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
            
            # Log cleanup summary
            cleanup_summary = result.get('cleanup_summary', {})
            if cleanup_summary:
                files_removed = cleanup_summary.get('files_removed', {})
                for split, counts in files_removed.items():
                    if counts.get('total', 0) > 0:
                        log_to_ui(ui_components, f"  📂 {split}: {counts['augmented']} aug + {counts['preprocessed']} prep = {counts['total']} files", "info")
            
            if progress_tracker:
                progress_tracker.complete(success_msg)
                
            # Show cleanup success info
            _show_success_info(
                ui_components,
                "Cleanup Berhasil!",
                f"✅ Cleanup selesai!\n\n📊 Summary:\n• {total_removed} file dihapus\n• Storage space dikosongkan"
            )
        else:
            info_msg = "💡 Cleanup selesai: tidak ada file untuk dihapus"
            log_to_ui(ui_components, info_msg, "info")
            if progress_tracker:
                progress_tracker.complete(info_msg)
            
            _show_success_info(
                ui_components,
                "Cleanup Selesai",
                "💡 Tidak ada file augmented yang perlu dihapus.\n\nDataset sudah bersih."
            )
    else:
        error_msg = f"❌ Cleanup gagal: {result.get('message', 'Unknown error')}"
        log_to_ui(ui_components, error_msg, "error")
        if progress_tracker:
            progress_tracker.error(error_msg)