"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handlers.py
Deskripsi: Complete handlers dengan dialog integration dan live preview handler
"""

from typing import Dict, Any, Callable, List
from smartcash.common.logger import get_logger

def setup_augmentation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan dialog integration dan live preview support"""
    
    # Setup config handler dengan UI integration
    config_handler = ui_components.get('config_handler')
    if config_handler and hasattr(config_handler, 'set_ui_components'):
        config_handler.set_ui_components(ui_components)
    
    # Setup handlers dengan dialog integration
    _setup_operation_handlers_with_dialog(ui_components, config)
    _setup_config_handlers_optimized(ui_components, config)
    _setup_preview_handler(ui_components, config)
    
    return ui_components

def _setup_operation_handlers_with_dialog(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup operation handlers dengan dialog confirmation menggunakan smartcash.ui.components.dialog"""
    
    def augment_handler(btn=None):
        from smartcash.ui.dataset.augmentation.utils.ui_utils import clear_ui_outputs, validate_form_inputs
        clear_ui_outputs(ui_components)
        
        # Validate form
        validation = validate_form_inputs(ui_components)
        if not validation['valid']:
            _show_validation_errors(ui_components, validation)
            return
        
        # Show confirmation dialog
        _show_augmentation_confirmation_dialog(ui_components)
    
    def check_handler(btn=None):
        from smartcash.ui.dataset.augmentation.utils.ui_utils import clear_ui_outputs
        clear_ui_outputs(ui_components)
        _execute_backend_operation(ui_components, 'dataset_check')
    
    def cleanup_handler(btn=None):
        from smartcash.ui.dataset.augmentation.utils.ui_utils import clear_ui_outputs
        clear_ui_outputs(ui_components)
        _show_cleanup_confirmation_dialog(ui_components)
    
    # Bind handlers dengan error handling
    handlers = {
        'augment_button': augment_handler,
        'check_button': check_handler,
        'cleanup_button': cleanup_handler
    }
    
    _bind_handlers_safe(ui_components, handlers)

def _setup_config_handlers_optimized(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup config handlers dengan save validation dan reset langsung"""
    from smartcash.ui.dataset.augmentation.utils.config_handlers import handle_save_config, handle_reset_config
    
    config_handlers = {
        'save_button': handle_save_config,
        'reset_button': handle_reset_config
    }
    
    _bind_handlers_safe(ui_components, config_handlers)

def _setup_preview_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup live preview handler untuk generate button"""
    
    def preview_handler(btn=None):
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        
        try:
            # Update status
            preview_status = ui_components.get('preview_status')
            if preview_status:
                preview_status.value = "<div style='text-align: center; color: #007bff; font-size: 12px;'>🔄 Generating preview...</div>"
            
            # Extract config untuk preview
            from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
            ui_config = extract_augmentation_config(ui_components)
            
            # Generate preview menggunakan backend service
            _generate_preview_image(ui_components, ui_config)
            
        except Exception as e:
            log_to_ui(ui_components, f"❌ Error generating preview: {str(e)}", "error")
            if preview_status:
                preview_status.value = "<div style='text-align: center; color: #dc3545; font-size: 12px;'>❌ Preview error</div>"
    
    # Bind preview handler
    generate_button = ui_components.get('generate_button')
    if generate_button and hasattr(generate_button, 'on_click'):
        generate_button.on_click(preview_handler)

def _show_augmentation_confirmation_dialog(ui_components: Dict[str, Any]):
    """Show augmentation confirmation menggunakan smartcash.ui.components.dialog"""
    try:
        from smartcash.ui.components.dialog import show_confirmation_dialog
        
        # Extract config untuk summary
        from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
        config = extract_augmentation_config(ui_components)
        aug_config = config.get('augmentation', {})
        
        # Create summary message
        summary_items = [
            f"🎯 Variasi: {aug_config.get('num_variations', 2)}",
            f"📊 Target: {aug_config.get('target_count', 500)}",
            f"🔄 Jenis: {', '.join(aug_config.get('types', ['combined']))}",
            f"📂 Split: {aug_config.get('target_split', 'train')}",
            f"🎚️ Intensitas: {aug_config.get('intensity', 0.7)}"
        ]
        
        message = "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px; margin: 8px 0;'>"
        message += "<br>".join(summary_items)
        message += "</div>"
        message += "<p style='font-size: 13px; margin-top: 8px;'>"
        message += "✅ Backend integration dengan progress tracking<br>"
        message += "🔄 FileNamingManager dengan variance support<br>"
        message += "📊 Real-time progress overall + current<br>"
        message += "🎯 YOLO normalization menggunakan preprocessor API</p>"
        
        show_confirmation_dialog(
            ui_components,
            title="🚀 Konfirmasi Augmentasi Pipeline",
            message=message,
            on_confirm=lambda: _execute_backend_operation(ui_components, 'augmentation_pipeline'),
            on_cancel=lambda: _handle_augmentation_cancel(ui_components),
            confirm_text="🚀 Mulai Augmentasi",
            cancel_text="❌ Batal"
        )
        
    except ImportError:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, "⚠️ Dialog tidak tersedia, langsung execute", "warning")
        _execute_backend_operation(ui_components, 'augmentation_pipeline')

def _show_cleanup_confirmation_dialog(ui_components: Dict[str, Any]):
    """Show cleanup confirmation menggunakan smartcash.ui.components.dialog"""
    try:
        from smartcash.ui.components.dialog import show_confirmation_dialog
        from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
        
        config = extract_augmentation_config(ui_components)
        cleanup_target = config.get('cleanup', {}).get('default_target', 'both')
        target_split = config.get('augmentation', {}).get('target_split', 'train')
        
        target_descriptions = {
            'augmented': '<strong>file augmented</strong> (aug_*.jpg + aug_*.txt + preprocessed *.npy)',
            'samples': '<strong>sample preview</strong> (sample_aug_*.jpg)',
            'both': '<strong>file augmented dan sample preview</strong>'
        }
        
        target_desc = target_descriptions.get(cleanup_target, cleanup_target)
        
        message = (
            f"Hapus {target_desc} dari split '{target_split}'?<br><br>"
            "<div style='background: #f8d7da; padding: 8px; border-radius: 4px; margin: 8px 0; color: #721c24;'>"
            "<strong>⚠️ Tindakan ini tidak dapat dibatalkan!</strong></div>"
            "<ul style='list-style: none; padding: 0; margin: 0;'><li>🗑️ Files akan dihapus permanent</li>"
            "<li>📊 Progress tracking tersedia</li><li>🧹 Target: {cleanup_target}</li></ul>".format(
                cleanup_target=cleanup_target
            )
        )
        
        show_confirmation_dialog(
            ui_components,
            title="⚠️ Konfirmasi Cleanup Dataset",
            message=message,
            on_confirm=lambda: _execute_backend_operation(ui_components, 'cleanup'),
            on_cancel=lambda: _handle_cleanup_cancel(ui_components),
            confirm_text="🗑️ Ya, Hapus",
            cancel_text="❌ Batal",
            danger_mode=True
        )
        
    except ImportError:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, "⚠️ Dialog tidak tersedia, langsung execute", "warning")
        _execute_backend_operation(ui_components, 'cleanup')

def _execute_backend_operation(ui_components: Dict[str, Any], operation_type: str):
    """Execute backend operation dengan proper error handling"""
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _backend_operation(ui_components):
        logger = get_logger('smartcash.ui.dataset.augmentation')
        progress_tracker = ui_components.get('progress_tracker')
        
        try:
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
                log_to_ui(ui_components, "🔍 Memulai comprehensive dataset check...", "info")
                result = service.get_augmentation_status()
                _handle_check_result(ui_components, result)
            
            elif operation_type == 'cleanup':
                log_to_ui(ui_components, "🧹 Memulai cleanup augmented files...", "info")
                cleanup_target = ui_config.get('cleanup', {}).get('default_target', 'both')
                result = service.cleanup_data(
                    target_split=ui_config['augmentation']['target_split'],
                    target=cleanup_target
                )
                _handle_cleanup_result(ui_components, result)
            
        except Exception as e:
            error_msg = f"❌ {operation_type.replace('_', ' ').title()} error: {str(e)}"
            logger.error(error_msg)
            log_to_ui(ui_components, error_msg, "error")
            if progress_tracker:
                progress_tracker.error(error_msg)
    
    _backend_operation(ui_components)

def _generate_preview_image(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Generate preview image ke /data/aug_preview.jpg"""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        import os
        
        # Create preview menggunakan backend service
        from smartcash.dataset.augmentor.service import create_augmentation_service
        service = create_augmentation_service(config)
        
        # Generate preview (service akan save ke /data/aug_preview.jpg)
        preview_result = service.create_live_preview(
            target_split=config['augmentation']['target_split']
        )
        
        if preview_result.get('success', False):
            # Load dan display image
            preview_path = '/data/aug_preview.jpg'
            if os.path.exists(preview_path):
                with open(preview_path, 'rb') as f:
                    image_data = f.read()
                
                preview_image = ui_components.get('preview_image')
                if preview_image:
                    preview_image.value = image_data
                
                preview_status = ui_components.get('preview_status')
                if preview_status:
                    preview_status.value = "<div style='text-align: center; color: #28a745; font-size: 12px;'>✅ Preview generated</div>"
                
                log_to_ui(ui_components, "✅ Preview berhasil di-generate", "success")
            else:
                raise FileNotFoundError("Preview file tidak ditemukan")
        else:
            error_msg = preview_result.get('message', 'Preview generation failed')
            raise Exception(error_msg)
            
    except Exception as e:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, f"❌ Error generating preview: {str(e)}", "error")
        
        preview_status = ui_components.get('preview_status')
        if preview_status:
            preview_status.value = "<div style='text-align: center; color: #dc3545; font-size: 12px;'>❌ Preview failed</div>"

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
            aug_imgs = result.get(f'{split}_augmented', 0)
            prep_imgs = result.get(f'{split}_preprocessed', 0)
            sample_imgs = result.get(f'{split}_sample_aug', 0)
            
            if aug_imgs > 0 or prep_imgs > 0:
                log_to_ui(ui_components, f"📂 {split.upper()}: {aug_imgs} aug | {prep_imgs} prep | {sample_imgs} samples", "info")
        
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
        target = result.get('target', 'both')
        
        success_msg = f"✅ Cleanup berhasil: {total_removed} files dihapus (target: {target})"
        log_to_ui(ui_components, success_msg, "success")
        
        if progress_tracker:
            progress_tracker.complete(success_msg)
    else:
        error_msg = f"❌ Cleanup gagal: {result.get('message', 'Unknown error')}"
        log_to_ui(ui_components, error_msg, "error")
        if progress_tracker:
            progress_tracker.error(error_msg)

def _handle_augmentation_cancel(ui_components: Dict[str, Any]):
    """Handle augmentation cancellation"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
    log_to_ui(ui_components, "🚫 Augmentasi dibatalkan oleh user", "info")

def _handle_cleanup_cancel(ui_components: Dict[str, Any]):
    """Handle cleanup cancellation"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
    log_to_ui(ui_components, "🚫 Cleanup dibatalkan oleh user", "info")

def _bind_handlers_safe(ui_components: Dict[str, Any], handlers: Dict[str, Callable]):
    """Safe handler binding dengan error handling"""
    for button_key, handler in handlers.items():
        button = ui_components.get(button_key)
        if button and hasattr(button, 'on_click'):
            try:
                button.on_click(handler)
            except Exception as e:
                from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
                log_to_ui(ui_components, f"⚠️ Error binding {button_key}: {str(e)}", "warning")

def _show_validation_errors(ui_components: Dict[str, Any], validation: Dict[str, Any]):
    """Show validation errors menggunakan UI utilities"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import show_validation_errors
    show_validation_errors(ui_components, validation)