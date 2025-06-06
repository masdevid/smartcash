"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Fixed download handler dengan proper integration ke smartcash.dataset.downloader dan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import try_operation_safe, show_status_safe
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.dataset.downloader.handlers.check_handler import setup_check_handler
from smartcash.ui.dataset.downloader.handlers.cleanup_handler import setup_cleanup_handler
from smartcash.ui.dataset.downloader.handlers.validation_handler import setup_validation_handler
from smartcash.ui.dataset.downloader.utils.operation_utils import get_streamlined_download_operations
from smartcash.dataset.downloader.download_service import create_download_service  # Fixed import
from smartcash.common.logger import get_logger

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handlers dengan fixed integration"""
    
    logger = ui_components.get('logger') or get_logger('downloader.handlers')
    
    # Setup handlers dengan fixed wrapper
    check_handler = _wrap_with_state_management(setup_check_handler(ui_components, config, logger), ui_components, logger)
    cleanup_handler = _wrap_with_state_management(setup_cleanup_handler(ui_components, config, logger), ui_components, logger)
    validation_handler = setup_validation_handler(ui_components, config, logger)
    download_handler = _wrap_with_state_management(_create_fixed_download_handler(ui_components, config, logger), ui_components, logger)
    
    # Bind handlers dengan fixed button management
    _bind_button_handlers_fixed(ui_components, {
        'check_handler': check_handler, 'download_handler': download_handler, 'cleanup_handler': cleanup_handler
    })
    
    ui_components.update({
        'check_handler': check_handler, 'download_handler': download_handler,
        'cleanup_handler': cleanup_handler, 'validation_handler': validation_handler
    })
    
    return ui_components

def _wrap_with_state_management(handler: Callable, ui_components: Dict[str, Any], logger) -> Callable:
    """Fixed wrapper dengan proper button state management"""
    
    def state_managed_handler(button):
        """Fixed wrapper untuk mutual exclusion buttons"""
        all_buttons = []
        try:
            # Get all buttons safely - one-liner collection
            all_buttons = (getattr(button, '_all_buttons', None) or 
                          [ui_components.get(btn) for btn in ['check_button', 'download_button', 'cleanup_button']] or [])
            all_buttons = [btn for btn in all_buttons if btn is not None and hasattr(btn, 'disabled')]
            
            # Disable all buttons - one-liner
            [setattr(btn, 'disabled', True) for btn in all_buttons]
            
            # Execute handler
            handler(button)
            
        except Exception as e:
            logger and logger.error(f"❌ Error in handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
        finally:
            # Re-enable buttons - one-liner
            [setattr(btn, 'disabled', False) for btn in all_buttons]
    
    return state_managed_handler

def _create_fixed_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Create fixed download handler dengan proper service integration"""
    
    def handle_download(button):
        """Handle download dengan fixed validation dan service call"""
        try:
            # Get current config
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                show_status_safe("❌ Config handler tidak ditemukan", "error", ui_components)
                return
            
            current_config = config_handler.extract_config(ui_components)
            
            # Validate config
            validation = config_handler.validate_config(current_config)
            if not validation['valid']:
                error_msg = f"❌ Config tidak valid: {'; '.join(validation['errors'])}"
                show_status_safe(error_msg, "error", ui_components)
                return
            
            # Check existing dataset untuk confirmation
            operations = get_streamlined_download_operations()
            has_existing = operations.check_existing_dataset()
            
            if has_existing:
                _show_download_confirmation_fixed(ui_components, current_config, logger)
            else:
                _execute_download_sync_fixed(ui_components, current_config, logger)
                
        except Exception as e:
            logger.error(f"❌ Error download handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
    
    return handle_download

def _show_download_confirmation_fixed(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Show fixed confirmation dialog"""
    
    workspace, project, version = config.get('workspace', ''), config.get('project', ''), config.get('version', '')
    dataset_id = f"{workspace}/{project}:v{version}"
    
    # Fixed handlers dengan proper function definitions
    def on_confirm_handler(button):
        _clear_confirmation_area_fixed(ui_components)
        _execute_download_sync_fixed(ui_components, config, logger)
    
    def on_cancel_handler(button):
        _clear_confirmation_area_fixed(ui_components)
    
    confirmation_dialog = create_confirmation_dialog(
        title="Konfirmasi Download Dataset",
        message=f"""📥 **Dataset Download Confirmation**

🎯 **Target Dataset:** {dataset_id}

⚠️ **Perhatian:**
• Dataset existing akan ditimpa jika ada
• Backup otomatis: {'✅ Ya' if config.get('backup_existing') else '❌ Tidak'}
• Validasi hasil: {'✅ Ya' if config.get('validate_download') else '❌ Tidak'}
• Format: YOLOv5 PyTorch (hardcoded)

🚀 Lanjutkan download?""",
        on_confirm=on_confirm_handler,
        on_cancel=on_cancel_handler,
        confirm_text="Ya, Download", 
        cancel_text="Batal"
    )
    
    _show_in_confirmation_area_fixed(ui_components, confirmation_dialog)

def _execute_download_sync_fixed(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Execute download dengan proper service integration"""
    try:
        # Clear confirmation area
        _clear_confirmation_area_fixed(ui_components)
        
        # Get progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            logger.error("❌ Progress tracker tidak ditemukan")
            show_status_safe("❌ Progress tracker tidak tersedia", "error", ui_components)
            return
        
        # Show progress dengan API yang benar
        download_steps = ["validate", "connect", "metadata", "download", "extract", "organize"]
        step_weights = {"validate": 5, "connect": 10, "metadata": 15, "download": 50, "extract": 15, "organize": 5}
        progress_tracker.show("Download Dataset", download_steps, step_weights)
        
        # Create download service dengan proper integration - FIXED
        download_service = create_download_service(config, logger)
        
        # Setup fixed dual-level progress callback
        download_service.set_progress_callback(_create_fixed_progress_callback(progress_tracker, logger))
        
        # Execute download dengan proper method call - FIXED
        result = download_service.download_dataset(
            workspace=config['workspace'], 
            project=config['project'], 
            version=config['version'],
            api_key=config['api_key'], 
            output_format=config.get('output_format', 'yolov5pytorch'),
            validate_download=config.get('validate_download', True), 
            organize_dataset=True,
            backup_existing=config.get('backup_existing', False)
        )
        
        # Handle result dengan fixed status display
        if result['status'] == 'success':
            success_msg = f"✅ Dataset berhasil didownload: {result['stats']['total_images']} gambar"
            progress_tracker.complete(success_msg)
            show_status_safe(f"{success_msg} ke {result['output_dir']}", "success", ui_components)
            logger.success(success_msg)
        else:
            error_msg = f"❌ Download gagal: {result['message']}"
            progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            logger.error(error_msg)
            
    except Exception as e:
        error_msg = f"❌ Error saat download: {str(e)}"
        progress_tracker = ui_components.get('progress_tracker')
        progress_tracker and progress_tracker.error(error_msg)
        show_status_safe(error_msg, "error", ui_components)
        logger.error(error_msg)

def _create_fixed_progress_callback(progress_tracker, logger) -> Callable:
    """Create fixed dual-level progress callback dengan API yang benar"""
    
    def progress_callback(step: str, current: int, total: int, message: str):
        """Fixed dual-level progress dengan API progress_tracker yang benar"""
        try:
            percentage = min(100, max(0, int((current / total) * 100) if total > 0 else 0))
            
            # Map steps ke progress levels dengan API yang benar
            step_mapping = {
                'validate': ('overall', "🔄 Validasi parameter"),
                'connect': ('overall', "🌐 Koneksi Roboflow"),
                'metadata': ('overall', "📊 Ambil metadata"), 
                'download': ('current', "📥 Download dataset"),
                'extract': ('current', "📦 Ekstrak files"),
                'organize': ('current', "🗂️ Organisir struktur")
            }
            
            if step in step_mapping:
                level_name, overall_msg = step_mapping[step]
                
                # Update dengan API yang benar - gunakan update method
                if level_name == 'overall':
                    # Update overall progress untuk high-level steps
                    progress_tracker.update('overall', percentage, overall_msg)
                else:
                    # Update current progress untuk detail steps
                    progress_tracker.update('current', percentage, message)
            else:
                # Generic progress update untuk unknown steps
                progress_tracker.update('current', percentage, message)
                
        except Exception as e:
            logger.debug(f"🔍 Progress callback error: {str(e)}")
    
    return progress_callback

def _bind_button_handlers_fixed(ui_components: Dict[str, Any], handlers: Dict[str, Callable]) -> None:
    """Bind button handlers dengan fixed approach - one-liner"""
    button_mappings = [
        ('check_button', handlers.get('check_handler')),
        ('download_button', handlers.get('download_handler')),
        ('cleanup_button', handlers.get('cleanup_handler'))
    ]
    
    # One-liner binding dengan safe checks
    [ui_components[btn_name].on_click(handler) 
     for btn_name, handler in button_mappings 
     if btn_name in ui_components and handler is not None and hasattr(ui_components[btn_name], 'on_click')]

def _show_in_confirmation_area_fixed(ui_components: Dict[str, Any], dialog_widget) -> None:
    """Show dialog dalam confirmation area dengan fixed display"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        try:
            confirmation_area.layout.display = 'block'
            confirmation_area.layout.visibility = 'visible'
            with confirmation_area:
                confirmation_area.clear_output(wait=True)
                from IPython.display import display
                display(dialog_widget)
        except Exception:
            pass

def _clear_confirmation_area_fixed(ui_components: Dict[str, Any]) -> None:
    """Clear confirmation area dengan fixed approach"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        try:
            confirmation_area.clear_output(wait=True)
            confirmation_area.layout.display = 'none'
            confirmation_area.layout.visibility = 'hidden'
        except Exception:
            pass

# Fixed utilities dengan one-liner style
get_download_status_fixed = lambda ui: {'ready': 'progress_tracker' in ui, 'handlers_count': len([k for k in ui.keys() if k.endswith('_handler')]), 'buttons_available': all(btn in ui for btn in ['check_button', 'download_button', 'cleanup_button'])}
validate_handlers_setup_fixed = lambda ui: all(handler in ui for handler in ['check_handler', 'download_handler', 'cleanup_handler'])
get_handler_summary_fixed = lambda ui: f"✅ Handlers setup: {len([k for k in ui.keys() if k.endswith('_handler')])} handlers | Progress: {'✅' if 'progress_tracker' in ui else '❌'}"