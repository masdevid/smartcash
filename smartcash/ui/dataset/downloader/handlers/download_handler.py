"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Fixed download handler dengan error fixes untuk 'bool' object is not callable dan variable scoping
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import try_operation_safe, show_status_safe
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.dataset.downloader.handlers.check_handler import setup_check_handler
from smartcash.ui.dataset.downloader.handlers.cleanup_handler import setup_cleanup_handler
from smartcash.ui.dataset.downloader.handlers.validation_handler import setup_validation_handler
from smartcash.ui.dataset.downloader.utils.operation_utils import get_streamlined_download_operations
from smartcash.dataset.downloader import get_downloader_instance
from smartcash.common.logger import get_logger

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handlers dengan fixed state management"""
    
    logger = ui_components.get('logger') or get_logger('downloader.handlers')
    
    # Setup handlers dengan fixed wrapper
    check_handler = _wrap_with_state_management(setup_check_handler(ui_components, config, logger), ui_components, logger)
    cleanup_handler = _wrap_with_state_management(setup_cleanup_handler(ui_components, config, logger), ui_components, logger)
    validation_handler = setup_validation_handler(ui_components, config, logger)
    download_handler = _wrap_with_state_management(_create_download_handler(ui_components, config, logger), ui_components, logger)
    
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
            # Get all buttons safely
            all_buttons = getattr(button, '_all_buttons', [])
            if not all_buttons:
                all_buttons = [
                    ui_components.get('check_button'),
                    ui_components.get('download_button'), 
                    ui_components.get('cleanup_button')
                ]
                all_buttons = [btn for btn in all_buttons if btn is not None]
            
            # Disable all buttons
            for btn in all_buttons:
                if hasattr(btn, 'disabled'):
                    btn.disabled = True
            
            # Execute handler
            handler(button)
            
        except Exception as e:
            logger and logger.error(f"❌ Error in handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
        finally:
            # Re-enable buttons
            for btn in all_buttons:
                if hasattr(btn, 'disabled'):
                    btn.disabled = False
    
    return state_managed_handler

def _create_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Create fixed download handler"""
    
    def handle_download(button):
        """Handle download dengan fixed validation dan confirmation"""
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
    
    workspace = config.get('workspace', '')
    project = config.get('project', '')
    version = config.get('version', '')
    dataset_id = f"{workspace}/{project}:v{version}"
    
    def on_confirm_handler(button):
        """Fixed confirm handler"""
        _clear_confirmation_area_fixed(ui_components)
        _execute_download_sync_fixed(ui_components, config, logger)
    
    def on_cancel_handler(button):
        """Fixed cancel handler"""
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
    """Execute download dengan fixed dual-level progress tracking"""
    try:
        # Clear confirmation area
        _clear_confirmation_area_fixed(ui_components)
        
        # Get progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            logger.error("❌ Progress tracker tidak ditemukan")
            show_status_safe("❌ Progress tracker tidak tersedia", "error", ui_components)
            return
        
        # Show progress dengan dual level
        progress_tracker.show("Download Dataset")
        
        # Create downloader instance
        downloader = get_downloader_instance(config, logger)
        
        # Setup fixed dual-level progress callback
        downloader.set_progress_callback(_create_fixed_progress_callback(progress_tracker, logger))
        
        # Execute download
        result = downloader.download_dataset(
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
        if progress_tracker:
            progress_tracker.error(error_msg)
        show_status_safe(error_msg, "error", ui_components)
        logger.error(error_msg)

def _create_fixed_progress_callback(progress_tracker, logger):
    """Create fixed dual-level progress callback"""
    
    def progress_callback(step: str, current: int, total: int, message: str):
        """Fixed dual-level progress dengan proper mapping"""
        try:
            percentage = min(100, max(0, int((current / total) * 100) if total > 0 else 0))
            
            # Map steps ke dual-level progress dengan fixed calculation
            step_mapping = {
                'validate': (0, 10, "🔄 Validasi parameter"),
                'connect': (10, 20, "🌐 Koneksi Roboflow"),
                'metadata': (20, 30, "📊 Ambil metadata"),
                'download': (30, 80, "📥 Download dataset"),
                'extract': (80, 90, "📦 Ekstrak files"),
                'organize': (90, 100, "🗂️ Organisir struktur")
            }
            
            if step in step_mapping:
                start_pct, end_pct, overall_msg = step_mapping[step]
                overall_percentage = start_pct + int(percentage * (end_pct - start_pct) / 100)
                progress_tracker.update_overall(overall_percentage, overall_msg)
                progress_tracker.update_primary(percentage, message)
            else:
                progress_tracker.update_primary(percentage, message)
                
        except Exception as e:
            logger.debug(f"🔍 Progress callback error: {str(e)}")
    
    return progress_callback

def _bind_button_handlers_fixed(ui_components: Dict[str, Any], handlers: Dict[str, Callable]) -> None:
    """Bind button handlers dengan fixed approach"""
    button_mappings = [
        ('check_button', handlers.get('check_handler')),
        ('download_button', handlers.get('download_handler')),
        ('cleanup_button', handlers.get('cleanup_handler'))
    ]
    
    for button_name, handler in button_mappings:
        if button_name in ui_components and handler is not None:
            button = ui_components[button_name]
            if hasattr(button, 'on_click'):
                button.on_click(handler)

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

# Fixed utilities
def get_download_status_fixed(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get download status dengan fixed approach"""
    return {
        'ready': 'progress_tracker' in ui_components,
        'handlers_count': len([k for k in ui_components.keys() if k.endswith('_handler')]),
        'buttons_available': all(btn in ui_components for btn in ['check_button', 'download_button', 'cleanup_button'])
    }

def validate_handlers_setup_fixed(ui_components: Dict[str, Any]) -> bool:
    """Validate handlers setup dengan fixed approach"""
    required_handlers = ['check_handler', 'download_handler', 'cleanup_handler']
    return all(handler in ui_components for handler in required_handlers)

def get_handler_summary_fixed(ui_components: Dict[str, Any]) -> str:
    """Get handler summary dengan fixed approach"""
    handler_count = len([k for k in ui_components.keys() if k.endswith('_handler')])
    progress_status = '✅' if 'progress_tracker' in ui_components else '❌'
    return f"✅ Handlers setup: {handler_count} handlers | Progress: {progress_status}"