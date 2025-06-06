"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Fixed download handler dengan proper callable checks dan one-liner style
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
    """Setup semua handlers dengan optimized state management"""
    
    logger = ui_components.get('logger') or get_logger('downloader.handlers')
    
    # Setup handlers dengan state management wrapper
    check_handler = _wrap_with_state_management(setup_check_handler(ui_components, config, logger), ui_components)
    cleanup_handler = _wrap_with_state_management(setup_cleanup_handler(ui_components, config, logger), ui_components)
    validation_handler = setup_validation_handler(ui_components, config, logger)
    download_handler = _wrap_with_state_management(_create_download_handler(ui_components, config, logger), ui_components)
    
    # Bind handlers dengan one-liner
    _bind_button_handlers(ui_components, {
        'check_handler': check_handler, 'download_handler': download_handler, 'cleanup_handler': cleanup_handler
    })
    
    ui_components.update({
        'check_handler': check_handler, 'download_handler': download_handler,
        'cleanup_handler': cleanup_handler, 'validation_handler': validation_handler
    })
    
    return ui_components

def _wrap_with_state_management(handler: Callable, ui_components: Dict[str, Any]) -> Callable:
    """Wrap handler dengan optimized button state management"""
    
    def state_managed_handler(button):
        """Wrapper untuk mutual exclusion buttons"""
        try:
            # Disable all buttons dengan one-liner
            all_buttons = getattr(button, '_all_buttons', [])
            [setattr(btn, 'disabled', True) for btn in all_buttons]
            
            # Execute handler
            handler(button)
            
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"❌ Error in handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
        finally:
            # Re-enable buttons
            all_buttons = getattr(button, '_all_buttons', [])
            [setattr(btn, 'disabled', False) for btn in all_buttons]
    
    return state_managed_handler

def _create_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Create optimized download handler"""
    
    def handle_download(button):
        """Handle download dengan streamlined validation dan confirmation"""
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
            if _has_existing_dataset():
                _show_download_confirmation(ui_components, current_config, logger)
            else:
                _execute_download_sync(ui_components, current_config, logger)
                
        except Exception as e:
            logger.error(f"❌ Error download handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
    
    return handle_download

def _show_download_confirmation(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Show streamlined confirmation dialog"""
    
    workspace, project, version = config.get('workspace', ''), config.get('project', ''), config.get('version', '')
    dataset_id = f"{workspace}/{project}:v{version}"
    
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
        on_confirm=lambda b: (_clear_confirmation_area(ui_components), _execute_download_sync(ui_components, config, logger)),
        on_cancel=lambda b: _clear_confirmation_area(ui_components),
        confirm_text="Ya, Download", cancel_text="Batal"
    )
    
    _show_in_confirmation_area(ui_components, confirmation_dialog)

def _execute_download_sync(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Execute download dengan optimized dual-level progress tracking"""
    try:
        # Clear confirmation area
        _clear_confirmation_area(ui_components)
        
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
        
        # Setup optimized dual-level progress callback
        downloader.set_progress_callback(_create_optimized_progress_callback(progress_tracker, logger))
        
        # Execute download
        result = downloader.download_dataset(
            workspace=config['workspace'], project=config['project'], version=config['version'],
            api_key=config['api_key'], output_format=config.get('output_format', 'yolov5pytorch'),
            validate_download=config.get('validate_download', True), organize_dataset=True,
            backup_existing=config.get('backup_existing', False)
        )
        
        # Handle result dengan optimized status display
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

def _create_optimized_progress_callback(progress_tracker, logger) -> Callable:
    """Create optimized dual-level progress callback"""
    
    def progress_callback(step: str, current: int, total: int, message: str):
        """Optimized dual-level progress dengan proper mapping"""
        try:
            percentage = min(100, max(0, int((current / total) * 100) if total > 0 else 0))
            
            # Map steps ke dual-level progress dengan optimized calculation
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
                progress_tracker.update_current(percentage, message)
            else:
                # Generic progress update
                progress_tracker.update_current(percentage, message)
                
        except Exception as e:
            logger.debug(f"🔍 Progress callback error: {str(e)}")
    
    return progress_callback

def _bind_button_handlers(ui_components: Dict[str, Any], handlers: Dict[str, Callable]) -> None:
    """Bind button handlers dengan optimized one-liner"""
    button_mappings = [
        ('check_button', handlers['check_handler']),
        ('download_button', handlers['download_handler']),
        ('cleanup_button', handlers['cleanup_handler'])
    ]
    
    [ui_components[button_name].on_click(handler) 
     for button_name, handler in button_mappings 
     if button_name in ui_components and hasattr(ui_components[button_name], 'on_click')]

def _show_in_confirmation_area(ui_components: Dict[str, Any], dialog_widget) -> None:
    """Show dialog dalam confirmation area dengan optimized display"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        setattr(confirmation_area.layout, 'display', 'block')
        with confirmation_area:
            confirmation_area.clear_output(wait=True)
            from IPython.display import display
            display(dialog_widget)

def _clear_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Clear confirmation area dengan one-liner"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        confirmation_area.clear_output(wait=True)
        confirmation_area.layout.display = 'none'

# Fixed utilities dengan proper callable checks
def _has_existing_dataset() -> bool:
    """Check existing dataset dengan proper callable handling"""
    try:
        operations = get_streamlined_download_operations()
        return operations.check_existing_dataset()
    except Exception:
        return False

# Utilities dengan one-liner optimization
get_download_status = lambda ui: {'ready': 'progress_tracker' in ui, 'handlers_count': len([k for k in ui.keys() if k.endswith('_handler')])}
validate_handlers_setup = lambda ui: all(key in ui for key in ['check_handler', 'download_handler', 'cleanup_handler'])
get_handler_summary = lambda ui: f"✅ Handlers setup: {len([k for k in ui.keys() if k.endswith('_handler')])} handlers | Progress: {'✅' if 'progress_tracker' in ui else '❌'}"