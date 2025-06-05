"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py  
Deskripsi: Fixed download handler dengan button state management dan proper dialog confirmation
"""

from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import try_operation_safe, show_status_safe
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.dataset.downloader.handlers.check_handler import setup_check_handler
from smartcash.ui.dataset.downloader.handlers.cleanup_handler import setup_cleanup_handler
from smartcash.ui.dataset.downloader.handlers.validation_handler import setup_validation_handler
from smartcash.ui.dataset.downloader.utils.operation_utils import consolidate_download_operations
from smartcash.dataset.downloader import get_downloader_instance
from smartcash.common.logger import get_logger

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handlers untuk download operations dengan proper state management"""
    
    logger = ui_components.get('logger') or get_logger('downloader.handlers')
    
    # Setup individual handlers dengan state management wrapper
    check_handler = _wrap_with_state_management(setup_check_handler(ui_components, config, logger), ui_components)
    cleanup_handler = _wrap_with_state_management(setup_cleanup_handler(ui_components, config, logger), ui_components)
    validation_handler = setup_validation_handler(ui_components, config, logger)
    
    # Setup main download handler dengan state management
    download_handler = _wrap_with_state_management(_create_download_handler(ui_components, config, logger), ui_components)
    
    # Bind button handlers dengan one-liner
    _bind_button_handlers(ui_components, {
        'check_handler': check_handler,
        'download_handler': download_handler,
        'cleanup_handler': cleanup_handler
    })
    
    ui_components.update({
        'check_handler': check_handler,
        'download_handler': download_handler,
        'cleanup_handler': cleanup_handler,
        'validation_handler': validation_handler
    })
    
    return ui_components

def _wrap_with_state_management(handler: Callable, ui_components: Dict[str, Any]) -> Callable:
    """Wrap handler dengan button state management untuk mutual exclusion"""
    
    def state_managed_handler(button):
        """Wrapper yang manage button states saat operation berjalan"""
        try:
            # Disable all buttons dengan one-liner state management
            all_buttons = getattr(button, '_all_buttons', [])
            [setattr(btn, 'disabled', True) for btn in all_buttons]
            
            # Execute original handler
            handler(button)
            
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"❌ Error in state managed handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
        finally:
            # Re-enable all buttons dengan one-liner restore
            all_buttons = getattr(button, '_all_buttons', [])
            [setattr(btn, 'disabled', False) for btn in all_buttons]
    
    return state_managed_handler

def _create_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Create main download handler dengan proper validation dan confirmation"""
    
    def handle_download(button):
        """Handle download operation dengan proper validation dan confirmation flow"""
        try:
            # Get current config dari config handler
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                show_status_safe("❌ Config handler tidak ditemukan", "error", ui_components)
                return
            
            # Extract config dengan proper widget access
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
                # Langsung download jika tidak ada data
                _execute_download_sync(ui_components, current_config, logger)
                
        except Exception as e:
            logger.error(f"❌ Error download handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
    
    return handle_download

def _show_download_confirmation(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Show proper confirmation dialog untuk download jika ada data existing"""
    
    workspace = config.get('workspace', '')
    project = config.get('project', '') 
    version = config.get('version', '')
    dataset_id = f"{workspace}/{project}:v{version}"
    backup_dir = config.get('backup_dir', 'data/backup')
    
    confirmation_dialog = create_confirmation_dialog(
        title="Konfirmasi Download Dataset",
        message=f"""📥 **Dataset Download Confirmation**

🎯 **Target Dataset:** {dataset_id}
📂 **Backup Directory:** {backup_dir}

⚠️ **Perhatian:**
• Dataset existing akan ditimpa jika ada
• Backup otomatis: {'✅ Ya' if config.get('backup_existing') else '❌ Tidak'}
• Validasi hasil: {'✅ Ya' if config.get('validate_download') else '❌ Tidak'}

🚀 Lanjutkan download?""",
        on_confirm=lambda b: (_clear_confirmation_area(ui_components), _execute_download_sync(ui_components, config, logger)),
        on_cancel=lambda b: _clear_confirmation_area(ui_components),
        confirm_text="Ya, Download",
        cancel_text="Batal"
    )
    
    _show_in_confirmation_area(ui_components, confirmation_dialog)

def _execute_download_sync(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Execute download operation dengan proper three-level progress tracking"""
    try:
        # Clear confirmation area
        _clear_confirmation_area(ui_components)
        
        # Show progress container dengan three-level tracking
        progress_tracker = ui_components.get('tracker')
        if progress_tracker:
            download_steps = ['validate', 'connect', 'metadata', 'download', 'extract', 'organize']
            step_weights = {'validate': 5, 'connect': 10, 'metadata': 10, 'download': 50, 'extract': 15, 'organize': 10}
            progress_tracker.show('download', download_steps, step_weights)
            progress_tracker.update_overall(0, "🚀 Memulai download dataset...")
        
        # Create downloader instance
        downloader = get_downloader_instance(config, logger)
        
        # Setup three-level progress callback
        progress_tracker and downloader.set_progress_callback(_create_three_level_progress_callback(progress_tracker, logger))
        
        # Execute download
        result = downloader.download_dataset(
            workspace=config['workspace'],
            project=config['project'],
            version=config['version'],
            api_key=config['api_key'],
            output_format=config.get('output_format', 'yolov5pytorch'),
            validate_download=config.get('validate_download', True),
            organize_dataset=True,  # Always true (no checkbox)
            backup_existing=config.get('backup_existing', False)
        )
        
        # Handle result dengan proper status display
        if result['status'] == 'success':
            success_msg = f"✅ Dataset berhasil didownload: {result['stats']['total_images']} gambar"
            progress_tracker and progress_tracker.complete(success_msg)
            show_status_safe(f"{success_msg} ke {result['output_dir']}", "success", ui_components)
            logger.success(success_msg)
        else:
            error_msg = f"❌ Download gagal: {result['message']}"
            progress_tracker and progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            logger.error(error_msg)
            
    except Exception as e:
        error_msg = f"❌ Error saat download: {str(e)}"
        progress_tracker = ui_components.get('tracker')
        progress_tracker and progress_tracker.error(error_msg)
        show_status_safe(error_msg, "error", ui_components)
        logger.error(error_msg)

def _create_three_level_progress_callback(progress_tracker, logger) -> Callable:
    """Create three-level progress callback untuk downloader dengan proper step mapping"""
    
    def progress_callback(step: str, current: int, total: int, message: str):
        """Three-level progress callback dengan step mapping ke overall/step/current"""
        try:
            # Map download steps ke three-level progress dengan one-liner conditionals
            if step in ['validate', 'connect', 'metadata']:
                progress_tracker.update_step(current, message)
                progress_tracker.update_overall(current, f"🔄 {message}")
            elif step == 'download':
                progress_tracker.update_current(current, message)
                progress_tracker.update_step(current, "📥 Downloading dataset")
                progress_tracker.update_overall(current, f"📥 {message}")
            elif step in ['extract', 'organize']:
                progress_tracker.update_step(current, message)
                progress_tracker.update_overall(current, f"🔄 {message}")
            else:
                # Generic progress update
                progress_tracker.update_current(current, message)
                
        except Exception as e:
            logger.debug(f"🔍 Three-level progress callback error: {str(e)}")
    
    return progress_callback

def _bind_button_handlers(ui_components: Dict[str, Any], handlers: Dict[str, Callable]) -> None:
    """Bind button handlers dengan one-liner safety checks"""
    button_mappings = [
        ('check_button', handlers['check_handler']),
        ('download_button', handlers['download_handler']),
        ('cleanup_button', handlers['cleanup_handler'])
    ]
    
    [ui_components[button_name].on_click(handler) 
     for button_name, handler in button_mappings 
     if button_name in ui_components and hasattr(ui_components[button_name], 'on_click')]

def _has_existing_dataset() -> bool:
    """Check apakah ada existing dataset dengan consolidated check"""
    return consolidate_download_operations().check_existing_dataset()

def _show_in_confirmation_area(ui_components: Dict[str, Any], dialog_widget) -> None:
    """Show dialog dalam confirmation area dengan proper display management"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        # Show confirmation area dan display dialog
        setattr(confirmation_area.layout, 'display', 'block')
        with confirmation_area:
            confirmation_area.clear_output(wait=True)
            from IPython.display import display
            display(dialog_widget)

def _clear_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Clear confirmation area dengan proper hiding - one-liner cleanup"""
    confirmation_area = ui_components.get('confirmation_area')
    confirmation_area and (
        confirmation_area.clear_output(wait=True),
        setattr(confirmation_area.layout, 'display', 'none')
    )