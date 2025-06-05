"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py  
Deskripsi: Main handler untuk download operations dengan progress tracking dan confirmation management
"""

from typing import Dict, Any, Callable
import threading
from smartcash.ui.utils.fallback_utils import try_operation_safe, show_status_safe
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.dataset.downloader.handlers.check_handler import setup_check_handler
from smartcash.ui.dataset.downloader.handlers.cleanup_handler import setup_cleanup_handler
from smartcash.ui.dataset.downloader.handlers.validation_handler import setup_validation_handler
from smartcash.ui.dataset.downloader.utils.operation_utils import consolidate_download_operations
from smartcash.dataset.downloader import get_downloader_instance
from smartcash.common.logger import get_logger

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handlers untuk download operations dengan progress tracking"""
    
    logger = ui_components.get('logger') or get_logger('downloader.handlers')
    
    # Setup individual handlers
    check_handler = setup_check_handler(ui_components, config, logger)
    cleanup_handler = setup_cleanup_handler(ui_components, config, logger)
    validation_handler = setup_validation_handler(ui_components, config, logger)
    
    # Setup main download handler
    download_handler = _create_download_handler(ui_components, config, logger)
    
    # Bind button handlers
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

def _create_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Create main download handler dengan progress tracking"""
    
    def handle_download(button):
        """Handle download operation dengan confirmation dan progress"""
        button.disabled = True
        
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
                show_status_safe(f"❌ Config tidak valid: {'; '.join(validation['errors'])}", "error", ui_components)
                return
            
            # Check existing dataset untuk confirmation
            if _has_existing_dataset():
                _show_download_confirmation(ui_components, current_config, logger)
            else:
                # Langsung download jika tidak ada data
                _execute_download(ui_components, current_config, logger)
                
        except Exception as e:
            logger.error(f"❌ Error download handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
        finally:
            button.disabled = False
    
    return handle_download

def _show_download_confirmation(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Show confirmation dialog untuk download jika ada data existing"""
    
    workspace = config.get('workspace', '')
    project = config.get('project', '') 
    version = config.get('version', '')
    dataset_id = f"{workspace}/{project}:v{version}"
    
    confirmation_dialog = create_confirmation_dialog(
        title="Konfirmasi Download Dataset",
        message=f"""Dataset akan didownload: {dataset_id}
        
⚠️ Dataset existing akan ditimpa jika ada.
        
Backup otomatis: {'Ya' if config.get('backup_existing') else 'Tidak'}
Organisir struktur: {'Ya' if config.get('organize_dataset') else 'Tidak'}
Validasi hasil: {'Ya' if config.get('validate_download') else 'Tidak'}

Lanjutkan download?""",
        on_confirm=lambda b: _execute_download(ui_components, config, logger),
        on_cancel=lambda b: _clear_confirmation_area(ui_components),
        confirm_text="Ya, Download",
        cancel_text="Batal"
    )
    
    _show_in_confirmation_area(ui_components, confirmation_dialog)

def _execute_download(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Execute download operation dengan progress tracking"""
    
    def download_thread():
        """Download thread dengan progress tracking"""
        try:
            # Clear confirmation area
            _clear_confirmation_area(ui_components)
            
            # Show progress container
            progress_tracker = ui_components.get('tracker')
            if progress_tracker:
                progress_tracker.show('download')
                progress_tracker.update('overall', 0, "🚀 Memulai download dataset...")
            
            # Create downloader instance
            downloader = get_downloader_instance(config, logger)
            
            # Setup progress callback
            if progress_tracker:
                downloader.set_progress_callback(_create_progress_callback(progress_tracker, logger))
            
            # Execute download
            result = downloader.download_dataset(
                workspace=config['workspace'],
                project=config['project'],
                version=config['version'],
                api_key=config['api_key'],
                output_format=config.get('output_format', 'yolov5pytorch'),
                validate_download=config.get('validate_download', True),
                organize_dataset=config.get('organize_dataset', True),
                backup_existing=config.get('backup_existing', False)
            )
            
            # Handle result
            if result['status'] == 'success':
                if progress_tracker:
                    progress_tracker.complete(f"✅ Download selesai: {result['stats']['total_images']} gambar")
                
                success_msg = f"✅ Dataset berhasil didownload ke {result['output_dir']}"
                show_status_safe(success_msg, "success", ui_components)
                logger.success(success_msg)
                
            else:
                error_msg = f"❌ Download gagal: {result['message']}"
                if progress_tracker:
                    progress_tracker.error(error_msg)
                show_status_safe(error_msg, "error", ui_components)
                logger.error(error_msg)
                
        except Exception as e:
            error_msg = f"❌ Error saat download: {str(e)}"
            if progress_tracker:
                progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            logger.error(error_msg)
    
    # Start download thread
    thread = threading.Thread(target=download_thread, daemon=True)
    thread.start()

def _create_progress_callback(progress_tracker, logger) -> Callable:
    """Create progress callback untuk downloader dengan consolidated operations"""
    
    def progress_callback(step: str, current: int, total: int, message: str):
        """Progress callback dengan step mapping"""
        try:
            # Map download steps ke progress tracker
            step_mapping = {
                'validate': ('overall', 'Validasi parameter'),
                'connect': ('step', 'Koneksi ke Roboflow'),
                'download': ('current', 'Download dataset'),
                'extract': ('step', 'Ekstraksi file'),
                'organize': ('overall', 'Organisir dataset')
            }
            
            if step in step_mapping:
                progress_type, default_msg = step_mapping[step]
                display_msg = message or default_msg
                progress_tracker.update(progress_type, current, display_msg)
            
        except Exception as e:
            logger.debug(f"🔍 Progress callback error: {str(e)}")
    
    return progress_callback

def _bind_button_handlers(ui_components: Dict[str, Any], handlers: Dict[str, Callable]) -> None:
    """Bind button handlers dengan one-liner"""
    button_mappings = {
        'check_button': handlers['check_handler'],
        'download_button': handlers['download_handler'],
        'cleanup_button': handlers['cleanup_handler']
    }
    
    [ui_components[button_name].on_click(handler) 
     for button_name, handler in button_mappings.items() 
     if button_name in ui_components and hasattr(ui_components[button_name], 'on_click')]

def _has_existing_dataset() -> bool:
    """Check apakah ada existing dataset dengan consolidated check"""
    return consolidate_download_operations().check_existing_dataset()

def _show_in_confirmation_area(ui_components: Dict[str, Any], dialog_widget) -> None:
    """Show dialog dalam confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        with confirmation_area:
            confirmation_area.clear_output(wait=True)
            from IPython.display import display
            display(dialog_widget)

def _clear_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Clear confirmation area dengan one-liner"""
    confirmation_area = ui_components.get('confirmation_area')
    confirmation_area and confirmation_area.clear_output(wait=True)