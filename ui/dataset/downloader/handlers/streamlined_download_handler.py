"""
File: smartcash/ui/dataset/downloader/handlers/streamlined_download_handler.py
Deskripsi: Streamlined download handler dengan reduced redundancy dan improved backend integration
"""

from typing import Dict, Any, Callable, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.common.logger import get_logger
from smartcash.ui.utils.fallback_utils import try_operation_safe, show_status_safe
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.dataset.downloader.handlers.validation_handler import validate_download_parameters
from smartcash.ui.dataset.downloader.handlers.progress_integration import create_progress_integrator

def setup_streamlined_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """Setup streamlined download handlers dengan reduced redundancy."""
    logger = ui_components.get('logger', get_logger('downloader.handler'))
    
    # Validate critical components dengan one-liner
    required = ['download_button', 'check_button', 'workspace_field', 'project_field', 'version_field']
    missing = [comp for comp in required if comp not in ui_components]
    
    if missing:
        error_msg = f"Missing components: {', '.join(missing)}"
        logger.error(f"❌ {error_msg}")
        return {'status': 'error', 'message': error_msg, 'missing': missing}
    
    try:
        # Setup progress integrator
        progress_integrator = create_progress_integrator(ui_components)
        ui_components['progress_integrator'] = progress_integrator
        
        # Setup button handlers dengan one-liner style
        button_handlers = {
            'download_button': lambda b: _handle_download_flow(ui_components, b, progress_integrator),
            'check_button': lambda b: _handle_check_flow(ui_components, b),
            'cleanup_button': lambda b: _handle_cleanup_flow(ui_components, b)
        }
        
        # Bind handlers dengan existence check
        [ui_components[btn].on_click(handler) for btn, handler in button_handlers.items() 
         if btn in ui_components and hasattr(ui_components[btn], 'on_click')]
        
        logger.info("✅ Streamlined download handlers configured")
        return {'status': 'success', 'handlers_configured': len(button_handlers)}
        
    except Exception as e:
        logger.error(f"❌ Handler setup error: {str(e)}")
        return {'status': 'error', 'message': str(e)}

def _handle_download_flow(ui_components: Dict[str, Any], button, progress_integrator) -> None:
    """Handle download flow dengan streamlined validation dan confirmation."""
    logger = ui_components.get('logger')
    
    try:
        # Clear outputs dan prepare UI
        _clear_ui_outputs(ui_components)
        _set_button_state(button, True, "Validating...")
        
        # Quick validation
        logger and logger.info("🔍 Validating download parameters...")
        validation_result = validate_download_parameters(ui_components, include_api_test=False)
        
        if not validation_result['valid']:
            show_status_safe(f"❌ Validation failed: {validation_result.get('message', 'Unknown error')}", 'error', ui_components)
            _set_button_state(button, False, "Download Dataset")
            return
        
        # Show confirmation dengan dataset info
        _show_streamlined_confirmation(ui_components, validation_result['config'], progress_integrator, button)
        
    except Exception as e:
        logger and logger.error(f"❌ Download flow error: {str(e)}")
        show_status_safe(f"❌ Download error: {str(e)}", 'error', ui_components)
        _set_button_state(button, False, "Download Dataset")

def _show_streamlined_confirmation(ui_components: Dict[str, Any], config: Dict[str, Any], 
                                 progress_integrator, button) -> None:
    """Show streamlined confirmation dialog dengan essential info."""
    
    # Build concise confirmation message
    roboflow = config.get('roboflow', {})
    local = config.get('local', {})
    
    dataset_info = f"{roboflow.get('workspace', 'N/A')}/{roboflow.get('project', 'N/A')}:{roboflow.get('version', 'N/A')}"
    storage_info = "Google Drive" if _is_drive_connected(ui_components) else "Local Storage"
    
    message = f"""
📥 **Download Dataset Confirmation**

**Dataset:** {dataset_info}
**Format:** {roboflow.get('format', 'yolov5pytorch')}
**Storage:** {storage_info}
**Options:** {'✅ Organize' if local.get('organize_dataset') else '❌ Organize'} | {'✅ Backup' if local.get('backup_enabled') else '❌ Backup'}

Lanjutkan download?
    """
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        _execute_streamlined_download(ui_components, config, progress_integrator, button)
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        _set_button_state(button, False, "Download Dataset")
        logger = ui_components.get('logger')
        logger and logger.info("❌ Download cancelled by user")
    
    # Create dan display dialog
    dialog = create_confirmation_dialog(
        title="📥 Download Confirmation",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text="Ya, Download",
        cancel_text="Batal"
    )
    
    _display_in_confirmation_area(ui_components, dialog)

def _execute_streamlined_download(ui_components: Dict[str, Any], config: Dict[str, Any], 
                                progress_integrator, button) -> None:
    """Execute download dengan streamlined flow dan proper progress tracking."""
    logger = ui_components.get('logger')
    
    try:
        # Start progress tracking
        progress_integrator.start_download('download')
        _set_button_state(button, True, "Downloading...")
        
        # Import dan setup download service
        from smartcash.dataset.downloader.download_service import DownloadService
        
        download_service = DownloadService(config, logger)
        download_service.set_progress_callback(progress_integrator.get_callback())
        
        logger and logger.info("🚀 Starting dataset download...")
        
        # Execute download dengan config dari UI
        roboflow = config.get('roboflow', {})
        result = download_service.download_dataset(
            workspace=roboflow.get('workspace'),
            project=roboflow.get('project'),
            version=roboflow.get('version'),
            api_key=roboflow.get('api_key'),
            output_format=roboflow.get('format', 'yolov5pytorch'),
            validate_download=True,
            organize_dataset=config.get('local', {}).get('organize_dataset', True),
            backup_existing=config.get('local', {}).get('backup_enabled', False)
        )
        
        # Handle result
        if result['status'] == 'success':
            _handle_download_success(ui_components, result, progress_integrator, button)
        else:
            _handle_download_error(ui_components, result.get('message', 'Unknown error'), progress_integrator, button)
            
    except Exception as e:
        logger and logger.error(f"❌ Download execution error: {str(e)}")
        _handle_download_error(ui_components, str(e), progress_integrator, button)

def _handle_download_success(ui_components: Dict[str, Any], result: Dict[str, Any], 
                           progress_integrator, button) -> None:
    """Handle successful download dengan concise feedback."""
    logger = ui_components.get('logger')
    
    # Complete progress
    progress_integrator.complete_download("Download completed successfully!")
    
    # Update UI state
    _set_button_state(button, False, "Download Dataset")
    
    # Show success summary
    stats = result.get('stats', {})
    duration = result.get('duration', 0)
    total_images = stats.get('total_images', 0)
    
    success_msg = f"✅ Download successful: {total_images} images in {duration:.1f}s"
    show_status_safe(success_msg, 'success', ui_components)
    
    # Log detailed info
    if logger:
        logger.success(f"🎉 {success_msg}")
        logger.info(f"📁 Output: {result.get('output_dir', 'Unknown')}")
        splits_info = ', '.join(f"{k}:{v.get('images', 0)}" for k, v in stats.get('splits', {}).items())
        splits_info and logger.info(f"📊 Splits: {splits_info}")

def _handle_download_error(ui_components: Dict[str, Any], error_message: str, 
                         progress_integrator, button) -> None:
    """Handle download error dengan proper cleanup."""
    logger = ui_components.get('logger')
    
    # Update progress dan UI state
    progress_integrator.error_download(error_message)
    _set_button_state(button, False, "Download Dataset")
    
    # Show error message
    show_status_safe(f"❌ Download failed: {error_message}", 'error', ui_components)
    logger and logger.error(f"💥 Download failed: {error_message}")

def _handle_check_flow(ui_components: Dict[str, Any], button) -> None:
    """Handle check dataset flow dengan streamlined validation."""
    logger = ui_components.get('logger')
    
    try:
        _set_button_state(button, True, "Checking...")
        logger and logger.info("🔍 Checking dataset structure...")
        
        # Import dataset organizer for check functionality
        from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer
        
        organizer = DatasetOrganizer(logger)
        check_result = organizer.check_organized_dataset()
        
        # Display check results
        if check_result['is_organized']:
            total_images = check_result['total_images']
            splits = ', '.join(f"{k}:{v.get('images', 0)}" for k, v in check_result['splits'].items())
            success_msg = f"✅ Dataset valid: {total_images} images | Splits: {splits}"
            show_status_safe(success_msg, 'success', ui_components)
            logger and logger.success(success_msg)
        else:
            issues = check_result.get('issues', ['Dataset not found'])
            error_msg = f"❌ Dataset issues: {'; '.join(issues[:2])}"
            show_status_safe(error_msg, 'warning', ui_components)
            logger and logger.warning(error_msg)
            
    except Exception as e:
        logger and logger.error(f"❌ Check dataset error: {str(e)}")
        show_status_safe(f"❌ Check error: {str(e)}", 'error', ui_components)
    finally:
        _set_button_state(button, False, "Check Dataset")

def _handle_cleanup_flow(ui_components: Dict[str, Any], button) -> None:
    """Handle cleanup flow dengan confirmation."""
    logger = ui_components.get('logger')
    
    def confirm_cleanup():
        try:
            _set_button_state(button, True, "Cleaning...")
            logger and logger.info("🧹 Starting dataset cleanup...")
            
            # Import dataset organizer for cleanup
            from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer
            
            organizer = DatasetOrganizer(logger)
            cleanup_result = organizer.cleanup_all_dataset_folders()
            
            if cleanup_result['status'] == 'success':
                show_status_safe(f"✅ {cleanup_result['message']}", 'success', ui_components)
                logger and logger.success(f"🧹 {cleanup_result['message']}")
            else:
                show_status_safe(f"⚠️ {cleanup_result['message']}", 'warning', ui_components)
                logger and logger.warning(f"⚠️ {cleanup_result['message']}")
                
        except Exception as e:
            logger and logger.error(f"❌ Cleanup error: {str(e)}")
            show_status_safe(f"❌ Cleanup error: {str(e)}", 'error', ui_components)
        finally:
            _set_button_state(button, False, "Hapus Hasil")
    
    # Show confirmation
    cleanup_dialog = create_confirmation_dialog(
        title="🗑️ Konfirmasi Hapus Dataset",
        message="Yakin ingin menghapus semua file dataset yang sudah didownload?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!",
        on_confirm=lambda b: (ui_components['confirmation_area'].clear_output(), confirm_cleanup()),
        on_cancel=lambda b: ui_components['confirmation_area'].clear_output(),
        confirm_text="Ya, Hapus",
        cancel_text="Batal",
        danger_mode=True
    )
    
    _display_in_confirmation_area(ui_components, cleanup_dialog)

# Helper functions dengan one-liner style
def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs dengan one-liner."""
    [widget.clear_output(wait=True) for key in ['log_output', 'confirmation_area', 'summary_container'] 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]

def _set_button_state(button, disabled: bool, description: str = None) -> None:
    """Set button state dengan one-liner."""
    try_operation_safe(lambda: setattr(button, 'disabled', disabled))
    description and try_operation_safe(lambda: setattr(button, 'description', description))

def _is_drive_connected(ui_components: Dict[str, Any]) -> bool:
    """Check drive connection dengan one-liner."""
    return try_operation_safe(
        lambda: ui_components.get('env_manager', type('', (), {'is_drive_mounted': False})()).is_drive_mounted,
        fallback_value=False
    )

def _display_in_confirmation_area(ui_components: Dict[str, Any], widget) -> None:
    """Display widget in confirmation area dengan one-liner."""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        confirmation_area.clear_output()
        with confirmation_area:
            display(widget)
        # Make confirmation area visible
        try_operation_safe(lambda: setattr(confirmation_area.layout, 'display', 'block'))