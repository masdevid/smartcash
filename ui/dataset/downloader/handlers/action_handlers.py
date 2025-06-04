"""
File: smartcash/ui/dataset/downloader/handlers/action_handlers.py
Deskripsi: Complete action handlers dengan proper integration ke dataset services dan progress tracking
"""

from typing import Dict, Any, Callable, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.common.logger import get_logger
from smartcash.ui.utils.fallback_utils import try_operation_safe, show_status_safe
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.dataset.downloader.handlers.validation_handler import validate_download_parameters

def setup_download_action_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup all action handlers dengan proper logging dan progress integration."""
    logger = ui_components.get('logger', get_logger('downloader.actions'))
    
    # Validate critical components
    required = ['download_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
    missing = [comp for comp in required if comp not in ui_components]
    
    if missing:
        logger.error(f"❌ Missing UI components: {', '.join(missing)}")
        return {'status': 'error', 'missing': missing}
    
    try:
        # Setup button handlers dengan one-liner style
        button_handlers = {
            'download_button': lambda b: _handle_download_with_confirmation(ui_components, b),
            'check_button': lambda b: _handle_check_with_confirmation(ui_components, b),
            'cleanup_button': lambda b: _handle_cleanup_with_confirmation(ui_components, b),
            'save_button': lambda b: _handle_save_config(ui_components, b),
            'reset_button': lambda b: _handle_reset_config(ui_components, b)
        }
        
        # Bind handlers dengan existence check
        [ui_components[btn].on_click(handler) for btn, handler in button_handlers.items() 
         if btn in ui_components and hasattr(ui_components[btn], 'on_click')]
        
        logger.info("✅ Action handlers configured with confirmation dialogs")
        return {'status': 'success', 'handlers_configured': len(button_handlers)}
        
    except Exception as e:
        logger.error(f"❌ Handler setup error: {str(e)}")
        return {'status': 'error', 'message': str(e)}

def _handle_download_with_confirmation(ui_components: Dict[str, Any], button) -> None:
    """Handle download dengan validation dan confirmation."""
    logger = ui_components.get('logger')
    
    # Reset UI state dan open log
    _reset_ui_state_and_open_log(ui_components)
    _set_button_state(button, True, "Validating...")
    
    try:
        # Quick validation
        logger and logger.info("🔍 Validating download parameters...")
        validation_result = validate_download_parameters(ui_components, include_api_test=False)
        
        if not validation_result['valid']:
            show_status_safe(f"❌ Validation failed: {validation_result.get('message', 'Unknown error')}", 'error', ui_components)
            _set_button_state(button, False, "📥 Download Dataset")
            return
        
        # Show confirmation dengan dataset info
        _show_download_confirmation(ui_components, validation_result, button)
        
    except Exception as e:
        logger and logger.error(f"❌ Download validation error: {str(e)}")
        show_status_safe(f"❌ Download error: {str(e)}", 'error', ui_components)
        _set_button_state(button, False, "📥 Download Dataset")

def _show_download_confirmation(ui_components: Dict[str, Any], validation_result: Dict[str, Any], button) -> None:
    """Show download confirmation dialog dengan dataset info."""
    config = validation_result.get('config', {})
    dataset_info = f"{config.get('workspace', 'N/A')}/{config.get('project', 'N/A')}:{config.get('version', 'N/A')}"
    storage_info = "Google Drive" if _is_drive_connected(ui_components) else "Local Storage"
    
    message = f"""
📥 **Download Dataset Confirmation**

**Dataset:** {dataset_info}
**Format:** YOLOv5 PyTorch (organize otomatis)
**Storage:** {storage_info}
**Backup:** {'✅ Enabled' if config.get('backup_existing') else '❌ Disabled'}

Dataset akan didownload dan diorganisir ke struktur:
• `/data/train/` - Gambar dan label training
• `/data/valid/` - Gambar dan label validation  
• `/data/test/` - Gambar dan label testing

Lanjutkan download?
    """
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        _execute_download(ui_components, config, button)
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        _set_button_state(button, False, "📥 Download Dataset")
    
    dialog = create_confirmation_dialog(
        title="📥 Download Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text="Ya, Download",
        cancel_text="Batal"
    )
    
    _display_in_confirmation_area(ui_components, dialog)

def _execute_download(ui_components: Dict[str, Any], config: Dict[str, Any], button) -> None:
    """Execute download menggunakan backend services dengan progress tracking."""
    logger = ui_components.get('logger')
    
    try:
        # Setup progress tracking
        _setup_progress_tracking(ui_components, 'download')
        _set_button_state(button, True, "Downloading...")
        
        logger and logger.info("🚀 Starting dataset download process...")
        
        # Import dan setup download service
        from smartcash.dataset.downloader.download_service import DownloadService
        download_service = DownloadService(config, logger)
        
        # Setup progress callback
        progress_callback = _create_progress_callback(ui_components)
        download_service.set_progress_callback(progress_callback)
        
        # Execute download
        result = download_service.download_dataset(
            workspace=config.get('workspace'),
            project=config.get('project'),
            version=config.get('version'),
            api_key=config.get('api_key'),
            output_format=config.get('format', 'yolov5pytorch'),
            validate_download=True,
            organize_dataset=True,  # Always TRUE
            backup_existing=config.get('backup_existing', False)
        )
        
        # Handle result
        if result['status'] == 'success':
            _handle_download_success(ui_components, result, button)
        else:
            _handle_download_error(ui_components, result.get('message', 'Unknown error'), button)
            
    except Exception as e:
        logger and logger.error(f"❌ Download execution error: {str(e)}")
        _handle_download_error(ui_components, str(e), button)

def _handle_check_with_confirmation(ui_components: Dict[str, Any], button) -> None:
    """Handle check dataset dengan confirmation."""
    _reset_ui_state_and_open_log(ui_components)
    
    message = """
🔍 **Check Dataset Confirmation**

Proses ini akan:
• Validasi struktur folder dataset
• Menghitung jumlah gambar dan label
• Check konsistensi data train/valid/test
• Tampilkan summary hasil validasi

Lanjutkan check dataset?
    """
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        _execute_check(ui_components, button)
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
    
    dialog = create_confirmation_dialog(
        title="🔍 Check Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text="Ya, Check",
        cancel_text="Batal"
    )
    
    _display_in_confirmation_area(ui_components, dialog)

def _execute_check(ui_components: Dict[str, Any], button) -> None:
    """Execute check dataset menggunakan dataset organizer."""
    logger = ui_components.get('logger')
    
    try:
        _set_button_state(button, True, "Checking...")
        logger and logger.info("🔍 Starting dataset validation...")
        
        # Import dataset organizer
        from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer
        organizer = DatasetOrganizer(logger)
        
        # Setup progress callback untuk organizer
        progress_callback = _create_progress_callback(ui_components)
        organizer.set_progress_callback(progress_callback)
        
        # Execute check
        check_result = organizer.check_organized_dataset()
        
        # Display results
        if check_result['is_organized']:
            total_images = check_result['total_images']
            splits_info = ', '.join(f"{k}:{v.get('images', 0)}" for k, v in check_result['splits'].items())
            success_msg = f"✅ Dataset valid: {total_images} images | Splits: {splits_info}"
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
        _set_button_state(button, False, "🔍 Check Dataset")

def _handle_cleanup_with_confirmation(ui_components: Dict[str, Any], button) -> None:
    """Handle cleanup dengan confirmation yang kuat."""
    _reset_ui_state_and_open_log(ui_components)
    
    message = """
🗑️ **PERINGATAN: Hapus Dataset**

⚠️ **TINDAKAN TIDAK DAPAT DIBATALKAN!**

Proses ini akan menghapus:
• Semua file di folder `/data/train/`
• Semua file di folder `/data/valid/`
• Semua file di folder `/data/test/`
• Semua file di folder `/data/downloads/`
• Backup yang ada

**PASTIKAN Anda sudah backup data penting!**

Yakin ingin melanjutkan?
    """
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        _execute_cleanup(ui_components, button)
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
    
    dialog = create_confirmation_dialog(
        title="🗑️ Hapus Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text="Ya, Hapus Semua",
        cancel_text="Batal",
        danger_mode=True
    )
    
    _display_in_confirmation_area(ui_components, dialog)

def _execute_cleanup(ui_components: Dict[str, Any], button) -> None:
    """Execute cleanup menggunakan dataset organizer."""
    logger = ui_components.get('logger')
    
    try:
        _set_button_state(button, True, "Cleaning...")
        logger and logger.info("🧹 Starting dataset cleanup...")
        
        # Setup progress tracking
        _setup_progress_tracking(ui_components, 'cleanup')
        
        # Import dataset organizer
        from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer
        organizer = DatasetOrganizer(logger)
        
        # Setup progress callback
        progress_callback = _create_progress_callback(ui_components)
        organizer.set_progress_callback(progress_callback)
        
        # Execute cleanup
        cleanup_result = organizer.cleanup_all_dataset_folders()
        
        # Handle result
        if cleanup_result['status'] == 'success':
            files_removed = cleanup_result['stats']['total_files_removed']
            success_msg = f"✅ Cleanup berhasil: {files_removed} file dihapus"
            show_status_safe(success_msg, 'success', ui_components)
            logger and logger.success(success_msg)
        else:
            show_status_safe(f"⚠️ {cleanup_result['message']}", 'warning', ui_components)
            logger and logger.warning(f"⚠️ {cleanup_result['message']}")
            
    except Exception as e:
        logger and logger.error(f"❌ Cleanup error: {str(e)}")
        show_status_safe(f"❌ Cleanup error: {str(e)}", 'error', ui_components)
    finally:
        _set_button_state(button, False, "🗑️ Hapus Hasil")

def _handle_save_config(ui_components: Dict[str, Any], button) -> None:
    """Handle save config dengan proper logging."""
    _reset_ui_state_and_open_log(ui_components)
    logger = ui_components.get('logger')
    _set_button_state(button, True, "Saving...")
    
    try:
        config_handler = ui_components.get('config_handler')
        if config_handler:
            success = config_handler.save_config(ui_components, 'downloader')
            if success:
                show_status_safe("✅ Konfigurasi berhasil disimpan", 'success', ui_components)
                logger and logger.success("💾 Configuration saved successfully")
            else:
                show_status_safe("❌ Gagal menyimpan konfigurasi", 'error', ui_components)
                logger and logger.error("💥 Failed to save configuration")
        else:
            show_status_safe("❌ Config handler tidak ditemukan", 'error', ui_components)
            logger and logger.error("💥 Config handler not found")
            
    except Exception as e:
        show_status_safe(f"❌ Error saving: {str(e)}", 'error', ui_components)
        logger and logger.error(f"💥 Save error: {str(e)}")
    finally:
        _set_button_state(button, False, "💾 Simpan")

def _handle_reset_config(ui_components: Dict[str, Any], button) -> None:
    """Handle reset config dengan proper logging."""
    _reset_ui_state_and_open_log(ui_components)
    logger = ui_components.get('logger')
    _set_button_state(button, True, "Resetting...")
    
    try:
        config_handler = ui_components.get('config_handler')
        if config_handler:
            success = config_handler.reset_config(ui_components, 'downloader')
            if success:
                show_status_safe("✅ Konfigurasi berhasil direset", 'success', ui_components)
                logger and logger.success("🔄 Configuration reset successfully")
            else:
                show_status_safe("❌ Gagal reset konfigurasi", 'error', ui_components)
                logger and logger.error("💥 Failed to reset configuration")
        else:
            show_status_safe("❌ Config handler tidak ditemukan", 'error', ui_components)
            logger and logger.error("💥 Config handler not found")
            
    except Exception as e:
        show_status_safe(f"❌ Error resetting: {str(e)}", 'error', ui_components)
        logger and logger.error(f"💥 Reset error: {str(e)}")
    finally:
        _set_button_state(button, False, "🔄 Reset")

# Helper functions dengan one-liner style
def _reset_ui_state_and_open_log(ui_components: Dict[str, Any]) -> None:
    """Reset UI state, clear logs, dan open log accordion."""
    # Clear log output
    log_output = ui_components.get('log_output')
    log_output and hasattr(log_output, 'clear_output') and log_output.clear_output(wait=True)
    
    # Open log accordion
    log_accordion = ui_components.get('log_accordion')
    log_accordion and hasattr(log_accordion, 'selected_index') and setattr(log_accordion, 'selected_index', 0)
    
    # Reset progress tracking
    try_operation_safe(lambda: ui_components.get('reset_all', lambda: None)())

def _set_button_state(button, disabled: bool, description: str = None) -> None:
    """Set button state dengan safe handling."""
    try_operation_safe(lambda: setattr(button, 'disabled', disabled))
    description and try_operation_safe(lambda: setattr(button, 'description', description))

def _setup_progress_tracking(ui_components: Dict[str, Any], operation: str) -> None:
    """Setup progress tracking untuk operation."""
    try_operation_safe(lambda: ui_components.get('show_for_operation', lambda x: None)(operation))

def _create_progress_callback(ui_components: Dict[str, Any]) -> Callable:
    """Create progress callback untuk backend services."""
    def progress_callback(step: str, current: int, total: int, message: str):
        """Unified progress callback."""
        try:
            # Route ke progress tracking methods
            if 'update_progress' in ui_components:
                ui_components['update_progress']('overall', current, message)
            elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'update'):
                ui_components['tracker'].update('overall', current, message)
        except Exception:
            pass  # Silent fail untuk prevent callback errors
    
    return progress_callback

def _handle_download_success(ui_components: Dict[str, Any], result: Dict[str, Any], button) -> None:
    """Handle successful download."""
    logger = ui_components.get('logger')
    
    # Complete progress
    try_operation_safe(lambda: ui_components.get('complete_operation', lambda x: None)("Download completed successfully!"))
    
    # Update UI state
    _set_button_state(button, False, "📥 Download Dataset")
    
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

def _handle_download_error(ui_components: Dict[str, Any], error_message: str, button) -> None:
    """Handle download error."""
    logger = ui_components.get('logger')
    
    # Update progress dan UI state
    try_operation_safe(lambda: ui_components.get('error_operation', lambda x: None)(f"❌ {error_message}"))
    _set_button_state(button, False, "📥 Download Dataset")
    
    # Show error message
    show_status_safe(f"❌ Download failed: {error_message}", 'error', ui_components)
    logger and logger.error(f"💥 Download failed: {error_message}")

def _is_drive_connected(ui_components: Dict[str, Any]) -> bool:
    """Check drive connection dengan one-liner."""
    return try_operation_safe(
        lambda: getattr(ui_components.get('env_manager', type('', (), {'is_drive_mounted': False})()), 'is_drive_mounted', False),
        fallback_value=False
    )

def _display_in_confirmation_area(ui_components: Dict[str, Any], widget) -> None:
    """Display widget in confirmation area."""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        confirmation_area.clear_output()
        try_operation_safe(lambda: setattr(confirmation_area.layout, 'display', 'block'))
        with confirmation_area:
            display(widget)