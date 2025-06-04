"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Download handler dengan progress callback integration dan error handling
"""

from typing import Dict, Any, Callable
from smartcash.ui.dataset.downloader.handlers.validation_handler import validate_download_parameters
from smartcash.ui.dataset.downloader.handlers.progress_handler import ProgressCallbackManager
from smartcash.ui.dataset.downloader.components.action_buttons import update_button_states
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from IPython.display import display

def setup_download_handlers(ui_components: Dict[str, Any], env=None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Setup download handlers dengan progress callback integration."""
    logger = ui_components.get('logger')
    
    try:
        # Setup progress callback manager
        progress_manager = ProgressCallbackManager(ui_components)
        ui_components['progress_manager'] = progress_manager
        
        # Download button handler
        if 'download_button' in ui_components:
            ui_components['download_button'].on_click(
                lambda b: _handle_download_click(ui_components, b, progress_manager)
            )
        
        # Validate button handler
        if 'validate_button' in ui_components:
            ui_components['validate_button'].on_click(
                lambda b: _handle_validate_click(ui_components, b)
            )
        
        # Quick validate button handler
        if 'quick_validate_button' in ui_components:
            ui_components['quick_validate_button'].on_click(
                lambda b: _handle_quick_validate_click(ui_components, b)
            )
        
        logger and logger.debug("✅ Download handlers configured")
        return {'download_handlers': True}
        
    except Exception as e:
        logger and logger.error(f"❌ Download handlers setup error: {str(e)}")
        return {'download_handlers': False, 'error': str(e)}

def _handle_download_click(ui_components: Dict[str, Any], button, progress_manager: ProgressCallbackManager) -> None:
    """Handle download button click dengan comprehensive flow."""
    logger = ui_components.get('logger')
    
    try:
        # Clear previous outputs
        _clear_outputs(ui_components)
        
        # Update button states
        update_button_states(ui_components, 'downloading')
        
        # Step 1: Validate parameters
        logger and logger.info("🔍 Validating download parameters...")
        validation_result = validate_download_parameters(ui_components)
        
        if not validation_result['valid']:
            _handle_validation_error(ui_components, validation_result, button)
            return
        
        # Step 2: Show confirmation dialog
        _show_download_confirmation(ui_components, validation_result['config'], progress_manager, button)
        
    except Exception as e:
        logger and logger.error(f"❌ Download click error: {str(e)}")
        _handle_download_error(ui_components, str(e), button)

def _handle_validate_click(ui_components: Dict[str, Any], button) -> None:
    """Handle validate button click untuk comprehensive validation."""
    logger = ui_components.get('logger')
    
    try:
        update_button_states(ui_components, 'validating')
        
        # Full validation termasuk API connectivity
        logger and logger.info("🔍 Running full validation...")
        validation_result = validate_download_parameters(ui_components, include_api_test=True)
        
        # Display validation results
        _display_validation_results(ui_components, validation_result)
        
    except Exception as e:
        logger and logger.error(f"❌ Validation error: {str(e)}")
        _show_status_message(ui_components, f"❌ Validation error: {str(e)}", "error")
    finally:
        update_button_states(ui_components, 'ready')

def _handle_quick_validate_click(ui_components: Dict[str, Any], button) -> None:
    """Handle quick validate untuk basic parameter check."""
    logger = ui_components.get('logger')
    
    try:
        # Quick validation tanpa API test
        validation_result = validate_download_parameters(ui_components, include_api_test=False)
        
        if validation_result['valid']:
            _show_status_message(ui_components, "✅ Quick validation passed", "success")
            logger and logger.info("✅ Quick validation passed")
        else:
            error_msg = "; ".join(validation_result['errors'][:3])
            _show_status_message(ui_components, f"❌ Quick validation failed: {error_msg}", "error")
            logger and logger.warning(f"⚠️ Quick validation failed: {error_msg}")
        
    except Exception as e:
        logger and logger.error(f"❌ Quick validation error: {str(e)}")
        _show_status_message(ui_components, f"❌ Quick validation error: {str(e)}", "error")

def _show_download_confirmation(ui_components: Dict[str, Any], config: Dict[str, Any], 
                               progress_manager: ProgressCallbackManager, button) -> None:
    """Show download confirmation dialog dengan dataset info."""
    
    # Build confirmation message
    storage_type = "Google Drive" if _is_drive_storage(ui_components) else "Local Storage"
    message = f"""
📥 **Konfirmasi Download Dataset**

**Dataset Information:**
• Workspace: {config.get('workspace', 'N/A')}
• Project: {config.get('project', 'N/A')}
• Version: {config.get('version', 'N/A')}
• Format: {config.get('output_format', 'yolov5pytorch')}

**Options:**
• Validation: {'✅' if config.get('validate_download') else '❌'}
• Organization: {'✅' if config.get('organize_dataset') else '❌'}
• Backup: {'✅' if config.get('backup_existing') else '❌'}

**Storage:** {storage_type}

Lanjutkan download?
    """
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        _execute_download(ui_components, config, progress_manager, button)
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        update_button_states(ui_components, 'ready')
        logger = ui_components.get('logger')
        logger and logger.info("❌ Download dibatalkan oleh user")
    
    # Create dan show dialog
    dialog = create_confirmation_dialog(
        title="📥 Konfirmasi Download Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text="Ya, Download",
        cancel_text="Batal"
    )
    
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)

def _execute_download(ui_components: Dict[str, Any], config: Dict[str, Any], 
                     progress_manager: ProgressCallbackManager, button) -> None:
    """Execute actual download dengan progress tracking."""
    logger = ui_components.get('logger')
    
    try:
        # Initialize progress tracking
        progress_manager.start_download_process()
        
        # Import download service
        from smartcash.dataset.downloader.download_service import DownloadService
        
        # Setup progress callback
        download_service = DownloadService(config, logger)
        download_service.set_progress_callback(progress_manager.get_progress_callback())
        
        logger and logger.info("🚀 Memulai download dataset...")
        
        # Execute download
        result = download_service.download_dataset(
            workspace=config['workspace'],
            project=config['project'],
            version=config['version'],
            api_key=config['api_key'],
            output_format=config['output_format'],
            validate_download=config['validate_download'],
            organize_dataset=config['organize_dataset'],
            backup_existing=config['backup_existing']
        )
        
        # Handle result
        if result['status'] == 'success':
            _handle_download_success(ui_components, result, progress_manager)
        else:
            _handle_download_error(ui_components, result.get('message', 'Unknown error'), button)
            
    except Exception as e:
        logger and logger.error(f"❌ Download execution error: {str(e)}")
        _handle_download_error(ui_components, str(e), button)
        progress_manager.error_download_process(str(e))

def _handle_download_success(ui_components: Dict[str, Any], result: Dict[str, Any], 
                           progress_manager: ProgressCallbackManager) -> None:
    """Handle successful download completion."""
    logger = ui_components.get('logger')
    
    # Complete progress
    progress_manager.complete_download_process("Download berhasil!")
    
    # Update button states
    update_button_states(ui_components, 'ready')
    
    # Show success message
    stats = result.get('stats', {})
    duration = result.get('duration', 0)
    total_images = stats.get('total_images', 0)
    
    success_msg = f"✅ Download berhasil: {total_images} gambar dalam {duration:.1f}s"
    _show_status_message(ui_components, success_msg, "success")
    
    if logger:
        logger.success(f"🎉 {success_msg}")
        logger.info(f"📁 Output: {result.get('output_dir', 'Unknown')}")
        logger.info(f"💾 Storage: {'Google Drive' if result.get('drive_storage') else 'Local'}")

def _handle_download_error(ui_components: Dict[str, Any], error_message: str, button) -> None:
    """Handle download error dengan proper cleanup."""
    logger = ui_components.get('logger')
    
    # Update button states
    update_button_states(ui_components, 'error')
    
    # Show error message
    _show_status_message(ui_components, f"❌ Download error: {error_message}", "error")
    
    # Log error
    logger and logger.error(f"💥 Download failed: {error_message}")

def _handle_validation_error(ui_components: Dict[str, Any], validation_result: Dict[str, Any], button) -> None:
    """Handle validation error dengan detailed feedback."""
    logger = ui_components.get('logger')
    
    # Update button states
    update_button_states(ui_components, 'ready')
    
    # Show validation errors
    errors = validation_result.get('errors', [])
    warnings = validation_result.get('warnings', [])
    
    if errors:
        error_msg = f"❌ Validation errors: {'; '.join(errors[:3])}"
        _show_status_message(ui_components, error_msg, "error")
        logger and logger.error(error_msg)
    
    if warnings:
        warning_msg = f"⚠️ Validation warnings: {'; '.join(warnings[:2])}"
        logger and logger.warning(warning_msg)

def _display_validation_results(ui_components: Dict[str, Any], validation_result: Dict[str, Any]) -> None:
    """Display comprehensive validation results."""
    logger = ui_components.get('logger')
    
    if validation_result['valid']:
        _show_status_message(ui_components, "✅ All validations passed", "success")
        
        # Log details
        if logger:
            config = validation_result.get('config', {})
            logger.success("✅ Validation berhasil!")
            logger.info(f"📋 Parameters validated:")
            logger.info(f"   • Dataset: {config.get('workspace')}/{config.get('project')}:{config.get('version')}")
            logger.info(f"   • Format: {config.get('output_format')}")
            logger.info(f"   • API: {'✅ Connected' if validation_result.get('api_connected') else '⚠️ Not tested'}")
    else:
        _handle_validation_error(ui_components, validation_result, None)

def _show_status_message(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Show status message pada status panel."""
    if 'status_panel' in ui_components:
        colors = {
            'success': '#d4edda',
            'error': '#f8d7da',
            'warning': '#fff3cd',
            'info': '#d1ecf1'
        }
        
        text_colors = {
            'success': '#155724',
            'error': '#721c24',
            'warning': '#856404',
            'info': '#0c5460'
        }
        
        bg_color = colors.get(status_type, colors['info'])
        text_color = text_colors.get(status_type, text_colors['info'])
        
        status_html = f"""
        <div style='padding:8px; background-color:{bg_color}; color:{text_color}; 
                   border-radius:4px; border-left:4px solid {text_color};'>
            {message}
        </div>
        """
        ui_components['status_panel'].value = status_html

def _clear_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear various output widgets."""
    output_widgets = ['log_output', 'confirmation_area', 'status_panel']
    
    for widget_key in output_widgets:
        if widget_key in ui_components:
            widget = ui_components[widget_key]
            if hasattr(widget, 'clear_output'):
                widget.clear_output(wait=True)
            elif hasattr(widget, 'value'):
                widget.value = ""

def _is_drive_storage(ui_components: Dict[str, Any]) -> bool:
    """Check apakah menggunakan Google Drive storage."""
    try:
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        return env_manager.is_colab and env_manager.is_drive_mounted
    except Exception:
        return False

# Quick action handlers
def setup_quick_action_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup quick action handlers untuk status bar."""
    try:
        # Refresh status handler
        if 'refresh_status_button' in ui_components:
            ui_components['refresh_status_button'].on_click(
                lambda b: _refresh_status_indicators(ui_components)
            )
        
        # Clear logs handler
        if 'clear_logs_button' in ui_components:
            ui_components['clear_logs_button'].on_click(
                lambda b: _clear_log_output(ui_components)
            )
        
        return {'quick_actions': True}
        
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Quick actions setup error: {str(e)}")
        return {'quick_actions': False}

def _refresh_status_indicators(ui_components: Dict[str, Any]) -> None:
    """Refresh status indicators dengan current state."""
    logger = ui_components.get('logger')
    
    try:
        # Update API status
        if 'api_status' in ui_components:
            api_key = ui_components.get('api_key_field', {}).get('value', '')
            if api_key:
                ui_components['api_status'].value = "<span style='color:#28a745;'>🔑 API: Key detected</span>"
            else:
                ui_components['api_status'].value = "<span style='color:#ffc107;'>🔑 API: No key</span>"
        
        # Update dataset status (placeholder - could check existing dataset)
        if 'dataset_status' in ui_components:
            ui_components['dataset_status'].value = "<span style='color:#17a2b8;'>📊 Dataset: Ready to download</span>"
        
        # Update connection status
        if 'connection_status' in ui_components:
            if _is_drive_storage(ui_components):
                ui_components['connection_status'].value = "<span style='color:#28a745;'>🌐 Connection: Drive connected</span>"
            else:
                ui_components['connection_status'].value = "<span style='color:#ffc107;'>🌐 Connection: Local storage</span>"
        
        logger and logger.debug("🔄 Status indicators refreshed")
        
    except Exception as e:
        logger and logger.error(f"❌ Status refresh error: {str(e)}")

def _clear_log_output(ui_components: Dict[str, Any]) -> None:
    """Clear log output widget."""
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        ui_components['log_output'].clear_output(wait=True)
        logger = ui_components.get('logger')
        logger and logger.info("🗑️ Log output cleared")