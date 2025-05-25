"""
File: smartcash/ui/dataset/preprocessing/handlers/cleanup_executor.py
Deskripsi: Fixed cleanup executor dengan proper button state management
"""

from typing import Dict, Any
from smartcash.dataset.preprocessor.utils.preprocessing_factory import PreprocessingFactory
from smartcash.ui.dataset.preprocessing.utils.config_extractor import get_config_extractor
from smartcash.ui.dataset.preprocessing.utils.dialog_manager import get_dialog_manager
from smartcash.ui.utils.button_state_manager import get_button_state_manager

def setup_cleanup_executor(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup cleanup executor dengan fixed button state management."""
    
    def execute_cleanup_action(button=None) -> None:
        """Execute cleanup dengan user confirmation."""
        # Clear outputs first
        _clear_ui_outputs(ui_components)
        
        dialog_manager = get_dialog_manager(ui_components)
        logger = ui_components.get('logger')
        
        # Show confirmation dialog
        dialog_manager.show_destructive_confirmation(
            title="⚠️ Konfirmasi Cleanup Dataset",
            message="""Operasi ini akan menghapus SEMUA data preprocessed yang ada.

Data asli (source dataset) akan tetap aman, tetapi hasil preprocessing akan hilang dan perlu dijalankan ulang.

Apakah Anda yakin ingin melanjutkan?""",
            on_confirm=lambda: _perform_cleanup_operation(ui_components, logger),
            item_name="data preprocessed"
        )
    
    # Register handler
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].on_click(execute_cleanup_action)
    
    ui_components['execute_cleanup'] = execute_cleanup_action
    return ui_components

def _perform_cleanup_operation(ui_components: Dict[str, Any], logger) -> None:
    """Perform actual cleanup operation setelah konfirmasi."""
    button_manager = get_button_state_manager(ui_components)
    
    with button_manager.operation_context('cleanup'):
        try:
            logger and logger.info("🧹 Memulai cleanup dataset preprocessed")
            
            # Setup progress
            ui_components.get('show_for_operation', lambda x: None)('cleanup')
            
            # Get config
            config_extractor = get_config_extractor(ui_components)
            config = config_extractor.get_full_config()
            
            # Create progress callback
            def progress_callback(**kwargs):
                progress = kwargs.get('progress', 0)
                message = kwargs.get('message', 'Cleaning up...')
                ui_components.get('update_progress', lambda *args: None)('overall', progress, message)
            
            # Create cleanup service
            cleanup_service = PreprocessingFactory.create_cleanup_executor(
                config, logger, progress_callback
            )
            
            # Execute cleanup
            result = cleanup_service.cleanup_preprocessed_data(safe_mode=True)
            
            # Handle results
            if result['success']:
                _handle_cleanup_success(ui_components, result, logger)
            else:
                raise Exception(result['message'])
                
        except Exception as e:
            logger and logger.error(f"💥 Error cleanup: {str(e)}")
            ui_components.get('error_operation', lambda x: None)(f"Cleanup gagal: {str(e)}")
            _update_status_panel_error(ui_components, f"Cleanup gagal: {str(e)}")
            raise

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs untuk fresh display."""
    for output_key in ['log_output', 'status']:
        if output_key in ui_components and hasattr(ui_components[output_key], 'clear_output'):
            ui_components[output_key].clear_output(wait=True)

def _handle_cleanup_success(ui_components: Dict[str, Any], result: Dict[str, Any], logger) -> None:
    """Handle successful cleanup completion."""
    stats = result.get('stats', {})
    files_removed = stats.get('files_removed', 0)
    size_freed_mb = stats.get('size_freed_mb', 0)
    
    # Update progress completion
    ui_components.get('complete_operation', lambda x: None)(
        f"Cleanup selesai: {files_removed:,} file dihapus ({size_freed_mb:.1f}MB freed)"
    )
    
    # Update status panel
    _update_status_panel_success(ui_components, f"🧹 Cleanup berhasil: {files_removed:,} file dihapus")
    
    # Log detailed stats
    if logger and files_removed > 0:
        splits_cleaned = stats.get('splits_cleaned', 0)
        logger.success(f"✅ Cleanup selesai: {splits_cleaned} split, {files_removed:,} file, {size_freed_mb:.1f}MB freed")

def _update_status_panel_success(ui_components: Dict[str, Any], message: str) -> None:
    """Update status panel dengan success state."""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, "success")

def _update_status_panel_error(ui_components: Dict[str, Any], message: str) -> None:
    """Update status panel dengan error state."""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, "error")