"""
File: smartcash/ui/dataset/preprocessing/handlers/cleanup_executor.py
Deskripsi: SRP handler untuk cleanup operations dengan service layer integration
"""

from typing import Dict, Any
from smartcash.dataset.preprocessor.utils.preprocessing_factory import PreprocessingFactory
from smartcash.ui.dataset.preprocessing.utils.config_extractor import get_config_extractor
from smartcash.ui.utils.button_state_manager import get_button_state_manager

def setup_cleanup_executor(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup cleanup executor dengan service integration dan confirmation."""
    
    def execute_cleanup_action(button=None) -> None:
        """Execute cleanup dengan user confirmation."""
        logger = ui_components.get('logger')
        
        # Show confirmation dialog
        _show_cleanup_confirmation(ui_components, _perform_cleanup_operation, logger)
    
    def _perform_cleanup_operation() -> None:
        """Perform actual cleanup operation setelah konfirmasi."""
        logger = ui_components.get('logger')
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
                raise
    
    # Register handler
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].on_click(execute_cleanup_action)
    
    ui_components['execute_cleanup'] = execute_cleanup_action
    return ui_components

def _show_cleanup_confirmation(ui_components: Dict[str, Any], confirm_callback: callable, logger) -> None:
    """Show cleanup confirmation dialog."""
    try:
        from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation
        from IPython.display import display
        
        def on_confirm(button):
            ui_components.get('confirmation_area', type('', (), {'clear_output': lambda **kw: None})).clear_output(wait=True)
            confirm_callback()
        
        def on_cancel(button):
            ui_components.get('confirmation_area', type('', (), {'clear_output': lambda **kw: None})).clear_output(wait=True)
            logger and logger.info("🚫 Cleanup dibatalkan oleh user")
        
        # Create confirmation dialog
        confirmation_dialog = create_destructive_confirmation(
            title="⚠️ Konfirmasi Cleanup Dataset",
            message="Operasi ini akan menghapus SEMUA data preprocessed yang ada.\n\n"
                   "Data asli (source dataset) akan tetap aman, tetapi hasil preprocessing "
                   "akan hilang dan perlu dijalankan ulang.\n\n"
                   "Apakah Anda yakin ingin melanjutkan?",
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            item_name="data preprocessed"
        )
        
        # Display in confirmation area
        if 'confirmation_area' in ui_components:
            with ui_components['confirmation_area']:
                display(confirmation_dialog)
        
    except Exception as e:
        logger and logger.error(f"❌ Error showing confirmation: {str(e)}")
        # Fallback direct execution jika confirmation dialog gagal
        confirm_callback()

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
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(
            ui_components['status_panel'],
            f"🧹 Cleanup berhasil: {files_removed:,} file dihapus",
            "success"
        )
    
    # Log detailed stats
    if logger and files_removed > 0:
        splits_cleaned = stats.get('splits_cleaned', 0)
        logger.success(f"✅ Cleanup selesai: {splits_cleaned} split, {files_removed:,} file, {size_freed_mb:.1f}MB freed")