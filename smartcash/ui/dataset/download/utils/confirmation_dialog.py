"""
File: smartcash/ui/dataset/download/utils/confirmation_dialog.py
Deskripsi: Updated confirmation dengan button_state_manager integration
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog

def show_download_confirmation(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Show download confirmation dengan storage info dan proper state management."""
    
    # Determine storage type
    env_manager = ui_components.get('env_manager')
    if env_manager and env_manager.is_drive_mounted:
        storage_info = f"📁 Storage: Google Drive ({env_manager.drive_path})"
    else:
        storage_info = "📁 Storage: Local (akan hilang saat restart)"
    
    message = (
        f"Download dataset dari Roboflow:\n\n"
        f"• Workspace: {params['workspace']}\n"
        f"• Project: {params['project']}\n" 
        f"• Version: {params['version']}\n"
        f"• Output: {params['output_dir']}\n"
        f"• {storage_info}\n\n"
        f"Lanjutkan download?"
    )
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        _execute_confirmed_download(ui_components, params)
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        _handle_cancel_download(ui_components)
    
    dialog = create_confirmation_dialog(
        title="Konfirmasi Download Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel
    )
    
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)
        
def show_cleanup_confirmation(ui_components: Dict[str, Any], output_dir: str, file_count: int) -> None:
    """Tampilkan dialog konfirmasi cleanup dengan proper state management."""
    message = (
        f"Anda akan menghapus {file_count} file dari:\n"
        f"{output_dir}\n\n"
        f"Tindakan ini tidak dapat dibatalkan.\n"
        f"Lanjutkan cleanup?"
    )
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        from smartcash.ui.dataset.download.handlers.cleanup_action import execute_cleanup_confirmed
        execute_cleanup_confirmed(ui_components, output_dir)
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        _handle_cancel_cleanup(ui_components)
    
    dialog = create_confirmation_dialog(
        title="Konfirmasi Hapus Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel
    )
    
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)

def _execute_confirmed_download(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Execute download setelah konfirmasi dengan proper state management."""
    from smartcash.ui.dataset.download.utils.download_executor import execute_roboflow_download
    from smartcash.ui.dataset.download.utils.button_state_manager import get_button_state_manager
    
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    if logger:
        logger.info("🚀 Memulai download dataset")
    
    try:
        # Use context manager untuk proper state handling
        with button_manager.operation_context('download'):
            result = execute_roboflow_download(ui_components, params)
            
            if result.get('status') == 'success':
                if logger:
                    storage_type = "Drive" if result.get('drive_storage', False) else "Local"
                    logger.success(f"✅ Download berhasil ke {storage_type}")
            else:
                if logger:
                    logger.error(f"❌ Download gagal: {result.get('message', 'Unknown error')}")
                    
    except Exception as e:
        if logger:
            logger.error(f"❌ Error: {str(e)}")

def _handle_cancel_download(ui_components: Dict[str, Any]) -> None:
    """Handle cancel download dengan proper state reset."""
    from smartcash.ui.dataset.download.utils.button_state_manager import get_button_state_manager
    
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    # Enable semua buttons kembali
    button_manager.enable_buttons('all')
    
    if logger:
        logger.info("❌ Download dibatalkan")

def _handle_cancel_cleanup(ui_components: Dict[str, Any]) -> None:
    """Handle cancel cleanup dengan proper state reset."""
    from smartcash.ui.dataset.download.utils.button_state_manager import get_button_state_manager
    
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    # Enable semua buttons kembali
    button_manager.enable_buttons('all')
    
    if logger:
        logger.info("❌ Cleanup dibatalkan")