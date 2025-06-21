"""
File: smartcash/ui/dataset/downloader/utils/dialog_utils.py
Deskripsi: Simplified dialog utilities menggunakan shared dialog API
"""

from typing import Dict, Any, Callable
from .ui_utils import clear_outputs
from .button_manager import get_button_manager

def show_download_confirmation_dialog(ui_config: Dict[str, Any], ui_components: Dict[str, Any], 
                                     existing_count: int, on_confirm: Callable, on_cancel: Callable = None):
    """Show download confirmation menggunakan shared dialog API"""
    try:
        from smartcash.ui.components.dialog import show_confirmation_dialog
        
        # Show confirmation area dan log waiting
        _show_confirmation_area(ui_components)
        _log_to_ui(ui_components, "⏳ Menunggu konfirmasi download dari user...", "info")
        
        roboflow = ui_config.get('data', {}).get('roboflow', {})
        download = ui_config.get('download', {})
        
        message_lines = [
            f"Dataset existing akan ditimpa! ({existing_count:,} gambar)",
            "",
            f"🎯 Target: {roboflow.get('workspace')}/{roboflow.get('project')}:v{roboflow.get('version')}",
            f"🔄 UUID Renaming: {'✅' if download.get('rename_files', True) else '❌'}",
            f"✅ Validasi: {'✅' if download.get('validate_download', True) else '❌'}",
            f"💾 Backup: {'✅' if download.get('backup_existing', False) else '❌'}",
            "",
            "Lanjutkan download?"
        ]
        
        show_confirmation_dialog(
            ui_components,
            title="⚠️ Konfirmasi Download Dataset",
            message="<br>".join(message_lines),
            on_confirm=lambda: _handle_confirm_with_hide(ui_components, on_confirm, "download"),
            on_cancel=lambda: _handle_cancel_with_hide(ui_components, on_cancel, "download"),
            confirm_text="Ya, Download",
            cancel_text="Batal",
            danger_mode=True
        )
        
    except ImportError:
        _log_to_ui(ui_components, "⚠️ Dialog API tidak tersedia, melanjutkan download", "warning")
        _hide_confirmation_area(ui_components)
        if on_confirm:
            on_confirm()

def show_cleanup_confirmation_dialog(ui_components: Dict[str, Any], targets_result: Dict[str, Any], 
                                   on_confirm: Callable, on_cancel: Callable = None):
    """Show cleanup confirmation menggunakan shared dialog API"""
    try:
        from smartcash.ui.components.dialog import show_confirmation_dialog
        
        # Show confirmation area dan log waiting
        _show_confirmation_area(ui_components)
        _log_to_ui(ui_components, "⏳ Menunggu konfirmasi cleanup dari user...", "info")
        
        summary = targets_result.get('summary', {})
        targets = targets_result.get('targets', {})
        
        message_lines = [
            f"Akan menghapus {summary.get('total_files', 0):,} file ({summary.get('size_formatted', '0 B')})",
            "",
            "📂 Target cleanup:"
        ]
        
        for target_name, target_info in targets.items():
            file_count = target_info.get('file_count', 0)
            size_formatted = target_info.get('size_formatted', '0 B')
            message_lines.append(f"  • {target_name}: {file_count:,} file ({size_formatted})")
        
        message_lines.extend(["", "⚠️ Direktori akan tetap dipertahankan", "Lanjutkan cleanup?"])
        
        show_confirmation_dialog(
            ui_components,
            title="⚠️ Konfirmasi Cleanup Dataset",
            message="<br>".join(message_lines),
            on_confirm=lambda: _handle_confirm_with_hide(ui_components, on_confirm, "cleanup"),
            on_cancel=lambda: _handle_cancel_with_hide(ui_components, on_cancel, "cleanup"),
            confirm_text="Ya, Hapus",
            cancel_text="Batal",
            danger_mode=True
        )
        
    except ImportError:
        _log_to_ui(ui_components, "⚠️ Dialog API tidak tersedia, operasi dibatalkan", "warning")
        _hide_confirmation_area(ui_components)
        button_manager = get_button_manager(ui_components)
        button_manager.enable_buttons()

def _handle_confirm_with_hide(ui_components: Dict[str, Any], on_confirm: Callable, operation_type: str):
    """Handle confirm dengan hide confirmation area"""
    _hide_confirmation_area(ui_components)
    _log_to_ui(ui_components, f"✅ {operation_type.capitalize()} dikonfirmasi oleh user", "success")
    if on_confirm:
        on_confirm()

def _handle_cancel_with_hide(ui_components: Dict[str, Any], on_cancel: Callable, operation_type: str):
    """Handle cancel dengan hide confirmation area"""
    from .ui_utils import hide_confirmation_area
    
    hide_confirmation_area(ui_components)
    _log_to_ui(ui_components, f"🚫 {operation_type.capitalize()} dibatalkan oleh user", "info")
    
    if on_cancel:
        on_cancel()
    else:
        # Default cancel behavior jika tidak ada custom callback
        button_manager = get_button_manager(ui_components)
        button_manager.enable_buttons()
        clear_outputs(ui_components)

def _show_confirmation_area(ui_components: Dict[str, Any]):
    """Show confirmation area dengan visibility management"""
    from .ui_utils import show_confirmation_area
    show_confirmation_area(ui_components)

def _hide_confirmation_area(ui_components: Dict[str, Any]):
    """Hide confirmation area dengan visibility management"""
    from .ui_utils import hide_confirmation_area
    hide_confirmation_area(ui_components)

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log message ke UI dengan fallback"""
    logger = ui_components.get('logger')
    if logger and hasattr(logger, level):
        getattr(logger, level)(message)
    else:
        # Fallback ke log_to_accordion
        try:
            from .ui_utils import log_to_accordion
            log_to_accordion(ui_components, message, level)
        except ImportError:
            print(f"[{level.upper()}] {message}")

def create_cancel_callback(ui_components: Dict[str, Any], operation_type: str) -> Callable:
    """Create cancel callback dengan proper state reset"""
    def on_cancel():
        logger = ui_components.get('logger')
        button_manager = get_button_manager(ui_components)
        
        if logger:
            logger.info(f"🚫 {operation_type.capitalize()} dibatalkan oleh user")
        
        # Reset state
        button_manager.enable_buttons()
        clear_outputs(ui_components)
        
        # Reset progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
    
    return on_cancel

def create_confirm_callback(ui_components: Dict[str, Any], operation_type: str, execute_function: Callable) -> Callable:
    """Create confirm callback dengan proper logging"""
    def on_confirm():
        logger = ui_components.get('logger')
        if logger:
            logger.info(f"✅ {operation_type.capitalize()} dikonfirmasi oleh user")
        
        # Execute operation
        if execute_function:
            execute_function()
    
    return on_confirm