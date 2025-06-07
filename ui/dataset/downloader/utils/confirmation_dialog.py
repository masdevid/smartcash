"""
File: smartcash/ui/dataset/downloader/utils/confirmation_dialog.py
Deskripsi: Fixed confirmation dialog khusus untuk downloader dengan cleanup behavior
"""

from typing import Callable, Dict, Any
from IPython.display import display, clear_output
from smartcash.ui.components.dialogs import show_destructive_confirmation

def show_downloader_confirmation_dialog(ui_components: Dict[str, Any], existing_count: int, 
                                       config: Dict[str, Any], on_confirm: Callable):
    """
    Show confirmation dialog khusus downloader dengan cleanup warning yang spesifik.
    
    Args:
        ui_components: UI components dict dengan confirmation_area
        existing_count: Jumlah file existing yang akan dihapus
        config: Download configuration
        on_confirm: Callback saat user confirm
    """
    confirmation_area = ui_components.get('confirmation_area')
    if not confirmation_area:
        # Fallback ke console confirmation
        _console_confirmation(existing_count, config, on_confirm)
        return
    
    # Force confirmation area visible
    _ensure_confirmation_area_visible(confirmation_area)
    
    # Build downloader-specific message
    message = _build_downloader_confirmation_message(existing_count, config)
    
    def safe_confirm(btn):
        """Safe confirm dengan cleanup confirmation area"""
        with confirmation_area:
            clear_output(wait=True)
        on_confirm(btn)
    
    def safe_cancel(btn):
        """Safe cancel dengan proper button state reset"""
        with confirmation_area:
            clear_output(wait=True)
        _handle_download_cancellation(ui_components)
    
    # Display dialog dalam confirmation area
    with confirmation_area:
        clear_output(wait=True)
        dialog = show_destructive_confirmation(
            "⚠️ Konfirmasi Download Dataset",
            message,
            "dataset existing", 
            safe_confirm, 
            safe_cancel
        )
        display(dialog)

def _build_downloader_confirmation_message(existing_count: int, config: Dict[str, Any]) -> str:
    """Build message yang spesifik untuk downloader cleanup behavior"""
    roboflow = config.get('data', {}).get('roboflow', {})
    download = config.get('download', {})
    backup_enabled = download.get('backup_existing', False)
    
    lines = [
        f"🗂️ Dataset existing: {existing_count:,} file akan DIHAPUS PERMANEN",
        f"🎯 Target: {roboflow.get('workspace')}/{roboflow.get('project')}:v{roboflow.get('version')}",
        "",
        "📋 Proses Download Downloader:",
        "  1️⃣ Download dataset baru dari Roboflow",
        "  2️⃣ Extract ke direktori temporary",
        "  3️⃣ 🚨 HAPUS dataset lama (train/valid/test)",
        "  4️⃣ Organize dataset baru ke struktur final", 
        "  5️⃣ UUID renaming untuk konsistensi research",
        "",
        f"🔄 UUID Renaming: {'✅' if download.get('rename_files', True) else '❌'} Aktif",
        f"✅ Validasi: {'✅' if download.get('validate_download', True) else '❌'} Aktif",
        f"💾 Backup: {'✅' if backup_enabled else '❌'} {_get_backup_status_text(backup_enabled)}",
        "",
        _get_safety_warning(backup_enabled)
    ]
    
    return '\n'.join(lines)

def _get_backup_status_text(backup_enabled: bool) -> str:
    """Get backup status text yang sesuai"""
    if backup_enabled:
        return "Dataset lama akan dibackup ke data/backup/"
    else:
        return "TIDAK ADA BACKUP - DATA HILANG PERMANEN!"

def _get_safety_warning(backup_enabled: bool) -> str:
    """Get safety warning berdasarkan backup status"""
    if backup_enabled:
        return "⚠️ Dataset lama akan dihapus tapi ada backup untuk recovery"
    else:
        return "🚨 BAHAYA: Dataset lama akan HILANG PERMANEN tanpa backup!"

def _ensure_confirmation_area_visible(confirmation_area):
    """Ensure confirmation area visible dan siap display dialog"""
    if hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.display = 'block'
        confirmation_area.layout.visibility = 'visible'
        confirmation_area.layout.height = 'auto'
        confirmation_area.layout.min_height = '100px'
        confirmation_area.layout.overflow = 'visible'

def _console_confirmation(existing_count: int, config: Dict[str, Any], on_confirm: Callable):
    """Fallback console confirmation jika UI tidak tersedia"""
    backup_enabled = config.get('download', {}).get('backup_existing', False)
    
    print("⚠️ KONFIRMASI DOWNLOAD DATASET")
    print("=" * 40)
    print(f"🗂️ File existing: {existing_count:,} akan dihapus")
    print(f"💾 Backup: {'✅ Aktif' if backup_enabled else '❌ TIDAK AKTIF'}")
    
    if not backup_enabled:
        print("🚨 PERINGATAN: DATA AKAN HILANG PERMANEN!")
    
    response = input("\nLanjutkan download? (y/n): ")
    if response.lower() in ['y', 'yes', 'ya']:
        on_confirm(None)

def _handle_download_cancellation(ui_components: Dict[str, Any]):
    """Handle download cancellation dengan proper state reset"""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🚫 Download dibatalkan oleh user")
    
    # Reset button states
    from smartcash.ui.dataset.downloader.utils.button_manager import get_button_manager
    button_manager = get_button_manager(ui_components)
    button_manager.enable_buttons()

def check_confirmation_area_availability(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check availability dan status confirmation area untuk debugging.
    
    Returns:
        Status dict untuk troubleshooting
    """
    confirmation_area = ui_components.get('confirmation_area')
    
    status = {
        'exists': confirmation_area is not None,
        'has_layout': hasattr(confirmation_area, 'layout') if confirmation_area else False,
        'has_clear_output': hasattr(confirmation_area, 'clear_output') if confirmation_area else False,
        'widget_type': type(confirmation_area).__name__ if confirmation_area else None
    }
    
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        layout = confirmation_area.layout
        status.update({
            'display': getattr(layout, 'display', 'not_set'),
            'visibility': getattr(layout, 'visibility', 'not_set'),
            'height': getattr(layout, 'height', 'not_set'),
            'width': getattr(layout, 'width', 'not_set')
        })
    
    return status