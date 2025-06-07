"""
File: smartcash/ui/components/dialogs/dialog_display.py
Deskripsi: Fixed dialog display untuk ensure confirmation dialogs muncul dengan proper visibility
"""

from typing import Callable, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output
from smartcash.ui.components.dialogs import show_destructive_confirmation

def display_confirmation_dialog(confirmation_area: widgets.Output, title: str, message: str, 
                               on_confirm: Callable, on_cancel: Callable = None):
    """
    Display confirmation dialog dengan force visibility dan proper output management.
    
    Args:
        confirmation_area: Output widget untuk display dialog
        title: Dialog title
        message: Dialog message  
        on_confirm: Callback saat confirm
        on_cancel: Callback saat cancel
    """
    if not confirmation_area or not hasattr(confirmation_area, 'clear_output'):
        # Fallback ke print jika output widget tidak tersedia
        print(f"⚠️ Confirmation required: {title}")
        print(f"📝 {message}")
        response = input("Lanjutkan? (y/n): ")
        if response.lower() in ['y', 'yes', 'ya']:
            on_confirm(None)
        elif on_cancel:
            on_cancel(None)
        return
    
    def safe_confirm(btn):
        """Safe confirm dengan cleanup output"""
        try:
            with confirmation_area:
                clear_output(wait=True)
            if on_confirm:
                on_confirm(btn)
        except Exception as e:
            print(f"❌ Error in confirmation: {str(e)}")
    
    def safe_cancel(btn):
        """Safe cancel dengan cleanup output"""
        try:
            with confirmation_area:
                clear_output(wait=True)
            if on_cancel:
                on_cancel(btn)
        except Exception as e:
            print(f"❌ Error in cancellation: {str(e)}")
    
    # Display dialog dalam output area
    with confirmation_area:
        clear_output(wait=True)
        
        # Create dan display dialog
        dialog = show_destructive_confirmation(title, message, "dataset", safe_confirm, safe_cancel)
        display(dialog)
    
    # Ensure output area visible
    if hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.display = 'block'
        confirmation_area.layout.visibility = 'visible'
        confirmation_area.layout.height = 'auto'

def show_download_confirmation_with_cleanup_info(ui_components: dict, existing_count: int, 
                                                config: dict, on_confirm: Callable):
    """
    Show download confirmation dengan info cleanup behavior yang jelas.
    
    Args:
        ui_components: UI components dict
        existing_count: Jumlah file existing
        config: Download config
        on_confirm: Callback saat konfirmasi
    """
    roboflow = config.get('data', {}).get('roboflow', {})
    download = config.get('download', {})
    
    # Build detailed message dengan cleanup info
    cleanup_info = [
        f"🗂️ Dataset existing: {existing_count:,} file akan ditimpa",
        f"🎯 Target: {roboflow.get('workspace')}/{roboflow.get('project')}:v{roboflow.get('version')}",
        "",
        "📋 Proses yang akan dilakukan:",
        "  1️⃣ Download dataset baru dari Roboflow",
        "  2️⃣ Extract ke direktori temporary", 
        "  3️⃣ Hapus dataset lama (train/valid/test)",
        "  4️⃣ Organize dataset baru ke struktur final",
        "  5️⃣ UUID renaming untuk konsistensi research",
        "",
        f"🔄 UUID Renaming: {'✅ Aktif' if download.get('rename_files', True) else '❌ Nonaktif'}",
        f"✅ Validasi: {'✅ Aktif' if download.get('validate_download', True) else '❌ Nonaktif'}",
        f"💾 Backup: {'✅ Aktif' if download.get('backup_existing', False) else '❌ Nonaktif'}",
        "",
        "⚠️ PERHATIAN: Dataset lama akan dihapus permanen!"
    ]
    
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        display_confirmation_dialog(
            confirmation_area,
            "⚠️ Konfirmasi Download Dataset", 
            '\n'.join(cleanup_info),
            on_confirm,
            lambda btn: _cancel_download(ui_components)
        )
    else:
        # Fallback confirmation
        print("⚠️ Confirmation Dialog")
        print('\n'.join(cleanup_info))
        response = input("\nLanjutkan download? (y/n): ")
        if response.lower() in ['y', 'yes', 'ya']:
            on_confirm(None)

def _cancel_download(ui_components: dict):
    """Cancel download dengan proper state reset"""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🚫 Download dibatalkan oleh user")
    
    # Enable buttons kembali
    from smartcash.ui.dataset.downloader.utils.button_manager import get_button_manager
    button_manager = get_button_manager(ui_components)
    button_manager.enable_buttons()
    
    # Clear confirmation area
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        with confirmation_area:
            clear_output(wait=True)

def ensure_confirmation_area_visible(ui_components: dict):
    """Ensure confirmation area visible dan ready untuk display dialog"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        # Force visibility
        confirmation_area.layout.display = 'block'
        confirmation_area.layout.visibility = 'visible'
        confirmation_area.layout.overflow = 'visible'
        confirmation_area.layout.height = 'auto'
        confirmation_area.layout.min_height = '50px'
        
        # Clear any existing content
        with confirmation_area:
            clear_output(wait=True)
        
        return True
    return False