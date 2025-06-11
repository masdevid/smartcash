"""
File: smartcash/ui/dataset/downloader/utils/confirmation_dialog.py
Deskripsi: Fixed confirmation dialog dengan sync execution dan debugging
"""

from typing import Callable, Dict, Any
from IPython.display import display, clear_output
import ipywidgets as widgets

def show_downloader_confirmation_dialog(ui_components: Dict[str, Any], existing_count: int, 
                                       config: Dict[str, Any], on_confirm: Callable):
    """
    Show confirmation dialog dengan sync execution dan force display.
    
    Args:
        ui_components: UI components dict dengan confirmation_area
        existing_count: Jumlah file existing yang akan dihapus
        config: Download configuration
        on_confirm: Callback saat user confirm
    """
    confirmation_area = ui_components.get('confirmation_area')
    logger = ui_components.get('logger')
    
    # Debug confirmation area
    if logger:
        area_status = _debug_confirmation_area(confirmation_area)
        logger.info(f"🔍 Confirmation area status: {area_status}")
    
    if not confirmation_area:
        # Fallback ke console confirmation
        if logger:
            logger.warning("⚠️ No confirmation_area, using console fallback")
        _console_confirmation(existing_count, config, on_confirm, logger)
        return
    
    try:
        # Force confirmation area visible dan ready
        _force_confirmation_area_ready(confirmation_area, logger)
        
        # Build message
        message = _build_downloader_confirmation_message(existing_count, config)
        
        # Create confirmation widget langsung tanpa external dialog
        confirmation_widget = _create_inline_confirmation_widget(
            message, 
            lambda: _handle_confirm(confirmation_area, on_confirm, logger),
            lambda: _handle_cancel(confirmation_area, ui_components, logger)
        )
        
        # Display dengan force update
        with confirmation_area:
            clear_output(wait=True)
            display(confirmation_widget)
        
        if logger:
            logger.info("✅ Confirmation dialog displayed successfully")
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Error showing confirmation: {str(e)}")
        # Fallback to console
        _console_confirmation(existing_count, config, on_confirm, logger)

def _debug_confirmation_area(confirmation_area) -> str:
    """Debug confirmation area untuk troubleshooting"""
    if not confirmation_area:
        return "None"
    
    info = {
        'type': type(confirmation_area).__name__,
        'has_layout': hasattr(confirmation_area, 'layout'),
        'has_clear_output': hasattr(confirmation_area, 'clear_output')
    }
    
    if hasattr(confirmation_area, 'layout'):
        layout = confirmation_area.layout
        info.update({
            'display': getattr(layout, 'display', 'not_set'),
            'visibility': getattr(layout, 'visibility', 'not_set'),
            'height': getattr(layout, 'height', 'not_set')
        })
    
    return str(info)

def _force_confirmation_area_ready(confirmation_area, logger):
    """Force confirmation area ready untuk display"""
    if not hasattr(confirmation_area, 'layout'):
        if logger:
            logger.warning("⚠️ Confirmation area has no layout attribute")
        return
    
    # Force visibility settings
    layout_updates = {
        'display': 'block',
        'visibility': 'visible', 
        'height': 'auto',
        'min_height': '200px',
        'max_height': '600px',
        'overflow': 'auto',
        'border': '1px solid #ddd',
        'padding': '10px',
        'background_color': 'white'
    }
    
    for attr, value in layout_updates.items():
        try:
            setattr(confirmation_area.layout, attr, value)
        except Exception as e:
            if logger:
                logger.debug(f"Could not set {attr}: {str(e)}")

def _create_inline_confirmation_widget(message: str, on_confirm: Callable, on_cancel: Callable) -> widgets.VBox:
    """Create inline confirmation widget tanpa external dialog dependency"""
    
    # Title
    title = widgets.HTML(
        '<h3 style="color: #dc3545; margin: 0 0 15px 0;">⚠️ Konfirmasi Download Dataset</h3>',
        layout=widgets.Layout(margin='0 0 10px 0')
    )
    
    # Message
    message_widget = widgets.HTML(
        f'<div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ffc107;"><pre style="white-space: pre-wrap; margin: 0; font-family: monospace; font-size: 12px;">{message}</pre></div>',
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Buttons
    confirm_button = widgets.Button(
        description="Ya, Lanjutkan Download",
        button_style='danger',
        icon='download',
        layout=widgets.Layout(width='200px', height='35px', margin='0 10px 0 0')
    )
    
    cancel_button = widgets.Button(
        description="Batal",
        button_style='',
        icon='times',
        layout=widgets.Layout(width='100px', height='35px')
    )
    
    # Button handlers
    def safe_confirm(btn):
        btn.disabled = True
        cancel_button.disabled = True
        btn.description = "Processing..."
        on_confirm()
    
    def safe_cancel(btn):
        btn.disabled = True
        confirm_button.disabled = True
        on_cancel()
    
    confirm_button.on_click(safe_confirm)
    cancel_button.on_click(safe_cancel)
    
    # Button container
    button_container = widgets.HBox(
        [confirm_button, cancel_button],
        layout=widgets.Layout(justify_content='flex-end', margin='15px 0 0 0')
    )
    
    # Main container
    return widgets.VBox([
        title,
        message_widget,
        button_container
    ], layout=widgets.Layout(
        width='100%',
        padding='20px',
        border='2px solid #dc3545',
        border_radius='8px',
        background_color='#fff'
    ))

def _handle_confirm(confirmation_area, on_confirm: Callable, logger):
    """Handle confirm dengan cleanup"""
    try:
        if logger:
            logger.info("✅ User confirmed download")
        
        # Clear confirmation area
        with confirmation_area:
            clear_output(wait=True)
        
        # Execute callback
        on_confirm(None)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error in confirm handler: {str(e)}")

def _handle_cancel(confirmation_area, ui_components: Dict[str, Any], logger):
    """Handle cancel dengan proper state reset"""
    try:
        if logger:
            logger.info("🚫 User cancelled download")
        
        # Clear confirmation area
        with confirmation_area:
            clear_output(wait=True)
        
        # Reset button states
        from smartcash.ui.dataset.downloader.utils.button_manager import get_button_manager
        button_manager = get_button_manager(ui_components)
        button_manager.enable_buttons()
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error in cancel handler: {str(e)}")

def _build_downloader_confirmation_message(existing_count: int, config: Dict[str, Any]) -> str:
    """Build confirmation message dengan cleanup info"""
    roboflow = config.get('data', {}).get('roboflow', {})
    download = config.get('download', {})
    backup_enabled = download.get('backup_existing', False)
    
    lines = [
        f"🗂️ Dataset Existing: {existing_count:,} file akan DIHAPUS PERMANEN",
        f"🎯 Target: {roboflow.get('workspace')}/{roboflow.get('project')}:v{roboflow.get('version')}",
        "",
        "📋 Proses Download:",
        "  1️⃣ Download dataset baru dari Roboflow",
        "  2️⃣ Extract ke direktori temporary",
        "  3️⃣ 🚨 HAPUS dataset lama (train/valid/test)", 
        "  4️⃣ Organize dataset baru ke struktur final",
        "  5️⃣ UUID renaming untuk konsistensi research",
        "",
        f"💾 Backup: {'✅ AKTIF' if backup_enabled else '❌ TIDAK AKTIF'}",
        f"🔄 UUID Rename: {'✅' if download.get('rename_files', True) else '❌'}",
        f"✅ Validasi: {'✅' if download.get('validate_download', True) else '❌'}",
        "",
    ]
    
    if backup_enabled:
        lines.append("✅ Dataset lama akan dibackup sebelum dihapus")
    else:
        lines.append("🚨 PERINGATAN: DATA AKAN HILANG PERMANEN!")
    
    return '\n'.join(lines)

def _console_confirmation(existing_count: int, config: Dict[str, Any], on_confirm: Callable, logger):
    """Fallback console confirmation"""
    backup_enabled = config.get('download', {}).get('backup_existing', False)
    
    print("\n" + "="*50)
    print("⚠️ KONFIRMASI DOWNLOAD DATASET")
    print("="*50)
    print(f"🗂️ File existing: {existing_count:,} akan dihapus")
    print(f"💾 Backup: {'✅ Aktif' if backup_enabled else '❌ TIDAK AKTIF'}")
    
    if not backup_enabled:
        print("🚨 PERINGATAN: DATA AKAN HILANG PERMANEN!")
    
    try:
        response = input("\nLanjutkan download? (y/n): ")
        if response.lower() in ['y', 'yes', 'ya']:
            if logger:
                logger.info("✅ Console confirmation: YES")
            on_confirm(None)
        else:
            if logger:
                logger.info("🚫 Console confirmation: NO")
    except KeyboardInterrupt:
        if logger:
            logger.info("🚫 Console confirmation: Interrupted")
        print("\nDownload cancelled.")
