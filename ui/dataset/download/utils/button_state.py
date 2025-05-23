"""
File: smartcash/ui/dataset/download/utils/button_state.py
Deskripsi: Manajemen state tombol UI
"""

def disable_download_buttons(ui_components: Dict[str, Any], disabled: bool) -> None:
    """Enable/disable semua tombol download."""
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    
    for key in button_keys:
        if key in ui_components and hasattr(ui_components[key], 'disabled'):
            ui_components[key].disabled = disabled