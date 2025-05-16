"""
File: smartcash/ui/setup/ui_helpers.py
Deskripsi: Helper functions untuk UI setup
"""

from typing import Dict, Any

def disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """
    Nonaktifkan UI selama proses berjalan
    
    Args:
        ui_components: Dictionary berisi komponen UI
        disable: True untuk nonaktifkan, False untuk aktifkan
    """
    # Daftar tombol yang akan dinonaktifkan
    buttons = ['drive_button', 'directory_button', 'check_button', 'save_button']
    
    # Nonaktifkan atau aktifkan tombol
    for button_name in buttons:
        if button_name in ui_components:
            ui_components[button_name].disabled = disable

def cleanup_ui(ui_components: Dict[str, Any]) -> None:
    """
    Bersihkan UI setelah proses selesai
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Aktifkan kembali tombol
    disable_ui_during_processing(ui_components, False)
    
    # Sembunyikan progress bar dan message
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].layout.visibility = 'hidden'
    
    if 'progress_message' in ui_components:
        ui_components['progress_message'].layout.visibility = 'hidden'
