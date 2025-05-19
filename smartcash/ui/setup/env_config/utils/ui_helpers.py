"""
File: smartcash/ui/setup/env_config/utils/ui_helpers.py
Deskripsi: Fungsi helper untuk UI
"""

from typing import Dict, Any

def disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """
    Nonaktifkan UI selama proses berjalan.
    
    Args:
        ui_components: Dictionary UI components
        disable: True untuk nonaktifkan, False untuk aktifkan
    """
    # Daftar tombol yang akan dinonaktifkan
    button_keys = ['drive_button', 'directory_button', 'check_button', 'save_button']
    
    # Nonaktifkan tombol
    for key in button_keys:
        if key in ui_components:
            ui_components[key].disabled = disable

def cleanup_ui(ui_components: Dict[str, Any]) -> None:
    """
    Bersihkan UI setelah proses selesai.
    
    Args:
        ui_components: Dictionary UI components
    """
    # Aktifkan kembali tombol
    disable_ui_during_processing(ui_components, False)
    
    # Sembunyikan progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].layout.visibility = 'hidden'
    
    # Sembunyikan progress message
    if 'progress_message' in ui_components:
        ui_components['progress_message'].layout.visibility = 'hidden'
