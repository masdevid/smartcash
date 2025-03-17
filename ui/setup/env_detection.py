"""
File: smartcash/ui/setup/env_detection.py
Deskripsi: Modul untuk deteksi environment SmartCash dengan styling konsisten
"""

import os
import sys
from typing import Dict, Any, Optional

def detect_environment(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """
    Deteksi dan konfigurasi environment UI dengan tema konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        
    Returns:
        Dictionary UI components yang telah diperbarui
    """
    # Import komponen UI yang sudah ada
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.components.alerts import create_info_alert
    
    # Coba gunakan environment manager atau gunakan fallback detection
    is_colab = False
    is_drive_mounted = False
    
    # Jika env sudah diberikan, gunakan itu
    if env and hasattr(env, 'is_colab'):
        is_colab = env.is_colab
        if hasattr(env, 'is_drive_mounted'):
            is_drive_mounted = env.is_drive_mounted
    else:
        # Coba import environment manager
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            is_colab = env_manager.is_colab
            is_drive_mounted = env_manager.is_drive_mounted
        except ImportError:
            # Fallback: Deteksi manual
            is_colab = 'google.colab' in sys.modules
            is_drive_mounted = os.path.exists('/content/drive/MyDrive')
    
    # Update colab panel dengan styling konsisten
    if 'colab_panel' in ui_components:
        if is_colab:
            # Tampilkan informasi Colab environment dengan komponen yang konsisten
            status_text = "terhubung" if is_drive_mounted else "tidak terhubung"
            icon = ICONS['success'] if is_drive_mounted else ICONS['warning']
            
            message = f"""
            <h3 style="color:{COLORS['primary']}; margin:5px 0">{ICONS['settings']} Environment: Google Colab</h3>
            <p style="margin:5px 0">{icon} Status Google Drive: <strong>{status_text}</strong></p>
            <p style="margin:5px 0">Klik tombol 'Hubungkan Google Drive' untuk mount drive dan menyinkronkan proyek.</p>
            """
            
            ui_components['colab_panel'].value = message
            # Aktifkan tombol drive
            ui_components['drive_button'].layout.display = 'block'
        else:
            # Tampilkan informasi local environment dengan komponen yang konsisten
            message = f"""
            <h3 style="color:{COLORS['success']}; margin:5px 0">{ICONS['settings']} Environment: Local</h3>
            <p style="margin:5px 0">Gunakan tombol 'Setup Direktori Lokal' untuk membuat struktur direktori proyek.</p>
            """
            
            ui_components['colab_panel'].value = message
            # Sembunyikan tombol drive
            ui_components['drive_button'].layout.display = 'none'
    
    return ui_components