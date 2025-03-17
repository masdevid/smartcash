"""
File: smartcash/ui/setup/env_detection.py
Deskripsi: Modul untuk deteksi environment SmartCash
"""

import os
import sys
from typing import Dict, Any, Optional

def detect_environment(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """
    Deteksi dan konfigurasi environment UI.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        
    Returns:
        Dictionary UI components yang telah diperbarui
    """
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
    
    # Update colab panel
    if 'colab_panel' in ui_components:
        if is_colab:
            # Tampilkan informasi Colab environment
            status = "terhubung" if is_drive_mounted else "tidak terhubung"
            icon = "✅" if is_drive_mounted else "⚠️"
            
            ui_components['colab_panel'].value = f"""
            <div style="padding:10px; background-color:#d1ecf1; color:#0c5460; border-radius:4px; margin:10px 0">
                <h3 style="color:inherit; margin:5px 0">🔍 Environment: Google Colab</h3>
                <p style="margin:5px 0">{icon} Status Google Drive: <strong>{status}</strong></p>
                <p style="margin:5px 0">Klik tombol 'Hubungkan Google Drive' untuk mount drive dan menyinkronkan proyek.</p>
            </div>
            """
            # Aktifkan tombol drive
            ui_components['drive_button'].layout.display = 'block'
        else:
            # Tampilkan informasi local environment
            ui_components['colab_panel'].value = """
            <div style="padding:10px; background-color:#d4edda; color:#155724; border-radius:4px; margin:10px 0">
                <h3 style="color:inherit; margin:5px 0">🔍 Environment: Local</h3>
                <p style="margin:5px 0">Gunakan tombol 'Setup Direktori Lokal' untuk membuat struktur direktori proyek.</p>
            </div>
            """
            # Sembunyikan tombol drive
            ui_components['drive_button'].layout.display = 'none'
    
    return ui_components