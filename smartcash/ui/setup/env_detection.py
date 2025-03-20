"""
File: smartcash/ui/setup/env_detection.py
Deskripsi: Modul untuk deteksi environment SmartCash dengan integrasi UI utils
"""

import os
import sys
from typing import Dict, Any, Optional
from IPython.display import HTML

from smartcash.ui.utils.constants import COLORS, ICONS 
from smartcash.ui.utils.alert_utils import create_info_alert

def detect_environment(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """
    Deteksi dan konfigurasi environment UI dengan integrasi UI utils.
    
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
        # Import dengan fallback
        from smartcash.ui.utils.fallback_utils import import_with_fallback
        
        # Coba import environment manager
        get_env_manager = import_with_fallback('smartcash.common.environment.get_environment_manager')
        if get_env_manager:
            env_manager = get_env_manager()
            is_colab = env_manager.is_colab
            is_drive_mounted = env_manager.is_drive_mounted
        else:
            # Fallback: Deteksi manual
            is_colab = 'google.colab' in sys.modules
            is_drive_mounted = os.path.exists('/content/drive/MyDrive')
    
    # Update colab panel dengan menggunakan komponen create_info_alert
    if 'colab_panel' in ui_components:
        if is_colab:
            # Tampilkan informasi Colab environment
            status = "terhubung" if is_drive_mounted else "tidak terhubung"
            icon = "✅" if is_drive_mounted else "⚠️"
            alert_type = "success" if is_drive_mounted else "warning"
            
            ui_components['colab_panel'].value = create_info_alert(
                f"""<h3 style="color:inherit; margin:5px 0">🔍 Environment: Google Colab</h3>
                <p style="margin:5px 0">{icon} Status Google Drive: <strong>{status}</strong></p>
                <p style="margin:5px 0">Klik tombol 'Hubungkan Google Drive' untuk mount drive dan menyinkronkan proyek.</p>""",
                alert_type
            ).value
            
            # Aktifkan tombol drive
            ui_components['drive_button'].layout.display = 'block'
        else:
            # Tampilkan informasi local environment
            ui_components['colab_panel'].value = create_info_alert(
                f"""<h3 style="color:inherit; margin:5px 0">🔍 Environment: Local</h3>
                <p style="margin:5px 0">✅ Status: <strong>Siap digunakan</strong></p>
                <p style="margin:5px 0">Gunakan tombol 'Setup Direktori Lokal' untuk membuat struktur direktori proyek.</p>""",
                "success"
            ).value
            
            # Sembunyikan tombol drive
            ui_components['drive_button'].layout.display = 'none'
    
    return ui_components