"""
File: smartcash/ui/setup/environment_detector.py
Deskripsi: Utilitas terstandarisasi untuk deteksi environment dengan satu tanggung jawab
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

def detect_environment(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """
    Deteksi dan konfigurasi environment UI.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        
    Returns:
        Dictionary UI components yang telah diperbarui
    """
    # Update status panel jika tersedia
    if 'status_panel' in ui_components:
        from smartcash.ui.utils.alert_utils import create_info_box
        ui_components['status_panel'].value = create_info_box(
            "Mendeteksi Environment", 
            "Sedang mendeteksi environment...",
            style="info"
        ).value
    
    # Deteksi environment menggunakan env atau fallback
    env_info = _detect_env_info(env)
    is_colab, is_drive_mounted, drive_path = env_info
    
    # Update status panel
    if 'status_panel' in ui_components and 'logger' in ui_components:
        ui_components['logger'].info(f"🔍 Environment: {'Google Colab' if is_colab else 'Local'}")
    
    # Update colab panel
    if 'colab_panel' in ui_components:
        _update_colab_panel(ui_components, is_colab, is_drive_mounted, drive_path)
    
    # Update status panel lagi
    if 'logger' in ui_components:
        ui_components['logger'].info(f"💾 Drive: {'terhubung' if is_drive_mounted else 'tidak terhubung'}")
    
    return ui_components

def _detect_env_info(env) -> Tuple[bool, bool, Optional[Path]]:
    """
    Deteksi informasi environment.
    
    Args:
        env: Environment manager
        
    Returns:
        Tuple (is_colab, is_drive_mounted, drive_path)
    """
    # Prefer env kalau sudah ada, jika tidak maka deteksi sendiri
    if env and hasattr(env, 'is_colab'):
        return env.is_colab, getattr(env, 'is_drive_mounted', False), getattr(env, 'drive_path', None)
    
    # Fallback: Deteksi manual dengan one-liner
    is_colab = 'google.colab' in sys.modules
    is_drive_mounted = os.path.exists('/content/drive/MyDrive')
    drive_path = Path('/content/drive/MyDrive/SmartCash') if is_drive_mounted else None
    
    return is_colab, is_drive_mounted, drive_path

def _update_colab_panel(ui_components: Dict[str, Any], is_colab: bool, is_drive_mounted: bool, drive_path: Optional[Path]) -> None:
    """
    Update panel informasi Colab.
    
    Args:
        ui_components: Dictionary komponen UI
        is_colab: Boolean menunjukkan apakah di Colab
        is_drive_mounted: Boolean menunjukkan apakah Drive terpasang
        drive_path: Path ke Drive jika terpasang
    """
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    if is_colab:
        # Tampilkan informasi Colab environment
        status = "terhubung" if is_drive_mounted else "tidak terhubung"
        icon = "✅" if is_drive_mounted else "⚠️"
        alert_type = "success" if is_drive_mounted else "warning"
        
        drive_info = f" ({drive_path})" if drive_path and is_drive_mounted else ""
        action_msg = 'Klik tombol "Setup Direktori Lokal" untuk membuat struktur direktori proyek.' if is_drive_mounted else \
                    'Klik tombol "Hubungkan Google Drive" untuk mount drive dan menyinkronkan proyek.'
        
        ui_components['colab_panel'].value = create_info_alert(
            f"""<h3 style="color:inherit; margin:5px 0">🔍 Environment: Google Colab</h3>
            <p style="margin:5px 0">{icon} Status Google Drive: <strong>{status}</strong>{drive_info}</p>
            <p style="margin:5px 0">{action_msg}</p>""",
            alert_type
        ).value
        
        # Tampilkan/sembunyikan tombol drive berdasarkan status
        ui_components['drive_button'].layout.display = 'none' if is_drive_mounted else 'block'
    else:
        # Tampilkan informasi local environment
        ui_components['colab_panel'].value = create_info_alert(
            f"""<h3 style="color:inherit; margin:5px 0">🔍 Environment: Local</h3>
            <p style="margin:5px 0">✅ Status: <strong>Siap digunakan</strong></p>
            <p style="margin:5px 0">Gunakan tombol 'Setup Direktori Lokal' untuk membuat struktur direktori proyek.</p>""",
            "success"
        ).value
        
        # Sembunyikan tombol drive di environment lokal
        ui_components['drive_button'].layout.display = 'none'

# Fungsi-fungsi terkait progress bar dihapus karena tidak diperlukan lagi