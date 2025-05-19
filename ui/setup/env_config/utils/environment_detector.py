"""
File: smartcash/ui/setup/env_config/utils/environment_detector.py
Deskripsi: Utilitas untuk deteksi environment
"""

from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display

def detect_environment(ui_components: Dict[str, Any], env_manager: Any) -> Dict[str, Any]:
    """
    Deteksi environment dan update UI sesuai dengan kondisi
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env_manager: Environment manager
        
    Returns:
        Dictionary berisi komponen UI yang telah diupdate
    """
    # Dapatkan komponen UI
    drive_button = ui_components.get('drive_button')
    directory_button = ui_components.get('directory_button')
    status = ui_components.get('status')
    logger = ui_components.get('logger')
    
    # Cek apakah environment sudah terdeteksi
    if not env_manager:
        if logger: logger.error("❌ Environment manager tidak tersedia")
        return ui_components
    
    # Cek apakah running di Colab
    if env_manager.is_colab:
        if logger: logger.info("🔍 Terdeteksi running di Google Colab")
        
        # Tampilkan tombol connect drive jika belum terhubung
        if drive_button and not env_manager.is_drive_mounted:
            drive_button.layout.display = 'block'
        else:
            drive_button.layout.display = 'none'
    else:
        if logger: logger.info("🔍 Terdeteksi running di environment lokal")
        
        # Sembunyikan tombol connect drive
        if drive_button:
            drive_button.layout.display = 'none'
    
    # Tampilkan tombol setup direktori
    if directory_button:
        directory_button.layout.display = 'block'
    
    # Tampilkan status
    with status:
        status.clear_output()
        if env_manager.is_drive_mounted:
            if logger: logger.info("✅ Google Drive sudah terhubung")
        else:
            if logger: logger.info("ℹ️ Google Drive belum terhubung")
    
    return ui_components
