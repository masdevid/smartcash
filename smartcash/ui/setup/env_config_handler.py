"""
File: smartcash/ui/setup/env_config_handler.py
Deskripsi: Handler untuk komponen UI konfigurasi environment yang terintegrasi dengan tema dan komponen UI yang ada
"""

import os
import sys
from typing import Dict, Any, Optional

def setup_env_config_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI environment config.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    from smartcash.ui.setup.env_detection import detect_environment
    from smartcash.ui.setup.drive_handler import handle_drive_connection
    from smartcash.ui.setup.directory_handler import handle_directory_setup
    
    # Deteksi environment jika belum ada
    ui_components = detect_environment(ui_components, env)
    
    # Setup logger untuk event handling jika tersedia
    logger = ui_components.get('logger', None)
    
    # Handler untuk tombol Drive
    def on_drive_button_clicked(b):
        if logger:
            logger.info("🔗 Tombol hubungkan Google Drive diklik")
        handle_drive_connection(ui_components)
    
    # Handler untuk tombol Directory
    def on_directory_button_clicked(b):
        if logger:
            logger.info("📁 Tombol setup direktori lokal diklik")
        handle_directory_setup(ui_components)
    
    # Daftarkan handler
    ui_components['drive_button'].on_click(on_drive_button_clicked)
    ui_components['directory_button'].on_click(on_directory_button_clicked)
    
    return ui_components

def check_smartcash_dir(root_dir: Optional[str] = None) -> bool:
    """
    Cek apakah direktori mengandung struktur SmartCash.
    
    Args:
        root_dir: Direktori root untuk dicek
        
    Returns:
        Boolean menunjukkan apakah direktori SmartCash
    """
    from pathlib import Path
    
    if root_dir is None:
        root_dir = os.getcwd()
    
    path = Path(root_dir)
    
    # Check for key folders and files that indicate a SmartCash project
    indicators = [
        path / "smartcash",
        path / "setup.py",
        path / "requirements.txt",
        path / "configs"
    ]
    
    # If at least 2 indicators exist, consider it a SmartCash directory
    existence_count = sum(1 for item in indicators if item.exists())
    return existence_count >= 2

def sync_configs(config_manager, sync_dir: str = 'configs', logger=None):
    """
    Sinkronisasi semua file konfigurasi.
    
    Args:
        config_manager: Manager konfigurasi
        sync_dir: Direktori konfigurasi
        logger: Logger untuk logging
        
    Returns:
        Hasil sinkronisasi
    """
    try:
        # Gunakan function sync_all_configs dari ConfigManager jika ada
        if hasattr(config_manager, 'sync_all_configs'):
            return config_manager.sync_all_configs('newest')
        else:
            # Implementasi fallback sederhana
            import os
            from pathlib import Path
            
            # Pastikan direktori configs ada
            Path(sync_dir).mkdir(parents=True, exist_ok=True)
            
            # Hitung jumlah file konfigurasi
            config_files = list(Path(sync_dir).glob('*.yaml')) + list(Path(sync_dir).glob('*.yml'))
            
            result = {
                "synced": [],
                "failed": [],
                "message": f"Ditemukan {len(config_files)} file konfigurasi"
            }
            
            if logger:
                logger.info(f"🔍 Ditemukan {len(config_files)} file konfigurasi")
                
            return result
            
    except Exception as e:
        error_msg = f"❌ Error saat sinkronisasi konfigurasi: {str(e)}"
        if logger:
            logger.error(error_msg)
        return {"error": error_msg}