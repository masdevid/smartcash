"""
File: smartcash/ui/handlers/environment_handler.py
Deskripsi: Handler untuk deteksi dan konfigurasi environment
"""

import os
import sys
from pathlib import Path
import shutil
from typing import Dict, Any, Optional, List, Union
from IPython.display import display, HTML

def detect_environment(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """
    Deteksi dan konfigurasi environment UI.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        
    Returns:
        Dictionary UI components yang telah diperbarui
    """
    # Log status awal
    logger = ui_components.get('logger')
    if logger:
        logger.info("🔍 Mendeteksi environment...")
    
    # Tampilkan informasi di status panel
    if 'status' in ui_components:
        with ui_components['status']:
            from IPython.display import display, HTML
            display(HTML("""
                <div style="padding: 8px; background-color: #d1ecf1; color: #0c5460; border-radius: 4px;">
                    🔄 Mendeteksi environment dan konfigurasi...
                </div>
            """))
            
    # Colab detection message
    try:
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
                if logger:
                    logger.warning("⚠️ EnvironmentManager tidak tersedia, menggunakan deteksi manual")
        
        # Update colab panel
        if 'colab_panel' in ui_components:
            logger = ui_components.get('logger')
            if is_colab:
                # Tampilkan informasi Colab environment
                style = "padding: 10px; background-color: #d1ecf1; color: #0c5460; border-radius: 4px; margin: 10px 0;"
                status = "terhubung" if is_drive_mounted else "tidak terhubung"
                icon = "✅" if is_drive_mounted else "⚠️"
                
                ui_components['colab_panel'].value = f"""
                <div style="{style}">
                    <h3>🔍 Environment: Google Colab</h3>
                    <p>{icon} Status Google Drive: <strong>{status}</strong></p>
                    <p>Klik tombol 'Hubungkan Google Drive' untuk mount drive dan menyinkronkan proyek.</p>
                </div>
                """
                
                # Log jika tersedia
                if logger:
                    logger.info("🔍 Environment terdeteksi: Google Colab")
                    
                # Aktifkan tombol drive
                ui_components['drive_button'].layout.display = 'block'
            else:
                # Tampilkan informasi local environment
                ui_components['colab_panel'].value = """
                <div style="padding: 10px; background-color: #d4edda; color: #155724; border-radius: 4px; margin: 10px 0;">
                    <h3>🔍 Environment: Local</h3>
                    <p>Gunakan tombol 'Setup Direktori Lokal' untuk membuat struktur direktori proyek.</p>
                </div>
                """
                
                # Log jika tersedia
                if logger:
                    logger.info("🔍 Environment terdeteksi: Local")
                    
                # Sembunyikan tombol drive
                ui_components['drive_button'].layout.display = 'none'
    except Exception as e:
        # Fallback: Tangani error
        from smartcash.ui.setup.env_config_fallbacks import handle_environment_detection_error
        ui_components = handle_environment_detection_error(ui_components, e)
    
    return ui_components

def filter_drive_tree(drive_tree_html: str) -> str:
    """
    Filter tree direktori Drive untuk hanya menampilkan yang relevan.
    
    Args:
        drive_tree_html: HTML tree direktori drive
        
    Returns:
        HTML tree yang sudah difilter
    """
    # Filter hanya direktori SmartCash dan subdirektori yang relevan
    if "SmartCash" not in drive_tree_html:
        return """
        <div style="padding: 10px; background-color: #fff3cd; color: #856404; border-radius: 4px;">
            ⚠️ Direktori SmartCash tidak ditemukan di Google Drive.
            <p>Pastikan direktori SmartCash sudah dibuat di Google Drive.</p>
        </div>
        """
    
    # Tambahkan header dan styling
    filtered_html = """
    <div style="padding: 10px; background-color: #f0f8ff; color: #0c5460; border-radius: 4px; margin: 10px 0;">
        <h3>📂 Struktur Direktori Project (Google Drive)</h3>
    """ + drive_tree_html + "</div>"
    
    return filtered_html

def fallback_get_directory_tree(directory: str, max_depth: int = 3) -> str:
    """
    Implementasi fallback untuk mendapatkan tree direktori.
    
    Args:
        directory: Path direktori
        max_depth: Kedalaman maksimum tree
        
    Returns:
        HTML representasi tree direktori
    """
    # Simple function to get directory tree when env_manager is not available
    dir_path = Path(directory)
    if not dir_path.exists():
        return f"<p style='color:red'>❌ Direktori tidak ditemukan: {directory}</p>"
    
    html = "<pre style='margin:0; padding:5px; background:#f8f9fa; font-family:monospace; color:#333;'>\n"
    html += f"<span style='color:#0366d6; font-weight:bold;'>{dir_path.name}/</span>\n"
    
    def _get_tree(path, prefix="", depth=0):
        if depth >= max_depth:
            return ""
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return f"{prefix}<span style='color:red'>❌ Akses ditolak</span>\n"
            
        result = ""
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└─ " if is_last else "├─ "
            
            if item.is_dir():
                result += f"{prefix}{connector}<span style='color:#0366d6; font-weight:bold;'>{item.name}/</span>\n"
                next_prefix = prefix + ("   " if is_last else "│  ")
                if depth < max_depth - 1:
                    result += _get_tree(item, next_prefix, depth + 1)
            else:
                result += f"{prefix}{connector}{item.name}\n"
        
        return result
    
    html += _get_tree(dir_path)
    html += "</pre>"
    
    return html

def check_smartcash_dir(ui_components: Dict[str, Any]) -> bool:
    """
    Cek apakah direktori SmartCash tersedia.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean status ketersediaan direktori
    """
    try:
        # Cek direktori SmartCash
        if os.path.exists("smartcash"):
            return True
            
        # Jika tidak ada, tampilkan pesan error
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML("""
                <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px;">
                    ❌ Direktori SmartCash tidak ditemukan.
                    <p>Pastikan repository SmartCash sudah di-clone dengan benar.</p>
                    <p>Gunakan cell pertama (Repository Clone) untuk mengunduh repository.</p>
                </div>
                """))
        return False
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].error(f"❌ Error cek direktori: {str(e)}")
        return False

def sync_configs(
    local_dirs: List[Path], 
    drive_dirs: List[Path],
    logger = None
) -> Dict[str, Any]:
    """
    Sinkronisasi konfigurasi antara lokal dan Google Drive.
    
    Args:
        local_dirs: List direktori lokal
        drive_dirs: List direktori Google Drive
        logger: Logger (opsional)
        
    Returns:
        Dictionary hasil sinkronisasi
    """
    if not drive_dirs or not local_dirs:
        return {'status': 'error', 'message': 'Direktori tidak valid'}
    
    result = {
        'synced': [],
        'errors': []
    }
    
    # Iterasi semua direktori
    for local_dir, drive_dir in zip(local_dirs, drive_dirs):
        if not local_dir.exists() or not drive_dir.exists():
            result['errors'].append(f"Direktori tidak ditemukan: {local_dir or drive_dir}")
            continue
        
        # Cek file YAML di lokal
        yaml_files = list(local_dir.glob('*.yaml')) + list(local_dir.glob('*.yml'))
        
        for yaml_file in yaml_files:
            try:
                # Path file di drive
                drive_file = drive_dir / yaml_file.name
                
                # Buat direktori di Drive jika belum ada
                drive_dir.mkdir(parents=True, exist_ok=True)
                
                # Cek apakah kedua file ada
                local_exists = yaml_file.exists()
                drive_exists = drive_file.exists()
                
                if local_exists and not drive_exists:
                    # Salin ke Drive
                    shutil.copy2(yaml_file, drive_file)
                    result['synced'].append(f"🔄 {yaml_file.name}: Lokal → Drive")
                    if logger:
                        logger.info(f"🔄 Konfigurasi {yaml_file.name}: Lokal → Drive")
                        
                elif not local_exists and drive_exists:
                    # Salin ke lokal
                    shutil.copy2(drive_file, yaml_file)
                    result['synced'].append(f"🔄 {yaml_file.name}: Drive → Lokal")
                    if logger:
                        logger.info(f"🔄 Konfigurasi {yaml_file.name}: Drive → Lokal")
                        
                elif local_exists and drive_exists:
                    # Jika kedua file ada, bandingkan timestamp
                    local_time = yaml_file.stat().st_mtime
                    drive_time = drive_file.stat().st_mtime
                    
                    if local_time > drive_time:
                        # Lokal lebih baru
                        shutil.copy2(yaml_file, drive_file)
                        result['synced'].append(f"🔄 {yaml_file.name}: Lokal → Drive (file lokal lebih baru)")
                        if logger:
                            logger.info(f"🔄 Konfigurasi {yaml_file.name}: Lokal → Drive (file lokal lebih baru)")
                    else:
                        # Drive lebih baru
                        shutil.copy2(drive_file, yaml_file)
                        result['synced'].append(f"🔄 {yaml_file.name}: Drive → Lokal (file drive lebih baru)")
                        if logger:
                            logger.info(f"🔄 Konfigurasi {yaml_file.name}: Drive → Lokal (file drive lebih baru)")
                            
            except Exception as e:
                result['errors'].append(f"❌ Error sinkronisasi {yaml_file.name}: {str(e)}")
                if logger:
                    logger.warning(f"❌ Error sinkronisasi {yaml_file.name}: {str(e)}")
    
    return result