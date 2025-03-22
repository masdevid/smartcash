"""
File: smartcash/ui/setup/drive_sync_initializer.py
Deskripsi: Modul untuk inisialisasi sinkronisasi Google Drive dengan progress tracking terintegrasi
"""

import os, sys
from typing import Dict, Any, Optional

def initialize_drive_sync(ui_components: Dict[str, Any] = None) -> bool:
    """
    Inisialisasi dan sinkronisasi Google Drive dengan progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI yang akan diupdate
        
    Returns:
        Boolean yang menunjukkan keberhasilan operasi
    """
    # Pastikan output tersedia
    if not ui_components or 'status' not in ui_components:
        print("Error: Output widget tidak tersedia")
        return False
    
    # Gunakan logger jika tersedia
    logger = ui_components.get('logger')
    
    # Fungsi log yang mengarahkan ke logger UI
    def log(message, status_type="info"):
        # Log menggunakan logger jika tersedia
        if logger:
            log_func = getattr(logger, status_type, logger.info)
            log_func(message)
        else:
            # Gunakan fungsi fallback jika logger tidak tersedia
            from smartcash.ui.utils.logging_utils import log_to_ui
            log_to_ui(ui_components, message, status_type)
    
    # Progress tracking objects
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    try:
        # 1. Deteksi environment Colab
        log("Mendeteksi environment Google Drive...", "info")
        is_colab = 'google.colab' in sys.modules
        drive_mounted = os.path.exists('/content/drive/MyDrive')
        
        if not is_colab:
            log("Bukan lingkungan Google Colab, lewati sinkronisasi Drive", "info")
            _update_colab_panel(ui_components, is_colab=False)
            return False
        
        # 2. Pastikan konfigurasi default tersedia
        log("Memastikan file konfigurasi default tersedia...", "info")
        _update_progress(progress_bar, progress_message, 1, 4, "Persiapan konfigurasi...")
        
        configs_created = _ensure_config_defaults()
        if configs_created is not None:
            log("Konfigurasi default " + ("berhasil dibuat" if configs_created else "sudah tersedia"), 
                "success" if configs_created else "info")
        
        # 3. Mount Drive jika diperlukan
        if not drive_mounted:
            log("Mounting Google Drive...", "info")
            _update_progress(progress_bar, progress_message, 2, 4, "Mounting Google Drive...")
            
            drive_mounted = _mount_google_drive()
            if not drive_mounted:
                log("Gagal mounting Google Drive", "error")
                return False
            
            log("Google Drive berhasil dimount", "success")
        else:
            log("Google Drive sudah terhubung", "success")
        
        # 4. Sinkronisasi konfigurasi
        log("Sinkronisasi konfigurasi dengan Drive...", "info")
        _update_progress(progress_bar, progress_message, 3, 4, "Sinkronisasi konfigurasi...")
        
        sync_results = _sync_configs_with_drive()
        if sync_results:
            success_count = len(sync_results.get("success", []))
            failure_count = len(sync_results.get("failure", []))
            
            if failure_count == 0:
                log(f"Sinkronisasi berhasil: {success_count} file ✓", "success")
            else:
                log(f"Sinkronisasi: {success_count} berhasil, {failure_count} gagal", "warning")
        
        # 5. Finalisasi
        _update_progress(progress_bar, progress_message, 4, 4, "Inisialisasi selesai")
        _update_colab_panel(ui_components, is_colab=True, drive_mounted=True)
        
        return True
        
    except Exception as e:
        log(f"Error saat inisialisasi Drive: {str(e)}", "error")
        _reset_progress(progress_bar, progress_message)
        return False

def _ensure_config_defaults() -> Optional[bool]:
    """Pastikan konfigurasi default tersedia."""
    try:
        from smartcash.common.default_config import ensure_all_configs_exist
        return ensure_all_configs_exist()
    except Exception:
        return None

def _mount_google_drive() -> bool:
    """Mount Google Drive dan kembalikan status keberhasilan."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return os.path.exists('/content/drive/MyDrive')
    except Exception:
        return False

def _sync_configs_with_drive() -> Optional[Dict[str, Any]]:
    """Sinkronisasi konfigurasi dengan Drive dan kembalikan hasilnya."""
    try:
        from smartcash.common.config_sync import sync_all_configs
        return sync_all_configs(sync_strategy='drive_priority')
    except ImportError:
        return None

def _update_progress(progress_bar, progress_message, value, total, message):
    """Update progress bar dan message."""
    if progress_bar:
        progress_bar.value = value
        progress_bar.max = total
    if progress_message:
        progress_message.value = message

def _reset_progress(progress_bar, progress_message):
    """Reset progress bar dan message."""
    if progress_bar:
        progress_bar.value = 0
    if progress_message:
        progress_message.value = "Error inisialisasi"

def _update_colab_panel(ui_components, is_colab=False, drive_mounted=False):
    """Update panel Colab dengan status terkini."""
    if 'colab_panel' not in ui_components:
        return
        
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    if is_colab:
        if drive_mounted:
            ui_components['colab_panel'].value = f"""
            <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                      color:{COLORS['alert_success_text']}; margin:10px 0; border-radius:4px; 
                      border-left:4px solid {COLORS['alert_success_text']};">
                <h3 style="color:inherit; margin:5px 0">🔍 Environment: Google Colab</h3>
                <p style="margin:5px 0">✅ Status Google Drive: <strong>terhubung</strong></p>
                <p style="margin:5px 0">Drive terhubung dan konfigurasi telah disinkronisasi</p>
            </div>
            """
        else:
            ui_components['colab_panel'].value = f"""
            <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                      color:{COLORS['alert_warning_text']}; margin:10px 0; border-radius:4px; 
                      border-left:4px solid {COLORS['alert_warning_text']};">
                <h3 style="color:inherit; margin:5px 0">🔍 Environment: Google Colab</h3>
                <p style="margin:5px 0">⚠️ Status Google Drive: <strong>tidak terhubung</strong></p>
                <p style="margin:5px 0">Klik tombol 'Hubungkan Google Drive' untuk mount drive.</p>
            </div>
            """
    else:
        ui_components['colab_panel'].value = f"""
        <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                  color:{COLORS['alert_success_text']}; margin:10px 0; border-radius:4px; 
                  border-left:4px solid {COLORS['alert_success_text']};">
            <h3 style="color:inherit; margin:5px 0">🔍 Environment: Local</h3>
            <p style="margin:5px 0">✅ Status: <strong>Siap digunakan</strong></p>
            <p style="margin:5px 0">Gunakan tombol 'Setup Direktori Lokal' untuk membuat struktur direktori proyek.</p>
        </div>
        """