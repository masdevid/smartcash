"""
File: smartcash/ui/setup/colab_initializer.py
Deskripsi: Modul untuk inisialisasi environment dan sinkronisasi konfigurasi Drive
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

# Komponen UI yang akan digunakan di beberapa fungsi
UI_COMPONENTS = {
    'status': None,
    'progress': None,
    'message': None,
    'output': None,
    'container': None
}

def create_alert(msg: str, type: str = 'info') -> HTML:
    """Membuat alert box dengan gaya yang konsisten."""
    styles = {
        'success': ('#d4edda', '#155724', '✅'), 
        'warning': ('#fff3cd', '#856404', '⚠️'), 
        'error': ('#f8d7da', '#721c24', '❌'), 
        'info': ('#d1ecf1', '#0c5460', '📘')
    }
    bg, text, icon = styles.get(type, styles['info'])
    return HTML(
        f"<div style='padding:10px;margin:5px 0;background-color:{bg};color:{text};border-radius:4px'>{icon} {msg}</div>"
    )

def initialize_ui() -> Dict[str, Any]:
    """Inisialisasi komponen UI."""
    # Status UI
    status_html = widgets.HTML("<h3 style='color:#2F58CD'>🚀 Memulai SmartCash dengan sinkronisasi konfigurasi...</h3>")
    progress = widgets.IntProgress(
        value=0, min=0, max=5, 
        description='Progres:',
        style={'description_width': 'initial', 'bar_color': '#3498db'},
        layout=widgets.Layout(width='50%', margin='10px 0')
    )
    message = widgets.HTML("Mendeteksi lingkungan...")
    output = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0',
            min_height='200px',
            max_height='300px',
            overflow='auto'
        )
    )

    # Tampilkan UI
    container = widgets.VBox([status_html, progress, message, output])
    display(container)
    
    # Update komponen UI global
    UI_COMPONENTS['status'] = status_html
    UI_COMPONENTS['progress'] = progress
    UI_COMPONENTS['message'] = message
    UI_COMPONENTS['output'] = output
    UI_COMPONENTS['container'] = container
    
    return UI_COMPONENTS

def update_ui(progress_value: int = None, message_text: str = None, status_text: str = None) -> None:
    """Update komponen UI dengan nilai baru."""
    if progress_value is not None and UI_COMPONENTS['progress']:
        UI_COMPONENTS['progress'].value = progress_value
    
    if message_text is not None and UI_COMPONENTS['message']:
        UI_COMPONENTS['message'].value = message_text
        
    if status_text is not None and UI_COMPONENTS['status']:
        UI_COMPONENTS['status'].value = f"<h3 style='color:#2F58CD'>{status_text}</h3>"

def setup_drive() -> bool:
    """Setup Google Drive di lingkungan Colab."""
    update_ui(2, "Menghubungkan Google Drive...")
    
    # Cek apakah kita di Colab
    is_colab = 'google.colab' in sys.modules
    
    with UI_COMPONENTS['output']:
        display(create_alert(f"Lingkungan: {'Google Colab' if is_colab else 'Local'}", "info"))
        
        # Jika di Colab, coba mount Google Drive jika belum
        if is_colab:
            if not os.path.exists('/content/drive/MyDrive'):
                display(create_alert("Menghubungkan Google Drive...", "info"))
                try:
                    from google.colab import drive
                    drive.mount('/content/drive')
                    time.sleep(1)  # Tunggu sebentar
                    display(create_alert("Google Drive terhubung! ✓", "success"))
                    return True
                except Exception as e:
                    display(create_alert(f"Error saat menghubungkan Google Drive: {str(e)}", "error"))
                    return False
            else:
                display(create_alert("Google Drive sudah terhubung ✓", "success"))
                return True
        else:
            display(create_alert("Bukan lingkungan Colab, lewati mounting Google Drive", "info"))
            return False

def setup_environment() -> bool:
    """Setup environment SmartCash."""
    update_ui(3, "Inisialisasi environment...")
    
    with UI_COMPONENTS['output']:
        try:
            # Pastikan direktori SmartCash ada di Drive jika kita di Colab
            if 'google.colab' in sys.modules and os.path.exists('/content/drive/MyDrive'):
                drive_smartcash_dir = '/content/drive/MyDrive/SmartCash'
                
                if not os.path.exists(drive_smartcash_dir):
                    display(create_alert("Membuat direktori SmartCash di Google Drive...", "info"))
                    os.makedirs(drive_smartcash_dir, exist_ok=True)
                    os.makedirs(f"{drive_smartcash_dir}/configs", exist_ok=True)
                    os.makedirs(f"{drive_smartcash_dir}/data", exist_ok=True)
                    os.makedirs(f"{drive_smartcash_dir}/runs", exist_ok=True)
                    os.makedirs(f"{drive_smartcash_dir}/logs", exist_ok=True)
                    display(create_alert("Direktori SmartCash berhasil dibuat di Google Drive ✓", "success"))
                else:
                    display(create_alert("Direktori SmartCash sudah ada di Google Drive ✓", "success"))
            
            # Coba load smartcash environment manager
            display(create_alert("Inisialisasi environment manager SmartCash...", "info"))
            try:
                from smartcash.common.environment import get_environment_manager
                env_manager = get_environment_manager()
                display(create_alert(f"Environment manager berhasil diinisialisasi ✓", "success"))
                display(create_alert(f"Drive status: {'Terhubung' if env_manager.is_drive_mounted else 'Tidak terhubung'}", "info"))
                return True
            except ImportError:
                display(create_alert("Environment manager belum tersedia, jalankan cell instalasi terlebih dahulu", "warning"))
                return False
        except Exception as e:
            display(create_alert(f"Error saat inisialisasi environment: {str(e)}", "error"))
            return False

def sync_config() -> bool:
    """Sinkronisasi konfigurasi dari Google Drive."""
    update_ui(4, "Sinkronisasi konfigurasi dengan Google Drive...")
    
    with UI_COMPONENTS['output']:
        try:
            # Coba gunakan fungsi inisialisasi
            try:
                from smartcash.common.initialization import initialize_config
                success, config = initialize_config()
                
                if success:
                    display(create_alert("Konfigurasi berhasil disinkronisasi dengan Google Drive ✓", "success"))
                    return True
                else:
                    display(create_alert("Konfigurasi berhasil dimuat tetapi dengan beberapa peringatan ⚠️", "warning"))
                    return True
            except ImportError:
                # Fallback: Coba gunakan fungsi sync jika tersedia
                try:
                    from smartcash.common.config_sync import sync_all_configs
                    results = sync_all_configs(sync_strategy='drive_priority')
                    
                    # Tampilkan hasil
                    success_count = len(results.get("success", []))
                    failure_count = len(results.get("failure", []))
                    
                    if failure_count == 0:
                        display(create_alert(f"Sinkronisasi berhasil: {success_count} file ✓", "success"))
                        return True
                    else:
                        display(create_alert(f"Sinkronisasi selesai dengan peringatan: {success_count} berhasil, {failure_count} gagal ⚠️", "warning"))
                        return True
                except ImportError:
                    display(create_alert("Modul sinkronisasi belum tersedia, silakan jalankan cell instalasi terlebih dahulu", "warning"))
                    return False
        
        except Exception as e:
            display(create_alert(f"Error saat sinkronisasi konfigurasi: {str(e)}", "error"))
            return False

def initialize_environment() -> None:
    """Fungsi utama untuk inisialisasi environment dan sinkronisasi konfigurasi."""
    # Inisialisasi UI
    initialize_ui()
    
    # Step 1: Check dependencies
    update_ui(1, "Memeriksa dependencies...")
    
    with UI_COMPONENTS['output']:
        display(create_alert("Memeriksa modul-modul yang diperlukan...", "info"))
        required_modules = ['yaml', 'tqdm', 'ipywidgets']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
                display(create_alert(f"{module} tersedia ✓", "success"))
            except ImportError:
                missing_modules.append(module)
                display(create_alert(f"{module} tidak tersedia ✗", "warning"))
        
        if missing_modules:
            display(create_alert(f"Menginstall modul yang diperlukan: {', '.join(missing_modules)}", "info"))
            import subprocess
            for module in missing_modules:
                display(create_alert(f"Menginstall {module}...", "info"))
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", module], check=True)
                display(create_alert(f"{module} berhasil diinstall", "success"))
    
    # Step 2: Setup Google Drive
    drive_connected = setup_drive()
    
    # Step 3: Setup environment
    env_setup = setup_environment()
    
    # Step 4: Sync config
    if drive_connected and env_setup:
        config_synced = sync_config()
    else:
        config_synced = False
        with UI_COMPONENTS['output']:
            display(create_alert("Lewati sinkronisasi konfigurasi karena environment tidak siap", "warning"))
    
    # Step 5: Complete
    update_ui(5, "Setup selesai.")
    
    if drive_connected and env_setup and config_synced:
        update_ui(status_text="✅ SmartCash siap digunakan!")
        with UI_COMPONENTS['output']:
            display(create_alert("SmartCash siap digunakan! 🚀", "success"))
    else:
        update_ui(status_text="⚠️ SmartCash siap dengan peringatan")
        with UI_COMPONENTS['output']:
            display(create_alert("SmartCash siap dengan beberapa peringatan. Cek log untuk detail.", "warning"))
    
    with UI_COMPONENTS['output']:
        display(create_alert("Silakan jalankan cell berikutnya untuk melanjutkan.", "info"))

if __name__ == "__main__":
    initialize_environment()