"""
File: smartcash/ui/dataset/download_initialization.py
Deskripsi: Inisialisasi komponen download dataset dengan penyembunyian API key otomatis
"""

from typing import Dict, Any
from IPython.display import HTML
import ipywidgets as widgets

try:
    from smartcash.ui.utils.constants import ICONS, COLORS, ALERT_STYLES
except ImportError:
    ICONS = {'info': 'ℹ️', 'folder': '📁', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
    COLORS = {'alert_info_bg': '#d1ecf1', 'alert_info_text': '#0c5460'}
    ALERT_STYLES = {'info': {'bg_color': '#d1ecf1', 'text_color': '#0c5460', 'icon': 'ℹ️'}}

def setup_initialization(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Inisialisasi komponen download dataset."""
    try:
        # Dapatkan direktori data
        data_dir = config.get('data', {}).get('dir', 'data') if config else 'data'
        
        # Gunakan Google Drive jika tersedia
        if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted and hasattr(env, 'drive_path'):
            data_dir = str(env.drive_path / 'data')
            
            # Log penggunaan Google Drive
            if 'logger' in ui_components:
                ui_components['logger'].info(f"{ICONS['folder']} Menggunakan Google Drive untuk penyimpanan dataset: {data_dir}")
            
            # Update status panel
            update_status_panel(ui_components, "info", f"Dataset akan didownload ke: {data_dir}")
        
        # Cek ketersediaan API key untuk Roboflow
        if 'roboflow_settings' in ui_components and hasattr(ui_components['roboflow_settings'], 'children'):
            api_settings = ui_components['roboflow_settings'].children
            if len(api_settings) > 1:  # Pastikan ada minimal 2 elemen
                workspace = api_settings[1].value
                # Hindari index out of range dengan menggunakan default
                project = api_settings[2].value if len(api_settings) > 2 else "rupiah-emisi-2022"
                
                update_status_panel(ui_components, "info", f"Konfig Roboflow: {workspace}/{project}")
                
                # Cek Google Secret untuk API key
                try:
                    from google.colab import userdata
                    secret_key = userdata.get('ROBOFLOW_API_KEY')
                    if secret_key and len(api_settings) > 0:
                        # Sembunyikan input API key karena sudah tersedia dari Secret
                        api_settings[0].layout.display = 'none'
                        # Tampilkan notifikasi
                        api_status = widgets.HTML(
                            value=f"""<div style="padding:5px 10px; background-color:{COLORS.get('alert_success_bg', '#d4edda')}; 
                                         color:{COLORS.get('alert_success_text', '#155724')}; border-radius:4px; margin:5px 0">
                                    <p style="margin:0">{ICONS.get('success', '✅')} API Key diambil dari Google Secret</p>
                                    </div>"""
                        )
                        # Tambahkan ke container roboflow settings
                        ui_components['api_key_status'] = api_status
                        # Jika UI dirender, tambahkan notifikasi
                        if 'download_settings_container' in ui_components and ui_components['download_settings_container'].children[0] is ui_components['roboflow_settings']:
                            ui_components['download_settings_container'].children = [
                                ui_components['roboflow_settings'], 
                                api_status
                            ]
                        
                        # Logika untuk menggunakan API key dari secret untuk operasi yang membutuhkan
                        if 'logger' in ui_components:
                            ui_components['logger'].info(f"{ICONS['success']} API Key Roboflow tersedia dari Google Secret")
                except ImportError:
                    pass
        
        # Tambahkan variabel tambahan ke ui_components
        ui_components['data_dir'] = data_dir
        
    except Exception as e:
        # Tangani error dengan fallback
        update_status_panel(ui_components, "error", f"Error inisialisasi: {str(e)}")
    
    return ui_components

def get_api_key_info(ui_components):
    """Dapatkan informasi API key dengan fallback."""
    try:
        # Cek API key dari Google Secret terlebih dahulu
        try:
            from google.colab import userdata
            secret_key = userdata.get('ROBOFLOW_API_KEY')
            
            if secret_key:
                return widgets.HTML(value=create_status_message(
                    "API Key tersedia dari Google Secret", 'success'))
        except ImportError:
            pass
            
        # Cek API key dari komponen
        if 'roboflow_settings' in ui_components and hasattr(ui_components['roboflow_settings'], 'children'):
            api_settings = ui_components['roboflow_settings'].children
            
            # Pastikan ada minimal 1 elemen sebelum mengakses
            api_key = api_settings[0].value if len(api_settings) > 0 else ""
            
            if api_key:
                return widgets.HTML(value=create_status_message("API Key Roboflow tersedia.", 'success'))
        
        # Tidak ada API key
        return widgets.HTML(value=create_status_message("API Key diperlukan untuk download.", 'warning'))
    
    except Exception as e:
        return widgets.HTML(value=create_status_message(f"Error mendapatkan API Key: {str(e)}", 'error'))

def create_status_message(message, status_type='info'):
    """Buat pesan status dalam format HTML."""
    style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
    return f"""
    <div style="padding:8px 12px; background-color:{style.get('bg_color', '#d1ecf1')}; 
              color:{style.get('text_color', '#0c5460')}; border-radius:4px; margin:5px 0;
              border-left:4px solid {style.get('text_color', '#0c5460')};">
        <p style="margin:3px 0">{style.get('icon', 'ℹ️')} {message}</p>
    </div>
    """

def update_status_panel(ui_components, status_type, message):
    """Update status panel dengan pesan dan jenis status."""
    if 'status_panel' not in ui_components:
        return
        
    try:
        from smartcash.ui.utils.ui_helpers import create_info_alert
        ui_components['status_panel'].value = create_info_alert(message, status_type).value
    except ImportError:
        # Fallback jika create_info_alert tidak tersedia
        ui_components['status_panel'].value = create_status_message(message, status_type)