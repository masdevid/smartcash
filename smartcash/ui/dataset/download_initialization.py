"""
File: smartcash/ui/dataset/download_initialization.py
Deskripsi: Perbaikan pada inisialisasi komponen download untuk sembunyikan field API key jika tersedia dari Google Secret
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
        
        # Cek ketersediaan API key dari Google Secret
        secret_api_key = None
        try:
            from google.colab import userdata
            secret_api_key = userdata.get('ROBOFLOW_API_KEY')
            
            if secret_api_key and 'roboflow_settings' in ui_components and hasattr(ui_components['roboflow_settings'], 'children'):
                # Ada API key, sembunyikan field API key
                api_settings = ui_components['roboflow_settings'].children
                if len(api_settings) > 0:
                    # Sembunyikan field API key
                    api_settings[0].layout.display = 'none'
                    
                    # Tampilkan informasi menggunakan API key dari secret
                    update_status_panel(
                        ui_components,
                        "info",
                        f"{ICONS['info']} Menggunakan API key dari Google Secret"
                    )
                    
                    if 'logger' in ui_components:
                        ui_components['logger'].info(f"{ICONS['info']} Menggunakan API key dari Google Secret")
                
                # Keluarkan juga informasi workspace dan project
                if len(api_settings) > 2:
                    workspace = api_settings[1].value
                    project = api_settings[2].value
                    
                    update_status_panel(
                        ui_components,
                        "info",
                        f"{ICONS['info']} Menggunakan API key dari Google Secret | Project: {project} di workspace: {workspace}"
                    )
        except ImportError:
            # Bukan di Colab, tidak perlu menyembunyikan field
            pass
        
        # Cek ketersediaan API key untuk Roboflow
        if not secret_api_key and 'roboflow_settings' in ui_components and hasattr(ui_components['roboflow_settings'], 'children'):
            api_settings = ui_components['roboflow_settings'].children
            if len(api_settings) > 1:  # Pastikan ada minimal 2 elemen
                workspace = api_settings[1].value
                # Hindari index out of range dengan menggunakan default
                project = api_settings[2].value if len(api_settings) > 2 else "rupiah-emisi-2022"
                
                update_status_panel(ui_components, "info", f"Konfig Roboflow: {workspace}/{project}")
        
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
        secret_key = None
        try:
            from google.colab import userdata
            secret_key = userdata.get('ROBOFLOW_API_KEY')
            
            if secret_key:
                return widgets.HTML(value=create_status_message(
                    "API Key tersedia dari Google Secret." if secret_key else "API Key diperlukan untuk download.", 
                    'success' if secret_key else 'warning'
                ))
        except ImportError:
            # Bukan di Colab
            pass
        
        # Cek API key dari komponen jika tidak ada secret
        if not secret_key and 'roboflow_settings' in ui_components and hasattr(ui_components['roboflow_settings'], 'children'):
            api_settings = ui_components['roboflow_settings'].children
            
            # Pastikan ada minimal 1 elemen sebelum mengakses
            api_key = api_settings[0].value if len(api_settings) > 0 else ""
            
            if api_key:
                return widgets.HTML(value=create_status_message("API Key Roboflow tersedia.", 'success'))
            else:
                return widgets.HTML(value=create_status_message("API Key diperlukan untuk download.", 'warning'))
        
        return widgets.HTML(value=create_status_message("Tidak dapat memeriksa API Key.", 'warning'))
    
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
        from smartcash.ui.utils.alerts import create_info_alert
        ui_components['status_panel'].value = create_info_alert(message, status_type).value
    except ImportError:
        # Fallback jika create_info_alert tidak tersedia
        ui_components['status_panel'].value = create_status_message(message, status_type)