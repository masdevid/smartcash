"""
File: smartcash/ui/dataset/download_initialization.py
Deskripsi: Inisialisasi komponen download dataset dengan error handling terpadu
"""

from typing import Dict, Any, Optional
from IPython.display import display, HTML
import ipywidgets as widgets

try:
    from smartcash.ui.utils.fallback import create_status_message, handle_download_status
    from smartcash.ui.utils.constants import ICONS, COLORS
except ImportError:
    # Fallback imports
    def create_status_message(message, status_type='info', **kwargs):
        return f"📢 {message}"
    
    def handle_download_status(ui_components, message, status_type='info', **kwargs):
        print(f"{status_type.upper()}: {message}")

def setup_initialization(
    ui_components: Dict[str, Any], 
    env=None, 
    config=None
) -> Dict[str, Any]:
    """
    Inisialisasi komponen download dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Dapatkan direktori data
        data_dir = config.get('data', {}).get('dir', 'data')
        
        # Gunakan Google Drive jika tersedia
        if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted:
            data_dir = str(env.drive_path / 'data')
            
            # Log penggunaan Google Drive
            if 'logger' in ui_components:
                ui_components['logger'].info(
                    f"{ICONS['folder']} Menggunakan Google Drive untuk penyimpanan dataset: {data_dir}"
                )
            
            # Update status panel dengan fallback
            handle_download_status(
                ui_components, 
                f"Dataset akan didownload dari sumber: {data_dir}", 
                'info'
            )
        
        # Cek ketersediaan API key untuk Roboflow
        if 'roboflow_settings' in ui_components:
            api_settings = ui_components['roboflow_settings'].children
            workspace = api_settings[1].value
            project = api_settings[2].value
            
            handle_download_status(
                ui_components, 
                f"Konfig Roboflow: {workspace}/{project}", 
                'info'
            )
        
        # Tambahkan variabel tambahan ke ui_components
        ui_components['data_dir'] = data_dir
        
    except Exception as e:
        # Tangani error dengan fallback
        handle_download_status(
            ui_components, 
            f"Error inisialisasi: {str(e)}", 
            'error'
        )
    
    return ui_components

def get_api_key_info(ui_components):
    """
    Dapatkan informasi API key dengan fallback.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        widgets.HTML dengan informasi API key
    """
    try:
        # Cek API key dari komponen
        api_settings = ui_components['roboflow_settings'].children
        api_key = api_settings[0].value
        
        if api_key:
            return create_status_message(
                "API Key Roboflow tersedia.", 
                'success', 
                as_widget=True
            )
        
        # Coba dapatkan dari Google Secret
        try:
            from google.colab import userdata
            secret_key = userdata.get('ROBOFLOW_API_KEY')
            
            return create_status_message(
                "API Key tersedia dari Google Secret.", 
                'info', 
                as_widget=True
            ) if secret_key else create_status_message(
                "API Key diperlukan untuk download.", 
                'warning', 
                as_widget=True
            )
        except ImportError:
            return create_status_message(
                "Tidak dapat memeriksa API Key.", 
                'error', 
                as_widget=True
            )
    
    except Exception as e:
        return create_status_message(
            f"Error mendapatkan API Key: {str(e)}", 
            'error', 
            as_widget=True
        )