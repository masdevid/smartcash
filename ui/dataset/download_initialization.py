"""
File: smartcash/ui/dataset/download_initialization.py
Deskripsi: Inisialisasi komponen download dengan integrasi utils terintegrasi dan fallback sederhana
"""

from typing import Dict, Any
import ipywidgets as widgets

def setup_initialization(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Inisialisasi komponen download dataset dengan UI utils yang terintegrasi."""
    try:
        # Import utils dengan pendekatan fallback yang terstandarisasi
        from smartcash.ui.utils.fallback_utils import import_with_fallback, update_status_panel
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        from smartcash.ui.utils.constants import ICONS
        
        # Dapatkan logger dari UI components dengan fallback
        logger = ui_components.get('logger')
        
        # Dapatkan direktori data dengan pendekatan terstandarisasi
        data_dir = config.get('data', {}).get('dir', 'data') if config else 'data'
        
        # Gunakan utils untuk mendeteksi Google Drive
        drive_mounted, drive_path = detect_drive_mount()
        
        # Gunakan drive path jika tersedia
        if drive_mounted and drive_path:
            data_dir = f"{drive_path}/data"
            if logger: logger.info(f"{ICONS['folder']} Menggunakan Google Drive untuk penyimpanan dataset: {data_dir}")
            update_status_panel(ui_components, "info", f"Dataset akan didownload ke: {data_dir}")
        
        # Cek ketersediaan API key dari Google Secret dengan utils standar
        api_key_available = False
        try:
            # Gunakan utils untuk memeriksa Google Secret
            from google.colab import userdata
            secret_api_key = userdata.get('ROBOFLOW_API_KEY')
            api_key_available = bool(secret_api_key)
            
            if api_key_available and 'roboflow_settings' in ui_components and hasattr(ui_components['roboflow_settings'], 'children'):
                # Ada API key, sembunyikan field API key
                api_settings = ui_components['roboflow_settings'].children
                if len(api_settings) > 0:
                    api_settings[0].layout.display = 'none'
                    update_status_panel(ui_components, "info", f"{ICONS['info']} Menggunakan API key dari Google Secret")
                    if logger: logger.info(f"{ICONS['info']} Menggunakan API key dari Google Secret")
                
                # Extract workspace/project info untuk ditampilkan
                if len(api_settings) > 2:
                    workspace = api_settings[1].value
                    project = api_settings[2].value
                    update_status_panel(ui_components, "info", 
                        f"{ICONS['info']} Menggunakan API key dari Google Secret | Project: {project} di workspace: {workspace}")
        except ImportError:
            # Bukan di Colab atau userdata tidak tersedia
            pass
        
        # Cek Roboflow info untuk tampilan status
        if not api_key_available and 'roboflow_settings' in ui_components and hasattr(ui_components['roboflow_settings'], 'children'):
            api_settings = ui_components['roboflow_settings'].children
            if len(api_settings) > 1:
                workspace = api_settings[1].value if len(api_settings) > 1 else "smartcash-wo2us"
                project = api_settings[2].value if len(api_settings) > 2 else "rupiah-emisi-2022"
                update_status_panel(ui_components, "info", f"Konfig Roboflow: {workspace}/{project}")
        
        # Tambahkan variabel penting ke ui_components
        ui_components['data_dir'] = data_dir
        ui_components['api_key_available'] = api_key_available
        
    except Exception as e:
        # Tangani error dengan utils standar
        from smartcash.ui.utils.alert_utils import create_status_indicator
        ui_components['status_panel'].value = create_status_indicator("error", f"Error inisialisasi: {str(e)}").value
        if logger: logger.error(f"{ICONS['error']} Error inisialisasi download: {str(e)}")
    
    return ui_components

def get_api_key_info(ui_components):
    """Dapatkan informasi API key dengan utils standar."""
    from smartcash.ui.utils.alert_utils import create_status_indicator
    
    try:
        # Cek API key dari Google Secret terlebih dahulu
        try:
            from google.colab import userdata
            secret_key = userdata.get('ROBOFLOW_API_KEY')
            
            if secret_key:
                return widgets.HTML(value=create_status_indicator(
                    "success", "API Key tersedia dari Google Secret."
                ).value)
        except ImportError:
            pass
        
        # Cek API key dari komponen UI
        if 'roboflow_settings' in ui_components and hasattr(ui_components['roboflow_settings'], 'children'):
            api_settings = ui_components['roboflow_settings'].children
            api_key = api_settings[0].value if len(api_settings) > 0 else ""
            
            if api_key:
                return widgets.HTML(value=create_status_indicator("success", "API Key Roboflow tersedia.").value)
            else:
                return widgets.HTML(value=create_status_indicator("warning", "API Key diperlukan untuk download.").value)
        
        return widgets.HTML(value=create_status_indicator("warning", "Tidak dapat memeriksa API Key.").value)
    
    except Exception as e:
        return widgets.HTML(value=create_status_indicator("error", f"Error mendapatkan API Key: {str(e)}").value)

def update_status_panel(ui_components, status_type, message):
    """Update status panel dengan utils standar."""
    from smartcash.ui.utils.fallback_utils import update_status_panel as update_panel
    update_panel(ui_components, status_type, message)