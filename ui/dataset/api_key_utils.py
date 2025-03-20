"""
File: smartcash/ui/dataset/api_key_utils.py
Deskripsi: Utilitas pengecekan API key untuk Roboflow dengan integrasi UI utils
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def get_api_key_info(ui_components: Dict[str, Any]) -> widgets.HTML:
    """
    Dapatkan informasi API key dengan utils standar.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        HTML widget dengan informasi API key
    """
    # Import komponen UI standar
    from smartcash.ui.utils.alert_utils import create_status_indicator
    from smartcash.ui.utils.constants import ICONS
    
    try:
        # Cek API key dari Google Secret terlebih dahulu
        secret_key = None
        api_key_available = False
        
        try:
            from google.colab import userdata
            secret_key = userdata.get('ROBOFLOW_API_KEY')
            api_key_available = bool(secret_key)
            
            if secret_key:
                return widgets.HTML(value=create_status_indicator(
                    "success", f"{ICONS['success']} API Key tersedia dari Google Secret."
                ).value)
        except ImportError:
            # Bukan di Colab
            pass
        
        # Cek API key dari komponen UI jika tidak ada secret
        if not secret_key and 'roboflow_settings' in ui_components and hasattr(ui_components['roboflow_settings'], 'children'):
            api_settings = ui_components['roboflow_settings'].children
            
            # Pastikan ada minimal 1 elemen sebelum mengakses dengan validasi
            if len(api_settings) > 0:
                api_key = api_settings[0].value
                
                if api_key:
                    return widgets.HTML(value=create_status_indicator(
                        "success", f"{ICONS['success']} API Key Roboflow tersedia."
                    ).value)
                else:
                    return widgets.HTML(value=create_status_indicator(
                        "warning", f"{ICONS['warning']} API Key diperlukan untuk download."
                    ).value)
        
        return widgets.HTML(value=create_status_indicator(
            "warning", f"{ICONS['warning']} Tidak dapat memeriksa API Key."
        ).value)
    
    except Exception as e:
        # Handle errors dengan utils standar
        return widgets.HTML(value=create_status_indicator(
            "error", f"{ICONS['error']} Error mendapatkan API Key: {str(e)}"
        ).value)

def check_api_key_availability(ui_components: Dict[str, Any]) -> bool:
    """
    Cek ketersediaan API key dari Google Secret atau UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean menunjukkan ketersediaan API key
    """
    try:
        # Cek dari Google Secret
        try:
            from google.colab import userdata
            secret_key = userdata.get('ROBOFLOW_API_KEY')
            if secret_key:
                return True
        except ImportError:
            pass
            
        # Cek dari UI components
        if 'roboflow_settings' in ui_components and hasattr(ui_components['roboflow_settings'], 'children'):
            api_settings = ui_components['roboflow_settings'].children
            if len(api_settings) > 0 and api_settings[0].value:
                return True
        
        return False
    except Exception:
        return False

def get_api_key(ui_components: Dict[str, Any]) -> Optional[str]:
    """
    Dapatkan API key dari Google Secret atau UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        API key jika tersedia, None jika tidak tersedia
    """
    try:
        # Cek dari Google Secret
        try:
            from google.colab import userdata
            secret_key = userdata.get('ROBOFLOW_API_KEY')
            if secret_key:
                return secret_key
        except ImportError:
            pass
            
        # Cek dari UI components
        if 'roboflow_settings' in ui_components and hasattr(ui_components['roboflow_settings'], 'children'):
            api_settings = ui_components['roboflow_settings'].children
            if len(api_settings) > 0:
                return api_settings[0].value
        
        return None
    except Exception:
        return None

def setup_api_key_field(ui_components: Dict[str, Any]) -> None:
    """
    Setup field API key berdasarkan ketersediaan di Google Secret.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Cek dari Google Secret
        secret_key_available = False
        try:
            from google.colab import userdata
            secret_key = userdata.get('ROBOFLOW_API_KEY')
            secret_key_available = bool(secret_key)
        except ImportError:
            pass
            
        # Update UI components jika API key tersedia dari Google Secret
        if secret_key_available and 'roboflow_settings' in ui_components and hasattr(ui_components['roboflow_settings'], 'children'):
            api_settings = ui_components['roboflow_settings'].children
            if len(api_settings) > 0:
                # Sembunyikan field API key dan tampilkan info
                api_settings[0].layout.display = 'none'
                
                # Update status panel
                from smartcash.ui.utils.fallback_utils import update_status_panel
                from smartcash.ui.utils.constants import ICONS
                update_status_panel(
                    ui_components, "info",
                    f"{ICONS['info']} Menggunakan API key dari Google Secret"
                )
                
                # Log info jika logger tersedia
                if 'logger' in ui_components and ui_components['logger']:
                    ui_components['logger'].info(f"{ICONS['info']} Menggunakan API key dari Google Secret")
    except Exception:
        pass