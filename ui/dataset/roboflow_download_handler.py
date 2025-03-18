"""
File: smartcash/ui/dataset/roboflow_download_handler.py
Deskripsi: Handler untuk download dataset dari Roboflow yang diperbaiki untuk mencegah index out of range
"""

from pathlib import Path
from typing import Dict, Any
from IPython.display import display
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.components.alerts import create_status_indicator

def get_roboflow_settings(ui_components):
    """Ambil pengaturan download Roboflow dengan verifikasi ketersediaan komponen."""
    # Periksa keberadaan komponen terlebih dahulu
    if 'roboflow_settings' not in ui_components:
        with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} Konfigurasi Roboflow tidak ditemukan"))
        return None
    
    roboflow_settings = ui_components['roboflow_settings']
    if not hasattr(roboflow_settings, 'children'):
        with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} Konfigurasi Roboflow tidak lengkap"))
        return None
    
    # Get settings dengan aman
    api_settings = roboflow_settings.children
    children_count = len(api_settings)
    
    # Default values
    api_key = ""
    workspace = "smartcash-wo2us"
    project = "rupiah-emisi-2022"
    version = "3"
    
    # Akses komponen dengan aman
    if children_count > 0: api_key = api_settings[0].value
    if children_count > 1: workspace = api_settings[1].value
    if children_count > 2: project = api_settings[2].value
    if children_count > 3: version = api_settings[3].value
    
    # Try to get API key from Google Secret if not provided
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get('ROBOFLOW_API_KEY')
            if api_key and children_count > 0:
                api_settings[0].value = api_key
            if not api_key:
                with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} API Key Roboflow tidak tersedia"))
                return None
        except Exception:
            with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} API Key Roboflow tidak tersedia"))
            return None
    
    return {
        'api_key': api_key,
        'workspace': workspace,
        'project': project,
        'version': version
    }