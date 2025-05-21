"""
File: smartcash/ui/dataset/download/handlers/api_key_handler.py
Deskripsi: Handler untuk manajemen API key Roboflow
"""

import os
from typing import Dict, Any, Optional, Tuple
from IPython.display import display, clear_output
from smartcash.ui.dataset.download.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.download.utils.ui_state_manager import highlight_required_fields, update_status_panel

def check_api_key(ui_components: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Periksa API key Roboflow dari berbagai sumber dan update UI jika perlu.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Tuple (has_api_key, api_key)
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Check sources in priority order:
    # 1. UI input field
    # 2. Google Colab secrets
    # 3. Environment variable
    
    # Check UI input first
    if 'api_key' in ui_components and ui_components['api_key'].value:
        api_key = ui_components['api_key'].value
        return True, api_key
    
    # Try to get from Google Colab secrets
    api_key = get_api_key_from_secrets('ROBOFLOW_API_KEY')
    if api_key:
        # Update UI component if available
        if 'api_key' in ui_components:
            ui_components['api_key'].value = api_key
        
        log_message(ui_components, "API key Roboflow berhasil diambil dari Google Colab secrets", "info", "🔑")
        update_status_panel(ui_components, "API key Roboflow berhasil diambil dari Google Colab secrets", "success")
        return True, api_key
    
    # Check environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    if api_key:
        # Update UI component if available
        if 'api_key' in ui_components:
            ui_components['api_key'].value = api_key
            
        log_message(ui_components, "API key Roboflow berhasil diambil dari environment variable", "info", "🔑")
        update_status_panel(ui_components, "API key Roboflow berhasil diambil dari environment variable", "success")
        return True, api_key
    
    # No API key found
    log_message(ui_components, "API key Roboflow tidak ditemukan", "warning", "⚠️")
    update_status_panel(ui_components, "API key Roboflow tidak ditemukan, silakan masukkan secara manual", "warning")
    return False, None

def get_api_key_from_secrets(secret_name: str) -> Optional[str]:
    """
    Ambil API key dari Google Colab secrets.
    
    Args:
        secret_name: Nama secret
        
    Returns:
        API key atau None jika tidak ditemukan
    """
    try:
        # Coba import Google Colab
        from google.colab import userdata
        
        # Coba ambil API key
        api_key = userdata.get(secret_name)
        return api_key if api_key else None
    except (ImportError, AttributeError, Exception):
        # Tidak di Colab atau error lainnya
        return None

def request_api_key_input(ui_components: Dict[str, Any]) -> None:
    """
    Tampilkan pesan meminta user untuk memasukkan API key jika belum diset.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Tampilkan pesan pada status output
    api_key_msg = (
        "API key Roboflow tidak ditemukan. Silakan masukkan API key pada "
        "field 'API Key' di bagian konfigurasi Roboflow."
    )
    
    # Update status panel
    update_status_panel(
        ui_components,
        api_key_msg,
        "warning"
    )
    
    # Jika RF accordion tertutup, buka
    if 'roboflow_accordion' in ui_components:
        ui_components['roboflow_accordion'].selected_index = 0
    
    # Log ke UI
    log_message(ui_components, api_key_msg, "warning", "🔑")
    
    # Highlight input field untuk API key
    if 'api_key' in ui_components:
        ui_components['api_key'].layout.border = "1px solid red"

def setup_api_key_input(ui_components: Dict[str, Any]) -> None:
    """
    Setup input API key dengan dukungan secrets dan validasi.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Hanya lakukan jika 'api_key' input ada di UI components
    if 'api_key' not in ui_components:
        return
        
    # Cek apakah API key sudah diset
    has_key, _ = check_api_key(ui_components)
    
    if not has_key:
        # Minta input API key
        request_api_key_input(ui_components)
        
        # Tambahkan callback untuk validasi saat ada perubahan
        def on_api_key_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                # Reset style jika nilai dimasukkan
                if change['new']:
                    ui_components['api_key'].layout.border = ""
                    
                    # Log sukses
                    log_message(ui_components, "API key berhasil dimasukkan", "success", "✅")
                    
                    # Update status panel
                    update_status_panel(
                        ui_components,
                        "API key berhasil dimasukkan",
                        "success"
                    )
        
        # Register callback
        ui_components['api_key'].observe(on_api_key_change, names='value')


def check_colab_secrets(ui_components: Dict[str, Any]) -> None:
    """
    Memeriksa Colab secret setelah init UI selesai.
    Fungsi ini harus dipanggil setelah UI diinisialisasi untuk memastikan
    API key dari Colab secret digunakan jika tersedia.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Coba mendapatkan API key dari Colab secrets
    api_key = get_api_key_from_secrets('ROBOFLOW_API_KEY')
    
    if api_key and 'api_key' in ui_components:
        # Update UI component dengan API key dari secret
        ui_components['api_key'].value = api_key
        
        # Log ke UI
        log_message(ui_components, "API key Roboflow berhasil diambil dari Google Colab secrets", "success", "🔑")
        
        # Update status panel
        update_status_panel(
            ui_components,
            "API key Roboflow berhasil diambil dari Google Colab secrets",
            "success"
        )
        
        # Reset style jika sebelumnya di-highlight
        ui_components['api_key'].layout.border = ""
    else:
        # Coba dari environment variable
        api_key = os.environ.get('ROBOFLOW_API_KEY')
        if api_key and 'api_key' in ui_components:
            # Update UI component
            ui_components['api_key'].value = api_key
            
            # Log ke UI
            log_message(ui_components, "API key Roboflow berhasil diambil dari environment variable", "success", "🔑")
            
            # Update status panel
            update_status_panel(
                ui_components,
                "API key Roboflow berhasil diambil dari environment variable",
                "success"
            )
            
            # Reset style
            ui_components['api_key'].layout.border = ""