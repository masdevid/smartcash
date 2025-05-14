"""
File: smartcash/ui/dataset/download/handlers/api_key_handler.py
Deskripsi: Handler untuk manajemen API key Roboflow
"""

import os
from typing import Dict, Any, Optional, Tuple
from IPython.display import display, clear_output

def check_api_key(ui_components: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Periksa API key Roboflow dari berbagai sumber dan update UI jika perlu.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Tuple (has_api_key, api_key)
    """
    logger = ui_components.get('logger')
    
    # Check sources in priority order:
    # 1. UI input field
    # 2. Google Colab secrets
    # 3. Environment variable
    
    # Check UI input first
    if 'rf_apikey' in ui_components and ui_components['rf_apikey'].value:
        api_key = ui_components['rf_apikey'].value
        return True, api_key
    
    # Try to get from Google Colab secrets
    api_key = get_api_key_from_secrets('ROBOFLOW_API_KEY')
    if api_key:
        # Update UI component if available
        if 'rf_apikey' in ui_components:
            ui_components['rf_apikey'].value = api_key
        
        if logger:
            logger.info("🔑 API key Roboflow berhasil diambil dari Google Colab secrets")
        return True, api_key
    
    # Check environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    if api_key:
        # Update UI component if available
        if 'rf_apikey' in ui_components:
            ui_components['rf_apikey'].value = api_key
            
        if logger:
            logger.info("🔑 API key Roboflow berhasil diambil dari environment variable")
        return True, api_key
    
    # No API key found
    if logger:
        logger.warning("⚠️ API key Roboflow tidak ditemukan")
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
    from smartcash.ui.utils.ui_logger import log_to_ui
    
    # Tampilkan pesan pada status output
    api_key_msg = (
        "API key Roboflow tidak ditemukan. Silakan masukkan API key pada "
        "field 'API Key' di bagian konfigurasi Roboflow."
    )
    
    # Jika RF accordion tertutup, buka
    if 'rf_accordion' in ui_components:
        ui_components['rf_accordion'].selected_index = 0
    
    # Log ke UI
    log_to_ui(ui_components, api_key_msg, "warning", "🔑")
    
    # Log ke logger
    logger = ui_components.get('logger')
    if logger:
        logger.warning(f"⚠️ {api_key_msg}")
        
    # Highlight input field jika ada
    if 'rf_apikey' in ui_components:
        # Berikan outline merah untuk highlight
        ui_components['rf_apikey'].layout.border = "1px solid red"

def setup_api_key_input(ui_components: Dict[str, Any]) -> None:
    """
    Setup input API key dengan dukungan secrets dan validasi.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Hanya lakukan jika 'rf_apikey' input ada di UI components
    if 'rf_apikey' not in ui_components:
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
                    ui_components['rf_apikey'].layout.border = ""
                    
                    # Log sukses
                    logger = ui_components.get('logger')
                    if logger:
                        logger.info("✅ API key berhasil dimasukkan")
        
        # Register callback
        ui_components['rf_apikey'].observe(on_api_key_change, names='value')