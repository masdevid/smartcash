"""
File: smartcash/ui/dataset/handlers/endpoint_handler.py
Deskripsi: Handler untuk manajemen endpoint download dataset
"""

from typing import Dict, Any, List, Optional

def handle_endpoint_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan endpoint dataset.
    
    Args:
        change: Dictionary perubahan dari widget observe
        ui_components: Dictionary komponen UI
    """
    new_endpoint = change.get('new')
    logger = ui_components.get('logger')
    
    # Reset visibility untuk semua accordion
    _reset_accordion_visibility(ui_components)
    
    # Update visibility berdasarkan endpoint yang dipilih
    if new_endpoint == 'Roboflow':
        ui_components['rf_accordion'].selected_index = 0
        if logger: logger.info("ℹ️ Endpoint diubah ke Roboflow")
    elif new_endpoint == 'Google Drive':
        ui_components['drive_accordion'].selected_index = 0
        if logger: logger.info("ℹ️ Endpoint diubah ke Google Drive")
    elif new_endpoint == 'URL Kustom':
        ui_components['url_accordion'].selected_index = 0
        if logger: logger.info("ℹ️ Endpoint diubah ke URL Kustom")
    
    # Gunakan custom logging UI
    log_change_to_ui(ui_components, new_endpoint)

def _reset_accordion_visibility(ui_components: Dict[str, Any]) -> None:
    """Reset visibility untuk semua accordion."""
    for accordion_key in ['rf_accordion', 'drive_accordion', 'url_accordion']:
        if accordion_key in ui_components:
            ui_components[accordion_key].selected_index = None

def log_change_to_ui(ui_components: Dict[str, Any], endpoint: str) -> None:
    """
    Log perubahan endpoint ke UI.
    
    Args:
        ui_components: Dictionary komponen UI
        endpoint: Endpoint yang dipilih
    """
    from smartcash.ui.utils.ui_logger import log_to_ui
    
    # Map emoji berdasarkan endpoint
    endpoint_emoji = {
        'Roboflow': '🧩',
        'Google Drive': '📁', 
        'URL Kustom': '🔗'
    }
    
    emoji = endpoint_emoji.get(endpoint, '📌')
    log_to_ui(ui_components, f"Endpoint diubah ke {endpoint}", "info", emoji)

def get_available_endpoints(ui_components: Dict[str, Any]) -> List[str]:
    """
    Dapatkan daftar endpoint yang tersedia.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        List endpoint yang tersedia
    """
    # Default endpoints
    endpoints = ['Roboflow', 'Google Drive', 'URL Kustom']
    
    # Cek ketersediaan tiap endpoint
    try:
        # Cek Roboflow
        import roboflow
        
        # Cek Drive (jika di Colab)
        try:
            from smartcash.common.environment import get_environment_manager
            env = get_environment_manager()
            if not env.is_colab:
                # Hapus Drive dari endpoint jika tidak di Colab
                endpoints.remove('Google Drive')
        except ImportError:
            pass
    except ImportError:
        # Roboflow tidak tersedia
        endpoints.remove('Roboflow')
    
    # Logging tersedia endpoint
    logger = ui_components.get('logger')
    if logger:
        logger.debug(f"📋 Endpoint tersedia: {', '.join(endpoints)}")
    
    return endpoints

def get_endpoint_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi endpoint yang dipilih.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi endpoint
    """
    endpoint = ui_components.get('endpoint_dropdown', {}).value
    config = {}
    
    if endpoint == 'Roboflow':
        config = {
            'type': 'roboflow',
            'workspace': ui_components.get('rf_workspace', {}).value,
            'project': ui_components.get('rf_project', {}).value,
            'version': ui_components.get('rf_version', {}).value,
            'api_key': ui_components.get('rf_apikey', {}).value,
        }
    elif endpoint == 'Google Drive':
        config = {
            'type': 'drive',
            'folder': ui_components.get('drive_folder', {}).value
        }
    elif endpoint == 'URL Kustom':
        config = {
            'type': 'url',
            'url': ui_components.get('url_input', {}).value
        }
    
    # Tambahkan konfigurasi output yang sama untuk semua endpoint
    config.update({
        'output_dir': ui_components.get('output_dir', {}).value,
        'output_format': ui_components.get('output_format', {}).value
    })
    
    return config