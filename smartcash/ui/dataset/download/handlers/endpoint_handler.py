"""
File: smartcash/ui/dataset/download/handlers/endpoint_handler.py
Deskripsi: Handler untuk manajemen endpoint download dataset
"""

import os
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
    
    # Gunakan custom logging UI
    log_change_to_ui(ui_components, new_endpoint)

def _reset_accordion_visibility(ui_components: Dict[str, Any]) -> None:
    """Reset visibility untuk semua accordion."""
    for accordion_key in ['rf_accordion', 'drive_accordion']:
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
        'Google Drive': '📁'
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
    # Default endpoints - hanya Roboflow dan Google Drive
    endpoints = ['Roboflow', 'Google Drive']
    
    # Cek ketersediaan tiap endpoint
    try:
        # Cek Drive (jika di Colab)
        from smartcash.common.environment import get_environment_manager
        env = get_environment_manager()
        if not env.is_colab:
            # Hapus Drive dari endpoint jika tidak di Colab
            endpoints.remove('Google Drive')
    except ImportError:
        pass
    
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
            'api_key': ui_components.get('rf_apikey', {}).value or os.environ.get('ROBOFLOW_API_KEY', ''),
            'format': 'yolov5pytorch'  # Format tetap
        }
    elif endpoint == 'Google Drive':
        config = {
            'type': 'drive',
            'folder': ui_components.get('drive_folder', {}).value,
            'format': 'yolov5pytorch'  # Format tetap
        }
    
    # Tambahkan konfigurasi output yang sama untuk semua endpoint
    config.update({
        'output_dir': ui_components.get('output_dir', {}).value,
    })
    
    return config