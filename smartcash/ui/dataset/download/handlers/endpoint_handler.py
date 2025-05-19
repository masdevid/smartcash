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
    # Fungsi ini dipertahankan untuk kompatibilitas dengan kode lama
    # tetapi tidak lagi digunakan karena kita hanya menggunakan Roboflow
    logger = ui_components.get('logger')
    if logger: 
        logger.debug("ℹ️ Endpoint handler dipanggil tapi tidak melakukan apa-apa karena hanya menggunakan Roboflow")

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
    from smartcash.ui.components.status_panel import update_status_panel
    
    # Map emoji berdasarkan endpoint
    endpoint_emoji = {
        'Roboflow': '🧩',
        'Google Drive': '📁'
    }
    
    emoji = endpoint_emoji.get(endpoint, '📌')
    
    # Log ke output area
    log_to_ui(ui_components, f"Endpoint diubah ke {endpoint}", "info", emoji)
    
    # Update status panel
    update_status_panel(
        ui_components['status_panel'],
        f"Siap untuk download dari {endpoint}",
        "info"
    )

def get_available_endpoints(ui_components: Dict[str, Any]) -> List[str]:
    """
    Dapatkan daftar endpoint yang tersedia.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        List endpoint yang tersedia
    """
    # Hanya menggunakan Roboflow
    endpoints = ['Roboflow']
    
    # Logging tersedia endpoint
    logger = ui_components.get('logger')
    if logger:
        logger.debug(f"📋 Endpoint tersedia: {', '.join(endpoints)}")
    
    return endpoints

def get_endpoint_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi Roboflow untuk download dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi endpoint
    """
    # Konfigurasi Roboflow
    config = {
        'type': 'roboflow',
        'workspace': ui_components.get('rf_workspace', {}).value,
        'project': ui_components.get('rf_project', {}).value,
        'version': ui_components.get('rf_version', {}).value,
        'api_key': ui_components.get('rf_apikey', {}).value or os.environ.get('ROBOFLOW_API_KEY', ''),
        'format': 'yolov5pytorch',  # Format tetap
        'output_dir': ui_components.get('output_dir', {}).value,
        'validate': ui_components.get('validate_dataset', {}).value
    }
    
    return config