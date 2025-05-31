"""
File: smartcash/ui/pretrained_model/utils/download_utils.py
Deskripsi: Utilitas untuk proses download model pretrained
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from smartcash.ui.pretrained_model.utils.logger_utils import get_module_logger, log_message
from smartcash.ui.pretrained_model.utils.progress import update_progress_ui

logger = get_module_logger()

def prepare_model_info(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Menyiapkan informasi model yang akan diunduh dari konfigurasi
    
    Args:
        config: Konfigurasi yang berisi informasi model
        
    Returns:
        List informasi model yang siap diunduh
    """
    models_info = []
    models = config.get('models', {})
    
    for model_name, model_data in models.items():
        models_info.append({
            'name': model_name,
            'path': model_data.get('path', ''),
            'url': model_data.get('url', ''),
            'size': model_data.get('size', 0)
        })
    
    return models_info

def check_model_exists(model_path: Union[str, Path]) -> bool:
    """
    Memeriksa apakah model sudah ada di path yang ditentukan
    
    Args:
        model_path: Path model yang akan diperiksa
        
    Returns:
        True jika model sudah ada, False jika belum
    """
    if isinstance(model_path, str):
        model_path = Path(model_path)
    
    return model_path.exists() and model_path.is_file() and model_path.stat().st_size > 0

def check_models_in_drive(config: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Memeriksa apakah model sudah ada di Google Drive
    
    Args:
        config: Konfigurasi yang berisi informasi model dan path Drive
        
    Returns:
        Tuple (semua_model_ada, list_model_info)
    """
    drive_models_dir = config.get('drive_models_dir', '')
    if not drive_models_dir or not os.path.exists(drive_models_dir):
        return False, []
    
    models_info = prepare_model_info(config)
    models_in_drive = []
    all_models_exist = True
    
    for model_info in models_info:
        model_name = model_info.get('name', '')
        model_path = model_info.get('path', '')
        
        if not model_path:
            continue
            
        # Konversi path lokal ke path Drive
        drive_path = os.path.join(drive_models_dir, os.path.basename(model_path))
        
        if check_model_exists(drive_path):
            models_in_drive.append({
                'name': model_name,
                'local_path': model_path,
                'drive_path': drive_path
            })
        else:
            all_models_exist = False
    
    return all_models_exist, models_in_drive

def get_models_to_download(config: Dict[str, Any], ui_components: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Mendapatkan daftar model yang perlu diunduh
    
    Args:
        config: Konfigurasi yang berisi informasi model
        ui_components: Komponen UI untuk logging
        
    Returns:
        List model yang perlu diunduh
    """
    models_info = prepare_model_info(config)
    models_to_download = []
    
    for model_info in models_info:
        model_name = model_info.get('name', '')
        model_path = model_info.get('path', '')
        model_url = model_info.get('url', '')
        
        if not model_path or not model_url:
            log_message(ui_components, f"Informasi model tidak lengkap untuk {model_name}", "warning")
            continue
            
        if not check_model_exists(model_path):
            log_message(ui_components, f"Model {model_name} perlu diunduh", "info")
            models_to_download.append(model_info)
    
    return models_to_download
