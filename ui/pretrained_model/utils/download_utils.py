"""
File: smartcash/ui/pretrained_model/utils/download_utils.py
Deskripsi: Utilitas untuk proses download model pretrained
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from smartcash.ui.pretrained_model.utils.logger_utils import get_module_logger, log_message
from smartcash.ui.pretrained_model.utils.progress import update_progress_ui
from smartcash.ui.pretrained_model.config.model_config import get_all_models, get_model_config, get_model_path, get_model_info_for_download

logger = get_module_logger()

def prepare_model_info(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Menyiapkan informasi model yang akan diunduh dari konfigurasi
    
    Args:
        config: Konfigurasi yang berisi informasi model
        
    Returns:
        List informasi model yang siap diunduh
    """
    # One-liner untuk mendapatkan informasi model dari konfigurasi terpusat
    models_dir = config.get('models_dir', '/content/models')
    return [get_model_info_for_download(model_key, models_dir) for model_key in get_all_models()]

def check_model_exists(model_path: Union[str, Path]) -> bool:
    """
    Memeriksa apakah model sudah ada di path yang ditentukan
    
    Args:
        model_path: Path model yang akan diperiksa
        
    Returns:
        True jika model sudah ada, False jika belum
    """
    # One-liner dengan konversi path dan validasi file
    path = Path(model_path) if isinstance(model_path, str) else model_path
    return path.exists() and path.is_file() and path.stat().st_size > 0

def get_models_to_download(config: Dict[str, Any], ui_components: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Mendapatkan daftar model yang perlu diunduh
    
    Args:
        config: Konfigurasi yang berisi informasi model
        ui_components: Komponen UI untuk logging
        
    Returns:
        List model yang perlu diunduh
    """
    # Dapatkan informasi model dari konfigurasi terpusat
    models_info = prepare_model_info(config)
    
    # Filter model yang perlu diunduh dengan one-liner
    models_to_download = []
    for model_info in models_info:
        model_name = model_info.get('name', '')
        model_path = model_info.get('path', '')
        model_url = model_info.get('url', '')
        
        # Validasi informasi model dan cek keberadaan file
        if not model_path or not model_url:
            log_message(ui_components, f"Informasi model tidak lengkap untuk {model_name}", "warning")
        elif not check_model_exists(model_path):
            log_message(ui_components, f"Model {model_name} perlu diunduh", "info")
            models_to_download.append(model_info)
    
    return models_to_download
