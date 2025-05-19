"""
File: smartcash/ui/setup/env_config/utils/env_utils.py
Deskripsi: Utilitas untuk konfigurasi environment
"""

from typing import Dict, Any, Tuple, List
import os
import platform
import sys
from pathlib import Path

def get_env_status(env_manager: Any) -> Dict[str, Any]:
    """
    Dapatkan status environment
    
    Args:
        env_manager: Environment manager
    
    Returns:
        Dictionary berisi status environment
    """
    # Dapatkan informasi sistem
    system_info = env_manager.get_system_info() if hasattr(env_manager, 'get_system_info') else {}
    
    # Dapatkan status drive
    drive_status = {
        'is_mounted': env_manager.is_drive_mounted if hasattr(env_manager, 'is_drive_mounted') else False,
        'drive_path': str(env_manager.drive_path) if hasattr(env_manager, 'drive_path') else None
    }
    
    # Dapatkan status direktori
    directory_status = {}
    if hasattr(env_manager, 'base_dir'):
        base_dir = Path(env_manager.base_dir)
        required_dirs = [
            "configs",
            "data",
            "data/raw",
            "data/processed",
            "models",
            "models/checkpoints",
            "models/weights",
            "output",
            "logs"
        ]
        
        for dir_name in required_dirs:
            dir_path = base_dir / dir_name
            directory_status[dir_name] = dir_path.exists()
    
    # Gabungkan semua status
    return {
        'system_info': system_info,
        'drive_status': drive_status,
        'directory_status': directory_status
    }

def format_env_info(env_status: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Format informasi environment untuk ditampilkan
    
    Args:
        env_status: Status environment
    
    Returns:
        List tuple (label, value) untuk ditampilkan
    """
    formatted_info = []
    
    # Format informasi sistem
    system_info = env_status.get('system_info', {})
    if system_info:
        formatted_info.append(("Sistem Operasi", system_info.get('os', 'Tidak diketahui')))
        formatted_info.append(("Python Version", system_info.get('python_version', 'Tidak diketahui')))
        formatted_info.append(("Colab", "Ya" if system_info.get('is_colab', False) else "Tidak"))
        formatted_info.append(("Kaggle", "Ya" if system_info.get('is_kaggle', False) else "Tidak"))
    
    # Format status drive
    drive_status = env_status.get('drive_status', {})
    if drive_status:
        formatted_info.append(("Google Drive", "Terhubung" if drive_status.get('is_mounted', False) else "Tidak terhubung"))
        if drive_status.get('is_mounted', False) and drive_status.get('drive_path'):
            formatted_info.append(("Drive Path", drive_status.get('drive_path', 'Tidak diketahui')))
    
    # Format status direktori
    directory_status = env_status.get('directory_status', {})
    if directory_status:
        dir_status_str = ", ".join([f"{dir_name}" for dir_name, exists in directory_status.items() if exists])
        formatted_info.append(("Direktori Tersedia", dir_status_str if dir_status_str else "Tidak ada"))
        
        missing_dirs = ", ".join([f"{dir_name}" for dir_name, exists in directory_status.items() if not exists])
        if missing_dirs:
            formatted_info.append(("Direktori Tidak Tersedia", missing_dirs))
    
    return formatted_info
