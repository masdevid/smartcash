"""
File: smartcash/ui/setup/env_config/utils/config_utils.py
Deskripsi: Fungsi-fungsi utilitas untuk operasi konfigurasi environment
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from smartcash.common.config import get_config_manager, SimpleConfigManager

def init_config_manager(base_dir: Optional[Path] = None, 
                       config_file: str = "configs/base_config.yaml") -> SimpleConfigManager:
    """
    Inisialisasi config manager
    
    Args:
        base_dir: Path direktori dasar
        config_file: Path file konfigurasi relatif terhadap base_dir
        
    Returns:
        Instance SimpleConfigManager
    """
    return get_config_manager(base_dir=base_dir, config_file=config_file)

def ensure_config_dir(config_dir: Path) -> Path:
    """
    Pastikan direktori konfigurasi ada
    
    Args:
        config_dir: Path direktori konfigurasi
        
    Returns:
        Path direktori konfigurasi yang terkonfirmasi
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir 