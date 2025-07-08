"""
File: smartcash/ui/dataset/visualization/configs/visualization_config.py
Deskripsi: Konfigurasi default untuk modul visualisasi dataset
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class VisualizationConfig:
    """Konfigurasi untuk modul visualisasi dataset."""
    
    # Daftar split yang akan ditampilkan
    splits: List[str] = field(default_factory=lambda: ['train', 'valid', 'test'])
    
    # Format tampilan persentase
    percentage_format: str = '{:.1f}%'
    
    # Warna untuk setiap split
    colors: Dict[str, str] = field(default_factory=lambda: {
        'train': '#4CAF50',  # Hijau
        'valid': '#2196F3',  # Biru
        'test': '#FF9800'    # Oranye
    })
    
    # Refresh interval dalam detik (0 untuk menonaktifkan auto-refresh)
    refresh_interval: int = 0
    
    # Tampilkan log accordion secara default
    show_log_accordion: bool = True
    
    # Tampilkan tombol refresh
    show_refresh_button: bool = True
    
    # Path default untuk dataset
    default_dataset_path: Optional[str] = None

# Konfigurasi default
DEFAULT_VISUALIZATION_CONFIG = VisualizationConfig()

# Fungsi helper untuk membuat konfigurasi dari dictionary
def create_visualization_config(config_dict: Optional[Dict[str, Any]] = None) -> VisualizationConfig:
    """Buat instance VisualizationConfig dari dictionary.
    
    Args:
        config_dict: Dictionary berisi konfigurasi
        
    Returns:
        Instance VisualizationConfig
    """
    if not config_dict:
        return VisualizationConfig()
    
    # Buat salinan dari konfigurasi default
    config = VisualizationConfig()
    
    # Update dengan nilai dari config_dict
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
