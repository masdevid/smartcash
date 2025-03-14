"""
File: smartcash/model/config/__init__.py
Deskripsi: Package initialization for model configss
"""

from smartcash.model.config.model_config import ModelConfig
from smartcash.model.config.backbone_config import BackboneConfig
from smartcash.model.config.experiment_config import ExperimentConfig

# Factory function untuk memudahkan akses
def load_config(config_path=None, **kwargs):
    """
    Factory function untuk memuat konfigurasi model.
    
    Args:
        config_path: Path ke file konfigurasi (opsional)
        **kwargs: Parameter tambahan untuk konfigurasi
        
    Returns:
        Instance ModelConfig yang dimuat
    """
    return ModelConfig(config_path, **kwargs)

# Ekspor kelas dan fungsi publik
__all__ = [
    'ModelConfig',        # Konfigurasi model dasar
    'BackboneConfig',     # Konfigurasi backbone network
    'ExperimentConfig',   # Konfigurasi eksperimen
    'load_config'         # Factory function
]