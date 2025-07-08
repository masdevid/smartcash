"""
File: smartcash/model/config/__init__.py
Deskripsi: Inisialisasi paket konfigurasi model (mengekspos semua konstanta dan fungsi dari model_constants)
"""

from smartcash.model.config.model_constants import (
    DETECTION_LAYERS,
    DETECTION_THRESHOLDS,
    LAYER_CONFIG,
    SUPPORTED_BACKBONES,
    SUPPORTED_EFFICIENTNET_MODELS,
    EFFICIENTNET_CHANNELS,
    YOLO_CHANNELS,
    YOLOV5_CONFIG,
    OPTIMIZED_MODELS,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_MODEL_CONFIG_FULL,
    get_model_config
)

# Factory function untuk memudahkan akses
def load_config(config_path=None, **kwargs):
    """Factory function untuk memuat konfigurasi model dari file atau parameter."""
    return ModelConfig(config_path, **kwargs)

# Ekspor kelas dan fungsi publik
__all__ = [
    # Kelas konfigurasi
    'ModelConfig',           # Konfigurasi model dasar
    'BackboneConfig',        # Konfigurasi backbone network
    'ExperimentConfig',      # Konfigurasi eksperimen
    
    # Konstanta
    'DETECTION_LAYERS',      # Layer deteksi yang didukung
    'DETECTION_THRESHOLDS',  # Threshold default untuk setiap layer
    'LAYER_CONFIG',          # Konfigurasi lengkap untuk setiap layer deteksi
    'SUPPORTED_BACKBONES',   # Backbone yang didukung
    'SUPPORTED_EFFICIENTNET_MODELS', # Model EfficientNet yang didukung
    'EFFICIENTNET_CHANNELS', # Channel output untuk setiap stage EfficientNet
    'YOLO_CHANNELS',         # Channel standar YOLOv5 untuk feature maps
    'YOLOV5_CONFIG',         # Konfigurasi model YOLOv5 untuk CSPDarknet
    'OPTIMIZED_MODELS',      # Konfigurasi model yang dioptimasi
    'DEFAULT_MODEL_CONFIG',  # Konfigurasi model default sederhana
    'DEFAULT_MODEL_CONFIG_FULL', # Konfigurasi model default lengkap
    
    # Fungsi
    'get_model_config',      # Mendapatkan konfigurasi model berdasarkan tipe
    'load_config'            # Factory function untuk memuat konfigurasi
]