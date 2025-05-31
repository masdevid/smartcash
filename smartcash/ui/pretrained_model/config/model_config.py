"""
File: smartcash/ui/pretrained_model/config/model_config.py
Deskripsi: Konfigurasi terpusat untuk model pretrained yang digunakan dalam aplikasi
"""

from pathlib import Path
from typing import Dict, Any

# Konfigurasi model terpusat sebagai satu sumber kebenaran
MODEL_CONFIG = {
    # Model YOLOv5
    'yolov5': {
        'name': 'YOLOv5s',
        'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt',
        'filename': 'yolov5s.pt',
        'size_mb': 14,
        'size_bytes': 14 * 1024 * 1024,
        'source': 'ultralytics/yolov5',
        'version': 'v6.1',
        'id': 'yolov5s-v6.1'
    },
    # Model EfficientNet-B4
    'efficientnet-b4': {
        'name': 'EfficientNet-B4',
        'url': 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin',
        'filename': 'efficientnet_b4_huggingface.bin',
        'size_mb': 75,
        'size_bytes': 75 * 1024 * 1024,
        'source': 'timm (Hugging Face)',
        'version': 'timm-1.0',
        'id': 'efficientnet_b4_timm-1.0'
    }
}

def get_model_path(model_key: str, base_dir: str = '/content/models') -> Path:
    """Mendapatkan path lengkap untuk model berdasarkan konfigurasi"""
    model_config = MODEL_CONFIG.get(model_key, {})
    filename = model_config.get('filename', '')
    return Path(base_dir) / filename if filename else Path(base_dir)

def get_model_config(model_key: str) -> Dict[str, Any]:
    """Mendapatkan konfigurasi lengkap untuk model tertentu"""
    return MODEL_CONFIG.get(model_key, {})

def get_all_models() -> Dict[str, Dict[str, Any]]:
    """Mendapatkan konfigurasi semua model"""
    return MODEL_CONFIG

def get_model_info_for_ui(model_key: str) -> Dict[str, Any]:
    """Mendapatkan informasi model untuk ditampilkan di UI"""
    model_config = MODEL_CONFIG.get(model_key, {})
    return {
        'name': model_config.get('name', ''),
        'size': f"{model_config.get('size_mb', 0)} MB",
        'url': model_config.get('url', ''),
        'source': model_config.get('source', '')
    }

def get_model_info_for_download(model_key: str, base_dir: str = '/content/models') -> Dict[str, Any]:
    """Mendapatkan informasi model untuk proses download"""
    model_config = MODEL_CONFIG.get(model_key, {})
    return {
        'name': model_key,
        'url': model_config.get('url', ''),
        'path': get_model_path(model_key, base_dir),
        'min_size': model_config.get('size_bytes', 0) // 2,  # Minimal setengah dari ukuran sebenarnya
        'size': model_config.get('size_bytes', 0),
        'id': model_config.get('id', ''),
        'version': model_config.get('version', ''),
        'source': model_config.get('source', '')
    }
