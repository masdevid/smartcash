"""
File: smartcash/model/architectures/heads/__init__.py
Deskripsi: Inisialisasi dan export modul heads untuk deteksi objek
"""

from smartcash.model.architectures.heads.detection_head import DetectionHead, LAYER_CONFIG

__all__ = [
    'DetectionHead',
    'LAYER_CONFIG'
]