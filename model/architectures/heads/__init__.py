"""
File: smartcash/model/architectures/heads/__init__.py
Deskripsi: File init untuk mengekspor komponen detection head
"""

from smartcash.model.architectures.heads.detection_head import DetectionHead, LAYER_CONFIG

__all__ = ['DetectionHead', 'LAYER_CONFIG']