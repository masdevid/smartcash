"""
File: smartcash/model/architectures/heads/__init__.py
Deskripsi: Ekspor komponen detection head
"""

from smartcash.model.architectures.heads.detection_head import DetectionHead
from smartcash.model.config.model_constants import LAYER_CONFIG

__all__ = ['DetectionHead', 'LAYER_CONFIG']