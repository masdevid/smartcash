"""
File: smartcash/ui/info_boxes/__init__.py
Deskripsi: Module untuk mengakses info box terpusat dengan pendekatan modular
"""

from smartcash.ui.info_boxes.environment_info import get_environment_info
from smartcash.ui.info_boxes.dependencies_info import get_dependencies_info
from smartcash.ui.info_boxes.preprocessing_info import get_preprocessing_info
from smartcash.ui.info_boxes.split_info import get_split_info
from smartcash.ui.info_boxes.download_info import get_download_info
from smartcash.ui.info_boxes.augmentation_info import get_augmentation_info
from smartcash.ui.info_boxes.dataset_info import get_dataset_info
from smartcash.ui.info_boxes.backbones_info import get_backbones_info
from smartcash.ui.components import create_info_accordion


__all__ = [
    'get_environment_info',
    'get_preprocessing_info',
    'get_dependencies_info',
    'get_split_info',
    'get_download_info',
    'get_augmentation_info',
    'get_dataset_info',
    'get_backbones_info',
    'create_info_accordion',
]