"""
File: smartcash/ui/info_boxes/__init__.py
Deskripsi: Module untuk mengakses info box terpusat dengan pendekatan modular
"""

from smartcash.ui.info_boxes.environment_info import get_environment_info
from smartcash.ui.info_boxes.dependencies_info import get_dependencies_info
from smartcash.ui.info_boxes.preprocessing_info import get_preprocessing_info
from smartcash.ui.utils.info_utils import create_info_accordion

__all__ = [
    'get_environment_info',
    'get_preprocessing_info',
    'get_dependencies_info',
    'create_info_accordion',
]