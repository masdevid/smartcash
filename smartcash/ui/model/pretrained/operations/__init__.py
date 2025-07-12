"""
File: smartcash/ui/model/pretrained/operations/__init__.py
Operation exports for pretrained module
"""

from .pretrained_operation_manager import PretrainedOperationManager
from .download_operation import DownloadOperation

__all__ = [
    'PretrainedOperationManager',
    'DownloadOperation'
]