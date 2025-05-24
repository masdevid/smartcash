"""
File: smartcash/ui/dataset/preprocessing/handlers/__init__.py
Deskripsi: Package initialization untuk preprocessing handlers yang sudah direfactor
"""

# New SRP handlers
from .config_save_handler import setup_config_save_handler
from .config_reset_handler import setup_config_reset_handler
from .preprocessing_executor import setup_preprocessing_executor
from .cleanup_executor import setup_cleanup_executor
from .dataset_checker import setup_dataset_checker

# Legacy handler yang masih digunakan untuk backward compatibility
from .config_handler import setup_config_handlers

__all__ = [
    # New SRP handlers
    'setup_config_save_handler',
    'setup_config_reset_handler', 
    'setup_preprocessing_executor',
    'setup_cleanup_executor',
    'setup_dataset_checker',
    
    # Legacy handlers (for backward compatibility)
    'setup_config_handlers'
]