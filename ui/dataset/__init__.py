"""
Dataset package for SmartCash UI.

This package contains modules for dataset manipulation including augmentation, preprocessing, and visualization.
"""
import importlib
import logging
from typing import Optional, Any, Callable

logger = logging.getLogger(__name__)

# Initialize empty modules dictionary to store imported modules
modules = {}

def safe_import(module_name: str, import_path: str, import_name: Optional[str] = None) -> Any:
    """Safely import a module or attribute, returning None on failure.
    
    Args:
        module_name: Name to use as the module key
        import_path: Full import path (e.g., '.split')
        import_name: Optional specific attribute to import (e.g., 'initialize_split_ui')
        
    Returns:
        The imported module/attribute or None if import fails
    """
    try:
        if import_name:
            module = importlib.import_module(import_path, package=__package__)
            attr = getattr(module, import_name)
            modules[module_name] = attr
            modules[f'initialize_{module_name}_ui'] = attr  # Also store with initialize_*_ui name
            return attr
        else:
            module = importlib.import_module(import_path, package=__package__)
            modules[module_name] = module
            return module
    except Exception as e:
        logger.warning(f"Failed to import {import_path}.{import_name or ''}: {str(e)}", exc_info=True)
        return None

# Import all dataset modules with error handling
downloader = safe_import('downloader', '.downloader')
initialize_downloader_ui = safe_import('downloader', '.downloader', 'initialize_downloader_ui')

split = safe_import('split', '.split')
initialize_split_ui = safe_import('split_ui', '.split', 'initialize_split_ui')

preprocessing = safe_import('preprocessing', '.preprocessing')
initialize_preprocessing_ui = safe_import('preprocessing_ui', '.preprocessing', 'initialize_preprocessing_ui')

augmentation = safe_import('augmentation', '.augmentation')
initialize_augmentation_ui = safe_import('augmentation_ui', '.augmentation', 'initialize_augmentation_ui')

# Define public API
__all__ = [
    'downloader',
    'initialize_downloader_ui',
    'split',
    'initialize_split_ui',
    'preprocessing',
    'initialize_preprocessing_ui',
    'augmentation',
    'initialize_augmentation_ui'
]

# Clean up
safe_import = None
