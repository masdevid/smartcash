"""
File: smartcash/ui/dataset/preprocessing/handlers/__init__.py
Deskripsi: Updated handlers module exports dengan base class pattern
"""

# Main coordinator
from .event_handlers import setup_all_handlers

# Base class
from .base_handler import BasePreprocessingHandler

# SRP handlers dengan base class
from .config_event_handlers import setup_config_handlers, ConfigEventHandler
from .operation_handlers import setup_operation_handlers, OperationHandler
from .confirmation_handlers import setup_confirmation_handlers, ConfirmationHandler

# Config management (unchanged)
from .config_handler import PreprocessingConfigHandler
from .config_extractor import extract_preprocessing_config
from .config_updater import update_preprocessing_ui, reset_preprocessing_ui
from .defaults import get_default_preprocessing_config

__all__ = [
    # Main coordinator
    'setup_all_handlers',
    
    # Base class
    'BasePreprocessingHandler',
    
    # SRP handlers
    'setup_config_handlers', 'ConfigEventHandler',
    'setup_operation_handlers', 'OperationHandler', 
    'setup_confirmation_handlers', 'ConfirmationHandler',
    
    # Config management
    'PreprocessingConfigHandler',
    'extract_preprocessing_config',
    'update_preprocessing_ui',
    'reset_preprocessing_ui',
    'get_default_preprocessing_config'
]