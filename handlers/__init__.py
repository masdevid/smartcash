# File: smartcash/handlers/__init__.py
# Author: Alfrida Sabar
# Deskripsi: Package untuk handlers SmartCash

from smartcash.handlers.base_handler import BaseHandler
from smartcash.handlers.handler_registry import HandlerRegistry
from smartcash.handlers.model_handler import ModelHandler
from smartcash.handlers.data_handler import DataHandler
from smartcash.handlers.model_factory import get_model_factory
from smartcash.handlers.optimizer_factory import get_optimizer_factory
from smartcash.handlers.checkpoint_manager import get_checkpoint_manager

# Handler dapat dibuat dengan factory pattern
def create_handler(handler_name: str, **kwargs):
    """
    Buat handler berdasarkan nama.
    
    Args:
        handler_name: Nama handler di registry
        **kwargs: Parameter tambahan untuk handler
        
    Returns:
        Instance handler yang dibuat
    """
    return HandlerRegistry.create(handler_name, **kwargs)

# Eksport semua komponen utama
__all__ = [
    'BaseHandler',
    'HandlerRegistry',
    'ModelHandler',
    'DataHandler',
    'get_model_factory',
    'get_optimizer_factory',
    'get_checkpoint_manager',
    'create_handler'
]