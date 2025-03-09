# File: smartcash/handlers/__init__.py
# Author: Alfrida Sabar
# Deskripsi: Package untuk handlers SmartCash

from smartcash.handlers.base_handler import BaseHandler
from smartcash.handlers.handler_registry import HandlerRegistry
    
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
    'create_handler',
]