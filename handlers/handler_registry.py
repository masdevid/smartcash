# File: smartcash/handlers/handler_registry.py
# Author: Alfrida Sabar
# Deskripsi: Registry untuk handler dengan factory method

from typing import Dict, Type, Any, Optional, Callable
import inspect

from smartcash.utils.logger import get_logger
from smartcash.exceptions.base import ConfigError

# Import BaseHandler dinamis untuk menghindari circular import
# BaseHandler akan didefinisikan nanti
BaseHandler = None

class HandlerRegistry:
    """
    Registry untuk handler dengan factory method.
    Memungkinkan pembuatan handler dengan nama dinamis.
    """
    
    # Registry internal untuk menyimpan handler class
    _registry: Dict[str, Type] = {}
    
    # Logger
    _logger = get_logger("handler_registry")
    
    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        """
        Dekorator untuk mendaftarkan handler class.
        
        Args:
            name: Nama handler di registry (opsional, default: class name)
            
        Returns:
            Decorator function
        """
        def decorator(handler_class):
            # Import BaseHandler jika belum
            global BaseHandler
            if BaseHandler is None:
                from smartcash.handlers.base_handler import BaseHandler as BH
                BaseHandler = BH
            
            # Validasi inheritance
            if not inspect.isclass(handler_class) or not issubclass(handler_class, BaseHandler):
                raise TypeError(f"Hanya class turunan BaseHandler yang dapat didaftarkan: {handler_class.__name__}")
                
            # Gunakan nama class jika name tidak disediakan
            handler_name = name or handler_class.__name__
            
            # Daftarkan handler
            cls._registry[handler_name] = handler_class
            cls._logger.debug(f"✅ Handler '{handler_name}' didaftarkan")
            
            return handler_class
            
        return decorator
    
    @classmethod
    def create(
        cls,
        handler_name: str,
        **kwargs
    ):
        """
        Buat instance handler berdasarkan nama.
        
        Args:
            handler_name: Nama handler di registry
            **kwargs: Parameter untuk konstruktor handler
            
        Returns:
            Instance handler
            
        Raises:
            ConfigError: Jika handler tidak ditemukan
        """
        # Cek apakah handler terdaftar
        if handler_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ConfigError(
                f"Handler '{handler_name}' tidak ditemukan. "
                f"Handler tersedia: {available}"
            )
            
        # Buat instance
        handler_class = cls._registry[handler_name]
        instance = handler_class(**kwargs)
        
        cls._logger.debug(f"🔧 Instance handler '{handler_name}' berhasil dibuat")
        return instance
    
    @classmethod
    def get_registered_handlers(cls) -> Dict[str, Type]:
        """
        Dapatkan semua handler yang terdaftar.
        
        Returns:
            Dictionary nama handler ke class handler
        """
        return cls._registry.copy()
    
    @classmethod
    def is_registered(cls, handler_name: str) -> bool:
        """
        Cek apakah handler terdaftar.
        
        Args:
            handler_name: Nama handler
            
        Returns:
            True jika handler terdaftar
        """
        return handler_name in cls._registry
    
    @classmethod
    def unregister(cls, handler_name: str) -> bool:
        """
        Hapus handler dari registry.
        
        Args:
            handler_name: Nama handler
            
        Returns:
            True jika handler berhasil dihapus
        """
        if handler_name in cls._registry:
            del cls._registry[handler_name]
            cls._logger.debug(f"❌ Handler '{handler_name}' dihapus dari registry")
            return True
        return False