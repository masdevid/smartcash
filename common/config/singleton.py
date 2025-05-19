"""
File: smartcash/common/config/singleton.py
Deskripsi: Implementasi singleton untuk ConfigManager
"""

from typing import Dict, Any, Optional, Type, TypeVar

# Type variable untuk dependency injection
T = TypeVar('T')

class Singleton:
    """
    Implementasi singleton pattern untuk memastikan hanya ada satu instance ConfigManager
    """
    _instance = None
    
    @classmethod
    def get_instance(cls, *args, **kwargs):
        """
        Dapatkan instance singleton
        
        Returns:
            Instance singleton
        """
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """
        Reset instance singleton (untuk testing)
        """
        cls._instance = None
