"""
File: smartcash/common/config/singleton.py
Deskripsi: Implementasi singleton untuk ConfigManager dengan dukungan Google Colab
"""

import os
from typing import Dict, Any, Optional, Type, TypeVar

# Type variable untuk dependency injection
T = TypeVar('T')

class Singleton:
    """
    Implementasi singleton pattern untuk memastikan hanya ada satu instance ConfigManager
    dengan dukungan Google Colab
    """
    _instance = None
    _is_colab = None
    
    @classmethod
    def is_colab_environment(cls) -> bool:
        """
        Deteksi apakah sedang berjalan di Google Colab
        
        Returns:
            bool: True jika berjalan di Colab, False jika tidak
        """
        if cls._is_colab is None:
            try:
                import google.colab
                cls._is_colab = True
            except ImportError:
                cls._is_colab = False
        return cls._is_colab
    
    @classmethod
    def get_instance(cls, *args, **kwargs):
        """
        Dapatkan instance singleton dengan dukungan Colab
        
        Returns:
            Instance singleton
        """
        if cls._instance is None:
            if cls.is_colab_environment():
                # Inisialisasi khusus untuk Colab
                from smartcash.common.config.colab_manager import ColabConfigManager
                cls._instance = ColabConfigManager(*args, **kwargs)
            else:
                cls._instance = cls(*args, **kwargs)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """
        Reset instance singleton (untuk testing)
        """
        cls._instance = None
        cls._is_colab = None
