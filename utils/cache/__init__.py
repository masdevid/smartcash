"""
File: smartcash/utils/cache/__init__.py
Author: Alfrida Sabar
Deskripsi: File inisialisasi untuk modul cache yang mengekspos kelas-kelas utama.
"""

from .cache_manager import CacheManager
from .cache_index import CacheIndex
from .cache_storage import CacheStorage
from .cache_cleanup import CacheCleanup
from .cache_stats import CacheStats

# Alias untuk kompatibilitas mundur
EnhancedCache = CacheManager

__all__ = ['CacheManager', 'CacheIndex', 'CacheStorage', 'CacheCleanup', 'CacheStats', 'EnhancedCache']