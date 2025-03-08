"""
File: smartcash/utils/cache/__init__.py
Author: Alfrida Sabar
Deskripsi: File inisialisasi untuk modul cache yang mengekspos kelas-kelas utama.
"""

from smartcash.utils.cache.cache_manager import CacheManager
from smartcash.utils.cache.cache_index import CacheIndex
from smartcash.utils.cache.cache_storage import CacheStorage
from smartcash.utils.cache.cache_cleanup import CacheCleanup
from smartcash.utils.cache.cache_stats import CacheStats

# Alias untuk kompatibilitas mundur
EnhancedCache = CacheManager

__all__ = [
    'CacheManager',
    'CacheIndex',
    'CacheStorage',
    'CacheCleanup',
    'CacheStats',
    'EnhancedCache',  # Alias untuk kompatibilitas
]