"""
File: smartcash/components/cache/__init__.py
Deskripsi: Package initialization untuk cache pattern terkonsolidasi di SmartCash
"""

from smartcash.components.cache.indexing_cache import CacheIndex
from smartcash.components.cache.storage_cache import CacheStorage
from smartcash.components.cache.cleanup_cache import CacheCleanup
from smartcash.components.cache.stats_cache import CacheStats
from smartcash.components.cache.manager_cache import CacheManager

# Fungsi factory untuk menciptakan dan mengembalikan instance CacheManager
def get_cache_manager(
    cache_dir: str = ".cache/preprocessing",
    max_size_gb: float = 1.0,
    ttl_hours: int = 24,
    auto_cleanup: bool = True,
    cleanup_interval_mins: int = 30,
    logger = None
) -> CacheManager:
    """
    Dapatkan instance CacheManager.
    
    Args:
        cache_dir: Path direktori cache
        max_size_gb: Ukuran maksimum cache dalam GB
        ttl_hours: Waktu hidup cache dalam jam
        auto_cleanup: Aktifkan pembersihan otomatis
        cleanup_interval_mins: Interval pembersihan dalam menit
        logger: Logger kustom (opsional)
        
    Returns:
        Instance CacheManager
    """
    return CacheManager(
        cache_dir=cache_dir,
        max_size_gb=max_size_gb,
        ttl_hours=ttl_hours,
        auto_cleanup=auto_cleanup,
        cleanup_interval_mins=cleanup_interval_mins,
        logger=logger
    )

__all__ = [
    'CacheManager', 
    'CacheIndex', 
    'CacheStorage', 
    'CacheCleanup', 
    'CacheStats',
    'get_cache_manager'
]