"""
File: smartcash/components/observer/cleanup_observer.py
Deskripsi: Modul pembersihan observer yang teregistrasi dengan integrasi atexit
"""

import atexit
import weakref
from typing import Any, Optional, List, Dict, Callable, Set

from smartcash.common.logger import get_logger
from smartcash.components.observer.event_dispatcher_observer import EventDispatcher


# Global variables untuk tracking
logger = get_logger()
_registered_cleanups = []
_registered_managers = weakref.WeakSet()


def register_observer_manager(manager: Any) -> None:
    """
    Daftarkan ObserverManager untuk dibersihkan saat aplikasi selesai.
    
    Args:
        manager: Instance ObserverManager
    """
    # Check jika manager sudah terdaftar dan daftarkan sekali cleanup
    if manager not in _registered_managers:
        _registered_managers.add(manager)
        
        # Daftarkan juga fungsi cleanup untuk saat aplikasi exit (sekali saja)
        if not _registered_cleanups:
            atexit.register(cleanup_all_observers)
            _registered_cleanups.append(True)
            logger.debug("🧹 Cleanup observer terdaftar ke atexit")


def register_cleanup_function(cleanup_func: Callable) -> None:
    """
    Daftarkan fungsi cleanup untuk dijalankan saat aplikasi selesai dengan one-liner.
    
    Args:
        cleanup_func: Fungsi cleanup yang akan dijalankan
    """
    # Daftarkan langsung ke atexit
    atexit.register(cleanup_func)
    logger.debug(f"🧹 Fungsi cleanup terdaftar: {cleanup_func.__name__ if hasattr(cleanup_func, '__name__') else 'anonymous'}")


def cleanup_observer_group(manager: Any, group: str) -> int:
    """
    Bersihkan observer dalam grup tertentu dengan penanganan error yang lebih baik.
    
    Args:
        manager: Instance ObserverManager
        group: Nama grup observer
        
    Returns:
        Jumlah observer yang dibersihkan
    """
    try:
        # Gunakan metode unregister_group manager
        if hasattr(manager, 'unregister_group'):
            count = manager.unregister_group(group)
            logger.info(f"🧹 Dibersihkan {count} observer dari grup '{group}'")
            return count
        return 0
    except Exception as e:
        logger.warning(f"⚠️ Gagal membersihkan observer grup '{group}': {str(e)}")
        return 0


def cleanup_all_observers() -> None:
    """
    Bersihkan semua observer ketika aplikasi selesai.
    Fungsi ini otomatis didaftarkan dengan atexit.
    """
    # Bersihkan semua manager yang terdaftar
    for manager in list(_registered_managers):
        try:
            if hasattr(manager, 'shutdown'):
                manager.shutdown()
            elif hasattr(manager, 'unregister_all'):
                manager.unregister_all()
        except Exception:
            # Tidak perlu log di atexit untuk menghindari error saat shutdown
            pass
    
    # Reset EventDispatcher dengan error handling
    try:
        EventDispatcher.shutdown()
    except Exception:
        pass


def register_notebook_cleanup(observer_managers: List[Any] = None,
                            cleanup_functions: List[Callable] = None,
                            auto_execute: bool = True) -> Callable:
    """
    Daftarkan fungsi cleanup untuk notebook Jupyter/Colab dengan optimasi one-liner.
    
    Args:
        observer_managers: List ObserverManager yang perlu dibersihkan
        cleanup_functions: List fungsi cleanup yang perlu dijalankan
        auto_execute: Jika True, akan terdaftar ke atexit
        
    Returns:
        Fungsi cleanup yang dapat dijalankan secara manual
    """
    # Default empty lists dengan one-liner
    managers, funcs = observer_managers or [], cleanup_functions or []
    
    # Buat fungsi cleanup dengan lambda
    def _cleanup():
        # One-liner untuk membersihkan semua manager
        [m.unregister_all() if hasattr(m, 'unregister_all') else None for m in managers]
        # One-liner untuk menjalankan semua fungsi cleanup
        [f() for f in funcs]
    
    # Daftarkan ke atexit jika diminta
    if auto_execute:
        atexit.register(_cleanup)
    
    # Daftarkan manager dengan one-liner
    [register_observer_manager(m) for m in managers]
    
    return _cleanup


def setup_cell_cleanup(ui_components: Dict[str, Any]) -> None:
    """
    Setup fungsi cleanup otomatis untuk cell Jupyter/Colab dengan one-liner.
    
    Args:
        ui_components: Dictionary komponen UI yang berisi fungsi cleanup
    """
    # Daftarkan cleanup function jika ada
    if 'cleanup' in ui_components and callable(ui_components['cleanup']):
        atexit.register(ui_components['cleanup'])
        
    # Daftarkan ObserverManager jika ada dengan one-liner
    if 'observer_manager' in ui_components and hasattr(ui_components['observer_manager'], 'unregister_all'):
        register_observer_manager(ui_components['observer_manager'])


def clean_event_observers(event_types: List[str] = None) -> int:
    """
    Bersihkan semua observer untuk event tertentu atau semua event.
    
    Args:
        event_types: List tipe event yang akan dibersihkan (None untuk semua)
        
    Returns:
        Jumlah event yang dibersihkan
    """
    try:
        # One-liner untuk unregister semua event
        if event_types is None:
            EventDispatcher.unregister_all()
            return 1
        
        # One-liner untuk unregister specific events
        [EventDispatcher.unregister_all(evt) for evt in event_types]
        return len(event_types)
    except Exception as e:
        logger.warning(f"⚠️ Error saat membersihkan observers: {str(e)}")
        return 0