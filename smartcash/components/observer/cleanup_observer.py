"""
File: smartcash/components/observer/cleanup_observer.py
Deskripsi: Modul pembersihan observer yang teregistrasi
"""

import atexit
import weakref
from typing import Any, Optional, List, Dict, Callable

from smartcash.common.logger import get_logger
from smartcash.components.observer.manager_observer import ObserverManager
from smartcash.components.observer.event_dispatcher_observer import EventDispatcher


logger = get_logger("observer_cleanup")
_registered_cleanups = []
_registered_managers = weakref.WeakSet()


def register_observer_manager(manager: ObserverManager) -> None:
    """
    Daftarkan ObserverManager untuk dibersihkan saat aplikasi selesai.
    
    Args:
        manager: Instance ObserverManager
    """
    if manager not in _registered_managers:
        _registered_managers.add(manager)
        
        # Daftarkan juga fungsi cleanup untuk saat aplikasi exit
        if not _registered_cleanups:
            atexit.register(cleanup_all_observers)
            _registered_cleanups.append(True)


def register_cleanup_function(cleanup_func: Callable) -> None:
    """
    Daftarkan fungsi cleanup untuk dijalankan saat aplikasi selesai.
    
    Args:
        cleanup_func: Fungsi cleanup yang akan dijalankan
    """
    # Daftarkan fungsi ke atexit
    atexit.register(cleanup_func)
    

def cleanup_observer_group(manager: ObserverManager, group: str) -> int:
    """
    Bersihkan observer dalam grup tertentu.
    
    Args:
        manager: Instance ObserverManager
        group: Nama grup observer
        
    Returns:
        Jumlah observer yang dibersihkan
    """
    try:
        count = manager.unregister_group(group)
        return count
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
            manager.unregister_all()
        except Exception:
            # Tidak perlu log di atexit
            pass
    
    # Reset EventDispatcher
    try:
        EventDispatcher.unregister_all()
    except Exception:
        pass


def register_notebook_cleanup(observer_managers: List[ObserverManager] = None,
                            cleanup_functions: List[Callable] = None,
                            auto_execute: bool = True) -> Callable:
    """
    Daftarkan fungsi cleanup untuk notebook Jupyter/Colab.
    
    Args:
        observer_managers: List ObserverManager yang perlu dibersihkan
        cleanup_functions: List fungsi cleanup yang perlu dijalankan
        auto_execute: Jika True, akan terdaftar ke atexit
        
    Returns:
        Fungsi cleanup yang dapat dijalankan secara manual
    """
    managers = observer_managers or []
    funcs = cleanup_functions or []
    
    def _cleanup():
        # Bersihkan observer manager
        for manager in managers:
            try:
                manager.unregister_all()
            except Exception as e:
                if logger:
                    logger.debug(f"⚠️ Gagal membersihkan observer manager: {str(e)}")
        
        # Jalankan fungsi cleanup
        for func in funcs:
            try:
                func()
            except Exception as e:
                if logger:
                    logger.debug(f"⚠️ Gagal menjalankan fungsi cleanup: {str(e)}")
    
    # Daftarkan ke atexit jika diperlukan
    if auto_execute:
        atexit.register(_cleanup)
    
    # Daftarkan manager
    for manager in managers:
        register_observer_manager(manager)
    
    return _cleanup


def setup_cell_cleanup(ui_components: Dict[str, Any]) -> None:
    """
    Setup fungsi cleanup otomatis untuk cell Jupyter/Colab.
    
    Args:
        ui_components: Dictionary komponen UI yang berisi fungsi cleanup
    """
    if 'cleanup' in ui_components and callable(ui_components['cleanup']):
        # Daftarkan ke atexit
        cleanup_func = ui_components['cleanup']
        atexit.register(cleanup_func)
        
    # Jika ada ObserverManager dalam ui_components
    if 'observer_manager' in ui_components and isinstance(ui_components['observer_manager'], ObserverManager):
        register_observer_manager(ui_components['observer_manager'])