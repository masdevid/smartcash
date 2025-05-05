"""
File: smartcash/components/observer/event_dispatcher_observer.py
Deskripsi: Dispatcher untuk event observer di SmartCash dengan optimasi menggunakan ThreadPoolExecutor
"""

import concurrent.futures
import traceback
import time
import weakref
from threading import RLock
from typing import Any, Dict, List, Optional, Set, Union, Callable

from smartcash.components.observer.base_observer import BaseObserver
from smartcash.components.observer.event_registry_observer import EventRegistry
from smartcash.common.logger import get_logger
import os

class EventDispatcher:
    """
    Dispatcher untuk event observer di SmartCash dengan optimasi performa.
    
    Menggunakan ThreadPoolExecutor untuk paralelisme yang lebih efisien dan
    manajemen resource yang lebih baik melalui weak references.
    """
    
    # Logger
    _logger = get_logger("event_dispatcher")
    
    # Lock untuk thread-safety
    _lock = RLock()
    
    # Thread pool untuk eksekusi asinkron
    _executor = None
    
    # Set untuk melacak notifikasi asinkron aktif dengan weak references
    _active_async_tasks = weakref.WeakValueDictionary()
    
    # Flag untuk mengaktifkan/menonaktifkan logging
    _logging_enabled = True
    
    @classmethod
    def _get_executor(cls):
        """
        Lazy initialization untuk thread pool executor.
        
        Returns:
            ThreadPoolExecutor instance
        """
        if cls._executor is None:
            with cls._lock:
                if cls._executor is None:
                    cls._executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(32, (os.cpu_count() or 4) * 2),  # 2x CPU count dengan batas 32
                        thread_name_prefix="EventDispatcher"
                    )
        return cls._executor
    
    @classmethod
    def register(cls, event_type: str, observer: Any, priority: Optional[int] = None) -> None:
        """
        Mendaftarkan observer untuk event tertentu dengan validasi tipe.
        
        Args:
            event_type: Tipe event yang akan diobservasi
            observer: Observer yang akan didaftarkan
            priority: Prioritas opsional (jika None, gunakan prioritas observer)
        """
        # Validasi observer secara eksplisit
        if not isinstance(observer, BaseObserver):
            raise TypeError(f"Observer harus merupakan instance dari BaseObserver, bukan {type(observer)}")
        
        EventRegistry().register(event_type, observer, priority)
        
        if cls._logging_enabled:
            cls._logger.debug(f"🔌 Observer {observer.name} terdaftar untuk event '{event_type}'")
    
    @classmethod
    def register_many(cls, event_types: List[str], observer: BaseObserver, priority: Optional[int] = None) -> None:
        """Mendaftarkan observer untuk beberapa event sekaligus dengan one-liner."""
        [cls.register(event_type, observer, priority) for event_type in event_types]
    
    @classmethod
    def unregister(cls, event_type: str, observer: BaseObserver) -> None:
        """Membatalkan registrasi observer untuk event tertentu."""
        EventRegistry().unregister(event_type, observer)
        
        if cls._logging_enabled:
            cls._logger.debug(f"🔌 Membatalkan registrasi observer {observer.name} dari event '{event_type}'")
    
    @classmethod
    def unregister_many(cls, event_types: List[str], observer: BaseObserver) -> None:
        """Membatalkan registrasi observer dari beberapa event sekaligus dengan one-liner."""
        [cls.unregister(event_type, observer) for event_type in event_types]
    
    @classmethod
    def unregister_from_all(cls, observer: BaseObserver) -> None:
        """Membatalkan registrasi observer dari semua event."""
        EventRegistry().unregister_from_all(observer)
        
        if cls._logging_enabled:
            cls._logger.debug(f"🔌 Observer {observer.name} dibatalkan dari semua event")
    
    @classmethod
    def unregister_all(cls, event_type: Optional[str] = None) -> None:
        """Membatalkan registrasi semua observer untuk event tertentu atau semua event."""
        EventRegistry().unregister_all(event_type)
        
        if cls._logging_enabled:
            message = "🧹 Semua observer dibersihkan" + (f" dari event '{event_type}'" if event_type else "")
            cls._logger.debug(message)
    
    @classmethod
    def notify(cls, event_type: str, sender: Any, async_mode: bool = False, **kwargs) -> Optional[List[concurrent.futures.Future]]:
        """
        Mengirim notifikasi ke semua observer yang terdaftar untuk event tertentu.
        
        Args:
            event_type: Tipe event yang terjadi
            sender: Objek yang mengirim event
            async_mode: True untuk menjalankan notifikasi secara asinkron
            **kwargs: Parameter tambahan yang spesifik untuk event
            
        Returns:
            None jika sinkron, list Future jika asinkron
        """
        registry = EventRegistry()
        registry.update_notify_stats()
        
        # Dapatkan semua observer untuk event ini
        observers = registry.get_observers(event_type)
        
        if not observers:
            return None if not async_mode else []
        
        # Tambahkan timestamp ke kwargs jika belum ada
        if 'timestamp' not in kwargs:
            kwargs['timestamp'] = time.time()
        
        # Notifikasi asinkron atau sinkron berdasarkan mode
        return cls._notify_async(event_type, sender, observers, **kwargs) if async_mode else cls._notify_sync(event_type, sender, observers, **kwargs)
    
    @classmethod
    def _notify_sync(cls, event_type: str, sender: Any, observers: List[BaseObserver], **kwargs) -> None:
        """Mengirim notifikasi secara sinkron dengan pencatatan error yang lebih baik."""
        errors = []
        
        for observer in observers:
            try:
                observer.update(event_type, sender, **kwargs)
            except Exception as e:
                # Catat error tapi tetap lanjutkan ke observer berikutnya
                error_info = {'observer': observer.name, 'error': str(e), 'traceback': traceback.format_exc()}
                errors.append(error_info)
                
                if cls._logging_enabled:
                    cls._logger.warning(f"⚠️ Error pada {observer.name}: {str(e)}")
        
        # Log ringkasan errors jika ada
        if errors and cls._logging_enabled:
            cls._logger.warning(f"⚠️ {len(errors)} error saat notifikasi event '{event_type}'")
        
        return None
    
    @classmethod
    def _notify_async(cls, event_type: str, sender: Any, observers: List[BaseObserver], **kwargs) -> List[concurrent.futures.Future]:
        """
        Mengirim notifikasi secara asinkron menggunakan ThreadPoolExecutor.
        
        Args:
            event_type: Tipe event yang terjadi
            sender: Objek yang mengirim event
            observers: List observer yang akan dinotifikasi
            **kwargs: Parameter tambahan
            
        Returns:
            List Future untuk operasi asinkron
        """
        # Buat ID unik untuk notifikasi ini
        notification_id = f"{event_type}_{id(sender)}_{time.time()}"
        
        # Tambahkan notification_id ke kwargs untuk referensi
        kwargs['notification_id'] = notification_id
        
        # Task class untuk menyimpan status tugas
        class AsyncTask:
            def __init__(self):
                self.complete = False
                self.futures = []
        
        # Buat objek task dan simpan sebagai referensi lemah
        task = AsyncTask()
        with cls._lock:
            cls._active_async_tasks[notification_id] = task
        
        # Fungsi untuk menjalankan update observer secara aman
        def update_observer(observer, evt_type, sndr, kw):
            try:
                observer.update(evt_type, sndr, **kw)
                return {'observer': observer.name, 'status': 'success'}
            except Exception as e:
                error_info = {'observer': observer.name, 'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()}
                
                if cls._logging_enabled:
                    cls._logger.warning(f"⚠️ Error pada {observer.name} (async): {str(e)}")
                return error_info
        
        # Dapatkan thread pool
        executor = cls._get_executor()
        futures = []
        
        # Submit tugas untuk setiap observer
        for observer in observers:
            future = executor.submit(update_observer, observer, event_type, sender, kwargs.copy())
            
            # Callback untuk update status tugas
            def done_callback(fut, task=task):
                try:
                    result = fut.result()
                    if result.get('status') == 'error' and cls._logging_enabled:
                        cls._logger.debug(f"🔄 Async observer {result['observer']} untuk event '{event_type}' selesai dengan error")
                    
                    # Cek apakah semua future sudah selesai
                    if all(f.done() for f in task.futures):
                        task.complete = True
                except Exception:
                    pass
            
            future.add_done_callback(done_callback)
            futures.append(future)
        
        # Simpan referensi ke future dalam task
        task.futures = futures
        
        return futures
    
    @classmethod
    def is_async_notification_active(cls, notification_id: str) -> bool:
        """Memeriksa apakah notifikasi asinkron masih aktif."""
        return notification_id in cls._active_async_tasks and not cls._active_async_tasks[notification_id].complete
    
    @classmethod
    def wait_for_async_notifications(cls, timeout: Optional[float] = None) -> bool:
        """Menunggu semua notifikasi asinkron selesai dengan polling yang efisien."""
        start_time = time.time()
        
        while True:
            # Cek apakah semua notifikasi sudah selesai
            active_tasks = [task for task in cls._active_async_tasks.values() if not task.complete]
            
            if not active_tasks:
                return True
                
            if timeout is not None and time.time() - start_time > timeout:
                return False
                
            # Lebih efisien: sleep lebih lama untuk polling yang tidak terlalu sering
            time.sleep(0.2)
    
    @classmethod
    def shutdown(cls) -> None:
        """Shutdown dispatcher dan bersihkan resources."""
        # Shutdown executor jika ada
        if cls._executor is not None:
            with cls._lock:
                if cls._executor is not None:
                    cls._executor.shutdown(wait=True)
                    cls._executor = None
        
        # Bersihkan weak references
        registry = EventRegistry()
        cleaned = registry.clean_references()
        
        if cls._logging_enabled and cleaned > 0:
            cls._logger.debug(f"🧹 Dibersihkan {cleaned} referensi yang tidak valid")
    
    @classmethod
    def enable_logging(cls) -> None:
        """Mengaktifkan logging untuk dispatcher."""
        cls._logging_enabled = True
    
    @classmethod
    def disable_logging(cls) -> None:
        """Menonaktifkan logging untuk dispatcher."""
        cls._logging_enabled = False
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Mendapatkan statistik dispatcher dan registry."""
        registry = EventRegistry()
        stats = registry.get_stats()
        
        with cls._lock:
            stats['active_async_notifications'] = len(cls._active_async_tasks)
            stats['thread_pool_active'] = cls._executor is not None
            
        return stats

# Import os untuk mendapatkan CPU count
import os