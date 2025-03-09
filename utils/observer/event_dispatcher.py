# File: smartcash/utils/observer/event_dispatcher.py
# Author: Alfrida Sabar
# Deskripsi: Dispatcher untuk event observer di SmartCash

import threading
import concurrent.futures
import traceback
import time
from typing import Any, Dict, List, Optional, Set, Union, Callable

from smartcash.utils.observer.base_observer import BaseObserver
from smartcash.utils.observer.event_registry import EventRegistry
from smartcash.utils.logger import get_logger


class EventDispatcher:
    """
    Dispatcher untuk event observer di SmartCash.
    
    EventDispatcher bertanggung jawab untuk mendaftarkan observer ke registry
    dan mengirimkan notifikasi ke observer saat event terjadi. Implementasi ini
    mendukung notifikasi sinkron dan asinkron, manajemen error, dan prioritas observer.
    """
    
    # Logger
    _logger = get_logger("event_dispatcher")
    
    # Lock untuk thread-safety
    _lock = threading.RLock()
    
    # Thread pool untuk eksekusi asinkron
    _thread_pool = None
    
    # Set untuk melacak observer yang sedang diproses secara asinkron
    _active_async_notifications: Set[str] = set()
    
    # Flag untuk mengaktifkan/menonaktifkan logging
    _logging_enabled = True
    
    @classmethod
    def _get_thread_pool(cls):
        """
        Lazy initialization untuk thread pool.
        
        Returns:
            ThreadPoolExecutor instance
        """
        if cls._thread_pool is None:
            with cls._lock:
                if cls._thread_pool is None:
                    cls._thread_pool = concurrent.futures.ThreadPoolExecutor(
                        max_workers=4,
                        thread_name_prefix="EventDispatcher"
                    )
        return cls._thread_pool
    
    @classmethod
    def register(cls, event_type: str, observer: BaseObserver, priority: Optional[int] = None) -> None:
        """
        Mendaftarkan observer untuk event tertentu.
        
        Args:
            event_type: Tipe event yang akan diobservasi
            observer: Observer yang akan didaftarkan
            priority: Prioritas opsional (jika None, gunakan prioritas observer)
        """
        registry = EventRegistry()
        registry.register(event_type, observer, priority)
        
        if cls._logging_enabled:
            cls._logger.debug(
                f"🔌 Mendaftarkan observer {observer.name} untuk event '{event_type}' dengan prioritas {priority or observer.priority}"
            )
    
    @classmethod
    def register_many(cls, event_types: List[str], observer: BaseObserver, priority: Optional[int] = None) -> None:
        """
        Mendaftarkan observer untuk beberapa event sekaligus.
        
        Args:
            event_types: List tipe event yang akan diobservasi
            observer: Observer yang akan didaftarkan
            priority: Prioritas opsional (jika None, gunakan prioritas observer)
        """
        for event_type in event_types:
            cls.register(event_type, observer, priority)
    
    @classmethod
    def unregister(cls, event_type: str, observer: BaseObserver) -> None:
        """
        Membatalkan registrasi observer untuk event tertentu.
        
        Args:
            event_type: Tipe event
            observer: Observer yang akan dibatalkan registrasinya
        """
        registry = EventRegistry()
        registry.unregister(event_type, observer)
        
        if cls._logging_enabled:
            cls._logger.debug(
                f"🔌 Membatalkan registrasi observer {observer.name} dari event '{event_type}'"
            )
    
    @classmethod
    def unregister_many(cls, event_types: List[str], observer: BaseObserver) -> None:
        """
        Membatalkan registrasi observer dari beberapa event sekaligus.
        
        Args:
            event_types: List tipe event
            observer: Observer yang akan dibatalkan registrasinya
        """
        for event_type in event_types:
            cls.unregister(event_type, observer)
    
    @classmethod
    def unregister_from_all(cls, observer: BaseObserver) -> None:
        """
        Membatalkan registrasi observer dari semua event.
        
        Args:
            observer: Observer yang akan dibatalkan registrasinya
        """
        registry = EventRegistry()
        registry.unregister_from_all(observer)
        
        if cls._logging_enabled:
            cls._logger.debug(
                f"🔌 Membatalkan registrasi observer {observer.name} dari semua event"
            )
    
    @classmethod
    def unregister_all(cls, event_type: Optional[str] = None) -> None:
        """
        Membatalkan registrasi semua observer untuk event tertentu atau semua event.
        
        Args:
            event_type: Tipe event yang akan dibersihkan (None untuk semua event)
        """
        registry = EventRegistry()
        registry.unregister_all(event_type)
        
        if cls._logging_enabled:
            if event_type is None:
                cls._logger.debug("🧹 Membersihkan semua observer dari registry")
            else:
                cls._logger.debug(f"🧹 Membersihkan semua observer untuk event '{event_type}'")
    
    @classmethod
    def notify(
        cls, 
        event_type: str, 
        sender: Any, 
        async_mode: bool = False,
        **kwargs
    ) -> Union[None, List[concurrent.futures.Future]]:
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
        
        # Log notification
        if cls._logging_enabled:
            cls._logger.debug(
                f"📢 Notifikasi event '{event_type}' ke {len(observers)} observer"
                f"{' (async)' if async_mode else ''}"
            )
        
        if async_mode:
            return cls._notify_async(event_type, sender, observers, **kwargs)
        else:
            cls._notify_sync(event_type, sender, observers, **kwargs)
            return None
    
    @classmethod
    def _notify_sync(cls, event_type: str, sender: Any, observers: List[BaseObserver], **kwargs) -> None:
        """
        Mengirim notifikasi secara sinkron.
        
        Args:
            event_type: Tipe event yang terjadi
            sender: Objek yang mengirim event
            observers: List observer yang akan dinotifikasi
            **kwargs: Parameter tambahan
        """
        errors = []
        
        # Update timestamp start
        timestamp_start = time.time()
        
        # Tambahkan timestamp ke kwargs jika belum ada
        if 'timestamp' not in kwargs:
            kwargs['timestamp'] = timestamp_start
        
        for observer in observers:
            try:
                observer.update(event_type, sender, **kwargs)
            except Exception as e:
                # Catat error tapi tetap lanjutkan ke observer berikutnya
                error_info = {
                    'observer': observer.name,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                errors.append(error_info)
                
                if cls._logging_enabled:
                    cls._logger.warning(
                        f"⚠️ Error pada observer {observer.name} saat memproses event '{event_type}': {str(e)}"
                    )
        
        # Log errors jika ada
        if errors and cls._logging_enabled:
            error_count = len(errors)
            cls._logger.warning(
                f"⚠️ {error_count} error terjadi saat notifikasi event '{event_type}'"
            )
    
    @classmethod
    def _notify_async(
        cls, 
        event_type: str, 
        sender: Any, 
        observers: List[BaseObserver], 
        **kwargs
    ) -> List[concurrent.futures.Future]:
        """
        Mengirim notifikasi secara asinkron menggunakan thread pool.
        
        Args:
            event_type: Tipe event yang terjadi
            sender: Objek yang mengirim event
            observers: List observer yang akan dinotifikasi
            **kwargs: Parameter tambahan
            
        Returns:
            List Future untuk operasi asinkron
        """
        thread_pool = cls._get_thread_pool()
        futures = []
        
        # Buat ID unik untuk notifikasi ini
        notification_id = f"{event_type}_{id(sender)}_{time.time()}"
        
        # Catat notifikasi aktif
        with cls._lock:
            cls._active_async_notifications.add(notification_id)
        
        # Update timestamp start
        timestamp_start = time.time()
        
        # Tambahkan timestamp ke kwargs jika belum ada
        if 'timestamp' not in kwargs:
            kwargs['timestamp'] = timestamp_start
        
        # Tambahkan notification_id ke kwargs untuk referensi
        kwargs['notification_id'] = notification_id
        
        # Fungsi untuk dijalankan secara asinkron
        def _notify_observer(observer, event_type, sender, **kwargs):
            try:
                observer.update(event_type, sender, **kwargs)
                return {'observer': observer.name, 'status': 'success'}
            except Exception as e:
                error_info = {
                    'observer': observer.name,
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                
                if cls._logging_enabled:
                    cls._logger.warning(
                        f"⚠️ Error pada observer {observer.name} saat memproses event '{event_type}' secara asinkron: {str(e)}"
                    )
                return error_info
                
        # Submit task untuk setiap observer
        for observer in observers:
            future = thread_pool.submit(_notify_observer, observer, event_type, sender, **kwargs)
            
            # Callback saat future selesai
            def _done_callback(future, observer_name=observer.name):
                result = future.result()
                if result.get('status') == 'error' and cls._logging_enabled:
                    cls._logger.debug(
                        f"🔄 Async observer {observer_name} untuk event '{event_type}' selesai dengan error"
                    )
                
                # Cek apakah semua future untuk notifikasi ini sudah selesai
                remaining = sum(1 for f in futures if not f.done())
                if remaining == 0:
                    with cls._lock:
                        if notification_id in cls._active_async_notifications:
                            cls._active_async_notifications.remove(notification_id)
            
            future.add_done_callback(_done_callback)
            futures.append(future)
        
        return futures
    
    @classmethod
    def is_async_notification_active(cls, notification_id: str) -> bool:
        """
        Memeriksa apakah notifikasi asinkron masih aktif.
        
        Args:
            notification_id: ID notifikasi
            
        Returns:
            True jika masih aktif, False jika sudah selesai
        """
        with cls._lock:
            return notification_id in cls._active_async_notifications
    
    @classmethod
    def wait_for_async_notifications(cls, timeout: Optional[float] = None) -> bool:
        """
        Menunggu semua notifikasi asinkron selesai.
        
        Args:
            timeout: Timeout dalam detik (None untuk menunggu tanpa batas waktu)
            
        Returns:
            True jika semua notifikasi selesai, False jika timeout
        """
        start_time = time.time()
        
        while True:
            with cls._lock:
                active_count = len(cls._active_async_notifications)
                
            if active_count == 0:
                return True
                
            if timeout is not None and time.time() - start_time > timeout:
                return False
                
            time.sleep(0.1)
    
    @classmethod
    def shutdown(cls) -> None:
        """
        Shutdown thread pool dan bersihkan resources.
        """
        if cls._thread_pool is not None:
            with cls._lock:
                if cls._thread_pool is not None:
                    cls._thread_pool.shutdown(wait=True)
                    cls._thread_pool = None
                    
        # Bersihkan weak references
        registry = EventRegistry()
        cleaned = registry.clean_references()
        
        if cls._logging_enabled and cleaned > 0:
            cls._logger.debug(f"🧹 Membersihkan {cleaned} weak references yang tidak valid")
    
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
        """
        Mendapatkan statistik dispatcher dan registry.
        
        Returns:
            Dictionary statistik
        """
        registry = EventRegistry()
        stats = registry.get_stats()
        
        with cls._lock:
            stats['active_async_notifications'] = len(cls._active_async_notifications)
            stats['thread_pool_active'] = cls._thread_pool is not None
            
        return stats