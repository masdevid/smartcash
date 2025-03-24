"""
File: smartcash/components/observer/event_registry_observer.py
Deskripsi: Registry untuk event dan observer di SmartCash dengan optimasi concurrent.futures
"""

import concurrent.futures
import weakref
import time
from threading import RLock
from typing import Dict, List, Set, Any, Optional

from smartcash.components.observer.base_observer import BaseObserver


class EventRegistry:
    """
    Registry untuk event dan observer di SmartCash.
    
    EventRegistry menyimpan informasi tentang observer yang terdaftar untuk
    setiap event dan menyediakan metode untuk mendaftarkan/membatalkan registrasi observer.
    Implementasi ini thread-safe dan menggunakan weak references untuk mencegah memory leak.
    """
    
    # Singleton instance
    _instance = None
    
    # Lock untuk thread-safety
    _lock = RLock()
    
    # Dictionary untuk menyimpan observer berdasarkan event_type
    # Format: {event_type: {observer_id: (observer_weakref, priority)}}
    _observers: Dict[str, Dict[int, tuple]] = {}
    
    # Set untuk menyimpan semua tipe event yang pernah digunakan
    _event_types: Set[str] = set()
    
    # Statistik untuk monitoring
    _stats = {
        'notify_count': 0,
        'register_count': 0,
        'unregister_count': 0,
        'last_notify_time': 0,
    }
    
    # ThreadPoolExecutor untuk operasi non-critical
    _executor = None
    
    def __new__(cls):
        """
        Implementasi singleton pattern dengan lazy initialization.
        
        Returns:
            Instance EventRegistry singleton
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventRegistry, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Inisialisasi registry dengan lazy initialization."""
        # Inisialisasi executor hanya jika diperlukan
        self._executor = None
    
    def _get_executor(self):
        """
        Lazy initialization untuk thread pool executor.
        
        Returns:
            ThreadPoolExecutor instance
        """
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=4,
                        thread_name_prefix="EventRegistry"
                    )
        return self._executor
    
    def register(self, event_type: str, observer: BaseObserver, priority: Optional[int] = None) -> None:
        """
        Mendaftarkan observer untuk event tertentu.
        
        Args:
            event_type: Tipe event yang akan diobservasi
            observer: Observer yang akan didaftarkan
            priority: Prioritas opsional (jika None, gunakan prioritas observer)
        """
        # Pastikan observer adalah instance dari BaseObserver
        if not isinstance(observer, BaseObserver):
            raise TypeError(f"Observer harus merupakan instance dari BaseObserver, bukan {type(observer)}")
        
        # Gunakan prioritas observer jika tidak disebutkan
        if priority is None:
            priority = observer.priority
            
        # Daftarkan event_type
        with self._lock:
            # Tambahkan event_type ke set jika belum ada
            self._event_types.add(event_type)
            
            # Inisialisasi dictionary untuk event_type jika belum ada
            if event_type not in self._observers:
                self._observers[event_type] = {}
            
            # Simpan observer dalam weak reference untuk mencegah memory leak
            observer_ref = weakref.ref(observer)
            
            # Simpan observer dan prioritasnya
            observer_id = id(observer)
            self._observers[event_type][observer_id] = (observer_ref, priority)
            
            # Update statistik
            self._stats['register_count'] += 1
    
    def unregister(self, event_type: str, observer: BaseObserver) -> None:
        """
        Membatalkan registrasi observer untuk event tertentu.
        
        Args:
            event_type: Tipe event
            observer: Observer yang akan dibatalkan registrasinya
        """
        with self._lock:
            if event_type in self._observers:
                observer_id = id(observer)
                if observer_id in self._observers[event_type]:
                    del self._observers[event_type][observer_id]
                    self._stats['unregister_count'] += 1
                    
                    # Hapus dictionary event_type jika tidak ada observer lagi
                    if not self._observers[event_type]:
                        del self._observers[event_type]
    
    def unregister_from_all(self, observer: BaseObserver) -> None:
        """
        Membatalkan registrasi observer dari semua event.
        
        Args:
            observer: Observer yang akan dibatalkan registrasinya
        """
        observer_id = id(observer)
        with self._lock:
            for event_type in list(self._observers.keys()):
                if observer_id in self._observers[event_type]:
                    del self._observers[event_type][observer_id]
                    self._stats['unregister_count'] += 1
                    
                    # Hapus dictionary event_type jika tidak ada observer lagi
                    if not self._observers[event_type]:
                        del self._observers[event_type]
    
    def unregister_all(self, event_type: Optional[str] = None) -> None:
        """
        Membatalkan registrasi semua observer untuk event tertentu atau semua event.
        
        Args:
            event_type: Tipe event yang akan dibersihkan (None untuk semua)
        """
        with self._lock:
            if event_type is None:
                # Hapus semua observer untuk semua event
                self._observers.clear()
            elif event_type in self._observers:
                # Hapus semua observer untuk event tertentu
                del self._observers[event_type]
    
    def get_observers(self, event_type: str) -> List[BaseObserver]:
        """
        Mendapatkan semua observer yang valid untuk event tertentu,
        diurutkan berdasarkan prioritas.
        
        Args:
            event_type: Tipe event
            
        Returns:
            List observer yang diurutkan berdasarkan prioritas (tinggi ke rendah)
        """
        result = []
        
        # Track observer IDs yang sudah diperiksa untuk mencegah duplikasi
        processed_ids = set()
        
        with self._lock:
            # One-liner untuk mendapatkan observers dari event_type yang cocok persis
            if event_type in self._observers:
                result.extend([(observer_ref(), priority) for observer_id, (observer_ref, priority) in self._observers[event_type].items()
                             if (observer := observer_ref()) is not None 
                             and observer.is_enabled 
                             and observer.should_process_event(event_type)
                             and not processed_ids.add(id(observer))])
            
            # Menangani hierarki event dengan one-liner untuk setiap level
            parts = event_type.split('.')
            for i in range(len(parts) - 1, 0, -1):
                parent_event = '.'.join(parts[:i])
                if parent_event in self._observers:
                    result.extend([(observer_ref(), priority) for observer_id, (observer_ref, priority) in self._observers[parent_event].items()
                                 if observer_id not in processed_ids
                                 and (observer := observer_ref()) is not None 
                                 and observer.is_enabled 
                                 and observer.should_process_event(event_type)
                                 and not processed_ids.add(observer_id)])
        
        # Urutkan berdasarkan prioritas dan kembalikan hanya observer dengan one-liner
        return [obs for obs, _ in sorted(result, key=lambda x: x[1], reverse=True)]
    
    def get_all_event_types(self) -> List[str]:
        """
        Mendapatkan semua tipe event yang telah didaftarkan.
        
        Returns:
            List tipe event
        """
        with self._lock:
            return list(self._event_types)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Mendapatkan statistik registry dengan one-liner.
        
        Returns:
            Dictionary statistik
        """
        with self._lock:
            stats = dict(self._stats)
            stats['observer_count'] = sum(len(observers) for observers in self._observers.values())
            stats['event_type_count'] = len(self._event_types)
            return stats
    
    def update_notify_stats(self) -> None:
        """Update statistik notifikasi."""
        with self._lock:
            self._stats['notify_count'] += 1
            self._stats['last_notify_time'] = time.time()
    
    def clean_references(self) -> int:
        """
        Membersihkan weak references yang tidak valid lagi.
        
        Returns:
            Jumlah referensi yang dibersihkan
        """
        cleaned_count = 0
        with self._lock:
            for event_type in list(self._observers.keys()):
                invalid_refs = [observer_id for observer_id, (observer_ref, _) in self._observers[event_type].items() if observer_ref() is None]
                for observer_id in invalid_refs:
                    del self._observers[event_type][observer_id]
                    cleaned_count += 1
                
                # Hapus dictionary event_type jika tidak ada observer lagi
                if not self._observers[event_type]:
                    del self._observers[event_type]
        
        return cleaned_count
    
    def clean_references_async(self) -> concurrent.futures.Future:
        """
        Membersihkan weak references secara asinkron.
        
        Returns:
            Future yang mewakili hasil pembersihan
        """
        executor = self._get_executor()
        return executor.submit(self.clean_references)
    
    def shutdown(self) -> None:
        """Shutdown executor dan bersihkan resources."""
        if self._executor is not None:
            with self._lock:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                    self._executor = None