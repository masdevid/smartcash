# File: smartcash/components/observer/event_registry.py
# Deskripsi: Registry untuk event dan observer di SmartCash

import threading
import weakref
from typing import Dict, List, Set, Any, Optional
import time

from .base_observer import BaseObserver


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
    _lock = threading.RLock()
    
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
    
    def __new__(cls):
        """
        Implementasi singleton pattern.
        
        Returns:
            Instance EventRegistry singleton
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EventRegistry, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, event_type: str, observer: BaseObserver, priority: Optional[int] = None) -> None:
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
        with cls._lock:
            # Tambahkan event_type ke set jika belum ada
            cls._event_types.add(event_type)
            
            # Inisialisasi dictionary untuk event_type jika belum ada
            if event_type not in cls._observers:
                cls._observers[event_type] = {}
            
            # Simpan observer dalam weak reference untuk mencegah memory leak
            observer_ref = weakref.ref(observer)
            
            # Simpan observer dan prioritasnya
            observer_id = id(observer)
            cls._observers[event_type][observer_id] = (observer_ref, priority)
            
            # Update statistik
            cls._stats['register_count'] += 1
    
    @classmethod
    def unregister(cls, event_type: str, observer: BaseObserver) -> None:
        """
        Membatalkan registrasi observer untuk event tertentu.
        
        Args:
            event_type: Tipe event
            observer: Observer yang akan dibatalkan registrasinya
        """
        with cls._lock:
            if event_type in cls._observers:
                observer_id = id(observer)
                if observer_id in cls._observers[event_type]:
                    del cls._observers[event_type][observer_id]
                    cls._stats['unregister_count'] += 1
                    
                    # Hapus dictionary event_type jika tidak ada observer lagi
                    if not cls._observers[event_type]:
                        del cls._observers[event_type]
    
    @classmethod
    def unregister_from_all(cls, observer: BaseObserver) -> None:
        """
        Membatalkan registrasi observer dari semua event.
        
        Args:
            observer: Observer yang akan dibatalkan registrasinya
        """
        observer_id = id(observer)
        with cls._lock:
            for event_type in list(cls._observers.keys()):
                if observer_id in cls._observers[event_type]:
                    del cls._observers[event_type][observer_id]
                    cls._stats['unregister_count'] += 1
                    
                    # Hapus dictionary event_type jika tidak ada observer lagi
                    if not cls._observers[event_type]:
                        del cls._observers[event_type]
    
    @classmethod
    def unregister_all(cls, event_type: Optional[str] = None) -> None:
        """
        Membatalkan registrasi semua observer untuk event tertentu atau semua event.
        
        Args:
            event_type: Tipe event yang akan dibersihkan (None untuk semua event)
        """
        with cls._lock:
            if event_type is None:
                # Hapus semua observer untuk semua event
                cls._observers.clear()
            elif event_type in cls._observers:
                # Hapus semua observer untuk event tertentu
                del cls._observers[event_type]
    
    @classmethod
    def get_observers(cls, event_type: str) -> List[BaseObserver]:
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
        
        with cls._lock:
            # Periksa event_type yang cocok secara persis
            if event_type in cls._observers:
                for observer_id, (observer_ref, priority) in cls._observers[event_type].items():
                    observer = observer_ref()
                    if observer is not None and observer.is_enabled and observer.should_process_event(event_type):
                        result.append((observer, priority))
                        processed_ids.add(observer_id)
            
            # Periksa event_type parent untuk hierarki event
            # Misalnya, untuk 'training.epoch.end', periksa juga 'training.epoch' dan 'training'
            parts = event_type.split('.')
            for i in range(len(parts) - 1, 0, -1):
                parent_event = '.'.join(parts[:i])
                if parent_event in cls._observers:
                    for observer_id, (observer_ref, priority) in cls._observers[parent_event].items():
                        # Skip jika observer sudah diproses
                        if observer_id in processed_ids:
                            continue
                            
                        observer = observer_ref()
                        if observer is not None and observer.is_enabled and observer.should_process_event(event_type):
                            result.append((observer, priority))
                            processed_ids.add(observer_id)
        
        # Urutkan berdasarkan prioritas dan kembalikan hanya observer
        result.sort(key=lambda x: x[1], reverse=True)
        return [obs for obs, _ in result]
    
    @classmethod
    def get_all_event_types(cls) -> List[str]:
        """
        Mendapatkan semua tipe event yang telah didaftarkan.
        
        Returns:
            List tipe event
        """
        with cls._lock:
            return list(cls._event_types)
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """
        Mendapatkan statistik registry.
        
        Returns:
            Dictionary statistik
        """
        with cls._lock:
            stats = cls._stats.copy()
            stats['observer_count'] = sum(len(observers) for observers in cls._observers.values())
            stats['event_type_count'] = len(cls._event_types)
            return stats
    
    @classmethod
    def update_notify_stats(cls) -> None:
        """Update statistik notifikasi."""
        with cls._lock:
            cls._stats['notify_count'] += 1
            cls._stats['last_notify_time'] = time.time()
    
    @classmethod
    def clean_references(cls) -> int:
        """
        Membersihkan weak references yang tidak valid lagi.
        
        Returns:
            Jumlah referensi yang dibersihkan
        """
        cleaned_count = 0
        with cls._lock:
            for event_type in list(cls._observers.keys()):
                for observer_id in list(cls._observers[event_type].keys()):
                    observer_ref, _ = cls._observers[event_type][observer_id]
                    if observer_ref() is None:
                        del cls._observers[event_type][observer_id]
                        cleaned_count += 1
                
                # Hapus dictionary event_type jika tidak ada observer lagi
                if not cls._observers[event_type]:
                    del cls._observers[event_type]
        
        return cleaned_count