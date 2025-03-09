# File: smartcash/utils/observer/base_observer.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi kelas dasar untuk observer pattern di SmartCash

from abc import ABC, abstractmethod
import re
from typing import Any, Dict, Optional, List, Callable, Union, Pattern


class BaseObserver(ABC):
    """
    Kelas dasar untuk semua observer di SmartCash.
    
    Observer mendaftarkan diri untuk event tertentu dan akan menerima notifikasi
    saat event tersebut terjadi. Implementasi ini mendukung:
    - Prioritas observer (observer dengan prioritas lebih tinggi dijalankan lebih dulu)
    - Filter event berdasarkan pattern
    - Konteks tambahan dalam notifikasi
    """
    
    def __init__(
        self, 
        name: Optional[str] = None, 
        priority: int = 0,
        event_filter: Optional[Union[str, Pattern, List[str], Callable]] = None
    ):
        """
        Inisialisasi observer.
        
        Args:
            name: Nama observer, default menggunakan nama kelas
            priority: Prioritas observer (semakin tinggi nilainya, semakin tinggi prioritasnya)
            event_filter: Filter untuk event yang akan diproses:
                - Jika string, akan dicocokkan persis dengan event_type
                - Jika regex pattern, akan dicocokkan dengan pattern
                - Jika list string, event_type harus ada dalam list
                - Jika callable, akan memanggil function dengan event_type sebagai argumen
        """
        self.name = name or self.__class__.__name__
        self.priority = priority
        self._event_filter = event_filter
        self._enabled = True
    
    @property
    def is_enabled(self) -> bool:
        """Mendapatkan status enabled dari observer."""
        return self._enabled
    
    def enable(self) -> None:
        """Mengaktifkan observer untuk menerima notifikasi."""
        self._enabled = True
    
    def disable(self) -> None:
        """Menonaktifkan observer sementara tanpa menghapus dari registry."""
        self._enabled = False
    
    def should_process_event(self, event_type: str) -> bool:
        """
        Menentukan apakah observer harus memproses event tertentu berdasarkan filter.
        
        Args:
            event_type: Tipe event yang akan diproses
            
        Returns:
            True jika event harus diproses, False jika tidak
        """
        # Jika observer tidak aktif, tidak perlu memproses event
        if not self._enabled:
            return False
            
        # Jika tidak ada filter, proses semua event
        if self._event_filter is None:
            return True
            
        # Filter berdasarkan tipe filter
        if isinstance(self._event_filter, str):
            # String: cocokkan persis atau event_type dimulai dengan filter + "."
            return event_type == self._event_filter or event_type.startswith(f"{self._event_filter}.")
            
        elif isinstance(self._event_filter, Pattern):
            # Regex pattern: gunakan regex.match
            return bool(self._event_filter.match(event_type))
            
        elif isinstance(self._event_filter, list):
            # List string: cek apakah event_type ada dalam list
            for filter_item in self._event_filter:
                if event_type == filter_item or event_type.startswith(f"{filter_item}."):
                    return True
            return False
            
        elif callable(self._event_filter):
            # Callable: panggil function
            return self._event_filter(event_type)
            
        # Default jika tipe filter tidak dikenali
        return False
    
    @abstractmethod
    def update(self, event_type: str, sender: Any, **kwargs) -> None:
        """
        Metode yang dipanggil saat event terjadi.
        
        Args:
            event_type: Tipe event yang terjadi
            sender: Objek yang mengirim event
            **kwargs: Parameter tambahan yang spesifik untuk event
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} (priority={self.priority})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', priority={self.priority})"
    
    def __eq__(self, other: Any) -> bool:
        """
        Membandingkan dua observer berdasarkan id objek.
        
        Args:
            other: Observer lain yang akan dibandingkan
            
        Returns:
            True jika sama, False jika berbeda
        """
        if not isinstance(other, BaseObserver):
            return False
        return id(self) == id(other)
    
    def __lt__(self, other: 'BaseObserver') -> bool:
        """
        Membandingkan dua observer untuk pengurutan berdasarkan prioritas.
        Observer dengan prioritas lebih tinggi akan diurutkan lebih dulu.
        
        Args:
            other: Observer lain yang akan dibandingkan
            
        Returns:
            True jika self harus diurutkan sebelum other
        """
        # Nilai lebih tinggi = prioritas lebih tinggi, sehingga dibalik dalam perbandingan (<)
        return self.priority > other.priority