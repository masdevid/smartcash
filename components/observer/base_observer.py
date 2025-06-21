"""
File: smartcash/components/observer/base_observer.py
Deskripsi: Implementasi kelas dasar untuk observer pattern di SmartCash dengan optimasi dan penerapan one-liner
"""
from abc import ABC, abstractmethod
import re
import uuid
from typing import Any, Dict, Optional, List, Callable, Union, Pattern


class BaseObserver(ABC):
    """Kelas dasar untuk semua observer di SmartCash dengan optimasi performa dan fleksibilitas."""
    
    def __init__(
        self, 
        name: Optional[str] = None, 
        priority: int = 0,
        event_filter: Optional[Union[str, Pattern, List[str], Callable]] = None,
        enabled: bool = True
    ):
        """Inisialisasi observer dengan parameter yang disederhanakan."""
        self.name = name or self.__class__.__name__
        self.priority = priority
        self._event_filter = event_filter
        self._enabled = enabled
        self._id = str(uuid.uuid4())  # ID unik untuk hashing
    
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
        """Menentukan apakah observer harus memproses event tertentu berdasarkan filter dengan one-liner."""
        # One-liner untuk pengecekan dasar
        if not self._enabled or self._event_filter is None: return self._enabled
        
        # One-liner untuk setiap tipe filter
        return (event_type == self._event_filter or event_type.startswith(f"{self._event_filter}.")) if isinstance(self._event_filter, str) else \
               bool(self._event_filter.match(event_type)) if isinstance(self._event_filter, Pattern) else \
               any(event_type == item or event_type.startswith(f"{item}.") for item in self._event_filter) if isinstance(self._event_filter, list) else \
               self._event_filter(event_type) if callable(self._event_filter) else False
    
    @abstractmethod
    def update(self, event_type: str, sender: Any, **kwargs) -> None:
        """Metode yang dipanggil saat event terjadi."""
        pass
    
    def __str__(self) -> str:
        """Representasi string observer."""
        return f"{self.name} (priority={self.priority})"
    
    def __repr__(self) -> str:
        """Representasi debug observer."""
        return f"{self.__class__.__name__}(name='{self.name}', priority={self.priority})"
    
    def __eq__(self, other: Any) -> bool:
        """Membandingkan dua observer berdasarkan ID unik."""
        return isinstance(other, BaseObserver) and self._id == other._id
    
    def __hash__(self) -> int:
        """Mengembalikan hash berdasarkan ID unik observer."""
        return hash(self._id)
    
    def __lt__(self, other: 'BaseObserver') -> bool:
        """Membandingkan dua observer untuk pengurutan berdasarkan prioritas (perhatikan: prioritas lebih tinggi = nilai lebih rendah)."""
        return self.priority > other.priority