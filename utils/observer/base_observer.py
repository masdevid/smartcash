# File: smartcash/utils/observer/base_observer.py
# Deskripsi: Implementasi kelas dasar untuk observer pattern di SmartCash

from abc import ABC, abstractmethod
import re
import uuid
from typing import Any, Dict, Optional, List, Callable, Union, Pattern


class BaseObserver(ABC):
    """Kelas dasar untuk semua observer di SmartCash."""
    
    def __init__(
        self, 
        name: Optional[str] = None, 
        priority: int = 0,
        event_filter: Optional[Union[str, Pattern, List[str], Callable]] = None
    ):
        """Inisialisasi observer."""
        self.name = name or self.__class__.__name__
        self.priority = priority
        self._event_filter = event_filter
        self._enabled = True
        # Tambahkan ID unik untuk hashing
        self._id = str(uuid.uuid4())
    
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
        """Menentukan apakah observer harus memproses event tertentu berdasarkan filter."""
        if not self._enabled:
            return False
        if self._event_filter is None:
            return True
        
        if isinstance(self._event_filter, str):
            return event_type == self._event_filter or event_type.startswith(f"{self._event_filter}.")
        elif isinstance(self._event_filter, Pattern):
            return bool(self._event_filter.match(event_type))
        elif isinstance(self._event_filter, list):
            return any(event_type == item or event_type.startswith(f"{item}.") for item in self._event_filter)
        elif callable(self._event_filter):
            return self._event_filter(event_type)
        return False
    
    @abstractmethod
    def update(self, event_type: str, sender: Any, **kwargs) -> None:
        """Metode yang dipanggil saat event terjadi."""
        pass
    
    def __str__(self) -> str:
        return f"{self.name} (priority={self.priority})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', priority={self.priority})"
    
    def __eq__(self, other: Any) -> bool:
        """Membandingkan dua observer berdasarkan ID unik."""
        if not isinstance(other, BaseObserver):
            return False
        return self._id == other._id
    
    def __hash__(self) -> int:
        """Mengembalikan hash berdasarkan ID unik observer."""
        return hash(self._id)
    
    def __lt__(self, other: 'BaseObserver') -> bool:
        """Membandingkan dua observer untuk pengurutan berdasarkan prioritas."""
        return self.priority > other.priority