# File: smartcash/handlers/evaluation/observers/base_observer.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk observer pattern pada pipeline evaluasi

from typing import Dict, Any
from abc import ABC, abstractmethod

class BaseObserver(ABC):
    """
    Kelas dasar untuk observer pattern.
    Memungkinkan monitoring proses evaluasi tanpa mengubah logika inti.
    """
    
    def __init__(self, name: str = "BaseObserver"):
        """
        Inisialisasi observer.
        
        Args:
            name: Nama observer
        """
        self.name = name
    
    @abstractmethod
    def update(self, event: str, data: Dict[str, Any] = None):
        """
        Metode yang dipanggil ketika event terjadi.
        
        Args:
            event: Nama event
            data: Data tambahan (opsional)
        """
        pass