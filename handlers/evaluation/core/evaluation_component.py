# File: smartcash/handlers/evaluation/core/evaluation_component.py
# Author: Alfrida Sabar
# Deskripsi: Komponen dasar untuk evaluasi model

from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod

from smartcash.utils.logger import SmartCashLogger, get_logger

class EvaluationComponent(ABC):
    """
    Komponen dasar untuk evaluasi model.
    Kelas abstrak yang digunakan sebagai base class untuk semua komponen evaluasi.
    """
    
    def __init__(
        self, 
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        name: Optional[str] = None
    ):
        """
        Inisialisasi komponen evaluasi.
        
        Args:
            config: Konfigurasi evaluasi
            logger: Logger kustom (opsional)
            name: Nama komponen (opsional, gunakan nama kelas)
        """
        self.config = config
        self.name = name or self.__class__.__name__
        self.logger = logger or get_logger(self.name.lower())
    
    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Proses evaluasi untuk komponen ini.
        
        Args:
            **kwargs: Parameter proses
            
        Returns:
            Dictionary hasil proses
        """
        pass
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validasi input sebelum proses.
        
        Args:
            **kwargs: Parameter untuk divalidasi
            
        Returns:
            True jika valid
            
        Raises:
            ValueError: Jika validasi gagal
        """
        return True
    
    def notify_observers(self, observers: List, event: str, data: Dict[str, Any] = None):
        """
        Notifikasi observer tentang event.
        
        Args:
            observers: List observer
            event: Nama event
            data: Data tambahan (opsional)
        """
        if not observers:
            return
            
        for observer in observers:
            if hasattr(observer, 'update'):
                observer.update(event, data or {})