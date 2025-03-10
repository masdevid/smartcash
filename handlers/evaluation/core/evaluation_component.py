# File: smartcash/handlers/evaluation/core/evaluation_component.py
# Author: Alfrida Sabar
# Deskripsi: Komponen dasar yang disederhanakan untuk evaluasi model

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.utils.observer.event_dispatcher import EventDispatcher

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
    
    def notify(self, event: str, data: Dict[str, Any] = None):
        """
        Notify observers menggunakan EventDispatcher.
        
        Args:
            event: Nama event
            data: Data event (opsional)
        """
        EventDispatcher.notify(f"evaluation.{event}", self, data or {})