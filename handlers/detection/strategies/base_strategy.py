# File: smartcash/handlers/detection/strategies/base_strategy.py
# Author: Alfrida Sabar
# Deskripsi: Strategi dasar untuk deteksi objek mata uang

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from smartcash.utils.logger import get_logger

class BaseDetectionStrategy(ABC):
    """
    Kelas abstrak untuk strategi deteksi objek mata uang.
    Menyediakan struktur dasar untuk implementasi konkrit.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        detector,
        preprocessor,
        postprocessor,
        output_manager,
        logger = None
    ):
        """
        Inisialisasi strategi deteksi.
        
        Args:
            config: Konfigurasi
            detector: Komponen detector
            preprocessor: Komponen preprocessor
            postprocessor: Komponen postprocessor
            output_manager: Komponen output manager
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.detector = detector
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.output_manager = output_manager
        self.logger = logger or get_logger(self.__class__.__name__)
        
        # List observer untuk monitoring
        self.observers = []
        
    def add_observer(self, observer):
        """
        Tambahkan observer untuk monitoring.
        
        Args:
            observer: Observer yang akan ditambahkan
            
        Returns:
            Self untuk method chaining
        """
        self.observers.append(observer)
        return self
    
    def notify_observers(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """
        Notifikasi semua observer tentang event.
        
        Args:
            event_type: Tipe event ('start', 'progress', 'complete', dll)
            data: Data terkait event (opsional)
        """
        for observer in self.observers:
            observer.update(event_type, data)
    
    @abstractmethod
    def detect(self, source, **kwargs):
        """
        Metode abstrak untuk deteksi objek.
        
        Args:
            source: Sumber gambar
            **kwargs: Parameter tambahan
            
        Returns:
            Hasil deteksi
        """
        pass