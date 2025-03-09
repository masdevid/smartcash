# File: smartcash/handlers/detection/pipeline/base_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk pipeline deteksi mata uang

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from smartcash.utils.logger import get_logger

class BasePipeline(ABC):
    """
    Kelas abstrak untuk pipeline deteksi.
    Mendefinisikan struktur utama pipeline.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        logger = None,
        name: str = "base_pipeline"
    ):
        """
        Inisialisasi pipeline.
        
        Args:
            config: Konfigurasi
            logger: Logger kustom (opsional)
            name: Nama pipeline
        """
        self.config = config
        self.logger = logger or get_logger(name)
        self.name = name
        self.observers = []
        
    def add_observer(self, observer):
        """Tambahkan observer untuk monitoring."""
        self.observers.append(observer)
        return self
        
    def notify_observers(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Notifikasi semua observer tentang event."""
        for observer in self.observers:
            observer.update(event_type, data)
            
    @abstractmethod
    def run(self, *args, **kwargs):
        """Jalankan pipeline."""
        pass