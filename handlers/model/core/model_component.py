# File: smartcash/handlers/model/core/model_component.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar komponen model yang diringkas, menyediakan fungsionalitas umum

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

from smartcash.utils.logger import get_logger, SmartCashLogger

class ModelComponent(ABC):
    """
    Kelas dasar untuk semua komponen model.
    Menyediakan fungsionalitas umum seperti logging dan konfigurasi.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        name: str = "model_component"
    ):
        """
        Inisialisasi model component.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom (opsional)
            name: Nama komponen untuk logging
        """
        self.config = config
        self.logger = logger or get_logger(f"model.{name}")
        self.name = name
        
        # Inisialisasi internal komponen
        self._initialize()
    
    def _initialize(self) -> None:
        """
        Inisialisasi internal komponen.
        Override oleh subclass untuk inisialisasi kustom.
        """
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Proses utama komponen.
        
        Args:
            *args: Argumen untuk proses
            **kwargs: Keyword argumen untuk proses
            
        Returns:
            Hasil proses
        """
        pass