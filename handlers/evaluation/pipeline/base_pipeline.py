# File: smartcash/handlers/evaluation/pipeline/base_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk pipeline evaluasi yang diringkas

import time
from typing import Dict, List, Optional, Any, Union, Callable
from abc import ABC, abstractmethod

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.utils.observer.event_dispatcher import EventDispatcher

class BasePipeline(ABC):
    """
    Pipeline dasar sebagai template untuk semua jenis pipeline evaluasi.
    Menyediakan struktur dasar pengelolaan observer dan eksekusi pipeline.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        name: str = "BasePipeline"
    ):
        """
        Inisialisasi pipeline dasar.
        
        Args:
            config: Konfigurasi evaluasi
            logger: Logger kustom (opsional)
            name: Nama pipeline
        """
        self.config = config
        self.name = name
        self.logger = logger or get_logger(name.lower())
        self.logger.debug(f"🔧 {name} diinisialisasi")
    
    def execute_with_timing(self, func: Callable, **kwargs) -> tuple:
        """
        Eksekusi fungsi dengan pengukuran waktu dan error handling.
        
        Args:
            func: Fungsi yang akan dieksekusi
            **kwargs: Parameter untuk fungsi
            
        Returns:
            Tuple (hasil eksekusi, waktu eksekusi)
        """
        start_time = time.time()
        
        try:
            # Eksekusi fungsi
            result = func(**kwargs)
            
            # Tambahkan waktu eksekusi
            execution_time = time.time() - start_time
            if isinstance(result, dict):
                result['execution_time'] = execution_time
            
            return result, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Eksekusi gagal: {str(e)}")
            raise
    
    def get_service(self, service, service_class: Callable, **kwargs):
        """
        Dapatkan service, buat baru jika None.
        
        Args:
            service: Service instance (bisa None)
            service_class: Class untuk membuat service
            **kwargs: Parameter tambahan
            
        Returns:
            Service yang siap digunakan
        """
        return service if service is not None else service_class(self.config, self.logger, **kwargs)
    
    def notify(self, event: str, data: Dict[str, Any] = None):
        """
        Notify observers dengan EventDispatcher.
        
        Args:
            event: Nama event
            data: Data event
        """
        EventDispatcher.notify(f"evaluation.{event}", self, data or {})
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Jalankan pipeline. Harus diimplementasikan oleh subclass.
        
        Args:
            **kwargs: Parameter eksekusi
            
        Returns:
            Hasil eksekusi
        """
        pass