# File: smartcash/handlers/evaluation/pipeline/base_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk pipeline evaluasi dengan komponen dan observer pattern

import time
from typing import Dict, List, Optional, Any, Union, Callable
from abc import ABC, abstractmethod

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.handlers.evaluation.observers.progress_observer import ProgressObserver

class BasePipeline(ABC):
    """
    Pipeline dasar sebagai template untuk semua jenis pipeline evaluasi.
    Menyediakan struktur dasar pengelolaan observer dan eksekusi pipeline.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        name: str = "BasePipeline",
        add_progress_observer: bool = True
    ):
        """
        Inisialisasi pipeline dasar.
        
        Args:
            config: Konfigurasi evaluasi
            logger: Logger kustom (opsional)
            name: Nama pipeline
            add_progress_observer: Tambahkan progress observer secara otomatis
        """
        self.config = config
        self.name = name
        self.logger = logger or get_logger(name.lower())
        
        # Komponen pipeline
        self.components = []
        
        # Observer
        self.observers = []
        
        # Tambahkan progress observer secara default jika diminta
        if add_progress_observer:
            self.add_observer(ProgressObserver())
        
        self.logger.debug(f"🔧 {name} diinisialisasi")
    
    def add_observer(self, observer) -> 'BasePipeline':
        """
        Tambahkan observer untuk monitoring pipeline.
        
        Args:
            observer: Observer
            
        Returns:
            Self untuk method chaining
        """
        self.observers.append(observer)
        self.logger.debug(f"👁️ Observer ditambahkan: {observer.__class__.__name__}")
        return self
    
    def notify_observers(self, event: str, data: Dict[str, Any] = None):
        """
        Notifikasi observer tentang event.
        
        Args:
            event: Nama event
            data: Data tambahan (opsional)
        """
        data = data or {}
        for observer in self.observers:
            if hasattr(observer, 'update'):
                observer.update(event, data)
    
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
            # Hitung waktu execution meskipun error
            execution_time = time.time() - start_time
            
            self.logger.error(f"❌ Eksekusi gagal: {str(e)}")
            raise  # Re-raise exception dengan traceback asli
    
    def get_adapter(self, adapter, default_factory: Callable, **kwargs):
        """
        Dapatkan adapter, gunakan factory jika None.
        
        Args:
            adapter: Adapter (bisa None)
            default_factory: Factory function untuk membuat adapter default
            **kwargs: Parameter tambahan untuk factory
            
        Returns:
            Adapter yang siap digunakan
        """
        return adapter or default_factory(self.config, self.logger, **kwargs)
    
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