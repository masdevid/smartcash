# File: smartcash/handlers/dataset/dataset_manager.py
# Deskripsi: Manager utama dataset dengan implementasi ObserverManager

from typing import Dict, Optional, List, Any

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.dataset.facades.pipeline_facade import PipelineFacade
from smartcash.utils.observer import BaseObserver
from smartcash.utils.observer.observer_manager import ObserverManager


class DatasetManager(PipelineFacade):
    """Manager utama untuk dataset SmartCash yang menyediakan antarmuka terpadu."""
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi DatasetManager."""
        super().__init__(config, data_dir, cache_dir, logger)
        
        # Inisialisasi ObserverManager untuk mengelola observer
        self.observer_manager = ObserverManager(auto_register=False)
        
        # Simpan observer terpisah untuk backward compatibility
        self.observers = []
    
    def register_observer(self, observer: BaseObserver) -> None:
        """
        Mendaftarkan observer untuk DatasetManager.
        
        Args:
            observer: Observer yang akan didaftarkan
        """
        # Validasi tipe observer
        if not isinstance(observer, BaseObserver):
            raise TypeError(f"Observer harus merupakan instance dari BaseObserver, bukan {type(observer)}")
        
        # Daftarkan observer untuk event-event preprocessing
        preprocessing_events = [
            "preprocessing",
            "preprocessing.start",
            "preprocessing.end", 
            "preprocessing.progress",
            "preprocessing.validation",
            "preprocessing.augmentation"
        ]
        
        for event in preprocessing_events:
            self.observer_manager._get_thread_pool()
            self.observer_manager.create_simple_observer(
                event_type=event,
                callback=observer.update,
                name=f"{observer.name}_proxy_{event}",
                priority=observer.priority
            )
        
        # Tambahkan ke list internal untuk backward compatibility
        if observer not in self.observers:
            self.observers.append(observer)
            
        self.logger.debug(f"✅ Observer {observer.name} berhasil didaftarkan")
    
    def unregister_observer(self, observer: BaseObserver) -> None:
        """
        Membatalkan pendaftaran observer.
        
        Args:
            observer: Observer yang akan dibatalkan pendaftarannya
        """
        # Hapus dari list internal
        if observer in self.observers:
            self.observers.remove(observer)
            
        # ObserverManager tidak memiliki mekanisme untuk menghapus observer berdasarkan
        # observer asli, jadi kita perlu menghapus semua observer yang kita buat
        self.observer_manager.unregister_all()
        
        # Daftarkan ulang semua observer yang tersisa
        for remaining_observer in self.observers:
            self.register_observer(remaining_observer)
            
        self.logger.debug(f"🗑️ Observer {observer.name} berhasil dihapus")
    
    def get_observers(self) -> List[BaseObserver]:
        """
        Mendapatkan daftar observer yang terdaftar.
        
        Returns:
            List observer yang terdaftar
        """
        return self.observers.copy()