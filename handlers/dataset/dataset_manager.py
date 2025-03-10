# File: smartcash/handlers/dataset/dataset_manager.py
# Deskripsi: Manager utama dataset yang menggabungkan semua facade (compatible dengan observer pattern SmartCash)

from typing import Dict, Optional, List

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.dataset.facades.pipeline_facade import PipelineFacade
from smartcash.utils.observer import BaseObserver, EventDispatcher, EventTopics

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
        self.observers = []  # Untuk kompatibilitas dengan kode lama
    
    def register_observer(self, observer: BaseObserver) -> None:
        """
        Mendaftarkan observer untuk DatasetManager.
        
        Args:
            observer: Observer yang akan didaftarkan
        """
        if not isinstance(observer, BaseObserver):
            raise TypeError(f"Observer harus merupakan instance dari BaseObserver, bukan {type(observer)}")
        
        # Daftarkan ke EventDispatcher untuk event dataset
        EventDispatcher.register(EventTopics.PREPROCESSING, observer)
        EventDispatcher.register(EventTopics.PREPROCESSING_START, observer)
        EventDispatcher.register(EventTopics.PREPROCESSING_END, observer)
        EventDispatcher.register(EventTopics.PREPROCESSING_PROGRESS, observer)
        EventDispatcher.register(EventTopics.VALIDATION_EVENT, observer)
        EventDispatcher.register(EventTopics.AUGMENTATION_EVENT, observer)
        
        # Juga simpan di list lokal untuk backward compatibility
        if observer not in self.observers:
            self.observers.append(observer)
            self.logger.debug(f"✅ Observer {observer.name} berhasil didaftarkan")
    
    def unregister_observer(self, observer: BaseObserver) -> None:
        """
        Membatalkan pendaftaran observer.
        
        Args:
            observer: Observer yang akan dibatalkan pendaftarannya
        """
        # Batalkan pendaftaran dari EventDispatcher
        EventDispatcher.unregister_from_all(observer)
        
        # Hapus dari list lokal
        if observer in self.observers:
            self.observers.remove(observer)
            self.logger.debug(f"🗑️ Observer {observer.name} berhasil dihapus")
    
    def get_observers(self) -> List[BaseObserver]:
        """
        Mendapatkan daftar observer yang terdaftar.
        
        Returns:
            List observer yang terdaftar
        """
        return self.observers.copy()