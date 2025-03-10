# File: smartcash/handlers/dataset/dataset_manager.py
# Deskripsi: Manager utama dataset yang menggabungkan semua facade dengan implementasi observer pattern yang benar

from typing import Dict, Optional, List, Any, Union

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.dataset.facades.pipeline_facade import PipelineFacade
from smartcash.utils.observer import BaseObserver, EventRegistry, EventDispatcher


class DatasetManager(PipelineFacade):
    """
    Manager utama untuk dataset SmartCash yang menyediakan antarmuka terpadu.
    
    Menggabungkan facade:
    - DataLoadingFacade: Loading data dan pembuatan dataloader
    - DataProcessingFacade: Validasi, augmentasi, dan balancing
    - DataOperationsFacade: Manipulasi dataset seperti split dan merge
    - VisualizationFacade: Visualisasi dataset
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi DatasetManager."""
        super().__init__(config, data_dir, cache_dir, logger)
        # Tidak perlu lagi menyimpan observers sebagai list, event registry akan menanganinya
    
    def register_observer(self, observer: BaseObserver) -> None:
        """
        Mendaftarkan observer untuk DatasetManager.
        
        Args:
            observer: Observer yang akan didaftarkan
        """
        if not isinstance(observer, BaseObserver):
            raise TypeError(f"Observer harus merupakan instance dari BaseObserver, bukan {type(observer)}")
        
        # Gunakan EventRegistry untuk mendaftarkan observer
        # Daftarkan untuk event preprocessing dan sub-events
        EventDispatcher.register("preprocessing", observer)
        EventDispatcher.register("preprocessing.start", observer)
        EventDispatcher.register("preprocessing.end", observer)
        EventDispatcher.register("preprocessing.progress", observer)
        EventDispatcher.register("preprocessing.validation", observer)
        EventDispatcher.register("preprocessing.augmentation", observer)
        
        self.logger.debug(f"✅ Observer {observer.name} berhasil didaftarkan")
    
    def unregister_observer(self, observer: BaseObserver) -> None:
        """
        Membatalkan pendaftaran observer.
        
        Args:
            observer: Observer yang akan dibatalkan pendaftarannya
        """
        # Gunakan EventDispatcher untuk membatalkan pendaftaran observer
        EventDispatcher.unregister_from_all(observer)
        
        self.logger.debug(f"🗑️ Observer {observer.name} berhasil dihapus dari semua event")
    
    def get_observers(self) -> List[BaseObserver]:
        """
        Mendapatkan daftar observer yang terdaftar.
        
        Returns:
            List observer yang terdaftar
        """
        # Gunakan EventRegistry untuk mendapatkan semua observer
        registry = EventRegistry()
        observers = set()
        
        # Kumpulkan observer dari semua event preprocessing
        for event_type in [
            "preprocessing",
            "preprocessing.start",
            "preprocessing.end",
            "preprocessing.progress",
            "preprocessing.validation",
            "preprocessing.augmentation"
        ]:
            event_observers = registry.get_observers(event_type)
            observers.update(event_observers)
        
        return list(observers)