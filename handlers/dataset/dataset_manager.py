# File: smartcash/handlers/dataset/dataset_manager.py
# Deskripsi: Manager utama dataset dengan adapter untuk kompatibilitas observer

from typing import Dict, Optional, List, Any, Union

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.dataset.facades.pipeline_facade import PipelineFacade
from smartcash.utils.observer import BaseObserver, EventDispatcher
from smartcash.utils.observer.compatibility_observer import CompatibilityObserver


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
        self.observers = []
    
    def register_observer(self, observer: Any) -> None:
        """
        Mendaftarkan observer untuk DatasetManager.
        
        Args:
            observer: Observer yang akan didaftarkan
        """
        # Periksa apakah observer sudah merupakan BaseObserver atau perlu dibungkus
        if not isinstance(observer, BaseObserver):
            try:
                # Coba bungkus dengan CompatibilityObserver
                compat_observer = CompatibilityObserver(observer)
                observer = compat_observer
                self.logger.debug(f"🔄 Membungkus observer dengan CompatibilityObserver: {compat_observer.name}")
            except Exception as e:
                raise TypeError(f"Observer harus merupakan instance dari BaseObserver atau memiliki metode yang kompatibel: {str(e)}")
        
        # Daftarkan untuk event preprocessing
        events = [
            "preprocessing",
            "preprocessing.start",
            "preprocessing.end",
            "preprocessing.progress",
            "preprocessing.validation",
            "preprocessing.augmentation"
        ]
        
        for event in events:
            EventDispatcher.register(event, observer)
        
        # Tambahkan ke list untuk tracking
        if observer not in self.observers:
            self.observers.append(observer)
            
        self.logger.debug(f"✅ Observer {observer.name} berhasil didaftarkan")
    
    def unregister_observer(self, observer: Any) -> None:
        """
        Membatalkan pendaftaran observer.
        
        Args:
            observer: Observer yang akan dibatalkan pendaftarannya
        """
        # Cari observer asli atau yang dibungkus
        target_observer = None
        for registered_observer in self.observers:
            if registered_observer == observer:
                target_observer = registered_observer
                break
            elif isinstance(registered_observer, CompatibilityObserver) and registered_observer.observer == observer:
                target_observer = registered_observer
                break
        
        if target_observer:
            # Batalkan pendaftaran
            EventDispatcher.unregister_from_all(target_observer)
            self.observers.remove(target_observer)
            self.logger.debug(f"🗑️ Observer {getattr(target_observer, 'name', 'unknown')} berhasil dihapus")
        else:
            self.logger.warning(f"⚠️ Observer tidak ditemukan, tidak dapat membatalkan pendaftaran")
    
    def get_observers(self) -> List[Any]:
        """
        Mendapatkan daftar observer yang terdaftar.
        
        Returns:
            List observer yang terdaftar
        """
        return self.observers.copy()