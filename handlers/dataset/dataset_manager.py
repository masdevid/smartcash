# File: smartcash/handlers/dataset/dataset_manager.py
# Deskripsi: Manager utama dataset yang menggabungkan semua facade

from typing import Dict, Optional, List, Any

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.dataset.facades.pipeline_facade import PipelineFacade
from smartcash.utils.observer.base_observer import BaseObserver


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
        # Validasi tipe observer
        if not isinstance(observer, BaseObserver):
            # Jika bukan instance BaseObserver, berikan pesan error yang jelas
            observer_type = type(observer).__name__
            supported_type = BaseObserver.__name__
            module_path = BaseObserver.__module__
            
            # Informasi detail untuk debugging
            error_msg = (
                f"Observer harus merupakan instance dari {supported_type}, bukan {observer_type}. "
                f"Pastikan observer mewarisi {module_path}.{supported_type}."
            )
            raise TypeError(error_msg)
        
        # Tambahkan ke daftar observer
        if observer not in self.observers:
            self.observers.append(observer)
            self.logger.debug(f"✅ Observer {observer.name} berhasil didaftarkan")
    
    def unregister_observer(self, observer: BaseObserver) -> None:
        """
        Membatalkan pendaftaran observer.
        
        Args:
            observer: Observer yang akan dibatalkan pendaftarannya
        """
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
    
    def _notify_observers(self, event: str, **kwargs) -> None:
        """
        Mengirimkan notifikasi ke semua observer.
        
        Args:
            event: Event yang terjadi
            **kwargs: Parameter tambahan
        """
        for observer in self.observers:
            try:
                # Cek apakah observer memiliki metode khusus untuk event ini
                event_method = f"on_{event}"
                if hasattr(observer, event_method) and callable(getattr(observer, event_method)):
                    getattr(observer, event_method)(**kwargs)
                # Jika tidak, gunakan metode update umum
                elif hasattr(observer, "update") and callable(getattr(observer, "update")):
                    observer.update(event, self, **kwargs)
            except Exception as e:
                self.logger.warning(f"⚠️ Error pada observer {observer.name}: {str(e)}")