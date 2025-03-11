# File: smartcash/handlers/dataset/dataset_manager.py
# Deskripsi: Manager utama dataset yang menggabungkan semua facade dan terintegrasi dengan ObserverManager

from typing import Dict, Optional, Any, List

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.observer import EventTopics
from smartcash.utils.observer.observer_manager import ObserverManager
from smartcash.handlers.dataset.facades.pipeline_facade import PipelineFacade


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
        
        # Inisialisasi ObserverManager
        self.observer_manager = ObserverManager(auto_register=True)
        
        # Setup observer untuk monitoring
        self._setup_observers()
    
    def _setup_observers(self):
        """Setup observer untuk monitoring operasi dataset."""
        # Observer untuk logging event preprocessing
        self.observer_manager.create_logging_observer(
            event_types=[
                EventTopics.PREPROCESSING_START,
                EventTopics.PREPROCESSING_END
            ],
            log_level="info",
            name="PreprocessingLogger",
            format_string="🔄 {event_type}: {action}",
            group="dataset_monitoring"
        )
        
        # Observer untuk logging event validasi
        self.observer_manager.create_logging_observer(
            event_types=[EventTopics.VALIDATION_EVENT],
            log_level="info",
            name="ValidationLogger",
            format_string="🔍 Validasi: {action} pada split {split}",
            group="dataset_monitoring"
        )
        
        # Observer untuk logging event augmentasi
        self.observer_manager.create_logging_observer(
            event_types=[EventTopics.AUGMENTATION_EVENT],
            log_level="info",
            name="AugmentationLogger",
            format_string="🎨 Augmentasi: {action}",
            group="dataset_monitoring"
        )
        
        # Observer untuk UI update
        self.observer_manager.create_logging_observer(
            event_types=[EventTopics.UI_UPDATE],
            log_level="debug",
            name="UILogger",
            format_string="🖼️ UI: {action}",
            group="dataset_ui"
        )
        
        # Observer untuk download events
        self.observer_manager.create_logging_observer(
            event_types=[
                EventTopics.DOWNLOAD_START,
                EventTopics.DOWNLOAD_END,
                EventTopics.DOWNLOAD_PROGRESS,
                EventTopics.DOWNLOAD_ERROR,
                EventTopics.DOWNLOAD_COMPLETE
            ],
            log_level="info",
            name="DownloadLogger",
            format_string="📥 {event_type}: {message}",
            group="dataset_download"
        )
    
    def setup_progress_observer(self, total: int, desc: str = "Dataset Processing") -> None:
        """
        Setup observer untuk monitoring progress dengan tqdm.
        
        Args:
            total: Total langkah untuk progress bar
            desc: Deskripsi progress
        """
        self.observer_manager.create_progress_observer(
            event_types=[EventTopics.PREPROCESSING_PROGRESS],
            total=total,
            desc=desc,
            name="DatasetProgressObserver",
            group="dataset_progress"
        )
    
    def setup_download_progress_observer(self, total: int, desc: str = "Downloading Dataset") -> None:
        """
        Setup observer untuk monitoring progress download dengan tqdm.
        
        Args:
            total: Total langkah untuk progress bar
            desc: Deskripsi progress
        """
        self.observer_manager.create_progress_observer(
            event_types=[EventTopics.DOWNLOAD_PROGRESS],
            total=total,
            desc=desc,
            name="DownloadProgressObserver",
            group="dataset_download"
        )
    
    def unregister_observers(self, group: Optional[str] = None) -> int:
        """
        Membatalkan registrasi observer dataset.
        
        Args:
            group: Nama grup observer yang akan dibersihkan, 
                  jika None maka semua observer akan dibersihkan
                  
        Returns:
            Jumlah observer yang dibersihkan
        """
        if group:
            count = self.observer_manager.unregister_group(group)
            return count
        else:
            count = self.observer_manager.unregister_all()
            return count
    
    def get_observer_stats(self) -> Dict[str, Any]:
        """
        Mendapatkan statistik observer.
        
        Returns:
            Dictionary statistik
        """
        return self.observer_manager.get_stats()
    
    def get_observer_groups(self) -> List[str]:
        """
        Mendapatkan daftar grup observer yang terdaftar.
        
        Returns:
            List nama grup
        """
        stats = self.observer_manager.get_stats()
        return list(stats.get('groups', {}).keys())
    
    def __del__(self):
        """Cleanup saat instance dihapus."""
        try:
            self.unregister_observers()
        except Exception:
            # Tidak perlu log di destructor untuk menghindari error saat Python shutdown
            pass