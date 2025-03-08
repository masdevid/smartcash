# File: smartcash/handlers/dataset/dataset_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager utama dataset yang menggabungkan semua facade ke dalam satu antarmuka

from typing import Dict, Optional

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.dataset.facades.pipeline_facade import PipelineFacade


class DatasetManager(PipelineFacade):
    """
    Manager utama untuk dataset SmartCash yang menyediakan antarmuka terpadu
    untuk semua operasi dan pipeline terkait dataset.
    
    Menggunakan pola composite facade yang menggabungkan:
    - DataLoadingFacade: Operasi loading data dan pembuatan dataloader
    - DataProcessingFacade: Operasi validasi, augmentasi, dan balancing
    - DataOperationsFacade: Operasi manipulasi dataset seperti split dan merge
    - VisualizationFacade: Operasi visualisasi dataset
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi DatasetManager.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori dataset (opsional)
            cache_dir: Direktori cache (opsional)
            logger: Logger kustom (opsional)
        """
        super().__init__(config, data_dir, cache_dir, logger)