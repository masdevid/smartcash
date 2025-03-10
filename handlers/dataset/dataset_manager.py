# File: smartcash/handlers/dataset/dataset_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager utama dataset yang menggabungkan semua facade

from typing import Dict, Optional

from smartcash.utils.logger import SmartCashLogger
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