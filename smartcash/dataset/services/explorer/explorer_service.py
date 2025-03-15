"""
File: smartcash/dataset/services/explorer/explorer_service.py
Deskripsi: Layanan utama untuk eksplorasi dataset sebagai koordinator
"""

from pathlib import Path
from typing import Dict, Any, Optional

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DatasetUtils


class ExplorerService:
    """Koordinator utama untuk eksplorasi dan analisis dataset."""
    
    def __init__(self, config: Dict, data_dir: str, logger=None, num_workers: int = 4):
        """
        Inisialisasi ExplorerService.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger("explorer_service")
        self.num_workers = num_workers
        
        # Setup utils
        self.utils = DatasetUtils(config, data_dir, logger)
        
        self.logger.info(f"🔍 ExplorerService diinisialisasi dengan {num_workers} workers")
    
    def analyze_class_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis distribusi kelas dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Hasil analisis distribusi kelas
        """
        from smartcash.dataset.services.explorer.class_explorer import ClassExplorer
        explorer = ClassExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
        return explorer.analyze_distribution(split, sample_size)
    
    def analyze_layer_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis distribusi layer dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Hasil analisis distribusi layer
        """
        from smartcash.dataset.services.explorer.layer_explorer import LayerExplorer
        explorer = LayerExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
        return explorer.analyze_distribution(split, sample_size)
    
    def analyze_bbox_statistics(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis statistik bounding box dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Hasil analisis statistik bbox
        """
        from smartcash.dataset.services.explorer.bbox_explorer import BBoxExplorer
        explorer = BBoxExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
        return explorer.analyze_bbox_statistics(split, sample_size)
    
    def analyze_image_sizes(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis ukuran gambar dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Hasil analisis ukuran gambar
        """
        from smartcash.dataset.services.explorer.image_explorer import ImageExplorer
        explorer = ImageExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
        return explorer.analyze_image_sizes(split, sample_size)