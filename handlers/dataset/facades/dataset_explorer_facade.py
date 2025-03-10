# File: smartcash/handlers/dataset/facades/dataset_explorer_facade.py
# Author: Alfrida Sabar
# Deskripsi: Facade untuk mengakses semua explorer dataset SmartCash

from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor
import time

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.dataset.explorers.validation_explorer import ValidationExplorer
from smartcash.handlers.dataset.explorers.class_explorer import ClassExplorer
from smartcash.handlers.dataset.explorers.layer_explorer import LayerExplorer
from smartcash.handlers.dataset.explorers.image_size_explorer import ImageSizeExplorer
from smartcash.handlers.dataset.explorers.bbox_explorer import BoundingBoxExplorer

class DatasetExplorerFacade:
    """Facade untuk akses terpadu ke semua explorer dataset."""
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi DatasetExplorerFacade."""
        self.config = config
        self.data_dir = data_dir or config.get('data_dir', 'data')
        self.logger = logger or SmartCashLogger(__name__)
        
        # Inisialisasi explorers
        self.explorers = {
            'validation': ValidationExplorer(config, data_dir, logger),
            'class': ClassExplorer(config, data_dir, logger),
            'layer': LayerExplorer(config, data_dir, logger),
            'image_size': ImageSizeExplorer(config, data_dir, logger),
            'bbox': BoundingBoxExplorer(config, data_dir, logger)
        }
        
        self.logger.info(f"🧭 DatasetExplorerFacade diinisialisasi: {self.data_dir}")
    
    def analyze_dataset(
        self, 
        split: str, 
        sample_size: int = 0,
        detailed: bool = True,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """Analisis komprehensif pada split dataset."""
        start_time = time.time()
        self.logger.info(f"📊 Analisis komprehensif: {split}")
        
        # Tentukan explorer yang akan digunakan
        explorers_to_use = {
            'validation': self.explorers['validation'],
            'class_balance': self.explorers['class'],
            'layer_balance': self.explorers['layer'],
            'image_size_distribution': self.explorers['image_size']
        }
        
        # Tambahkan bbox explorer jika detailed
        if detailed:
            explorers_to_use['bbox_statistics'] = self.explorers['bbox']
        
        # Jalankan analisis
        results = {}
        
        if parallel:
            # Analisis paralel
            with ThreadPoolExecutor(max_workers=min(4, len(explorers_to_use))) as executor:
                # Submit tasks
                futures = {
                    executor.submit(explorer.explore, split, sample_size): name 
                    for name, explorer in explorers_to_use.items()
                }
                
                # Kumpulkan hasil
                for future in futures:
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        self.logger.error(f"❌ Explorer {name} error: {e}")
                        results[name] = {'error': str(e)}
        else:
            # Analisis sekuensial
            for name, explorer in explorers_to_use.items():
                try:
                    results[name] = explorer.explore(split, sample_size)
                except Exception as e:
                    self.logger.error(f"❌ Explorer {name} error: {e}")
                    results[name] = {'error': str(e)}
        
        # Tambahkan waktu eksekusi
        results['execution_time'] = time.time() - start_time
        
        self.logger.success(f"✅ Analisis '{split}' selesai: {results['execution_time']:.2f}s")
        return results
    
    def get_split_statistics(self) -> Dict[str, Dict[str, int]]:
        """Dapatkan statistik dasar untuk semua split dataset."""
        return self.explorers['validation'].get_split_statistics()
    
    def get_class_statistics(self, split: str) -> Dict[str, Any]:
        """Dapatkan statistik kelas untuk split tertentu."""
        return self.explorers['class'].get_class_statistics(split)
    
    def get_layer_statistics(self, split: str) -> Dict[str, Any]:
        """Dapatkan statistik layer untuk split tertentu."""
        return self.explorers['layer'].get_layer_statistics(split)
    
    def get_dataset_sizes(self, split: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Dapatkan ukuran gambar dalam dataset."""
        return self.explorers['image_size'].get_dataset_sizes(split)
    
    # Metode khusus per explorer
    
    def analyze_class_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """Analisis khusus untuk distribusi kelas."""
        return self.explorers['class'].explore(split, sample_size)
    
    def analyze_layer_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """Analisis khusus untuk distribusi layer."""
        return self.explorers['layer'].explore(split, sample_size)
    
    def analyze_image_sizes(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """Analisis khusus untuk ukuran gambar."""
        return self.explorers['image_size'].explore(split, sample_size)
    
    def analyze_bounding_boxes(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """Analisis khusus untuk bounding box."""
        return self.explorers['bbox'].explore(split, sample_size)