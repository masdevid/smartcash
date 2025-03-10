# File: smartcash/handlers/dataset/facades/dataset_explorer_facade.py
# Deskripsi: Facade untuk eksplorasi dan analisis dataset

from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor
import time
import concurrent

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset import EnhancedDatasetValidator, DatasetAnalyzer
from smartcash.utils.dataset.dataset_utils import DatasetUtils


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
        
        # Inisialisasi komponen
        self.validator = EnhancedDatasetValidator(config, self.data_dir, logger)
        self.analyzer = DatasetAnalyzer(config, self.data_dir, logger)
        self.utils = DatasetUtils(config, self.data_dir, logger)
        
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
        
        # Gunakan analyzer dari utils/dataset
        result = self.analyzer.analyze_dataset(split, sample_size, detailed)
        
        # Tambahkan waktu eksekusi
        result['execution_time'] = time.time() - start_time
        
        self.logger.success(f"✅ Analisis '{split}' selesai: {result['execution_time']:.2f}s")
        return result
    
    def get_split_statistics(self) -> Dict[str, Dict[str, int]]:
        """Dapatkan statistik dasar untuk semua split dataset."""
        return self.utils.get_split_statistics()
    
    def get_class_statistics(self, split: str) -> Dict[str, Any]:
        """Dapatkan statistik kelas untuk split tertentu."""
        # Validasi dataset untuk dapatkan statistik kelas
        validation_results = self.validator.validate_dataset(
            split=split, 
            fix_issues=False, 
            move_invalid=False, 
            visualize=False
        )
        
        if 'class_stats' in validation_results:
            return {'class_stats': validation_results['class_stats']}
        
        # Fallback: gunakan analyzer untuk dapatkan statistik kelas
        analysis = self.analyzer.analyze_dataset(split, sample_size=0, detailed=False)
        return analysis.get('class_balance', {})
    
    def get_layer_statistics(self, split: str) -> Dict[str, Any]:
        """Dapatkan statistik layer untuk split tertentu."""
        # Validasi dataset untuk dapatkan statistik layer
        validation_results = self.validator.validate_dataset(
            split=split, 
            fix_issues=False, 
            move_invalid=False, 
            visualize=False
        )
        
        if 'layer_stats' in validation_results:
            return {'layer_stats': validation_results['layer_stats']}
        
        # Fallback: gunakan analyzer untuk dapatkan statistik layer
        analysis = self.analyzer.analyze_dataset(split, sample_size=0, detailed=False)
        return analysis.get('layer_balance', {})
    
    def get_dataset_sizes(self, split: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Dapatkan ukuran gambar dalam dataset."""
        splits = [split] if split else ['train', 'valid', 'test']
        sizes = {}
        
        for current_split in splits:
            sizes[current_split] = self.analyzer.analyze_image_sizes(current_split)
            
        return sizes
    
    # Metode khusus per analisis
    
    def analyze_class_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """Analisis khusus untuk distribusi kelas."""
        analysis = self.analyzer.analyze_dataset(split, sample_size, detailed=False)
        return analysis.get('class_balance', {})
    
    def analyze_layer_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """Analisis khusus untuk distribusi layer."""
        analysis = self.analyzer.analyze_dataset(split, sample_size, detailed=False)
        return analysis.get('layer_balance', {})
    
    def analyze_image_sizes(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """Analisis khusus untuk ukuran gambar."""
        return self.analyzer.analyze_image_sizes(split, sample_size)
    
    def analyze_bounding_boxes(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """Analisis khusus untuk bounding box."""
        analysis = self.analyzer.analyze_dataset(split, sample_size, detailed=True)
        return analysis.get('bbox_statistics', {})