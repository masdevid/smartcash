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
    """
    Facade yang menyediakan akses terpadu ke semua explorer.
    Menerapkan pola Facade untuk menyederhanakan penggunaan explorer.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi DatasetExplorerFacade.
        
        Args:
            config: Konfigurasi dataset/aplikasi
            data_dir: Direktori dataset (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.data_dir = data_dir or config.get('data_dir', 'data')
        self.logger = logger or SmartCashLogger(__name__)
        
        # Inisialisasi explorer yang diperlukan
        self.validation_explorer = ValidationExplorer(config, data_dir, logger)
        self.class_explorer = ClassExplorer(config, data_dir, logger)
        self.layer_explorer = LayerExplorer(config, data_dir, logger)
        self.image_size_explorer = ImageSizeExplorer(config, data_dir, logger)
        self.bbox_explorer = BoundingBoxExplorer(config, data_dir, logger)
        
        self.logger.info(f"🧭 DatasetExplorerFacade diinisialisasi untuk: {self.data_dir}")
    
    def analyze_dataset(
        self, 
        split: str, 
        sample_size: int = 0,
        detailed: bool = True,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Lakukan analisis komprehensif pada split dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            sample_size: Jumlah sampel yang akan dianalisis (0 untuk semua)
            detailed: Jika True, lakukan analisis yang lebih mendalam
            parallel: Jika True, jalankan analisis secara paralel
            
        Returns:
            Dict berisi hasil analisis
        """
        start_time = time.time()
        self.logger.info(f"📊 Memulai analisis komprehensif dataset: {split}")
        
        # Definsikan explorer dan nama yang akan digunakan
        explorers = {
            'validation': self.validation_explorer,
            'class_balance': self.class_explorer,
            'layer_balance': self.layer_explorer,
            'image_size_distribution': self.image_size_explorer
        }
        
        # Tambahkan bbox explorer jika detailed
        if detailed:
            explorers['bbox_statistics'] = self.bbox_explorer
        
        # Jalankan analisis setiap explorer
        analysis_results = {}
        
        if parallel:
            # Jalankan secara paralel dengan ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(4, len(explorers))) as executor:
                # Submit semua task
                future_to_name = {}
                for name, explorer in explorers.items():
                    future = executor.submit(explorer.explore, split, sample_size)
                    future_to_name[future] = name
                
                # Kumpulkan hasil
                for future in future_to_name:
                    name = future_to_name[future]
                    try:
                        result = future.result()
                        analysis_results[name] = result
                    except Exception as exc:
                        self.logger.error(f"❌ Explorer {name} mendapatkan exception: {exc}")
                        analysis_results[name] = {'error': str(exc)}
        else:
            # Jalankan secara sekuensial
            for name, explorer in explorers.items():
                try:
                    result = explorer.explore(split, sample_size)
                    analysis_results[name] = result
                except Exception as exc:
                    self.logger.error(f"❌ Explorer {name} mendapatkan exception: {exc}")
                    analysis_results[name] = {'error': str(exc)}
        
        # Waktu eksekusi
        elapsed_time = time.time() - start_time
        analysis_results['execution_time'] = elapsed_time
        
        self.logger.success(
            f"✅ Analisis dataset '{split}' selesai dalam {elapsed_time:.2f} detik\n"
            f"   • Validation: {'✓' if 'validation' in analysis_results and 'error' not in analysis_results['validation'] else '✗'}\n"
            f"   • Class Balance: {'✓' if 'class_balance' in analysis_results and 'error' not in analysis_results['class_balance'] else '✗'}\n"
            f"   • Layer Balance: {'✓' if 'layer_balance' in analysis_results and 'error' not in analysis_results['layer_balance'] else '✗'}\n"
            f"   • Image Size: {'✓' if 'image_size_distribution' in analysis_results and 'error' not in analysis_results['image_size_distribution'] else '✗'}"
        )
        
        if detailed:
            self.logger.info(f"   • Bounding Box: {'✓' if 'bbox_statistics' in analysis_results and 'error' not in analysis_results['bbox_statistics'] else '✗'}")
        
        return analysis_results
    
    def get_split_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Dapatkan statistik dasar untuk semua split dataset.
        
        Returns:
            Dict statistik per split
        """
        return self.validation_explorer.get_split_statistics()
    
    def get_class_statistics(self, split: str) -> Dict[str, Any]:
        """
        Dapatkan statistik kelas untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Dict statistik kelas
        """
        return self.class_explorer.get_class_statistics(split)
    
    def get_layer_statistics(self, split: str) -> Dict[str, Any]:
        """
        Dapatkan statistik layer untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Dict statistik layer
        """
        return self.layer_explorer.get_layer_statistics(split)
    
    def get_dataset_sizes(self, split: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Dapatkan ukuran gambar dalam dataset.
        
        Args:
            split: Split dataset tertentu (opsional, jika None semua split dianalisis)
            
        Returns:
            Dict berisi statistik ukuran untuk setiap split
        """
        return self.image_size_explorer.get_dataset_sizes(split)
    
    # Method khusus untuk explorer tertentu
    
    def analyze_class_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis khusus untuk distribusi kelas.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Dict hasil analisis kelas
        """
        return self.class_explorer.explore(split, sample_size)
    
    def analyze_layer_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis khusus untuk distribusi layer.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Dict hasil analisis layer
        """
        return self.layer_explorer.explore(split, sample_size)
    
    def analyze_image_sizes(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis khusus untuk ukuran gambar.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Dict hasil analisis ukuran gambar
        """
        return self.image_size_explorer.explore(split, sample_size)
    
    def analyze_bounding_boxes(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis khusus untuk bounding box.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Dict hasil analisis bounding box
        """
        return self.bbox_explorer.explore(split, sample_size)