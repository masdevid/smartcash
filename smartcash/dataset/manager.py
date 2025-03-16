"""
File: smartcash/dataset/manager.py
Deskripsi: Koordinator alur kerja dataset tingkat tinggi
"""

from pathlib import Path
from typing import Dict, Optional, Any, List, Union

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.dataset.utils.dataset_utils import DatasetUtils


class DatasetManager:
    """
    Manager utama untuk dataset yang bertindak sebagai koordinator untuk
    semua operasi dataset dengan lazy-loading service.
    """
    
    def __init__(self, config: Dict, data_dir: Optional[str] = None, logger=None):
        """
        Inisialisasi DatasetManager.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori utama data (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.logger = logger or get_logger("dataset_manager")
        
        # Inisialisasi layer config
        self.layer_config = get_layer_config()
        self.active_layers = config.get('layers', self.layer_config.get_layer_names())
        
        # Utilitas dataset
        self.utils = DatasetUtils(config, str(self.data_dir), self.logger)
        
        # Layanan di-inisialisasi secara lazy saat diperlukan
        self._services = {}
        
        self.logger.info(f"🚀 DatasetManager diinisialisasi dengan data_dir: {self.data_dir}")
    
    def get_service(self, service_name: str) -> Any:
        """
        Dapatkan service dengan lazy initialization.
        
        Args:
            service_name: Nama service yang diinginkan
            
        Returns:
            Instance dari service yang diminta
        """
        if service_name not in self._services:
            if service_name == 'loader':
                from smartcash.dataset.services.loader import DatasetLoaderService
                self._services[service_name] = DatasetLoaderService(
                    self.config, str(self.data_dir), self.logger)
                
            elif service_name == 'validator':
                from smartcash.dataset.services.validator import DatasetValidatorService
                self._services[service_name] = DatasetValidatorService(
                    self.config, str(self.data_dir), self.logger)
                
            elif service_name == 'augmentor':
                from smartcash.dataset.services.augmentor import AugmentationService
                self._services[service_name] = AugmentationService(
                    self.config, str(self.data_dir), self.logger)
                
            elif service_name == 'downloader':
                from smartcash.dataset.services.downloader import DownloadService
                self._services[service_name] = DownloadService(
                    self.config, str(self.data_dir), self.logger)
                
            elif service_name == 'explorer':
                from smartcash.dataset.services.explorer import ExplorerService
                self._services[service_name] = ExplorerService(
                    self.config, str(self.data_dir), self.logger)
                
            elif service_name == 'balancer':
                from smartcash.dataset.services.balancer import BalanceService
                self._services[service_name] = BalanceService(
                    self.config, str(self.data_dir), self.logger)
                
            elif service_name == 'reporter':
                from smartcash.dataset.services.reporter import ReportService
                self._services[service_name] = ReportService(
                    self.config, str(self.data_dir), self.logger)
                
            elif service_name == 'visualizer':
                from smartcash.dataset.visualization import DataVisualizationHelper, ReportVisualizer
                self._services[service_name] = {
                    'data_viz': DataVisualizationHelper(str(self.data_dir / 'visualizations'), self.logger),
                    'report_viz': ReportVisualizer(str(self.data_dir / 'reports'), self.logger)
                }
                
            else:
                self.logger.warning(f"⚠️ Service tidak dikenal: {service_name}")
                return None
                
        return self._services[service_name]
    
    # ===== Delegasi ke Loader Service =====
    
    def get_dataset(self, split: str, **kwargs):
        """
        Dapatkan dataset untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            **kwargs: Parameter tambahan untuk dataset
            
        Returns:
            Instance dari Dataset
        """
        return self.get_service('loader').get_dataset(split, **kwargs)
    
    def get_dataloader(self, split: str, **kwargs):
        """
        Dapatkan dataloader untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            **kwargs: Parameter tambahan untuk dataloader
            
        Returns:
            Instance dari DataLoader
        """
        return self.get_service('loader').get_dataloader(split, **kwargs)
    
    def get_all_dataloaders(self, **kwargs) -> Dict[str, Any]:
        """
        Dapatkan semua dataloader untuk semua split yang tersedia.
        
        Args:
            **kwargs: Parameter tambahan untuk dataloader
            
        Returns:
            Dictionary berisi dataloader untuk setiap split
        """
        return self.get_service('loader').get_all_dataloaders(**kwargs)
    
    # ===== Delegasi ke Validator Service =====
    
    def validate_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Validasi dataset untuk memastikan integritas data.
        
        Args:
            split: Split dataset yang akan divalidasi
            **kwargs: Parameter tambahan untuk validasi
            
        Returns:
            Hasil validasi
        """
        return self.get_service('validator').validate_dataset(split, **kwargs)
    
    def fix_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Perbaiki masalah yang ditemukan dalam dataset.
        
        Args:
            split: Split dataset yang akan diperbaiki
            **kwargs: Parameter tambahan untuk perbaikan
            
        Returns:
            Hasil perbaikan
        """
        return self.get_service('validator').fix_dataset(split, **kwargs)
    
    # ===== Delegasi ke Augmentor Service =====
    
    def augment_dataset(self, **kwargs) -> Dict[str, Any]:
        """
        Augmentasi dataset untuk meningkatkan variasi data.
        
        Args:
            **kwargs: Parameter untuk augmentasi
            
        Returns:
            Hasil augmentasi
        """
        return self.get_service('augmentor').augment_dataset(**kwargs)
    
    # ===== Delegasi ke Downloader Service =====
    
    def download_from_roboflow(self, **kwargs) -> str:
        """
        Download dataset dari Roboflow.
        
        Args:
            **kwargs: Parameter untuk download
            
        Returns:
            Path ke dataset yang didownload
        """
        return self.get_service('downloader').download_dataset(**kwargs)
    
    def upload_local_dataset(self, zip_path: str, **kwargs) -> Dict[str, Any]:
        """
        Upload dataset lokal dari file zip.
        
        Args:
            zip_path: Path ke file zip
            **kwargs: Parameter tambahan
            
        Returns:
            Hasil import
        """
        return self.get_service('downloader').import_from_zip(zip_path, **kwargs)
    
    # ===== Delegasi ke Explorer Service =====
    
    def explore_class_distribution(self, split: str) -> Dict[str, Any]:
        """
        Analisis distribusi kelas dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            
        Returns:
            Hasil analisis distribusi kelas
        """
        return self.get_service('explorer').analyze_class_distribution(split)
    
    def explore_layer_distribution(self, split: str) -> Dict[str, Any]:
        """
        Analisis distribusi layer dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            
        Returns:
            Hasil analisis distribusi layer
        """
        return self.get_service('explorer').analyze_layer_distribution(split)
    
    def explore_bbox_statistics(self, split: str) -> Dict[str, Any]:
        """
        Analisis statistik bounding box dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            
        Returns:
            Hasil analisis statistik bbox
        """
        return self.get_service('explorer').analyze_bbox_statistics(split)
    
    # ===== Delegasi ke Balance Service =====
    
    def balance_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Seimbangkan dataset untuk mengatasi ketidakseimbangan kelas.
        
        Args:
            split: Split dataset yang akan diseimbangkan
            **kwargs: Parameter tambahan
            
        Returns:
            Hasil penyeimbangan
        """
        return self.get_service('balancer').balance_by_undersampling(split, **kwargs)
    
    # ===== Delegasi ke Reporter Service =====
    
    def generate_dataset_report(self, splits: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Buat laporan komprehensif tentang dataset.
        
        Args:
            splits: List split dataset yang akan dimasukkan dalam laporan
            **kwargs: Parameter tambahan
            
        Returns:
            Laporan dataset
        """
        if splits is None:
            splits = ['train', 'valid', 'test']
        return self.get_service('reporter').generate_dataset_report(splits, **kwargs)
    
    # ===== Delegasi ke Visualizer Service =====
    
    def visualize_class_distribution(self, class_stats: Dict[str, int], **kwargs) -> str:
        """
        Visualisasikan distribusi kelas dalam dataset.
        
        Args:
            class_stats: Dictionary dengan class_name: count
            **kwargs: Parameter tambahan untuk visualisasi
            
        Returns:
            Path ke file visualisasi
        """
        visualizer = self.get_service('visualizer')['data_viz']
        return visualizer.plot_class_distribution(class_stats, **kwargs)
    
    def visualize_sample_images(self, data_dir: str, **kwargs) -> str:
        """
        Visualisasikan sampel gambar dengan bounding box.
        
        Args:
            data_dir: Direktori dataset
            **kwargs: Parameter tambahan untuk visualisasi
            
        Returns:
            Path ke file visualisasi
        """
        visualizer = self.get_service('visualizer')['data_viz']
        return visualizer.plot_sample_images(data_dir, **kwargs)
    
    def create_dataset_dashboard(self, report: Dict[str, Any], **kwargs) -> str:
        """
        Buat dashboard visualisasi dari laporan dataset.
        
        Args:
            report: Laporan dataset
            **kwargs: Parameter tambahan untuk visualisasi
            
        Returns:
            Path ke file dashboard
        """
        visualizer = self.get_service('visualizer')['report_viz']
        return visualizer.create_dataset_dashboard(report, **kwargs)
    
    # ===== Metode Utilitas =====
    
    def get_split_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Dapatkan statistik dasar untuk semua split.
        
        Returns:
            Dictionary berisi statistik setiap split
        """
        return self.utils.get_split_statistics()
    
    def split_dataset(self, **kwargs) -> Dict[str, int]:
        """
        Pecah dataset menjadi train/valid/test.
        
        Args:
            **kwargs: Parameter split
            
        Returns:
            Hasil pemecahan dataset
        """
        from smartcash.dataset.utils.split import DatasetSplitter
        splitter = DatasetSplitter(self.config, str(self.data_dir), self.logger)
        return splitter.split_dataset(**kwargs)