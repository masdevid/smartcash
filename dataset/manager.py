"""
File: smartcash/dataset/manager.py
Deskripsi: Koordinator utama untuk alur kerja dataset dengan implementasi Factory Pattern
"""

import torch
from typing import Dict, List, Optional, Union, Any, Callable
from smartcash.common.logger import get_logger
from smartcash.common.exceptions import DatasetError, DatasetProcessingError

from smartcash.dataset.services.service_factory import ServiceFactory
from smartcash.dataset.services.preprocessing_manager import PreprocessingManager

class DatasetManager:
    """Koordinator utama untuk alur kerja dataset dengan implementasi Factory Pattern."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[Any] = None):
        """
        Inisialisasi dataset manager dengan Factory pattern.
        
        Args:
            config: Konfigurasi dataset
            logger: Logger kustom
        """
        self.logger = logger or get_logger("dataset_manager")
        
        # Default config
        self.config = self._initialize_config(config)
        
        # Lazy-loaded services
        self._services = {}
        self._progress_callback = None
        
        # Service Factory
        self.service_factory = ServiceFactory(self.config, self.logger)
        
        # Preprocessing Manager
        self.preprocessing = PreprocessingManager(self.config, self.logger)
        
        self.logger.info(f"📊 DatasetManager diinisialisasi (dataset_dir: {self.config['dataset_dir']})")
    
    def _initialize_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Inisialisasi konfigurasi dengan default values dan update dari parameter."""
        base_config = {
            'dataset_dir': 'data/',
            'img_size': (640, 640),
            'batch_size': 16,
            'num_workers': 4,
            'multilayer': True,
            'preprocessing': {
                'enable_preprocessing': True,
                'use_preprocessed': True,
                'img_size': (640, 640),
                'preprocessed_dir': 'data/preprocessed',
                'raw_dataset_dir': 'data/'
            }
        }
        
        if config:
            # Update base config
            if 'dataset' in config:
                base_config.update(config['dataset'])
            elif isinstance(config, dict):
                base_config.update(config)
                
            # Update preprocessing config
            if 'preprocessing' in config:
                base_config['preprocessing'].update(config['preprocessing'])
        
        return base_config
    
    def register_progress_callback(self, callback: Callable) -> None:
        """Register progress callback untuk digunakan di preprocessing."""
        self._progress_callback = callback
        self.preprocessing.register_progress_callback(callback)
    
    def get_service(self, service_name: str) -> Any:
        """Dapatkan service instance dengan lazy initialization menggunakan Factory."""
        if service_name in self._services:
            return self._services[service_name]
            
        try:
            service = self.service_factory.create_service(service_name)
            self._services[service_name] = service
            return service
        except Exception as e:
            raise DatasetError(f"💥 Gagal mendapatkan service '{service_name}': {str(e)}")
    
    # ===== Preprocessing Methods =====
    
    def preprocess_dataset(self, split='all', force_reprocess=False, **kwargs) -> Dict[str, Any]:
        """Preprocess dataset dan simpan hasilnya."""
        return self.preprocessing.preprocess_dataset(split, force_reprocess, **kwargs)
    
    def clean_preprocessed(self, split='all'):
        """Bersihkan hasil preprocessing."""
        return self.preprocessing.clean_preprocessed(split)
    
    def get_preprocessed_stats(self):
        """Dapatkan statistik hasil preprocessing."""
        return self.preprocessing.get_preprocessed_stats()
    
    # ===== Dataset Methods =====
    
    def get_dataset(self, split: str, **kwargs) -> torch.utils.data.Dataset:
        """Dapatkan dataset untuk split tertentu."""
        return self.preprocessing.get_dataset(split, **kwargs) if self.config['preprocessing'].get('use_preprocessed', True) else self.get_service('loader').get_dataset(split, **kwargs)
    
    def get_dataloader(self, split: str, **kwargs) -> torch.utils.data.DataLoader:
        """Dapatkan dataloader untuk split tertentu."""
        return self.preprocessing.get_dataloader(split, **kwargs) if self.config['preprocessing'].get('use_preprocessed', True) else self.get_service('loader').get_dataloader(split, **kwargs)
    
    def get_all_dataloaders(self, **kwargs) -> Dict[str, torch.utils.data.DataLoader]:
        """Dapatkan dataloader untuk semua split."""
        return self.preprocessing.get_all_dataloaders(**kwargs) if self.config['preprocessing'].get('use_preprocessed', True) else self.get_service('loader').get_all_dataloaders(**kwargs)
    
    # ===== Data Processing Methods =====
    
    def validate_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """Validasi dataset untuk split tertentu."""
        validator = self.get_service('validator')
        
        if split == 'all':
            return {s: validator.validate_dataset(s, **kwargs) for s in ['train', 'val', 'test']}
        return validator.validate_dataset(split, **kwargs)
    
    def fix_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """Perbaiki masalah pada dataset."""
        validator = self.get_service('validator')
        
        if split == 'all':
            return {s: validator.fix_dataset(s, **kwargs) for s in ['train', 'val', 'test']}
        return validator.fix_dataset(split, **kwargs)
    
    def augment_dataset(self, **kwargs) -> Dict[str, Any]:
        """Augmentasi dataset."""
        return self.get_service('augmentor').augment_dataset(**kwargs)
    
    def download_from_roboflow(self, **kwargs) -> Dict[str, Any]:
        """Download dataset dari Roboflow."""
        return self.get_service('downloader').download_from_roboflow(**kwargs)
    
    # ===== Analysis Methods =====
    
    def explore_class_distribution(self, split: str) -> Dict[str, Any]:
        """Analisis distribusi kelas dalam dataset."""
        explorer = self.get_service('explorer')
        
        if split == 'all':
            return {s: explorer.analyze_class_distribution(s) for s in ['train', 'val', 'test']}
        return explorer.analyze_class_distribution(split)
    
    def explore_layer_distribution(self, split: str) -> Dict[str, Any]:
        """Analisis distribusi layer dalam dataset."""
        explorer = self.get_service('explorer')
        
        if split == 'all':
            return {s: explorer.analyze_layer_distribution(s) for s in ['train', 'val', 'test']}
        return explorer.analyze_layer_distribution(split)
    
    def explore_bbox_statistics(self, split: str) -> Dict[str, Any]:
        """Analisis statistik bounding box dalam dataset."""
        explorer = self.get_service('explorer')
        
        if split == 'all':
            return {s: explorer.analyze_bbox_statistics(s) for s in ['train', 'val', 'test']}
        return explorer.analyze_bbox_statistics(split)
    
    def balance_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """Seimbangkan dataset."""
        return self.get_service('balancer').balance_by_undersampling(split, **kwargs)
    
    def generate_dataset_report(self, splits: List[str], **kwargs) -> Dict[str, Any]:
        """Buat laporan dataset."""
        return self.get_service('reporter').generate_dataset_report(splits, **kwargs)
    
    def visualize_class_distribution(self, class_stats: Dict[str, Any], **kwargs) -> Dict[str, str]:
        """Visualisasi distribusi kelas dalam dataset."""
        return self.get_service('reporter').visualize_class_distribution(class_stats, **kwargs)
    
    def create_dataset_dashboard(self, report: Dict[str, Any], **kwargs) -> Dict[str, str]:
        """Buat dashboard visualisasi dataset."""
        return self.get_service('reporter').create_dashboard(report, **kwargs)
    
    def get_split_statistics(self) -> Dict[str, Dict[str, int]]:
        """Dapatkan statistik dasar tentang split dataset."""
        return self.get_preprocessed_stats() if self.config['preprocessing'].get('use_preprocessed', True) else self.get_service('loader').get_dataset_stats()
    
    def split_dataset(self, **kwargs) -> Dict[str, Any]:
        """Pecah dataset menjadi split train/val/test."""
        raise NotImplementedError("Splitting dataset belum diimplementasikan")
    
    def cleanup_dataset(self, output_dir=None):
        """Stub for test compatibility. Does nothing."""
        self.logger.info(f"Called cleanup_dataset with output_dir={output_dir}")