"""
File: smartcash/dataset/manager.py
Deskripsi: Koordinator utama untuk alur kerja dataset dengan implementasi exception handling
"""

import torch
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from smartcash.common.logger import get_logger
# Import exceptions
from smartcash.common.exceptions import (
    DatasetError, DatasetFileError, DatasetValidationError, 
    DatasetProcessingError, DatasetCompatibilityError
)

from smartcash.dataset.services.loader.dataset_loader import DatasetLoaderService
from smartcash.dataset.services.validator.dataset_validator import DatasetValidatorService
from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
from smartcash.dataset.services.explorer.explorer_service import ExplorerService
from smartcash.dataset.services.balancer.balance_service import BalanceService
from smartcash.dataset.services.reporter.report_service import ReportService
from smartcash.dataset.services.downloader.download_service import DownloadService

# Import baru untuk preprocessing
from smartcash.dataset.services.preprocessor.dataset_preprocessor import DatasetPreprocessor
from smartcash.dataset.services.loader.preprocessed_dataset_loader import PreprocessedDatasetLoader


class DatasetManager:
    """
    Koordinator utama untuk alur kerja dataset.
    Menyediakan akses terpadu ke berbagai layanan dataset seperti 
    loading, validasi, augmentasi, dan eksplorasi.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi dataset manager.
        
        Args:
            config: Konfigurasi dataset
            logger: Logger kustom
        """
        self.logger = logger or get_logger("dataset_manager")
        
        # Default config
        self.config = {
            'dataset_dir': 'data/',
            'img_size': (640, 640),
            'batch_size': 16,
            'num_workers': 4,
            'multilayer': True
        }
        
        # Tambahkan konfigurasi preprocessing
        self.preprocess_config = {
            'enable_preprocessing': True,
            'use_preprocessed': True,  # Gunakan dataset yang sudah di-preprocess
            'img_size': (640, 640),
            'preprocessed_dir': 'data/preprocessed',
            'raw_dataset_dir': 'data/'
        }
        
        # Update config dari parameter
        if config:
            if 'dataset' in config:
                self.config.update(config['dataset'])
            elif isinstance(config, dict):
                # Legacy support
                self.config.update(config)
                
            # Update preprocessing config
            if 'preprocessing' in config:
                self.preprocess_config.update(config['preprocessing'])
        
        # Dictionary services dengan lazy initialization
        self._services = {}
        
        self.logger.info(f"📊 DatasetManager diinisialisasi (dataset_dir: {self.config['dataset_dir']})")
        self.logger.info(f"📊 Pre-processing (preprocessed_dir: {self.config['preprocessed_dir']}); use_preprocessed: {self.preprocess_config['use_preprocessed']}")
    
    def get_service(self, service_name: str) -> Any:
        """
        Dapatkan instance service dengan lazy initialization.
        
        Args:
            service_name: Nama service
            
        Returns:
            Instance service
            
        Raises:
            DatasetError: Jika service_name tidak valid
        """
        if service_name in self._services:
            return self._services[service_name]
            
        # Inisialisasi service sesuai permintaan
        try:
            if service_name == 'loader':
                service = DatasetLoaderService(
                    dataset_dir=self.config['dataset_dir'],
                    img_size=self.config['img_size'],
                    multilayer=self.config.get('multilayer', True),
                    logger=self.logger
                )
            elif service_name == 'validator':
                service = DatasetValidatorService(
                    dataset_dir=self.config['dataset_dir'],
                    logger=self.logger
                )
            elif service_name == 'augmentor':
                service = AugmentationService(
                    dataset_dir=self.config['dataset_dir'],
                    logger=self.logger
                )
            elif service_name == 'explorer':
                service = ExplorerService(
                    dataset_dir=self.config['dataset_dir'],
                    logger=self.logger
                )
            elif service_name == 'balancer':
                service = BalanceService(
                    dataset_dir=self.config['dataset_dir'],
                    logger=self.logger
                )
            elif service_name == 'reporter':
                service = ReportService(
                    dataset_dir=self.config['dataset_dir'],
                    logger=self.logger
                )
            elif service_name == 'downloader':
                service = DownloadService(
                    output_dir=self.config['dataset_dir'],
                    logger=self.logger
                )
            else:
                raise DatasetError(f"Service '{service_name}' tidak dikenal")
        except ImportError as e:
            raise DatasetError(f"💥 Gagal mengimpor module untuk service '{service_name}': {str(e)}")
        except Exception as e:
            raise DatasetError(f"💥 Gagal menginisialisasi service '{service_name}': {str(e)}")
            
        # Simpan instance untuk penggunaan berikutnya
        self._services[service_name] = service
        return service
    
    def _get_preprocessor(self):
        """
        Dapatkan preprocessor dataset dengan lazy initialization.
        
        Returns:
            DatasetPreprocessor: Preprocessor dataset
            
        Raises:
            DatasetError: Jika gagal menginisialisasi preprocessor
        """
        try:
            if not hasattr(self, '_preprocessor'):
                self._preprocessor = DatasetPreprocessor(
                    config={
                        'img_size': self.preprocess_config['img_size'],
                        'preprocessed_dir': self.preprocess_config['preprocessed_dir'],
                        'dataset_dir': self.preprocess_config['raw_dataset_dir']
                    },
                    logger=self.logger
                )
            return self._preprocessor
        except ImportError as e:
            raise DatasetError(f"🚨 Error saat memuat DatasetPreprocessor: {str(e)}")
        except Exception as e:
            raise DatasetError(f"🚨 Error saat menginisialisasi preprocessor: {str(e)}")
    
    def _get_preprocessed_loader(self):
        """
        Dapatkan loader untuk dataset preprocessed dengan lazy initialization.
        
        Returns:
            PreprocessedDatasetLoader: Loader dataset preprocessed
            
        Raises:
            DatasetError: Jika gagal menginisialisasi loader
        """
        try:
            if not hasattr(self, '_preprocessed_loader'):
                self._preprocessed_loader = PreprocessedDatasetLoader(
                    preprocessed_dir=self.preprocess_config['preprocessed_dir'],
                    fallback_to_raw=True,
                    auto_preprocess=self.preprocess_config.get('auto_preprocess', True),
                    config={
                        'raw_dataset_dir': self.preprocess_config['raw_dataset_dir'],
                        'img_size': self.preprocess_config['img_size'],
                        'batch_size': self.config.get('batch_size', 16),
                        'num_workers': self.config.get('num_workers', 4)
                    },
                    logger=self.logger
                )
            return self._preprocessed_loader
        except ImportError as e:
            raise DatasetError(f"🚨 Error saat memuat PreprocessedDatasetLoader: {str(e)}")
        except Exception as e:
            raise DatasetError(f"🚨 Error saat menginisialisasi preprocessed loader: {str(e)}")
    
    def preprocess_dataset(self, split='all', force_reprocess=False):
        """
        Preprocess dataset dan simpan hasilnya.
        
        Args:
            split: Split dataset ('train', 'val', 'test', 'all')
            force_reprocess: Paksa proses ulang meskipun sudah ada
            
        Returns:
            Dict: Statistik hasil preprocessing
            
        Raises:
            DatasetProcessingError: Jika terjadi error saat preprocessing
        """
        try:
            preprocessor = self._get_preprocessor()
            return preprocessor.preprocess_dataset(split=split, force_reprocess=force_reprocess)
        except Exception as e:
            raise DatasetProcessingError(f"🔄 Error saat preprocessing dataset: {str(e)}")
    
    def clean_preprocessed(self, split='all'):
        """
        Bersihkan hasil preprocessing.
        
        Args:
            split: Split dataset yang akan dibersihkan ('train', 'val', 'test', 'all')
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            preprocessor = self._get_preprocessor()
            return preprocessor.clean_preprocessed(split=split)
        except Exception as e:
            raise DatasetProcessingError(f"🧹 Error saat membersihkan data preprocessed: {str(e)}")
    
    def get_preprocessed_stats(self):
        """
        Dapatkan statistik hasil preprocessing.
        
        Returns:
            Dict: Statistik data preprocessed
        """
        try:
            preprocessor = self._get_preprocessor()
            return preprocessor.get_preprocessed_stats()
        except Exception as e:
            raise DatasetError(f"📊 Error saat mengambil statistik preprocessed: {str(e)}")
    
    def get_dataset(self, split: str, **kwargs) -> torch.utils.data.Dataset:
        """
        Dapatkan dataset untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            **kwargs: Parameter tambahan untuk loader dataset
            
        Returns:
            Dataset
            
        Raises:
            DatasetFileError: Jika file dataset tidak ditemukan
        """
        try:
            # Gunakan preprocessed loader jika diaktifkan
            if self.preprocess_config.get('use_preprocessed', True):
                return self._get_preprocessed_loader().get_dataset(split, **kwargs)
            
            # Fallback ke loader asli
            loader = self.get_service('loader')
            return loader.get_dataset(split, **kwargs)
        except FileNotFoundError as e:
            raise DatasetFileError(f"📁 File dataset tidak ditemukan: {str(e)}")
        except Exception as e:
            raise DatasetError(f"📊 Error saat memuat dataset: {str(e)}")
    
    def get_dataloader(self, split: str, **kwargs) -> torch.utils.data.DataLoader:
        """
        Dapatkan dataloader untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            **kwargs: Parameter tambahan untuk dataloader
            
        Returns:
            DataLoader
            
        Raises:
            DatasetError: Jika terjadi error saat membuat dataloader
        """
        try:
            # Gunakan preprocessed loader jika diaktifkan
            if self.preprocess_config.get('use_preprocessed', True):
                return self._get_preprocessed_loader().get_dataloader(split, **kwargs)
            
            # Fallback ke loader asli
            loader = self.get_service('loader')
            return loader.get_dataloader(split, **kwargs)
        except FileNotFoundError as e:
            raise DatasetFileError(f"📁 File dataset tidak ditemukan: {str(e)}")
        except Exception as e:
            raise DatasetError(f"📊 Error saat membuat dataloader: {str(e)}")
    
    def get_all_dataloaders(self, **kwargs) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Dapatkan dataloader untuk semua split.
        
        Args:
            **kwargs: Parameter tambahan untuk dataloader
            
        Returns:
            Dictionary dengan dataloader per split
            
        Raises:
            DatasetError: Jika terjadi error saat membuat dataloader
        """
        try:
            # Gunakan preprocessed loader jika diaktifkan
            if self.preprocess_config.get('use_preprocessed', True):
                return self._get_preprocessed_loader().get_all_dataloaders(**kwargs)
            
            # Fallback ke loader asli
            loader = self.get_service('loader')
            return loader.get_all_dataloaders(**kwargs)
        except Exception as e:
            raise DatasetError(f"📊 Error saat membuat semua dataloader: {str(e)}")
    
    def validate_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Validasi dataset untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'val', 'test', 'all')
            **kwargs: Parameter tambahan untuk validasi
            
        Returns:
            Hasil validasi
            
        Raises:
            DatasetValidationError: Jika terjadi error saat validasi
        """
        try:
            validator = self.get_service('validator')
            
            if split == 'all':
                results = {}
                for s in ['train', 'val', 'test']:
                    try:
                        results[s] = validator.validate_dataset(s, **kwargs)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Tidak dapat memvalidasi split {s}: {str(e)}")
                        results[s] = {"error": str(e)}
                return results
            else:
                return validator.validate_dataset(split, **kwargs)
        except Exception as e:
            raise DatasetValidationError(f"✅ Error saat validasi dataset: {str(e)}")
    
    def fix_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Perbaiki masalah pada dataset.
        
        Args:
            split: Split dataset ('train', 'val', 'test', 'all')
            **kwargs: Parameter tambahan untuk perbaikan
            
        Returns:
            Hasil perbaikan
            
        Raises:
            DatasetError: Jika terjadi error saat perbaikan dataset
        """
        try:
            validator = self.get_service('validator')
            
            if split == 'all':
                results = {}
                for s in ['train', 'val', 'test']:
                    try:
                        results[s] = validator.fix_dataset(s, **kwargs)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Tidak dapat memperbaiki split {s}: {str(e)}")
                        results[s] = {"error": str(e)}
                return results
            else:
                return validator.fix_dataset(split, **kwargs)
        except Exception as e:
            raise DatasetError(f"🔧 Error saat memperbaiki dataset: {str(e)}")
    
    def augment_dataset(self, **kwargs) -> Dict[str, Any]:
        """
        Augmentasi dataset.
        
        Args:
            **kwargs: Parameter tambahan untuk augmentasi
            
        Returns:
            Hasil augmentasi
            
        Raises:
            DatasetProcessingError: Jika terjadi error saat augmentasi
        """
        try:
            augmentor = self.get_service('augmentor')
            return augmentor.augment_dataset(**kwargs)
        except Exception as e:
            raise DatasetProcessingError(f"🖼️ Error saat augmentasi dataset: {str(e)}")
    
    def download_from_roboflow(self, **kwargs) -> Dict[str, Any]:
        """
        Download dataset dari Roboflow.
        
        Args:
            **kwargs: Parameter untuk download (api_key, workspace, project, version)
            
        Returns:
            Info hasil download
            
        Raises:
            DatasetError: Jika terjadi error saat download
        """
        try:
            downloader = self.get_service('downloader')
            return downloader.download_from_roboflow(**kwargs)
        except ImportError as e:
            raise DatasetError(f"📚 Package tidak tersedia: {str(e)}")
        except Exception as e:
            raise DatasetError(f"📥 Error saat download dataset: {str(e)}")
    
    def explore_class_distribution(self, split: str) -> Dict[str, Any]:
        """
        Analisis distribusi kelas dalam dataset.
        
        Args:
            split: Split dataset ('train', 'val', 'test', 'all')
            
        Returns:
            Statistik distribusi kelas
            
        Raises:
            DatasetError: Jika terjadi error saat analisis
        """
        try:
            explorer = self.get_service('explorer')
            
            if split == 'all':
                results = {}
                for s in ['train', 'val', 'test']:
                    try:
                        results[s] = explorer.analyze_class_distribution(s)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Tidak dapat menganalisis split {s}: {str(e)}")
                        results[s] = None
                return results
            else:
                return explorer.analyze_class_distribution(split)
        except Exception as e:
            raise DatasetError(f"📊 Error saat analisis distribusi kelas: {str(e)}")
    
    def explore_layer_distribution(self, split: str) -> Dict[str, Any]:
        """
        Analisis distribusi layer dalam dataset.
        
        Args:
            split: Split dataset ('train', 'val', 'test', 'all')
            
        Returns:
            Statistik distribusi layer
            
        Raises:
            DatasetError: Jika terjadi error saat analisis
        """
        try:
            explorer = self.get_service('explorer')
            
            if split == 'all':
                results = {}
                for s in ['train', 'val', 'test']:
                    try:
                        results[s] = explorer.analyze_layer_distribution(s)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Tidak dapat menganalisis split {s}: {str(e)}")
                        results[s] = None
                return results
            else:
                return explorer.analyze_layer_distribution(split)
        except Exception as e:
            raise DatasetError(f"📊 Error saat analisis distribusi layer: {str(e)}")
    
    def explore_bbox_statistics(self, split: str) -> Dict[str, Any]:
        """
        Analisis statistik bounding box dalam dataset.
        
        Args:
            split: Split dataset ('train', 'val', 'test', 'all')
            
        Returns:
            Statistik bounding box
            
        Raises:
            DatasetError: Jika terjadi error saat analisis
        """
        try:
            explorer = self.get_service('explorer')
            
            if split == 'all':
                results = {}
                for s in ['train', 'val', 'test']:
                    try:
                        results[s] = explorer.analyze_bbox_statistics(s)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Tidak dapat menganalisis split {s}: {str(e)}")
                        results[s] = None
                return results
            else:
                return explorer.analyze_bbox_statistics(split)
        except Exception as e:
            raise DatasetError(f"📊 Error saat analisis statistik bbox: {str(e)}")
    
    def balance_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Seimbangkan dataset.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            **kwargs: Parameter tambahan untuk balancing
            
        Returns:
            Hasil balancing
            
        Raises:
            DatasetProcessingError: Jika terjadi error saat balancing
        """
        try:
            balancer = self.get_service('balancer')
            return balancer.balance_by_undersampling(split, **kwargs)
        except Exception as e:
            raise DatasetProcessingError(f"⚖️ Error saat menyeimbangkan dataset: {str(e)}")
    
    def generate_dataset_report(self, splits: List[str], **kwargs) -> Dict[str, Any]:
        """
        Buat laporan dataset.
        
        Args:
            splits: List split untuk laporan
            **kwargs: Parameter tambahan untuk laporan
            
        Returns:
            Laporan dataset
            
        Raises:
            DatasetError: Jika terjadi error saat membuat laporan
        """
        try:
            reporter = self.get_service('reporter')
            return reporter.generate_dataset_report(splits, **kwargs)
        except Exception as e:
            raise DatasetError(f"📋 Error saat membuat laporan dataset: {str(e)}")
    
    def visualize_class_distribution(self, class_stats: Dict[str, Any], **kwargs) -> Dict[str, str]:
        """
        Visualisasi distribusi kelas dalam dataset.
        
        Args:
            class_stats: Statistik kelas (dari explore_class_distribution)
            **kwargs: Parameter tambahan untuk visualisasi
            
        Returns:
            Path visualisasi
            
        Raises:
            DatasetError: Jika terjadi error saat visualisasi
        """
        try:
            reporter = self.get_service('reporter')
            return reporter.visualize_class_distribution(class_stats, **kwargs)
        except Exception as e:
            raise DatasetError(f"📊 Error saat visualisasi distribusi kelas: {str(e)}")
    
    def create_dataset_dashboard(self, report: Dict[str, Any], **kwargs) -> Dict[str, str]:
        """
        Buat dashboard visualisasi dataset.
        
        Args:
            report: Laporan dataset (dari generate_dataset_report)
            **kwargs: Parameter tambahan untuk dashboard
            
        Returns:
            Path visualisasi
            
        Raises:
            DatasetError: Jika terjadi error saat membuat dashboard
        """
        try:
            reporter = self.get_service('reporter')
            return reporter.create_dashboard(report, **kwargs)
        except Exception as e:
            raise DatasetError(f"📊 Error saat membuat dashboard dataset: {str(e)}")
    
    def get_split_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Dapatkan statistik dasar tentang split dataset.
        
        Returns:
            Dictionary dengan informasi jumlah file per split
            
        Raises:
            DatasetError: Jika terjadi error saat mengambil statistik
        """
        try:
            # Jika menggunakan preprocessed data, dapatkan statistik dari sana
            if self.preprocess_config.get('use_preprocessed', True):
                return self.get_preprocessed_stats()
                
            # Fallback ke loader asli
            loader = self.get_service('loader')
            return loader.get_dataset_stats()
        except Exception as e:
            raise DatasetError(f"📊 Error saat mengambil statistik split: {str(e)}")
    
    def split_dataset(self, **kwargs) -> Dict[str, Any]:
        """
        Pecah dataset menjadi split train/val/test.
        
        Args:
            **kwargs: Parameter untuk splitting
            
        Returns:
            Hasil splitting
            
        Raises:
            NotImplementedError: Fungsi belum diimplementasikan
        """
        raise NotImplementedError("Splitting dataset belum diimplementasikan")