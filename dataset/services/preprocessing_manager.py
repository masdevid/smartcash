"""
File: smartcash/dataset/services/preprocessing_manager.py
Deskripsi: Manager khusus untuk preprocessing dengan UI progress notifications dan suppressed logging
"""

import torch
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path

from smartcash.common.exceptions import DatasetError, DatasetFileError, DatasetProcessingError
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_IMG_SIZE, DEFAULT_PREPROCESSED_DIR
from smartcash.dataset.services.preprocessor.preprocessing_service import PreprocessingService

class PreprocessingManager:
    """Manager untuk preprocessing dengan UI progress notifications dan suppressed logging."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Inisialisasi PreprocessingManager.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger untuk logging (akan disuppress untuk console)
        """
        self.config = config
        self.logger = logger
        self.preprocess_config = config.get('preprocessing', {})
        self._preprocessing_service = None
        self._preprocessed_loader = None
        self._progress_callback = None
        
        # Setup suppressed logging
        self._setup_suppressed_logging()
    
    def _setup_suppressed_logging(self):
        """Setup suppressed logging untuk prevent console output."""
        import logging
        
        # Suppress semua preprocessing related logging ke console
        loggers_to_suppress = [
            'smartcash.dataset.services.preprocessor',
            'smartcash.dataset.utils',
            'tqdm',
            'cv2',
            'PIL'
        ]
        
        for logger_name in loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
    
    def register_progress_callback(self, callback: Callable) -> None:
        """
        Register progress callback untuk UI notifications.
        
        Args:
            callback: Fungsi callback untuk progress tracking
        """
        self._progress_callback = callback
        
        # Register ke service jika sudah ada
        if self._preprocessing_service:
            self._preprocessing_service.register_progress_callback(callback)
    
    def _get_preprocessing_service(self):
        """
        Dapatkan preprocessing service dengan lazy initialization.
        
        Returns:
            PreprocessingService instance
        """
        if not self._preprocessing_service:
            try:
                # Integrasikan konfigurasi untuk preprocessing service
                integrated_config = {
                    'preprocessing': {
                        'img_size': self.preprocess_config.get('img_size', DEFAULT_IMG_SIZE),
                        'output_dir': self.preprocess_config.get('preprocessed_dir', DEFAULT_PREPROCESSED_DIR),
                        'normalization': {
                            'enabled': self.preprocess_config.get('normalize', True),
                            'preserve_aspect_ratio': self.preprocess_config.get('preserve_aspect_ratio', True)
                        }
                    },
                    'data': {
                        'dir': self.preprocess_config.get('raw_dataset_dir', 'data')
                    }
                }
                
                # Inisialisasi preprocessing service dengan progress callback
                self._preprocessing_service = PreprocessingService(
                    config=integrated_config,
                    logger=self.logger,
                    progress_callback=self._progress_callback
                )
                
            except Exception as e:
                error_msg = f"Error saat menginisialisasi preprocessing service: {str(e)}"
                if self._progress_callback:
                    self._progress_callback(
                        progress=0, total=100,
                        message=error_msg,
                        status="error"
                    )
                raise DatasetError(error_msg)
                
        return self._preprocessing_service
    
    def _get_preprocessed_loader(self):
        """
        Dapatkan loader untuk dataset preprocessed dengan lazy initialization.
        
        Returns:
            PreprocessedDatasetLoader instance
        """
        if not self._preprocessed_loader:
            try:
                from smartcash.dataset.services.loader.preprocessed_dataset_loader import PreprocessedDatasetLoader
                
                self._preprocessed_loader = PreprocessedDatasetLoader(
                    preprocessed_dir=self.preprocess_config.get('preprocessed_dir', DEFAULT_PREPROCESSED_DIR),
                    fallback_to_raw=True,
                    auto_preprocess=self.preprocess_config.get('auto_preprocess', True),
                    config={
                        'raw_dataset_dir': self.preprocess_config.get('raw_dataset_dir', 'data'),
                        'img_size': self.preprocess_config.get('img_size', DEFAULT_IMG_SIZE),
                        'batch_size': self.config.get('batch_size', 16),
                        'num_workers': self.config.get('num_workers', 4)
                    },
                    logger=self.logger
                )
            except ImportError as e:
                error_msg = f"Error saat memuat PreprocessedDatasetLoader: {str(e)}"
                if self._progress_callback:
                    self._progress_callback(
                        progress=0, total=100,
                        message=error_msg,
                        status="error"
                    )
                raise DatasetError(error_msg)
            except Exception as e:
                error_msg = f"Error saat menginisialisasi preprocessed loader: {str(e)}"
                if self._progress_callback:
                    self._progress_callback(
                        progress=0, total=100,
                        message=error_msg,
                        status="error"
                    )
                raise DatasetError(error_msg)
                
        return self._preprocessed_loader
    
    def preprocess_dataset(self, split='all', force_reprocess=False, **kwargs) -> Dict[str, Any]:
        """
        Preprocess dataset dengan UI progress notifications.
        
        Args:
            split: Split dataset ('train', 'val', 'test', 'all')
            force_reprocess: Paksa proses ulang meskipun sudah ada
            **kwargs: Parameter tambahan untuk preprocessing
            
        Returns:
            Dict: Statistik hasil preprocessing
        """
        try:
            # Notify start via progress callback
            if self._progress_callback:
                split_info = split if split != 'all' else 'semua split'
                self._progress_callback(
                    progress=0, total=100,
                    message=f"Memulai preprocessing dataset ({split_info})",
                    status="info", step=0, split_step="Persiapan"
                )
                
            preprocessing_service = self._get_preprocessing_service()
            
            # Extract parameter yang diperlukan untuk preprocessing
            valid_preprocessor_params = {
                'show_progress': False,  # Always disable tqdm
                'normalize': kwargs.get('normalize', True),
                'preserve_aspect_ratio': kwargs.get('preserve_aspect_ratio', True),
            }
            
            # Jalankan preprocessing
            result = preprocessing_service.preprocess_dataset(
                split=split, 
                force_reprocess=force_reprocess,
                **valid_preprocessor_params
            )
            
            # Log hasil preprocessing via progress callback
            if self._progress_callback and result:
                total_images = result.get('total_images', 0)
                processing_time = result.get('processing_time', 0)
                self._progress_callback(
                    progress=100, total=100,
                    message=f"Preprocessing selesai: {total_images} gambar, waktu: {processing_time:.2f} detik",
                    status="success", step=3, split_step="Selesai"
                )
                
            return result
            
        except Exception as e:
            error_msg = f"Error saat preprocessing dataset: {str(e)}"
            if self._progress_callback:
                self._progress_callback(
                    progress=0, total=100,
                    message=error_msg,
                    status="error", step=0, split_step="Error"
                )
            raise DatasetProcessingError(error_msg)
    
    def cleanup(self):
        """Cleanup resources."""
        if self._preprocessing_service:
            self._preprocessing_service.cleanup()
            self._preprocessing_service = None
        self._preprocessed_loader = None
        self._progress_callback = None
    
    def clean_preprocessed(self, split='all') -> bool:
        """
        Bersihkan hasil preprocessing.
        
        Args:
            split: Split dataset yang akan dibersihkan ('train', 'val', 'test', 'all')
            
        Returns:
            Boolean menunjukkan keberhasilan
        """
        try:
            if self._progress_callback:
                split_info = split if split != 'all' else 'semua split'
                self._progress_callback(
                    progress=0, total=100,
                    message=f"Membersihkan data preprocessed ({split_info})",
                    status="info", step=0, split_step="Cleanup"
                )
                
            preprocessing_service = self._get_preprocessing_service()
            preprocessing_service.preprocessor.clean_preprocessed(split=None if split == 'all' else split)
            
            if self._progress_callback:
                self._progress_callback(
                    progress=100, total=100,
                    message="Pembersihan data preprocessing selesai",
                    status="success", step=1, split_step="Selesai"
                )
                
            return True
        except Exception as e:
            error_msg = f"Error saat membersihkan data preprocessed: {str(e)}"
            if self._progress_callback:
                self._progress_callback(
                    progress=0, total=100,
                    message=error_msg,
                    status="error", step=0, split_step="Error"
                )
            raise DatasetProcessingError(error_msg)
    
    def get_preprocessed_stats(self) -> Dict[str, Any]:
        """
        Dapatkan statistik hasil preprocessing.
        
        Returns:
            Dictionary berisi statistik dataset preprocessed
        """
        try:
            preprocessing_service = self._get_preprocessing_service()
            stats = preprocessing_service.preprocessor.get_preprocessed_stats()
            
            if self._progress_callback:
                total_stats = sum(
                    stats.get(split, {}).get('processed', 0) 
                    for split in DEFAULT_SPLITS 
                    if split in stats
                )
                self._progress_callback(
                    progress=100, total=100,
                    message=f"Statistik dataset preprocessed: {total_stats} total gambar",
                    status="info", step=1, split_step="Info"
                )
                
            return stats
        except Exception as e:
            error_msg = f"Error saat mengambil statistik preprocessed: {str(e)}"
            if self._progress_callback:
                self._progress_callback(
                    progress=0, total=100,
                    message=error_msg,
                    status="error", step=0, split_step="Error"
                )
            raise DatasetError(error_msg)
    
    def get_dataset(self, split: str, **kwargs) -> torch.utils.data.Dataset:
        """
        Dapatkan dataset untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            **kwargs: Parameter tambahan untuk loader dataset
            
        Returns:
            Dataset
        """
        try:
            dataset = self._get_preprocessed_loader().get_dataset(split, **kwargs)
            return dataset
        except FileNotFoundError as e:
            error_msg = f"File dataset tidak ditemukan: {str(e)}"
            if self._progress_callback:
                self._progress_callback(
                    progress=0, total=100,
                    message=error_msg,
                    status="error"
                )
            raise DatasetFileError(error_msg)
        except Exception as e:
            error_msg = f"Error saat memuat dataset: {str(e)}"
            if self._progress_callback:
                self._progress_callback(
                    progress=0, total=100,
                    message=error_msg,
                    status="error"
                )
            raise DatasetError(error_msg)
    
    def get_dataloader(self, split: str, **kwargs) -> torch.utils.data.DataLoader:
        """
        Dapatkan dataloader untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            **kwargs: Parameter tambahan untuk dataloader
            
        Returns:
            DataLoader
        """
        try:
            dataloader = self._get_preprocessed_loader().get_dataloader(split, **kwargs)
            return dataloader
        except FileNotFoundError as e:
            error_msg = f"File dataset tidak ditemukan: {str(e)}"
            if self._progress_callback:
                self._progress_callback(
                    progress=0, total=100,
                    message=error_msg,
                    status="error"
                )
            raise DatasetFileError(error_msg)
        except Exception as e:
            error_msg = f"Error saat membuat dataloader: {str(e)}"
            if self._progress_callback:
                self._progress_callback(
                    progress=0, total=100,
                    message=error_msg,
                    status="error"
                )
            raise DatasetError(error_msg)
    
    def get_all_dataloaders(self, **kwargs) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Dapatkan dataloader untuk semua split.
        
        Args:
            **kwargs: Parameter tambahan untuk dataloader
            
        Returns:
            Dictionary dengan dataloader per split
        """
        try:
            dataloaders = self._get_preprocessed_loader().get_all_dataloaders(**kwargs)
            
            if self._progress_callback and dataloaders:
                split_info = ", ".join([f"{k}: {len(v.dataset)}" for k, v in dataloaders.items()])
                self._progress_callback(
                    progress=100, total=100,
                    message=f"Semua dataloader berhasil dibuat ({split_info} sampel)",
                    status="success"
                )
                
            return dataloaders
        except Exception as e:
            error_msg = f"Error saat membuat semua dataloader: {str(e)}"
            if self._progress_callback:
                self._progress_callback(
                    progress=0, total=100,
                    message=error_msg,
                    status="error"
                )
            raise DatasetError(error_msg)