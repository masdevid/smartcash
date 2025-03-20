"""
File: smartcash/dataset/services/preprocessing_manager.py
Deskripsi: Manager khusus untuk fungsionalitas preprocessing dataset
"""

import torch
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path

from smartcash.common.exceptions import DatasetError, DatasetFileError, DatasetProcessingError

class PreprocessingManager:
    """Manager khusus untuk fungsionalitas preprocessing dataset dengan SRP approach."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Inisialisasi PreprocessingManager.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger untuk logging
        """
        self.config = config
        self.logger = logger
        self.preprocess_config = config.get('preprocessing', {})
        self._preprocessor = None
        self._preprocessed_loader = None
        self._progress_callback = None
    
    def register_progress_callback(self, callback: Callable) -> None:
        """
        Register progress callback untuk preprocessing.
        
        Args:
            callback: Fungsi callback untuk progress tracking
        """
        self._progress_callback = callback
        if self._preprocessor:
            self._preprocessor.register_progress_callback(callback)
    
    def _get_preprocessor(self):
        """
        Dapatkan preprocessor dataset dengan lazy initialization.
        
        Returns:
            DatasetPreprocessor instance
        """
        if not self._preprocessor:
            try:
                from smartcash.dataset.services.preprocessor.dataset_preprocessor import DatasetPreprocessor
                
                integrated_config = {
                    'preprocessing': {
                        'img_size': self.preprocess_config.get('img_size', [640, 640]),
                        'output_dir': self.preprocess_config.get('preprocessed_dir', 'data/preprocessed'),
                    },
                    'data': {
                        'dir': self.preprocess_config.get('raw_dataset_dir', 'data')
                    }
                }
                
                self._preprocessor = DatasetPreprocessor(
                    config=integrated_config,
                    logger=self.logger
                )
                
                # Register callback jika ada
                if self._progress_callback:
                    self._preprocessor.register_progress_callback(self._progress_callback)
            except ImportError as e:
                raise DatasetError(f"🚨 Error saat memuat DatasetPreprocessor: {str(e)}")
            except Exception as e:
                raise DatasetError(f"🚨 Error saat menginisialisasi preprocessor: {str(e)}")
                
        return self._preprocessor
    
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
                    preprocessed_dir=self.preprocess_config.get('preprocessed_dir', 'data/preprocessed'),
                    fallback_to_raw=True,
                    auto_preprocess=self.preprocess_config.get('auto_preprocess', True),
                    config={
                        'raw_dataset_dir': self.preprocess_config.get('raw_dataset_dir', 'data'),
                        'img_size': self.preprocess_config.get('img_size', [640, 640]),
                        'batch_size': self.config.get('batch_size', 16),
                        'num_workers': self.config.get('num_workers', 4)
                    },
                    logger=self.logger
                )
            except ImportError as e:
                raise DatasetError(f"🚨 Error saat memuat PreprocessedDatasetLoader: {str(e)}")
            except Exception as e:
                raise DatasetError(f"🚨 Error saat menginisialisasi preprocessed loader: {str(e)}")
                
        return self._preprocessed_loader
    
    def preprocess_dataset(self, split='all', force_reprocess=False, **kwargs) -> Dict[str, Any]:
        """
        Preprocess dataset dan simpan hasilnya.
        
        Args:
            split: Split dataset ('train', 'val', 'test', 'all')
            force_reprocess: Paksa proses ulang meskipun sudah ada
            **kwargs: Parameter tambahan untuk preprocessing
            
        Returns:
            Dict: Statistik hasil preprocessing
        """
        try:
            preprocessor = self._get_preprocessor()
            
            # Extract parameter yang diperlukan untuk preprocessing
            valid_preprocessor_params = {
                'show_progress': kwargs.get('show_progress', True),
                'normalize': kwargs.get('normalize', True),
                'preserve_aspect_ratio': kwargs.get('preserve_aspect_ratio', True),
            }
            
            return preprocessor.preprocess_dataset(
                split=split, 
                force_reprocess=force_reprocess,
                **valid_preprocessor_params
            )
        except Exception as e:
            raise DatasetProcessingError(f"🔄 Error saat preprocessing dataset: {str(e)}")
    
    def clean_preprocessed(self, split='all') -> bool:
        """
        Bersihkan hasil preprocessing.
        
        Args:
            split: Split dataset yang akan dibersihkan ('train', 'val', 'test', 'all')
            
        Returns:
            Boolean menunjukkan keberhasilan
        """
        try:
            preprocessor = self._get_preprocessor()
            preprocessor.clean_preprocessed(split=None if split == 'all' else split)
            return True
        except Exception as e:
            raise DatasetProcessingError(f"🧹 Error saat membersihkan data preprocessed: {str(e)}")
    
    def get_preprocessed_stats(self) -> Dict[str, Any]:
        """
        Dapatkan statistik hasil preprocessing.
        
        Returns:
            Dictionary berisi statistik dataset preprocessed
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
        """
        try:
            return self._get_preprocessed_loader().get_dataset(split, **kwargs)
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
        """
        try:
            return self._get_preprocessed_loader().get_dataloader(split, **kwargs)
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
        """
        try:
            return self._get_preprocessed_loader().get_all_dataloaders(**kwargs)
        except Exception as e:
            raise DatasetError(f"📊 Error saat membuat semua dataloader: {str(e)}")