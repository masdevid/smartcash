"""
File: smartcash/dataset/services/preprocessor/preprocessing_service.py
Deskripsi: Layanan preprocessing dataset dengan integrasi observer pattern dan UI progress notifications
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable

from smartcash.common.logger import get_logger
from smartcash.common.exceptions import DatasetError
from smartcash.common.config import ConfigManager, get_config_manager
from smartcash.dataset.services.preprocessor.dataset_preprocessor import DatasetPreprocessor
from smartcash.dataset.utils.dataset_constants import DEFAULT_IMG_SIZE, DEFAULT_PREPROCESSED_DIR

class PreprocessingService:
    """Layanan preprocessing dataset dengan integrasi UI progress notifications."""
    
    def __init__(self, config: Dict[str, Any], logger=None, progress_callback=None):
        """
        Inisialisasi PreprocessingService.
        
        Args:
            config: Konfigurasi preprocessing
            logger: Logger untuk logging
            progress_callback: Callback untuk UI progress notifications
        """
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.progress_callback = progress_callback
        self.config_manager = get_config_manager()
        
        # Validasi base directory
        base_dir = config.get('data', {}).get('dir')
        if not base_dir:
            raise DatasetError("base_dir must not be None. Please provide a valid base directory for configuration.")
        
        # Setup paths
        self.base_dir = Path(base_dir)
        self.preprocessed_dir = Path(config.get('preprocessing', {}).get('output_dir', DEFAULT_PREPROCESSED_DIR))
        
        # Ensure directories exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessor
        self._preprocessor = None
        
        # Setup suppressed logging untuk prevent console leaks
        self._setup_suppressed_logging()
    
    def _setup_suppressed_logging(self):
        """Setup logging yang disuppress untuk prevent console output."""
        import logging
        
        # Suppress semua logging dari preprocessing ke console
        preprocessing_loggers = [
            'smartcash.dataset.services.preprocessor',
            'smartcash.dataset.utils',
            'cv2',
            'PIL'
        ]
        
        for logger_name in preprocessing_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
    
    @property
    def preprocessor(self) -> DatasetPreprocessor:
        """Lazy initialization of preprocessor dengan UI progress callback."""
        if self._preprocessor is None:
            # Get config from config manager
            config = self.config_manager.get_config()
            
            # Create integrated config
            integrated_config = {
                'preprocessing': {
                    'img_size': config.get('preprocessing', {}).get('img_size', DEFAULT_IMG_SIZE),
                    'output_dir': str(self.preprocessed_dir),
                    'normalization': {
                        'enabled': config.get('preprocessing', {}).get('normalize', True),
                        'preserve_aspect_ratio': config.get('preprocessing', {}).get('preserve_aspect_ratio', True)
                    }
                },
                'data': {
                    'dir': str(self.base_dir)
                }
            }
            
            # Initialize preprocessor
            self._preprocessor = DatasetPreprocessor(
                config=integrated_config,
                logger=self.logger
            )
            
            # Register progress callback untuk UI notifications
            if self.progress_callback:
                self._preprocessor.register_progress_callback(self.progress_callback)
        
        return self._preprocessor
    
    def preprocess_dataset(self, split: str = 'all', force_reprocess: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Preprocess dataset dan simpan hasilnya dengan UI progress notifications.
        
        Args:
            split: Split dataset ('train', 'val', 'test', 'all')
            force_reprocess: Paksa proses ulang meskipun sudah ada
            **kwargs: Parameter tambahan untuk preprocessing
            
        Returns:
            Dict: Statistik hasil preprocessing
        """
        try:
            # Notify start melalui progress callback
            if self.progress_callback:
                self.progress_callback(
                    progress=0,
                    total=100,
                    message=f"Memulai preprocessing dataset {split}",
                    status="info",
                    step=0,
                    split_step="Persiapan"
                )
            
            # Force disable show_progress untuk prevent tqdm
            kwargs['show_progress'] = False
            
            # Get preprocessor and run preprocessing
            result = self.preprocessor.preprocess_dataset(
                split=split,
                force_reprocess=force_reprocess,
                **kwargs
            )
            
            # Notify completion melalui progress callback
            if self.progress_callback and result:
                total_images = result.get('total_images', 0)
                processing_time = result.get('processing_time', 0)
                
                self.progress_callback(
                    progress=100,
                    total=100,
                    message=f"Preprocessing selesai: {total_images} gambar dalam {processing_time:.1f} detik",
                    status="success",
                    step=3,
                    split_step="Selesai"
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Error saat preprocessing dataset: {str(e)}"
            
            # Notify error melalui progress callback
            if self.progress_callback:
                self.progress_callback(
                    progress=0,
                    total=100,
                    message=error_msg,
                    status="error",
                    step=0,
                    split_step="Error"
                )
            
            raise DatasetError(error_msg)
    
    def register_progress_callback(self, callback: Callable):
        """Register progress callback untuk UI notifications."""
        self.progress_callback = callback
        
        # Register ke preprocessor jika sudah ada
        if self._preprocessor:
            self._preprocessor.register_progress_callback(callback)
    
    def cleanup(self):
        """Cleanup resources."""
        self._preprocessor = None
        self.progress_callback = None