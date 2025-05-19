"""
File: smartcash/dataset/services/preprocessor/preprocessing_service.py
Deskripsi: Layanan preprocessing dataset dengan integrasi observer pattern
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
    """Layanan preprocessing dataset dengan integrasi observer pattern."""
    
    def __init__(self, config: Dict[str, Any], logger=None, observer_manager=None):
        """
        Inisialisasi PreprocessingService.
        
        Args:
            config: Konfigurasi preprocessing
            logger: Logger untuk logging
            observer_manager: Observer manager untuk UI notifications
        """
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.observer_manager = observer_manager
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
        
        # Log initialization
        self.logger.info(f"✅ PreprocessingService diinisialisasi:")
        self.logger.info(f"  - Base dir: {self.base_dir}")
        self.logger.info(f"  - Preprocessed dir: {self.preprocessed_dir}")
    
    @property
    def preprocessor(self) -> DatasetPreprocessor:
        """Lazy initialization of preprocessor."""
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
            
            # Register progress callback if observer manager exists
            if self.observer_manager:
                self._preprocessor.register_progress_callback(
                    lambda **kwargs: self._notify_progress(**kwargs)
                )
        
        return self._preprocessor
    
    def _notify_progress(self, **kwargs):
        """Notify progress through observer manager."""
        if self.observer_manager:
            from smartcash.dataset.services.downloader.notification_utils import notify_service_event
            notify_service_event(
                "preprocessing",
                "progress",
                self,
                self.observer_manager,
                **kwargs
            )
    
    def preprocess_dataset(self, split: str = 'all', force_reprocess: bool = False, **kwargs) -> Dict[str, Any]:
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
            # Get latest config
            config = self.config_manager.get_config()
            
            # Notify start
            if self.observer_manager:
                self._notify_progress(
                    message=f"Memulai preprocessing dataset {split}",
                    step="start"
                )
            
            # Get preprocessor and run preprocessing
            result = self.preprocessor.preprocess_dataset(
                split=split,
                force_reprocess=force_reprocess,
                **kwargs
            )
            
            # Notify completion
            if self.observer_manager:
                self._notify_progress(
                    message=f"Preprocessing dataset {split} selesai",
                    step="complete",
                    result=result
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Error saat preprocessing dataset: {str(e)}"
            self.logger.error(error_msg)
            
            # Notify error
            if self.observer_manager:
                self._notify_progress(
                    message=error_msg,
                    step="error"
                )
            
            raise DatasetError(error_msg)
    
    def cleanup(self):
        """Cleanup resources."""
        self._preprocessor = None
        self.observer_manager = None 