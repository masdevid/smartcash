"""
File: smartcash/dataset/services/preprocessor/preprocessor_factory.py
Deskripsi: Factory pattern untuk membuat instansi DatasetPreprocessor dengan konfigurasi yang tepat
"""

from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.dataset.services.preprocessor.dataset_preprocessor import DatasetPreprocessor
from smartcash.dataset.utils.dataset_constants import DEFAULT_IMG_SIZE, DEFAULT_PREPROCESSED_DIR

class PreprocessorFactory:
    """Factory untuk membuat dan mengkonfigurasi DatasetPreprocessor."""
    
    @staticmethod
    def create_preprocessor(config: Dict[str, Any], logger=None) -> DatasetPreprocessor:
        """
        Buat dan konfigurasi DatasetPreprocessor.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger untuk logging (opsional)
            
        Returns:
            DatasetPreprocessor yang sudah dikonfigurasi
        """
        logger = logger or get_logger()
        
        # Ekstrak konfigurasi preprocessing
        preprocess_config = config.get('preprocessing', {})
        
        # Buat konfigurasi terintegrasi untuk preprocessor
        integrated_config = {
            'preprocessing': {
                'img_size': preprocess_config.get('img_size', DEFAULT_IMG_SIZE),
                'output_dir': preprocess_config.get('output_dir', DEFAULT_PREPROCESSED_DIR),
                'normalization': {
                    'enabled': preprocess_config.get('normalize', True),
                    'preserve_aspect_ratio': preprocess_config.get('preserve_aspect_ratio', True)
                },
                'file_prefix': preprocess_config.get('file_prefix', 'rp')
            },
            'data': {
                'dir': config.get('data', {}).get('dir', 'data'),
                'local': config.get('data', {}).get('local', {})
            }
        }
        
        # Buat dan kembalikan preprocessor
        return DatasetPreprocessor(integrated_config, logger)