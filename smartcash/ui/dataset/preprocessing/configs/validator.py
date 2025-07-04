"""
File: smartcash/ui/dataset/preprocessing/configs/validator.py
Deskripsi: Config validation utilities untuk preprocessing module.
"""

from typing import Dict, Any, List
import logging


def validate_preprocessing_config(config: Dict[str, Any]) -> bool:
    """Validate preprocessing configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    logger = logging.getLogger("preprocessing.config")
    logger.debug("🔍 Validating preprocessing config")
    
    try:
        # Extract config sections
        preprocessing_config = config.get('preprocessing', {})
        cleanup_config = config.get('cleanup', {})
        data_config = config.get('data', {})
        
        # Validate preprocessing settings
        if 'batch_size' in preprocessing_config:
            batch_size = preprocessing_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                logger.warning(f"⚠️ Invalid batch_size: {batch_size}, should be positive integer")
                return False
                
        if 'resolution' in preprocessing_config:
            resolution = preprocessing_config['resolution']
            if not isinstance(resolution, str) or 'x' not in resolution:
                logger.warning(f"⚠️ Invalid resolution format: {resolution}, should be WxH")
                return False
                
            # Check resolution format
            try:
                width, height = resolution.split('x')
                int(width), int(height)  # Try to convert to int
            except (ValueError, TypeError):
                logger.warning(f"⚠️ Invalid resolution values: {resolution}, should be integers")
                return False
                
        if 'normalization' in preprocessing_config:
            normalization = preprocessing_config['normalization']
            valid_normalizations = ['minmax', 'standard', 'none']
            if normalization not in valid_normalizations:
                logger.warning(f"⚠️ Invalid normalization: {normalization}, should be one of {valid_normalizations}")
                return False
                
        if 'target_splits' in preprocessing_config:
            target_splits = preprocessing_config['target_splits']
            valid_splits = ['train', 'valid', 'test']
            
            if not isinstance(target_splits, list):
                logger.warning(f"⚠️ Invalid target_splits: {target_splits}, should be a list")
                return False
                
            for split in target_splits:
                if split not in valid_splits:
                    logger.warning(f"⚠️ Invalid split: {split}, should be one of {valid_splits}")
                    return False
        
        # Validate cleanup settings
        if 'target' in cleanup_config:
            target = cleanup_config['target']
            valid_targets = ['preprocessed', 'original', 'all']
            if target not in valid_targets:
                logger.warning(f"⚠️ Invalid cleanup target: {target}, should be one of {valid_targets}")
                return False
        
        logger.debug("✅ Config validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating config: {str(e)}")
        return False
