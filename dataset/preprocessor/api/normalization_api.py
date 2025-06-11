"""
File: smartcash/dataset/preprocessor/api/normalization_api.py
Deskripsi: Updated normalization API menggunakan FileNamingManager
"""

import numpy as np
from typing import Dict, Any, Tuple, Union, Optional, Callable
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.common.utils.file_naming_manager import create_file_naming_manager
from ..core.normalizer import YOLONormalizer
from ..config.defaults import NORMALIZATION_PRESETS
from ..config.validator import validate_normalization_preset

def normalize_for_yolo(image: np.ndarray, 
                      preset: str = 'default',
                      **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """🎯 Normalize image untuk YOLOv5"""
    try:
        validate_normalization_preset(preset)
        config = NORMALIZATION_PRESETS[preset].copy()
        config.update(kwargs)
        
        normalizer = YOLONormalizer(config)
        return normalizer.normalize(image)
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Normalization error: {str(e)}")
        raise ValueError(f"Normalization failed: {str(e)}")

def denormalize_for_visualization(normalized_image: np.ndarray, 
                                metadata: Dict[str, Any]) -> np.ndarray:
    """🔄 Denormalize untuk visualization"""
    try:
        norm_info = metadata.get('normalization', {})
        target_size = metadata.get('target_shape', [640, 640])
        
        config = {
            'target_size': target_size,
            'pixel_range': norm_info.get('pixel_range', [0, 1]),
            'preserve_aspect_ratio': True
        }
        
        normalizer = YOLONormalizer(config)
        return normalizer.denormalize(normalized_image, metadata)
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"⚠️ Denormalization warning: {str(e)}")
        return (normalized_image * 255).astype(np.uint8)

def batch_normalize_for_yolo(images: list, 
                           preset: str = 'default',
                           **kwargs) -> Tuple[np.ndarray, list]:
    """📦 Batch normalize multiple images"""
    config = NORMALIZATION_PRESETS[preset].copy()
    config.update(kwargs)
    config['batch_processing'] = True
    
    normalizer = YOLONormalizer(config)
    return normalizer.batch_normalize(images)

def transform_coordinates_for_yolo(coordinates: np.ndarray,
                                 metadata: Dict[str, Any],
                                 reverse: bool = False) -> np.ndarray:
    """🔧 Transform YOLO coordinates"""
    norm_info = metadata.get('normalization', {})
    target_size = metadata.get('target_shape', [640, 640])
    
    config = {
        'target_size': target_size,
        'pixel_range': norm_info.get('pixel_range', [0, 1]),
        'preserve_aspect_ratio': True
    }
    
    normalizer = YOLONormalizer(config)
    return normalizer.transform_coordinates(coordinates, metadata, reverse)

def create_normalizer(preset: str = 'default', **kwargs) -> YOLONormalizer:
    """🏭 Create normalizer instance"""
    validate_normalization_preset(preset)
    config = NORMALIZATION_PRESETS[preset].copy()
    config.update(kwargs)
    return YOLONormalizer(config)

def get_normalization_info(preset: str = 'default') -> Dict[str, Any]:
    """📋 Get normalization preset info"""
    validate_normalization_preset(preset)
    return NORMALIZATION_PRESETS[preset].copy()

def list_available_presets() -> list:
    """📜 List available presets"""
    return list(NORMALIZATION_PRESETS.keys())

def normalize_image_file(image_path: Union[str, Path],
                        output_path: Union[str, Path] = None,
                        preset: str = 'default',
                        save_metadata: bool = True,
                        **kwargs) -> Dict[str, Any]:
    """📁 Normalize image file menggunakan naming manager"""
    try:
        from ..core.file_processor import FileProcessor
        
        naming_manager = create_file_naming_manager()
        fp = FileProcessor()
        image = fp.read_image(image_path)
        
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Normalize
        normalized, metadata = normalize_for_yolo(image, preset, **kwargs)
        
        # Generate output path menggunakan naming manager
        if output_path is None:
            input_path = Path(image_path)
            output_filename = naming_manager.generate_corresponding_filename(
                input_path.name, 'preprocessed', '.npy'
            )
            output_path = input_path.parent / output_filename
        
        # Save normalized array
        success = fp.save_normalized_array(output_path, normalized, metadata if save_metadata else None)
        
        if not success:
            raise ValueError(f"Failed to save normalized array: {output_path}")
        
        return {
            'success': True,
            'input_path': str(image_path),
            'output_path': str(output_path),
            'metadata_path': str(Path(output_path).with_suffix('.meta.json')) if save_metadata else None,
            'metadata': metadata
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'input_path': str(image_path)}

def denormalize_npy_file(npy_path: Union[str, Path],
                        output_path: Union[str, Path] = None,
                        format: str = 'jpg') -> Dict[str, Any]:
    """📁 Denormalize .npy file menggunakan naming manager"""
    try:
        from ..core.file_processor import FileProcessor
        import cv2
        
        naming_manager = create_file_naming_manager()
        fp = FileProcessor()
        
        # Load normalized array
        normalized, metadata = fp.load_normalized_array(npy_path)
        if normalized is None:
            raise ValueError(f"Cannot load normalized array: {npy_path}")
        
        # Denormalize
        denormalized = denormalize_for_visualization(normalized, metadata or {})
        
        # Generate output path menggunakan naming manager
        if output_path is None:
            input_path = Path(npy_path)
            output_filename = naming_manager.generate_corresponding_filename(
                input_path.name, 'sample', f'.{format}'
            )
            output_path = input_path.parent / output_filename
        
        # Save image
        success = cv2.imwrite(str(output_path), cv2.cvtColor(denormalized, cv2.COLOR_RGB2BGR))
        
        if not success:
            raise ValueError(f"Failed to save image: {output_path}")
        
        return {
            'success': True,
            'input_path': str(npy_path),
            'output_path': str(output_path),
            'image_shape': denormalized.shape
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'input_path': str(npy_path)}