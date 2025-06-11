"""
File: smartcash/dataset/preprocessor/api/normalization_api.py
Deskripsi: Standalone normalization API untuk reuse di modules lain
"""

import numpy as np
from typing import Dict, Any, Tuple, Union, Optional
from pathlib import Path

from smartcash.common.logger import get_logger
from ..core.normalizer import YOLONormalizer
from ..config.defaults import NORMALIZATION_PRESETS
from ..config.validator import validate_normalization_preset

def normalize_for_yolo(image: np.ndarray, 
                      preset: str = 'default',
                      **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """🎯 Normalize image untuk YOLOv5 inference/training
    
    Args:
        image: Input image array (RGB format)
        preset: Normalization preset ('default', 'yolov5s', 'yolov5m', etc)
        **kwargs: Override preset parameters
        
    Returns:
        Tuple of (normalized_image, metadata)
        
    Example:
        >>> normalized, meta = normalize_for_yolo(image, 'yolov5s')
        >>> # For custom augmentation module
        >>> normalized, meta = normalize_for_yolo(image, 'inference', target_size=[832, 832])
    """
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
    """🔄 Denormalize untuk visualization
    
    Args:
        normalized_image: Normalized image array
        metadata: Metadata dari normalize_for_yolo()
        
    Returns:
        Denormalized image untuk display
        
    Example:
        >>> original = denormalize_for_visualization(normalized, metadata)
        >>> # Display atau save untuk preview
    """
    try:
        # Extract config dari metadata
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
        # Fallback: simple pixel denormalization
        return (normalized_image * 255).astype(np.uint8)

def batch_normalize_for_yolo(images: list, 
                           preset: str = 'default',
                           **kwargs) -> Tuple[np.ndarray, list]:
    """📦 Batch normalize multiple images
    
    Args:
        images: List of image arrays
        preset: Normalization preset
        **kwargs: Override parameters
        
    Returns:
        Tuple of (batch_array, metadata_list)
    """
    config = NORMALIZATION_PRESETS[preset].copy()
    config.update(kwargs)
    config['batch_processing'] = True
    
    normalizer = YOLONormalizer(config)
    return normalizer.batch_normalize(images)

def transform_coordinates_for_yolo(coordinates: np.ndarray,
                                 metadata: Dict[str, Any],
                                 reverse: bool = False) -> np.ndarray:
    """🔧 Transform YOLO coordinates
    
    Args:
        coordinates: YOLO format coordinates [class, x, y, w, h]
        metadata: Metadata dari normalize_for_yolo()
        reverse: True untuk normalized→original, False untuk original→normalized
        
    Returns:
        Transformed coordinates
        
    Example:
        >>> # Training: original → normalized
        >>> norm_coords = transform_coordinates_for_yolo(orig_coords, metadata, reverse=False)
        >>> # Inference: normalized → original
        >>> orig_coords = transform_coordinates_for_yolo(norm_coords, metadata, reverse=True)
    """
    # Extract config dari metadata untuk create normalizer
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
    """🏭 Create normalizer instance untuk advanced use
    
    Args:
        preset: Normalization preset
        **kwargs: Override parameters
        
    Returns:
        YOLONormalizer instance
        
    Example:
        >>> normalizer = create_normalizer('yolov5l', target_size=[832, 832])
        >>> # Use for multiple operations
        >>> normalized1, meta1 = normalizer.normalize(image1)
        >>> normalized2, meta2 = normalizer.normalize(image2)
    """
    validate_normalization_preset(preset)
    config = NORMALIZATION_PRESETS[preset].copy()
    config.update(kwargs)
    return YOLONormalizer(config)

def get_normalization_info(preset: str = 'default') -> Dict[str, Any]:
    """📋 Get normalization preset information
    
    Args:
        preset: Preset name
        
    Returns:
        Preset configuration dict
    """
    validate_normalization_preset(preset)
    return NORMALIZATION_PRESETS[preset].copy()

def list_available_presets() -> list:
    """📜 List available normalization presets"""
    return list(NORMALIZATION_PRESETS.keys())

# === File-based operations ===

def normalize_image_file(image_path: Union[str, Path],
                        output_path: Union[str, Path] = None,
                        preset: str = 'default',
                        save_metadata: bool = True,
                        **kwargs) -> Dict[str, Any]:
    """📁 Normalize image file dan save hasil
    
    Args:
        image_path: Path ke input image
        output_path: Path untuk output .npy (auto-generated jika None)
        preset: Normalization preset
        save_metadata: Save metadata ke .meta.json
        **kwargs: Override parameters
        
    Returns:
        Dict dengan output paths dan metadata
    """
    try:
        from ..core.file_processor import FileProcessor
        
        fp = FileProcessor()
        image = fp.read_image(image_path)
        
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Normalize
        normalized, metadata = normalize_for_yolo(image, preset, **kwargs)
        
        # Determine output path
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.with_suffix('.npy')
        
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
        return {
            'success': False,
            'error': str(e),
            'input_path': str(image_path)
        }

def denormalize_npy_file(npy_path: Union[str, Path],
                        output_path: Union[str, Path] = None,
                        format: str = 'jpg') -> Dict[str, Any]:
    """📁 Denormalize .npy file ke image untuk visualization
    
    Args:
        npy_path: Path ke .npy file
        output_path: Output image path (auto-generated jika None)
        format: Output format ('jpg', 'png')
        
    Returns:
        Dict dengan output path dan info
    """
    try:
        from ..core.file_processor import FileProcessor
        import cv2
        
        fp = FileProcessor()
        
        # Load normalized array dan metadata
        normalized, metadata = fp.load_normalized_array(npy_path)
        if normalized is None:
            raise ValueError(f"Cannot load normalized array: {npy_path}")
        
        # Denormalize
        denormalized = denormalize_for_visualization(normalized, metadata or {})
        
        # Determine output path
        if output_path is None:
            input_path = Path(npy_path)
            output_path = input_path.with_suffix(f'.{format}')
        
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
        return {
            'success': False,
            'error': str(e),
            'input_path': str(npy_path)
        }