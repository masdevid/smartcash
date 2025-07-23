"""
File: smartcash/dataset/preprocessor/api/readiness_api.py
Deskripsi: Readiness check API untuk preprocessing service
"""

from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import os

from smartcash.common.logger import get_logger


def check_service_readiness(data_dir: Union[str, Path] = "data") -> Dict[str, Any]:
    """
    Enhanced backend service readiness check.
    
    Check both raw data availability and preprocessed directories:
    - Raw data: /data/{train, valid, test}/{labels, images}
    - Preprocessed structure: /data/preprocessed/{train, valid, test}/{labels, images}
    
    Args:
        data_dir: Base data directory path
        
    Returns:
        Dict with readiness status and details
        
    Example:
        >>> result = check_service_readiness('data')
        >>> if result['ready']:
        >>>     print("Service ready for operations")
    """
    try:
        logger = get_logger(__name__)
        data_path = Path(data_dir)
        preprocessed_path = data_path / "preprocessed"
        
        # Check raw data availability first (rp_* images)
        raw_data_status = _check_raw_data_availability(data_path)
        
        # Check augmented data (aug_* regular images before normalization)
        augmented_data_status = _check_augmented_data_availability(data_path)
        
        # Check preprocessed directory structure (pre_* and aug_* .npy files)
        preprocessed_status = _check_preprocessed_structure(preprocessed_path)
        
        # Service is ready if any of the data stages exist
        ready = (raw_data_status['has_raw_data'] or 
                augmented_data_status['has_augmented_data'] or 
                preprocessed_status['structure_ready'])
        
        # Determine readiness message based on what's available
        total_files = (raw_data_status['total_raw_files'] + 
                      augmented_data_status['total_augmented_files'] + 
                      preprocessed_status['total_preprocessed_files'])
        
        available_stages = []
        if raw_data_status['has_raw_data']:
            available_stages.append(f"raw data ({raw_data_status['total_raw_files']} rp_* files)")
        if augmented_data_status['has_augmented_data']:
            available_stages.append(f"augmented data ({augmented_data_status['total_augmented_files']} aug_* files)")
        if preprocessed_status['structure_ready']:
            available_stages.append(f"preprocessed structure ({preprocessed_status['total_preprocessed_files']} .npy files)")
        
        if available_stages:
            message = f"✅ Service ready - {', '.join(available_stages)} available"
        else:
            message = "❌ Service not ready - no data found (missing rp_*, aug_*, pre_* files)"
        
        return {
            'success': True,
            'ready': ready,
            'has_raw_data': raw_data_status['has_raw_data'],
            'has_augmented_data': augmented_data_status['has_augmented_data'],
            'has_preprocessed_structure': preprocessed_status['structure_ready'],
            'message': message,
            'details': {
                'raw_data': raw_data_status,
                'augmented_data': augmented_data_status,
                'preprocessed': preprocessed_status,
                'total_files': total_files
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking service readiness: {str(e)}")
        return {
            'success': False,
            'ready': False,
            'has_raw_data': False,
            'has_preprocessed_structure': False,
            'message': f'Error checking readiness: {str(e)}',
            'details': {}
        }


def _check_raw_data_availability(data_path: Path) -> Dict[str, Any]:
    """Check for raw data availability in /data/{train,valid,test}/{images,labels}.
    
    Checks for raw images with prefix 'rp_*' in regular image formats.
    """
    splits = ['train', 'valid', 'test']
    raw_data_status = {}
    total_raw_files = 0
    
    for split in splits:
        split_path = data_path / split
        images_path = split_path / "images"
        labels_path = split_path / "labels"
        
        # Count raw data files with rp_* prefix
        images_count = 0
        if images_path.exists():
            # Check for raw images with rp_* prefix in common formats
            image_patterns = ["rp_*.jpg", "rp_*.jpeg", "rp_*.png", "rp_*.bmp", "rp_*.tiff"]
            for pattern in image_patterns:
                images_count += len(list(images_path.glob(pattern)))
        
        labels_count = len(list(labels_path.glob("*.txt"))) if labels_path.exists() else 0
        
        raw_data_status[split] = {
            'split_exists': split_path.exists(),
            'images_dir_exists': images_path.exists(),
            'labels_dir_exists': labels_path.exists(),
            'raw_images_count': images_count,
            'labels_count': labels_count,
            'has_data': images_count > 0 or labels_count > 0
        }
        
        total_raw_files += images_count + labels_count
    
    has_raw_data = total_raw_files > 0
    
    return {
        'has_raw_data': has_raw_data,
        'total_raw_files': total_raw_files,
        'splits_status': raw_data_status
    }


def _check_augmented_data_availability(data_path: Path) -> Dict[str, Any]:
    """Check for augmented data availability in /data/augmented/{train,valid,test}/{images,labels}.
    
    Checks for augmented images with prefix 'aug_*' in regular image formats.
    """
    augmented_path = data_path / "augmented"
    if not augmented_path.exists():
        return {
            'has_augmented_data': False,
            'total_augmented_files': 0,
            'splits_status': {}
        }
    
    splits = ['train', 'valid', 'test']
    augmented_data_status = {}
    total_augmented_files = 0
    
    for split in splits:
        split_path = augmented_path / split
        images_path = split_path / "images"
        labels_path = split_path / "labels"
        
        # Count augmented data files with aug_* prefix
        images_count = 0
        if images_path.exists():
            # Check for augmented images with aug_* prefix in common formats
            image_patterns = ["aug_*.jpg", "aug_*.jpeg", "aug_*.png", "aug_*.bmp", "aug_*.tiff"]
            for pattern in image_patterns:
                images_count += len(list(images_path.glob(pattern)))
        
        labels_count = len(list(labels_path.glob("*.txt"))) if labels_path.exists() else 0
        
        augmented_data_status[split] = {
            'split_exists': split_path.exists(),
            'images_dir_exists': images_path.exists(),
            'labels_dir_exists': labels_path.exists(),
            'augmented_images_count': images_count,
            'labels_count': labels_count,
            'has_data': images_count > 0 or labels_count > 0
        }
        
        total_augmented_files += images_count + labels_count
    
    has_augmented_data = total_augmented_files > 0
    
    return {
        'has_augmented_data': has_augmented_data,
        'total_augmented_files': total_augmented_files,
        'splits_status': augmented_data_status
    }


def _check_preprocessed_structure(preprocessed_path: Path) -> Dict[str, Any]:
    """Check for preprocessed directory structure with pre_* and aug_* prefixed .npy files."""
    if not preprocessed_path.exists():
        return {
            'structure_ready': False,
            'total_preprocessed_files': 0,
            'splits_status': {}
        }
    
    splits = ['train', 'valid', 'test']
    splits_status = {}
    total_preprocessed_files = 0
    
    for split in splits:
        split_path = preprocessed_path / split
        images_path = split_path / "images"
        labels_path = split_path / "labels"
        
        # Count preprocessed .npy files with pre_* and aug_* prefixes
        preprocessed_count = 0
        augmented_count = 0
        if images_path.exists():
            preprocessed_count = len(list(images_path.glob("pre_*.npy")))
            augmented_count = len(list(images_path.glob("aug_*.npy")))
        
        total_images = preprocessed_count + augmented_count
        labels_count = len(list(labels_path.glob("*.txt"))) if labels_path.exists() else 0
        
        splits_status[split] = {
            'exists': split_path.exists(),
            'images_dir_exists': images_path.exists(),
            'labels_dir_exists': labels_path.exists(),
            'preprocessed_count': preprocessed_count,
            'augmented_count': augmented_count,
            'total_images': total_images,
            'labels_count': labels_count,
            'has_data': total_images > 0 or labels_count > 0
        }
        
        total_preprocessed_files += total_images + labels_count
    
    # Structure is ready if all directories exist (regardless of content)
    all_dirs_exist = all(split_info['exists'] for split_info in splits_status.values())
    structure_ready = preprocessed_path.exists() and all_dirs_exist
    
    return {
        'structure_ready': structure_ready,
        'total_preprocessed_files': total_preprocessed_files,
        'splits_status': splits_status
    }


def check_existing_data(data_dir: Union[str, Path] = "data", 
                       splits: List[str] = None) -> Dict[str, Any]:
    """
    Check for existing preprocessed data in specific splits.
    
    Args:
        data_dir: Base data directory path
        splits: List of splits to check (default: ['train', 'valid', 'test'])
        
    Returns:
        Dict with existing data information
    """
    try:
        logger = get_logger(__name__)
        data_path = Path(data_dir)
        preprocessed_path = data_path / "preprocessed"
        splits = splits or ['train', 'valid', 'test']
        
        existing_data = {}
        total_existing_files = 0
        
        for split in splits:
            split_path = preprocessed_path / split
            images_path = split_path / "images"
            labels_path = split_path / "labels"
            
            if split_path.exists():
                # Count existing files
                image_files = list(images_path.glob("*.npy")) if images_path.exists() else []
                label_files = list(labels_path.glob("*.txt")) if labels_path.exists() else []
                
                existing_data[split] = {
                    'images': len(image_files),
                    'labels': len(label_files),
                    'total': len(image_files) + len(label_files),
                    'sample_files': {
                        'images': [f.name for f in image_files[:3]],  # Show first 3 as samples
                        'labels': [f.name for f in label_files[:3]]
                    }
                }
                
                total_existing_files += len(image_files) + len(label_files)
            else:
                existing_data[split] = {
                    'images': 0,
                    'labels': 0,
                    'total': 0,
                    'sample_files': {'images': [], 'labels': []}
                }
        
        has_existing_data = total_existing_files > 0
        
        return {
            'success': True,
            'has_existing_data': has_existing_data,
            'total_existing_files': total_existing_files,
            'splits_data': existing_data,
            'message': _generate_existing_data_message(has_existing_data, total_existing_files)
        }
        
    except Exception as e:
        logger.error(f"Error checking existing data: {str(e)}")
        return {
            'success': False,
            'has_existing_data': False,
            'total_existing_files': 0,
            'splits_data': {},
            'message': f'Error checking existing data: {str(e)}'
        }


def get_preprocessing_directory_info(data_dir: Union[str, Path] = "data") -> Dict[str, Any]:
    """
    Get detailed information about preprocessing directories.
    
    Args:
        data_dir: Base data directory path
        
    Returns:
        Dict with directory structure information
    """
    try:
        data_path = Path(data_dir)
        preprocessed_path = data_path / "preprocessed"
        
        directory_info = {
            'base_dir': {
                'path': str(data_path),
                'exists': data_path.exists(),
                'readable': os.access(data_path, os.R_OK) if data_path.exists() else False,
                'writable': os.access(data_path, os.W_OK) if data_path.exists() else False
            },
            'preprocessed_dir': {
                'path': str(preprocessed_path),
                'exists': preprocessed_path.exists(),
                'readable': os.access(preprocessed_path, os.R_OK) if preprocessed_path.exists() else False,
                'writable': os.access(preprocessed_path, os.W_OK) if preprocessed_path.exists() else False
            },
            'splits': {}
        }
        
        # Check each split directory
        splits = ['train', 'valid', 'test']
        for split in splits:
            split_path = preprocessed_path / split
            images_path = split_path / "images"
            labels_path = split_path / "labels"
            
            directory_info['splits'][split] = {
                'split_dir': {
                    'path': str(split_path),
                    'exists': split_path.exists(),
                    'readable': os.access(split_path, os.R_OK) if split_path.exists() else False,
                    'writable': os.access(split_path, os.W_OK) if split_path.exists() else False
                },
                'images_dir': {
                    'path': str(images_path),
                    'exists': images_path.exists(),
                    'readable': os.access(images_path, os.R_OK) if images_path.exists() else False,
                    'writable': os.access(images_path, os.W_OK) if images_path.exists() else False
                },
                'labels_dir': {
                    'path': str(labels_path),
                    'exists': labels_path.exists(),
                    'readable': os.access(labels_path, os.R_OK) if labels_path.exists() else False,
                    'writable': os.access(labels_path, os.W_OK) if labels_path.exists() else False
                }
            }
        
        return {
            'success': True,
            'directory_info': directory_info
        }
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error getting directory info: {str(e)}")
        return {
            'success': False,
            'message': f'Error getting directory info: {str(e)}',
            'directory_info': {}
        }


def _generate_readiness_message(ready: bool, has_data: bool, total_files: int) -> str:
    """Generate human-readable readiness message."""
    if not ready:
        return "❌ Service not ready - preprocessed directory structure missing"
    elif not has_data:
        return "⚠️ Service ready but no preprocessed data found"
    else:
        return f"✅ Service ready with {total_files} preprocessed files"


def _generate_existing_data_message(has_data: bool, total_files: int) -> str:
    """Generate human-readable existing data message."""
    if not has_data:
        return "No existing preprocessed data found"
    else:
        return f"Found {total_files} existing preprocessed files"