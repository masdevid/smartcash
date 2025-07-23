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
    Quick backend service readiness check.
    
    Check if preprocessed directories exist and contain data:
    - /dataset/preprocessed/{train, valid, test}/{labels, images}
    
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
        
        # Check if preprocessed directory exists
        if not preprocessed_path.exists():
            return {
                'success': True,
                'ready': False,
                'message': 'Preprocessed directory does not exist',
                'details': {
                    'preprocessed_dir_exists': False,
                    'splits_status': {},
                    'total_files': 0
                }
            }
        
        # Check each split (train, valid, test)
        splits = ['train', 'valid', 'test']
        splits_status = {}
        total_files = 0
        
        for split in splits:
            split_path = preprocessed_path / split
            images_path = split_path / "images"
            labels_path = split_path / "labels"
            
            # Count files in each subdirectory
            images_count = len(list(images_path.glob("*.npy"))) if images_path.exists() else 0
            labels_count = len(list(labels_path.glob("*.txt"))) if labels_path.exists() else 0
            
            splits_status[split] = {
                'exists': split_path.exists(),
                'images_dir_exists': images_path.exists(),
                'labels_dir_exists': labels_path.exists(),
                'images_count': images_count,
                'labels_count': labels_count,
                'has_data': images_count > 0 or labels_count > 0
            }
            
            total_files += images_count + labels_count
        
        # Determine overall readiness
        has_any_data = any(split_info['has_data'] for split_info in splits_status.values())
        all_dirs_exist = all(split_info['exists'] for split_info in splits_status.values())
        
        # Service is ready if preprocessed directory exists and has basic structure
        ready = preprocessed_path.exists() and all_dirs_exist
        
        return {
            'success': True,
            'ready': ready,
            'has_data': has_any_data,
            'message': _generate_readiness_message(ready, has_any_data, total_files),
            'details': {
                'preprocessed_dir_exists': preprocessed_path.exists(),
                'preprocessed_dir_path': str(preprocessed_path),
                'splits_status': splits_status,
                'total_files': total_files
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking service readiness: {str(e)}")
        return {
            'success': False,
            'ready': False,
            'has_data': False,
            'message': f'Error checking readiness: {str(e)}',
            'details': {}
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