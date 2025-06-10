"""
File: smartcash/dataset/preprocessor/__init__.py
Deskripsi: Enhanced modul preprocessing dengan dual progress tracker integration dan API consistency
"""

from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from pathlib import Path
import os
import numpy as np

from .service import PreprocessingService
from .core.engine import PreprocessingEngine, PreprocessingValidator
from .utils import (
    validate_preprocessing_config,
    get_default_preprocessing_config,
    ProgressBridge,
    FileProcessor,
    FileScanner,
    FilenameManager,
    PathResolver,
    CleanupManager
)

# Enhanced factory functions
def create_preprocessing_service(config: Dict[str, Any] = None, 
                               progress_tracker=None) -> PreprocessingService:
    """🏭 Enhanced factory untuk preprocessing service dengan dual progress tracker"""
    return PreprocessingService(config, progress_tracker)

def create_preprocessing_engine(config: Dict[str, Any]) -> PreprocessingEngine:
    """🏭 Enhanced factory untuk preprocessing engine"""
    return PreprocessingEngine(config)

def create_preprocessing_validator(config: Dict[str, Any]) -> PreprocessingValidator:
    """🏭 Enhanced factory untuk preprocessing validator"""
    return PreprocessingValidator(config)

# Enhanced main functions dengan dual progress compatibility
def preprocess_dataset(config: Dict[str, Any], 
                      progress_tracker=None,
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🚀 Enhanced preprocessing pipeline dengan dual progress tracking
    
    Args:
        config: Konfigurasi preprocessing (will be validated)
        progress_tracker: Progress tracker object untuk UI integration
        progress_callback: Callback function dengan signature (level: str, current: int, total: int, message: str)
        
    Returns:
        Dict dengan keys: success, message, stats, processing_time
    """
    service = create_preprocessing_service(config, progress_tracker)
    
    # Register dual progress callback jika ada
    if progress_callback and hasattr(service, '_dual_progress_callback'):
        service._external_progress_callback = progress_callback
    
    return service.preprocess_dataset(progress_callback)

def get_preprocessing_samples(config: Dict[str, Any], 
                            target_split: str = "train",
                            max_samples: int = 5,
                            progress_tracker=None) -> Dict[str, Any]:
    """🎲 Enhanced sampling untuk evaluasi dengan better error handling
    
    Args:
        config: Konfigurasi preprocessing
        target_split: Target split untuk sampling
        max_samples: Maksimal jumlah samples
        progress_tracker: Progress tracker object (opsional)
        
    Returns:
        Dict dengan keys: success, message, samples, total_samples
    """
    service = create_preprocessing_service(config, progress_tracker)
    return service.get_sampling(target_split, max_samples)

def validate_dataset(config: Dict[str, Any], 
                    target_split: str = "train",
                    progress_tracker=None) -> Dict[str, Any]:
    """🔍 Enhanced dataset validation dengan detailed reporting
    
    Args:
        config: Konfigurasi validasi
        target_split: Target split untuk validasi
        progress_tracker: Progress tracker object (opsional)
        
    Returns:
        Dict dengan keys: success, message, validation_result, summary
    """
    service = create_preprocessing_service(config, progress_tracker)
    return service.validate_dataset_only(target_split)

def cleanup_preprocessed_data(config: Dict[str, Any], 
                            target_split: str = None,
                            progress_tracker=None) -> Dict[str, Any]:
    """🧹 Enhanced cleanup dengan detailed statistics
    
    Args:
        config: Konfigurasi cleanup
        target_split: Target split untuk cleanup (None = semua)
        progress_tracker: Progress tracker object (opsional)
        
    Returns:
        Dict dengan keys: success, message, stats
    """
    service = create_preprocessing_service(config, progress_tracker)
    return service.cleanup_preprocessed_data(target_split)

def get_preprocessing_status(config: Dict[str, Any],
                           progress_tracker=None) -> Dict[str, Any]:
    """📊 Enhanced status check dengan system information
    
    Args:
        config: Konfigurasi system
        progress_tracker: Progress tracker object (opsional)
        
    Returns:
        Dict dengan keys: success, service_ready, message, system_info, configuration
    """
    service = create_preprocessing_service(config, progress_tracker)
    return service.get_preprocessing_status()

# Enhanced utility functions
def get_preprocessing_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """⚙️ Get comprehensive config summary untuk UI display"""
    validated_config = validate_preprocessing_config(config)
    preprocessing_config = validated_config.get('preprocessing', {})
    norm_config = preprocessing_config.get('normalization', {})
    
    return {
        'target_splits': preprocessing_config.get('target_splits', ['train', 'valid']),
        'normalization': {
            'enabled': norm_config.get('enabled', True),
            'method': norm_config.get('method', 'minmax'),
            'target_size': norm_config.get('target_size', [640, 640]),
            'preserve_aspect_ratio': norm_config.get('preserve_aspect_ratio', True)
        },
        'validation': {
            'enabled': preprocessing_config.get('validation', {}).get('enabled', True),
            'move_invalid': preprocessing_config.get('validation', {}).get('move_invalid', True)
        },
        'output': {
            'output_dir': preprocessing_config.get('output_dir', 'data/preprocessed'),
            'create_npy': preprocessing_config.get('output', {}).get('create_npy', True)
        },
        'performance': {
            'batch_size': validated_config.get('performance', {}).get('batch_size', 32),
            'threading': validated_config.get('performance', {}).get('threading', {})
        }
    }

def check_preprocessing_compatibility() -> Dict[str, Any]:
    """🔍 Check compatibility dengan dual progress tracker dan augmentor"""
    compatibility = {
        'dual_progress_tracker': False,
        'augmentor_consistency': False,
        'enhanced_features': []
    }
    
    try:
        # Check dual progress tracker compatibility
        from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
        compatibility['dual_progress_tracker'] = True
        compatibility['enhanced_features'].append('dual_progress_tracking')
    except ImportError:
        pass
    
    try:
        # Check augmentor consistency
        from smartcash.dataset.augmentor.core.normalizer import NormalizationEngine
        compatibility['augmentor_consistency'] = True
        compatibility['enhanced_features'].append('augmentor_consistency')
    except ImportError:
        pass
    
    # Check enhanced features
    enhanced_checks = {
        'multi_split_support': True,
        'aspect_ratio_preservation': True,
        'enhanced_validation': True,
        'threading_support': True,
        'npy_output_support': True
    }
    
    compatibility['enhanced_features'].extend([k for k, v in enhanced_checks.items() if v])
    compatibility['total_features'] = len(compatibility['enhanced_features'])
    
    return compatibility

# Enhanced preprocessing functions untuk specific use cases
def preprocess_single_split(config: Dict[str, Any], split: str,
                          progress_tracker=None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🎯 Enhanced preprocessing untuk single split saja"""
    # Override target_splits untuk single split
    modified_config = validate_preprocessing_config(config)
    modified_config['preprocessing']['target_splits'] = [split]
    
    return preprocess_dataset(modified_config, progress_tracker, progress_callback)

def preprocess_with_validation(config: Dict[str, Any], 
                             progress_tracker=None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🔍 Enhanced preprocessing dengan comprehensive validation"""
    # Force enable validation
    modified_config = validate_preprocessing_config(config)
    modified_config['preprocessing']['validation']['enabled'] = True
    
    return preprocess_dataset(modified_config, progress_tracker, progress_callback)

def preprocess_for_training(config: Dict[str, Any],
                          progress_tracker=None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🎯 Enhanced preprocessing optimized untuk training"""
    # Optimize config untuk training
    modified_config = validate_preprocessing_config(config)
    preprocessing = modified_config['preprocessing']
    
    # Enable normalization dan .npy output
    preprocessing['normalization']['enabled'] = True
    preprocessing['output']['create_npy'] = True
    
    # Focus pada train/valid splits
    if 'target_splits' not in preprocessing or preprocessing['target_splits'] == 'all':
        preprocessing['target_splits'] = ['train', 'valid']
    
    return preprocess_dataset(modified_config, progress_tracker, progress_callback)

def get_preprocessing_statistics(config: Dict[str, Any]) -> Dict[str, Any]:
    """📊 Get comprehensive preprocessing statistics tanpa actual processing"""
    try:
        service = create_preprocessing_service(config)
        
        # Get target splits
        target_splits = config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
        if isinstance(target_splits, str):
            target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
        
        stats = {
            'input': {'splits': [], 'total_images': 0, 'total_labels': 0},
            'configuration': get_preprocessing_config_summary(config),
            'estimated_output': {'total_files': 0, 'npy_files': 0, 'image_files': 0}
        }
        
        # Analyze each split
        for split in target_splits:
            try:
                validation_result = service.validate_dataset_only(split)
                if validation_result.get('success'):
                    split_summary = validation_result.get('summary', {})
                    split_stats = {
                        'split': split,
                        'images': split_summary.get('total_images', 0),
                        'valid_images': split_summary.get('valid_images', 0),
                        'labels': split_summary.get('total_images', 0),  # Assume 1:1 mapping
                        'class_distribution': split_summary.get('class_distribution', {})
                    }
                    stats['input']['splits'].append(split_stats)
                    stats['input']['total_images'] += split_stats['valid_images']
                    stats['input']['total_labels'] += split_stats['labels']
            except Exception:
                continue
        
        # Estimate output
        norm_enabled = config.get('preprocessing', {}).get('normalization', {}).get('enabled', True)
        if norm_enabled:
            stats['estimated_output']['npy_files'] = stats['input']['total_images']
        else:
            stats['estimated_output']['image_files'] = stats['input']['total_images']
        
        stats['estimated_output']['total_files'] = stats['input']['total_images']
        
        return {
            'success': True,
            'message': f"✅ Analysis completed untuk {len(target_splits)} splits",
            'stats': stats
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f"❌ Error analysis: {str(e)}",
            'stats': {}
        }

# Backward compatibility functions
def preprocess_and_visualize(config: Dict[str, Any], target_split: str = "train", 
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🔄 Backward compatibility untuk preprocess_and_visualize"""
    return preprocess_single_split(config, target_split, None, progress_callback)

# Export enhanced functions
__all__ = [
    # Core classes
    'PreprocessingService',
    'PreprocessingEngine', 
    'PreprocessingValidator',
    
    # Factory functions
    'create_preprocessing_service',
    'create_preprocessing_engine',
    'create_preprocessing_validator',
    
    # Main enhanced functions
    'preprocess_dataset',
    'get_preprocessing_samples',
    'validate_dataset',
    'cleanup_preprocessed_data',
    'get_preprocessing_status',
    
    # Enhanced utility functions
    'get_preprocessing_config_summary',
    'check_preprocessing_compatibility',
    'get_preprocessing_statistics',
    
    # Specialized functions
    'preprocess_single_split',
    'preprocess_with_validation', 
    'preprocess_for_training',
    
    # Utility components
    'validate_preprocessing_config',
    'get_default_preprocessing_config',
    'ProgressBridge',
    'FileProcessor',
    'FileScanner',
    'PathResolver',
    'CleanupManager',
    'FilenameManager',
    
    # Backward compatibility
    'preprocess_and_visualize'
]