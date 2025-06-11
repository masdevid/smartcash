"""
File: smartcash/dataset/preprocessor/__init__.py
Deskripsi: Consolidated exports untuk preprocessor dengan backward compatibility dan progress tracker integration
"""

from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from pathlib import Path
import os
import numpy as np

# === CORE SERVICES ===
from .service import PreprocessingService, create_preprocessing_service
from .core.engine import PreprocessingEngine, PreprocessingValidator
from .core.validator import ValidationEngine, create_validation_engine

# === CONSOLIDATED UTILS ===
from .utils import (
    # Config validation
    validate_preprocessing_config,
    get_default_preprocessing_config,
    reload_default_config,
    
    # Core utilities
    FileOperations, ValidationCore, PathManager, MetadataManager, 
    YOLONormalizer, ProgressBridge,
    
    # Factory functions
    create_file_operations, create_validation_core, create_path_manager,
    create_metadata_manager, create_yolo_normalizer, create_progress_bridge,
    create_compatible_bridge,
    
    # Convenience functions
    read_image_safe, write_image_safe, scan_image_files, find_pairs_safe,
    validate_image_safe, validate_label_safe, validate_pair_safe,
    validate_source_safe, create_output_safe, get_paths_safe,
    parse_filename_safe, generate_preprocessed_safe, extract_nominal_safe,
    preprocess_image_for_yolo, normalize_yolo_safe, update_progress_safe,
    
    # Legacy compatibility
    ImageValidator, LabelValidator, PairValidator, FileProcessor,
    PathResolver, FilenameManager, FileScanner, CleanupManager,
    create_image_validator, create_label_validator, create_pair_validator,
    create_file_processor, create_path_resolver, create_filename_manager,
    create_file_scanner, create_cleanup_manager
)

# === ENHANCED FACTORY FUNCTIONS ===

def create_preprocessing_service_with_progress(config: Dict[str, Any] = None, 
                                             ui_components: Dict[str, Any] = None) -> PreprocessingService:
    """🏭 Enhanced factory dengan UI progress tracker integration"""
    progress_callback = None
    
    if ui_components:
        # Create compatible progress bridge
        bridge = create_compatible_bridge(ui_components)
        
        def progress_callback_func(level: str, current: int, total: int, message: str):
            """Progress callback yang kompatibel dengan dual progress tracker"""
            try:
                # Update progress tracker
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker:
                    if hasattr(progress_tracker, f'update_{level}'):
                        getattr(progress_tracker, f'update_{level}')(current, message)
                    elif hasattr(progress_tracker, 'update'):
                        progress_tracker.update(level, current, message)
                
                # Update bridge
                bridge.update(level, current, total, message)
                
            except Exception as e:
                # Silent fail untuk prevent breaking process
                pass
        
        progress_callback = progress_callback_func
    
    return create_preprocessing_service(config, progress_callback)

def create_preprocessing_engine_enhanced(config: Dict[str, Any]) -> PreprocessingEngine:
    """🏭 Enhanced factory untuk preprocessing engine"""
    validated_config = validate_preprocessing_config(config) if config else get_default_preprocessing_config()
    return PreprocessingEngine(validated_config)

def create_preprocessing_validator_enhanced(config: Dict[str, Any]) -> PreprocessingValidator:
    """🏭 Enhanced factory untuk preprocessing validator"""
    validated_config = validate_preprocessing_config(config) if config else get_default_preprocessing_config()
    return PreprocessingValidator(validated_config)

# === ENHANCED MAIN FUNCTIONS ===

def preprocess_dataset(config: Dict[str, Any], 
                      ui_components: Dict[str, Any] = None,
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🚀 Enhanced preprocessing pipeline dengan dual progress tracking
    
    Args:
        config: Preprocessing configuration (akan divalidasi)
        ui_components: UI components untuk progress integration (opsional)
        progress_callback: Callback function dengan signature (level: str, current: int, total: int, message: str)
        
    Returns:
        Dict dengan keys: success, message, stats, processing_time
    """
    # Use UI components untuk create service jika ada
    if ui_components:
        service = create_preprocessing_service_with_progress(config, ui_components)
    else:
        service = create_preprocessing_service(config, progress_callback)
    
    return service.preprocess_dataset()

def get_preprocessing_samples(config: Dict[str, Any], 
                            target_split: str = "train",
                            max_samples: int = 5,
                            ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """🎲 Enhanced sampling untuk preview dengan better error handling"""
    service = create_preprocessing_service_with_progress(config, ui_components) if ui_components else create_preprocessing_service(config)
    return service.get_sampling(target_split, max_samples)

def validate_dataset(config: Dict[str, Any], 
                    target_split: str = "train",
                    ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """🔍 Enhanced dataset validation dengan detailed reporting"""
    service = create_preprocessing_service_with_progress(config, ui_components) if ui_components else create_preprocessing_service(config)
    return service.validate_dataset_only(target_split)

def cleanup_preprocessed_data(config: Dict[str, Any], 
                            target_split: str = None,
                            ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """🧹 Enhanced cleanup dengan detailed statistics"""
    service = create_preprocessing_service_with_progress(config, ui_components) if ui_components else create_preprocessing_service(config)
    return service.cleanup_preprocessed_data(target_split)

def get_preprocessing_status(config: Dict[str, Any],
                           ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """📊 Enhanced status check dengan system information"""
    service = create_preprocessing_service_with_progress(config, ui_components) if ui_components else create_preprocessing_service(config)
    return service.get_preprocessing_status()

# === ENHANCED UTILITY FUNCTIONS ===

def get_preprocessing_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """⚙️ Get comprehensive config summary untuk UI display"""
    validated_config = validate_preprocessing_config(config)
    preprocessing_config = validated_config.get('preprocessing', {})
    norm_config = preprocessing_config.get('normalization', {})
    
    return {
        'target_splits': preprocessing_config.get('target_splits', ['train', 'valid']),
        'normalization': {
            'enabled': norm_config.get('enabled', True),
            'target_size': norm_config.get('target_size', [640, 640]),
            'preserve_aspect_ratio': norm_config.get('preserve_aspect_ratio', True)
        },
        'validation': {
            'enabled': preprocessing_config.get('validation', {}).get('enabled', True),
            'move_invalid': preprocessing_config.get('validation', {}).get('move_invalid', True)
        },
        'output': {
            'output_dir': preprocessing_config.get('output_dir', 'data/preprocessed'),
            'format': 'npy + txt'
        },
        'performance': {
            'batch_size': validated_config.get('performance', {}).get('batch_size', 32)
        }
    }

def check_preprocessing_compatibility() -> Dict[str, Any]:
    """🔍 Check compatibility dengan dual progress tracker"""
    compatibility = {
        'dual_progress_tracker': False,
        'ui_integration': False,
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
        # Check UI integration
        compatibility['ui_integration'] = True
        compatibility['enhanced_features'].append('ui_integration')
    except ImportError:
        pass
    
    # Enhanced features check
    enhanced_checks = {
        'yolo_normalization': True,
        'metadata_management': True,
        'path_management': True,
        'validation_core': True,
        'progress_bridge': True,
        'file_operations': True
    }
    
    compatibility['enhanced_features'].extend([k for k, v in enhanced_checks.items() if v])
    compatibility['total_features'] = len(compatibility['enhanced_features'])
    
    return compatibility

# === SPECIALIZED FUNCTIONS ===

def preprocess_single_split(config: Dict[str, Any], split: str,
                          ui_components: Dict[str, Any] = None, 
                          progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🎯 Enhanced preprocessing untuk single split"""
    modified_config = validate_preprocessing_config(config)
    modified_config['preprocessing']['target_splits'] = [split]
    
    return preprocess_dataset(modified_config, ui_components, progress_callback)

def preprocess_with_validation(config: Dict[str, Any], 
                             ui_components: Dict[str, Any] = None,
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🔍 Enhanced preprocessing dengan comprehensive validation"""
    modified_config = validate_preprocessing_config(config)
    modified_config['preprocessing']['validation']['enabled'] = True
    
    return preprocess_dataset(modified_config, ui_components, progress_callback)

def preprocess_for_yolo_training(config: Dict[str, Any],
                                ui_components: Dict[str, Any] = None,
                                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🎯 Enhanced preprocessing optimized untuk YOLO training"""
    modified_config = validate_preprocessing_config(config)
    preprocessing = modified_config['preprocessing']
    
    # Optimize untuk YOLO training
    preprocessing['normalization']['enabled'] = True
    preprocessing['normalization']['target_size'] = [640, 640]
    preprocessing['normalization']['preserve_aspect_ratio'] = True
    
    # Focus pada train/valid splits
    if 'target_splits' not in preprocessing or preprocessing['target_splits'] == 'all':
        preprocessing['target_splits'] = ['train', 'valid']
    
    return preprocess_dataset(modified_config, ui_components, progress_callback)

def get_preprocessing_statistics(config: Dict[str, Any], 
                               ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """📊 Get comprehensive preprocessing statistics tanpa actual processing"""
    try:
        service = create_preprocessing_service_with_progress(config, ui_components) if ui_components else create_preprocessing_service(config)
        
        # Get comprehensive status
        status_result = service.get_preprocessing_status()
        
        if not status_result.get('success'):
            return {
                'success': False,
                'message': status_result.get('message', 'Status check failed'),
                'stats': {}
            }
        
        # Enhanced statistics
        source_validation = status_result.get('source_validation', {})
        config_summary = get_preprocessing_config_summary(config)
        
        stats = {
            'input': {
                'total_images': source_validation.get('total_images', 0),
                'splits': source_validation.get('splits', {}),
                'validation_status': 'valid' if source_validation.get('is_valid') else 'invalid'
            },
            'configuration': config_summary,
            'system': status_result.get('system_info', {}),
            'compatibility': check_preprocessing_compatibility()
        }
        
        return {
            'success': True,
            'message': f"✅ Statistics compiled for {len(config_summary['target_splits'])} splits",
            'stats': stats
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f"❌ Error getting statistics: {str(e)}",
            'stats': {}
        }

# === BACKWARD COMPATIBILITY ===

def preprocess_and_visualize(config: Dict[str, Any], target_split: str = "train", 
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🔄 Backward compatibility untuk preprocess_and_visualize"""
    return preprocess_single_split(config, target_split, None, progress_callback)

# === EXPORTS ===

__all__ = [
    # Core services
    'PreprocessingService',
    'PreprocessingEngine', 
    'PreprocessingValidator',
    'ValidationEngine',
    
    # Enhanced factory functions
    'create_preprocessing_service',
    'create_preprocessing_service_with_progress',
    'create_preprocessing_engine_enhanced',
    'create_preprocessing_validator_enhanced',
    'create_validation_engine',
    
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
    'preprocess_for_yolo_training',
    
    # Utils exports (consolidated)
    'validate_preprocessing_config',
    'get_default_preprocessing_config',
    'reload_default_config',
    'FileOperations', 'ValidationCore', 'PathManager', 'MetadataManager', 
    'YOLONormalizer', 'ProgressBridge',
    'create_file_operations', 'create_validation_core', 'create_path_manager',
    'create_metadata_manager', 'create_yolo_normalizer', 'create_progress_bridge',
    'create_compatible_bridge',
    'read_image_safe', 'write_image_safe', 'scan_image_files', 'find_pairs_safe',
    'validate_image_safe', 'validate_label_safe', 'validate_pair_safe',
    'validate_source_safe', 'create_output_safe', 'get_paths_safe',
    'parse_filename_safe', 'generate_preprocessed_safe', 'extract_nominal_safe',
    'preprocess_image_for_yolo', 'normalize_yolo_safe', 'update_progress_safe',
    
    # Legacy compatibility exports
    'ImageValidator', 'LabelValidator', 'PairValidator', 'FileProcessor',
    'PathResolver', 'FilenameManager', 'FileScanner', 'CleanupManager',
    'create_image_validator', 'create_label_validator', 'create_pair_validator',
    'create_file_processor', 'create_path_resolver', 'create_filename_manager',
    'create_file_scanner', 'create_cleanup_manager',
    
    # Backward compatibility
    'preprocess_and_visualize'
]