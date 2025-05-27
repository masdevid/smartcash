"""
File: smartcash/dataset/augmentor/utils/config_extractor.py
Deskripsi: SRP module untuk config extraction dan context creation
"""

from typing import Dict, Any
from smartcash.dataset.augmentor.utils.path_operations import resolve_drive_path, get_best_data_location
from smartcash.dataset.augmentor.utils.progress_tracker import create_progress_tracker
from smartcash.dataset.augmentor.utils.dataset_detector import detect_structure
from smartcash.dataset.augmentor.utils.cleanup_operations import cleanup_files

# =============================================================================
# CONFIG EXTRACTION - One-liner utilities
# =============================================================================

extract_config = lambda config: {
    'raw_dir': resolve_drive_path(config.get('data', {}).get('dir', 'data')),
    'aug_dir': resolve_drive_path(config.get('augmentation', {}).get('output_dir', 'data/augmented')),
    'prep_dir': resolve_drive_path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')),
    'num_variations': config.get('augmentation', {}).get('num_variations', 2),
    'target_count': config.get('augmentation', {}).get('target_count', 500),
    'types': config.get('augmentation', {}).get('types', ['combined']),
    'intensity': config.get('augmentation', {}).get('intensity', 0.7)
}

extract_split_aware_config = lambda config: {
    **extract_config(config),
    'target_split': config.get('augmentation', {}).get('target_split', 'train'),
    'split_aware': True
}

# =============================================================================
# CONTEXT CREATION
# =============================================================================

def create_context(config: Dict[str, Any], communicator=None) -> Dict[str, Any]:
    """Enhanced context creation dengan communicator integration"""
    aug_config = extract_config(config)
    progress_tracker = create_progress_tracker(communicator)
    
    from smartcash.dataset.augmentor.utils.path_operations import build_paths
    
    return {
        'config': aug_config,
        'progress': progress_tracker,
        'paths': build_paths(aug_config['raw_dir'], aug_config['aug_dir'], aug_config['prep_dir']),
        'detector': lambda: detect_structure(aug_config['raw_dir']),
        'cleaner': lambda: cleanup_files(aug_config['aug_dir'], aug_config['prep_dir'], progress_tracker),
        'comm': communicator
    }

def create_split_aware_context(config: Dict[str, Any], communicator=None) -> Dict[str, Any]:
    """Create context dengan split awareness"""
    aug_config = extract_split_aware_config(config)
    progress_tracker = create_progress_tracker(communicator)
    target_split = aug_config.get('target_split', 'train')
    
    from smartcash.dataset.augmentor.utils.path_operations import build_split_aware_paths
    from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
    from smartcash.dataset.augmentor.utils.cleanup_operations import cleanup_split_aware
    
    return {
        'config': aug_config,
        'progress': progress_tracker,
        'paths': build_split_aware_paths(aug_config['raw_dir'], aug_config['aug_dir'], 
                                       aug_config['prep_dir'], target_split),
        'detector': lambda: detect_split_structure(aug_config['raw_dir']),
        'cleaner': lambda split=None: cleanup_split_aware(aug_config['aug_dir'], aug_config['prep_dir'], 
                                                        split or target_split, progress_tracker),
        'comm': communicator,
        'split_aware': True,
        'target_split': target_split
    }