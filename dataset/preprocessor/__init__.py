"""
File: smartcash/dataset/preprocessor/__init__.py
Deskripsi: Main API exports untuk preprocessor module dengan clean interface
"""

# === MAIN SERVICES ===
from .service import PreprocessingService
from .api.preprocessing_api import preprocess_dataset, get_preprocessing_status, validate_dataset_structure, validate_filenames
from .api.normalization_api import normalize_for_yolo, create_normalizer, denormalize_for_visualization
from .api.samples_api import get_samples, generate_sample_previews
from .api.stats_api import get_dataset_stats, get_file_stats
from .api.cleanup_api import cleanup_preprocessing_files

# === CORE COMPONENTS ===
from .core.normalizer import YOLONormalizer
from .core.file_processor import FileProcessor
from .core.stats_collector import StatsCollector

# === CONFIGURATION ===
from .config.defaults import get_default_config, NORMALIZATION_PRESETS
from .config.validator import validate_preprocessing_config

# === UTILITIES ===
from .utils.path_manager import PathManager
from .utils.progress_bridge import create_preprocessing_bridge
from .utils.metadata_extractor import extract_file_metadata, parse_research_filename

# === FACTORY FUNCTIONS ===
def create_preprocessing_service(config=None, progress_callback=None):
    """🏭 Factory untuk create preprocessing service"""
    return PreprocessingService(config, progress_callback)

def create_yolo_normalizer(preset='default', **kwargs):
    """🏭 Factory untuk create YOLO normalizer"""
    from .config.defaults import NORMALIZATION_PRESETS
    config = NORMALIZATION_PRESETS[preset].copy()
    config.update(kwargs)
    return YOLONormalizer(config)

def create_file_processor(config=None):
    """🏭 Factory untuk create file processor"""
    return FileProcessor(config)

def create_stats_collector(config=None):
    """🏭 Factory untuk create statistics collector"""
    return StatsCollector(config)

# === MAIN API FUNCTIONS ===
__all__ = [
    # Main services
    'PreprocessingService',
    'preprocess_dataset',
    'get_preprocessing_status',
    'validate_dataset_structure',
    'validate_filenames',
    
    # Normalization API
    'normalize_for_yolo',
    'create_normalizer', 
    'denormalize_for_visualization',
    
    # Samples API
    'get_samples',
    'generate_sample_previews',
    
    # Statistics API
    'get_dataset_stats',
    'get_file_stats',
    
    # Cleanup API
    'cleanup_preprocessing_files',
    
    # Core components
    'YOLONormalizer',
    'FileProcessor',
    'StatsCollector',
    
    # Configuration
    'get_default_config',
    'validate_preprocessing_config',
    'NORMALIZATION_PRESETS',
    
    # Utilities
    'PathManager',
    'create_preprocessing_bridge',
    'extract_file_metadata',
    'parse_research_filename',
    
    # Factory functions
    'create_preprocessing_service',
    'create_yolo_normalizer',
    'create_file_processor',
    'create_stats_collector'
]

# === BACKWARD COMPATIBILITY ALIASES ===
def validate_dataset(config=None, target_split=None, ui_components=None):
    """🔄 Compatibility alias untuk validate_dataset_structure dengan config handling"""
    if config is None:
        return {'success': False, 'message': 'Config required'}
    
    data_dir = config.get('data', {}).get('dir', 'data')
    splits = config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
    
    return validate_dataset_structure(data_dir, splits, auto_fix=True)

def cleanup_preprocessed_data(config=None, target_split=None, ui_components=None):
    """🔄 Compatibility alias untuk cleanup_preprocessing_files dengan config handling"""
    if config is None:
        return {'success': False, 'message': 'Config required'}
    
    data_dir = config.get('data', {}).get('dir', 'data')
    target = 'preprocessed'  # Default untuk backward compatibility
    splits = [target_split] if target_split else None
    
    return cleanup_preprocessing_files(data_dir, target, splits, confirm=True)

def get_preprocessing_samples(config=None, target_split='train', max_samples=5, ui_components=None):
    """🔄 Compatibility alias untuk get_samples dengan config handling"""
    if config is None:
        return {'success': False, 'message': 'Config required', 'samples': []}
    
    data_dir = config.get('data', {}).get('preprocessed_dir', 'data/preprocessed')
    return get_samples(data_dir, target_split, max_samples)