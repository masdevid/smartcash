"""
File: smartcash/dataset/augmentor/__init__.py
Deskripsi: Main augmentor module dengan fixed structure dan one-liner exports
"""

# Core engines
from .core.engine import AugmentationEngine, create_augmentation_engine, process_raw_to_augmented, validate_raw_directory
from .core.normalizer import NormalizationEngine, create_normalization_engine, normalize_augmented_to_split, cleanup_aug_prefix, get_augmented_file_stats
from .core.pipeline import PipelineFactory, create_augmentation_pipeline, get_research_pipeline_types, get_available_pipeline_types, validate_intensity, get_pipeline_description

# Service layer
from .service import AugmentationService, create_augmentation_service, create_service_from_ui, augment_raw_data, normalize_augmented_data, run_full_pipeline, cleanup_augmented_files

# Configuration
from .config import create_aug_config, extract_ui_config, get_raw_dir, get_aug_dir, get_prep_dir, get_num_variations, get_target_count, get_output_prefix, get_augmentation_types, validate_config, DEFAULT_CONFIG, COLAB_CONFIG
from .types import AugConfig, ProcessingStats, ClassBalanceInfo, AugmentationResult, ProgressCallback, ResultDict, FileList, ClassDistribution, AugmentationTypes, DEFAULT_AUGMENTATION_TYPES, LAYER_1_CLASSES, LAYER_2_CLASSES, TARGET_CLASSES

# Communication
from .communicator import UICommunicator, create_communicator, log_to_ui, progress_to_ui, start_ui_operation, complete_ui_operation, error_ui_operation

# Utilities
from .utils.cleaner import AugmentedDataCleaner, create_augmented_data_cleaner, cleanup_all_aug_files, preview_cleanup, find_aug_files, count_aug_files
from .utils.dataset_detector import detect_dataset_structure, detect_structure, is_yolo_structure, get_image_directories, get_label_directories, count_dataset_files, has_valid_dataset

__all__ = [
    # Core engines
    'AugmentationEngine', 'create_augmentation_engine', 'process_raw_to_augmented', 'validate_raw_directory',
    'NormalizationEngine', 'create_normalization_engine', 'normalize_augmented_to_split', 'cleanup_aug_prefix', 'get_augmented_file_stats',
    'PipelineFactory', 'create_augmentation_pipeline', 'get_research_pipeline_types', 'get_available_pipeline_types', 'validate_intensity', 'get_pipeline_description',
    
    # Service layer  
    'AugmentationService', 'create_augmentation_service', 'create_service_from_ui',
    'augment_raw_data', 'normalize_augmented_data', 'run_full_pipeline', 'cleanup_augmented_files',
    
    # Configuration
    'create_aug_config', 'extract_ui_config', 'get_raw_dir', 'get_aug_dir', 'get_prep_dir',
    'get_num_variations', 'get_target_count', 'get_output_prefix', 'get_augmentation_types', 'validate_config',
    'DEFAULT_CONFIG', 'COLAB_CONFIG',
    
    # Types
    'AugConfig', 'ProcessingStats', 'ClassBalanceInfo', 'AugmentationResult',
    'ProgressCallback', 'ResultDict', 'FileList', 'ClassDistribution', 'AugmentationTypes',
    'DEFAULT_AUGMENTATION_TYPES', 'LAYER_1_CLASSES', 'LAYER_2_CLASSES', 'TARGET_CLASSES',
    
    # Communication
    'UICommunicator', 'create_communicator', 'log_to_ui', 'progress_to_ui',
    'start_ui_operation', 'complete_ui_operation', 'error_ui_operation',
    
    # Utilities
    'AugmentedDataCleaner', 'create_augmented_data_cleaner', 'cleanup_all_aug_files', 
    'preview_cleanup', 'find_aug_files', 'count_aug_files',
    'detect_dataset_structure', 'detect_structure', 'is_yolo_structure', 
    'get_image_directories', 'get_label_directories', 'count_dataset_files', 'has_valid_dataset'
]