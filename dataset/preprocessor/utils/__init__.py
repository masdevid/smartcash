"""
File: smartcash/dataset/preprocessor/utils/__init__.py
Deskripsi: Consolidated exports untuk preprocessor utils dengan backward compatibility
"""

# === CONFIG VALIDATION ===
from .config_validator import (
    validate_preprocessing_config,
    get_default_preprocessing_config,
    reload_default_config,
    PreprocessingConfigValidator
)

# === FILE OPERATIONS ===
from .file_operations import (
    FileOperations,
    create_file_operations,
    # Convenience functions
    read_image_safe,
    write_image_safe,
    scan_image_files,
    find_pairs_safe
)

# === VALIDATION CORE ===
from .validation_core import (
    ValidationCore,
    ValidationResult,
    create_validation_core,
    # Convenience functions
    validate_image_safe,
    validate_label_safe,
    validate_pair_safe
)

# === PATH MANAGEMENT ===
from .path_manager import (
    PathManager,
    PathConfig,
    create_path_manager,
    create_path_config,
    # Convenience functions
    validate_source_safe,
    create_output_safe,
    get_paths_safe
)

# === PROGRESS BRIDGE ===
from .progress_bridge import (
    ProgressBridge,
    ProgressLevel,
    ProgressUpdate,
    SubProgressBridge,
    PhaseProgressManager,
    create_progress_bridge,
    create_compatible_bridge,
    # Specialized bridges
    create_preprocessing_bridge,
    create_validation_bridge,
    create_augmentation_bridge,
    # Convenience functions
    update_progress_safe
)

# === METADATA MANAGEMENT ===
from .metadata_manager import (
    MetadataManager,
    FileMetadata,
    create_metadata_manager,
    # Convenience functions
    parse_filename_safe,
    generate_preprocessed_safe,
    extract_nominal_safe,
    validate_pairs_safe
)

# === YOLO NORMALIZATION ===
from .normalization import (
    YOLONormalizer,
    create_yolo_normalizer,
    # Convenience functions
    preprocess_image_for_yolo,
    normalize_yolo_safe
)

# === LEGACY COMPATIBILITY ===
# Backward compatibility dengan existing validators
from .validation_core import ValidationCore as ImageValidator
from .validation_core import ValidationCore as LabelValidator
from .validation_core import ValidationCore as PairValidator

# Factory functions untuk backward compatibility
def create_image_validator(config=None):
    """🔄 Backward compatibility wrapper"""
    return ValidationCore(config)

def create_label_validator(config=None):
    """🔄 Backward compatibility wrapper"""
    return ValidationCore(config)

def create_pair_validator(config=None):
    """🔄 Backward compatibility wrapper"""
    return ValidationCore(config)

# Legacy file processor compatibility
FileProcessor = FileOperations
def create_file_processor(config=None):
    """🔄 Backward compatibility wrapper"""
    return FileOperations(config)

# Legacy path resolver compatibility
PathResolver = PathManager
def create_path_resolver(config=None):
    """🔄 Backward compatibility wrapper"""
    return PathManager(config)

# Legacy filename manager compatibility
FilenameManager = MetadataManager
def create_filename_manager(config=None):
    """🔄 Backward compatibility wrapper"""
    return MetadataManager(config)

# Legacy file scanner compatibility
class FileScanner:
    """🔄 Backward compatibility wrapper"""
    def __init__(self):
        self.file_ops = FileOperations()
    
    def scan_directory(self, directory, extensions=None):
        if extensions:
            return self.file_ops.scan_files(directory, set(extensions))
        return self.file_ops.scan_images(directory)
    
    def find_image_label_pairs(self, image_dir, label_dir):
        return self.file_ops.find_image_label_pairs(image_dir, label_dir)

def create_file_scanner():
    """🔄 Backward compatibility wrapper"""
    return FileScanner()

# Legacy cleanup manager compatibility
class CleanupManager:
    """🔄 Backward compatibility wrapper"""
    def __init__(self, config=None):
        self.path_manager = PathManager(config)
    
    def cleanup_output_dirs(self, split=None):
        splits = [split] if split else None
        return self.path_manager.cleanup_output_dirs(splits, confirm=True)

def create_cleanup_manager(config=None):
    """🔄 Backward compatibility wrapper"""
    return CleanupManager(config)

# === CONSOLIDATED EXPORTS ===
__all__ = [
    # Config validation
    'validate_preprocessing_config',
    'get_default_preprocessing_config',
    'reload_default_config',
    'PreprocessingConfigValidator',
    
    # File operations
    'FileOperations',
    'create_file_operations',
    'read_image_safe',
    'write_image_safe',
    'scan_image_files',
    'find_pairs_safe',
    
    # Validation
    'ValidationCore',
    'ValidationResult',
    'create_validation_core',
    'validate_image_safe',
    'validate_label_safe', 
    'validate_pair_safe',
    
    # Path management
    'PathManager',
    'PathConfig',
    'create_path_manager',
    'create_path_config',
    'validate_source_safe',
    'create_output_safe',
    'get_paths_safe',
    
    # Progress bridge
    'ProgressBridge',
    'ProgressLevel',
    'ProgressUpdate',
    'SubProgressBridge',
    'PhaseProgressManager',
    'create_progress_bridge',
    'create_compatible_bridge',
    'create_preprocessing_bridge',
    'create_validation_bridge',
    'create_augmentation_bridge',
    'update_progress_safe',
    
    # Metadata management
    'MetadataManager',
    'FileMetadata',
    'create_metadata_manager',
    'parse_filename_safe',
    'generate_preprocessed_safe',
    'extract_nominal_safe',
    'validate_pairs_safe',
    
    # YOLO normalization
    'YOLONormalizer',
    'create_yolo_normalizer',
    'preprocess_image_for_yolo',
    'normalize_yolo_safe',
    
    # Legacy compatibility
    'ImageValidator',
    'LabelValidator', 
    'PairValidator',
    'create_image_validator',
    'create_label_validator',
    'create_pair_validator',
    'FileProcessor',
    'create_file_processor',
    'PathResolver',
    'create_path_resolver',
    'FilenameManager',
    'create_filename_manager',
    'FileScanner',
    'create_file_scanner',
    'CleanupManager',
    'create_cleanup_manager'
]