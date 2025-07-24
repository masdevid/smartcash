"""
File: smartcash/ui/dataset/preprocessing/constants.py
Description: Constants and enums for preprocessing module
"""

from enum import Enum
from typing import Dict, List, Tuple, Any

# ==================== PREPROCESSING CONSTANTS ====================

class PreprocessingOperation(Enum):
    """Preprocessing operation types"""
    PREPROCESS = "preprocess"
    CHECK = "check"
    CLEANUP = "cleanup"
    PREVIEW = "preview"

class ValidationLevel(Enum):
    """Validation levels for preprocessing"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"

class NormalizationMethod(Enum):
    """Normalization methods"""
    MINMAX = "minmax"
    ZSCORE = "zscore"
    ROBUST = "robust"

class CleanupTarget(Enum):
    """Cleanup target types"""
    PREPROCESSED = "preprocessed"
    AUGMENTED = "augmented"
    SAMPLES = "samples"
    BOTH = "both"

class ProcessingPhase(Enum):
    """Processing phases for progress tracking"""
    VALIDATION = "validation"
    PROCESSING = "processing"
    FINALIZATION = "finalization"

# ==================== YOLO PRESETS ====================

YOLO_PRESETS: Dict[str, Dict[str, Any]] = {
    'default': {
        'target_size': [640, 640],
        'preserve_aspect_ratio': True,
        'pixel_range': [0, 1],
        'interpolation': 'linear'
    },
    'yolov5s': {
        'target_size': [640, 640],
        'preserve_aspect_ratio': True,
        'pixel_range': [0, 1],
        'interpolation': 'linear'
    },
    'yolov5m': {
        'target_size': [640, 640],
        'preserve_aspect_ratio': True,
        'pixel_range': [0, 1],
        'interpolation': 'linear'
    },
    'yolov5l': {
        'target_size': [832, 832],
        'preserve_aspect_ratio': True,
        'pixel_range': [0, 1],
        'interpolation': 'cubic'
    },
    'yolov5x': {
        'target_size': [1024, 1024],
        'preserve_aspect_ratio': True,
        'pixel_range': [0, 1],
        'interpolation': 'cubic'
    },
    'inference': {
        'target_size': [640, 640],
        'preserve_aspect_ratio': True,
        'pixel_range': [0, 1],
        'interpolation': 'linear',
        'batch_optimized': True
    }
}

# ==================== FILE PATTERNS ====================

FILE_PATTERNS = {
    'raw_images': r'^rp_(\d{6})_([a-zA-Z0-9]+)\.jpg$',
    'preprocessed_npy': r'^pre_(\d{6})_([a-zA-Z0-9]+)\.npy$',
    'augmented_npy': r'^aug_(\d{6})_([a-zA-Z0-9]+)_(\d{3})\.npy$',
    'sample_images': r'^sample_pre_(\d{6})_([a-zA-Z0-9]+)\.jpg$',
    'augmented_samples': r'^sample_aug_(\d{6})_([a-zA-Z0-9]+)_(\d{3})\.jpg$',
    'label_files': r'^(pre|aug)_(\d{6})_([a-zA-Z0-9]+)(_\d{3})?\.txt$'
}

# ==================== BANKNOTE CLASSES ====================

BANKNOTE_CLASSES = {
    0: {'nominal': '001000', 'display': 'Rp1K', 'value': 1000, 'layer': 'l1_main'},
    1: {'nominal': '002000', 'display': 'Rp2K', 'value': 2000, 'layer': 'l1_main'},
    2: {'nominal': '005000', 'display': 'Rp5K', 'value': 5000, 'layer': 'l1_main'},
    3: {'nominal': '010000', 'display': 'Rp10K', 'value': 10000, 'layer': 'l1_main'},
    4: {'nominal': '020000', 'display': 'Rp20K', 'value': 20000, 'layer': 'l1_main'},
    5: {'nominal': '050000', 'display': 'Rp50K', 'value': 50000, 'layer': 'l1_main'},
    6: {'nominal': '100000', 'display': 'Rp100K', 'value': 100000, 'layer': 'l1_main'}
}

MAIN_BANKNOTE_CLASSES = list(range(7))  # Classes 0-6

# ==================== LAYER DEFINITIONS ====================

LAYER_CLASSES = {
    'l1_main': list(range(0, 7)),      # Main banknotes (0-6)
    'l2_security': list(range(7, 14)), # Security features (7-13)
    'l3_micro': list(range(14, 17))    # Micro text (14-16)
}

# ==================== UI CONFIGURATION ====================

UI_CONFIG = {
    'title': 'Dataset Preprocessing',
    'subtitle': 'Preprocessing dataset dengan YOLO normalization dan real-time progress',
    'icon': '🚀',
    'logo_path': 'path/to/your/logo.png',  # TODO: Update with actual path
    'module_name': 'preprocessing',
    'parent_module': 'dataset'
}

# ==================== DEFAULT SPLITS ====================

DEFAULT_SPLITS = ['train', 'valid', 'test']
SUPPORTED_SPLITS = ['train', 'valid', 'test']

# ==================== PROGRESS SETTINGS ====================

PROGRESS_PHASES = {
    ProcessingPhase.VALIDATION: {
        'weight': 0.20,
        'description': 'Validating dataset structure'
    },
    ProcessingPhase.PROCESSING: {
        'weight': 0.70,
        'description': 'Processing images'
    },
    ProcessingPhase.FINALIZATION: {
        'weight': 0.10,
        'description': 'Finalizing results'
    }
}

# ==================== VALIDATION SETTINGS ====================

VALIDATION_CONFIG = {
    'filename_pattern': True,
    'auto_fix': True,
    'create_directories': True,
    'min_files_per_split': 1
}

# ==================== PERFORMANCE SETTINGS ====================

PERFORMANCE_CONFIG = {
    'batch_size': 32,
    'io_workers': 8,
    'cpu_workers': None,  # Auto-detect
    'memory_limit_mb': 2048
}

# ==================== ERROR MESSAGES ====================

ERROR_MESSAGES = {
    'no_data_dir': 'Data directory not found or not accessible',
    'no_splits': 'No valid splits found in data directory',
    'invalid_config': 'Invalid preprocessing configuration',
    'normalization_failed': 'Image normalization failed',
    'cleanup_failed': 'Cleanup operation failed',
    'validation_failed': 'Dataset validation failed'
}

# ==================== SUCCESS MESSAGES ====================

SUCCESS_MESSAGES = {
    'preprocessing_complete': '✅ Preprocessing completed successfully',
    'validation_passed': '✅ Dataset validation passed',
    'cleanup_complete': '✅ Cleanup completed successfully',
    'check_complete': '✅ Dataset check completed'
}

# ==================== TIPS AND INFO ====================

PREPROCESSING_TIPS = [
    'Gunakan resolusi yang sesuai dengan model target',
    'Min-Max normalization (0-1) direkomendasikan untuk YOLO',
    'Aktifkan validasi untuk memastikan dataset berkualitas',
    'Backup data asli sebelum preprocessing',
    'Perhatikan aspect ratio untuk hasil optimal'
]

# ==================== BUTTON CONFIGURATIONS ====================

BUTTON_CONFIG = {
    'preprocess': {
        'id': 'preprocess',
        'text': '🚀 Mulai Preprocessing',
        'style': 'success',
        'tooltip': 'Mulai proses preprocessing dataset',
        'order': 1
    },
    'check': {
        'id': 'check',
        'text': '🔍 Check Dataset',
        'style': 'info',
        'tooltip': 'Periksa status dataset dan konfigurasi',
        'order': 2
    },
    'cleanup': {
        'id': 'cleanup',
        'text': '🗑️ Cleanup',
        'style': 'danger',  # Changed from 'warning' to 'danger' to make it more prominent
        'tooltip': 'Hapus data preprocessing yang sudah ada',
        'order': 3
    }
}

# ==================== MODULE METADATA ====================

MODULE_METADATA = {
    'name': 'preprocessing',
    'display_name': 'Dataset Preprocessing',
    'version': '2.0.0',
    'description': 'YOLO-compatible dataset preprocessing with normalization and validation',
    'category': 'dataset',
    'requires_gpu': False,
    'supports_batch': True,
    'supports_progress': True
}
