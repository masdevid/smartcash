"""
File: smartcash/ui/dataset/augment/constants.py
Description: Constants and enums for the augment module

This module contains all constants, enums, and configuration data
used throughout the augment module following the UI structure guidelines.
"""

from typing import Dict, Any, List
from enum import Enum

# =============================================================================
# ENUMS
# =============================================================================

class AugmentationOperation(Enum):
    """Available augmentation operations."""
    AUGMENT = "augment"
    CHECK = "check"
    CLEANUP = "cleanup"
    PREVIEW = "preview"

class AugmentationTypes(Enum):
    """Available augmentation types."""
    COMBINED = "combined"
    POSITION = "position"
    LIGHTING = "lighting"
    CUSTOM = "custom"

class CleanupTarget(Enum):
    """Cleanup target options."""
    AUGMENTED = "augmented"
    SAMPLES = "samples"
    BOTH = "both"

class ProcessingPhase(Enum):
    """Processing phases for augmentation."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    FINALIZATION = "finalization"

# =============================================================================
# UI CONFIGURATION
# =============================================================================

UI_CONFIG = {
    'module_name': 'augment',
    'parent_module': 'dataset',
    'title': '🎨 Data Augmentation',
    'subtitle': 'Enhance dataset with augmentation techniques',
    'icon': '🎨',
    'version': '2.0.0'
}

# =============================================================================
# BUTTON CONFIGURATION
# =============================================================================

BUTTON_CONFIG = {
    'augment': {
        'text': '🚀 Start Augmentation',
        'style': 'success',
        'tooltip': 'Start augmentation process with current settings',
        'order': 1
    },
    'check': {
        'text': '🔍 Check Status',
        'style': 'info',
        'tooltip': 'Check augmentation status and statistics',
        'order': 2
    },
    'cleanup': {
        'text': '🗑️ Cleanup Files',
        'style': 'danger',
        'tooltip': 'Clean up augmented or sample files',
        'order': 3
    },
    'preview': {
        'text': '👁️ Live Preview',
        'style': 'warning',
        'tooltip': 'Preview augmentation results in real-time',
        'order': 4
    }
}

# =============================================================================
# FORM CONFIGURATION
# =============================================================================

AUGMENTATION_TYPES_OPTIONS = [
    ('Combined (Position + Lighting)', 'combined'),
    ('Position Only (Flip, Rotate, Scale)', 'position'),
    ('Lighting Only (Brightness, Contrast, HSV)', 'lighting'),
    ('Custom Configuration', 'custom')
]

TARGET_SPLIT_OPTIONS = [
    ('Training Set', 'train'),
    ('Validation Set', 'valid'),
    ('Test Set', 'test'),
    ('All Splits', 'all')
]

CLEANUP_TARGET_OPTIONS = [
    ('Augmented Files Only', 'augmented'),
    ('Sample Files Only', 'samples'),
    ('Both Augmented and Samples', 'both')
]

# =============================================================================
# STYLING CONSTANTS
# =============================================================================

AUGMENT_COLORS = {
    'primary': '#007bff',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

SECTION_STYLES = {
    'basic_options': {
        'border_color': AUGMENT_COLORS['info'],
        'background': '#f0f8ff'
    },
    'advanced_options': {
        'border_color': AUGMENT_COLORS['warning'],
        'background': '#fffaf0'
    },
    'augmentation_types': {
        'border_color': AUGMENT_COLORS['success'],
        'background': '#f0fff0'
    },
    'live_preview': {
        'border_color': AUGMENT_COLORS['primary'],
        'background': '#f8f9ff'
    }
}

# =============================================================================
# PROGRESS PHASES
# =============================================================================

PROGRESS_PHASES = {
    ProcessingPhase.VALIDATION: {
        'weight': 0.2,
        'description': 'Validating configuration and dataset'
    },
    ProcessingPhase.PROCESSING: {
        'weight': 0.7,
        'description': 'Processing augmentation operations'
    },
    ProcessingPhase.FINALIZATION: {
        'weight': 0.1,
        'description': 'Finalizing results and cleanup'
    }
}

# =============================================================================
# DEFAULT VALUES
# =============================================================================

DEFAULT_AUGMENTATION_PARAMS = {
    'num_variations': 2,
    'target_count': 500,
    'intensity': 0.7,
    'balance_classes': True,
    'target_split': 'train',
    'types': ['combined']
}

DEFAULT_POSITION_PARAMS = {
    'horizontal_flip': 0.5,
    'rotation_limit': 12,
    'translate_limit': 0.08,
    'scale_limit': 0.04
}

DEFAULT_LIGHTING_PARAMS = {
    'brightness_limit': 0.2,
    'contrast_limit': 0.15,
    'hsv_hue': 10,
    'hsv_saturation': 15
}

DEFAULT_CLEANUP_PARAMS = {
    'default_target': 'both',
    'confirm_before_cleanup': True,
    'backup_before_cleanup': False,
    'cleanup_empty_dirs': True
}

# =============================================================================
# MESSAGES
# =============================================================================

SUCCESS_MESSAGES = {
    'augmentation_complete': '✅ Augmentation completed successfully',
    'check_complete': '✅ Status check completed',
    'cleanup_complete': '✅ Cleanup completed successfully',
    'preview_ready': '✅ Preview generated successfully'
}

ERROR_MESSAGES = {
    'augmentation_failed': '❌ Augmentation process failed',
    'check_failed': '❌ Status check failed',
    'cleanup_failed': '❌ Cleanup operation failed',
    'preview_failed': '❌ Preview generation failed',
    'invalid_config': '❌ Invalid configuration provided',
    'no_data_found': '❌ No data found for augmentation'
}

WARNING_MESSAGES = {
    'low_data_count': '⚠️ Low data count detected',
    'overwrite_warning': '⚠️ This will overwrite existing files',
    'cleanup_warning': '⚠️ This action cannot be undone'
}

# =============================================================================
# TIPS AND HELP TEXT
# =============================================================================

AUGMENTATION_TIPS = [
    "💡 Use 'Combined' type for best results with balanced augmentation",
    "📊 Monitor target count to avoid dataset imbalance",
    "🔄 Intensity values between 0.5-0.8 work well for most datasets",
    "📁 Always backup your data before running cleanup operations",
    "👁️ Use Live Preview to fine-tune parameters before processing",
    "⚖️ Enable class balancing for multi-class datasets",
    "🎯 Target specific splits to control where augmentation is applied"
]

HELP_TEXT = {
    'num_variations': 'Number of augmented versions per original image',
    'target_count': 'Target number of images per class after augmentation',
    'intensity': 'Overall intensity of augmentation effects (0.0-1.0)',
    'balance_classes': 'Automatically balance classes during augmentation',
    'target_split': 'Dataset split to apply augmentation to',
    'types': 'Types of augmentation transformations to apply',
    'horizontal_flip': 'Probability of horizontal flip (0.0-1.0)',
    'rotation_limit': 'Maximum rotation angle in degrees',
    'translate_limit': 'Maximum translation as fraction of image size',
    'scale_limit': 'Maximum scale variation as fraction',
    'brightness_limit': 'Maximum brightness adjustment',
    'contrast_limit': 'Maximum contrast adjustment',
    'hsv_hue': 'Maximum hue shift in HSV color space',
    'hsv_saturation': 'Maximum saturation adjustment',
    'cleanup_target': 'Type of files to clean up'
}

# =============================================================================
# BANKNOTE CLASSES (Business Logic)
# =============================================================================

BANKNOTE_CLASSES = [
    '001', '002', '005', '010', '020','050','100',
    'l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020','l2_050','l2_100',
    'l3_text', 'l3_thread', 'l3_security'
]

CLASS_WEIGHTS = {
    '001': 1.0,
    '002': 1.0,
    '005': 1.0,
    '010': 1.0,
    '020': 1.0,
    '050': 1.0,
    '100': 1.0,
    'l2_001': 0.8,
    'l2_002': 0.8,
    'l2_005': 0.8,
    'l2_010': 0.8,
    'l2_020': 0.8,
    'l2_050': 0.8,
    'l2_100': 0.8,
    'l3_text': 0.3,
    'l3_thread': 0.3,
    'l3_security': 0.3
}

# =============================================================================
# BACKEND CONFIGURATION
# =============================================================================

BACKEND_CONFIG = {
    'service_enabled': True,
    'progress_tracking': True,
    'async_processing': False,
    'max_workers': 4,
    'timeout_seconds': 300,
    'retry_count': 3,
    'validation_enabled': True
}

FILE_PROCESSING_CONFIG = {
    'max_workers': 4,
    'batch_size': 100,
    'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp'],
    'output_format': '.jpg',
    'quality': 95
}

PERFORMANCE_CONFIG = {
    'num_workers': 4,
    'memory_limit_mb': 1024,
    'cache_enabled': True,
    'cache_size_mb': 256
}

# =============================================================================
# MODULE METADATA
# =============================================================================

MODULE_METADATA = {
    'name': 'augment',
    'display_name': 'Data Augmentation',
    'version': '2.0.0',
    'description': 'Dataset augmentation with position and lighting transforms',
    'category': 'dataset',
    'requires_gpu': False,
    'supports_batch': True,
    'supports_progress': True,
    'supports_preview': True,
    'augmentation_types': ['position', 'lighting', 'combined', 'custom'],
    'max_variations': 10,
    'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp']
}