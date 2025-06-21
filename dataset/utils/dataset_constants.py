"""
File: smartcash/dataset/utils/dataset_constants.py
Deskripsi: Enhanced konstanta dataset dengan UUID support dan research naming
"""
import re
from smartcash.common.constants.core import APP_NAME
from smartcash.common.constants.paths import (
    DEFAULT_DATA_DIR, PREPROCESSED_DIR as DEFAULT_PREPROCESSED_DIR,
    AUGMENTED_DIR as DEFAULT_AUGMENTED_DIR, BACKUP_DIR as DEFAULT_BACKUP_DIR,
    INVALID_DIR as DEFAULT_INVALID_DIR, VISUALIZATION_DIR as DEFAULT_VISUALIZATION_DIR,
    CHECKPOINT_DIR as DEFAULT_CHECKPOINT_DIR, TRAIN_DIR as DEFAULT_TRAIN_DIR,
    VALID_DIR as DEFAULT_VALID_DIR, TEST_DIR as DEFAULT_TEST_DIR,
    COLAB_PATH, COLAB_DATASET_PATH, COLAB_PREPROCESSED_PATH, COLAB_AUGMENTED_PATH,
    COLAB_BACKUP_PATH, COLAB_INVALID_PATH, COLAB_VISUALIZATION_PATH,
    DRIVE_PATH, DRIVE_DATASET_PATH, DRIVE_PREPROCESSED_PATH, DRIVE_AUGMENTED_PATH,
    DRIVE_BACKUP_PATH, DRIVE_INVALID_PATH, DRIVE_VISUALIZATION_PATH
)
from smartcash.common.constants.dataset import DEFAULT_TRAIN_SPLIT, DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT

# Enhanced konstanta untuk format file gambar
IMG_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP']
IMG_EXTENSIONS_SET = {'.jpg', '.jpeg', '.png', '.bmp'}  # Untuk faster lookup

# Enhanced konstanta untuk split dataset
DEFAULT_SPLITS = ['train', 'valid', 'test']
DEFAULT_SPLIT_RATIOS = {'train': DEFAULT_TRAIN_SPLIT, 'valid': DEFAULT_VAL_SPLIT, 'test': DEFAULT_TEST_SPLIT}

# Konstanta untuk random seed
DEFAULT_RANDOM_SEED = 42

# Enhanced konstanta untuk prefix file dengan UUID support
DEFAULT_ORIGINAL_PREFIX = 'rp'
DEFAULT_AUGMENTATION_PREFIX = 'aug'
DEFAULT_PREPROCESSING_PREFIX = 'pre'
UUID_PREFIX = 'rp'  # Research-ready prefix

# UUID file naming patterns
UUID_FILE_PATTERN = r'^rp_(\d{6})_([0-9a-f-]{36})_(\d{3})'
UUID_FILENAME_REGEX = re.compile(UUID_FILE_PATTERN)
LEGACY_PATTERNS = ['aug_', 'IMG_', 'img_', 'image_', 'photo_', 'pic_']

# Enhanced konstanta untuk ukuran gambar
DEFAULT_IMG_SIZE = [640, 640]

# Enhanced denomination mapping dengan research format
DENOMINATION_CLASS_MAP = {
    # Layer 1 - Primary banknote detection
    '0': '001000', '1': '002000', '2': '005000', '3': '010000', 
    '4': '020000', '5': '050000', '6': '100000',
    
    # Layer 2 - Nominal detection  
    '7': '001000', '8': '002000', '9': '005000', '10': '010000',
    '11': '020000', '12': '050000', '13': '100000',
    
    # Layer 3 - Security features (undetermined nominal)
    '14': '000000', '15': '000000', '16': '000000'
}

# Reverse mapping untuk lookup
NOMINAL_TO_CLASS_MAP = {
    '001000': [0, 7], '002000': [1, 8], '005000': [2, 9], '010000': [3, 10],
    '020000': [4, 11], '050000': [5, 12], '100000': [6, 13], '000000': [14, 15, 16]
}

# Research-ready nominal descriptions
NOMINAL_DESCRIPTIONS = {
    '001000': 'Rp1000', '002000': 'Rp2000', '005000': 'Rp5000', 
    '010000': 'Rp10000', '020000': 'Rp20000', '050000': 'Rp50000', 
    '100000': 'Rp100000', '000000': 'Undetermined'
}

# Class information dengan layer context
CLASS_INFO = {
    # Layer 1 - Banknote detection
    0: {'name': '001', 'desc': 'Rp1000', 'layer': 'banknote', 'nominal': '001000'},
    1: {'name': '002', 'desc': 'Rp2000', 'layer': 'banknote', 'nominal': '002000'},
    2: {'name': '005', 'desc': 'Rp5000', 'layer': 'banknote', 'nominal': '005000'},
    3: {'name': '010', 'desc': 'Rp10000', 'layer': 'banknote', 'nominal': '010000'},
    4: {'name': '020', 'desc': 'Rp20000', 'layer': 'banknote', 'nominal': '020000'},
    5: {'name': '050', 'desc': 'Rp50000', 'layer': 'banknote', 'nominal': '050000'},
    6: {'name': '100', 'desc': 'Rp100000', 'layer': 'banknote', 'nominal': '100000'},
    
    # Layer 2 - Nominal detection
    7: {'name': 'l2_001', 'desc': 'Rp1000', 'layer': 'nominal', 'nominal': '001000'},
    8: {'name': 'l2_002', 'desc': 'Rp2000', 'layer': 'nominal', 'nominal': '002000'},
    9: {'name': 'l2_005', 'desc': 'Rp5000', 'layer': 'nominal', 'nominal': '005000'},
    10: {'name': 'l2_010', 'desc': 'Rp10000', 'layer': 'nominal', 'nominal': '010000'},
    11: {'name': 'l2_020', 'desc': 'Rp20000', 'layer': 'nominal', 'nominal': '020000'},
    12: {'name': 'l2_050', 'desc': 'Rp50000', 'layer': 'nominal', 'nominal': '050000'},
    13: {'name': 'l2_100', 'desc': 'Rp100000', 'layer': 'nominal', 'nominal': '100000'},
    
    # Layer 3 - Security features
    14: {'name': 'l3_sign', 'desc': 'Tanda tangan', 'layer': 'security', 'nominal': '000000'},
    15: {'name': 'l3_text', 'desc': 'Teks mikro', 'layer': 'security', 'nominal': '000000'},
    16: {'name': 'l3_thread', 'desc': 'Benang pengaman', 'layer': 'security', 'nominal': '000000'}
}

# Layer priority untuk extraction (layer 1 & 2 prioritized)
LAYER_PRIORITY = {
    'banknote': 10,   # Highest priority
    'nominal': 9,     # Second priority  
    'security': 1     # Lowest priority (undetermined nominal)
}

# File naming strategies
NAMING_STRATEGIES = {
    'research_uuid': {
        'pattern': 'rp_{nominal}_{uuid}_{sequence}',
        'nominal_digits': 6,
        'sequence_digits': 3,
        'use_uuid': True
    },
    'simple': {
        'pattern': 'rp_{nominal}_{counter}',
        'nominal_digits': 6,
        'sequence_digits': 4,
        'use_uuid': False
    },
    'legacy': {
        'pattern': '{original_name}',
        'preserve_original': True,
        'use_uuid': False
    }
}

# Default naming strategy
DEFAULT_NAMING_STRATEGY = 'research_uuid'

# File validation patterns
VALID_FILENAME_PATTERNS = {
    'uuid_format': UUID_FILE_PATTERN,
    'legacy_rp': r'^rp_\d+_.*',
    'legacy_aug': r'^aug_.*',
    'legacy_img': r'^(IMG_|img_|image_).*'
}

# Batch processing constants
DEFAULT_BATCH_SIZE = 1000
MAX_BATCH_SIZE = 5000
MIN_BATCH_SIZE = 100

# Progress reporting intervals
PROGRESS_REPORT_INTERVAL = 0.05  # Report every 5%
BATCH_PROGRESS_INTERVAL = 0.10   # Report every 10% for batch ops

# Memory management
UUID_REGISTRY_MAX_SIZE = 10000
UUID_REGISTRY_CLEANUP_SIZE = 5000

# Performance tuning
PARALLEL_THRESHOLD = 100  # Minimum files untuk parallel processing
MAX_WORKERS_IO = 8       # Maximum workers untuk I/O operations
MAX_WORKERS_CPU = None   # CPU count untuk CPU operations

# Utility functions dengan one-liner style
def get_nominal_from_class_id(class_id: int) -> str:
    """Get nominal dari class ID dengan one-liner lookup"""
    return CLASS_INFO.get(class_id, {}).get('nominal', '000000')

def get_layer_from_class_id(class_id: int) -> str:
    """Get layer dari class ID dengan one-liner lookup"""
    return CLASS_INFO.get(class_id, {}).get('layer', 'unknown')

def is_uuid_format(filename: str) -> bool:
    """Check apakah filename menggunakan UUID format dengan one-liner regex"""
    return bool(UUID_FILENAME_REGEX.match(filename))

def extract_nominal_from_uuid_filename(filename: str) -> str:
    """Extract nominal dari UUID filename dengan one-liner regex"""
    match = UUID_FILENAME_REGEX.match(filename)
    return match.group(1) if match else '000000'

def extract_uuid_from_filename(filename: str) -> str:
    """Extract UUID dari filename dengan one-liner regex"""
    match = UUID_FILENAME_REGEX.match(filename)
    return match.group(2) if match else None

def get_class_priority(class_id: int) -> int:
    """Get priority untuk class ID berdasarkan layer dengan one-liner lookup"""
    layer = get_layer_from_class_id(class_id)
    return LAYER_PRIORITY.get(layer, 0)

def is_valid_image_extension(filename: str) -> bool:
    """Check apakah file extension valid untuk image dengan one-liner check"""
    return any(filename.lower().endswith(ext.strip('*')) for ext in IMG_EXTENSIONS)

def is_legacy_filename(filename: str) -> bool:
    """Check apakah filename menggunakan legacy format dengan one-liner check"""
    return any(pattern in filename.lower() for pattern in LEGACY_PATTERNS)

def get_denomination_info(class_id: int) -> dict:
    """Get comprehensive denomination info dengan one-liner lookup"""
    return CLASS_INFO.get(class_id, {'name': 'unknown', 'desc': 'Unknown', 'layer': 'unknown', 'nominal': '000000'})

# Validation functions
def validate_uuid_format(filename: str) -> dict:
    """Validate UUID format dan return detailed info"""
    match = UUID_FILENAME_REGEX.match(filename)
    if match:
        return {
            'valid': True,
            'nominal': match.group(1),
            'uuid': match.group(2),
            'sequence': match.group(3),
            'description': NOMINAL_DESCRIPTIONS.get(match.group(1), 'Unknown')
        }
    return {'valid': False, 'reason': 'Invalid UUID format'}

def get_filename_type(filename: str) -> str:
    """Determine filename type dengan one-liner checks"""
    if is_uuid_format(filename):
        return 'uuid_format'
    elif is_legacy_filename(filename):
        return 'legacy_format'  
    elif any(filename.startswith(prefix) for prefix in ['rp_', 'rupiah']):
        return 'rp_format'
    else:
        return 'unknown_format'

# Export constants untuk backward compatibility
__all__ = [
    'IMG_EXTENSIONS', 'IMG_EXTENSIONS_SET', 'DEFAULT_SPLITS', 'DEFAULT_SPLIT_RATIOS',
    'DEFAULT_RANDOM_SEED', 'DEFAULT_ORIGINAL_PREFIX', 'DEFAULT_AUGMENTATION_PREFIX',
    'UUID_PREFIX', 'UUID_FILE_PATTERN', 'UUID_FILENAME_REGEX', 'LEGACY_PATTERNS',
    'DEFAULT_IMG_SIZE', 'DENOMINATION_CLASS_MAP', 'NOMINAL_TO_CLASS_MAP',
    'NOMINAL_DESCRIPTIONS', 'CLASS_INFO', 'LAYER_PRIORITY', 'NAMING_STRATEGIES',
    'DEFAULT_NAMING_STRATEGY', 'VALID_FILENAME_PATTERNS', 'DEFAULT_BATCH_SIZE',
    'get_nominal_from_class_id', 'get_layer_from_class_id', 'is_uuid_format',
    'extract_nominal_from_uuid_filename', 'extract_uuid_from_filename',
    'get_class_priority', 'is_valid_image_extension', 'is_legacy_filename',
    'get_denomination_info', 'validate_uuid_format', 'get_filename_type'
]