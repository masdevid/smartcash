"""
File: smartcash/dataset/preprocessor/config/defaults.py
Deskripsi: Default configurations dan presets untuk preprocessing
"""

# === NORMALIZATION PRESETS ===
NORMALIZATION_PRESETS = {
    'default': {
        'target_size': [640, 640],
        'pixel_range': [0, 1],
        'preserve_aspect_ratio': True,
        'pad_color': 114,
        'interpolation': 'linear'
    },
    'yolov5s': {
        'target_size': [640, 640],
        'pixel_range': [0, 1],
        'preserve_aspect_ratio': True,
        'pad_color': 114,
        'interpolation': 'linear'
    },
    'yolov5m': {
        'target_size': [640, 640],
        'pixel_range': [0, 1],
        'preserve_aspect_ratio': True,
        'pad_color': 114,
        'interpolation': 'linear'
    },
    'yolov5l': {
        'target_size': [832, 832],
        'pixel_range': [0, 1],
        'preserve_aspect_ratio': True,
        'pad_color': 114,
        'interpolation': 'linear'
    },
    'yolov5x': {
        'target_size': [1024, 1024],
        'pixel_range': [0, 1],
        'preserve_aspect_ratio': True,
        'pad_color': 114,
        'interpolation': 'linear'
    },
    'inference': {
        'target_size': [640, 640],
        'pixel_range': [0, 1],
        'preserve_aspect_ratio': True,
        'pad_color': 114,
        'interpolation': 'linear',
        'batch_processing': True
    }
}

# === MAIN BANKNOTES CLASSES ===
MAIN_BANKNOTE_CLASSES = {
    0: {'nominal': '001000', 'display': 'Rp1.000', 'value': 1000},
    1: {'nominal': '002000', 'display': 'Rp2.000', 'value': 2000},
    2: {'nominal': '005000', 'display': 'Rp5.000', 'value': 5000},
    3: {'nominal': '010000', 'display': 'Rp10.000', 'value': 10000},
    4: {'nominal': '020000', 'display': 'Rp20.000', 'value': 20000},
    5: {'nominal': '050000', 'display': 'Rp50.000', 'value': 50000},
    6: {'nominal': '100000', 'display': 'Rp100.000', 'value': 100000}
}

# === LAYER MAPPING ===
LAYER_CLASSES = {
    'l1_main': list(range(0, 7)),      # Main banknotes (0-6)
    'l2_security': list(range(7, 14)), # Security features (7-13)  
    'l3_micro': list(range(14, 21))    # Micro features (14-20)
}

def get_default_config():
    """🔧 Default preprocessing configuration"""
    return {
        'preprocessing': {
            'enabled': True,
            'validation': {
                'enabled': True,   # Enable sample validation by default
                'filename_pattern': True,
                'directory_structure': True,
                'auto_fix': True,
                'min_bbox_size': 0.001,
                'min_valid_boxes': 1,
                'quarantine_invalid': True
            },
            'normalization': NORMALIZATION_PRESETS['default'].copy(),
            'output': {
                'format': 'npy',
                'preserve_originals': False,
                'organize_by_split': True
            },
            'target_splits': ['train', 'valid'],
            'sample_size': 0  # 0 = process all
        },
        'data': {
            'dir': 'data',
            'preprocessed_dir': 'data/preprocessed',
            'samples_dir': 'data/samples',
            'invalid_dir': 'data/invalid',
            'local': {}
        },
        'performance': {
            'batch_size': 32,
            'use_threading': True,
            'max_workers': 4
        },
        'file_naming': {
            'raw_pattern': r'rp_(\d{6})_([a-f0-9-]{36})_(\d+)\.(\w+)',
            'preprocessed_prefix': 'pre_',
            'augmented_prefix': 'aug_',
            'sample_prefix': 'sample_'
        }
    }

def get_yolo_config(model_variant='yolov5s'):
    """🎯 Get YOLO-specific configuration"""
    config = get_default_config()
    if model_variant in NORMALIZATION_PRESETS:
        config['preprocessing']['normalization'] = NORMALIZATION_PRESETS[model_variant].copy()
    return config

def get_inference_config():
    """🔮 Get inference-optimized configuration"""
    config = get_default_config()
    config['preprocessing']['normalization'] = NORMALIZATION_PRESETS['inference'].copy()
    config['preprocessing']['validation']['enabled'] = False
    config['performance']['batch_size'] = 64
    return config