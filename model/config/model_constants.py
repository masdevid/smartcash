"""
File: smartcash/model/config/model_constants.py
Deskripsi: Konstanta-konstanta untuk konfigurasi model
"""

import torch
from typing import Dict, Any, List

# Definisi layer deteksi untuk digunakan di seluruh aplikasi
DETECTION_LAYERS = ['banknote', 'nominal', 'security']

# Definisi threshold default untuk setiap layer deteksi
DETECTION_THRESHOLDS = {
    'banknote': 0.25,
    'nominal': 0.30,
    'security': 0.35
}

# Konfigurasi lengkap untuk setiap layer deteksi
LAYER_CONFIG = {
    'banknote': {
        'num_classes': 7,
        'classes': [
            {'id': 0, 'name': '001', 'desc': 'Rp1000'},
            {'id': 1, 'name': '002', 'desc': 'Rp2000'},
            {'id': 2, 'name': '005', 'desc': 'Rp5000'},
            {'id': 3, 'name': '010', 'desc': 'Rp10000'},
            {'id': 4, 'name': '020', 'desc': 'Rp20000'},
            {'id': 5, 'name': '050', 'desc': 'Rp50000'},
            {'id': 6, 'name': '100', 'desc': 'Rp100000'},
        ],
        'description': 'Deteksi uang kertas utuh'
    },
    'nominal': {
        'num_classes': 7,
        'classes': [
            {'id': 7, 'name': 'l2_001', 'desc': 'Rp1000'},
            {'id': 8, 'name': 'l2_002', 'desc': 'Rp2000'},
            {'id': 9, 'name': 'l2_005', 'desc': 'Rp5000'},
            {'id': 10, 'name': 'l2_010', 'desc': 'Rp10000'},
            {'id': 11, 'name': 'l2_020', 'desc': 'Rp20000'},
            {'id': 12, 'name': 'l2_050', 'desc': 'Rp50000'},
            {'id': 13, 'name': 'l2_100', 'desc': 'Rp100000'},
        ],
        'description': 'Deteksi area nominal'
    },
    'security': {
        'num_classes': 3,
        'classes': [
            {'id': 14, 'name': 'l3_sign', 'desc': 'Tanda tangan'},
            {'id': 15, 'name': 'l3_text', 'desc': 'Teks mikro'},
            {'id': 16, 'name': 'l3_thread', 'desc': 'Benang pengaman'},
        ],
        'description': 'Deteksi fitur keamanan'
    }
}

LAYER_CONFIG_FLAT = [
    {'id': 0, 'layer': 'banknote', 'name': '001', 'desc': 'Rp1000'},
    {'id': 1, 'layer': 'banknote', 'name': '002', 'desc': 'Rp2000'},
    {'id': 2, 'layer': 'banknote', 'name': '005', 'desc': 'Rp5000'},
    {'id': 3, 'layer': 'banknote', 'name': '010', 'desc': 'Rp10000'},
    {'id': 4, 'layer': 'banknote', 'name': '020', 'desc': 'Rp20000'},
    {'id': 5, 'layer': 'banknote', 'name': '050', 'desc': 'Rp50000'},
    {'id': 6, 'layer': 'banknote', 'name': '100', 'desc': 'Rp100000'},
    
    {'id': 7, 'layer': 'nominal', 'name': 'l2_001', 'desc': 'Rp1000'},
    {'id': 8, 'layer': 'nominal', 'name': 'l2_002', 'desc': 'Rp2000'},
    {'id': 9, 'layer': 'nominal', 'name': 'l2_005', 'desc': 'Rp5000'},
    {'id': 10, 'layer': 'nominal', 'name': 'l2_010', 'desc': 'Rp10000'},
    {'id': 11, 'layer': 'nominal', 'name': 'l2_020', 'desc': 'Rp20000'},
    {'id': 12, 'layer': 'nominal', 'name': 'l2_050', 'desc': 'Rp50000'},
    {'id': 13, 'layer': 'nominal', 'name': 'l2_100', 'desc': 'Rp100000'},
    
    {'id': 14, 'layer': 'security', 'name': 'l3_sign', 'desc': 'Tanda tangan'},
    {'id': 15, 'layer': 'security', 'name': 'l3_text', 'desc': 'Teks mikro'},
    {'id': 16, 'layer': 'security', 'name': 'l3_thread', 'desc': 'Benang pengaman'}
]


# Daftar variasi model EfficientNet yang didukung
SUPPORTED_EFFICIENTNET_MODELS = ['efficientnet_b4']

# Definisi channel output untuk setiap stage EfficientNet
EFFICIENTNET_CHANNELS = {
    'efficientnet_b4': [56, 160, 448],
}

# Output channels standar yang digunakan YOLOv5 untuk feature maps
YOLO_CHANNELS = [128, 256, 512]  # P3, P4, P5 stages

# Default feature indices untuk EfficientNet (P3, P4, P5 stages)
DEFAULT_EFFICIENTNET_INDICES = [2, 3, 4]  # Indeks untuk mengambil feature maps dari stage 3, 4, dan 5

# Konfigurasi model YOLOv5 untuk CSPDarknet backbone
YOLOV5_CONFIG = {
    'yolov5s': {
        'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt',
        'feature_indices': [4, 6, 9],  # P3, P4, P5 layers
        'expected_channels': [128, 256, 512],
        'expected_shapes': [(80, 80), (40, 40), (20, 20)],  # untuk input 640x640
    }
}

# Enum untuk backbone yang didukung
SUPPORTED_BACKBONES = {
    'efficientnet_b4': {
        'type': 'efficientnet', 
        'variant': 'efficientnet_b4',
        'stride': 32,
        'width_coefficient': 1.4,
        'depth_coefficient': 1.8,
        'features': 1792,
        'stages': [32, 56, 160, 1792]
    },
    'cspdarknet_s': {
        'type': 'cspdarknet', 
        'variant': 'yolov5s',
        'stride': 32,
        'width_coefficient': 1.0,
        'depth_coefficient': 1.0,
        'features': 1024,
        'stages': [64, 128, 256, 1024]
    }
}

# Model optimasi yang didukung
OPTIMIZED_MODELS = {
    'yolov5s': {
        'description': 'YOLOv5s dengan CSPDarknet sebagai backbone (model pembanding)',
        'backbone': 'cspdarknet_s',
        'use_attention': False,
        'use_residual': False,
        'use_ciou': False,
        'detection_layers': ['banknote'],
        'num_classes': 7,
        'img_size': 640,
        'pretrained': True
    },
    'efficient_basic': {
        'description': 'Model dasar tanpa optimasi khusus',
        'backbone': 'efficientnet_b4',
        'use_attention': False,
        'use_residual': False,
        'use_ciou': False,
        'detection_layers': ['banknote'],
        'num_classes': 7,
        'img_size': 640,
        'pretrained': True
    },
    'efficient_optimized': {
        'description': 'Model dengan EfficientNet-B4 dan FeatureAdapter',
        'backbone': 'efficientnet_b4',
        'use_attention': True,
        'use_residual': False,
        'use_ciou': False,
        'detection_layers': ['banknote'],
        'num_classes': 7,
        'img_size': 640,
        'pretrained': True
    },
    'efficient_advanced': {
        'description': 'Model dengan semua optimasi: FeatureAdapter, ResidualAdapter, dan CIoU',
        'backbone': 'efficientnet_b4',
        'use_attention': True,
        'use_residual': True,
        'use_ciou': True,
        'detection_layers': ['banknote'],
        'num_classes': 7,
        'img_size': 640,
        'pretrained': True
    }
}

# Default konfigurasi model
DEFAULT_MODEL_CONFIG = {
    'backbone': 'efficientnet_b4',
    'img_size': 640,
    'pretrained': True,
    'use_attention': True,
    'use_residual': True,
    'use_ciou': True,
    'detection_layers': DETECTION_LAYERS,
    'num_classes': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Default konfigurasi model lengkap untuk ModelConfig
DEFAULT_MODEL_CONFIG_FULL = {
    'model': {
        'name': 'smartcash_model',
        'img_size': [640, 640],
        'batch_size': 16,
        'workers': 4,
        'backbone': 'efficientnet_b4',
        'model_type': 'efficient_optimized',
        'pretrained': True,
        'freeze_backbone': True,
        'use_attention': True,
        'use_residual': False,
        'use_ciou': False,
        'num_repeats': 3
    },
    'training': {
        'epochs': 100,
        'lr': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'fliplr': 0.5,
        'flipud': 0.0,
        'scale': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'translate': 0.1,
        'degrees': 0.0
    },
    'layers': {
        'banknote': {'enabled': True, 'threshold': DETECTION_THRESHOLDS['banknote']},
        'nominal': {'enabled': True, 'threshold': DETECTION_THRESHOLDS['nominal']},
        'security': {'enabled': True, 'threshold': DETECTION_THRESHOLDS['security']}
    },
    'optimizer': {
        'type': 'SGD',
        'params': {}
    },
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'params': {}
    }
}

# Fungsi utilitas untuk mendapatkan konfigurasi model berdasarkan tipe
def get_model_config(model_type: str) -> Dict[str, Any]:
    """Dapatkan konfigurasi untuk model tertentu berdasarkan tipe model."""
    return OPTIMIZED_MODELS.get(model_type, OPTIMIZED_MODELS['efficient_optimized']).copy()
