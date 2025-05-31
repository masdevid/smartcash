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
        'num_classes': 7,  # 7 denominasi ('001', '002', '005', '010', '020', '050', '100')
        'description': 'Deteksi uang kertas utuh'
    },
    'nominal': {
        'num_classes': 7,  # 7 area nominal ('l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100')
        'description': 'Deteksi area nominal'
    },
    'security': {
        'num_classes': 3,  # 3 fitur keamanan ('l3_sign', 'l3_text', 'l3_thread')
        'description': 'Deteksi fitur keamanan'
    }
}

# Daftar variasi model EfficientNet yang didukung
SUPPORTED_EFFICIENTNET_MODELS = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5']

# Definisi channel output untuk setiap stage EfficientNet
EFFICIENTNET_CHANNELS = {
    'efficientnet_b0': [24, 48, 208],  # P3, P4, P5 stages
    'efficientnet_b1': [32, 88, 320],
    'efficientnet_b2': [32, 112, 352], 
    'efficientnet_b3': [40, 112, 384],
    'efficientnet_b4': [56, 160, 448],
    'efficientnet_b5': [64, 176, 512],
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
    },
    'yolov5m': {
        'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m.pt',
        'feature_indices': [4, 6, 9],
        'expected_channels': [192, 384, 768],
        'expected_shapes': [(80, 80), (40, 40), (20, 20)],
    },
    'yolov5l': {
        'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l.pt',
        'feature_indices': [4, 6, 9],
        'expected_channels': [256, 512, 1024],
        'expected_shapes': [(80, 80), (40, 40), (20, 20)],
    }
}

# Enum untuk backbone yang didukung
SUPPORTED_BACKBONES = {
    'efficientnet_b0': {
        'type': 'efficientnet', 
        'variant': 'efficientnet_b0',
        'stride': 32,
        'width_coefficient': 1.0,
        'depth_coefficient': 1.0,
        'features': 1280,
        'stages': [24, 40, 112, 1280]
    },
    'efficientnet_b1': {
        'type': 'efficientnet', 
        'variant': 'efficientnet_b1',
        'stride': 32,
        'width_coefficient': 1.0,
        'depth_coefficient': 1.1,
        'features': 1280,
        'stages': [24, 40, 112, 1280]
    },
    'efficientnet_b2': {
        'type': 'efficientnet', 
        'variant': 'efficientnet_b2',
        'stride': 32,
        'width_coefficient': 1.1,
        'depth_coefficient': 1.2,
        'features': 1408,
        'stages': [24, 48, 120, 1408]
    },
    'efficientnet_b3': {
        'type': 'efficientnet', 
        'variant': 'efficientnet_b3',
        'stride': 32,
        'width_coefficient': 1.2,
        'depth_coefficient': 1.4,
        'features': 1536,
        'stages': [32, 48, 136, 1536]
    },
    'efficientnet_b4': {
        'type': 'efficientnet', 
        'variant': 'efficientnet_b4',
        'stride': 32,
        'width_coefficient': 1.4,
        'depth_coefficient': 1.8,
        'features': 1792,
        'stages': [32, 56, 160, 1792]
    },
    'efficientnet_b5': {
        'type': 'efficientnet', 
        'variant': 'efficientnet_b5',
        'stride': 32,
        'width_coefficient': 1.6,
        'depth_coefficient': 2.2,
        'features': 2048,
        'stages': [40, 64, 176, 2048]
    },
    'cspdarknet_s': {
        'type': 'cspdarknet', 
        'variant': 'yolov5s',
        'stride': 32,
        'width_coefficient': 1.0,
        'depth_coefficient': 1.0,
        'features': 1024,
        'stages': [64, 128, 256, 1024]
    },
    'cspdarknet_m': {
        'type': 'cspdarknet', 
        'variant': 'yolov5m',
        'stride': 32,
        'width_coefficient': 1.0,
        'depth_coefficient': 1.0,
        'features': 1024,
        'stages': [96, 192, 384, 1024]
    },
    'cspdarknet_l': {
        'type': 'cspdarknet', 
        'variant': 'yolov5l',
        'stride': 32,
        'width_coefficient': 1.0,
        'depth_coefficient': 1.0,
        'features': 1024,
        'stages': [128, 256, 512, 1024]
    },
    'cspdarknet_x': {
        'type': 'cspdarknet', 
        'variant': 'yolov5l', # Menggunakan yolov5l karena yolov5x tidak didukung
        'stride': 32,
        'width_coefficient': 1.0,
        'depth_coefficient': 1.0,
        'features': 1024,
        'stages': [160, 320, 640, 1024]
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
