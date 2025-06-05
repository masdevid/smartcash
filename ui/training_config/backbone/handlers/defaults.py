"""
File: smartcash/ui/training_config/backbone/handlers/defaults.py
Deskripsi: Default backbone configuration values dengan one-liner structure
"""

from typing import Dict, Any

def get_default_backbone_config() -> Dict[str, Any]:
    """Get default backbone configuration dengan struktur yang sesuai dengan model_config.yaml"""
    return {
        # Override konfigurasi model dari base_config
        'model': {
            # Parameter baru yang tidak ada di base_config
            'type': 'efficient_basic',  # Tipe model: 'efficient_basic' atau 'yolov5s'
            'backbone': 'efficientnet_b4',  # Backbone yang didukung
            'backbone_weights': '',  # Path ke file weights kustom
            'backbone_unfreeze_epoch': 5,  # Unfreeze backbone setelah epoch ini
            'pretrained_weights': '',  # Path ke file weights YOLOv5
            
            # Parameter processing yang baru
            'anchors': [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326],
            'strides': [8, 16, 32],
            
            # Parameter spesifikasi model yang baru
            'depth_multiple': 0.67,  # Faktor skala untuk kedalaman model
            'width_multiple': 0.75,  # Faktor skala untuk lebar model
            
            # Parameter integrasi khusus yang baru
            'use_efficient_blocks': True,  # Gunakan blok EfficientNet di FPN/PAN
            'use_adaptive_anchors': True,  # Adaptasi anchors berdasarkan dataset
            
            # Parameter optimasi model yang baru
            'quantization': False,
            'quantization_aware_training': False,
            'fp16_training': True,
            
            # Parameter fitur optimasi tambahan yang baru
            'use_attention': False,
            'use_residual': False,
            'use_ciou': False,
            
            # Parameter dari base_config yang dipertahankan
            'input_size': [640, 640],
            'confidence': 0.25,
            'iou_threshold': 0.45,
            'max_detections': 100,
            'transfer_learning': True,
            'pretrained': True,
            'workers': 4,
            'freeze_backbone': False
        },
        
        # Parameter khusus EfficientNet-B4 (parameter baru)
        'efficientnet': {
            'width_coefficient': 1.4,
            'depth_coefficient': 1.8,
            'resolution': 380,
            'dropout_rate': 0.4
        },
        
        # Parameter transfer learning (parameter baru)
        'transfer_learning': {
            'freeze_batch_norm': False,
            'unfreeze_after_epochs': 10
        },
        
        # Parameter regularisasi khusus model (parameter baru)
        'regularization': {
            'label_smoothing': 0.0
        },
        
        # Konfigurasi kompilasi model untuk deployment (parameter baru)
        'export': {
            'formats': ['onnx', 'torchscript'],
            'dynamic_batch': True,
            'optimize': True,
            'half_precision': True,
            'simplify': True
        },
        
        # Konfigurasi khusus untuk uji coba model (parameter baru)
        'experiments': {
            'backbones': [
                {
                    'name': 'cspdarknet_s',
                    'description': 'YOLOv5s default backbone',
                    'config': {
                        'backbone': 'cspdarknet_s',
                        'pretrained': True
                    }
                },
                {
                    'name': 'efficientnet_b4',
                    'description': 'EfficientNet-B4 backbone',
                    'config': {
                        'backbone': 'efficientnet_b4',
                        'pretrained': True
                    }
                }
            ],
            'scenarios': [
                {
                    'name': 'efficient_advanced',
                    'description': 'Model dengan semua optimasi: FeatureAdapter, ResidualAdapter, dan CIoU',
                    'config': {
                        'type': 'efficient_advanced',
                        'backbone': 'efficientnet_b4',
                        'use_attention': True,
                        'use_residual': True,
                        'use_ciou': True
                    }
                }
            ]
        },
        
        # Metadata
        'config_version': '1.0',
        'description': 'Default backbone configuration untuk SmartCash detection'
    }

# One-liner option getters
get_backbone_options = lambda: [('EfficientNet-B4', 'efficientnet_b4'), ('CSPDarknet-S', 'cspdarknet_s')]
get_model_type_options = lambda: [('EfficientNet Basic', 'efficient_basic'), ('YOLOv5s', 'yolov5s')]
get_optimization_features = lambda: [('FeatureAdapter (Attention)', 'use_attention'), ('ResidualAdapter (Residual)', 'use_residual'), ('CIoU Loss', 'use_ciou')]