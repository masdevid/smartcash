"""
File: smartcash/ui/training_config/backbone/handlers/defaults.py
Deskripsi: Default backbone configuration values dengan one-liner structure
"""

from typing import Dict, Any

def get_default_backbone_config() -> Dict[str, Any]:
    """Get default backbone configuration dengan complete structure"""
    return {
        'model': {
            # Type model
            'type': 'efficient_basic',  # Tipe model: 'efficient_basic' atau 'yolov5s'
            
            # Backbone model
            'backbone': 'efficientnet_b4',  # Backbone yang didukung: 'efficientnet_b4' atau 'cspdarknet_s'
            'backbone_pretrained': True,
            'backbone_weights': '',  # Path ke file weights kustom
            'backbone_freeze': False,  # Freeze backbone selama training
            'backbone_unfreeze_epoch': 5,  # Unfreeze backbone setelah epoch ini
            
            # Ukuran input dan preprocessing
            'input_size': [640, 640],
            
            # Thresholds
            'confidence': 0.25,  # Confidence threshold untuk deteksi
            'iou_threshold': 0.45,  # IoU threshold untuk NMS
            'max_detections': 100,  # Jumlah maksimum deteksi yang dihasilkan
            
            # Transfer learning
            'transfer_learning': True,
            'pretrained': True,
            'pretrained_weights': '',  # Path ke file weights YOLOv5
            
            # Processing
            'anchors': [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326],
            'strides': [8, 16, 32],
            'workers': 4,  # Jumlah worker untuk dataloader
            
            # Spesifikasi model
            'depth_multiple': 0.67,  # Faktor skala untuk kedalaman model (YOLOv5s)
            'width_multiple': 0.75,  # Faktor skala untuk lebar model (YOLOv5s)
            
            # Integrasi khusus untuk SmartCash
            'use_efficient_blocks': True,  # Gunakan blok EfficientNet di FPN/PAN
            'use_adaptive_anchors': True,  # Adaptasi anchors berdasarkan dataset
            
            # Optimasi model
            'quantization': False,
            'quantization_aware_training': False,  # Gunakan QAT untuk model lebih ringan
            'fp16_training': True,  # Gunakan mixed precision untuk training
            
            # Feature optimization (khusus UI)
            'use_attention': True,
            'use_residual': True,
            'use_ciou': False
        },
        'config_version': '1.0',
        'description': 'Default backbone configuration untuk SmartCash detection'
    }

# One-liner option getters
get_backbone_options = lambda: [('EfficientNet-B4', 'efficientnet_b4'), ('CSPDarknet-S', 'cspdarknet_s')]
get_model_type_options = lambda: [('EfficientNet Basic', 'efficient_basic'), ('YOLOv5s', 'yolov5s')]
get_optimization_features = lambda: [('FeatureAdapter (Attention)', 'use_attention'), ('ResidualAdapter (Residual)', 'use_residual'), ('CIoU Loss', 'use_ciou')]