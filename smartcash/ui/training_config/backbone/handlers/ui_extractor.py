"""
File: smartcash/ui/training_config/backbone/handlers/ui_extractor.py
Deskripsi: Extract backbone configuration dari UI components dengan one-liner pattern
"""

from typing import Dict, Any
from datetime import datetime

def extract_backbone_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract backbone configuration dari UI components.
    
    Args:
        ui_components: Dictionary berisi UI components
        
    Returns:
        Dictionary konfigurasi backbone
    """
    # One-liner extraction dengan safe get dan fallback values
    backbone_value = getattr(ui_components.get('backbone_dropdown'), 'value', 'efficientnet_b4')
    model_type_value = getattr(ui_components.get('model_type_dropdown'), 'value', 'efficient_basic')
    
    # Dapatkan nilai dari UI atau gunakan default
    model_config = {
        # Type model
        'type': model_type_value,
        
        # Backbone model
        'backbone': backbone_value,
        'backbone_pretrained': True,
        'backbone_weights': '',
        'backbone_freeze': False,
        'backbone_unfreeze_epoch': 5,
        
        # Ukuran input dan preprocessing
        'input_size': [640, 640],
        
        # Thresholds
        'confidence': 0.25,
        'iou_threshold': 0.45,
        'max_detections': 100,
        
        # Transfer learning
        'transfer_learning': True,
        'pretrained': True,
        'pretrained_weights': '',
        
        # Feature optimization dari UI
        'use_attention': getattr(ui_components.get('use_attention_checkbox'), 'value', True),
        'use_residual': getattr(ui_components.get('use_residual_checkbox'), 'value', True),
        'use_ciou': getattr(ui_components.get('use_ciou_checkbox'), 'value', False),
        
        # Processing dan spesifikasi model default
        'anchors': [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326],
        'strides': [8, 16, 32],
        'workers': 4,
        'depth_multiple': 0.67,
        'width_multiple': 0.75,
        
        # Integrasi khusus SmartCash
        'use_efficient_blocks': True,
        'use_adaptive_anchors': True,
        
        # Optimasi model
        'quantization': False,
        'quantization_aware_training': False,
        'fp16_training': True
    }
    
    return {
        'model': model_config,
        'updated_at': datetime.now().isoformat(),
        'config_version': '1.0'
    }