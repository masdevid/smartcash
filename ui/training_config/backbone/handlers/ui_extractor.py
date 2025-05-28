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
    model_config = {
        'backbone': getattr(ui_components.get('backbone_dropdown'), 'value', 'efficientnet_b4'),
        'model_type': getattr(ui_components.get('model_type_dropdown'), 'value', 'efficient_basic'),
        'use_attention': getattr(ui_components.get('use_attention_checkbox'), 'value', True),
        'use_residual': getattr(ui_components.get('use_residual_checkbox'), 'value', True),
        'use_ciou': getattr(ui_components.get('use_ciou_checkbox'), 'value', False),
        'pretrained': True,  # Always use pretrained
        'freeze_backbone': False,
        'freeze_bn': False,
        'dropout': 0.2,
        'activation': 'relu',
        'normalization': {'type': 'batch_norm', 'momentum': 0.1},
        'weights': {'path': '', 'strict': True}
    }
    
    return {
        'model': model_config,
        'updated_at': datetime.now().isoformat(),
        'config_version': '1.0'
    }