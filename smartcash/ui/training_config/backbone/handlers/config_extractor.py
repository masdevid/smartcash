"""
File: smartcash/ui/training_config/backbone/handlers/config_extractor.py
Deskripsi: Extract backbone configuration dari UI components dengan one-liner pattern
"""

from typing import Dict, Any
from datetime import datetime

def extract_backbone_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract backbone configuration dari UI components sesuai struktur model_config.yaml.
    
    Args:
        ui_components: Dictionary berisi UI components
        
    Returns:
        Dictionary konfigurasi backbone
    """
    # One-liner extraction dengan safe get dan fallback values
    get_val = lambda key, default: getattr(ui_components.get(key), 'value', default) if key in ui_components else default
    
    # Dapatkan nilai backbone dan model type dari UI
    backbone_value = get_val('backbone_dropdown', 'efficientnet_b4')
    model_type_value = get_val('model_type_dropdown', 'efficient_optimized')
    
    # Metadata untuk config yang diperbarui
    current_time = datetime.now().isoformat()
    
    # Buat config sesuai dengan model_config.yaml yang diperbarui
    return {
        # Inherit dari base_config.yaml
        '_base_': 'base_config.yaml',
        
        # Konfigurasi model
        'model': {
            # Type model dan backbone
            'type': model_type_value,
            'backbone': backbone_value,
            
            # Konfigurasi backbone
            'backbone_pretrained': True,
            'backbone_weights': '',
            'backbone_freeze': get_val('freeze_backbone_checkbox', False),
            'backbone_unfreeze_epoch': get_val('unfreeze_epoch_slider', 5),
            
            # Ukuran input dan preprocessing
            'input_size': [640, 640],
            'normalize': True,
            
            # Thresholds untuk deteksi
            'confidence': get_val('confidence_threshold_slider', 0.25),
            'iou_threshold': get_val('iou_threshold_slider', 0.45),
            'max_detections': 100,
            
            # Transfer learning
            'transfer_learning': {
                'enabled': get_val('transfer_learning_checkbox', True),
                'weights': get_val('pretrained_weights_text', ''),
                'freeze_layers': get_val('freeze_layers_slider', 0),
                'unfreeze_schedule': {
                    'enabled': get_val('unfreeze_schedule_checkbox', True),
                    'start_epoch': get_val('unfreeze_start_epoch_slider', 5),
                    'layers_per_epoch': get_val('layers_per_epoch_slider', 1)
                }
            },
            
            # Feature optimization dari UI
            'use_attention': get_val('use_attention_checkbox', True),
            'use_residual': get_val('use_residual_checkbox', True),
            'use_ciou': get_val('use_ciou_checkbox', False),
            
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
            'quantization': get_val('quantization_checkbox', False),
            'quantization_aware_training': get_val('qat_checkbox', False),
            'fp16_training': get_val('fp16_checkbox', True),
            
            # EfficientNet-B4 specific parameters
            'efficientnet': {
                'version': 'b4',
                'width_coefficient': 1.4,
                'depth_coefficient': 1.8,
                'resolution': 640,
                'dropout_rate': 0.4,
                'attention': {
                    'enabled': get_val('use_attention_checkbox', True),
                    'type': 'se',  # Squeeze-and-Excitation
                    'ratio': 0.25
                },
                'activation': 'swish'
            },
            
            # Export settings
            'export': {
                'formats': ['onnx', 'tflite', 'coreml'],
                'quantize_export': get_val('quantize_export_checkbox', False),
                'include_nms': True,
                'batch_size': 1
            },
            
            # Eksperimen backbone
            'experiments': {
                'enabled': get_val('experiments_enabled_checkbox', False),
                'scenario': get_val('experiment_scenario_dropdown', 'baseline'),
                'compare_with_baseline': True,
                'metrics': ['mAP_0.5', 'inference_time']
            }
        },
        
        # Metadata
        'updated_at': current_time,
        'config_version': '1.0'
    }