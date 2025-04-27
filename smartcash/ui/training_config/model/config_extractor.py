"""
File: smartcash/ui/training_config/model_config_extractor.py
Deskripsi: Ekstraksi konfigurasi dari komponen UI backbone & model
"""

from typing import Dict, Any, Optional

from smartcash.ui.training_config.model_config_definitions import get_model_config, MODEL_TYPE_CONFIGS

def extract_config_from_ui(ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ekstrak konfigurasi model dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi existing (opsional)
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    if config is None:
        config = {}
        
    # Get model type
    model_options = ui_components.get('model_options')
    if not model_options or not hasattr(model_options, 'children') or len(model_options.children) < 1:
        return config
            
    model_dropdown = model_options.children[0]
    model_option = model_dropdown.value
    model_type = model_option.split(' - ')[0].strip()
    
    # Get backbone config
    backbone_options = ui_components.get('backbone_options')
    if not backbone_options or not hasattr(backbone_options, 'children') or len(backbone_options.children) < 3:
        # Fallback ke default
        backbone_type = MODEL_TYPE_CONFIGS[model_type]['backbone']
        pretrained = True
        freeze_backbone = True
    else:
        backbone_dropdown = backbone_options.children[0]
        if hasattr(backbone_dropdown, 'options') and hasattr(backbone_dropdown, 'index'):
            backbone_option = backbone_dropdown.options[backbone_dropdown.index]
            backbone_type = backbone_option.split(' - ')[0].strip()
            pretrained = backbone_options.children[1].value
            freeze_backbone = backbone_options.children[2].value
        else:
            backbone_type = MODEL_TYPE_CONFIGS[model_type]['backbone']
            pretrained = True
            freeze_backbone = True
    
    # Get features config
    features_options = ui_components.get('features_options')
    if not features_options or not hasattr(features_options, 'children') or len(features_options.children) < 4:
        # Fallback ke default untuk tipe model
        use_attention = MODEL_TYPE_CONFIGS[model_type]['use_attention']
        use_residual = MODEL_TYPE_CONFIGS[model_type]['use_residual']
        use_ciou = MODEL_TYPE_CONFIGS[model_type]['use_ciou']
        num_repeats = MODEL_TYPE_CONFIGS[model_type]['num_repeats']
    else:
        use_attention = features_options.children[0].value
        use_residual = features_options.children[1].value
        use_ciou = features_options.children[2].value
        num_repeats = features_options.children[3].value
    
    # Get layer config
    layer_config = ui_components.get('layer_config')
    layers = {}
    layer_names = ['banknote', 'nominal', 'security']
    
    if layer_config and hasattr(layer_config, 'children'):
        for i, layer_name in enumerate(layer_names):
            if i < len(layer_config.children):
                layer_row = layer_config.children[i]
                if hasattr(layer_row, 'children') and len(layer_row.children) >= 2:
                    layers[layer_name] = {
                        'enabled': layer_row.children[0].value,
                        'threshold': layer_row.children[1].value
                    }
    else:
        # Fallback untuk layer config - default values
        for layer_name in layer_names:
            layers[layer_name] = {'enabled': True, 'threshold': 0.25 + (i * 0.05)}
    
    # Memastikan model config ada
    if 'model' not in config:
        config['model'] = {}
    
    # Update model & settings
    config['model'].update({
        'model_type': model_type,
        'backbone': backbone_type,
        'pretrained': pretrained,
        'freeze_backbone': freeze_backbone,
        'use_attention': use_attention,
        'use_residual': use_residual,
        'use_ciou': use_ciou,
        'num_repeats': num_repeats
    })
    
    # Update layer config
    config['layers'] = layers
    
    return config