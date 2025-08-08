"""
Layer-related constants and configurations.

This module contains constants and configurations related to model layers,
including layer names, detection layers, and layer-specific settings.
"""

# Layer names and configurations
LAYER_NAMES = ['banknote', 'denomination', 'security_feature']

# Detection layers configuration
DETECTION_LAYERS = {
    'banknote': {
        'num_classes': 7,  # Banknote denominations
        'description': 'Banknote detection layer',
        'active': True,
    },
    'denomination': {
        'num_classes': 7,  # Same as banknote for now
        'description': 'Denomination detection layer',
        'active': True,
    },
    'security_feature': {
        'num_classes': 3,  # Security features
        'description': 'Security feature detection layer',
        'active': True,
    }
}

# Layer configuration for backward compatibility
LAYER_CONFIG = {
    layer_name: {
        'num_classes': config['num_classes'],
        'description': config['description']
    }
    for layer_name, config in DETECTION_LAYERS.items()
}
# Add layer-specific configurations
for layer in LAYER_NAMES:
    if layer not in LAYER_CONFIG:
        LAYER_CONFIG[layer] = {
            'num_classes': 1,
            'description': f'{layer.capitalize()} detection layer'
        }
