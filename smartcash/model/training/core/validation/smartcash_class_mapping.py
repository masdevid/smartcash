#!/usr/bin/env python3
"""
SmartCash class mapping configuration aligned with loss.json.

This module provides the centralized class mapping and configuration
for SmartCash's 17→7+1 class structure as specified in loss.json.
"""

# SmartCash class mapping aligned with loss.json
SMARTCASH_CLASS_CONFIG = {
    # Fine-grained classes (17 total)
    'fine_classes': {
        0: '1000_whole', 1: '2000_whole', 2: '5000_whole', 3: '10000_whole',
        4: '20000_whole', 5: '50000_whole', 6: '100000_whole',
        7: '1000_nominal_feature', 8: '2000_nominal_feature', 9: '5000_nominal_feature',
        10: '10000_nominal_feature', 11: '20000_nominal_feature', 12: '50000_nominal_feature',
        13: '100000_nominal_feature', 14: 'security_thread', 15: 'watermark', 16: 'special_sign'
    },
    # Mapping to 7 main + 1 feature classes as per loss.json
    'fine_to_main_mapping': {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,  # Main denominations (0-6)
        7: 0, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6,  # Nominal features map to main denominations
        14: 'feature', 15: 'feature', 16: 'feature'  # Authentication features map to feature class
    },
    # Layer-specific class ranges (aligned with prediction reorganization)
    'layer_ranges': {
        'layer_1': list(range(0, 7)),    # Main denominations (0-6)
        'layer_2': list(range(7, 14)),   # Nominal features (7-13)
        'layer_3': list(range(14, 17))   # Authentication features (14-16)
    }
}


def get_fine_class_name(class_id: int) -> str:
    """Get human-readable name for fine-grained class ID."""
    return SMARTCASH_CLASS_CONFIG['fine_classes'].get(class_id, f'class_{class_id}')


def get_main_class_id(fine_class_id: int):
    """Map fine-grained class ID to main class ID."""
    return SMARTCASH_CLASS_CONFIG['fine_to_main_mapping'].get(fine_class_id)


def get_layer_classes(layer_name: str) -> list:
    """Get class range for specific layer."""
    return SMARTCASH_CLASS_CONFIG['layer_ranges'].get(layer_name, [])


def is_denomination_class(class_id: int) -> bool:
    """Check if class ID is a denomination class (0-6 or 7-13)."""
    return 0 <= class_id <= 13


def is_feature_class(class_id: int) -> bool:
    """Check if class ID is an authentication feature class (14-16)."""
    return 14 <= class_id <= 16