"""
File: smartcash/ui/dataset/augment/configs/augment_defaults.py
Description: Default configuration values for augment module

This module provides the default configuration structure that preserves
all business logic from the original augmentation module.
"""

from typing import Dict, Any
from ..constants import (
    DEFAULT_AUGMENTATION_PARAMS, DEFAULT_POSITION_PARAMS, 
    DEFAULT_LIGHTING_PARAMS, DEFAULT_CLEANUP_PARAMS,
    BACKEND_CONFIG, FILE_PROCESSING_CONFIG, PERFORMANCE_CONFIG,
    CLASS_WEIGHTS, BANKNOTE_CLASSES
)

def get_default_augment_config() -> Dict[str, Any]:
    """
    Get default augment configuration preserving all business logic.
    
    Returns:
        Dictionary containing complete default configuration
    """
    return {
        # Data paths - backend essentials
        'data': {
            'dir': 'data'
        },
        
        # Form fields mapping (preserved from original)
        'augmentation': {
            # Basic form fields
            'num_variations': DEFAULT_AUGMENTATION_PARAMS['num_variations'],
            'target_count': DEFAULT_AUGMENTATION_PARAMS['target_count'],
            'intensity': DEFAULT_AUGMENTATION_PARAMS['intensity'],
            'balance_classes': DEFAULT_AUGMENTATION_PARAMS['balance_classes'],
            'target_split': DEFAULT_AUGMENTATION_PARAMS['target_split'],
            'types': DEFAULT_AUGMENTATION_PARAMS['types'],
            
            # Advanced form fields (position)
            'position': DEFAULT_POSITION_PARAMS.copy(),
            
            # Advanced form fields (lighting) - UPDATED: Added HSV parameters
            'lighting': DEFAULT_LIGHTING_PARAMS.copy(),
            
            # Combined params (sync dengan position + lighting) - UPDATED: Added HSV
            'combined': {
                **DEFAULT_POSITION_PARAMS,
                **DEFAULT_LIGHTING_PARAMS
            }
        },
        
        # UPDATED: Cleanup configuration menggantikan preprocessing.normalization
        'cleanup': {
            **DEFAULT_CLEANUP_PARAMS,
            
            # Target-specific settings
            'targets': {
                'augmented': {
                    'include_preprocessed': True,
                    'patterns': ['aug_*']
                },
                'samples': {
                    'patterns': ['sample_aug_*'],
                    'preserve_originals': True
                },
                'both': {
                    'sequential': True
                }
            }
        },
        
        # Backend structure yang diharapkan service
        'backend': BACKEND_CONFIG.copy(),
        
        # Backend essentials only
        'balancing': {
            'enabled': True,
            'layer_weights': CLASS_WEIGHTS.copy(),
            'banknote_classes': BANKNOTE_CLASSES.copy()
        },
        
        'file_processing': FILE_PROCESSING_CONFIG.copy(),
        
        'performance': PERFORMANCE_CONFIG.copy()
    }

def get_minimal_augment_config() -> Dict[str, Any]:
    """
    Get minimal configuration for testing purposes.
    
    Returns:
        Dictionary containing minimal working configuration
    """
    return {
        'data': {'dir': 'data'},
        'augmentation': {
            'num_variations': 1,
            'target_count': 100,
            'intensity': 0.5,
            'balance_classes': False,
            'target_split': 'train',
            'types': ['combined']
        }
    }

def get_augment_config_schema() -> Dict[str, Any]:
    """
    Get configuration schema for validation.
    
    Returns:
        Dictionary containing configuration schema
    """
    return {
        'type': 'object',
        'properties': {
            'data': {
                'type': 'object',
                'properties': {
                    'dir': {'type': 'string'}
                },
                'required': ['dir']
            },
            'augmentation': {
                'type': 'object',
                'properties': {
                    'num_variations': {'type': 'integer', 'minimum': 1, 'maximum': 10},
                    'target_count': {'type': 'integer', 'minimum': 10, 'maximum': 10000},
                    'intensity': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
                    'balance_classes': {'type': 'boolean'},
                    'target_split': {'type': 'string', 'enum': ['train', 'valid', 'test', 'all']},
                    'types': {'type': 'array', 'items': {'type': 'string'}}
                },
                'required': ['num_variations', 'target_count', 'intensity', 'target_split']
            }
        },
        'required': ['data', 'augmentation']
    }