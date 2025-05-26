"""
File: smartcash/dataset/augmentor/config.py
Deskripsi: Consolidated config operations menggantikan config.py dengan one-liner utilities
"""

from typing import Dict, Any, List
from .types import AugConfig, DEFAULT_AUGMENTATION_TYPES

# One-liner path resolvers yang sudah dikonsolidasi
from .utils.core import resolve_drive_path, get_best_data_location

# One-liner config extractors
get_raw_dir = lambda cfg: resolve_drive_path(cfg.get('data', {}).get('dir', 'data'))
get_aug_dir = lambda cfg: resolve_drive_path(cfg.get('augmentation', {}).get('output_dir', 'data/augmented'))
get_prep_dir = lambda cfg: resolve_drive_path(cfg.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))

# One-liner parameter extractors
get_num_variations = lambda cfg: cfg.get('augmentation', {}).get('num_variations', 2)
get_target_count = lambda cfg: cfg.get('augmentation', {}).get('target_count', 500)
get_augmentation_types = lambda cfg: cfg.get('augmentation', {}).get('types', DEFAULT_AUGMENTATION_TYPES)
get_intensity = lambda cfg: cfg.get('augmentation', {}).get('intensity', 0.7)

# One-liner validators
validate_config = lambda cfg: cfg and isinstance(cfg, dict)
validate_paths = lambda cfg: all([get_raw_dir(cfg), get_aug_dir(cfg), get_prep_dir(cfg)])

def create_aug_config(config: Dict[str, Any]) -> AugConfig:
    """Create AugConfig dengan consolidated path resolution."""
    if not validate_config(config):
        return AugConfig()
    
    return AugConfig(
        raw_dir=get_raw_dir(config),
        aug_dir=get_aug_dir(config),
        prep_dir=get_prep_dir(config),
        num_variations=max(1, get_num_variations(config)),
        target_count=max(1, get_target_count(config)),
        validate_results=False
    )

def extract_ui_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI dengan consolidated utilities."""
    config = ui_components.get('config', {})
    base_data_dir = get_best_data_location()
    
    # Extract augmentation types dari UI
    aug_options = ui_components.get('aug_options', {})
    if hasattr(aug_options, 'children') and aug_options.children:
        selected_types = getattr(aug_options.children[0], 'value', ['combined'])
    else:
        aug_types_widget = ui_components.get('augmentation_types')
        selected_types = getattr(aug_types_widget, 'value', ['combined']) if aug_types_widget else ['combined']
    
    return {
        'data': {'dir': base_data_dir},
        'augmentation': {
            'types': list(selected_types) if selected_types else ['combined'],
            'num_variations': config.get('num_variations', 2),
            'target_count': config.get('target_count', 500),
            'intensity': config.get('intensity', 0.7),
            'output_dir': resolve_drive_path('data/augmented')
        },
        'preprocessing': {
            'output_dir': resolve_drive_path('data/preprocessed')
        }
    }

# Auto-detect config dengan consolidated utilities
auto_detect_config = lambda: {
    'data': {'dir': get_best_data_location()},
    'augmentation': {
        'types': DEFAULT_AUGMENTATION_TYPES,
        'num_variations': 2,
        'target_count': 500,
        'intensity': 0.7,
        'output_dir': resolve_drive_path('data/augmented')
    },
    'preprocessing': {
        'output_dir': resolve_drive_path('data/preprocessed')
    }
}

# One-liner config mergers
merge_configs = lambda base, override: {**base, **{k: {**base.get(k, {}), **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v for k, v in override.items()}}
normalize_config = lambda cfg: {k: v for k, v in cfg.items() if v is not None}