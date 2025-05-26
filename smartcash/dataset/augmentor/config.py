"""
File: smartcash/dataset/augmentor/config.py
Deskripsi: Config validators dan extractors dengan one-liners untuk augmentasi
"""

from typing import Dict, Any, List
from .types import AugConfig, DEFAULT_AUGMENTATION_TYPES

# One-liner path extractors
get_raw_dir = lambda cfg: cfg.get('data', {}).get('dir', 'data')
get_aug_dir = lambda cfg: cfg.get('augmentation', {}).get('output_dir', 'data/augmented')
get_prep_dir = lambda cfg: cfg.get('preprocessing', {}).get('output_dir', 'data/preprocessed')

# One-liner augmentation parameter extractors
get_num_variations = lambda cfg: cfg.get('augmentation', {}).get('num_variations', 2)
get_target_count = lambda cfg: cfg.get('augmentation', {}).get('target_count', 500)
get_output_prefix = lambda cfg: cfg.get('augmentation', {}).get('output_prefix', 'aug')
get_process_bboxes = lambda cfg: cfg.get('augmentation', {}).get('process_bboxes', True)
get_balance_classes = lambda cfg: cfg.get('augmentation', {}).get('balance_classes', False)

# One-liner type extractors
get_augmentation_types = lambda cfg: cfg.get('augmentation', {}).get('types', DEFAULT_AUGMENTATION_TYPES)
get_intensity = lambda cfg: cfg.get('augmentation', {}).get('intensity', 1.0)
get_num_workers = lambda cfg: cfg.get('augmentation', {}).get('num_workers', 1)

# One-liner validators
validate_config = lambda cfg: cfg and isinstance(cfg, dict)
validate_paths = lambda cfg: all(get_raw_dir(cfg), get_aug_dir(cfg), get_prep_dir(cfg))
validate_num_variations = lambda n: isinstance(n, int) and n > 0
validate_target_count = lambda t: isinstance(t, int) and t > 0
validate_aug_types = lambda types: isinstance(types, list) and len(types) > 0

def create_aug_config(config: Dict[str, Any]) -> AugConfig:
    """
    Buat AugConfig dari dictionary konfigurasi dengan validasi.
    
    Args:
        config: Dictionary konfigurasi aplikasi
        
    Returns:
        AugConfig object yang tervalidasi
    """
    if not validate_config(config):
        return AugConfig()  # Return default config
    
    return AugConfig(
        raw_dir=get_raw_dir(config),
        aug_dir=get_aug_dir(config),
        prep_dir=get_prep_dir(config),
        num_variations=max(1, get_num_variations(config)),
        target_count=max(1, get_target_count(config)),
        output_prefix=get_output_prefix(config),
        process_bboxes=get_process_bboxes(config),
        validate_results=False  # Always False untuk memastikan gambar dihasilkan
    )

def extract_ui_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract konfigurasi dari UI components dengan one-liners.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang diekstrak
    """
    # Base extraction dengan fallback
    config = ui_components.get('config', {})
    
    # Extract augmentation parameters dari UI
    aug_options = ui_components.get('aug_options', {})
    if hasattr(aug_options, 'children') and len(aug_options.children) > 0:
        selector = aug_options.children[0]
        selected_types = getattr(selector, 'value', ['combined'])
    else:
        selected_types = ['combined']
    
    # Build config dengan one-liners
    extracted_config = {
        'data': {'dir': config.get('data_dir', 'data')},
        'augmentation': {
            'types': list(selected_types) if selected_types else ['combined'],
            'num_variations': config.get('num_variations', 2),
            'target_count': config.get('target_count', 500),
            'output_prefix': config.get('output_prefix', 'aug'),
            'process_bboxes': config.get('process_bboxes', True),
            'balance_classes': config.get('balance_classes', False),
            'intensity': config.get('intensity', 1.0),
            'num_workers': 1,  # Always 1 for Colab compatibility
            'output_dir': config.get('augmented_dir', 'data/augmented')
        },
        'preprocessing': {
            'output_dir': config.get('preprocessed_dir', 'data/preprocessed')
        }
    }
    
    return extracted_config

def get_safe_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Safely extract nested config value dengan dot notation.
    
    Args:
        config: Dictionary konfigurasi
        key_path: Path dengan dot notation (e.g., 'augmentation.num_variations')
        default: Nilai default jika key tidak ditemukan
        
    Returns:
        Nilai konfigurasi atau default
    """
    try:
        keys = key_path.split('.')
        value = config
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

# One-liner config mergers
merge_configs = lambda base, override: {**base, **{k: {**base.get(k, {}), **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v for k, v in override.items()}}
normalize_config = lambda cfg: {k: v for k, v in cfg.items() if v is not None}

# Predefined config templates
DEFAULT_CONFIG = {
    'data': {'dir': 'data'},
    'augmentation': {
        'types': DEFAULT_AUGMENTATION_TYPES,
        'num_variations': 2,
        'target_count': 500,
        'output_prefix': 'aug',
        'process_bboxes': True,
        'balance_classes': False,
        'intensity': 1.0,
        'num_workers': 1,
        'output_dir': 'data/augmented'
    },
    'preprocessing': {
        'output_dir': 'data/preprocessed'
    }
}

COLAB_CONFIG = merge_configs(DEFAULT_CONFIG, {
    'augmentation': {'num_workers': 2, 'validate_results': False}
})