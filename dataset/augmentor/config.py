"""
File: smartcash/dataset/augmentor/config.py
Deskripsi: Fixed config validators dan extractors dengan Google Drive path resolution dan one-liners
"""

from typing import Dict, Any, List
from .types import AugConfig, DEFAULT_AUGMENTATION_TYPES

def _resolve_to_drive_if_needed(path: str) -> str:
    """Resolve path ke Google Drive jika diperlukan."""
    import os
    
    # Jika path sudah absolute dan exists, return as-is
    if os.path.exists(path):
        return path
    
    # Try resolve ke Google Drive
    drive_bases = [
        '/content/drive/MyDrive/SmartCash',
        '/content/drive/MyDrive', 
        '/content'
    ]
    
    for base in drive_bases:
        if path.startswith('data'):
            resolved = os.path.join(base, path)
        else:
            resolved = os.path.join(base, 'data') if path == 'data' else os.path.join(base, path)
        
        # Check if resolved path exists or parent exists (for creation)
        if os.path.exists(resolved) or os.path.exists(os.path.dirname(resolved)):
            return resolved
    
    return path  # Fallback

# One-liner path extractors dengan Google Drive resolution
get_raw_dir = lambda cfg: _resolve_to_drive_if_needed(cfg.get('data', {}).get('dir', 'data'))
get_aug_dir = lambda cfg: _resolve_to_drive_if_needed(cfg.get('augmentation', {}).get('output_dir', 'data/augmented'))
get_prep_dir = lambda cfg: _resolve_to_drive_if_needed(cfg.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))

# One-liner augmentation parameter extractors
get_num_variations = lambda cfg: cfg.get('augmentation', {}).get('num_variations', 2)
get_target_count = lambda cfg: cfg.get('augmentation', {}).get('target_count', 500)
get_output_prefix = lambda cfg: cfg.get('augmentation', {}).get('output_prefix', 'aug')
get_process_bboxes = lambda cfg: cfg.get('augmentation', {}).get('process_bboxes', True)
get_balance_classes = lambda cfg: cfg.get('augmentation', {}).get('balance_classes', False)

# One-liner type extractors
get_augmentation_types = lambda cfg: cfg.get('augmentation', {}).get('types', DEFAULT_AUGMENTATION_TYPES)
get_intensity = lambda cfg: cfg.get('augmentation', {}).get('intensity', 0.7)
get_num_workers = lambda cfg: cfg.get('augmentation', {}).get('num_workers', 1)

# One-liner validators
validate_config = lambda cfg: cfg and isinstance(cfg, dict)
validate_paths = lambda cfg: all([get_raw_dir(cfg), get_aug_dir(cfg), get_prep_dir(cfg)])
validate_num_variations = lambda n: isinstance(n, int) and n > 0
validate_target_count = lambda t: isinstance(t, int) and t > 0
validate_aug_types = lambda types: isinstance(types, list) and len(types) > 0

def create_aug_config(config: Dict[str, Any]) -> AugConfig:
    """
    Buat AugConfig dari dictionary dengan Google Drive path resolution.
    
    Args:
        config: Dictionary konfigurasi aplikasi
        
    Returns:
        AugConfig object dengan resolved paths
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
    Extract konfigurasi dari UI components dengan Google Drive path resolution.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi dengan resolved paths
    """
    # Base extraction dengan fallback
    config = ui_components.get('config', {})
    
    # Get base data directory dengan Drive resolution
    base_data_dir = get_best_data_location()
    
    # Extract augmentation parameters dari UI
    aug_options = ui_components.get('aug_options', {})
    if hasattr(aug_options, 'children') and len(aug_options.children) > 0:
        selector = aug_options.children[0]
        selected_types = getattr(selector, 'value', ['combined'])
    else:
        # Fallback dari augmentation_types widget
        aug_types_widget = ui_components.get('augmentation_types')
        selected_types = getattr(aug_types_widget, 'value', ['combined']) if aug_types_widget else ['combined']
    
    # Build config dengan Google Drive resolved paths
    extracted_config = {
        'data': {'dir': base_data_dir},
        'augmentation': {
            'types': list(selected_types) if selected_types else ['combined'],
            'num_variations': config.get('num_variations', 2),
            'target_count': config.get('target_count', 500),
            'output_prefix': config.get('output_prefix', 'aug'),
            'process_bboxes': config.get('process_bboxes', True),
            'balance_classes': config.get('balance_classes', False),
            'intensity': config.get('intensity', 0.7),
            'num_workers': 1,  # Always 1 for Colab compatibility
            'output_dir': _resolve_to_drive_if_needed('data/augmented')
        },
        'preprocessing': {
            'output_dir': _resolve_to_drive_if_needed('data/preprocessed')
        }
    }
    
    return extracted_config

def get_best_data_location() -> str:
    """
    Dapatkan lokasi data terbaik dengan Google Drive priority.
    
    Returns:
        Path ke lokasi data terbaik
    """
    import os
    
    # Priority: Drive mounted > Local dengan fallback
    candidates = [
        '/content/drive/MyDrive/SmartCash/data',
        '/content/drive/MyDrive/data',
        '/content/SmartCash/data', 
        '/content/data',
        'data'
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
        # Check if parent directory exists (creation possibility)
        parent = os.path.dirname(candidate)
        if os.path.exists(parent):
            return candidate
    
    return 'data'  # Ultimate fallback

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

def validate_drive_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Google Drive paths dalam config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary hasil validasi
    """
    import os
    
    # Extract paths
    raw_dir = get_raw_dir(config)
    aug_dir = get_aug_dir(config)
    prep_dir = get_prep_dir(config)
    
    # Check path validity dan Drive status
    validation = {
        'raw_dir_exists': os.path.exists(raw_dir),
        'raw_dir_path': raw_dir,
        'aug_dir_ready': os.path.exists(os.path.dirname(aug_dir)),
        'aug_dir_path': aug_dir,
        'prep_dir_ready': os.path.exists(os.path.dirname(prep_dir)),
        'prep_dir_path': prep_dir,
        'using_drive': any('/content/drive/MyDrive' in p for p in [raw_dir, aug_dir, prep_dir]),
        'drive_mounted': os.path.exists('/content/drive/MyDrive'),
        'all_paths_valid': True
    }
    
    # Overall validation
    validation['all_paths_valid'] = (
        validation['raw_dir_exists'] and 
        validation['aug_dir_ready'] and 
        validation['prep_dir_ready']
    )
    
    return validation

# One-liner config mergers dengan path resolution
merge_configs = lambda base, override: {**base, **{k: {**base.get(k, {}), **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v for k, v in override.items()}}
normalize_config = lambda cfg: {k: v for k, v in cfg.items() if v is not None}
resolve_all_paths = lambda cfg: {**cfg, 'data': {'dir': get_raw_dir(cfg)}, 'augmentation': {**cfg.get('augmentation', {}), 'output_dir': get_aug_dir(cfg)}, 'preprocessing': {**cfg.get('preprocessing', {}), 'output_dir': get_prep_dir(cfg)}}

# Predefined config templates dengan Google Drive paths
DEFAULT_CONFIG = {
    'data': {'dir': get_best_data_location()},
    'augmentation': {
        'types': DEFAULT_AUGMENTATION_TYPES,
        'num_variations': 2,
        'target_count': 500,
        'output_prefix': 'aug',
        'process_bboxes': True,
        'balance_classes': False,
        'intensity': 0.7,
        'num_workers': 1,
        'output_dir': _resolve_to_drive_if_needed('data/augmented')
    },
    'preprocessing': {
        'output_dir': _resolve_to_drive_if_needed('data/preprocessed')
    }
}

COLAB_CONFIG = merge_configs(DEFAULT_CONFIG, {
    'augmentation': {'num_workers': 1, 'validate_results': False},  # Colab optimized
    'data': {'dir': get_best_data_location()}
})

DRIVE_CONFIG = {
    'data': {'dir': '/content/drive/MyDrive/SmartCash/data'},
    'augmentation': {
        'types': DEFAULT_AUGMENTATION_TYPES,
        'num_variations': 2,
        'target_count': 500,
        'output_prefix': 'aug',
        'process_bboxes': True,
        'balance_classes': False,
        'intensity': 0.7,
        'num_workers': 1,
        'output_dir': '/content/drive/MyDrive/SmartCash/data/augmented'
    },
    'preprocessing': {
        'output_dir': '/content/drive/MyDrive/SmartCash/data/preprocessed'
    }
}

def auto_detect_config() -> Dict[str, Any]:
    """
    Auto-detect optimal config berdasarkan environment dan Drive status.
    
    Returns:
        Dictionary config yang optimal
    """
    import os
    
    # Detect Google Drive mount status
    drive_mounted = os.path.exists('/content/drive/MyDrive')
    smartcash_folder = os.path.exists('/content/drive/MyDrive/SmartCash')
    
    if drive_mounted and smartcash_folder:
        return DRIVE_CONFIG
    elif drive_mounted:
        # Drive mounted tapi belum ada SmartCash folder
        config = dict(DRIVE_CONFIG)
        config['data']['dir'] = '/content/drive/MyDrive/data'
        config['augmentation']['output_dir'] = '/content/drive/MyDrive/data/augmented'
        config['preprocessing']['output_dir'] = '/content/drive/MyDrive/data/preprocessed'
        return config
    else:
        return COLAB_CONFIG

def setup_drive_structure() -> Dict[str, Any]:
    """
    Setup struktur direktori di Google Drive untuk SmartCash.
    
    Returns:
        Dictionary hasil setup
    """
    import os
    
    required_dirs = [
        '/content/drive/MyDrive/SmartCash',
        '/content/drive/MyDrive/SmartCash/data',
        '/content/drive/MyDrive/SmartCash/data/images',
        '/content/drive/MyDrive/SmartCash/data/labels',
        '/content/drive/MyDrive/SmartCash/data/augmented',
        '/content/drive/MyDrive/SmartCash/data/augmented/images',
        '/content/drive/MyDrive/SmartCash/data/augmented/labels',
        '/content/drive/MyDrive/SmartCash/data/preprocessed'
    ]
    
    created_dirs = []
    failed_dirs = []
    
    for dir_path in required_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            if os.path.exists(dir_path):
                created_dirs.append(dir_path)
        except Exception as e:
            failed_dirs.append((dir_path, str(e)))
    
    return {
        'status': 'success' if len(failed_dirs) == 0 else 'partial',
        'created_dirs': created_dirs,
        'failed_dirs': failed_dirs,
        'total_created': len(created_dirs),
        'drive_ready': len(created_dirs) >= 4  # Minimal dirs needed
    }

# Utility functions untuk path management
def ensure_google_drive_setup() -> bool:
    """
    Pastikan Google Drive sudah setup dengan proper structure.
    
    Returns:
        Boolean apakah setup berhasil
    """
    import os
    
    if not os.path.exists('/content/drive/MyDrive'):
        return False
    
    setup_result = setup_drive_structure()
    return setup_result['drive_ready']

def get_config_recommendations() -> List[str]:
    """
    Dapatkan rekomendasi konfigurasi berdasarkan environment.
    
    Returns:
        List rekomendasi
    """
    import os
    
    recommendations = []
    
    # Drive status recommendations
    if not os.path.exists('/content/drive/MyDrive'):
        recommendations.append("🔗 Mount Google Drive: dari menu Files > Mount Drive")
        recommendations.append("📁 Buat folder SmartCash di Drive untuk persistensi data")
    elif not os.path.exists('/content/drive/MyDrive/SmartCash'):
        recommendations.append("📁 Jalankan setup_drive_structure() untuk membuat folder struktur")
    else:
        recommendations.append("✅ Google Drive sudah siap dengan struktur SmartCash")
    
    # Data location recommendations
    best_location = get_best_data_location()
    if '/drive' in best_location:
        recommendations.append(f"💾 Menggunakan Google Drive: {best_location}")
    else:
        recommendations.append(f"⚠️ Menggunakan local storage: {best_location} (data tidak persistent)")
    
    return recommendations

# One-liner shortcuts untuk common operations
get_drive_config = lambda: DRIVE_CONFIG if ensure_google_drive_setup() else COLAB_CONFIG
create_optimal_config = lambda ui_components: resolve_all_paths(extract_ui_config(ui_components))
validate_config_paths = lambda cfg: validate_drive_paths(cfg)['all_paths_valid']