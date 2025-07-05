"""
File: smartcash/ui/dataset/split/handlers/config_extractor.py
Deskripsi: Ekstraksi config dari UI components dengan centralized error handling
"""

from typing import Dict, Any, Tuple
import logging
from datetime import datetime

# Import error handling
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Import validation utilities
from smartcash.ui.dataset.split.handlers.defaults import normalize_split_ratios, validate_split_ratios

# Logger
logger = logging.getLogger(__name__)


@handle_ui_errors(log_error=True)
def extract_split_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI components dengan centralized error handling
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Dictionary konfigurasi split dataset
    """
    logger.debug("Mengekstrak konfigurasi split dari UI components")
    
    # Extract values dengan error handling
    config = _extract_config_values(ui_components)
    
    # Validate config
    is_valid, message = validate_extracted_values(config)
    if not is_valid:
        logger.warning(f"Konfigurasi tidak valid: {message}")
    else:
        logger.debug("Konfigurasi berhasil diekstrak dan valid")
    
    return config


@handle_ui_errors(log_error=True)
def _extract_config_values(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract values dari UI components dengan centralized error handling
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Dictionary konfigurasi split dataset
    """
    # Helper function untuk ekstraksi nilai dengan fallback
    def get_value(key, default):
        component = ui_components.get(key)
        if component is None:
            return default
        return getattr(component, 'value', default)
    
    # Extract ratio values
    train_ratio, valid_ratio, test_ratio = extract_ratios(ui_components)
    
    # Validate dan normalize ratio
    valid, message = validate_split_ratios(train_ratio, valid_ratio, test_ratio)
    if not valid:
        logger.warning(f"Split ratio tidak valid: {message}, melakukan normalisasi")
        train_ratio, valid_ratio, test_ratio = normalize_split_ratios(train_ratio, valid_ratio, test_ratio)
    
    # Metadata untuk config yang diperbarui
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Construct config sesuai dengan dataset_config.yaml
    return {
        # Inherit dari base_config.yaml
        '_base_': 'base_config.yaml',
        
        # Override konfigurasi data dari base_config
        'data': {
            'dir': 'data',                    # Direktori utama data (path relatif)
            'preprocessed_dir': 'data/preprocessed',  # Direktori untuk hasil preprocessed
            'split_ratios': {
                'train': train_ratio, 
                'valid': valid_ratio, 
                'test': test_ratio
            },
            'stratified_split': get_value('stratified_checkbox', True),
            'random_seed': get_value('random_seed', 42),
            'source': 'roboflow',             # Sumber dataset ('roboflow', 'local')
            'roboflow': {                      # Konfigurasi Roboflow
                'api_key': '',                  # Diisi oleh user atau dari Google Secret
                'workspace': 'smartcash-wo2us',
                'project': 'rupiah-emisi-2022',
                'version': '3',
                'output_format': 'yolov5pytorch'
            },
            'validation': {
                'enabled': get_value('validation_enabled', True),
                'fix_issues': get_value('fix_issues', True),
                'move_invalid': get_value('move_invalid', True),
                'invalid_dir': get_value('invalid_dir', 'data/invalid'),
                'visualize_issues': get_value('visualize_issues', False)
            }
        },
        
        # Konfigurasi untuk akses dan backup dataset
        'dataset': {
            'backup': {
                'enabled': get_value('backup_checkbox', True),
                'dir': get_value('backup_dir', 'data/backup/dataset'),
                'count': get_value('backup_count', 2),
                'auto': get_value('auto_backup', False)
            },
            'export': {
                'enabled': True,
                'formats': ['yolo', 'coco'],
                'dir': 'data/exported'
            },
            'import': {
                'allowed_formats': ['yolo', 'coco'],
                'temp_dir': 'data/temp'
            }
        },
        
        # Override konfigurasi cache dari base_config
        'cache': {
            'dir': '.cache/smartcash/dataset'  # Override dari base_config (.cache/smartcash)
        },
        
        # Split settings untuk backward compatibility
        'split_settings': {
            'backup_before_split': get_value('backup_checkbox', True),
            'backup_dir': get_value('backup_dir', 'data/splits_backup'),
            'dataset_path': 'data',
            'preprocessed_path': 'data/preprocessed'
        },
        
        # Metadata
        'config_version': '1.0',
        'updated_at': current_time
    }


@handle_ui_errors(log_error=True)
def extract_ratios(ui_components: Dict[str, Any]) -> Tuple[float, float, float]:
    """Extract split ratios dari UI components
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Tuple berisi (train_ratio, valid_ratio, test_ratio)
    """
    defaults = {'train': 0.7, 'valid': 0.15, 'test': 0.15}
    result = []
    
    for name, default in defaults.items():
        slider = ui_components.get(f'{name}_slider')
        if slider is None:
            logger.warning(f"{name}_slider tidak ditemukan, menggunakan nilai default {default}")
            result.append(default)
        else:
            result.append(getattr(slider, 'value', default))
    
    return tuple(result)


@handle_ui_errors(log_error=True)
def validate_extracted_values(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate extracted config values
    
    Args:
        config: Dictionary konfigurasi split dataset
        
    Returns:
        Tuple berisi (is_valid, message)
    """
    # Check config structure
    if 'data' not in config:
        return False, "Struktur config tidak valid: 'data' tidak ditemukan"
    
    if 'split_ratios' not in config['data']:
        return False, "Struktur config tidak valid: 'split_ratios' tidak ditemukan"
    
    # Get split ratios
    ratios = config['data']['split_ratios']
    if not all(k in ratios for k in ['train', 'valid', 'test']):
        return False, "Split ratios tidak lengkap"
    
    # Validate ratios
    return validate_split_ratios(ratios['train'], ratios['valid'], ratios['test'])
