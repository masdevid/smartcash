"""
File: smartcash/ui/dataset/split/handlers/config_extractor.py
Deskripsi: Ekstraksi config dari UI components - refactored dengan one-liner approach
"""

from typing import Dict, Any
from smartcash.ui.dataset.split.handlers.defaults import normalize_split_ratios, validate_split_ratios


def extract_split_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI components dengan one-liner extraction"""
    # One-liner value extraction dengan fallback
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    # Extract ratio values
    train_ratio, valid_ratio, test_ratio = (
        get_value('train_slider', 0.7),
        get_value('valid_slider', 0.15), 
        get_value('test_slider', 0.15)
    )
    
    # Validate dan normalize ratio dengan one-liner
    train_ratio, valid_ratio, test_ratio = normalize_split_ratios(train_ratio, valid_ratio, test_ratio) if not validate_split_ratios(train_ratio, valid_ratio, test_ratio)[0] else (train_ratio, valid_ratio, test_ratio)
    
    # Metadata untuk config yang diperbarui
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # One-liner config construction sesuai dengan dataset_config.yaml
    return {
        'config_version': '1.0',
        'updated_at': current_time,
        'data': {
            'split_ratios': {
                'train': train_ratio, 
                'valid': valid_ratio, 
                'test': test_ratio
            },
            'stratified_split': get_value('stratified_checkbox', True),
            'random_seed': get_value('random_seed', 42),
            'validation': {
                'enabled': get_value('validation_enabled', True),
                'fix_issues': get_value('fix_issues', True),
                'move_invalid': get_value('move_invalid', True),
                'invalid_dir': get_value('invalid_dir', 'data/invalid'),
                'visualize_issues': get_value('visualize_issues', False)
            }
        },
        'dataset': {
            'backup': {
                'enabled': get_value('backup_checkbox', True),
                'dir': get_value('backup_dir', 'data/backup/dataset'),
                'count': get_value('backup_count', 3),
                'auto': get_value('auto_backup', False)
            }
        }
    }


# One-liner extraction utilities
extract_ratios = lambda ui_components: tuple(getattr(ui_components.get(f'{t}_slider', type('', (), {'value': d})()), 'value', d) for t, d in [('train', 0.7), ('valid', 0.15), ('test', 0.15)])
validate_extracted_values = lambda config: validate_split_ratios(*config['data']['split_ratios'].values()) if 'data' in config and 'split_ratios' in config['data'] else (False, "Invalid config structure")