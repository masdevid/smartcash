"""
File: smartcash/ui/dataset/split/handlers/defaults.py
Deskripsi: Default values dan validation untuk split dataset config
"""

from typing import Dict, Any, Tuple


def get_default_split_config() -> Dict[str, Any]:
    """Default split config dengan struktur yang konsisten"""
    return {
        'data': {
            'split_ratios': {'train': 0.7, 'valid': 0.15, 'test': 0.15},
            'stratified_split': True,
            'random_seed': 42
        },
        'split_settings': {
            'backup_before_split': True,
            'backup_dir': 'data/splits_backup',
            'dataset_path': 'data',
            'preprocessed_path': 'data/preprocessed'
        }
    }


def validate_split_ratios(train: float, valid: float, test: float) -> Tuple[bool, str]:
    """Validate split ratios dengan one-liner checks"""
    total = round(train + valid + test, 2)
    return (total == 1.0 and all(0 < r < 1 for r in [train, valid, test]), 
            f"Total harus 1.0 (saat ini: {total})" if total != 1.0 else "Invalid ratios" if not all(0 < r < 1 for r in [train, valid, test]) else "Valid")


def normalize_split_ratios(train: float, valid: float, test: float) -> Tuple[float, float, float]:
    """Normalize ratios dengan one-liner calculation"""
    total = train + valid + test
    factor = 1.0 / total if total > 0 else 1.0
    return tuple(round(r * factor, 2) for r in [train, valid, test]) if total > 0 else (0.7, 0.15, 0.15)


# One-liner utilities
get_default_ui_values = lambda: {'train_ratio': 0.7, 'valid_ratio': 0.15, 'test_ratio': 0.15, 'stratified_split': True, 'random_seed': 42, 'backup_before_split': True}
is_valid_config = lambda config: 'data' in config and 'split_ratios' in config['data'] and 'split_settings' in config