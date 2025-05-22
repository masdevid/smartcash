"""
File: smartcash/ui/dataset/split/handlers/defaults.py
Deskripsi: Handler untuk nilai default konfigurasi split dataset
"""

from typing import Dict, Any


def get_default_split_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk split dataset.
    
    Returns:
        Dictionary konfigurasi default yang sesuai dengan base_config.yaml
    """
    return {
        'data': {
            'split_ratios': {
                'train': 0.7,
                'valid': 0.15,
                'test': 0.15
            },
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


def get_default_ui_values() -> Dict[str, Any]:
    """
    Dapatkan nilai default untuk komponen UI.
    
    Returns:
        Dictionary nilai default UI
    """
    return {
        'train_ratio': 0.7,
        'valid_ratio': 0.15,
        'test_ratio': 0.15,
        'stratified_split': True,
        'random_seed': 42,
        'backup_before_split': True,
        'backup_dir': 'data/splits_backup',
        'dataset_path': 'data',
        'preprocessed_path': 'data/preprocessed'
    }


def validate_split_ratios(train: float, valid: float, test: float) -> tuple[bool, str]:
    """
    Validasi rasio split untuk memastikan total = 1.0.
    
    Args:
        train: Rasio training
        valid: Rasio validation  
        test: Rasio testing
        
    Returns:
        Tuple (is_valid, message)
    """
    total = round(train + valid + test, 2)
    
    if total != 1.0:
        return False, f"Total rasio harus 1.0, saat ini: {total}"
    
    if any(ratio <= 0 for ratio in [train, valid, test]):
        return False, "Semua rasio harus lebih besar dari 0"
    
    if any(ratio >= 1.0 for ratio in [train, valid, test]):
        return False, "Tidak ada rasio yang boleh 1.0 atau lebih"
    
    return True, "Rasio split valid"


def normalize_split_ratios(train: float, valid: float, test: float) -> tuple[float, float, float]:
    """
    Normalisasi rasio split agar total = 1.0.
    
    Args:
        train: Rasio training
        valid: Rasio validation
        test: Rasio testing
        
    Returns:
        Tuple rasio yang dinormalisasi (train, valid, test)
    """
    total = train + valid + test
    if total == 0:
        return 0.7, 0.15, 0.15  # Default fallback
    
    factor = 1.0 / total
    return (
        round(train * factor, 2),
        round(valid * factor, 2), 
        round(test * factor, 2)
    )