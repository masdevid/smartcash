"""
File: smartcash/dataset/utils/denomination_utils.py
Deskripsi: Utilitas untuk penanganan dan mapping denominasi mata uang Rupiah
"""

from typing import Dict, Any, Optional, Set, Tuple
import re
from pathlib import Path

# Mapping denominasi mata uang Rupiah berdasarkan class id
DENOMINATION_CLASS_MAP = {
    '0': '1k',    # 1.000 Rupiah
    '1': '2k',    # 2.000 Rupiah
    '2': '5k',    # 5.000 Rupiah
    '3': '10k',   # 10.000 Rupiah
    '4': '20k',   # 20.000 Rupiah
    '5': '50k',   # 50.000 Rupiah
    '6': '100k',  # 100.000 Rupiah
    '7': '1k',    # 1.000 Rupiah (variant)
    '8': '2k',    # 2.000 Rupiah (variant)
    '9': '5k',    # 5.000 Rupiah (variant)
    '10': '10k',  # 10.000 Rupiah (variant)
    '11': '20k',  # 20.000 Rupiah (variant)
    '12': '50k',  # 50.000 Rupiah (variant)
    '13': '100k', # 100.000 Rupiah (variant)
}

def get_denomination_label(class_id: str) -> str:
    """
    Dapatkan label denominasi dari class ID.
    
    Args:
        class_id: ID kelas yang akan dicari
        
    Returns:
        Label denominasi atau 'unknown' jika tidak ditemukan
    """
    return DENOMINATION_CLASS_MAP.get(class_id, 'unknown')

def extract_info_from_filename(filename: str) -> Dict[str, Any]:
    """
    Ekstrak informasi dari nama file dengan format denominasi.
    
    Format ekspektasi:
    - Original: rp_100k_uuid.jpg
    - Augmented: aug_rp_100k_uuid_var_1.jpg
    
    Args:
        filename: Nama file yang akan diekstrak info-nya
        
    Returns:
        Dictionary berisi informasi yang diekstrak
    """
    info = {'is_valid': False}
    
    # Pattern untuk augmented
    aug_pattern = r'aug_(?P<prefix>\w+)_(?P<denomination>\w+)_(?P<uuid>[^_]+)_var_(?P<variation>\d+)'
    # Pattern untuk original
    orig_pattern = r'(?P<prefix>\w+)_(?P<denomination>\w+)_(?P<uuid>[^_\.]+)'
    
    # Coba match dengan pattern augmented
    aug_match = re.match(aug_pattern, filename)
    if aug_match:
        info.update({
            'is_valid': True,
            'is_augmented': True,
            'prefix': aug_match.group('prefix'),
            'denomination': aug_match.group('denomination'),
            'uuid': aug_match.group('uuid'),
            'variation': int(aug_match.group('variation'))
        })
        return info
    
    # Coba match dengan pattern original
    orig_match = re.match(orig_pattern, filename)
    if orig_match:
        info.update({
            'is_valid': True,
            'is_augmented': False,
            'prefix': orig_match.group('prefix'),
            'denomination': orig_match.group('denomination'),
            'uuid': orig_match.group('uuid')
        })
        return info
    
    return info

def generate_denomination_filename(class_id: str, uuid: str, prefix: str = 'rp', variation: Optional[int] = None) -> str:
    """
    Generate nama file dengan format denominasi.
    
    Args:
        class_id: ID kelas untuk denominasi
        uuid: Identifier unik untuk file
        prefix: Prefix untuk nama file
        variation: Nomor variasi (untuk file augmentasi)
        
    Returns:
        Nama file dengan format denominasi
    """
    denomination = get_denomination_label(class_id)
    
    # Format: rp_100k_uuid.jpg or aug_rp_100k_uuid_var_1.jpg
    if variation is not None:
        return f"aug_{prefix}_{denomination}_{uuid}_var_{variation}.jpg"
    else:
        return f"{prefix}_{denomination}_{uuid}.jpg"