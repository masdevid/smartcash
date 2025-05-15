"""
File: smartcash/ui/dataset/augmentation/handlers/config_validator.py
Deskripsi: Validator untuk konfigurasi augmentasi dataset
"""

from typing import Dict, Any, Optional, List, Union
from smartcash.common.logger import get_logger

logger = get_logger("augmentation_validator")

def validate_augmentation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validasi konfigurasi augmentasi dan pastikan semua parameter yang diperlukan tersedia.
    
    Args:
        config: Konfigurasi augmentasi yang akan divalidasi
        
    Returns:
        Konfigurasi augmentasi yang sudah divalidasi
    """
    if not config:
        logger.warning("⚠️ Konfigurasi kosong, menggunakan default")
        config = {}
    
    # Pastikan struktur dasar tersedia
    if 'augmentation' not in config:
        config['augmentation'] = {}
    
    aug_config = config['augmentation']
    
    # Validasi parameter dasar
    aug_config['enabled'] = _validate_param(aug_config.get('enabled'), True, bool)
    aug_config['num_variations'] = _validate_param(aug_config.get('num_variations'), 2, int, range_=(1, 10))
    aug_config['output_prefix'] = _validate_param(aug_config.get('output_prefix'), 'aug', str)
    aug_config['process_bboxes'] = _validate_param(aug_config.get('process_bboxes'), True, bool)
    aug_config['validate_results'] = _validate_param(aug_config.get('validate_results'), True, bool)
    aug_config['resume'] = _validate_param(aug_config.get('resume'), False, bool)
    aug_config['num_workers'] = _validate_param(aug_config.get('num_workers'), 4, int, range_=(1, 16))
    aug_config['balance_classes'] = _validate_param(aug_config.get('balance_classes'), True, bool)
    aug_config['target_count'] = _validate_param(aug_config.get('target_count'), 1000, int, range_=(100, 10000))
    aug_config['move_to_preprocessed'] = _validate_param(aug_config.get('move_to_preprocessed'), True, bool)
    
    # Validasi jenis augmentasi
    aug_config['types'] = _validate_aug_types(aug_config.get('types'))
    
    # Validasi parameter posisi
    if 'position' not in aug_config:
        aug_config['position'] = {}
    
    position_config = aug_config['position']
    position_config['fliplr'] = _validate_param(position_config.get('fliplr'), 0.5, float, range_=(0, 1))
    position_config['degrees'] = _validate_param(position_config.get('degrees'), 15, int, range_=(0, 180))
    position_config['translate'] = _validate_param(position_config.get('translate'), 0.15, float, range_=(0, 1))
    position_config['scale'] = _validate_param(position_config.get('scale'), 0.15, float, range_=(0, 1))
    position_config['shear_max'] = _validate_param(position_config.get('shear_max'), 10, int, range_=(0, 45))
    
    # Validasi parameter pencahayaan
    if 'lighting' not in aug_config:
        aug_config['lighting'] = {}
    
    lighting_config = aug_config['lighting']
    lighting_config['hsv_h'] = _validate_param(lighting_config.get('hsv_h'), 0.025, float, range_=(0, 1))
    lighting_config['hsv_s'] = _validate_param(lighting_config.get('hsv_s'), 0.7, float, range_=(0, 1))
    lighting_config['hsv_v'] = _validate_param(lighting_config.get('hsv_v'), 0.4, float, range_=(0, 1))
    
    # Validasi parameter contrast dan brightness sebagai list
    contrast_default = [0.7, 1.3]
    brightness_default = [0.7, 1.3]
    
    lighting_config['contrast'] = _validate_range_param(lighting_config.get('contrast'), contrast_default)
    lighting_config['brightness'] = _validate_range_param(lighting_config.get('brightness'), brightness_default)
    
    lighting_config['blur'] = _validate_param(lighting_config.get('blur'), 0.2, float, range_=(0, 1))
    lighting_config['noise'] = _validate_param(lighting_config.get('noise'), 0.1, float, range_=(0, 1))
    
    return config

def _validate_param(value: Any, default: Any, type_: type = None, range_: tuple = None) -> Any:
    """
    Validasi parameter dengan tipe dan rentang yang diharapkan.
    
    Args:
        value: Nilai parameter
        default: Nilai default jika validasi gagal
        type_: Tipe yang diharapkan
        range_: Rentang nilai yang valid (min, max)
        
    Returns:
        Nilai parameter yang sudah divalidasi
    """
    # Cek apakah nilai None
    if value is None:
        return default
    
    # Cek tipe
    if type_ and not isinstance(value, type_):
        try:
            value = type_(value)
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Gagal mengkonversi nilai {value} ke tipe {type_.__name__}, menggunakan default {default}")
            return default
    
    # Cek rentang
    if range_ and isinstance(value, (int, float)):
        min_val, max_val = range_
        if value < min_val or value > max_val:
            logger.warning(f"⚠️ Nilai {value} di luar rentang ({min_val}, {max_val}), menggunakan default {default}")
            return default
    
    return value

def _validate_range_param(value: Any, default: List[float]) -> List[float]:
    """
    Validasi parameter rentang (list dengan 2 elemen).
    
    Args:
        value: Nilai parameter
        default: Nilai default jika validasi gagal
        
    Returns:
        List dengan 2 elemen yang sudah divalidasi
    """
    if value is None:
        return default
    
    # Cek apakah list atau tuple
    if not isinstance(value, (list, tuple)):
        logger.warning(f"⚠️ Nilai {value} bukan list atau tuple, menggunakan default {default}")
        return default
    
    # Cek jumlah elemen
    if len(value) != 2:
        logger.warning(f"⚠️ Nilai {value} tidak memiliki 2 elemen, menggunakan default {default}")
        return default
    
    # Cek tipe elemen
    try:
        result = [float(value[0]), float(value[1])]
        # Cek urutan
        if result[0] > result[1]:
            logger.warning(f"⚠️ Nilai min {result[0]} lebih besar dari max {result[1]}, menukar urutan")
            result = [result[1], result[0]]
        return result
    except (ValueError, TypeError):
        logger.warning(f"⚠️ Gagal mengkonversi nilai {value} ke float, menggunakan default {default}")
        return default

def _validate_aug_types(aug_types: Any) -> List[str]:
    """
    Validasi jenis augmentasi.
    
    Args:
        aug_types: Jenis augmentasi
        
    Returns:
        List jenis augmentasi yang sudah divalidasi
    """
    valid_types = ['combined', 'position', 'lighting', 'extreme_rotation']
    default_types = ['combined']
    
    # Cek apakah None
    if aug_types is None:
        return default_types
    
    # Cek apakah string
    if isinstance(aug_types, str):
        aug_types = [aug_types]
    
    # Cek apakah list atau tuple
    if not isinstance(aug_types, (list, tuple)):
        logger.warning(f"⚠️ Nilai {aug_types} bukan list atau tuple, menggunakan default {default_types}")
        return default_types
    
    # Filter nilai yang valid
    result = []
    for aug_type in aug_types:
        if aug_type is None:
            continue
            
        # Konversi ke lowercase
        aug_type_lower = str(aug_type).lower()
        
        # Normalisasi nama
        if aug_type_lower == 'combined (recommended)':
            aug_type_lower = 'combined'
            
        # Cek apakah valid
        if aug_type_lower in valid_types:
            result.append(aug_type_lower)
    
    # Jika kosong, gunakan default
    if not result:
        logger.warning(f"⚠️ Tidak ada jenis augmentasi yang valid, menggunakan default {default_types}")
        return default_types
    
    return result
