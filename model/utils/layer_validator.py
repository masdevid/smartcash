"""
File: smartcash/model/utils/layer_validator.py
Deskripsi: Utilitas untuk validasi parameter layer mode dan detection layers
"""

from typing import List, Dict, Tuple, Any
from smartcash.common.logger import get_logger
from smartcash.model.config.model_constants import DETECTION_LAYERS

logger = get_logger(__name__)

def validate_layer_params(
    layer_mode: str, 
    detection_layers: List[str]
) -> Tuple[str, List[str]]:
    """
    Memvalidasi parameter layer_mode dan detection_layers dan memperbaiki jika tidak valid.
    
    Args:
        layer_mode: Mode layer ('single' atau 'multilayer')
        detection_layers: Daftar layer deteksi yang digunakan
        
    Returns:
        Tuple berisi (layer_mode, detection_layers) yang sudah divalidasi
    """
    # Validasi detection_layers
    valid_detection_layers = []
    for layer in detection_layers:
        if layer in DETECTION_LAYERS:
            valid_detection_layers.append(layer)
    
    # Jika tidak ada layer valid, gunakan default 'banknote'
    if not valid_detection_layers:
        logger.warning(f"⚠️ Tidak ada detection_layers valid, menggunakan default 'banknote'")
        valid_detection_layers = ['banknote']
    
    # Validasi layer_mode berdasarkan jumlah detection_layers
    valid_layer_mode = layer_mode
    
    # Kasus 1: Jika hanya satu detection_layer tapi mode multilayer, ubah ke single
    if len(valid_detection_layers) == 1 and layer_mode == 'multilayer':
        logger.warning(f"⚠️ Hanya satu detection_layer tetapi layer_mode adalah '{layer_mode}', mengubah ke 'single'")
        valid_layer_mode = 'single'
    
    # Kasus 2: Jika banyak detection_layers dengan mode multilayer, tetap multilayer
    elif len(valid_detection_layers) > 1 and layer_mode == 'multilayer':
        valid_layer_mode = 'multilayer'
    
    # Kasus 3: Jika banyak detection_layers dengan mode single, tetap single
    # tapi beri peringatan bahwa hanya layer pertama yang akan digunakan
    elif len(valid_detection_layers) > 1 and layer_mode == 'single':
        logger.warning(f"⚠️ Ada {len(valid_detection_layers)} detection_layers dengan mode 'single', hanya layer pertama yang akan digunakan")
        valid_layer_mode = 'single'
        
    # Kasus 4: Jika satu detection_layer dengan mode single, tetap single
    else:  # len(valid_detection_layers) == 1 and layer_mode == 'single'
        valid_layer_mode = 'single'
    
    return valid_layer_mode, valid_detection_layers

def get_num_classes_for_layers(detection_layers: List[str]) -> int:
    """
    Menghitung total jumlah kelas berdasarkan detection_layers yang digunakan.
    
    Args:
        detection_layers: Daftar layer deteksi yang digunakan
        
    Returns:
        Total jumlah kelas
    """
    # Jumlah kelas untuk setiap layer (sesuai dengan LAYER_CONFIG di model_constants.py)
    layer_classes = {
        'banknote': 7,  # 7 denominasi
        'nominal': 7,   # 7 area nominal
        'security': 3   # 3 fitur keamanan
    }
    
    total_classes = 0
    for layer in detection_layers:
        if layer in layer_classes:
            total_classes += layer_classes[layer]
    
    # Jika tidak ada layer valid, gunakan default 7 (untuk banknote)
    if total_classes == 0:
        total_classes = 7
        
    return total_classes
