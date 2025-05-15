"""
File: smartcash/ui/dataset/split/handlers/config_handlers.py
Deskripsi: Handler untuk operasi konfigurasi split dataset
"""

from typing import Dict, Any
import yaml
from pathlib import Path
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.common.logger import get_logger

# Import ConfigManager untuk persistensi
try:
    from smartcash.common.config.manager import ConfigManager
except ImportError:
    # Fallback jika ConfigManager tidak tersedia
    ConfigManager = None
    print(f"{ICONS['warning']} ConfigManager tidak tersedia, persistensi konfigurasi mungkin tidak berfungsi dengan baik")

def update_config_from_ui(config: Dict[str, Any], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari komponen UI dengan validasi.
    
    Args:
        config: Konfigurasi aplikasi
        ui_components: Dictionary komponen UI
        
    Returns:
        Konfigurasi yang telah diupdate
    """
    # Pastikan config ada dan memiliki struktur yang benar
    if not config:
        config = {}
    if 'data' not in config:
        config['data'] = {}
    if 'split' not in config['data']:
        config['data']['split'] = {}
    
    # Validasi komponen yang diperlukan
    if 'split_sliders' not in ui_components or not ui_components['split_sliders'] or len(ui_components['split_sliders']) < 3:
        print(f"{ICONS['warning']} Komponen split_sliders tidak tersedia atau tidak valid")
        return config
    
    # Dapatkan nilai dari UI
    train_slider, valid_slider, test_slider = ui_components['split_sliders']
    
    # Normalisasi nilai slider untuk memastikan total 1.0
    total = train_slider.value + valid_slider.value + test_slider.value
    if total > 0:  # Hindari division by zero
        train_ratio = train_slider.value / total
        valid_ratio = valid_slider.value / total
        test_ratio = test_slider.value / total
    else:
        # Default jika total 0
        train_ratio, valid_ratio, test_ratio = 0.8, 0.1, 0.1
    
    # Update konfigurasi
    config['data']['split'] = {
        'train': train_ratio,
        'val': valid_ratio,  # Menggunakan 'val' bukan 'valid' untuk konsistensi dengan dataset_config.yaml
        'test': test_ratio,
        'stratified': ui_components['stratified'].value if 'stratified' in ui_components else True
    }
    
    # Advanced options
    if 'advanced_options' in ui_components and hasattr(ui_components['advanced_options'], 'children') and len(ui_components['advanced_options'].children) >= 3:
        config['data']['random_seed'] = ui_components['advanced_options'].children[0].value
        config['data']['backup_before_split'] = ui_components['advanced_options'].children[1].value
        config['data']['backup_dir'] = ui_components['advanced_options'].children[2].value
    
    # Data paths
    if 'data_paths' in ui_components and hasattr(ui_components['data_paths'], 'children') and len(ui_components['data_paths'].children) >= 2:
        config['data']['dataset_path'] = ui_components['data_paths'].children[0].value
        config['data']['preprocessed_path'] = ui_components['data_paths'].children[1].value
    
    return config

def save_config(config: Dict[str, Any], config_path: str = 'config/dataset_config.yaml') -> bool:
    """
    Simpan konfigurasi dataset ke file.
    
    Args:
        config: Konfigurasi aplikasi
        config_path: Path ke file konfigurasi
        
    Returns:
        Boolean yang menunjukkan keberhasilan penyimpanan
    """
    try:
        # Cek apakah file konfigurasi ada
        config_file = Path(config_path)
        if not config_file.exists():
            # Coba cari di lokasi alternatif
            alt_path = Path("../config/dataset_config.yaml")
            if alt_path.exists():
                config_path = str(alt_path)
        
        # Pastikan direktori ada
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Simpan konfigurasi ke file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return True
    except Exception as e:
        print(f"{ICONS['error']} Error saat menyimpan konfigurasi: {str(e)}")
        return False

def load_default_config() -> Dict[str, Any]:
    """
    Load konfigurasi default untuk dataset.
    
    Returns:
        Dictionary berisi konfigurasi default
    """
    # Nilai default untuk split dataset
    return {
        'data': {
            'split': {
                'train': 0.8,
                'val': 0.1,
                'test': 0.1,
                'stratified': True
            },
            'random_seed': 42,
            'backup_before_split': True,
            'backup_dir': 'data/splits_backup',
            'dataset_path': 'data',
            'preprocessed_path': 'data/preprocessed'
        }
    }

def load_split_config_config() -> Dict[str, Any]:
    """
    Load konfigurasi split dataset dari file YAML.
    
    Returns:
        Dictionary berisi konfigurasi split dataset
    """
    try:
        # Path ke file konfigurasi dataset
        config_path = Path("config/dataset_config.yaml")
        if not config_path.exists():
            # Coba cari di lokasi alternatif
            alt_path = Path("../config/dataset_config.yaml")
            if alt_path.exists():
                config_path = alt_path
            else:
                return load_default_config()
        
        # Load konfigurasi dari file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Pastikan struktur konfigurasi benar
        if not config:
            config = {}
        if 'data' not in config:
            config['data'] = {}
        if 'split' not in config['data']:
            config['data']['split'] = {'train': 0.8, 'val': 0.1, 'test': 0.1, 'stratified': True}
        
        return config
    except Exception as e:
        print(f"{ICONS['error']} Error saat memuat konfigurasi split dataset: {str(e)}")
        # Kembalikan konfigurasi default
        return load_default_config()

def get_config_manager():
    """
    Dapatkan instance ConfigManager dengan penanganan error.
    
    Returns:
        Instance ConfigManager atau None jika tidak tersedia
    """
    if ConfigManager is None:
        return None
    
    try:
        return ConfigManager.get_instance()
    except Exception as e:
        print(f"{ICONS['warning']} Gagal mendapatkan instance ConfigManager: {str(e)}")
        return None

def save_config_with_manager(config: Dict[str, Any], ui_components: Dict[str, Any], logger=None) -> bool:
    """
    Simpan konfigurasi menggunakan ConfigManager dengan fallback.
    
    Args:
        config: Konfigurasi aplikasi
        ui_components: Dictionary komponen UI
        logger: Logger untuk logging
        
    Returns:
        Boolean yang menunjukkan keberhasilan penyimpanan
    """
    success = False
    
    # Coba simpan dengan ConfigManager terlebih dahulu
    config_manager = get_config_manager()
    if config_manager:
        try:
            # Pastikan UI components terdaftar untuk persistensi
            config_manager.register_ui_components('dataset_split', ui_components)
            # Simpan konfigurasi
            success = config_manager.save_module_config('dataset', config)
            if logger: logger.debug(f"{ICONS['info']} Konfigurasi disimpan melalui ConfigManager: {success}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan dengan ConfigManager: {str(e)}")
            # Fallback ke metode save_config tradisional
            success = save_config(config)
    else:
        # Fallback ke metode save_config tradisional
        success = save_config(config)
    
    return success
