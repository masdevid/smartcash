"""
File: smartcash/ui/dataset/split/handlers/config_handlers.py
Deskripsi: Handler untuk operasi konfigurasi split dataset
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path
import os
from smartcash.ui.utils.constants import ICONS

def load_default_config() -> Dict[str, Any]:
    """
    Load konfigurasi default untuk split dataset.
    
    Returns:
        Dictionary berisi konfigurasi default
    """
    # Nilai default untuk split dataset dengan ratio 70-15-15
    return {
        'data': {
            'split': {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15,
                'stratified': True
            },
            'random_seed': 42,
            'backup_before_split': True,
            'backup_dir': 'data/splits_backup',
            'dataset_path': 'data',
            'preprocessed_path': 'data/preprocessed'
        }
    }

def load_config() -> Dict[str, Any]:
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
            config['data']['split'] = {'train': 0.7, 'val': 0.15, 'test': 0.15, 'stratified': True}
        
        return config
    except Exception as e:
        print(f"{ICONS['error']} Error saat memuat konfigurasi split dataset: {str(e)}")
        # Kembalikan konfigurasi default
        return load_default_config()

def save_config(config: Dict[str, Any]) -> bool:
    """
    Simpan konfigurasi split dataset ke file YAML.
    
    Args:
        config: Konfigurasi yang akan disimpan
        
    Returns:
        Boolean yang menunjukkan keberhasilan penyimpanan
    """
    try:
        # Path ke file konfigurasi dataset
        config_path = Path("config/dataset_config.yaml")
        
        # Buat direktori jika belum ada
        os.makedirs(config_path.parent, exist_ok=True)
        
        # Simpan konfigurasi ke file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Coba sinkronkan dengan drive jika tersedia
        try:
            from google.colab import drive
            # Cek apakah drive sudah di-mount
            if os.path.exists('/content/drive'):
                # Copy file ke drive
                drive_path = Path("/content/drive/MyDrive/smartcash/config/dataset_config.yaml")
                os.makedirs(drive_path.parent, exist_ok=True)
                with open(drive_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                print(f"{ICONS['success']} Konfigurasi berhasil disinkronkan dengan Google Drive")
        except ImportError:
            # Bukan di Google Colab, abaikan
            pass
        
        return True
    except Exception as e:
        print(f"{ICONS['error']} Error saat menyimpan konfigurasi split dataset: {str(e)}")
        return False

def get_config_manager_instance():
    """
    Dapatkan instance ConfigManager jika tersedia.
    
    Returns:
        Instance ConfigManager atau None jika tidak tersedia
    """
    try:
        from smartcash.common.config.manager import get_config_manager
        return get_config_manager()
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
    config_manager = get_config_manager_instance()
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

def update_config_from_ui(config: Dict[str, Any], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari nilai UI.
    
    Args:
        config: Konfigurasi aplikasi
        ui_components: Dictionary komponen UI
        
    Returns:
        Konfigurasi yang diupdate
    """
    # Pastikan struktur konfigurasi benar
    if not config:
        config = {}
    if 'data' not in config:
        config['data'] = {}
    if 'split' not in config['data']:
        config['data']['split'] = {}
    
    # Update nilai split dari slider
    if 'train_slider' in ui_components:
        config['data']['split']['train'] = round(ui_components['train_slider'].value, 2)
    if 'val_slider' in ui_components:
        config['data']['split']['val'] = round(ui_components['val_slider'].value, 2)
    if 'test_slider' in ui_components:
        config['data']['split']['test'] = round(ui_components['test_slider'].value, 2)
    
    # Update nilai stratified dari checkbox
    if 'stratified_checkbox' in ui_components:
        config['data']['split']['stratified'] = ui_components['stratified_checkbox'].value
    
    # Update nilai random seed dari input
    if 'random_seed' in ui_components:
        config['data']['random_seed'] = ui_components['random_seed'].value
    
    # Update nilai backup dari checkbox
    if 'backup_checkbox' in ui_components:
        config['data']['backup_before_split'] = ui_components['backup_checkbox'].value
    
    # Update nilai backup dir dari input
    if 'backup_dir' in ui_components:
        config['data']['backup_dir'] = ui_components['backup_dir'].value
    
    # Update nilai dataset path dari input
    if 'dataset_path' in ui_components:
        config['data']['dataset_path'] = ui_components['dataset_path'].value
    
    # Update nilai preprocessed path dari input
    if 'preprocessed_path' in ui_components:
        config['data']['preprocessed_path'] = ui_components['preprocessed_path'].value
    
    return config
