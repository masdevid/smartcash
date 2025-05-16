"""
File: smartcash/ui/dataset/preprocessing/handlers/persistence_handler.py
Deskripsi: Handler untuk persistensi konfigurasi preprocessing
"""

from typing import Dict, Any, Optional, List, Union
import os
import yaml
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager

logger = get_logger("preprocessing_persistence")

def validate_param(value: Any, default_value: Any, 
                  valid_types: Optional[Union[type, List[type]]] = None, 
                  valid_values: Optional[List[Any]] = None) -> Any:
    """
    Validasi parameter dengan fallback ke nilai default.
    
    Args:
        value: Nilai yang akan divalidasi
        default_value: Nilai default jika validasi gagal
        valid_types: Tipe data yang valid
        valid_values: Nilai-nilai yang valid
        
    Returns:
        Nilai yang sudah divalidasi atau default_value jika validasi gagal
    """
    # Validasi None
    if value is None:
        return default_value
    
    # Validasi tipe data
    if valid_types:
        # Konversi ke list jika bukan list
        if not isinstance(valid_types, (list, tuple)):
            valid_types = [valid_types]
        
        # Cek apakah value memiliki tipe yang valid
        if not any(isinstance(value, t) for t in valid_types):
            return default_value
    
    # Validasi nilai
    if valid_values and value not in valid_values:
        return default_value
    
    return value

def ensure_ui_persistence(ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Pastikan persistensi UI components dengan ConfigManager.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi tambahan (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Dapatkan instance ConfigManager
        config_manager = get_config_manager()
        
        # Register UI components untuk persistensi
        config_manager.register_ui_components('preprocessing', ui_components)
        
        # Dapatkan konfigurasi preprocessing jika tidak disediakan
        if not config:
            config = config_manager.get_module_config('preprocessing')
            
            # Jika masih None, coba load dari file
            if not config:
                try:
                    from smartcash.ui.dataset.preprocessing.handlers.config_handler import load_preprocessing_config
                    config = load_preprocessing_config(ui_components=ui_components)
                except Exception as e:
                    logger.warning(f"{ICONS['warning']} Gagal memuat konfigurasi: {str(e)}")
        
        # Update UI dari konfigurasi jika ada
        if config:
            from smartcash.ui.dataset.preprocessing.handlers.config_handler import update_ui_from_config
            ui_components = update_ui_from_config(ui_components, config)
        
        # Log info
        logger.info(f"{ICONS['success']} UI components berhasil terdaftar untuk persistensi")
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat memastikan persistensi UI: {str(e)}")
        return ui_components

def get_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi preprocessing dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi preprocessing
    """
    # Coba dapatkan dari ConfigManager terlebih dahulu
    try:
        config_manager = get_config_manager()
        config = config_manager.get_module_config('preprocessing')
        if config and isinstance(config, dict) and 'preprocessing' in config:
            logger.debug(f"{ICONS['info']} Menggunakan konfigurasi dari ConfigManager")
            return config
    except Exception as e:
        logger.debug(f"{ICONS['info']} Tidak dapat memuat dari ConfigManager: {str(e)}")
    
    # Jika tidak ada di ConfigManager, ekstrak dari UI
    try:
        # Import parameter extractor
        from smartcash.ui.dataset.preprocessing.handlers.parameter_handler import extract_preprocess_params
        
        # Ekstrak parameter preprocessing dari UI
        preprocess_params = extract_preprocess_params(ui_components)
        
        # Dapatkan split dari UI
        split_option = ui_components.get('split_selector', {}).value if 'split_selector' in ui_components else 'All Splits'
        split_map = {'All Splits': None, 'Train Only': 'train', 'Validation Only': 'valid', 'Test Only': 'test'}
        split = split_map.get(split_option)
        
        # Tambahkan split ke parameter
        preprocess_params['split'] = split
        
        # Buat struktur konfigurasi yang benar
        config = {
            'preprocessing': preprocess_params,
            'data': {
                'dir': ui_components.get('data_dir', 'data')
            }
        }
        
        return config
    except Exception as e:
        logger.warning(f"{ICONS['warning']} Error saat ekstrak konfigurasi dari UI: {str(e)}")
        
        # Fallback ke konfigurasi yang tersimpan di ui_components
        if 'config' in ui_components and ui_components['config']:
            return ui_components['config']
        
        # Fallback ke konfigurasi default
        from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_PREPROCESSED_DIR, DEFAULT_INVALID_DIR, DEFAULT_IMG_SIZE
        return {
            "preprocessing": {
                "enabled": True,
                "output_dir": DEFAULT_PREPROCESSED_DIR,
                "img_size": DEFAULT_IMG_SIZE,
                "normalization": {
                    "enabled": True,
                    "preserve_aspect_ratio": True
                },
                "validate": {
                    "enabled": True,
                    "fix_issues": True,
                    "move_invalid": True,
                    "invalid_dir": DEFAULT_INVALID_DIR
                },
                "splits": DEFAULT_SPLITS,
                "num_workers": 4
            },
            "data": {
                "dir": "data"
            }
        }

def sync_config_with_drive(ui_components: Dict[str, Any]) -> bool:
    """
    Sinkronisasi konfigurasi dengan file di drive.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean status keberhasilan
    """
    try:
        # Dapatkan instance ConfigManager
        config_manager = get_config_manager()
        
        # Dapatkan konfigurasi dari UI
        from smartcash.ui.dataset.preprocessing.handlers.config_handler import update_config_from_ui
        updated_config = update_config_from_ui(ui_components, ui_components.get('config', {}))
        
        # Validasi konfigurasi sebelum disimpan
        if not updated_config or not isinstance(updated_config, dict):
            logger.warning(f"{ICONS['warning']} Konfigurasi tidak valid untuk disinkronkan")
            return False
            
        # Pastikan struktur konfigurasi benar
        if 'preprocessing' not in updated_config:
            updated_config['preprocessing'] = {}
        if 'data' not in updated_config:
            updated_config['data'] = {}
        
        # Simpan konfigurasi ke drive melalui ConfigManager
        success = config_manager.save_module_config('preprocessing', updated_config)
        
        # Simpan juga ke file lokal untuk kompatibilitas
        try:
            config_path = "configs/preprocessing_config.yaml"
            # Pastikan direktori ada
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            # Simpan ke file
            with open(config_path, 'w') as f:
                yaml.dump(updated_config, f, default_flow_style=False)
        except Exception as e:
            logger.warning(f"{ICONS['warning']} Gagal menyimpan ke file lokal: {str(e)}")
        
        # Coba sinkronkan dengan Google Drive jika tersedia
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if env_manager.is_drive_mounted:
                drive_config_path = str(env_manager.drive_path / 'configs' / Path(config_path).name)
                
                # Buat direktori jika belum ada
                os.makedirs(os.path.dirname(drive_config_path), exist_ok=True)
                
                # Salin file ke Google Drive
                with open(drive_config_path, 'w') as f:
                    yaml.dump(updated_config, f, default_flow_style=False)
                    
                logger.info(f"{ICONS['success']} Konfigurasi disimpan ke drive: {drive_config_path}")
        except Exception as e:
            logger.debug(f"{ICONS['info']} Tidak dapat menyalin ke drive: {str(e)}")
        
        # Log info
        if success:
            logger.info(f"{ICONS['success']} Konfigurasi berhasil disinkronkan dengan drive")
            # Simpan kembali config yang diupdate ke ui_components
            ui_components['config'] = updated_config
        else:
            logger.error(f"{ICONS['error']} Gagal menyinkronkan konfigurasi dengan drive")
        
        return success
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat menyinkronkan konfigurasi: {str(e)}")
        return False


def get_persisted_ui_components() -> Dict[str, Any]:
    """
    Dapatkan UI components yang tersimpan dari ConfigManager.
    
    Returns:
        Dictionary UI components atau None jika tidak ditemukan
    """
    try:
        config_manager = get_config_manager()
        return config_manager.get_ui_components('preprocessing')
    except Exception as e:
        logger.debug(f"{ICONS['info']} Tidak dapat memuat UI components dari ConfigManager: {str(e)}")
        return None

def reset_config_to_default(ui_components: Dict[str, Any]) -> bool:
    """
    Reset konfigurasi ke default dan perbarui UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean status keberhasilan
    """
    try:
        from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_PREPROCESSED_DIR, DEFAULT_INVALID_DIR, DEFAULT_IMG_SIZE
        
        # Buat konfigurasi default
        default_config = {
            "preprocessing": {
                "enabled": True,
                "output_dir": DEFAULT_PREPROCESSED_DIR,
                "img_size": DEFAULT_IMG_SIZE,
                "normalization": {
                    "enabled": True,
                    "preserve_aspect_ratio": True
                },
                "validate": {
                    "enabled": True,
                    "fix_issues": True,
                    "move_invalid": True,
                    "invalid_dir": DEFAULT_INVALID_DIR
                },
                "splits": DEFAULT_SPLITS,
                "num_workers": 4
            },
            "data": {
                "dir": "data"
            }
        }
        
        # Simpan ke ConfigManager
        config_manager = get_config_manager()
        success = config_manager.save_module_config('preprocessing', default_config)
        
        # Update UI dari konfigurasi default
        if success and ui_components:
            from smartcash.ui.dataset.preprocessing.handlers.config_handler import update_ui_from_config
            ui_components = update_ui_from_config(ui_components, default_config)
            ui_components['config'] = default_config
            
            # Simpan juga ke file lokal
            try:
                config_path = "configs/preprocessing_config.yaml"
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
            except Exception as e:
                logger.warning(f"{ICONS['warning']} Gagal menyimpan default ke file: {str(e)}")
        
        return success
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat reset konfigurasi: {str(e)}")
        return False
